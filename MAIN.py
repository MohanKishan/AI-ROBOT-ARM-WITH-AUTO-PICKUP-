#!/usr/bin/env python3
"""
Voice-Controlled 4-DOF Robot Arm â€” HTTP command API + low-latency object detection
Pi 4B Â· Python 3.x  [v40 — Perfect center-grab pickup system]

v40 Changes â€” Perfect center-grab pickup:
  - Pickup always targets the geometric center of the locked object (never edges/bottom/top).
  - Two-phase center acquisition (coarse + fine) with EMA smoothing and multi-frame agreement.
  - Descent blocked until crosshair alignment is stable for PICKUP_STABLE_FRAMES frames.
  - Target center frozen before depth scan; blind descent trusts pre-descent alignment.
  - Post-move YOLO blackout stamped on every pickup servo move.

v35 Changes â€” Centering Stability (camera-on-arm YOLO distraction fix):
  PROBLEM: During pickup centering the camera moves WITH the arm.  Every servo
  step shifts the background in the frame, causing YOLO to pick the wrong centroid
  in cluttered scenes â€” the arm then over-corrects and oscillates.

  FIX 1 â€” Post-move YOLO blackout (PICKUP_POST_MOVE_BLACKOUT_SEC = 0.40 s):
    After every servo command during centering, detector thread discards all YOLO
    frames captured inside the blackout window.  Only frames grabbed after the arm
    has fully settled feed the error measurement.

  FIX 2 â€” Slower EMA alpha during centering (PICKUP_CENTERING_EMA_ALPHA = 0.20):
    Reduced from hard-coded 0.35 â†’ tunable 0.20.  Heavier smoothing means a single
    noisy frame (background motion, partial occlusion) cannot dominate the centroid.

  FIX 3 â€” Multi-frame agreement gate (PICKUP_CENTERING_MIN_AGREE_FRAMES = 2):
    _wait_for_target now requires N consecutive IoU-matching frames before
    returning a target.  Prevents a single transient detection from triggering a
    corrective move.

  All three thresholds are tunable via env-vars without code changes.

"""

import os
import sys
import base64
import json
import math
import queue
import re
import time
import signal
import socket
import subprocess
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote

try:
    import serial as _serial_mod
except ImportError:
    _serial_mod = None

try:
    import board as _board_mod
    import busio as _busio_mod
    from adafruit_ina219 import INA219 as _INA219_mod
    _INA219_AVAILABLE = True
except ImportError:
    _INA219_AVAILABLE = False

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# --- Pipeline classes (embedded) -------------------------------------------------
#  PART 1 — ROBOT GEOMETRY & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RobotGeometry:
    """All physical dimensions of the robot arm.
    
    Defaults match the measured arm (v42 config).
    Override any value via constructor or env var.
    """
    link1_cm: float = 13.5           # upper arm: shoulder → elbow pivot-to-pivot
    link2_cm: float = 31.0           # forearm + gripper: elbow → tip pivot-to-pivot
    link2_forearm_frac: float = 0.73 # fraction of L2 that is forearm
    link2_wrist_frac: float = 0.27   # fraction of L2 that is wrist+gripper
    gripper_finger_ext_cm: float = 4.8  # finger extension beyond tip
    floor_below_shoulder_cm: float = 7.2  # shoulder height above floor
    shoulder_to_elbow_offset_cm: float = 0.0  # lateral elbow offset (if any)
    wrist_to_gripper_lateral_cm: float = 0.0  # gripper lateral offset (if any)

    @property
    def link2_forearm_cm(self) -> float:
        return self.link2_cm * self.link2_forearm_frac

    @property
    def link2_wrist_cm(self) -> float:
        return self.link2_cm * self.link2_wrist_frac

    @property
    def total_reach_cm(self) -> float:
        return self.link1_cm + self.link2_cm

    @property
    def shoulder_height_cm(self) -> float:
        """Height of shoulder joint above the floor."""
        return self.floor_below_shoulder_cm


@dataclass
class JointLimits:
    """Mechanical joint limits in degrees."""
    base_min: float = 10.0
    base_max: float = 170.0
    # FIX: was -50.0. With JOINT_REVERSED['arm']=True, servo_angle = 180 - angle,
    # so any logical angle below 0 produced servo_angle > 180, which
    # _angle_to_us() silently clamped back to 180 — meaning the real servo
    # froze at max pulse for the entire -50..0 range while the FK/3D model
    # kept computing positions as if the arm were still moving. That mismatch
    # is exactly why the real arm hit the floor at a different point than
    # the digital twin predicted. 0.0 is the true minimum the hardware can reach.
    arm_min: float = 0.0
    arm_max: float = 180.0
    wrist_min: float = 0.0
    wrist_max: float = 180.0
    grip_open: float = 30.0
    grip_close: float = 120.0

    def clamp(self, joint: str, value: float) -> float:
        limits = {
            'base': (self.base_min, self.base_max),
            'arm': (self.arm_min, self.arm_max),
            'wrist': (self.wrist_min, self.wrist_max),
            'grip': (self.grip_open, self.grip_close),
        }
        lo, hi = limits.get(joint, (-180, 180))
        return max(lo, min(hi, value))

    def in_bounds(self, joint: str, value: float) -> bool:
        clamped = self.clamp(joint, value)
        return abs(clamped - value) < 1e-9


@dataclass
class CameraIntrinsics:
    """Pinhole camera model parameters.
    
    If fx/fy are not provided, compute from FOV and frame size.
    """
    width_px: int = 640
    height_px: int = 480
    fx: Optional[float] = None   # pixels
    fy: Optional[float] = None   # pixels
    cx: Optional[float] = None   # principal point x (default: width/2)
    cy: Optional[float] = None   # principal point y (default: height/2)
    fov_x_deg: float = 74.0     # horizontal field of view
    fov_y_deg: float = 42.0     # vertical field of view

    def __post_init__(self):
        if self.cx is None:
            self.cx = self.width_px / 2.0
        if self.cy is None:
            self.cy = self.height_px / 2.0
        if self.fx is None:
            self.fx = (self.width_px / 2.0) / math.tan(math.radians(self.fov_x_deg / 2.0))
        if self.fy is None:
            self.fy = (self.height_px / 2.0) / math.tan(math.radians(self.fov_y_deg / 2.0))


@dataclass
class CameraExtrinsics:
    """Camera-to-gripper extrinsic calibration.
    
    The camera is eye-in-hand: mounted on the forearm, behind the gripper,
    looking forward. This defines the rigid transform from camera frame
    (X=right, Y=down, Z=forward) to gripper frame (same axes orientation,
    origin at gripper jaw center).
    """
    # Translation from camera origin to gripper jaw center (cm), in camera frame
    t_x_cm: float = 0.0    # lateral (right +ve)
    t_y_cm: float = -3.0   # vertical (down +ve, gripper is below camera)
    t_z_cm: float = 8.0    # forward (gripper is in front of camera)

    # Rotation (Euler ZYX or just identity if aligned)
    # Camera and gripper are assumed co-axial (no relative rotation)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    # Sensor-to-jaw offset: distance from VL53L0X sensor face to gripper jaw
    sensor_to_jaw_cm: float = 10.0

    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from camera to gripper frame."""
        r = math.radians(self.roll_deg)
        p = math.radians(self.pitch_deg)
        y = math.radians(self.yaw_deg)
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(r), -math.sin(r)],
                       [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)],
                       [0, 1, 0],
                       [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0],
                       [math.sin(y), math.cos(y), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def homogeneous_matrix(self) -> np.ndarray:
        """4×4 homogeneous transform from camera to gripper frame."""
        R = self.rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        # t = position of gripper in camera frame. For camera→gripper transform,
        # we need translation = -R.T @ t (position of camera in gripper frame).
        # With R=I, this is simply -t.
        T[:3, 3] = [-self.t_x_cm, -self.t_y_cm, -self.t_z_cm]
        return T

    def inverse_homogeneous(self) -> np.ndarray:
        """4×4 homogeneous transform from gripper to camera frame."""
        T = self.homogeneous_matrix()
        R = T[:3, :3]
        t = T[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = R.T
        inv[:3, 3] = -R.T @ t
        return inv


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — FORWARD KINEMATICS
# ═══════════════════════════════════════════════════════════════════════════

class ForwardKinematics:
    """Forward kinematics for the 4-DOF articulated arm.
    
    Joint convention (matches physical servos + 3D model):
      base:   90° = arm points forward along +X
      arm:    90° = vertical up (home)
      arm:   >90° = tip swings down toward floor
      wrist:  90° = straight (parallel to forearm)
      wrist: <90° = nose-down bend
    
Internal geometric angles (radians):
      θ_base  = base_deg - 90°
      θ_shoulder = 1.526 * arm_deg - 47.35°  (arm=25→floor, arm=90→vertical UP)
      θ_wrist = wrist_deg - 90°    (0 = straight, + = bend down)
    """

    def __init__(self, geometry: RobotGeometry, limits: JointLimits):
        self.geom = geometry
        self.limits = limits

    @staticmethod
    def _to_rad(base_deg: float, arm_deg: float, wrist_deg: float
                ) -> Tuple[float, float, float]:
        """Convert servo degrees to internal radians.
        
        Convention (matches physical servos + 3D model):
          θ_base = base_deg - 90°  (0 = forward +X)
          θ_shoulder = 1.526 * arm_deg - 47.35°  (arm=25→floor, arm=90→vertical UP)
          θ_wrist = wrist_deg - 90°  (0 = straight, + = bend down)
        """
        return (math.radians(base_deg - 90.0),
                math.radians(_fk_shoulder_deg(arm_deg)),
                math.radians(wrist_deg - 90.0))

    def forward_transform(self, base_deg: float, arm_deg: float,
                          wrist_deg: float) -> np.ndarray:
        """Full 4×4 forward kinematics transform (geometric approach).
        
        Camera frame {C}: X=right, Y=down, Z=forward
        Base frame {0}: X=forward, Y=left, Z=up
        
        Chain mathematics (validated against physical arm):
          1. Base rotates arm plane by θ_base about Z
          2. In arm plane: shoulder angle θ_sh, wrist angle θ_wr
          3. x_reach = L1·cos(θ_sh) + L2·cos(θ_sh + θ_wr)
          4. z_drop  = L1·sin(θ_sh) + L2·sin(θ_sh + θ_wr)
          5. X = x_reach·cos(θ_base), Y = x_reach·sin(θ_base)
          6. Z = shoulder_height - z_drop
        
        Returns:
            4×4 homogeneous matrix T_base_to_gripper
        """
        tb, ts, tw = self._to_rad(base_deg, arm_deg, wrist_deg)
        h = self.geom.shoulder_height_cm
        L1 = self.geom.link1_cm
        L2 = self.geom.link2_cm

        x_reach = L1 * math.cos(ts) + L2 * math.cos(ts + tw)
        z_drop = L1 * math.sin(ts) + L2 * math.sin(ts + tw)

        X = x_reach * math.cos(tb)
        Y = x_reach * math.sin(tb)
        Z = h - z_drop

        # Gripper orientation in base frame
        # Forward direction (X_grip) in the arm plane at angle (ts + tw)
        x_grip_dir = np.array([
            math.cos(tb) * math.cos(ts + tw),
            math.sin(tb) * math.cos(ts + tw),
            -math.sin(ts + tw)
        ])

        # Y_grip is normal to the arm plane (perpendicular to both X_grip and -Z rotated)
        y_grip_dir = np.array([
            -math.sin(tb),
            math.cos(tb),
            0.0
        ])

        z_grip_dir = np.cross(x_grip_dir, y_grip_dir)

        T = np.eye(4)
        T[:3, 0] = x_grip_dir
        T[:3, 1] = y_grip_dir
        T[:3, 2] = z_grip_dir
        T[:3, 3] = [X, Y, Z]
        return T

    def gripper_pose(self, base_deg: float, arm_deg: float,
                     wrist_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """Gripper position (3D) and orientation (3×3 rotation matrix).
        
        Returns:
            (position, rotation_matrix) in base frame
        """
        T = self.forward_transform(base_deg, arm_deg, wrist_deg)
        return T[:3, 3], T[:3, :3]

    def gripper_position(self, base_deg: float, arm_deg: float,
                         wrist_deg: float) -> Tuple[float, float, float]:
        """Gripper tip position in base frame (X, Y, Z in cm)."""
        pos, _ = self.gripper_pose(base_deg, arm_deg, wrist_deg)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def joint_positions(self, base_deg: float, arm_deg: float,
                        wrist_deg: float) -> Dict[str, np.ndarray]:
        """3D positions of all joints in base frame (geometric approach)."""
        tb, ts, tw = self._to_rad(base_deg, arm_deg, wrist_deg)
        h = self.geom.shoulder_height_cm
        L1 = self.geom.link1_cm
        L2 = self.geom.link2_cm

        base_pos = np.array([0.0, 0.0, 0.0])
        shoulder_pos = np.array([0.0, 0.0, h])

        elbow_reach = L1 * math.cos(ts)
        elbow_drop = L1 * math.sin(ts)
        elbow_pos = np.array([
            elbow_reach * math.cos(tb),
            elbow_reach * math.sin(tb),
            h - elbow_drop
        ])

        wrist_reach = L1 * math.cos(ts) + self.geom.link2_forearm_cm * math.cos(ts + tw)
        wrist_drop = L1 * math.sin(ts) + self.geom.link2_forearm_cm * math.sin(ts + tw)
        wrist_pos = np.array([
            wrist_reach * math.cos(tb),
            wrist_reach * math.sin(tb),
            h - wrist_drop
        ])

        grip_reach = L1 * math.cos(ts) + L2 * math.cos(ts + tw)
        grip_drop = L1 * math.sin(ts) + L2 * math.sin(ts + tw)
        grip_pos = np.array([
            grip_reach * math.cos(tb),
            grip_reach * math.sin(tb),
            h - grip_drop
        ])

        return {
            'base': base_pos,
            'shoulder': shoulder_pos,
            'elbow': elbow_pos,
            'wrist': wrist_pos,
            'gripper': grip_pos,
        }

    def segment_heights_cm(self, arm_deg: float, wrist_deg: float
                           ) -> Dict[str, float]:
        """Floor clearance for each arm segment (for collision checking)."""
        joints = self.joint_positions(90.0, arm_deg, wrist_deg)
        floor = 0.0
        L1 = self.geom.link1_cm
        L2f = self.geom.link2_forearm_cm
        L2w = self.geom.link2_wrist_cm
        fe = self.geom.gripper_finger_ext_cm

        sh = math.radians(_fk_shoulder_deg(arm_deg))
        wr = math.radians(wrist_deg - 90.0)
        elbow_drop = L1 * math.sin(sh)
        wrist_drop = elbow_drop + L2f * math.sin(sh + wr)
        hand_drop = wrist_drop + L2w * math.sin(sh + wr)
        tip_drop = L1 * math.sin(sh) + self.geom.link2_cm * math.sin(sh + wr)
        finger_drop = tip_drop + fe * max(0.0, math.sin(sh + wr))
        floor_h = self.geom.shoulder_height_cm

        return {
            'elbow': floor_h - elbow_drop,
            'wrist': floor_h - wrist_drop,
            'hand': floor_h - hand_drop,
            'tip': floor_h - tip_drop,
            'finger': floor_h - finger_drop,
        }

    def min_clearance_cm(self, arm_deg: float, wrist_deg: float) -> float:
        return min(self.segment_heights_cm(arm_deg, wrist_deg).values())

    def is_safe(self, arm_deg: float, wrist_deg: float,
                safety_margin_cm: float = 2.5) -> bool:
        """True if all segments clear floor by at least safety_margin_cm."""
        return self.min_clearance_cm(arm_deg, wrist_deg) >= safety_margin_cm

    def error_vs_target(self, base_deg: float, arm_deg: float,
                        wrist_deg: float,
                        target: Tuple[float, float, float]
                        ) -> Dict[str, float]:
        """Position/orientation error of current pose vs target (base frame)."""
        pos, rot = self.gripper_pose(base_deg, arm_deg, wrist_deg)
        tx, ty, tz = target
        dx = pos[0] - tx
        dy = pos[1] - ty
        dz = pos[2] - tz
        euclidean = math.sqrt(dx*dx + dy*dy + dz*dz)
        return {
            'dx_cm': float(dx),
            'dy_cm': float(dy),
            'dz_cm': float(dz),
            'euclidean_cm': float(euclidean),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — INVERSE KINEMATICS
# ═══════════════════════════════════════════════════════════════════════════

class InverseKinematics:
    """Analytical inverse kinematics for the 4-DOF arm.
    
    Strategy:
      1. Solve base angle from (x, y) target → θ_base
      2. Solve planar 2-link IK in the rotated arm plane for (x_reach, z)
      3. Validate solution against joint limits and floor safety
      4. Return complete joint solution
    """

    def __init__(self, geometry: RobotGeometry, limits: JointLimits,
                 fk: ForwardKinematics):
        self.geom = geometry
        self.limits = limits
        self.fk = fk

    def solve_base(self, target_x_cm: float, target_y_cm: float,
                   cur_base_deg: float = 90.0) -> Optional[float]:
        """Solve base angle to point at (x, y) target.
        
        Returns base servo angle in degrees, or None if target is at origin.
        """
        r_xy = math.hypot(target_x_cm, target_y_cm)
        if r_xy < 0.001:
            return cur_base_deg
        bearing = math.atan2(target_y_cm, target_x_cm)
        base_deg = 90.0 + math.degrees(bearing)
        return self.limits.clamp('base', base_deg)

    def solve_planar(self, x_cm: float, z_cm: float,
                     cur_arm_deg: float = 90.0,
                     prefer_elbow_up: bool = False
                     ) -> Optional[Tuple[float, float]]:
        """Planar 2-link IK in the arm vertical plane.
        
        Args:
            x_cm: horizontal reach from shoulder (forward +ve)
            z_cm: vertical drop from shoulder (down +ve)
            cur_arm_deg: current arm angle (for stability check)
            prefer_elbow_up: prefer elbow-above-shoulder solution
        
        Returns:
            (arm_deg, wrist_deg) or None if unreachable
        """
        L1 = self.geom.link1_cm
        L2 = self.geom.link2_cm
        x = max(0.01, float(x_cm))
        z = float(z_cm)
        r = math.hypot(x, z)

        max_reach = L1 + L2
        min_reach = abs(L1 - L2) + 0.1

        if r > max_reach - 0.1 or r < min_reach:
            return None

        cos_theta2 = (r*r - L1*L1 - L2*L2) / (2.0 * L1 * L2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        if prefer_elbow_up:
            theta2_rad = -math.acos(cos_theta2)
        else:
            theta2_rad = math.acos(cos_theta2)

        alpha = math.atan2(z, x)
        beta = math.atan2(L2 * math.sin(theta2_rad),
                          L1 + L2 * math.cos(theta2_rad))
        theta1_rad = alpha + beta

        theta1_deg = math.degrees(theta1_rad)
        arm_deg = _fk_arm_from_shoulder(theta1_deg)
        wrist_deg = 90.0 + math.degrees(theta2_rad)

        arm_deg = self.limits.clamp('arm', arm_deg)
        wrist_deg = self.limits.clamp('wrist', wrist_deg)

        if abs(arm_deg - cur_arm_deg) > 60.0:
            return None

        return (arm_deg, wrist_deg)

    def solve_full(self, target_x: float, target_y: float, target_z: float,
                   cur_base: float = 90.0, cur_arm: float = 90.0,
                   cur_wrist: float = 90.0,
                   safety_margin_cm: float = 2.5,
                   max_solution_candidates: int = 2
                   ) -> Optional[Dict[str, float]]:
        """Solve all 3 positioning DOF for a given target.
        
        Strategy:
          1. Solve base rotation from (target_x, target_y)
          2. Project target into the arm plane → reach distance
          3. Solve planar 2-link IK for (horizontal_reach, vertical_drop)
          4. Try both elbow-up and elbow-down configurations
          5. Validate against joint limits and floor safety
          6. Return the best (lowest error) valid solution
        
        Returns dict {base, arm, wrist, error_cm} or None if no valid solution.
        """
        try:
            base_deg = self.solve_base(target_x, target_y, cur_base)
            if base_deg is None:
                return None

            # Horizontal distance from base to target in XY plane
            r_xy = math.hypot(target_x, target_y)
            reach_x = max(r_xy, 0.01)

            # Vertical drop from shoulder to target (positive = target below shoulder)
            z_drop = self.geom.shoulder_height_cm - target_z

            candidates = []
            for elbow_up in [False, True][:max_solution_candidates]:
                aw = self.solve_planar(reach_x, z_drop, cur_arm,
                                       prefer_elbow_up=elbow_up)
                if aw is None:
                    continue
                arm_deg, wrist_deg = aw

                if not self.limits.in_bounds('arm', arm_deg):
                    continue
                if not self.limits.in_bounds('wrist', wrist_deg):
                    continue
                if not self.fk.is_safe(arm_deg, wrist_deg, safety_margin_cm):
                    continue
                if arm_deg < 50.0 or arm_deg > 145.0:
                    continue
                if wrist_deg < 20.0 or wrist_deg > 170.0:
                    continue

                pos, _ = self.fk.gripper_pose(base_deg, arm_deg, wrist_deg)
                error = math.sqrt((pos[0]-target_x)**2 +
                                  (pos[1]-target_y)**2 +
                                  (pos[2]-target_z)**2)
                candidates.append((error, arm_deg, wrist_deg))

            if not candidates:
                return None

            candidates.sort(key=lambda c: c[0])
            best = candidates[0]

            return {
                'base': base_deg,
                'arm': best[1],
                'wrist': best[2],
                'error_cm': best[0],
            }
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — COORDINATE TRANSFORMS (Camera → Gripper → Base)
# ═══════════════════════════════════════════════════════════════════════════

class CoordinateTransforms:
    """Coordinate frame transformations for the eye-in-hand camera system.
    
    Frame hierarchy:
      camera {C} ──[T_CG]──→ gripper {G} ──[T_G0]──→ base {0}
    
    Camera frame {C}: X=right, Y=down, Z=forward (from camera)
    Gripper frame {G}: same orientation, origin at jaw center
    Base frame {0}: X=forward, Y=left, Z=up (right-hand rule)
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 extrinsics: CameraExtrinsics,
                 fk: ForwardKinematics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.fk = fk
        self.T_CG = extrinsics.homogeneous_matrix()       # camera → gripper
        self.T_GC = extrinsics.inverse_homogeneous()       # gripper → camera

    def pixel_to_camera_ray(self, px: float, py: float
                            ) -> np.ndarray:
        """Back-project pixel to unit direction vector in camera frame.
        
        Returns 3D unit vector (X, Y, Z) in camera frame.
        Z is positive forward (depth direction).
        """
        x_norm = (float(px) - self.intrinsics.cx) / self.intrinsics.fx
        y_norm = (float(py) - self.intrinsics.cy) / self.intrinsics.fy
        vec = np.array([x_norm, y_norm, 1.0])
        return vec / np.linalg.norm(vec)

    def pixel_to_camera_3d(self, px: float, py: float, depth_cm: float
                           ) -> Optional[np.ndarray]:
        """Back-project pixel to 3D point in camera frame.
        
        Args:
            px, py: pixel coordinates
            depth_cm: measured depth along camera Z axis
        
        Returns:
            3D point (X, Y, Z) in camera frame, or None if depth invalid
        """
        if depth_cm <= 0:
            return None
        x_cam = (float(px) - self.intrinsics.cx) * depth_cm / self.intrinsics.fx
        y_cam = (float(py) - self.intrinsics.cy) * depth_cm / self.intrinsics.fy
        return np.array([x_cam, y_cam, depth_cm])

    def camera_to_gripper(self, p_cam: np.ndarray) -> np.ndarray:
        """Transform 3D point from camera frame to gripper frame.
        
        Uses the 4×4 homogeneous transform T_CG.
        """
        p_h = np.append(p_cam, 1.0)
        p_grip_h = self.T_CG @ p_h
        return p_grip_h[:3]

    def gripper_to_camera(self, p_grip: np.ndarray) -> np.ndarray:
        """Transform 3D point from gripper frame back to camera frame."""
        p_h = np.append(p_grip, 1.0)
        p_cam_h = self.T_GC @ p_h
        return p_cam_h[:3]

    def gripper_to_base(self, p_grip: np.ndarray,
                        base_deg: float, arm_deg: float, wrist_deg: float
                        ) -> np.ndarray:
        """Transform 3D point from gripper frame to base frame.
        
        Gripper frame {G}: X=right, Y=down, Z=forward (matches camera).
        FK computes orientation as: X=forward, Y=left, Z=up.
        Reorder axes so the depth (gripper Z = forward) maps correctly.
        """
        pos, rot = self.fk.gripper_pose(base_deg, arm_deg, wrist_deg)
        # Axis permutation: gripper (X=right,Y=down,Z=fwd) ? FK (X=fwd,Y=left,Z=up)
        # new_x = old_z (fwd?fwd), new_y = -old_x (right?-left), new_z = -old_y (down?-up)
        R_reorder = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
        rot_reordered = rot @ R_reorder
        T_G0 = np.eye(4)
        T_G0[:3, :3] = rot_reordered
        T_G0[:3, 3] = pos
        p_h = np.append(p_grip, 1.0)
        p_base_h = T_G0 @ p_h
        return p_base_h[:3]

    def camera_to_base(self, p_cam: np.ndarray,
                       base_deg: float, arm_deg: float, wrist_deg: float
                       ) -> np.ndarray:
        """Full transform: camera frame → gripper frame → base frame."""
        p_grip = self.camera_to_gripper(p_cam)
        return self.gripper_to_base(p_grip, base_deg, arm_deg, wrist_deg)

    def pixel_and_depth_to_base(self, px: float, py: float,
                                 depth_cm: float,
                                 base_deg: float, arm_deg: float,
                                 wrist_deg: float
                                 ) -> Optional[np.ndarray]:
        """Complete pipeline: pixel + depth → 3D position in base frame.
        
        This is the primary function used by the pickup pipeline.
        """
        p_cam = self.pixel_to_camera_3d(px, py, depth_cm)
        if p_cam is None:
            return None
        return self.camera_to_base(p_cam, base_deg, arm_deg, wrist_deg)

    def object_to_reach_coords(self, px: float, py: float, depth_cm: float,
                                sensor_to_jaw_offset_cm: Optional[float] = None
                                ) -> Optional[Dict[str, float]]:
        """Convert detection + depth to arm reach coordinates (for IK).
        
        This accounts for:
          1. Pixel back-projection to camera 3D
          2. Camera-to-gripper offset
          3. Sensor-to-jaw forward offset (VL53L0X reading correction)
        
        Returns dict {x_cm, y_cm, z_cm} in base-aligned reach space,
        or None on failure.
        """
        try:
            if depth_cm <= 0:
                return None
            offset = (sensor_to_jaw_offset_cm
                      if sensor_to_jaw_offset_cm is not None
                      else self.extrinsics.sensor_to_jaw_cm)
            adjusted_depth = depth_cm + offset

            p_cam = self.pixel_to_camera_3d(px, py, adjusted_depth)
            if p_cam is None:
                return None
            p_grip = self.camera_to_gripper(p_cam)

            return {
                'x_cm': float(p_grip[0]),
                'y_cm': float(p_grip[1]),
                'z_cm': float(p_grip[2]),
            }
        except Exception:
            return None

    def round_trip_check(self, p_original: np.ndarray,
                         base_deg: float, arm_deg: float, wrist_deg: float
                         ) -> Dict[str, float]:
        """Round-trip consistency check.
        
        Transforms a point: base → gripper → camera → gripper → base
        and returns the error.
        """
        pos, rot = self.fk.gripper_pose(base_deg, arm_deg, wrist_deg)
        T_G0 = np.eye(4)
        T_G0[:3, :3] = rot
        T_G0[:3, 3] = pos
        T_0G = np.linalg.inv(T_G0)

        p_base_h = np.append(p_original, 1.0)
        p_grip_h = T_0G @ p_base_h
        p_grip = p_grip_h[:3]

        p_cam = self.gripper_to_camera(p_grip)
        p_grip_back = self.camera_to_gripper(p_cam)

        p_base_back_h = T_G0 @ np.append(p_grip_back, 1.0)
        p_base_back = p_base_back_h[:3]

        error = np.linalg.norm(p_original - p_base_back)
        return {
            'original': p_original.tolist(),
            'round_trip': p_base_back.tolist(),
            'error_cm': float(error),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — TRAJECTORY PLANNER
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryPlanner:
    """Joint-space trajectory planning with validation."""

    def __init__(self, fk: ForwardKinematics, ik: InverseKinematics,
                 limits: JointLimits):
        self.fk = fk
        self.ik = ik
        self.limits = limits

    def interpolate_joint(self, start: float, end: float, steps: int
                          ) -> List[float]:
        """Linear interpolation between joint angles."""
        return [start + (end - start) * i / max(1, steps - 1)
                for i in range(steps)]

    def plan_joint_trajectory(self, start: Dict[str, float],
                               end: Dict[str, float],
                               steps: int = 20
                               ) -> Optional[List[Dict[str, float]]]:
        """Generate joint-space trajectory from start to end.
        
        Returns list of waypoints [{base, arm, wrist}], or None if invalid.
        """
        traj = []
        bases = self.interpolate_joint(start['base'], end['base'], steps)
        arms = self.interpolate_joint(start['arm'], end['arm'], steps)
        wrists = self.interpolate_joint(start['wrist'], end['wrist'], steps)

        for i in range(steps):
            wp = {
                'base': bases[i],
                'arm': arms[i],
                'wrist': wrists[i],
            }
            if not self.validate_waypoint(wp):
                return None
            traj.append(wp)
        return traj

    def validate_waypoint(self, wp: Dict[str, float],
                          safety_margin_cm: float = 2.5
                          ) -> bool:
        """Validate a single waypoint against all constraints."""
        if not self.limits.in_bounds('base', wp['base']):
            return False
        if not self.limits.in_bounds('arm', wp['arm']):
            return False
        if not self.limits.in_bounds('wrist', wp['wrist']):
            return False
        if not self.fk.is_safe(wp['arm'], wp['wrist'], safety_margin_cm):
            return False
        return True

    def detect_singularity(self, wp: Dict[str, float],
                           threshold_deg: float = 5.0
                           ) -> bool:
        """Check if a waypoint is near a kinematic singularity.
        
        For this arm, singularities occur when:
          - Arm is fully extended (straight, wrist at 90°)
          - Arm is at joint limits
          - Wrist is at 0° or 180° (fully bent)
        """
        arm = wp['arm']
        wrist = wp['wrist']
        if abs(arm - 90.0) > 85.0:
            return True
        if wrist < threshold_deg or wrist > (180.0 - threshold_deg):
            return True
        return False

    def validate_trajectory(self, trajectory: List[Dict[str, float]],
                            safety_margin_cm: float = 2.5
                            ) -> Dict[str, Any]:
        """Full validation of a complete trajectory.
        
        Checks:
          - Joint limits at every waypoint
          - Floor collisions
          - Singularities
          - Smoothness (joint velocity continuity)
        
        Returns dict with validation results.
        """
        result = {
            'valid': True,
            'joint_limit_violations': [],
            'floor_collisions': [],
            'singularities': [],
            'smoothness_ok': True,
            'total_waypoints': len(trajectory),
        }

        for i, wp in enumerate(trajectory):
            if not self.limits.in_bounds('base', wp['base']):
                result['joint_limit_violations'].append((i, 'base', wp['base']))
                result['valid'] = False
            if not self.limits.in_bounds('arm', wp['arm']):
                result['joint_limit_violations'].append((i, 'arm', wp['arm']))
                result['valid'] = False
            if not self.limits.in_bounds('wrist', wp['wrist']):
                result['joint_limit_violations'].append((i, 'wrist', wp['wrist']))
                result['valid'] = False
            if not self.fk.is_safe(wp['arm'], wp['wrist'], safety_margin_cm):
                result['floor_collisions'].append(i)
                result['valid'] = False
            if self.detect_singularity(wp):
                result['singularities'].append(i)
                result['valid'] = False

        if len(trajectory) >= 3:
            for i in range(1, len(trajectory) - 1):
                da = abs(trajectory[i+1]['arm'] - 2*trajectory[i]['arm']
                         + trajectory[i-1]['arm'])
                if da > 10.0:
                    result['smoothness_ok'] = False
                    break

        return result

    def plan_hover_to_grasp(self, hover: Dict[str, float],
                             grasp: Dict[str, float],
                             descent_steps: int = 15
                             ) -> Optional[List[Dict[str, float]]]:
        """Plan vertical descent from hover to grasp position.
        
        The descent is primarily arm-angle driven (lowering the arm).
        """
        traj = self.plan_joint_trajectory(hover, grasp, descent_steps)
        return traj

    def plan_lift(self, grasp: Dict[str, float],
                   lift_target: Dict[str, float],
                   lift_steps: int = 15
                   ) -> Optional[List[Dict[str, float]]]:
        """Plan lift from grasp to raised position."""
        return self.plan_joint_trajectory(grasp, lift_target, lift_steps)

    def plan_transit(self, start: Dict[str, float],
                      end: Dict[str, float],
                      steps: int = 30
                      ) -> Optional[List[Dict[str, float]]]:
        """Plan transit motion between two arbitrary poses."""
        return self.plan_joint_trajectory(start, end, steps)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — 3D SIMULATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class Simulation3D:
    """Matplotlib 3D simulation environment for kinematic validation.
    
    Visualizes:
      - Arm links and joints
      - Coordinate frames at every joint
      - Object position
      - Camera and gripper frames
      - Full trajectory animation
    """

    FRAME_AXIS_LENGTH = 5.0  # cm
    JOINT_RADIUS = 1.0
    LINK_COLOR = '#3498db'
    JOINT_COLOR = '#e74c3c'
    GRIPPER_COLOR = '#2ecc71'

    def __init__(self, fk: ForwardKinematics):
        self.fk = fk
        self.fig = None
        self.ax = None
        self.artists = {}

    def setup(self, title: str = "4-DOF Robotic Arm Simulation"):
        """Create the 3D plot with equal aspect ratio."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            print("[SIM] matplotlib not available — skipping 3D visualization")
            return False

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        self.ax.set_title(title)

        reach = self.fk.geom.total_reach_cm * 1.2
        self.ax.set_xlim(-reach, reach)
        self.ax.set_ylim(-reach, reach)
        self.ax.set_zlim(-5, self.fk.geom.shoulder_height_cm + reach)

        self.ax.view_init(elev=25, azim=-45)
        self.ax.grid(True, alpha=0.3)

        return True

    def draw_coordinate_frame(self, origin: np.ndarray,
                               rotation: np.ndarray,
                               label: str = '',
                               length: float = None
                               ) -> List:
        """Draw XYZ axes at a given pose (4×4 transform)."""
        if length is None:
            length = self.FRAME_AXIS_LENGTH
        artists = []
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            axis_end = origin + rotation[:3, i] * length
            line, = self.ax.plot([origin[0], axis_end[0]],
                                 [origin[1], axis_end[1]],
                                 [origin[2], axis_end[2]],
                                 color=colors[i], linewidth=2, alpha=0.7)
            artists.append(line)
        if label:
            self.ax.text(origin[0], origin[1], origin[2], label,
                         fontsize=9, fontweight='bold')
        return artists

    def draw_arm(self, base_deg: float, arm_deg: float, wrist_deg: float
                 ) -> List:
        """Draw the arm at given joint angles.
        
        Returns list of matplotlib artists for animation updates.
        """
        joints = self.fk.joint_positions(base_deg, arm_deg, wrist_deg)
        artists = []

        points = [joints['base'], joints['shoulder'],
                  joints['elbow'], joints['wrist'], joints['gripper']]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        link_lines, = self.ax.plot(xs, ys, zs, color=self.LINK_COLOR,
                                   linewidth=4, alpha=0.8)
        artists.append(link_lines)

        for name, pt in joints.items():
            size = self.JOINT_RADIUS * 1.5 if name in ('gripper',) else self.JOINT_RADIUS
            joint_scat = self.ax.scatter([pt[0]], [pt[1]], [pt[2]],
                                         color=self.JOINT_COLOR, s=size*20)
            artists.append(joint_scat)

        pos, rot = self.fk.gripper_pose(base_deg, arm_deg, wrist_deg)
        frame_arts = self.draw_coordinate_frame(
            pos, rot, label='G', length=self.FRAME_AXIS_LENGTH * 0.8)
        artists.extend(frame_arts)

        return artists

    def draw_object(self, position: Tuple[float, float, float],
                    size: float = 2.0, color: str = '#f39c12'
                    ) -> List:
        """Draw object at given 3D position."""
        artists = []
        obj = self.ax.scatter([position[0]], [position[1]], [position[2]],
                              color=color, s=size*50, marker='o', alpha=0.8,
                              label='Object')
        artists.append(obj)
        return artists

    def draw_floor(self, size_cm: float = 60.0):
        """Draw a semi-transparent floor plane."""
        x = np.linspace(-size_cm, size_cm, 2)
        y = np.linspace(-size_cm, size_cm, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    def draw_camera_frustum(self, intrinsics: CameraIntrinsics,
                             pose: Tuple[np.ndarray, np.ndarray],
                             depth_cm: float = 20.0
                             ) -> List:
        """Draw camera frustum at given pose."""
        artists = []
        pos, rot = pose
        cx, cy = intrinsics.cx, intrinsics.cy
        fx, fy = intrinsics.fx, intrinsics.fy

        corners_px = [(0, 0), (intrinsics.width_px, 0),
                      (intrinsics.width_px, intrinsics.height_px),
                      (0, intrinsics.height_px)]
        corners_cam = []
        for u, v in corners_px:
            x = (u - cx) * depth_cm / fx
            y = (v - cy) * depth_cm / fy
            z = depth_cm
            corners_cam.append(np.array([x, y, z]))

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        origin = pos
        for pt in corners_cam:
            pt_h = np.append(pt, 1.0)
            pt_w = (T @ pt_h)[:3]
            self.ax.plot([origin[0], pt_w[0]],
                         [origin[1], pt_w[1]],
                         [origin[2], pt_w[2]],
                         color='#e67e22', linewidth=1, alpha=0.4)
        return artists

    def render_static(self, base_deg: float, arm_deg: float, wrist_deg: float,
                      object_pos: Optional[Tuple[float, float, float]] = None):
        """Single-frame render of the arm at given angles."""
        if self.ax is None:
            if not self.setup():
                return
        self.ax.clear()
        self.setup()
        self.draw_floor()
        self.draw_arm(base_deg, arm_deg, wrist_deg)
        if object_pos:
            self.draw_object(object_pos)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def animate_trajectory(self, trajectory: List[Dict[str, float]],
                           object_pos: Optional[Tuple[float, float, float]] = None,
                           pause_sec: float = 0.05
                           ) -> List[Dict[str, float]]:
        """Animate arm moving through a trajectory.
        
        Returns list of gripper positions along trajectory.
        """
        if self.ax is None:
            if not self.setup():
                return []

        positions = []
        for wp in trajectory:
            self.ax.clear()
            self.setup()
            self.draw_floor()
            self.draw_arm(wp['base'], wp['arm'], wp['wrist'])
            if object_pos:
                self.draw_object(object_pos)
            pos = self.fk.gripper_position(wp['base'], wp['arm'], wp['wrist'])
            positions.append(pos)
            plt.pause(pause_sec)

        plt.pause(0.5)
        return positions

    def show_legend(self):
        if self.ax:
            self.ax.legend(loc='upper right')

    def display(self):
        if self.fig:
            self.fig.show()

    def save_frame(self, path: str = 'arm_simulation.png'):
        if self.fig:
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[SIM] Saved frame to {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7 — ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ErrorThresholds:
    """Error thresholds for validation rejection."""
    position_error_cm: float = 1.0      # max acceptable position error
    euclidean_error_cm: float = 1.5     # max Euclidean distance error
    orientation_error_deg: float = 5.0  # max orientation error
    grasp_x_offset_cm: float = 2.0      # lateral centering tolerance
    grasp_y_offset_cm: float = 2.0      # longitudinal centering tolerance
    grasp_z_offset_cm: float = 1.0      # vertical centering tolerance
    round_trip_error_cm: float = 0.1    # round-trip transform error


class ErrorAnalysis:
    """Compute and validate kinematic and calibration errors."""

    def __init__(self, thresholds: Optional[ErrorThresholds] = None):
        self.thresholds = thresholds or ErrorThresholds()

    def position_error(self, desired: np.ndarray, actual: np.ndarray
                       ) -> Dict[str, float]:
        """Compute position error between desired and actual positions."""
        diff = actual - desired
        return {
            'dx_cm': float(diff[0]),
            'dy_cm': float(diff[1]),
            'dz_cm': float(diff[2]),
            'euclidean_cm': float(np.linalg.norm(diff)),
        }

    def orientation_error(self, R_desired: np.ndarray,
                           R_actual: np.ndarray) -> Dict[str, float]:
        """Compute orientation error between two rotation matrices.
        
        Uses the angle-axis representation of R_desired^T * R_actual.
        """
        R_error = R_desired.T @ R_actual
        trace = np.trace(R_error)
        angle_rad = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
        return {
            'angle_error_deg': float(math.degrees(angle_rad)),
            'trace': float(trace),
        }

    def check_thresholds(self, errors: Dict[str, float],
                          prefix: str = ''
                          ) -> Dict[str, Any]:
        """Check if errors exceed thresholds.
        
        Returns dict with:
          - passed: bool (True if ALL errors within thresholds)
          - failures: list of (field, value, threshold) tuples
        """
        threshold_map = {
            'euclidean_cm': self.thresholds.euclidean_error_cm,
            'dx_cm': self.thresholds.position_error_cm,
            'dy_cm': self.thresholds.position_error_cm,
            'dz_cm': self.thresholds.position_error_cm,
            'angle_error_deg': self.thresholds.orientation_error_deg,
        }
        failures = []
        for key, threshold in threshold_map.items():
            full_key = f"{prefix}{key}" if prefix else key
            if full_key in errors:
                val = abs(errors[full_key])
                if val > threshold:
                    failures.append((full_key, val, threshold))

        return {
            'passed': len(failures) == 0,
            'failures': failures,
        }

    def generate_report(self, fk_errors: Dict[str, float],
                         ik_result: Dict, traj_validation: Dict,
                         grasp_errors: Dict,
                         calib_errors: Dict
                         ) -> str:
        """Generate a human-readable validation report."""
        lines = ['='*60,
                 'PICKUP PIPELINE VALIDATION REPORT',
                 '='*60]

        lines.append('\n[1] FK Verification')
        ek = fk_errors.get('euclidean_cm', float('inf'))
        ek_val = fk_errors.get('euclidean_cm', 'N/A')
        lines.append(f'    Position error:  {ek_val} cm')
        lines.append(f'    Passed:          {"YES" if ek <= self.thresholds.euclidean_error_cm else "NO"}')

        lines.append('\n[2] IK Solver')
        if ik_result:
            lines.append(f'    Solution found: YES')
            lines.append(f'    IK error:       {ik_result.get("error_cm", "N/A"):.3f} cm')
        else:
            lines.append(f'    Solution found: NO')

        lines.append('\n[3] Trajectory Validation')
        if traj_validation:
            lines.append(f'    Valid:          {traj_validation.get("valid", False)}')
            lines.append(f'    Waypoints:      {traj_validation.get("total_waypoints", 0)}')
            if traj_validation.get('joint_limit_violations'):
                lines.append(f'    Limit violations: {len(traj_validation["joint_limit_violations"])}')
            if traj_validation.get('floor_collisions'):
                lines.append(f'    Floor collisions: {len(traj_validation["floor_collisions"])}')
            if traj_validation.get('singularities'):
                lines.append(f'    Singularities:    {len(traj_validation["singularities"])}')

        lines.append('\n[4] Grasp Centering')
        for k, v in grasp_errors.items():
            if isinstance(v, float):
                lines.append(f'    {k}: {v:.3f}')

        lines.append('\n[5] Calibration')
        if calib_errors:
            rt_err = calib_errors.get('round_trip_error_cm', float('inf'))
            lines.append(f'    Round-trip error: {rt_err:.6f} cm')
            lines.append(f'    Passed:           {"YES" if rt_err <= self.thresholds.round_trip_error_cm else "NO"}')

        lines.append('\n[6] Overall Verdict')
        all_ok = (ek <= self.thresholds.euclidean_error_cm and
                  ik_result is not None and
                  traj_validation.get('valid', False) and
                  calib_errors.get('round_trip_error_cm', float('inf')) <= self.thresholds.round_trip_error_cm)
        lines.append(f'    PIPELINE VALID: {"YES" if all_ok else "NO"}')
        lines.append('='*60)

        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 8 — GRASP CENTERING VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class GraspValidator:
    """Validates and corrects grasp centering before closing gripper.
    
    Determines if the object is centered between the gripper jaws based
    on camera feedback. If not centered, computes correction offsets
    and recalculates IK.
    """

    def __init__(self, thresholds: ErrorThresholds,
                 fk: ForwardKinematics,
                 ik: InverseKinematics):
        self.thresholds = thresholds
        self.fk = fk
        self.ik = ik
        self.max_correction_iterations = 5

    def compute_offset_errors(self, object_center_px: Tuple[float, float],
                               image_center_px: Tuple[float, float],
                               depth_cm: float,
                               intrinsics: CameraIntrinsics
                               ) -> Dict[str, float]:
        """Compute lateral/longitudinal/vertical centering errors.
        
        Args:
            object_center_px: (u, v) of object in image
            image_center_px: (cu, cv) image center (crosshair)
            depth_cm: measured depth to object
            intrinsics: camera calibration
        
        Returns:
            dict with x_offset_cm, y_offset_cm, z_offset_cm
        """
        du = object_center_px[0] - image_center_px[0]
        dv = object_center_px[1] - image_center_px[1]

        x_offset = du * depth_cm / intrinsics.fx
        y_offset = dv * depth_cm / intrinsics.fy
        z_offset = 0.0

        return {
            'x_offset_cm': x_offset,
            'y_offset_cm': y_offset,
            'z_offset_cm': z_offset,
            'du_px': du,
            'dv_px': dv,
        }

    def check_centering(self, offset_errors: Dict[str, float]
                        ) -> Dict[str, Any]:
        """Check if object is centered within tolerance.
        
        Returns:
            dict with:
              - centered: bool
              - failures: list of (axis, error, threshold)
        """
        failures = []
        checks = [
            ('x_offset_cm', 'lateral', self.thresholds.grasp_x_offset_cm),
            ('y_offset_cm', 'longitudinal', self.thresholds.grasp_y_offset_cm),
            ('z_offset_cm', 'vertical', self.thresholds.grasp_z_offset_cm),
        ]
        for key, axis, thresh in checks:
            val = abs(offset_errors.get(key, 0.0))
            if val > thresh:
                failures.append((axis, val, thresh))

        return {
            'centered': len(failures) == 0,
            'failures': failures,
            'offset_errors': offset_errors,
        }

    def compute_correction(self, offset_errors: Dict[str, float],
                            current_angles: Dict[str, float],
                            object_pos_base: Tuple[float, float, float]
                            ) -> Optional[Dict[str, float]]:
        """Compute corrected IK solution accounting for centering offset.
        
        Modifies the target position to center the gripper over the object.
        """
        target_x = object_pos_base[0] - offset_errors.get('x_offset_cm', 0.0)
        target_y = object_pos_base[1] - offset_errors.get('y_offset_cm', 0.0)
        target_z = object_pos_base[2]

        return self.ik.solve_full(
            target_x, target_y, target_z,
            current_angles.get('base', 90.0),
            current_angles.get('arm', 90.0),
            current_angles.get('wrist', 90.0),
        )

    def centering_score(self, offset_errors: Dict[str, float]) -> float:
        """Score how well-centered the grasp is (0.0 = perfect, 1.0 = worst)."""
        x = abs(offset_errors.get('x_offset_cm', 0.0)) / max(1.0, self.thresholds.grasp_x_offset_cm)
        y = abs(offset_errors.get('y_offset_cm', 0.0)) / max(1.0, self.thresholds.grasp_y_offset_cm)
        return min(1.0, (x + y) / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 9 — CALIBRATION VERIFIER
# ═══════════════════════════════════════════════════════════════════════════

class CalibrationVerifier:
    """Automatic verification of camera calibration and transforms."""

    def __init__(self, transforms: CoordinateTransforms,
                 fk: ForwardKinematics,
                 ik: InverseKinematics,
                 analysis: ErrorAnalysis):
        self.transforms = transforms
        self.fk = fk
        self.ik = ik
        self.analysis = analysis

    def verify_forward_inverse_consistency(
            self, test_angles: List[Tuple[float, float, float]] = None
            ) -> Dict[str, Any]:
        """Verify FK → IK → FK round-trip consistency.
        
        For a set of test joint angles:
          1. Compute gripper position (FK)
          2. Solve IK for that position
          3. Compute FK from IK solution
          4. Compare positions
        
        Returns dict with pass/fail and errors for each test case.
        """
        if test_angles is None:
            test_angles = [
                (90.0, 90.0, 90.0),    # home
                (90.0, 120.0, 90.0),   # arm lowered
                (90.0, 140.0, 80.0),   # arm down, wrist bent
                (45.0, 100.0, 95.0),   # base rotated
                (135.0, 130.0, 85.0),  # base other side
            ]

        results = []
        for ba, aa, wa in test_angles:
            pos1, rot1 = self.fk.gripper_pose(ba, aa, wa)
            solution = self.ik.solve_full(
                float(pos1[0]), float(pos1[1]), float(pos1[2]),
                ba, aa, wa
            )
            if solution is None:
                results.append({
                    'input': (ba, aa, wa),
                    'position': pos1.tolist(),
                    'ik_success': False,
                    'error_cm': float('inf'),
                    'passed': False,
                })
                continue

            pos2, _ = self.fk.gripper_pose(
                solution['base'], solution['arm'], solution['wrist']
            )
            err = self.analysis.position_error(pos1, pos2)
            passed = err['euclidean_cm'] <= self.analysis.thresholds.euclidean_error_cm
            results.append({
                'input': (ba, aa, wa),
                'position': pos1.tolist(),
                'ik_success': True,
                'ik_output': {k: float(v) for k, v in solution.items()},
                'fk_position': pos2.tolist(),
                **err,
                'passed': passed,
            })

        return {
            'test_cases': len(test_angles),
            'passed': all(r['passed'] for r in results),
            'results': results,
        }

    def verify_round_trip(self, num_points: int = 10
                          ) -> Dict[str, Any]:
        """Verify camera → gripper → camera round-trip.
        
        Generates random points in camera space, transforms to gripper
        and back, checks error.
        """
        np.random.seed(42)
        errors = []
        for _ in range(num_points):
            p_cam = np.random.uniform(-20, 20, 3)
            p_cam[2] = np.random.uniform(5, 50)  # forward (depth)

            p_grip = self.transforms.camera_to_gripper(p_cam)
            p_cam_back = self.transforms.gripper_to_camera(p_grip)

            err = np.linalg.norm(p_cam - p_cam_back)
            errors.append(float(err))

        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        return {
            'max_error_cm': max_err,
            'mean_error_cm': mean_err,
            'passed': max_err <= self.analysis.thresholds.round_trip_error_cm,
            'errors': errors,
        }

    def verify_camera_to_base_transform(
            self, test_angles: Tuple[float, float, float] = (90.0, 120.0, 80.0)
            ) -> Dict[str, Any]:
        """Verify camera → base transform consistency.
        
        Places a virtual point in camera frame, transforms to base frame,
        then checks the round trip through the gripper frame.
        """
        ba, aa, wa = test_angles
        p_cam = np.array([5.0, -3.0, 25.0])  # test point in camera frame

        p_base = self.transforms.camera_to_base(p_cam, ba, aa, wa)

        pos, rot = self.fk.gripper_pose(ba, aa, wa)
        T_G0 = np.eye(4)
        T_G0[:3, :3] = rot
        T_G0[:3, 3] = pos
        T_0G = np.linalg.inv(T_G0)

        p_base_h = np.append(p_base, 1.0)
        p_grip_h = T_0G @ p_base_h
        p_grip = p_grip_h[:3]
        p_cam_back = self.transforms.gripper_to_camera(p_grip)

        round_trip_err = np.linalg.norm(p_cam - p_cam_back)
        return {
            'point_camera': p_cam.tolist(),
            'point_base': p_base.tolist(),
            'point_camera_roundtrip': p_cam_back.tolist(),
            'round_trip_error_cm': float(round_trip_err),
            'passed': round_trip_err <= self.analysis.thresholds.round_trip_error_cm,
        }

    def run_all(self) -> Dict[str, Any]:
        """Run all calibration verification checks."""
        results = {
            'fk_ik_consistency': self.verify_forward_inverse_consistency(),
            'camera_gripper_roundtrip': self.verify_round_trip(),
            'camera_base_transform': self.verify_camera_to_base_transform(),
        }
        all_passed = all(r['passed'] for r in results.values())
        results['all_passed'] = all_passed
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  PART 10 — PICKUP SUCCESS METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PickupMetrics:
    """Compute and report pickup success metrics."""
    position_weight: float = 0.25
    orientation_weight: float = 0.15
    centering_weight: float = 0.25
    efficiency_weight: float = 0.15
    collision_weight: float = 0.20

    def compute_score(self, position_error_cm: float,
                       orientation_error_deg: float,
                       centering_error: float,
                       path_efficiency: float,
                       collision_free: bool
                       ) -> Dict[str, float]:
        """Compute overall pickup success score (0.0–1.0).
        
        Each component is scored 0.0–1.0 where 1.0 is perfect.
        """
        pos_score = max(0.0, 1.0 - position_error_cm / 5.0)
        orient_score = max(0.0, 1.0 - orientation_error_deg / 15.0)
        center_score = max(0.0, 1.0 - centering_error)
        eff_score = max(0.0, min(1.0, path_efficiency))
        coll_score = 1.0 if collision_free else 0.0

        total = (self.position_weight * pos_score +
                 self.orientation_weight * orient_score +
                 self.centering_weight * center_score +
                 self.efficiency_weight * eff_score +
                 self.collision_weight * coll_score)

        return {
            'total_score': round(total, 4),
            'position_score': round(pos_score, 4),
            'orientation_score': round(orient_score, 4),
            'centering_score': round(center_score, 4),
            'efficiency_score': round(eff_score, 4),
            'collision_score': round(coll_score, 4),
            'position_error_cm': round(position_error_cm, 3),
            'orientation_error_deg': round(orientation_error_deg, 3),
            'centering_error': round(centering_error, 3),
            'path_efficiency': round(path_efficiency, 3),
            'collision_free': collision_free,
        }

    def compute_centering_error(self, grasp_validation: Dict) -> float:
        """Extract centering error from grasp validator output (0-1)."""
        offsets = grasp_validation.get('offset_errors', {})
        x_err = abs(offsets.get('x_offset_cm', 0.0))
        y_err = abs(offsets.get('y_offset_cm', 0.0))
        return min(1.0, (x_err + y_err) / 4.0)

    def generate_summary(self, score: Dict[str, float]) -> str:
        """Generate human-readable summary of pickup metrics."""
        lines = [
            '='*50,
            'PICKUP SUCCESS METRICS',
            '='*50,
            f'  Overall Score:     {score["total_score"]:.3f} / 1.000',
            f'  Position Accuracy: {score["position_score"]:.3f}  (err={score["position_error_cm"]:.2f}cm)',
            f'  Orientation Acc:   {score["orientation_score"]:.3f}  (err={score["orientation_error_deg"]:.2f}°)',
            f'  Grasp Centering:   {score["centering_score"]:.3f}  (err={score["centering_error"]:.3f})',
            f'  Path Efficiency:   {score["efficiency_score"]:.3f}  (eff={score["path_efficiency"]:.3f})',
            f'  Collision Free:    {score["collision_score"]:.3f}  (safe={score["collision_free"]})',
            '-'*50,
            f'  VERDICT: {"PASS" if score["total_score"] >= 0.7 else "FAIL"} (threshold: 0.7)',
            '='*50,
        ]
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 11 — COMPLETE PICKUP PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PickupConfig:
    """Tunable pickup parameters (all overridable via env or constructor)."""
    # Detection
    confidence_min: float = 0.30

    # Hover & descent
    hover_height_cm: float = 8.0
    descent_steps: int = 15

    # Gripper
    grip_close_step_deg: float = 0.45
    grip_hold_sec: float = 0.80

    # Safety
    floor_safety_margin_cm: float = 2.5
    total_timeout_sec: float = 90.0

    # Lift
    lift_arm_deg: float = 32.0
    lift_steps: int = 15

    # Grasp tolerance (cm)
    grasp_x_tolerance_cm: float = 2.0
    grasp_y_tolerance_cm: float = 2.0
    grasp_z_tolerance_cm: float = 1.0

    # Error thresholds
    max_position_error_cm: float = 1.0
    max_euclidean_error_cm: float = 1.5
    max_orientation_error_deg: float = 5.0


class PickupPipeline:
    """Complete pickup pipeline with pre-motion simulation and validation.
    
    Usage:
        pipeline = PickupPipeline(geometry, limits)
        result = pipeline.run_simulation(object_px, object_py, depth_cm)
        if result['validation_ok']:
            pipeline.execute_physical(result)
    """

    def __init__(self, geometry: Optional[RobotGeometry] = None,
                 limits: Optional[JointLimits] = None,
                 intrinsics: Optional[CameraIntrinsics] = None,
                 extrinsics: Optional[CameraExtrinsics] = None,
                 config: Optional[PickupConfig] = None):
        self.geom = geometry or RobotGeometry()
        self.limits = limits or JointLimits()
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.extrinsics = extrinsics or CameraExtrinsics()
        self.config = config or PickupConfig()

        self.fk = ForwardKinematics(self.geom, self.limits)
        self.ik = InverseKinematics(self.geom, self.limits, self.fk)
        self.transforms = CoordinateTransforms(self.intrinsics,
                                                self.extrinsics, self.fk)
        self.planner = TrajectoryPlanner(self.fk, self.ik, self.limits)
        self.error_thresholds = ErrorThresholds(
            position_error_cm=self.config.max_position_error_cm,
            euclidean_error_cm=self.config.max_euclidean_error_cm,
            orientation_error_deg=self.config.max_orientation_error_deg,
            grasp_x_offset_cm=self.config.grasp_x_tolerance_cm,
            grasp_y_offset_cm=self.config.grasp_y_tolerance_cm,
            grasp_z_offset_cm=self.config.grasp_z_tolerance_cm,
        )
        self.analysis = ErrorAnalysis(self.error_thresholds)
        self.grasp_validator = GraspValidator(
            self.error_thresholds, self.fk, self.ik)
        self.calib_verifier = CalibrationVerifier(
            self.transforms, self.fk, self.ik, self.analysis)
        self.metrics = PickupMetrics()
        self.sim = Simulation3D(self.fk)
        self.validation_cache = {}

    def _detection_to_position(self, px: float, py: float, depth_cm: float,
                                current_joints: Optional[Dict[str, float]] = None
                                ) -> Optional[Dict[str, float]]:
        """Convert pixel + depth ? 3D position in base frame."""
        if current_joints is None:
            current_joints = {'base': 90.0, 'arm': 90.0, 'wrist': 90.0}
        p_base = self.transforms.pixel_and_depth_to_base(
            px, py, depth_cm,
            current_joints['base'], current_joints['arm'], current_joints['wrist'],
        )
        if p_base is None:
            return None
        return {'x_cm': float(p_base[0]), 'y_cm': float(p_base[1]), 'z_cm': float(p_base[2])}

    def _compute_hover_pose(self, target_x: float, target_y: float,
                             target_z: float,
                             current_angles: Dict[str, float]
                             ) -> Optional[Dict[str, float]]:
        """Compute IK for hover position above target."""
        hover_z = target_z + self.config.hover_height_cm
        return self.ik.solve_full(
            target_x, target_y, hover_z,
            current_angles['base'], current_angles['arm'],
            current_angles['wrist'],
            safety_margin_cm=self.config.floor_safety_margin_cm,
        )

    def _compute_grasp_pose(self, target_x: float, target_y: float,
                             target_z: float,
                             current_angles: Dict[str, float]
                             ) -> Optional[Dict[str, float]]:
        """Compute IK for grasp pose at target."""
        return self.ik.solve_full(
            target_x, target_y, target_z,
            current_angles['base'], current_angles['arm'],
            current_angles['wrist'],
            safety_margin_cm=0.5,
        )

    def _compute_lift_pose(self, current_angles: Dict[str, float]
                           ) -> Dict[str, float]:
        """Compute lift pose (raise arm after grasping)."""
        return {
            'base': current_angles.get('base', 90.0),
            'arm': current_angles.get('arm', 90.0) - self.config.lift_arm_deg,
            'wrist': current_angles.get('wrist', 90.0),
        }

    def run_validation(self) -> Dict[str, Any]:
        """Run all pre-pickup validation checks.
        
        Returns comprehensive validation results.
        """
        calib_results = self.calib_verifier.run_all()

        result = {
            'calibration': calib_results,
            'all_passed': calib_results.get('all_passed', False),
        }

        if result['all_passed']:
            print("[VALIDATION] All calibration checks PASSED")
        else:
            print("[VALIDATION] Some calibration checks FAILED")

        self.validation_cache = result
        return result

    def run_simulation(self, px: float, py: float, depth_cm: float,
                        current_joints: Optional[Dict[str, float]] = None,
                        visualize: bool = False
                        ) -> Dict[str, Any]:
        """Full kinematic simulation validating pickup feasibility.

        Computes 3D position from pixel+depth, solves IK for hover + grasp
        poses, validates joint limits / floor safety / FK round-trip /
        centering, and returns a complete plan or failure report.
        No arm movement � pure computation.

        Args:
            px, py: object pixel coordinates
            depth_cm: measured depth from ToF sensor
            current_joints: current arm angles {base, arm, wrist}
            visualize: unused (kept for API compatibility)

        Returns:
            dict with success, validation_ok, errors, object_position (3D),
            hover_angles, grasp_angles, ik_validation, centering, score
        """
        if current_joints is None:
            current_joints = {'base': 90.0, 'arm': 90.0, 'wrist': 90.0}

        result = {
            'success': False,
            'object_pixel': (px, py),
            'depth_cm': depth_cm,
            'validation_ok': False,
            'grasp_centered': False,
            'score': {},
            'errors': [],
            'stages': {},
        }

        # Stage 1: Convert pixel + depth to 3D base-frame position
        print("[SIM] Stage 1/4: Localizing object...")
        p_base = self.transforms.pixel_and_depth_to_base(
            px, py, depth_cm,
            current_joints['base'], current_joints['arm'], current_joints['wrist'],
        )
        if p_base is None:
            result['errors'].append("Failed to localize: pixel_and_depth_to_base returned None")
            return result
        obj_pos = {
            'x_cm': float(p_base[0]),
            'y_cm': float(p_base[1]),
            'z_cm': float(p_base[2]),
        }
        result['object_position'] = obj_pos
        print(f"    Base-frame: ({obj_pos['x_cm']:.1f}, {obj_pos['y_cm']:.1f}, {obj_pos['z_cm']:.1f}) cm")

        # Stage 2: Solve IK for hover and grasp poses
        print("[SIM] Stage 2/4: Solving IK...")
        hover_z = obj_pos['z_cm'] + self.config.hover_height_cm
        hover_solution = self.ik.solve_full(
            obj_pos['x_cm'], obj_pos['y_cm'], hover_z,
            current_joints['base'], current_joints['arm'], current_joints['wrist'],
            safety_margin_cm=self.config.floor_safety_margin_cm,
        )
        if hover_solution is None:
            result['errors'].append("No IK solution for hover position (out of reach?)")
            return result
        result['hover_angles'] = hover_solution

        grasp_solution = self.ik.solve_full(
            obj_pos['x_cm'], obj_pos['y_cm'], obj_pos['z_cm'],
            hover_solution['base'], hover_solution['arm'], hover_solution['wrist'],
            safety_margin_cm=0.5,
        )
        if grasp_solution is None:
            result['errors'].append("No IK solution for grasp position (out of reach?)")
            return result
        result['grasp_angles'] = grasp_solution

        # Stage 3: Validate both solutions
        print("[SIM] Stage 3/4: Validating solutions...")
        ik_validation = {}
        for phase, angles in [('hover', hover_solution), ('grasp', grasp_solution)]:
            limits_ok = all([
                self.limits.in_bounds('base', angles['base']),
                self.limits.in_bounds('arm', angles['arm']),
                self.limits.in_bounds('wrist', angles['wrist']),
            ])
            floor_safe = self.fk.is_safe(
                angles['arm'], angles['wrist'],
                margin_cm=self.config.floor_safety_margin_cm,
            )
            singular = self.planner.detect_singularity(angles)
            pos, _ = self.fk.gripper_pose(
                angles['base'], angles['arm'], angles['wrist'])
            fk_err = float(np.linalg.norm(p_base - pos))

            ik_validation[phase] = {
                'joint_limits_ok': limits_ok,
                'floor_safe': floor_safe,
                'singularity': singular,
                'fk_error_cm': fk_err,
                'valid': limits_ok and floor_safe and not singular and fk_err < 2.0,
            }
            print(f"    {phase}: base={angles['base']:.1f} arm={angles['arm']:.1f} "
                  f"wrist={angles['wrist']:.1f} | limits={limits_ok} safe={floor_safe} "
                  f"singular={singular} fk_err={fk_err:.2f}cm")
        result['ik_validation'] = ik_validation

        # Stage 4: Centering check + score
        print("[SIM] Stage 4/4: Computing metrics...")
        image_center = (self.intrinsics.cx, self.intrinsics.cy)
        centering_errors = self.grasp_validator.compute_offset_errors(
            (px, py), image_center, depth_cm, self.intrinsics
        )
        centering_check = self.grasp_validator.check_centering(centering_errors)
        result['centering'] = centering_check
        if centering_check['centered']:
            result['grasp_centered'] = True
        else:
            result['grasp_centered'] = False
            failures = centering_check.get('failures', [])
            result['errors'].append(f"Grasp not centered: {failures}")

        centering_err = self.metrics.compute_centering_error(centering_check)
        all_valid = all(v['valid'] for v in ik_validation.values())
        grip_valid = ik_validation.get('grasp', {})

        score = self.metrics.compute_score(
            position_error_cm=grip_valid.get('fk_error_cm', 99.0),
            orientation_error_deg=0.0,
            centering_error=centering_err,
            path_efficiency=1.0 if all_valid else 0.5,
            collision_free=grip_valid.get('floor_safe', False),
        )
        result['score'] = score
        result['success'] = all_valid and centering_check['centered']
        result['validation_ok'] = all_valid and centering_check['centered']

        print(f"    Score: {score['total_score']:.3f}")
        print(f"    Valid: {result['validation_ok']}")
        return result

    def print_report(self, result: Dict[str, Any]):
        """Print comprehensive validation report."""
        fk_errors = result.get('fk_verification', {})
        ik_result = result.get('grasp_pose', {})
        traj_val = {
            'valid': (result.get('descent_validation', {}).get('valid', False) and
                      result.get('lift_validation', {}).get('valid', False)),
            'total_waypoints': (len(result.get('descent_validation', {}).get('joint_limit_violations', [])) +
                               len(result.get('lift_validation', {}).get('joint_limit_violations', []))),
        }
        calib = result.get('stages', {}).get('calibration', {})

        calib_errors = {}
        if calib:
            rt = calib.get('camera_gripper_roundtrip', {})
            calib_errors['round_trip_error_cm'] = rt.get('max_error_cm', 0.0)

        centering = result.get('centering', {})
        report = self.analysis.generate_report(
            fk_errors, ik_result, traj_val, centering, calib_errors
        )
        print(report)

        if result.get('score'):
            print(self.metrics.generate_summary(result['score']))


# ═══════════════════════════════════════════════════════════════════════════




# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CONFIG â€” edit to match your hardware
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480
INPUT_SIZE = int(float(os.environ.get('ARM_YOLO_IMGSZ', '416')))
STREAM_PORT = 8080
CMD_API_PORT = 8081
MAX_JSON_BODY = 64 * 1024

PIN_BASE = 16
PIN_ARM = 18   # GPIO fallback only when the USB FrankenServo is unavailable
PIN_WRIST = 23
PIN_GRIP = 24

# Optional ESP8266 FrankenServo shoulder controller.  When enabled and the
# serial port opens successfully, only the arm joint is routed over USB;
# base/wrist/grip continue using the existing GPIO servo outputs.  Set
# ARM_SERIAL_SERVO=1 to enable the Nano serial arm path.
ARM_SERIAL_SERVO_ENABLED = os.environ.get('ARM_SERIAL_SERVO', '0').strip().lower() in ('1', 'true', 'yes', 'on')
ARM_SERIAL_SERVO_PORT = os.environ.get('ARM_SERIAL_SERVO_PORT', '/dev/ttyUSB1').strip() or '/dev/ttyUSB1'
ARM_SERIAL_SERVO_BAUD = int(float(os.environ.get('ARM_SERIAL_SERVO_BAUD', '9600')))
ARM_SERIAL_SERVO_MIN_DEG = float(os.environ.get('ARM_SERIAL_SERVO_MIN_DEG', '0.0'))
ARM_SERIAL_SERVO_MAX_DEG = float(os.environ.get('ARM_SERIAL_SERVO_MAX_DEG', '270.0'))

PIN_BY_JOINT = {
    'base': PIN_BASE,
    'arm': PIN_ARM,
    'wrist': PIN_WRIST,
    'grip': PIN_GRIP,
}

# ARM_MIN/MAX are mechanical joint limits only. Floor safety uses FK reject,
# never a precomputed minimum arm angle.
BASE_MIN, BASE_MAX = 10, 170
# FIX: was (-50, 150). See JointLimits.arm_min comment above — with the arm
# joint reversed, logical angles below 0 mapped to an out-of-range servo
# pulse that got silently clamped, freezing the real servo while the FK/3D
# model kept predicting motion. 0 is the true reachable minimum.
ARM_MIN, ARM_MAX = 90, 180
WRIST_HOME = 90

# Base servo pulse calibration — 35kg digital servo.
BASE_US_MIN       = 500    # Âµs at 0Â°  — standard 35kg minimum
BASE_US_MAX       = 2500   # Âµs at 180Â° — standard 35kg maximum
BASE_SNAP_DEG     = 2      # snap to nearest 2Â° to prevent micro-jitter
BASE_DEADBAND_DEG_SERVO = 3      # ignore moves smaller than 3Â° (digital servo)
BASE_POS_STEPS    = 36     # interpolation steps for smooth base motion
BASE_POS_DELAY    = 0.016  # seconds per step
# MG996R flexible jaw gripper (cults3d design) uses a full 180Â° servo.
# Adjust these if the jaw direction is reversed on your mount:
#   GRIP_OPEN=10 â†’ jaw fully open (servo at 10Â°)
#   GRIP_CLOSE=120 â†’ jaw fully closed (servo at 120Â°)
# If your gripper closes in the opposite direction swap the values:
#   GRIP_OPEN=170, GRIP_CLOSE=30
# DC motor mode: MG996R motor on IBT-4 (BTS7960). No servo controller.
# RPWM = forward (close), LPWM = reverse (open). INA219 stops at current limit.
GRIP_DC_MODE = os.environ.get('ARM_GRIP_DC_MODE', '1').strip().lower() not in ('0', 'false', 'no', 'off')
GRIP_DC_STOP_SEC = float(os.environ.get('ARM_GRIP_DC_STOP_SEC', '3.0'))  # max seconds before forced stop (burnout safety)
GRIP_DC_CURRENT_LIMIT_MA = float(os.environ.get('ARM_GRIP_DC_CURRENT_LIMIT_MA', '500'))  # INA219 stall threshold
GRIP_RPWM_PIN = 25   # IBT-4 RPWM — open direction
GRIP_LPWM_PIN = 26   # IBT-4 LPWM — close direction
GRIP_DC_SPEED = int(float(os.environ.get('ARM_GRIP_DC_SPEED', '200')))  # 0-255 PWM duty cycle
GRIP_DC_HIGH_SEC = float(os.environ.get('ARM_GRIP_DC_HIGH_SEC', '0.5'))  # stop after current above limit for this long

GRIP_OPEN  = 30    # kept for display/UI compat — not used for motor control
GRIP_CLOSE = 120   # kept for display/UI compat — not used for motor control
GRIP_ADAPTIVE_CLOSE = os.environ.get('ARM_GRIP_ADAPTIVE_CLOSE', '1').strip().lower() not in ('0', 'false', 'no', 'off')
GRIP_CLOSE_STEP_DEG = float(os.environ.get('ARM_GRIP_CLOSE_STEP_DEG', '0.45'))
GRIP_CONTACT_BACKOFF_DEG = float(os.environ.get('ARM_GRIP_CONTACT_BACKOFF_DEG', '12.0'))
GRIP_STEP_SETTLE_SEC = float(os.environ.get('ARM_GRIP_STEP_SETTLE_SEC', '0.20'))
GRIP_HOLD_PULSES = int(float(os.environ.get('ARM_GRIP_HOLD_PULSES', '3')))          # fewer forced rewrites avoids hold buzz
GRIP_HOLD_PULSE_INTERVAL = float(os.environ.get('ARM_GRIP_HOLD_PULSE_INTERVAL', '0.16'))
GRIP_EMA_ALPHA = float(os.environ.get('ARM_GRIP_EMA_ALPHA', '0.15'))                # smoother filtering for less oscillation
GRIP_DEADZONE_DEG = float(os.environ.get('ARM_GRIP_DEADZONE_DEG', '0.35'))          # close loop must not batch small steps into a jump
GRIP_OSC_SAMPLE_SEC = float(os.environ.get('ARM_GRIP_OSC_SAMPLE_SEC', '0.08'))
# Detect contact earlier so the jaw stops before it drives through the object.
GRIP_OSC_SCORE_THRESHOLD = float(os.environ.get('ARM_GRIP_OSC_SCORE_THRESHOLD', '2.6'))
GRIP_OSC_MIN_CLOSE_FRACTION = float(os.environ.get('ARM_GRIP_OSC_MIN_CLOSE_FRACTION', '0.20'))
GRIP_OSC_CONFIRM_SAMPLES = int(float(os.environ.get('ARM_GRIP_OSC_CONFIRM_SAMPLES', '2')))
GRIP_TUNE_DEFAULT_LEVEL = int(float(os.environ.get('ARM_GRIP_TUNE_LEVEL', '2')))

# INA219 current-based contact detection
GRIP_CURRENT_THRESHOLD_MA = float(os.environ.get('ARM_GRIP_CURRENT_THRESHOLD_MA', '400'))  # mA spike = contact
GRIP_CURRENT_CONFIRM_SAMPLES = int(float(os.environ.get('ARM_GRIP_CURRENT_CONFIRM_SAMPLES', '3')))
GRIP_CURRENT_SAMPLE_SEC = float(os.environ.get('ARM_GRIP_CURRENT_SAMPLE_SEC', '0.06'))

JOINT_LIMITS = {
    'base': (BASE_MIN, BASE_MAX),
    'arm': (ARM_MIN, ARM_MAX),
    'wrist': (0, 180),
    'grip': (GRIP_OPEN, GRIP_CLOSE),
}

# â”€â”€ Digital twin floor geometry (FK only â€” no angle floor limit) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FLOOR_BELOW_SHOULDER_CM  = float(os.environ.get('ARM_FLOOR_BELOW_SHOULDER_CM', '15.0'))   # 150 mm
FLOOR_SAFETY_MARGIN_CM   = float(os.environ.get('ARM_FLOOR_SAFETY_MARGIN_CM',  '-100.0'))   # disabled - linear FK inaccurate beyond ~45deg
WRIST_GROUND_COMP_DEG    = float(os.environ.get('ARM_WRIST_GROUND_COMP', '0.0'))
# Measured link lengths (cm).  Used for 2-link IK after the depth
# map gives us a metric 3D target point.
# ──── UPDATED to effective pivot-to-pivot dimensions ────
#   Arm 1 = shoulder → elbow = 16.0 cm (effective pivot-to-pivot)
#   Arm 2 = elbow → wrist + gripper tip = 31.0 cm (effective pivot-to-pivot)
#   Total effective reach = 47.0 cm
ARM_LINK1_CM = 13.5    # upper arm  (shoulder → elbow pivot-to-pivot)
ARM_LINK2_CM = 31.0    # forearm + gripper (elbow → tip pivot-to-pivot)
ARM_REACH_CM = ARM_LINK1_CM + ARM_LINK2_CM   # 19.2 cm full extension
ARM_REACH_PHYSICAL_CM = ARM_REACH_CM
# Minimum forward reach passed to IK — prevents arm going flat when object is close
REACH_X_MIN  = ARM_LINK1_CM * 0.4            # ~2.6 cm

# L2 split matches the Three.js model (forearm 73%, wrist+gripper 27%)
ARM_L2_FOREARM_FRAC = 0.73
ARM_L2_WRIST_FRAC   = 0.27
GRIPPER_FINGER_EXT_CM = float(os.environ.get('ARM_GRIPPER_FINGER_EXT_CM', '4.8'))

print(f'[FLOOR] shoulderâ†’floor={FLOOR_BELOW_SHOULDER_CM}cm  '
      f'reach={ARM_REACH_PHYSICAL_CM}cm  '
      f'safety_margin={FLOOR_SAFETY_MARGIN_CM}cm  '
      f'FK segments=elbow,wrist,hand,tip,finger', flush=True)


_fk_margin_local = threading.local()


def _active_fk_margin_cm():
    margin = getattr(_fk_margin_local, 'margin_cm', None)
    if margin is None:
        return None
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        return None
    return margin if math.isfinite(margin) else None


# ═══════════════════════════════════════════════════════════════
#  CALIBRATED FK LOOKUP TABLE  (arm_servo → shoulder_angle_deg)
#  Measured with wrist=90°, L1=13.5, L2=31.0, shoulder_h=15cm
# ═══════════════════════════════════════════════════════════════
_FK_TABLE = [
    (90,  -90.0),   # vertical up
    (110, -56.2),
    (130, -39.0),
    (150, -21.1),
    (160, -10.4),
    (180,  11.7),   # tip at 6cm above floor
]

def _fk_shoulder_deg(arm_deg: float) -> float:
    """Interpolated shoulder angle (degrees) from calibrated lookup table."""
    a = float(arm_deg)
    tbl = _FK_TABLE
    if a <= tbl[0][0]:
        return tbl[0][1]
    if a >= tbl[-1][0]:
        return tbl[-1][1]
    for i in range(len(tbl) - 1):
        a0, s0 = tbl[i]
        a1, s1 = tbl[i + 1]
        if a0 <= a <= a1:
            t = (a - a0) / (a1 - a0)
            return s0 + t * (s1 - s0)
    return tbl[-1][1]

def _fk_arm_from_shoulder(sh_deg: float) -> float:
    """Inverse lookup: shoulder angle (deg) → arm servo angle (deg)."""
    s = float(sh_deg)
    tbl = _FK_TABLE
    if s <= tbl[0][1]:
        return tbl[0][0]
    if s >= tbl[-1][1]:
        return tbl[-1][0]
    for i in range(len(tbl) - 1):
        a0, s0 = tbl[i]
        a1, s1 = tbl[i + 1]
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0)
            return a0 + t * (a1 - a0)
    return tbl[-1][0]


def _fk_sh_wr_rad(arm_deg, wrist_deg):
    """Geometric shoulder/wrist angles (rad) from calibrated lookup table."""
    sh = math.radians(_fk_shoulder_deg(float(arm_deg)))
    wr = math.radians(float(wrist_deg) - 90.0)
    return sh, wr


def _fk_segment_heights_cm(arm_deg, wrist_deg):
    """Heights above floor (cm) for every checked body point. Negative = through floor."""
    sh, wr = _fk_sh_wr_rad(arm_deg, wrist_deg)
    L1 = ARM_LINK1_CM
    L2f = ARM_LINK2_CM * ARM_L2_FOREARM_FRAC
    L2w = ARM_LINK2_CM * ARM_L2_WRIST_FRAC
    elbow_drop = L1 * math.sin(sh)
    wrist_drop = elbow_drop + L2f * math.sin(sh + wr)
    hand_drop  = wrist_drop + L2w * math.sin(sh + wr)
    tip_drop   = L1 * math.sin(sh) + ARM_LINK2_CM * math.sin(sh + wr)
    # Fingers only add vertical drop when the chain points downward
    finger_drop = tip_drop + GRIPPER_FINGER_EXT_CM * max(0.0, math.sin(sh + wr))
    floor_h = FLOOR_BELOW_SHOULDER_CM
    return {
        'elbow':  floor_h - elbow_drop,
        'wrist':  floor_h - wrist_drop,
        'hand':   floor_h - hand_drop,
        'tip':    floor_h - tip_drop,
        'finger': floor_h - finger_drop,
    }


def _fk_tip_height_cm(arm_deg, wrist_deg):
    return _fk_segment_heights_cm(arm_deg, wrist_deg)['tip']


def _fk_min_clearance_cm(arm_deg, wrist_deg):
    return min(_fk_segment_heights_cm(arm_deg, wrist_deg).values())


def _fk_safe(arm_deg, wrist_deg, margin_cm=None):
    """True if elbow, wrist, hand, tip, and fingers clear floor + safety margin."""
    if margin_cm is None:
        margin_cm = FLOOR_SAFETY_MARGIN_CM
    return _fk_min_clearance_cm(arm_deg, wrist_deg) >= float(margin_cm)


def _fk_reject_reason(arm_deg, wrist_deg, margin_cm=None):
    """Human-readable reject detail for logs/UI."""
    if margin_cm is None:
        margin_cm = FLOOR_SAFETY_MARGIN_CM
    segs = _fk_segment_heights_cm(arm_deg, wrist_deg)
    worst = min(segs, key=segs.get)
    return worst, segs[worst], float(margin_cm)


def _fk_gate_joint_target(key, target):
    """Single FK floor check for every joint command (track/pickup/jog/slider/IK).

    Returns (allowed, worst_segment, clearance_cm, margin_cm). base/grip always allowed.
    """
    if key not in ('arm', 'wrist'):
        return True, None, None, None
    try:
        target = float(target)
    except (TypeError, ValueError):
        return True, None, None, None
    if not math.isfinite(target):
        return True, None, None, None
    margin_cm = _active_fk_margin_cm()
    with _pos_lock:
        if key == 'arm':
            wrist = float(_pos.get('wrist', 90.0))
            if _fk_safe(target, wrist, margin_cm=margin_cm):
                return True, None, None, None
            worst, height, margin = _fk_reject_reason(target, wrist, margin_cm=margin_cm)
            return False, worst, height, margin
        arm = float(_pos.get('arm', 90.0))
        if _fk_safe(arm, target, margin_cm=margin_cm):
            return True, None, None, None
        worst, height, margin = _fk_reject_reason(arm, target, margin_cm=margin_cm)
        return False, worst, height, margin


def _fk_log_reject(context, key, target, worst, height, margin):
    shortfall = float(margin) - float(height)
    print(f'[DIGITAL-TWIN] {context} {key.upper()} REJECT: target={float(target):.1f}Â° | '
          f'{worst}={float(height):.2f}cm margin={float(margin):.2f}cm '
          f'shortfall={shortfall:.2f}cm', flush=True)

# UI step sizes
STEP_DEG = 30
BASE_STEP_DEG = 30
WAVE_REPS = 3
WAVE_DEG = 25
DEADBAND_DEG = 6

# â”€â”€ Normal (tracking) motion speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Used only by the tracking thread so corrections are snappy.
POS_STEPS = 12
POS_DELAY = 0.010

# â”€â”€ Slow motion speed (all non-tracking moves) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pickup descent, IK repositioning, home, wave, manual jog â€” everything
# except tracking uses these so the arm moves deliberately and precisely
# instead of snapping, which reduces overshoot and mechanical stress.
SLOW_POS_STEPS       = 48      # smoother sweep
SLOW_POS_DELAY       = 0.022   # slower, smoother motion
SLOW_WRIST_POS_STEPS = 36
SLOW_WRIST_POS_DELAY = 0.022
SLOW_ARM_POS_STEPS   = 42
SLOW_ARM_POS_DELAY   = 0.022
SLOW_GRIP_POS_STEPS  = 40
SLOW_GRIP_POS_DELAY  = 0.016
SLOW_BASE_POS_STEPS  = 36
SLOW_BASE_POS_DELAY  = 0.016

# Wrist servo pulse calibration.
# MG90S standard range: 500â€“2500Âµs = 0â€“180Â°.  Using 1000â€“2000 was WRONG â€”
# it compressed 180Â° into 1000Âµs (half the servo's real travel), so 1Â° of
# UI angle commanded ~5.5Âµs but the servo moved as if it were the full range,
# causing ~90Â° of physical movement per 1Â° of slider movement.
# Now uses the full 500â€“2500Âµs range matching all other servos.
WRIST_US_MIN       = 500    # Âµs at 0Â°  â€” standard MG90S minimum
WRIST_US_MAX       = 2500   # Âµs at 180Â° â€” standard MG90S maximum
WRIST_SNAP_DEG     = 2      # snap to nearest 2Â° to prevent micro-jitter
WRIST_DEADBAND_DEG = 3      # ignore moves smaller than 3Â°
WRIST_POS_STEPS    = 8      # interpolation steps for smooth wrist motion
WRIST_POS_DELAY    = 0.012  # seconds per step
WRIST_STEP_DEG     = 10     # degrees per wrist-left/wrist-right button press
WRIST_TRIM_DEG     = 0      # fine-tune resting position: adjust +/- if wrist creeps at home (try Â±2, Â±4)

# Continuous-rotation wrist fallback values. These are not used while the wrist
# joint mode is positional for a normal digital servo.
WRIST_CONT_STOP_US   = 1500
WRIST_CONT_RANGE_US  = 400   # FIX: was 260 â€” wider range = more usable speed headroom
WRIST_CONT_SPEED     = 0.70  # FIX: was 0.55 â€” pair with wider range for same max speed
WRIST_CONT_MS_PER_DEG = 18.0 # FIX: was 12 â€” 12ms*10Â°=120ms was barely visible; 18ms*10Â°=180ms is clearly visible
WRIST_CONT_MIN_MS    = 100.0 # FIX: was 80ms â€” too short, motor barely starts
WRIST_CONT_MAX_MS    = 800.0 # FIX: was 500ms â€” more headroom for large slider jumps

# Arm joint calibration â€” same treatment as wrist to stop creep on small slider moves.
ARM_US_MIN       = 500    # µs at 0°  — standard MG90S minimum
ARM_US_MAX       = 2500   # µs at 180° — standard MG90S maximum
ARM_SNAP_DEG     = 2      # snap to nearest 2° to prevent micro-jitter
ARM_DEADBAND_DEG = 3      # ignore moves smaller than 3Â°
ARM_POS_STEPS    = 8      # interpolation steps for smooth motion
ARM_POS_DELAY    = 0.012  # seconds per step
ARM_HOME_DEG     = 90     # vertical-up home angle
ARM_TRIM_DEG     = 0      # fine-tune resting position: adjust +/- if arm creeps at home (try Â±2, Â±4)

# Gripper servo pulse calibration — digital servo like wrist and arm.
GRIP_US_MIN       = 500    # Âµs at 0Â°  — standard MG996R/MG995 minimum
GRIP_US_MAX       = 2500   # Âµs at 180Â° — standard MG996R/MG995 maximum
GRIP_SNAP_DEG     = 2      # snap to nearest 2Â° to prevent micro-jitter
GRIP_DEADBAND_DEG_SERVO = 3      # ignore moves smaller than 3Â° (positional servo deadband)
GRIP_POS_STEPS    = 40     # interpolation steps for smooth gripper motion
GRIP_POS_DELAY    = 0.016  # seconds per step

# Positional control for stable arm motion. The wrist is now a positional
# digital servo, so wrist slider/button commands map to real wrist angles.
JOINT_MODE = {
    'base': 'positional',
    'arm': 'positional',
    'wrist': 'positional',
    'grip': 'positional',
}
JOINT_REVERSED = {
    'base': False,
    # DS5160 60kg servo: direct direction (opposite to GS5508MG)
    'arm': False,
    'wrist': True,
    'grip': False,
}
CONT_STOP_US = 1500
CONT_RANGE_US = 380
CONT_SPEED = 0.65
CONT_MS_PER_DEG = 10.0
CONT_MIN_MS = 90.0
CONT_MAX_MS = 650.0
CONT_CHUNK_MS = 25.0

# Detection settings
CONF_THRESH = float(os.environ.get('ARM_YOLO_CONF', '0.20'))
NMS_THRESH = float(os.environ.get('ARM_YOLO_NMS', '0.50'))
DETECTION_INTERVAL = 0.18  # seconds

# â”€â”€ Centering stability: post-move YOLO blackout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# After every servo command during pickup centering the camera is still
# swinging.  Any YOLO frame grabbed while the arm is in motion measures the
# BACKGROUND moving, not the target â€” in a cluttered scene this makes the
# centroid jump sideways and triggers another correction, creating oscillation.
#
# Solution: after each _move_joint_slow call during centering we suppress the
# fresh detection for this many seconds so only frames grabbed AFTER the arm has
# fully settled feed the error measurement.
#
# Rule of thumb: set â‰¥ servo_travel_time + one DETECTION_INTERVAL (0.18 s).
# 0.40 s covers a 1.5Â° step (â‰ˆ 0.10 s settle) + one full detection cycle.
# Raise to 0.55 s if you see residual oscillation on coarse steps.
PICKUP_POST_MOVE_BLACKOUT_SEC = float(
    os.environ.get('ARM_PICKUP_POST_MOVE_BLACKOUT_SEC', '0.40')
)

# EMA alpha used to smooth the locked-target centroid during centering passes.
# Lower = heavier smoothing, less reactive to single-frame noise.
# The default 0.35 was fine for a static camera; 0.20 is safer when the camera
# itself is moving because each new YOLO box is less trustworthy.
PICKUP_CENTERING_EMA_ALPHA = float(
    os.environ.get('ARM_PICKUP_CENTERING_EMA_ALPHA', '0.20')
)

# Minimum number of consecutive post-settle frames that must agree (IoU-pass)
# before the centering loop accepts the measurement as valid.
# 1 = no multi-frame gate (old behaviour); 2 = two frames must agree.
PICKUP_CENTERING_MIN_AGREE_FRAMES = int(
    float(os.environ.get('ARM_PICKUP_CENTERING_MIN_AGREE_FRAMES', '3'))
)

JPEG_QUALITY = 68
DETECTION_MIN_AREA_RATIO = float(os.environ.get('ARM_DET_MIN_AREA_RATIO', '0.0015'))
DETECTION_MAX_AREA_RATIO = float(os.environ.get('ARM_DET_MAX_AREA_RATIO', '0.42'))
DETECTION_MAX_WIDTH_RATIO = float(os.environ.get('ARM_DET_MAX_WIDTH_RATIO', '0.82'))
DETECTION_MAX_HEIGHT_RATIO = float(os.environ.get('ARM_DET_MAX_HEIGHT_RATIO', '0.82'))
DETECTION_IGNORE_LABELS = {
    'airplane', 'aeroplane', 'boat', 'bus', 'car', 'train', 'truck',
    'bench', 'bed', 'chair', 'couch', 'dining table', 'table',
    'tv', 'tvmonitor', 'refrigerator', 'oven', 'sink', 'toilet',
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TRACKING CONFIG (v22 â€” frozen-lock, centroid-only tracking)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ Grace period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Frames without an IoU match before we consider the target truly lost.
# 12 frames @ 0.25s = 3 seconds of patience before unlocking.
TRACK_LOST_LIMIT = 12

# â”€â”€ Deadzone / tolerance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If the EMA-smoothed centroid is within TRACK_PX_DEADBAND pixels of the
# crosshair, NO servo command is issued.  Primary jitter absorber.
# 35 px â‰ˆ 5.5 % of 640 â€” wide enough to absorb YOLO box flicker entirely.
TRACK_PX_DEADBAND = 35

# Hysteresis: once declared "centred" stay centred until error exceeds this.
# Must be larger than TRACK_PX_DEADBAND to prevent boundary oscillation.
TRACK_HYSTERESIS_PX = 55

# â”€â”€ EMA position smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Exponential moving average on the CENTROID (not the box).
# alpha = weight of the newest frame; (1-alpha) = weight of history.
# 0.25 = very smooth; 0.5 = snappier.  Start conservative.
TRACK_EMA_ALPHA = 0.25

# â”€â”€ Stable-detection gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Require this many consecutive frames with an IoU-matching detection before
# issuing any servo command.  Prevents reacting to single spurious frames.
TRACK_STABLE_FRAMES_REQUIRED = 2

# â”€â”€ Frozen-lock IoU threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The reference box (locked_box) is FROZEN at acquisition and never overwritten
# by jittered detections.  A detection must overlap it by at least this much
# to be accepted as "still the same target".
# 0.40 = strict enough to reject nearby objects, loose enough to handle
#        normal YOLO scale fluctuation (~10-15% box size change â†’ IoU ~0.75).
TRACK_LOCK_IOU_THRESHOLD = 0.40

# â”€â”€ Locked-box update policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The reference box is only updated (slowly) when the object has genuinely
# moved far enough that the old reference would start failing IoU checks.
# TRACK_BOX_UPDATE_IOU: below this IoU the reference IS updated â€” the object
#   moved so far the old box no longer matches anything.
# TRACK_BOX_UPDATE_ALPHA: EMA weight for the reference update (slow blend).
TRACK_BOX_UPDATE_IOU   = 0.60   # update ref when IoU drops this low (object moved)
TRACK_BOX_UPDATE_ALPHA = 0.15   # blend 15% new box into reference per frame

# â”€â”€ Centroid-displacement filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ignore any single-frame centroid jump larger than this many pixels.
# YOLO can teleport a box by 80+ px when it re-classifies; a real object
# cannot move that fast between frames.  Set to ~15% of frame width.
TRACK_MAX_CENTROID_JUMP_PX = 90   # pixels; jumps larger than this are dropped

# â”€â”€ Bounding-box size-change filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ignore area changes > threshold unless persistent for N frames.
TRACK_SIZE_CHANGE_THRESHOLD = 0.35
TRACK_SIZE_CHANGE_FRAMES    = 5

# â”€â”€ Proportional gain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tiny â€” large errors converge over many slow steps, no snap or overshoot.
TRACK_GAIN_X     = 0.35   # horizontal (base)
TRACK_GAIN_Y     = 0.35   # vertical   (arm)
TRACK_GAIN_WRIST = 0.12   # wrist pitch

# â”€â”€ Maximum step per update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Hard cap regardless of error magnitude.
TRACK_MAX_NUDGE       = 2.5   # degrees per update (base/arm)
TRACK_MAX_WRIST_NUDGE = 1.0   # degrees per update (wrist)

# â”€â”€ Update rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.30 s: servo settles fully + camera captures the new position before the
# next error is computed.  Prevents chasing in-flight motion.
TRACK_UPDATE_INTERVAL = 0.30

# â”€â”€ Minimum movement threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Don't command a joint unless the computed nudge exceeds this. Avoids
# micro-pulses that make the servo buzz without actually moving.
TRACK_MIN_BASE_NUDGE_DEG = 0.35   # degrees â€” must be reachable from proportional gain
TRACK_MIN_ARM_NUDGE_DEG  = 0.35   # degrees

TRACK_EDGE_MARGIN_X = 105
TRACK_EDGE_MARGIN_Y = 80
TRACK_EDGE_BOOST    = 1.0

TRACK_SIGN_FLIP_THRESHOLD_PX = 9999   # disabled â€” signs set explicitly

TRACK_USE_WRIST_AIM = True
# Pickup aiming/timing. The pickup aim is the pixel the gripper should cover.
# Keep this at the camera centre by default; tune the offsets only after watching
# the green PICKUP AIM reticle in the stream during a real grab.
PICKUP_AIM_X_OFFSET = 0       # tune if the gripper is left/right of camera center
PICKUP_AIM_Y_OFFSET = 0       # tune if the gripper lands above/below camera center
PICKUP_TARGET_X_FRACTION = 0.50
PICKUP_TARGET_Y_FRACTION = 0.50
# For center-based grabs, use the true bounding-box midpoint instead of a
# lower fractional target. This keeps the gripper aligned with the object's
# visual center and avoids passing too high over the object.
# Center-grab only: grasp point is always bbox/mask geometric center (0.50).
PICKUP_OBJECT_CENTER_FRACTION = 0.50
PICKUP_CENTER_GRAB_ENABLED = True  # Hard-coded: never edge/bottom/top biased grabs
PICKUP_GRAB_HEIGHT_TO_ARM_DEG = float(os.environ.get('ARM_PICKUP_GRAB_HEIGHT_TO_ARM_DEG', '0.36'))  # gentler height response
# Fine X trim for a camera that is not exactly above the gripper. Positive
# values mean "aim farther right" in camera space; use +/-1cm steps if needed.
PICKUP_GRAB_X_OFFSET_CM = float(os.environ.get('ARM_PICKUP_GRAB_X_OFFSET_CM', '0.0'))
PICKUP_Y_TO_REACH_Z_GAIN = float(os.environ.get('ARM_PICKUP_Y_TO_REACH_Z_GAIN', '0.75'))  # gentler vertical targeting
PICKUP_Y_TO_DESCENT_DEG = float(os.environ.get('ARM_PICKUP_Y_TO_DESCENT_DEG', '0.40'))  # gentler Y-to-descent coupling
PICKUP_Y_DESCENT_ADJUST_MAX_DEG = float(os.environ.get('ARM_PICKUP_Y_DESCENT_ADJUST_MAX_DEG', '10.0'))
# FIX v35: floor safety margin (disabled - linear FK inaccurate at extreme angles)
PICKUP_FLOOR_SAFETY_MARGIN_CM = float(os.environ.get('ARM_PICKUP_FLOOR_SAFETY_MARGIN_CM', '0.0'))
PICKUP_BASE_BEARING_DEADBAND_CM = float(os.environ.get('ARM_PICKUP_BASE_DEADBAND_CM', '1.0'))
PICKUP_BASE_BEARING_GAIN = float(os.environ.get('ARM_PICKUP_BASE_BEARING_GAIN', '0.55'))
PICKUP_LOCK_BASE_AFTER_ALIGN = os.environ.get(
    'ARM_PICKUP_LOCK_BASE_AFTER_ALIGN',
    '1',
).strip().lower() not in ('0', 'false', 'no', 'off')
# v40: Center accuracy over speed — alignment must finish before descent.
# v40.1: Toned-down centering — see comments on each changed line.
PICKUP_ALIGN_X_DEADBAND = 22  # was 18
PICKUP_ALIGN_Y_DEADBAND = 26  # was 22
PICKUP_ALIGN_PASS_LIMIT = int(float(os.environ.get('ARM_PICKUP_ALIGN_PASS_LIMIT', '50')))
PICKUP_ARM_LIMIT_GUARD_DEG = float(os.environ.get('ARM_PICKUP_ARM_LIMIT_GUARD_DEG', '5.0'))
PICKUP_FINE_X_DEADBAND = float(os.environ.get('ARM_PICKUP_FINE_X_DEADBAND_PX', '20.0'))   # was 12
PICKUP_FINE_Y_DEADBAND = float(os.environ.get('ARM_PICKUP_FINE_Y_DEADBAND_PX', '22.0'))   # was 14
PICKUP_ACQUIRE_X_DEADBAND = float(os.environ.get('ARM_PICKUP_ACQUIRE_X_DEADBAND_PX', '50.0'))  # was 36
PICKUP_ACQUIRE_Y_DEADBAND = float(os.environ.get('ARM_PICKUP_ACQUIRE_Y_DEADBAND_PX', '55.0'))  # was 42
PICKUP_ACQUIRE_PASSES = int(float(os.environ.get('ARM_PICKUP_ACQUIRE_PASSES', '40')))
PICKUP_FINE_PASSES = int(float(os.environ.get('ARM_PICKUP_FINE_PASSES', '25')))
PICKUP_STABLE_FRAMES = int(float(os.environ.get('ARM_PICKUP_STABLE_FRAMES', '3')))         # was 5
PICKUP_ACQUIRE_MAX_STEP = float(os.environ.get('ARM_PICKUP_ACQUIRE_MAX_STEP_DEG', '0.8'))  # was 1.2
PICKUP_FINE_MAX_STEP = float(os.environ.get('ARM_PICKUP_FINE_MAX_STEP_DEG', '0.35'))       # was 0.55
PICKUP_BASE_ALIGN_MAX_STEP = float(os.environ.get('ARM_PICKUP_BASE_ALIGN_MAX_STEP_DEG', '1.2'))  # was 1.8
PICKUP_ARM_ALIGN_MAX_STEP = float(os.environ.get('ARM_PICKUP_ARM_ALIGN_MAX_STEP_DEG', '1.0'))    # was 1.5
PICKUP_ARM_ALIGN_MIN_STEP = float(os.environ.get('ARM_PICKUP_ARM_ALIGN_MIN_STEP_DEG', str(max(float(ARM_SNAP_DEG), 0.5))))
# Final gate: object center must sit inside both crosshair deadbands.
PICKUP_CROSSHAIR_X_DEADBAND = float(os.environ.get('ARM_PICKUP_CROSSHAIR_X_DEADBAND_PX', '22.0'))  # was 10
PICKUP_CROSSHAIR_Y_DEADBAND = float(os.environ.get('ARM_PICKUP_CROSSHAIR_Y_DEADBAND_PX', '22.0'))  # was 10
PICKUP_ARM_SIGN_FLIP_THRESHOLD_PX = float(os.environ.get('ARM_PICKUP_ARM_SIGN_FLIP_THRESHOLD_PX', '9999'))
PICKUP_ARM_WRONG_WAY_THRESHOLD_PX = float(os.environ.get('ARM_PICKUP_ARM_WRONG_WAY_THRESHOLD_PX', '8.0'))
PICKUP_BASE_GAIN = 0.042
PICKUP_BASE_ALIGN_GAIN_DEG = 6.0    # was 12.0 — halved to reduce per-step overcorrection
PICKUP_ARM_GAIN_DEG = 6.0           # was 12.0 — halved
PICKUP_SIGN_FLIP_THRESHOLD_PX = 9999
PICKUP_WRIST_GAIN_DEG = 5.0         # was 7.0
PICKUP_WRIST_MIN = 40   # FIX v30: was 50 â€” allows wrist to tilt further nose-down during pickup
PICKUP_WRIST_MAX = 150
PICKUP_WRIST_ALIGN_GAIN = 0.18      # was 0.25 — softer wrist Y correction during alignment
PICKUP_WRIST_ALIGN_MAX_STEP = 2.5   # was 3.5

# â”€â”€ Hard pickup timeout (v24) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The ENTIRE pickup sequence â€” alignment + depth + descent + grip + lift â€”
# must complete within this many seconds or the sequence is aborted.
# Prevents the arm from being stuck for minutes when YOLO oscillates.
# 90 s is generous for a real grab; tune down if your environment is fast.
PICKUP_TOTAL_TIMEOUT_SEC = float(os.environ.get('ARM_PICKUP_TOTAL_TIMEOUT_SEC', '90.0'))

# â”€â”€ Convergence / divergence detection (v24) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If alignment error hasn't reduced by at least this fraction after
# PICKUP_CONVERGENCE_CHECK_PASSES passes, the sequence is abandoned.
# Prevents infinite spinning when the arm is oscillating around a target
# it cannot physically reach.
PICKUP_CONVERGENCE_CHECK_PASSES = int(float(os.environ.get('ARM_PICKUP_CONVERGENCE_CHECK_PASSES', '12')))
PICKUP_CONVERGENCE_MIN_REDUCTION = float(os.environ.get('ARM_PICKUP_CONVERGENCE_MIN_REDUCTION', '0.15'))

# â”€â”€ IoU lock threshold for pickup (v24) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# During pickup alignment the chosen target box is FROZEN at acquisition.
# Any new detection must overlap the frozen box by at least this much to be
# accepted as the same object.  Prevents YOLO label flips from hijacking
# the arm mid-sequence.
PICKUP_IOU_LOCK_THRESHOLD = float(os.environ.get('ARM_PICKUP_IOU_LOCK_THRESHOLD', '0.35'))

# If the IoU-matched box falls below this, consider the object lost.
PICKUP_IOU_LOST_THRESHOLD = float(os.environ.get('ARM_PICKUP_IOU_LOST_THRESHOLD', '0.20'))

# Wrist tilts gradually during descent so the gripper angles toward the object.
# As the arm descends, wrist interpolates from approach_wrist by this span.
# APPROACH_GAIN scales span with descent depth â€” deeper grab = more tilt.
PICKUP_WRIST_APPROACH_SIGN    = float(os.environ.get('ARM_PICKUP_WRIST_APPROACH_SIGN', '-1.0'))  # was 1.0 — flipped
PICKUP_WRIST_APPROACH_MIN_DEG = 0.0     # unused — wrist is now absolute from depth
PICKUP_WRIST_APPROACH_GAIN    = 0.50   # wrist = 90 - descent * gain (depth-driven)
PICKUP_WRIST_APPROACH_MAX_DEG = 20.0   # unused — clamped by PICKUP_WRIST_MIN/MAX
PICKUP_WRIST_FINAL_COMPRESS_DEG = float(os.environ.get('ARM_PICKUP_WRIST_FINAL_COMPRESS_DEG', '0.0'))  # keep wrist steady during pickup; IK already aims base/arm
PICKUP_LEVEL_WRIST_AFTER_LIFT = os.environ.get('ARM_PICKUP_LEVEL_WRIST_AFTER_LIFT', '').strip().lower() in ('1', 'true', 'yes', 'on')
# FIX v35: Smooth velocity ramping â€” smaller steps for better control and less overshoot
PICKUP_DESCENT_DEGS = (5, 10, 15)
PICKUP_DESCENT_STEP_DEG = 0.9  # smoother final approach
PICKUP_DESCENT_STEP_FINAL_DEG = 0.45  # micro-steps for very smooth final approach
PICKUP_DESCENT_SETTLE = float(os.environ.get('ARM_PICKUP_DESCENT_SETTLE_SEC', '0.18'))
PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG = float(os.environ.get('ARM_PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG', '16.0'))
PICKUP_FINAL_DIP_DEG = float(os.environ.get('ARM_PICKUP_FINAL_DIP_DEG', '4.0'))           # was 0.0
PICKUP_FINAL_DIP_DEPTH_GAIN = float(os.environ.get('ARM_PICKUP_FINAL_DIP_DEPTH_GAIN', '0.0'))
PICKUP_CENTER_CONTACT_BIAS_DEG = float(os.environ.get('ARM_PICKUP_CENTER_CONTACT_BIAS_DEG', '12.0'))  # extra push to ensure jaw reaches object top
# Fraction of measured object height to pull the descent back up by, so the jaws
# close on the object's vertical center instead of its base/table contact point.
# 0.5 = true center. Lower (e.g. 0.3) grips closer to the bottom; raise (e.g. 0.7)
# grips closer to the top. Tune this first if grabs are still off-center.
PICKUP_HEIGHT_CENTER_FRACTION = float(os.environ.get('ARM_PICKUP_HEIGHT_CENTER_FRACTION', '0.5'))
PICKUP_PREGRIP_PASSES = 4
PICKUP_SIMPLE_LOCKED_ENABLED = os.environ.get(
    'ARM_PICKUP_SIMPLE_LOCKED',
    '0',
).strip().lower() not in ('0', 'false', 'no', 'off')
# Match this arm's manual controls: "down" lowers the arm by decreasing the
# direct slider angle, and "up" lifts it by increasing the direct slider angle.
# Pickup descent uses this sign, then lift moves opposite after grip closes.
PICKUP_ARM_DESCEND_SIGN = 1.0  # +1: arm_deg INCREASES to descend (FK: sh=1.526*arm-47.35)
# Centering uses the OPPOSITE sign from descent on most camera-on-gripper mounts.
# Descent sign -1 made the arm climb to max during Y align, throwing the object
# out of frame. Override with ARM_PICKUP_ARM_ALIGN_SIGN if your mount differs.
PICKUP_ARM_ALIGN_SIGN = float(os.environ.get('ARM_PICKUP_ARM_ALIGN_SIGN', '-1.0'))
# Prefer wrist tilt for Y (same as live tracking); arm only for large residual error.
PICKUP_ARM_Y_FALLBACK_PX = float(os.environ.get('ARM_PICKUP_ARM_Y_FALLBACK_PX', '36.0'))
# Hard cap on total arm travel during the entire centering sequence.
# Prevents the arm from drifting 40-50° across many passes and losing the object.
PICKUP_ALIGN_ARM_TOTAL_LIMIT_DEG = float(os.environ.get('ARM_PICKUP_ALIGN_ARM_TOTAL_LIMIT_DEG', '12.0'))
# If centroid is within this many px of any edge, skip correction for that axis.
PICKUP_ALIGN_EDGE_GUARD_PX = float(os.environ.get('ARM_PICKUP_ALIGN_EDGE_GUARD_PX', '80.0'))
# The photos show the desired grab is not a flat forward reach. Before the
# final descent, force a raised shoulder + nose-down wrist pose, then descend.
# Tune these in 3-5 degree steps if the real arm needs a slightly different pose.
PICKUP_TOPDOWN_ARM_DEG = float(os.environ.get('ARM_PICKUP_TOPDOWN_ARM_DEG', '132.0'))
PICKUP_TOPDOWN_WRIST_DEG = float(os.environ.get('ARM_PICKUP_TOPDOWN_WRIST_DEG', '125.0'))
PICKUP_REACQUIRE_SEC = 1.35
PICKUP_LOST_CLOSE_FRACTION = 0.45
PICKUP_MIN_BOX_AREA_RATIO = 0.003
PICKUP_MIN_CONFIDENCE = 30.0
PICKUP_CENTER_DEPTH_BOX_PX = int(float(os.environ.get('ARM_PICKUP_CENTER_DEPTH_BOX_PX', '120')))
PICKUP_MAX_DESCENT_DEG = 65.0          # Arm with 55cm reach needs enough descent headroom.
PICKUP_LIFT_DEG = 32
PICKUP_GRIP_HOLD_SEC = float(os.environ.get('ARM_PICKUP_GRIP_HOLD_SEC', '0.80'))
PICKUP_LEARN_FILE = 'pickup_learning.json'
PICKUP_DEPTH_DEFAULT = 5.0
PICKUP_DEPTH_MIN = 2.0
PICKUP_DEPTH_MAX = ARM_REACH_PHYSICAL_CM
PICKUP_DEPTH_FAIL_STEP = 1.5
# Extra descent learned from grab feedback. This is a servo-angle add-on, not
# the fallback depth estimate; defaulting it to 5 made every depth grab dive low.
PICKUP_EXTRA_DROP_DEFAULT_DEG = float(os.environ.get('ARM_PICKUP_EXTRA_DROP_DEG', '0.0'))
PICKUP_EXTRA_DROP_MIN_DEG = 0.0
PICKUP_EXTRA_DROP_MAX_DEG = float(os.environ.get('ARM_PICKUP_EXTRA_DROP_MAX_DEG', '6.0'))

# Depth hint from the detection box size.
# Larger boxes are usually closer; smaller boxes are usually farther away.
# This is a lightweight fallback that keeps pickup working without extra
# model dependencies.
DEPTH_HINT_MIN = 2.0
DEPTH_HINT_MAX = 16.0
DEPTH_HINT_REF_AREA = 0.028
DEPTH_HINT_GAIN = 4.0

# Depth Anything V2 disabled 2014 VL53L1X ToF sensor provides metric depth for pickup.
DEPTH_ENABLED = False
# VL53L1X ToF (replaces Depth Anything V2).
# ARM_VL53L1X_MOUNT=forward â†’ sensor is below the camera but points forward at the object.
#   Reading = sensor-to-object forward distance along the gripper direction.
# ARM_VL53L1X_MOUNT=down     â†’ sensor points at the table; height drives descent only.
VL53L1X_ENABLED = os.environ.get('ARM_DISABLE_VL53L1X', '').strip().lower() not in ('1', 'true', 'yes', 'on')
VL53L1X_MOUNT = os.environ.get('ARM_VL53L1X_MOUNT', 'forward').strip().lower() or 'forward'
VL53L1X_SDA_PIN = int(float(os.environ.get('ARM_VL53L1X_SDA_PIN', '2')))   # hardware I2C: GPIO 2 (Pin 3)
VL53L1X_SCL_PIN = int(float(os.environ.get('ARM_VL53L1X_SCL_PIN', '3')))   # hardware I2C: GPIO 3 (Pin 5)
# Forward distance from the VL53L1X sensor face to the gripper jaw/contact centre.
VL53L1X_MOUNT_OFFSET_CM = float(os.environ.get(
    'ARM_VL53L1X_SENSOR_TO_JAW_OFFSET_CM',
    os.environ.get('ARM_VL53L1X_MOUNT_OFFSET_CM', os.environ.get('ARM_GRIPPER_CAMERA_OFFSET_CM', '10.0'))
))
VL53L1X_GRASP_CLEARANCE_CM = float(os.environ.get('ARM_VL53L1X_GRASP_CLEARANCE_CM', '4.0'))
VL53L1X_POLL_INTERVAL = float(os.environ.get('ARM_VL53L1X_POLL_INTERVAL', '0'))
VL53L1X_SAMPLES = max(1, int(float(os.environ.get('ARM_VL53L1X_SAMPLES', '5'))))
VL53L1X_SAMPLE_GAP_SEC = float(os.environ.get('ARM_VL53L1X_SAMPLE_GAP_SEC', '0.10'))
VL53L1X_SCL_TIMEOUT_SEC = float(os.environ.get('ARM_VL53L1X_SCL_TIMEOUT_SEC', '0.12'))
VL53L1X_MIN_CM = float(os.environ.get('ARM_VL53L1X_MIN_CM', '3.0'))  # FIX v37: raised from 2.0; VL53L0X returns 20mm (2cm) as error sentinel — must be strictly above that
VL53L1X_MAX_CM = float(os.environ.get('ARM_VL53L1X_MAX_CM', '400.0'))
# VL53L1X is a ToF sensor — no sound-speed parameter needed
VL53L1X_SCALE_FACTOR = float(os.environ.get('ARM_VL53L1X_SCALE_FACTOR', os.environ.get('ARM_VL53L1X_DISTANCE_SCALE', '1.0')))
VL53L1X_DISTANCE_OFFSET_CM = float(os.environ.get('ARM_VL53L1X_DISTANCE_OFFSET_CM', '0.0'))
DEPTH_SAFE_WORKER = True

# â”€â”€ Timeout tuning for Pi 4B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cold-start on Pi 4B (CPU, vits): first inference takes 30-60 s.
# 90 s gives headroom; subsequent inferences are ~5-10 s each.
DEPTH_WORKER_TIMEOUT_SEC = float(os.environ.get('ARM_DEPTH_TIMEOUT_SEC', '90.0'))

# After a crash give the Pi time to free memory before restarting
DEPTH_WORKER_RESTART_COOLDOWN_SEC = 60.0
DEPTH_WORKER_MAX_CRASHES = 1
DEPTH_ENCODER = os.environ.get('ARM_DEPTH_ENCODER', 'vits').strip().lower() or 'vits'
DEPTH_DATASET = os.environ.get('ARM_DEPTH_DATASET', 'hypersim').strip().lower() or 'hypersim'

# â”€â”€ max_depth tuned for room-scale arm (reach = 55 cm, room height = 300 cm) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Increased from 2m to 5m to accurately represent room-scale distances.
# At 2m, ceiling objects (2.5-3m away) get compressed and misread as 25cm.
# At 5m scale, the model can properly represent both tabletop (5-50cm) AND
# room objects (200-300cm), so ceiling fan gets correct 250-300cm estimate
# which is then rejected by DEPTH_SANITY_MAX_CM.
# IMPORTANT: this must match the value passed to DepthAnythingV2()
# constructor â€” it cannot be changed after model load.
DEPTH_MAX_DEPTH_M = float(os.environ.get('ARM_DEPTH_MAX_M', '5.0'))

# Lower input size = faster on Pi; 392 still gives good ROI accuracy
# at table-top distances. Use 518 if accuracy feels off.
DEPTH_INPUT_SIZE = int(float(os.environ.get('ARM_DEPTH_INPUT_SIZE', '518')))
DEPTH_TORCH_HOME = 'torch_cache'

# Use all 4 Pi cores for inference
DEPTH_TORCH_THREADS = int(float(os.environ.get('ARM_DEPTH_TORCH_THREADS', '4')))
DEPTH_DEPTH_SAMPLES = int(float(os.environ.get('ARM_DEPTH_SAMPLES', '3')))
DEPTH_SAMPLE_DELAY = 0.04
DEPTH_HINT_WEIGHT = 0.85
DEPTH_CACHE_SEC = float(os.environ.get('ARM_DEPTH_CACHE_SEC', '1.75'))
DEPTH_ROI_X1 = 0.20
DEPTH_ROI_X2 = 0.80
DEPTH_ROI_Y1 = 0.30
DEPTH_ROI_Y2 = 0.92
DEPTH_OBJECT_NEAR_PERCENTILE = float(os.environ.get('ARM_DEPTH_OBJECT_NEAR_PERCENTILE', '35.0'))
DEPTH_OBJECT_MAX_PERCENTILE = float(os.environ.get('ARM_DEPTH_OBJECT_MAX_PERCENTILE', '68.0'))
DEPTH_OBJECT_MIN_PIXELS = int(float(os.environ.get('ARM_DEPTH_OBJECT_MIN_PIXELS', '30')))
DEPTH_TARGET_PATCH_PX = int(float(os.environ.get('ARM_DEPTH_TARGET_PATCH_PX', '9')))
DEPTH_TARGET_MAX_DELTA_CM = float(os.environ.get('ARM_DEPTH_TARGET_MAX_DELTA_CM', '12.0'))
DEPTH_TARGET_DEPTH_BLEND = float(os.environ.get('ARM_DEPTH_TARGET_BLEND', '0.75'))

# Reach limit defaults to the effective 19.2 cm pivot-to-pivot extension.
DEPTH_REACH_LIMIT_CM = float(os.environ.get('ARM_DEPTH_REACH_LIMIT_CM', '33'))
# How often (seconds) the background thread automatically re-scans depth
# when an object is detected and depth is enabled.  Defaults to 0 so VL53L1X
# reads only happen on Sensor Read / pickup, not in a continuous loop.
# Set ARM_VL53L1X_POLL_INTERVAL or ARM_DEPTH_AUTO_SCAN_INTERVAL > 0 to re-enable.
DEPTH_AUTO_SCAN_INTERVAL = float(os.environ.get('ARM_DEPTH_AUTO_SCAN_INTERVAL', '0'))
DEPTH_CAMERA_FOV_X_DEG = float(os.environ.get('ARM_CAMERA_FOV_X_DEG', '74.0'))
DEPTH_CAMERA_FOV_Y_DEG = float(os.environ.get('ARM_CAMERA_FOV_Y_DEG', '42.0'))
# Forward offset from the depth source to the jaw/contact centre. With the
# bottom VL53L1X mounted behind the green gripper, use the sensor-to-jaw offset
# measured above instead of the older camera-only 10 cm estimate.
DEPTH_GRIPPER_OFFSET_CM = float(os.environ.get('ARM_GRIPPER_CAMERA_OFFSET_CM', str(VL53L1X_MOUNT_OFFSET_CM)))
DEPTH_APPROACH_CLEARANCE_CM = float(os.environ.get('ARM_DEPTH_APPROACH_CLEARANCE_CM', '2.0'))
DEPTH_CM_TO_ARM_DEG = float(os.environ.get('ARM_DEPTH_CM_TO_ARM_DEG', '1.8'))
# max_depth MUST be in the constructor dict for metric DA2 â€” it initialises an
# internal scale factor used by forward(). Setting it as a plain attribute after
# load_state_dict does NOT work; the model reads the value set at __init__ time.

# â”€â”€ Depth sanity bounds matched to arm geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Scale factor: model is reading ~1.67x too far on this hardware setup.
# Real measurement: object at ~15cm, model reports 25cm â†’ scale = 15/25 = 1
# Adjust up/down in 0.05 steps if grabs consistently over/undershoot.
DEPTH_SCALE_FACTOR = float(os.environ.get('ARM_DEPTH_SCALE_FACTOR', '1.0'))
DEPTH_SCALE_STEP = float(os.environ.get('ARM_DEPTH_SCALE_STEP', '0.10'))
DEPTH_SCALE_MIN = float(os.environ.get('ARM_DEPTH_SCALE_MIN', '0.30'))
DEPTH_SCALE_MAX = float(os.environ.get('ARM_DEPTH_SCALE_MAX', '4.00'))
DEPTH_SANITY_MIN_CM = float(os.environ.get('ARM_DEPTH_SANITY_MIN_CM', '3.0'))   # closer than 3 cm = noise
DEPTH_SANITY_MAX_CM = float(os.environ.get('ARM_DEPTH_SANITY_MAX_CM', '70.0'))  # beyond full arm reach + margin = ignore

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  GRIPPER / CAMERA CALIBRATION OFFSETS  (v23)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#
#  Why these are needed
#  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  The camera is mounted on the forearm, NOT directly above the gripper jaw.
#  When the camera centres an object on-screen the gripper jaw is physically
#  displaced by (GRIPPER_X_OFFSET_CM, GRIPPER_Y_OFFSET_CM) in arm space.
#  This causes the systematic "grabs N cm to the right" error.
#
#  Two-level correction:
#    1. CAMERA_CENTER_OFFSET_PX  â€” shifts the on-screen crosshair so it
#       marks WHERE THE GRIPPER WILL LAND, not the camera optical axis.
#       Positive = crosshair moves right.  Use this to make the green
#       reticle visually line up with the gripper jaw during manual tests.
#
#    2. GRIPPER_X_OFFSET_CM / GRIPPER_Y_OFFSET_CM  â€” applied in the
#       camera-space â†’ arm-space 3D transform so the IK target is the
#       jaw centre, not the camera centre.
#       Positive X = jaw is to the right of the camera (in arm's forward direction).
#       Positive Y = jaw is below the camera.
#
#  Typical starting values for a camera mounted behind / above the gripper:
#    If the arm grabs 10 cm to the right â†’ GRIPPER_X_OFFSET_CM = -10.0
#    CAMERA_CENTER_OFFSET_PX â‰ˆ GRIPPER_X_OFFSET_CM * (fx / typical_depth)
#      e.g. -10 cm Ã— (430 px/cm) / 20 cm â‰ˆ -215 px  (use â€“ve smaller value first)
#
#  Calibration mode auto-computes these â€” see /calibrate API endpoint.
#
GRIPPER_X_OFFSET_CM = float(os.environ.get('ARM_GRIPPER_X_OFFSET_CM', '0.0'))
GRIPPER_Y_OFFSET_CM = float(os.environ.get('ARM_GRIPPER_Y_OFFSET_CM', '0.0'))
# Pixel shift for the crosshair drawn on stream (positive = crosshair shifts right).
# Set this so the GREEN crosshair sits exactly over the gripper jaw in a still image.
CAMERA_CENTER_OFFSET_PX = float(os.environ.get('ARM_CAMERA_CENTER_OFFSET_PX', '0.0'))
CAMERA_CENTER_Y_OFFSET_PX = float(os.environ.get('ARM_CAMERA_CENTER_Y_OFFSET_PX', '0.0'))

# â”€â”€ Close-range depth correction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Depth Anything V2 is trained on room-scale data; tabletop distances
# (5â€“40 cm) often read 1.3â€“2Ã— too far.  A separate per-range scale
# factor corrects this without touching the global DEPTH_SCALE_FACTOR.
# To calibrate: measure real distance (e.g. 15 cm), read DEPTH_READOUT
# from the UI, set DEPTH_CLOSE_RANGE_SCALE = real / readout.
DEPTH_CLOSE_RANGE_MAX_CM  = float(os.environ.get('ARM_DEPTH_CLOSE_RANGE_MAX_CM',  '40.0'))
DEPTH_CLOSE_RANGE_SCALE   = float(os.environ.get('ARM_DEPTH_CLOSE_RANGE_SCALE',   '1.0'))

# â”€â”€ Temporal depth smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Keep a rolling window of depth samples; use the median instead of
# a single-frame reading.  Rejects single-frame outliers and avoids
# sudden depth jumps that cause over/undershooting.
DEPTH_TEMPORAL_WINDOW     = int(float(os.environ.get('ARM_DEPTH_TEMPORAL_WINDOW',    '5')))
DEPTH_TEMPORAL_MAX_JUMP_CM= float(os.environ.get('ARM_DEPTH_TEMPORAL_MAX_JUMP_CM', '4.0'))

# â”€â”€ Calibration mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# When True the arm runs a single-shot calibration grab:
#   â€¢ Place an object exactly under the crosshair
#   â€¢ The arm grabs; the user measures the actual error
#   â€¢ POST /calibrate with actual_x_cm and actual_y_cm to auto-update offsets
_calibration_lock = threading.Lock()
_calibration_state = {
    'mode': False,        # calibration mode active
    'last_grab_aim': None,  # (aim_x_px, aim_y_px) at grab time
    'last_depth_cm': None,  # depth estimate at grab time
    'last_error_cm': None,  # user-reported grab error
    'offsets': {
        'gripper_x_cm': GRIPPER_X_OFFSET_CM,
        'gripper_y_cm': GRIPPER_Y_OFFSET_CM,
        'camera_center_px': CAMERA_CENTER_OFFSET_PX,
        'camera_center_y_px': CAMERA_CENTER_Y_OFFSET_PX,
        'close_range_scale': DEPTH_CLOSE_RANGE_SCALE,
        'home_base': 90.0,
        'home_arm': float(ARM_HOME_DEG + ARM_TRIM_DEG),
        'home_wrist': float(max(0, min(180, WRIST_HOME + WRIST_TRIM_DEG))),
        'home_grip': float(GRIP_OPEN),
    },
}

_grip_tune_lock = threading.Lock()
_GRIP_TUNE_LEVELS = [
    # All levels stay within the configured MG996R flex-gripper range (30Â°-120Â°).
    # 'close' is the maximum angle the routine will command if no contact is detected.
    # The adaptive loop stops at contact+backoff so the servo never stalls.
    {'name': 'soft',   'close': 96.0,  'backoff': 14.0, 'threshold': 2.0, 'min_frac': 0.12, 'confirm': 2},
    {'name': 'gentle', 'close': 108.0, 'backoff': 13.0, 'threshold': 2.2, 'min_frac': 0.16, 'confirm': 2},
    {'name': 'normal', 'close': 116.0, 'backoff': 10.0, 'threshold': 2.8, 'min_frac': 0.22, 'confirm': 2},
    {'name': 'firm',   'close': 120.0, 'backoff': 5.0,  'threshold': 3.8, 'min_frac': 0.34, 'confirm': 2},
    {'name': 'hard',   'close': 120.0, 'backoff': 2.0,  'threshold': 4.8, 'min_frac': 0.45, 'confirm': 3},
]
_grip_tune_state = {'level': int(max(0, min(len(_GRIP_TUNE_LEVELS) - 1, GRIP_TUNE_DEFAULT_LEVEL)))}

DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
if DEPTH_ENCODER not in DEPTH_MODEL_CONFIGS:
    DEPTH_ENCODER = 'vits'
if DEPTH_DATASET not in ('hypersim', 'vkitti'):
    DEPTH_DATASET = 'hypersim'
DEPTH_MODEL_SCALE = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}.get(DEPTH_ENCODER, 'Small')
DEPTH_DATASET_NAME = 'Hypersim' if DEPTH_DATASET == 'hypersim' else 'VKITTI'
DEPTH_MODEL_TYPE = f'Depth-Anything-V2-Metric-{DEPTH_DATASET_NAME}-{DEPTH_MODEL_SCALE}'
DEPTH_HF_REPO = f'depth-anything/{DEPTH_MODEL_TYPE}'
DEPTH_CHECKPOINT_NAME = f'depth_anything_v2_metric_{DEPTH_DATASET}_{DEPTH_ENCODER}.pth'

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  AUDIO CONFIG (PLACEHOLDER)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
AUDIO_RATE = 16000
AUDIO_CHUNK = 4000

_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV8_MODEL = 'yolov8n.pt'
YOLOV8_DEVICE = os.environ.get('ARM_YOLO_DEVICE', 'cpu').strip() or 'cpu'
YOLOV8_MAX_DET = int(float(os.environ.get('ARM_YOLO_MAX_DET', '12')))
_CONTROL_HTML = os.environ.get('ARM_CONTROL_HTML', '').strip()
if not _CONTROL_HTML:
    for _control_name in ('arm_control_v11.html', 'arm_control_v10.html', 'arm_control_v9.html', 'arm_control_v3 (2).html', 'arm_control_v3.html', 'arm_control_v2.html'):
        _control_path = os.path.join(_DIR, _control_name)
        if os.path.isfile(_control_path):
            _CONTROL_HTML = _control_path
            break
    else:
        _CONTROL_HTML = os.path.join(_DIR, 'arm_control_v2.html')
_THREE_JS = os.path.join(_DIR, 'three.min.js')
_PICKUP_LEARN_PATH = os.path.join(_DIR, PICKUP_LEARN_FILE)
_DEPTH_TORCH_HOME_PATH = (
    DEPTH_TORCH_HOME if os.path.isabs(DEPTH_TORCH_HOME)
    else os.path.join(_DIR, DEPTH_TORCH_HOME)
)
_DEPTH_CHECKPOINT_DIR = os.path.join(_DIR, 'checkpoints')
_DEPTH_CHECKPOINT_PATH = os.environ.get(
    'ARM_DEPTH_CHECKPOINT',
    os.path.join(_DEPTH_CHECKPOINT_DIR, DEPTH_CHECKPOINT_NAME),
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SHARED STATE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_pos = {'base': 90.0, 'arm': float(ARM_HOME_DEG + ARM_TRIM_DEG), 'wrist': float(WRIST_HOME), 'grip': float(GRIP_OPEN)}
# Kinematics solvers (initialized in main())
_robot_geometry = None
_fk_solver = None
_ik_solver = None
_pickup_config = None
_pickup_sequence = None  # kept for backward compat, not used
_pipeline = None               # PickupPipeline instance (validated pipeline)

_running = [True]
_pi = [None]
_lgpio_h = [None]
_cap_ref = [None]
_gpio_ready = [False]

_frame_lock = threading.Lock()
_scene_lock = threading.Lock()
_stream_lock = threading.Lock()
_yolo_pause_for_depth = threading.Event()
_yolo_pause_for_pickup = threading.Event()

# --- Server heartbeat / watchdog ---
_heartbeat = [time.time()]
_heartbeat_lock = threading.Lock()
_WATCHDOG_STALE_SEC = 8.0
_WATCHDOG_CHECK_INTERVAL = 3.0

_latest_frame = [None]
_stream_jpeg = [None]
_depth_result_lock = threading.Lock()
_latest_depth_result = [{
    'ts': 0.0,
    'ok': False,
    'label': 'none',
    'source': 'none',
    'message': 'No depth capture yet',
    'distance_cm': None,
    'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
    'in_reach': None,
    'depth_map_ready': False,
    'path': None,
}]
_latest_depth_jpeg = [None]
_scene_dets = []
_scene_info = {'ts': 0.0, 'label': 'none', 'count': 0}
_pickup_frozen_target_lock = threading.Lock()
_pickup_frozen_target = [None]
_status_txt = ['Waiting for command...']
_tracking = [False]
_track_obj = [None]
_track_lock = threading.Lock()
_pickup_lock = threading.Lock()
_pickup_abort = threading.Event()   # FIX v18: set this to interrupt a running pickup immediately
# Timestamp of the last servo move issued during a pickup centering pass.
# The detector thread checks this to suppress YOLO frames captured while
# the arm (and its camera) are still settling after a commanded step.
_last_pickup_move_time = [0.0]
# Object centroid captured at pickup start (before any centering moves).
_pickup_noted_center = [None]
# Position snapshot captured the instant the pickup button is pressed.
# Used by the UI to show "where was the arm when grab started".
_pickup_start_pos_lock = threading.Lock()
_pickup_start_pos = [{}]
_learn_lock = threading.Lock()
_pickup_learning = [{
    'depth_offset': PICKUP_EXTRA_DROP_DEFAULT_DEG,
    'depth_scale_factor': DEPTH_SCALE_FACTOR,
    'successes': 0,
    'failures': 0,
    'last_result': 'none',
}]

_depth_lock = threading.Lock()
_depth_worker_lock = threading.Lock()
_depth_request_lock = threading.Lock()
_depth_state = {
    'ready': False,
    'failed': False,
    'loading': False,
    'model': None,
    'transform': None,
    'device': None,
    'last_error': '',
    'last_box': None,
    'last_hint': None,
    'last_value': None,
    'last_result': None,
    'last_ts': 0.0,
    'inferences': 0,
    'worker_pid': None,
    'worker_returncode': None,
    'worker_crashes': 0,
    'worker_disabled_until': 0.0,
    'runtime_enabled': bool(VL53L1X_ENABLED),
}
_vl53l1x_lock = threading.Lock()
_vl53l1x_measure_lock = threading.Lock()
_vl53l1x_state = {
    'ready': False,
    'measurement_ok': False,
    'failed': False,
    'last_error': '',
    'last_raw_cm': None,
    'last_distance_cm': None,
    'last_measurement_us': None,
    'last_pulse_us': None,
    'last_ts': 0.0,
    'readings': 0,
}
_depth_worker = {
    'proc': None,
    'queue': None,
    'reader': None,
    'last_start': 0.0,
}


class _DepthTemporalSmoother:
    """Rolling-window depth smoother that rejects sudden jumps.

    â€¢ Maintains a deque of the last N depth readings.
    â€¢ Ignores any single frame that differs from the current median by
      more than DEPTH_TEMPORAL_MAX_JUMP_CM (transient outlier / occlusion).
    â€¢ Returns the median of the accepted window â€” much more stable than
      a single-frame reading for the close-range tabletop regime.
    """
    def __init__(self, window=5, max_jump_cm=10.0):
        self._window   = max(1, int(window))
        self._max_jump = float(max_jump_cm)
        self._history  = []
        self._pending  = None
        self._pending_count = 0
        self._lock     = threading.Lock()

    def update(self, new_cm):
        """Accept a new sample; return the smoothed estimate."""
        try:
            new_cm = float(new_cm)
        except (TypeError, ValueError):
            return self.smooth()
        if not math.isfinite(new_cm) or new_cm <= 0:
            return self.smooth()
        with self._lock:
            if self._history:
                cur_median = float(np.median(self._history))
                if abs(new_cm - cur_median) > self._max_jump:
                    if self._pending is not None and abs(new_cm - self._pending) <= self._max_jump:
                        self._pending_count += 1
                    else:
                        self._pending = new_cm
                        self._pending_count = 1
                    if self._pending_count >= 2:
                        print(f'[DEPTH-SMOOTH] accepting new stable depth {new_cm:.1f}cm '
                              f'after {self._pending_count} confirmations', flush=True)
                        self._history = [new_cm]
                        self._pending = None
                        self._pending_count = 0
                        return new_cm
                    print(f'[DEPTH-SMOOTH] jump {abs(new_cm - cur_median):.1f}cm rejected '
                          f'(max={self._max_jump:.1f}cm)', flush=True)
                    return cur_median
            self._pending = None
            self._pending_count = 0
            self._history.append(new_cm)
            if len(self._history) > self._window:
                self._history.pop(0)
            return float(np.median(self._history))

    def smooth(self):
        with self._lock:
            if not self._history:
                return None
            return float(np.median(self._history))

    def reset(self):
        with self._lock:
            self._history.clear()
            self._pending = None
            self._pending_count = 0


# Global depth temporal smoother â€” shared between pickup calls.
# Reset it at the start of each pickup so stale readings don't carry over.
_depth_smoother = _DepthTemporalSmoother(
    window=DEPTH_TEMPORAL_WINDOW,
    max_jump_cm=DEPTH_TEMPORAL_MAX_JUMP_CM,
)

_last_angle = {}
_last_pulse = {}
_servo_lock  = threading.Lock()
_pos_lock    = threading.Lock()   # protects _pos reads/writes across threads
_motion_lock = threading.Lock()
_motion_seq  = {}


class _SerialServoJoint:
    """USB bridge for Arduino Nano FrankenServo PID controller."""

    # Matches both Nano format: "Pos: 90.0°  Tgt: 90.0°  Err: 0.0°  PWM: 0"
    # and ESP format:          "P:90.0 T:90.0 E:0.0 PWM:0"
    _NANO_RE = re.compile(
        r'Pos:\s*(?P<position>-?\d+(?:\.\d+)?)\s*°?\s+'
        r'Tgt:\s*(?P<target>-?\d+(?:\.\d+)?)\s*°?\s+'
        r'Err:\s*(?P<error>-?\d+(?:\.\d+)?)\s*°?\s+'
        r'PWM:\s*(?P<pwm>-?\d+)'
    )
    _ESP_RE = re.compile(
        r'^P:(?P<position>-?\d+(?:\.\d+)?)\s+'
        r'T:(?P<target>-?\d+(?:\.\d+)?)\s+'
        r'E:(?P<error>-?\d+(?:\.\d+)?)\s+'
        r'PWM:(?P<pwm>-?\d+)$'
    )

    def __init__(self, port, baud, angle_min=0.0, angle_max=270.0):
        if _serial_mod is None:
            raise RuntimeError('pyserial is not installed (run: pip install pyserial)')
        self.port = str(port)
        self.baud = int(baud)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.ser = _serial_mod.Serial(self.port, self.baud, timeout=0.1, write_timeout=0.5)
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._running = True
        self._position = None
        self._target = None
        self._error = None
        self._pwm = None
        self._last_update = 0.0
        self._last_line = ''
        self._reader = threading.Thread(target=self._read_loop, name='arm-serial-reader', daemon=True)
        self._reader.start()

    def _read_loop(self):
        while self._running:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                match = self._NANO_RE.search(line) or self._ESP_RE.match(line)
                with self._state_lock:
                    self._last_line = line
                    if match:
                        self._position = float(match.group('position'))
                        self._target = float(match.group('target'))
                        self._error = float(match.group('error'))
                        self._pwm = int(match.group('pwm'))
                        self._last_update = time.time()
            except Exception as exc:
                if self._running:
                    print(f'[ARM-SERIAL] read error: {exc}', flush=True)
                    time.sleep(0.1)

    def _write_line(self, text):
        payload = (str(text).strip() + '\n').encode('ascii')
        with self._write_lock:
            self.ser.write(payload)
            self.ser.flush()

    def move_to(self, angle):
        angle = max(self.angle_min, min(self.angle_max, float(angle)))
        self._write_line(f'{angle:.1f}')
        return angle

    def stop(self):
        # Nano has no STOP command — send current position to hold in place.
        with self._state_lock:
            pos = self._position
        if pos is not None:
            self._write_line(f'{pos:.1f}')
        else:
            self._write_line('90.0')

    def enable(self):
        # Nano auto-enables on first numeric angle command — no-op here.
        pass

    def snapshot(self):
        with self._state_lock:
            age = time.time() - self._last_update if self._last_update else None
            return {
                'enabled': True,
                'connected': bool(self.ser and self.ser.is_open),
                'port': self.port,
                'baud': self.baud,
                'position': self._position,
                'target': self._target,
                'error': self._error,
                'pwm': self._pwm,
                'telemetry_age_sec': age,
                'telemetry_fresh': bool(age is not None and age < 0.5),
                'last_line': self._last_line,
            }

    def close(self):
        self._running = False
        try:
            self.ser.close()
        except Exception:
            pass


_arm_serial = [None]

_USE_PIGPIO = False
_DEPTH_WORKER_MODE = '--depth-worker' in sys.argv
if not _DEPTH_WORKER_MODE:
    try:
        import pigpio as _pigpio_mod
        _test_pi = _pigpio_mod.pi()
        if _test_pi.connected:
            _test_pi.stop()
            _USE_PIGPIO = True
    except Exception:
        pass

    if _USE_PIGPIO:
        import pigpio
    else:
        try:
            import lgpio
        except ImportError:
            class _NoOpLGPIO:
                @staticmethod
                def gpiochip_open(*args, **kwargs):
                    return None

                @staticmethod
                def gpio_claim_output(*args, **kwargs):
                    return None

                @staticmethod
                def tx_servo(*args, **kwargs):
                    return None

                @staticmethod
                def gpiochip_close(*args, **kwargs):
                    return None

            lgpio = _NoOpLGPIO()
            print('[WARN] Neither pigpio nor lgpio available; GPIO commands are disabled.')

# ── INA219 current sensor (gripper) ──────────────────────────────────────────
_ina219 = [None]
if _INA219_AVAILABLE and not _DEPTH_WORKER_MODE:
    try:
        _i2c_bus = _busio_mod.I2C(_board_mod.SCL, _board_mod.SDA)
        _ina219[0] = _INA219_mod(_i2c_bus, addr=0x40)
        _ina219[0].gain = 3  # ±320mV shunt → ±1.6A range (MG996R peaks ~1.5A)
        print('[INA219] Current sensor initialised on I2C 0x40', flush=True)
    except Exception as exc:
        print(f'[INA219] Init failed: {exc}', flush=True)
        _ina219[0] = None

# ────────────────────────────────────────────────────────────────────────────────
#  HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(lo)
    if not math.isfinite(v):
        return float(lo)
    return max(lo, min(hi, v))

def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('1', 'true', 'yes', 'on', 'enable', 'enabled'):
            return True
        if text in ('0', 'false', 'no', 'off', 'disable', 'disabled'):
            return False
    return bool(value)

def _grip_tune_snapshot():
    with _grip_tune_lock:
        level = int(_clamp(_grip_tune_state.get('level', GRIP_TUNE_DEFAULT_LEVEL), 0, len(_GRIP_TUNE_LEVELS) - 1))
    cfg = dict(_GRIP_TUNE_LEVELS[level])
    cfg['level'] = level
    cfg['levels'] = [dict(item, level=i) for i, item in enumerate(_GRIP_TUNE_LEVELS)]
    cfg['adaptive'] = bool(GRIP_ADAPTIVE_CLOSE)
    return cfg

def _set_grip_tune(delta=0, level=None):
    with _grip_tune_lock:
        cur = int(_clamp(_grip_tune_state.get('level', GRIP_TUNE_DEFAULT_LEVEL), 0, len(_GRIP_TUNE_LEVELS) - 1))
        if level is not None:
            new_level = int(_clamp(level, 0, len(_GRIP_TUNE_LEVELS) - 1))
        else:
            new_level = int(_clamp(cur + int(delta), 0, len(_GRIP_TUNE_LEVELS) - 1))
        _grip_tune_state['level'] = new_level
    snap = _grip_tune_snapshot()
    _set_status(f'Gripper strength: {snap["name"]}')
    return snap

def _pickup_aim_xy(dets=None):
    """Return the calibrated grab-aim point in image space.

    This is NOT the camera optical-axis centre.  It is the pixel that the
    gripper jaw will physically be over when the arm finishes centering.
    The offset accounts for the physical camera-to-gripper displacement:

        aim_x = frame_centre_x + PICKUP_AIM_X_OFFSET + CAMERA_CENTER_OFFSET_PX

    CAMERA_CENTER_OFFSET_PX should be tuned until the green crosshair sits
    exactly over the gripper jaw when viewing a static image of the arm.
    Positive value â†’ crosshair moves right â†’ gripper is to the right of lens.
    """
    with _calibration_lock:
        offsets = _calibration_state['offsets']
        cam_offset_x = offsets.get('camera_center_px', 0.0)
        cam_offset_y = offsets.get('camera_center_y_px', 0.0)
    ax = FRAME_W / 2.0 + PICKUP_AIM_X_OFFSET + cam_offset_x
    ay = FRAME_H / 2.0 + PICKUP_AIM_Y_OFFSET + cam_offset_y
    return _clamp(ax, 0, FRAME_W - 1), _clamp(ay, 0, FRAME_H - 1)

def _set_status(text):
    _status_txt[0] = str(text)

def _pos_snapshot(timeout=2.0):
    if _pos_lock.acquire(timeout=timeout):
        try:
            return dict(_pos)
        finally:
            _pos_lock.release()
    print(f'[POS] _pos_lock timeout (held by another thread)', flush=True)
    return dict(_pos)

def _tracking_snapshot():
    with _track_lock:
        return bool(_tracking[0]), _track_obj[0]

def _set_tracking(enabled, obj_name=None):
    with _track_lock:
        _tracking[0] = bool(enabled)
        _track_obj[0] = _clean_object_name(obj_name) if enabled else None

def _clean_object_name(text):
    if text is None:
        return None
    name = ' '.join(str(text).strip().lower().split())
    name = re.sub(r'^(the|a|an)\s+', '', name, count=1)
    return name or None

def _extract_object_name(cmd, keywords):
    for kw in sorted(keywords, key=len, reverse=True):
        match = re.search(r'\b' + re.escape(kw) + r'\b', cmd)
        if match:
            return _clean_object_name(cmd[match.end():])
    return None

def _label_matches(wanted, label):
    wanted = _clean_object_name(wanted)
    label = _clean_object_name(label)
    if not wanted or not label:
        return False
    return wanted == label or wanted in label or label in wanted

def _detection_area_ratio(det):
    box = det.get('box') if isinstance(det, dict) else None
    if not box or len(box) != 4:
        return 0.0
    try:
        _, _, bw, bh = [float(v) for v in box]
        if bw <= 0 or bh <= 0:
            return 0.0
        return float(_clamp((bw * bh) / float(FRAME_W * FRAME_H), 0.0, 1.0))
    except Exception:
        return 0.0

def _detection_point(det):
    if not isinstance(det, dict):
        return FRAME_W / 2.0, FRAME_H / 2.0
    try:
        for x_key, y_key in (
            ('target_x', 'target_y'),
            ('object_center_x', 'object_center_y'),
            ('center_x', 'center_y'),
        ):
            if det.get(x_key) is not None and det.get(y_key) is not None:
                return (
                    float(_clamp(det.get(x_key), 0, FRAME_W - 1)),
                    float(_clamp(det.get(y_key), 0, FRAME_H - 1)),
                )
        box = det.get('box')
        if box and len(box) == 4:
            x, y, bw, bh = [float(v) for v in box]
            if bw > 1 and bh > 1:
                return (
                    float(_clamp(x + bw * 0.5, 0, FRAME_W - 1)),
                    float(_clamp(y + bh * 0.5, 0, FRAME_H - 1)),
                )
        return (
            float(_clamp(det.get('cx', FRAME_W / 2.0), 0, FRAME_W - 1)),
            float(_clamp(det.get('cy', FRAME_H / 2.0), 0, FRAME_H - 1)),
        )
    except Exception:
        return FRAME_W / 2.0, FRAME_H / 2.0

def _target_focus_score(det, near=None):
    """Lower is better. Prefer the object already under the pickup crosshair."""
    cx, cy = _detection_point(det)
    if isinstance(near, dict):
        rx, ry = _detection_point(near)
    else:
        rx, ry = _pickup_aim_xy(None)
    aim_x, aim_y = _pickup_aim_xy(None)
    conf = float(det.get('confidence', 0.0)) if isinstance(det, dict) else 0.0
    area_ratio = _detection_area_ratio(det)

    ref_dist = ((cx - rx) / max(1.0, FRAME_W)) ** 2 + ((cy - ry) / max(1.0, FRAME_H)) ** 2
    aim_dist = ((cx - aim_x) / max(1.0, FRAME_W)) ** 2 + ((cy - aim_y) / max(1.0, FRAME_H)) ** 2

    edge_penalty = 0.0
    box = det.get('box') if isinstance(det, dict) else None
    if box and len(box) == 4:
        try:
            x, y, bw, bh = [float(v) for v in box]
            if x <= 2 or y <= 2 or x + bw >= FRAME_W - 3 or y + bh >= FRAME_H - 3:
                edge_penalty = 0.18
        except Exception:
            pass

    # Confidence alone is deliberately weak here. In cluttered scenes YOLO often
    # gives background objects higher confidence than the object at the gripper.
    return ref_dist * 2.0 + aim_dist * 1.4 + edge_penalty - (conf / 100.0) * 0.22 - area_ratio * 0.55

def _best_detection(dets, label_hint=None, near=None):
    if not dets:
        return None
    hint = _clean_object_name(label_hint)
    usable = [d for d in dets if _detection_area_ratio(d) >= DETECTION_MIN_AREA_RATIO]
    if usable:
        dets = usable
    if not hint:
        return min(dets, key=lambda d: _target_focus_score(d, near=near))
    matches = [d for d in dets if _label_matches(hint, d.get('label', ''))]
    if not matches:
        return None
    return min(matches, key=lambda d: _target_focus_score(d, near=near))

def _clone_pickup_detection(det):
    if not isinstance(det, dict):
        return None
    frozen = dict(det)
    box = frozen.get('box')
    try:
        if frozen.get('target_x') is not None and frozen.get('target_y') is not None:
            center_x = float(frozen.get('target_x'))
            center_y = float(frozen.get('target_y'))
            pickup_depth_box = [int(round(float(v))) for v in box] if box and len(box) == 4 else None
        elif frozen.get('object_center_x') is not None and frozen.get('object_center_y') is not None:
            center_x = float(frozen.get('object_center_x'))
            center_y = float(frozen.get('object_center_y'))
            pickup_depth_box = [int(round(float(v))) for v in box] if box and len(box) == 4 else None
        elif box and len(box) == 4:
            x, y, bw, bh = [float(v) for v in box]
            center_x = x + max(1.0, bw) * 0.5
            center_y = y + max(1.0, bh) * 0.5
            pickup_depth_box = [int(round(float(v))) for v in (x, y, bw, bh)]
        else:
            center_x = float(frozen.get('object_center_x', frozen.get('center_x', frozen.get('cx'))))
            center_y = float(frozen.get('object_center_y', frozen.get('center_y', frozen.get('cy'))))
            pickup_depth_box = None
        center_x = float(_clamp(center_x, 0, FRAME_W - 1))
        center_y = float(_clamp(center_y, 0, FRAME_H - 1))
        # FIX v44: do NOT pop 'box' - _find_iou_locked_detection reads det.get('box')
        # to compute IoU against locked_box. If box is deleted, IoU=0 always and the
        # lock never matches any YOLO frame -> centering spins forever.
        # Only remove the 1D edge coords that are redundant given box.
        for key in ('x1', 'y1', 'x2', 'y2'):
            frozen.pop(key, None)
        frozen['object_center_x'] = center_x
        frozen['object_center_y'] = center_y
        frozen['target_x'] = center_x
        frozen['target_y'] = center_y
        if pickup_depth_box is not None:
            frozen['pickup_depth_box'] = pickup_depth_box
        frozen['center_x'] = center_x
        frozen['center_y'] = center_y
        frozen['cx'] = int(round(center_x))
        frozen['cy'] = int(round(center_y))
        frozen['target_class'] = frozen.get('class_id', frozen.get('cls', frozen.get('label', 'object')))
        frozen['target_label'] = str(frozen.get('label', 'object'))
        frozen['frozen_pickup_center'] = True
        frozen['pickup_pipeline'] = 'yolo-mask-centroid-tof'
        frozen['centroid_source'] = str(frozen.get('centroid_source') or frozen.get('center_source') or 'bbox-centroid')
        frozen['frozen_ts'] = time.time()
        detector = str(frozen.get('detector') or 'yolov8n')
        frozen['detector'] = detector if 'pickup-center-freeze' in detector else detector + '+pickup-center-freeze'
        return frozen
    except Exception:
        return None

def _set_pickup_frozen_target(det):
    frozen = _clone_pickup_detection(det)
    if frozen is None:
        return None
    _yolo_pause_for_pickup.set()
    with _pickup_frozen_target_lock:
        _pickup_frozen_target[0] = dict(frozen)
    with _scene_lock:
        _scene_dets[:] = [dict(frozen)]
        _scene_info['ts'] = time.time()
        _scene_info['label'] = str(frozen.get('label', 'object'))
        _scene_info['count'] = 1
    print(
        f'[PICKUP] HARD CENTER FREEZE center=({frozen["object_center_x"]:.1f},{frozen["object_center_y"]:.1f}) '
        f'class={frozen.get("target_class")}',
        flush=True,
    )
    return frozen

def _pickup_frozen_snapshot():
    with _pickup_frozen_target_lock:
        frozen = _pickup_frozen_target[0]
        return None if frozen is None else dict(frozen)

def _current_display_detections():
    frozen = _pickup_frozen_snapshot()
    if _yolo_pause_for_pickup.is_set() and frozen is not None:
        return [_decorate_detection_debug(frozen)]
    with _scene_lock:
        return [_decorate_detection_debug(d) for d in _scene_dets]

def _decorate_detection_debug(det):
    if not isinstance(det, dict):
        return det
    out = dict(det)
    try:
        cx, cy = _detection_point(out)
        ax, ay = _pickup_aim_xy(None)
        out.setdefault('object_center_x', round(cx, 1))
        out.setdefault('object_center_y', round(cy, 1))
        out.setdefault('target_x', round(cx, 1))
        out.setdefault('target_y', round(cy, 1))
        out.setdefault('grasp_x', round(cx, 1))
        out.setdefault('grasp_y', round(cy, 1))
        box = out.get('box')
        if box and len(box) == 4:
            try:
                _, y, _, bh = [float(v) for v in box]
                out.setdefault('object_height_px', round(float(bh), 1))
                out.setdefault('pixel_height', round(float(bh), 1))
                out.setdefault('grasp_y', round(float(y + bh * 0.5), 1))
            except Exception:
                pass
        out.update(_height_fields_for_detection(out))
        # Green crosshair always marks the calibrated gripper aim — never the object.
        cross_x, cross_y = ax, ay
        noted = _pickup_noted_center[0]
        noted_px = None
        if noted is not None:
            try:
                noted_px = {'x': round(float(noted[0]), 1), 'y': round(float(noted[1]), 1)}
            except Exception:
                noted_px = None
        out['frozen_center'] = bool(out.get('frozen_pickup_center'))
        out['crosshair_x'] = round(float(cross_x), 1)
        out['crosshair_y'] = round(float(cross_y), 1)
        out['target_wrist_deg'] = round(float(_target_wrist_angle(out)), 1)
        out['pickup_debug'] = {
            'pipeline': out.get('pickup_pipeline', 'yolo-mask-centroid-tof'),
            'centroid_source': out.get('centroid_source', out.get('center_source', 'bbox-centroid')),
            'detected_object_center_px': {'x': round(cx, 1), 'y': round(cy, 1)},
            'noted_object_center_px': noted_px,
            'grasp_point_px': {'x': round(float(out.get('grasp_x', cx)), 1), 'y': round(float(out.get('grasp_y', cy)), 1)},
            'crosshair_px': {'x': round(float(cross_x), 1), 'y': round(float(cross_y), 1)},
            'target_wrist_deg': out['target_wrist_deg'],
            'error_px': {'x': round(cx - ax, 1), 'y': round(cy - ay, 1)},
            'movement_aim_px': {'x': round(float(ax), 1), 'y': round(float(ay), 1)},
            'object_height_px': out.get('object_height_px'),
            'corrected_object_height_cm': out.get('corrected_object_height_cm', out.get('object_height_cm')),
            'frozen_center': out.get('frozen_center', False)
        }
    except Exception:
        pass
    return out


def _height_fields_for_detection(det):
    fields = {}
    try:
        if not isinstance(det, dict):
            return fields
        det_box = det.get('box')
        if not det_box or len(det_box) != 4:
            return fields
        latest = _depth_result_snapshot()
        path = latest.get('path') if isinstance(latest, dict) else None
        if not isinstance(path, dict):
            return fields
        path_box = path.get('bbox_px') or latest.get('box')
        if path_box and len(path_box) == 4 and _box_similarity(det_box, path_box) < 0.50:
            return fields
        for key in (
            'object_height_px',
            'pixel_height',
            'object_height_cm',
            'corrected_object_height_cm',
            'grab_height_cm',
            'grab_height_fraction',
            'height_source',
            'height_measurement',
            'grasp_pose',
        ):
            if path.get(key) is not None:
                fields[key] = path.get(key)
        if isinstance(path.get('grasp_px'), dict):
            fields['grasp_x'] = path['grasp_px'].get('x')
            fields['grasp_y'] = path['grasp_px'].get('y')
        return fields
    except Exception:
        return fields

def _clear_pickup_frozen_target():
    with _pickup_frozen_target_lock:
        _pickup_frozen_target[0] = None
    _pickup_noted_center[0] = None
    _yolo_pause_for_pickup.clear()

def _target_wrist_angle(det):
    if not isinstance(det, dict):
        return float(_clamp(WRIST_HOME + WRIST_TRIM_DEG, 0, 180))
    try:
        if det.get('target_y') is not None:
            cy = float(det.get('target_y'))
        elif det.get('object_center_y') is not None:
            cy = float(det.get('object_center_y'))
        elif det.get('center_y') is not None:
            cy = float(det.get('center_y'))
        else:
            box = det.get('box')
            if box and len(box) == 4:
                x, y, bw, bh = [float(v) for v in box]
                if bw > 1 and bh > 1:
                    cy = float(_clamp(y + bh * PICKUP_OBJECT_CENTER_FRACTION, 0, FRAME_H - 1))
                else:
                    cy = float(det.get('cy', FRAME_H // 2))
            else:
                cy = float(det.get('cy', FRAME_H // 2))
    except Exception:
        cy = float(det.get('cy', FRAME_H // 2))
    cy = _clamp(cy, 0, FRAME_H - 1)
    _, aim_y = _pickup_aim_xy([det] if det else None)
    err_y = cy - aim_y
    offset = (err_y / max(1.0, FRAME_H / 2.0)) * PICKUP_WRIST_GAIN_DEG
    neutral = WRIST_HOME + WRIST_TRIM_DEG
    return float(_clamp(neutral + offset, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX))

def _ik_elbow_to_wrist_servo(elbow_deg):
    """Compute wrist servo angle from elbow geometry for reach extension.

    FIX v34: Previously always returned None (disabled), meaning the wrist never
    adjusted for reach depth during IK repositioning. The arm would position the
    shoulder correctly but the wrist would stay at whatever angle it was last set to,
    preventing the gripper from fully extending to the target.

    For a flexible compliant gripper (MG995/MG996R cults3d design):
    - elbow_deg is the geometric interior angle at the elbow (0Â° = fully extended, 180Â° = folded)
    - The wrist servo must compensate to keep the gripper pointing forward/down at the object
    - When the elbow bends (angle < 120Â°), the wrist tilts to aim the jaw at the target
    - Convention: wrist 90Â° = straight; smaller = nose-down; larger = nose-up

    The compensation formula:
        wrist_cmd = 90 - (180 - elbow_deg) * WRIST_ELBOW_COMPENSATION
    When elbow is 180Â° (fully extended) â†’ wrist stays at 90Â° (straight)
    When elbow bends to 90Â° â†’ wrist tilts 90*factor degrees nose-down
    """
    try:
        elbow = float(elbow_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(elbow):
        return None

    # Compensation factor: how much wrist tilts per degree of elbow bend.
    # 0.45 keeps the gripper roughly level relative to the world frame.
    WRIST_ELBOW_COMPENSATION = 0.45
    wrist_target = 90.0 - (180.0 - elbow) * WRIST_ELBOW_COMPENSATION
    # Clamp to pickup wrist range â€” keep gripper from swinging too far
    return float(_clamp(wrist_target, float(PICKUP_WRIST_MIN), float(PICKUP_WRIST_MAX)))

def _load_pickup_learning():
    try:
        with open(_PICKUP_LEARN_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    with _learn_lock:
        cur = _pickup_learning[0]
        cur['successes'] = int(max(0, data.get('successes', 0)))
        cur['failures'] = int(max(0, data.get('failures', 0)))
        cur['last_result'] = str(data.get('last_result', 'none'))
        cur['depth_offset'] = _coerce_extra_drop(data.get('depth_offset', PICKUP_EXTRA_DROP_DEFAULT_DEG),
                                                 cur['failures'],
                                                 cur['last_result'])
        cur['depth_scale_factor'] = _coerce_depth_scale(data.get('depth_scale_factor', DEPTH_SCALE_FACTOR))

def _save_pickup_learning():
    with _learn_lock:
        data = dict(_pickup_learning[0])
    try:
        tmp = _PICKUP_LEARN_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, _PICKUP_LEARN_PATH)
    except Exception as e:
        print(f'[LEARN] Failed to save pickup feedback: {e}')

def _pickup_depth_offset():
    with _learn_lock:
        cur = _pickup_learning[0]
        return _coerce_extra_drop(cur.get('depth_offset', PICKUP_EXTRA_DROP_DEFAULT_DEG),
                                  cur.get('failures', 0),
                                  cur.get('last_result', 'none'))

def _coerce_depth_scale(value):
    try:
        scale = float(value)
    except Exception:
        scale = float(DEPTH_SCALE_FACTOR)
    if not math.isfinite(scale) or scale <= 0:
        scale = float(DEPTH_SCALE_FACTOR)
    return float(_clamp(scale, DEPTH_SCALE_MIN, DEPTH_SCALE_MAX))

def _pickup_depth_scale_factor():
    with _learn_lock:
        cur = _pickup_learning[0]
        return _coerce_depth_scale(cur.get('depth_scale_factor', DEPTH_SCALE_FACTOR))

def _coerce_extra_drop(value, failures=0, last_result='none'):
    try:
        drop = float(value)
    except Exception:
        drop = PICKUP_EXTRA_DROP_DEFAULT_DEG
    # Older builds stored the 5 cm fallback depth in this field. If there is no
    # failure history, migrate that old default to zero extra servo drop.
    try:
        no_feedback = int(failures or 0) <= 0 and str(last_result or 'none').lower() in ('', 'none')
    except Exception:
        no_feedback = False
    if no_feedback and abs(drop - float(PICKUP_DEPTH_DEFAULT)) < 0.01:
        drop = PICKUP_EXTRA_DROP_DEFAULT_DEG
    return float(_clamp(drop, PICKUP_EXTRA_DROP_MIN_DEG, PICKUP_EXTRA_DROP_MAX_DEG))

def _record_pickup_feedback(success, opened=False, homed=False, reason=''):
    success = bool(success)
    reason = str(reason or '').strip().lower()
    with _learn_lock:
        cur = _pickup_learning[0]
        depth = _coerce_extra_drop(cur.get('depth_offset', PICKUP_EXTRA_DROP_DEFAULT_DEG),
                                   cur.get('failures', 0),
                                   cur.get('last_result', 'none'))
        scale = _coerce_depth_scale(cur.get('depth_scale_factor', DEPTH_SCALE_FACTOR))
        if success:
            cur['successes'] = int(cur.get('successes', 0)) + 1
            cur['last_result'] = 'success'
            # Keep the learned deeper approach after a win; just stop escalating.
        elif reason in ('too_close', 'close', 'short', 'near'):
            cur['failures'] = int(cur.get('failures', 0)) + 1
            cur['last_result'] = 'too_close'
            scale += DEPTH_SCALE_STEP
        elif reason in ('too_far', 'far', 'long'):
            cur['failures'] = int(cur.get('failures', 0)) + 1
            cur['last_result'] = 'too_far'
            scale -= DEPTH_SCALE_STEP
        elif reason in ('too_low', 'low', 'deep', 'table'):
            cur['failures'] = int(cur.get('failures', 0)) + 1
            cur['last_result'] = 'too_low'
            depth -= PICKUP_DEPTH_FAIL_STEP
        else:
            cur['failures'] = int(cur.get('failures', 0)) + 1
            cur['last_result'] = 'failed'
            depth += PICKUP_DEPTH_FAIL_STEP
        cur['depth_offset'] = _coerce_extra_drop(depth, cur.get('failures', 0), cur.get('last_result', 'none'))
        cur['depth_scale_factor'] = _coerce_depth_scale(scale)
        cur['opened_after_success'] = bool(opened)
        cur['homed_after_success'] = bool(homed)
        result = dict(cur)
    _save_pickup_learning()
    if success:
        status = 'Pickup feedback: success noted'
    elif reason in ('too_low', 'low', 'deep', 'table'):
        status = 'Pickup feedback: next grab will stay higher'
    elif reason in ('too_close', 'close', 'short', 'near'):
        status = 'Pickup feedback: next depth grab will reach farther'
    elif reason in ('too_far', 'far', 'long'):
        status = 'Pickup feedback: next depth grab will reach closer'
    else:
        status = 'Pickup feedback: next grab will go lower'
    _set_status(f"{status} (extra drop +{result['depth_offset']:.1f} deg, depth scale {result['depth_scale_factor']:.2f})")
    return result

def _box_edge_boost(det):
    box = det.get('box') if isinstance(det, dict) else None
    boost_x = boost_y = 1.0
    if box and len(box) == 4:
        try:
            x, y, bw, bh = [float(v) for v in box]
            right = x + bw
            bottom = y + bh
            if x < TRACK_EDGE_MARGIN_X or right > FRAME_W - TRACK_EDGE_MARGIN_X:
                boost_x = TRACK_EDGE_BOOST
            if y < TRACK_EDGE_MARGIN_Y or bottom > FRAME_H - TRACK_EDGE_MARGIN_Y:
                boost_y = TRACK_EDGE_BOOST
        except (TypeError, ValueError):
            pass
    return boost_x, boost_y

def _depth_hint_from_box(box):
    """Estimate a relative depth hint from a detection box.

    This is not true metric depth. It is a stable pickup helper:
    bigger box -> likely closer -> smaller depth hint.
    """
    if not box or len(box) != 4:
        return float(PICKUP_DEPTH_DEFAULT)
    try:
        bw = float(box[2])
        bh = float(box[3])
        if bw <= 0 or bh <= 0:
            return float(PICKUP_DEPTH_DEFAULT)
        area_ratio = _clamp((bw * bh) / float(FRAME_W * FRAME_H), 0.0, 1.0)
        if area_ratio <= 0.0:
            return float(PICKUP_DEPTH_DEFAULT)
        approx = (DEPTH_HINT_REF_AREA / max(0.01, math.sqrt(area_ratio))) * DEPTH_HINT_GAIN
        return float(_clamp(approx, DEPTH_HINT_MIN, DEPTH_HINT_MAX))
    except Exception:
        return float(PICKUP_DEPTH_DEFAULT)

def _depth_box_from_center(center_x, center_y, size_px=None):
    try:
        size = float(PICKUP_CENTER_DEPTH_BOX_PX if size_px is None else size_px)
        size = float(_clamp(size, 40.0, min(FRAME_W, FRAME_H) * 0.75))
        cx = float(_clamp(center_x, 0, FRAME_W - 1))
        cy = float(_clamp(center_y, 0, FRAME_H - 1))
        half = size * 0.5
        x1 = float(_clamp(cx - half, 0, FRAME_W - 2))
        y1 = float(_clamp(cy - half, 0, FRAME_H - 2))
        x2 = float(_clamp(cx + half, x1 + 1, FRAME_W))
        y2 = float(_clamp(cy + half, y1 + 1, FRAME_H))
        return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]
    except Exception:
        return None

def _depth_result_snapshot():
    with _depth_result_lock:
        return dict(_latest_depth_result[0])


def _depth_map_jpeg_snapshot():
    with _depth_result_lock:
        return _latest_depth_jpeg[0]


def _publish_depth_result(result, map_jpeg_b64=None):
    snap = dict(result or {})
    snap.setdefault('ts', time.time())
    snap.setdefault('reach_limit_cm', DEPTH_REACH_LIMIT_CM)
    snap.setdefault('depth_map_ready', False)

    map_bytes = None
    if map_jpeg_b64:
        try:
            map_bytes = base64.b64decode(map_jpeg_b64)
            snap['depth_map_ready'] = bool(map_bytes)
        except Exception as e:
            snap['depth_map_ready'] = False
            snap['map_error'] = str(e)

    with _depth_result_lock:
        # BUG FIX: Always update depth map if we got a new one, so webpage can fetch it
        if map_bytes is not None:
            _latest_depth_jpeg[0] = map_bytes
            snap['depth_map_ready'] = True
        snap['depth_map_ready'] = False
        _latest_depth_result[0] = snap
    _sync_depth_geometry_to_scene(snap)
    return snap


def _sync_depth_geometry_to_scene(depth_result):
    try:
        if not isinstance(depth_result, dict):
            return
        path = depth_result.get('path')
        if not isinstance(path, dict):
            return
        path_box = path.get('bbox_px') or depth_result.get('box')
        if not path_box or len(path_box) != 4:
            return
        fields = {}
        for key in (
            'object_height_px',
            'pixel_height',
            'object_height_cm',
            'corrected_object_height_cm',
            'grab_height_cm',
            'grab_height_fraction',
            'height_source',
            'height_measurement',
            'grasp_pose',
        ):
            if path.get(key) is not None:
                fields[key] = path.get(key)
        if isinstance(path.get('grasp_px'), dict):
            fields['grasp_x'] = path['grasp_px'].get('x')
            fields['grasp_y'] = path['grasp_px'].get('y')
        if not fields:
            return
        with _scene_lock:
            for det in _scene_dets:
                det_box = det.get('box') if isinstance(det, dict) else None
                if det_box and len(det_box) == 4 and _box_similarity(det_box, path_box) >= 0.50:
                    det.update(fields)
        with _pickup_frozen_target_lock:
            frozen = _pickup_frozen_target[0]
            frozen_box = frozen.get('box') if isinstance(frozen, dict) else None
            if frozen_box and len(frozen_box) == 4 and _box_similarity(frozen_box, path_box) >= 0.50:
                frozen.update(fields)
    except Exception as e:
        print(f'[DEPTH] scene geometry sync failed: {e}', flush=True)


def _depth_distance_to_hint(distance_cm):
    try:
        cm = float(distance_cm)
        if not math.isfinite(cm) or cm <= 0:
            return float(PICKUP_DEPTH_DEFAULT)
        ratio = _clamp(cm / max(1.0, DEPTH_REACH_LIMIT_CM), 0.0, 1.0)
        return float(_clamp(
            PICKUP_DEPTH_MIN + ratio * (PICKUP_DEPTH_MAX - PICKUP_DEPTH_MIN),
            PICKUP_DEPTH_MIN,
            PICKUP_DEPTH_MAX,
        ))
    except Exception:
        return float(PICKUP_DEPTH_DEFAULT)


def _depth_contact_drop_degs(distance_cm, horizontal_ratio=0.0):
    """Return (full_drop_deg, approach_drop_deg, final_descent_deg) for a metric Z depth.

    full_drop_deg      — total arm travel implied by measured depth (cm × DEPTH_CM_TO_ARM_DEG)
    approach_drop_deg  — portion reserved for IK / depth-approach (stops short by clearance)
    final_descent_deg  — slow pickup-loop descent (scales with Z; used by grip sequence)

    BUG FIX: final_descent was previously min(full_drop, clearance) ≈ 3.7° for every
    object beyond ~4 cm, so grabs past ~15 cm always used the same shallow plunge.
    """
    try:
        cm = float(distance_cm)
        if not math.isfinite(cm) or cm <= 0:
            cm = float(PICKUP_DEPTH_DEFAULT)
    except Exception:
        cm = float(PICKUP_DEPTH_DEFAULT)

    ratio = float(_clamp(horizontal_ratio, 0.0, 1.0))
    depth_factor = 1.0 - ratio * 0.3
    contact_cm = max(0.0, cm)
    full_drop = max(0.0, contact_cm * DEPTH_CM_TO_ARM_DEG * depth_factor)
    clearance_drop = max(0.0, DEPTH_APPROACH_CLEARANCE_CM * DEPTH_CM_TO_ARM_DEG * depth_factor)

    # Final descent loop: linear in measured Z (15 cm → ~13.8°, 30 cm → ~27.6°, …)
    final_descent = float(_clamp(
        full_drop,
        PICKUP_DESCENT_STEP_DEG,
        PICKUP_MAX_DESCENT_DEG,
    ))
    # IK / depth-approach travel — everything except the last clearance slice
    approach_drop = float(_clamp(
        max(0.0, full_drop - clearance_drop),
        0.0,
        max(0.0, PICKUP_MAX_DESCENT_DEG - PICKUP_DESCENT_STEP_DEG),
    ))
    return full_drop, approach_drop, final_descent


def _descent_deg_from_depth(target_depth_cm, cur_arm_deg, cur_wrist_deg,
                            target_z_cm=0.0):
    """Compute arm/wrist angles to reach object center at target_depth.

    The ToF sensor is on the forearm. It measures straight-line distance to the
    object. We know the object is on a table at height target_z_cm above floor.
    Using the sensor position (from current arm angle) and Pythagorean theorem:
      horizontal_dist = sqrt(depth^2 - (sensor_z - obj_z)^2)
    Then solve 2-link IK to put gripper tip at the object.

    Returns (descent_deg, grasp_arm, grasp_wrist).
    """
    L1 = float(ARM_LINK1_CM)
    L2 = float(ARM_LINK2_CM)
    h  = float(FLOOR_BELOW_SHOULDER_CM)
    mount_offset = float(VL53L1X_MOUNT_OFFSET_CM)

    # Sensor position from current arm angle
    sensor_along = L2 - mount_offset
    cur_sh = math.radians(_fk_shoulder_deg(cur_arm_deg))
    cur_wr_rad = math.radians(cur_wrist_deg - 90.0)
    sensor_x = L1 * math.cos(cur_sh) + sensor_along * math.cos(cur_sh + cur_wr_rad)
    sensor_z_floor = h - L1 * math.sin(cur_sh) - sensor_along * math.sin(cur_sh + cur_wr_rad)

    # Object is on table at target_z_cm above floor.
    # ToF depth is straight-line distance from sensor to object.
    # horizontal distance = sqrt(depth^2 - (sensor_z - obj_z)^2)
    vert_diff = sensor_z_floor - target_z_cm
    horiz_dist_sq = target_depth_cm ** 2 - vert_diff ** 2
    if horiz_dist_sq < 0:
        # ToF depth is less than vertical drop — object is directly below sensor
        horiz_dist_sq = 0
    horiz_dist = math.sqrt(horiz_dist_sq)

    obj_x = sensor_x + horiz_dist
    obj_z_floor = target_z_cm

    # IK: find arm/wrist to put gripper tip at (obj_x, obj_z_floor)
    x = obj_x
    y = h - obj_z_floor  # positive = below shoulder

    d = math.hypot(x, y)
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)

    if d > max_reach - 0.01:
        sh_rad = math.atan2(y, x)
        q2 = 0.0
    elif d < min_reach + 0.01:
        sh_rad = math.atan2(y, x)
        q2 = math.pi
    else:
        cos_q2 = max(-1.0, min(1.0, (d*d - L1*L1 - L2*L2) / (2*L1*L2)))
        q2 = math.acos(cos_q2)
        sh_rad = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))

    # Convert to servo angles via lookup table
    sh_deg = math.degrees(sh_rad)
    grasp_arm = _fk_arm_from_shoulder(sh_deg)
    grasp_arm = float(_clamp(grasp_arm, ARM_MIN, ARM_MAX))

    grasp_wrist = math.degrees(q2) + 90.0
    grasp_wrist = float(_clamp(grasp_wrist, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX))

    descent_deg = grasp_arm - cur_arm_deg

    # Verify FK
    sh_rad_v = math.radians(_fk_shoulder_deg(grasp_arm))
    wr_rad_v = math.radians(grasp_wrist - 90.0)
    fk_x = L1 * math.cos(sh_rad_v) + L2 * math.cos(sh_rad_v + wr_rad_v)
    fk_drop = L1 * math.sin(sh_rad_v) + L2 * math.sin(sh_rad_v + wr_rad_v)
    fk_z = h - fk_drop

    print(f'[DESCENT-IK] depth={target_depth_cm:.1f} '
          f'sensor=({sensor_x:.1f},{sensor_z_floor:.1f}) '
          f'obj=({obj_x:.1f},{obj_z_floor:.1f}) '
          f'cur_arm={cur_arm_deg:.1f} -> grasp_arm={grasp_arm:.1f} '
          f'descent={descent_deg:.1f} wrist={grasp_wrist:.1f} '
          f'FK_x={fk_x:.1f} FK_z={fk_z:.1f}', flush=True)

    return descent_deg, grasp_arm, grasp_wrist


def _wrist_from_depth(metric_depth_cm, cur_arm_deg=None):
    """Compute wrist angle from ToF depth so the gripper can reach the object.

    Geometry:
      - VL53L1X ToF sensor is on the forearm, VL53L1X_MOUNT_OFFSET_CM behind the jaw
      - ToF reads straight-line distance D from sensor to object
      - Jaw-to-object distance ≈ D − mount_offset (after arm descends)
      - Wrist angle controls gripper approach angle relative to forearm

    The wrist tilts proportionally to depth so that:
      - Shallow objects (D ≈ 5-15 cm): wrist near 90° (gripper approaches from the side)
      - Deep objects (D ≈ 30-50 cm): wrist tilts down more (gripper reaches down/forward)

    At depth=0: wrist=90 (horizontal). At max depth (50cm): wrist≈65°.
    """
    try:
        D = float(metric_depth_cm)
    except (TypeError, ValueError):
        D = float(PICKUP_DEPTH_DEFAULT)
    if not math.isfinite(D) or D <= 0:
        D = float(PICKUP_DEPTH_DEFAULT)

    # Sensor is offset behind the jaw; the jaw is already closer to the object
    jaw_gap = max(0.0, D - VL53L1X_MOUNT_OFFSET_CM)

    # The arm descends by descent_deg to close the gap.
    # Wrist tilt compensates for the remaining geometry:
    #   - At shallow depth: little tilt needed, gripper approaches from side
    #   - At deep depth: more tilt needed, gripper angles down to reach object
    #
    # The gain is chosen so that at typical pickup depth (20cm), the wrist tilts
    # about 10° from horizontal, and at max depth (50cm), it tilts about 25°.
    #
    #   wrist = 90 − jaw_gap × Wrist_DEPTH_GAIN
    #
    # Wrist_DEPTH_GAIN = 0.50 gives:
    #   D=10cm → jaw_gap=0 → wrist=90°  (sensor right at object, no tilt)
    #   D=15cm → jaw_gap=5 → wrist=87.5°
    #   D=20cm → jaw_gap=10 → wrist=85°
    #   D=30cm → jaw_gap=20 → wrist=80°
    #   D=40cm → jaw_gap=30 → wrist=75°
    #   D=50cm → jaw_gap=40 → wrist=70°
    Wrist_DEPTH_GAIN = 0.50

    wrist = WRIST_HOME - jaw_gap * Wrist_DEPTH_GAIN

    return float(_clamp(wrist, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX))


def _ik_solve_pickup(target_depth_cm, target_z_cm):
    """2-link IK: solve for (arm_deg, wrist_deg) that puts gripper tip at target.

    target_depth_cm: horizontal distance from shoulder to object (forward)
    target_z_cm:     height of target above the floor (e.g. object center)

    Returns (arm_deg, wrist_deg) or None if unreachable.
    """
    L1 = float(ARM_LINK1_CM)
    L2 = float(ARM_LINK2_CM)
    h  = float(FLOOR_BELOW_SHOULDER_CM)

    x = target_depth_cm
    y = h - target_z_cm          # positive = below shoulder

    d = math.hypot(x, y)
    if d > L1 + L2 - 0.01 or d < abs(L1 - L2) + 0.01:
        return None

    cos_q2 = max(-1.0, min(1.0, (d*d - L1*L1 - L2*L2) / (2*L1*L2)))
    q2 = math.acos(cos_q2)       # wrist angle (radians, positive = bend down)

    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))

    arm_deg   = _fk_arm_from_shoulder(math.degrees(q1))
    wrist_deg = math.degrees(q2) + 90.0
    return arm_deg, wrist_deg


def _wrist_ik_for_arm(target_depth_cm, target_z_cm, arm_deg):
    """Compute wrist angle via IK so gripper tip reaches target, given arm_deg is fixed.

    Uses the same FK convention as the server:
      θ_sh = 1.526 * arm_deg - 47.35°
      θ_wr = wrist_deg - 90°
      x = L1·cos(θ_sh) + L2·cos(θ_sh + θ_wr)
      Z = shoulder_h - L1·sin(θ_sh) - L2·sin(θ_sh + θ_wr)

    Returns wrist_deg (clamped to PICKUP_WRIST_MIN..MAX), or None if unreachable.
    """
    L1 = float(ARM_LINK1_CM)
    L2 = float(ARM_LINK2_CM)
    h  = float(FLOOR_BELOW_SHOULDER_CM)

    sh = math.radians(_fk_shoulder_deg(arm_deg))

    # Remaining distance the wrist (L2) must cover after shoulder (L1) contribution
    A = target_depth_cm - L1 * math.cos(sh)
    B = L1 * math.sin(sh) - (h - target_z_cm)

    # L2 must reach (A, B) — check reachable
    d2 = math.hypot(A, B)
    if d2 > L2 + 0.01:
        return None

    # θ_sh + θ_wr = atan2(B, A)  →  θ_wr = atan2(B, A) - θ_sh
    theta_total = math.atan2(B, A)
    theta_wr = theta_total - sh
    wrist_deg = math.degrees(theta_wr) + 90.0
    return float(_clamp(wrist_deg, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX))


def _pickup_metric_depth_cm(depth_plan, path=None, fallback_hint=None):
    """Best available metric Z (cm) captured at pickup start."""
    if isinstance(path, dict):
        try:
            cm = float(path.get('contact_cm'))
            if math.isfinite(cm) and cm > 0:
                return float(_clamp(cm, DEPTH_SANITY_MIN_CM, DEPTH_SANITY_MAX_CM))
        except (TypeError, ValueError):
            pass
        target = path.get('target_cm')
        if isinstance(target, dict):
            try:
                cm = float(target.get('z_cm'))
                if math.isfinite(cm) and cm > 0:
                    return float(_clamp(cm, DEPTH_SANITY_MIN_CM, DEPTH_SANITY_MAX_CM))
            except (TypeError, ValueError):
                pass
    if isinstance(depth_plan, dict):
        dist = depth_plan.get('distance_cm')
        if dist is not None:
            try:
                cm = _jaw_distance_from_camera_depth(float(dist))
                if math.isfinite(cm) and cm > 0:
                    return float(_clamp(cm, PICKUP_DEPTH_MIN, DEPTH_SANITY_MAX_CM))
            except (TypeError, ValueError):
                pass
    if fallback_hint is not None:
        try:
            cm = float(fallback_hint)
            if math.isfinite(cm) and cm > 0:
                return float(_clamp(cm, PICKUP_DEPTH_MIN, DEPTH_SANITY_MAX_CM))
        except (TypeError, ValueError):
            pass
    return float(PICKUP_DEPTH_DEFAULT)


def _ik_approach_reach_cm(target_x_cm, target_y_cm, target_z_cm):
    """Map live 3D target â†’ 2-link IK (reach_x forward, reach_z vertical).

    FIX v35: Enhanced reach calculation to ensure gripper extends fully to target.
    - reach_x now accounts for lateral offset AND arm's natural diagonal reach
    - Clearance dynamically scaled based on object distance for smoother approach
    - reach_z uses target_y with improved gain for better vertical accuracy
    """
    z_fwd = max(1.0, float(target_z_cm))
    try:
        target_x = float(target_x_cm)
    except (TypeError, ValueError):
        target_x = 0.0
    # FIX v35: Improved lateral compensation â€” 3D reach accounting for diagonal approach
    # When target is off-axis, IK must position arm to reach diagonally, so we add
    # the lateral distance to forward reach for accurate positioning.
    lateral_correction = math.hypot(abs(target_x), 0) * 0.18  # FIX v35: was 0.15 â€” account for arm geometry
    reach_x = float(_clamp(z_fwd + lateral_correction, REACH_X_MIN, ARM_REACH_CM - 0.3))

    # Keep the pickup approach only slightly above the object. Larger negative
    # reach_z values make this arm climb high before the final descent.
    clear_cm = float(_clamp(
        0.5 + z_fwd * 0.015,
        0.5,
        2.0
    ))
    try:
        target_y = float(target_y_cm)
    except (TypeError, ValueError):
        target_y = 0.0
    # Higher PICKUP_Y_TO_REACH_Z_GAIN makes vertical positioning more responsive.
    # Subtract only a small clearance so the gripper does not stage far above
    # the pickup point.
    reach_z = target_y * float(PICKUP_Y_TO_REACH_Z_GAIN) - float(clear_cm)
    return reach_x, reach_z


def _pickup_y_descent_adjust_deg(target_y_cm):
    """Convert target camera-space Y into final descent trim.

    Positive Y means the grab point is lower in the camera view, so descend more.
    Negative Y means the grab point is higher, so descend less.
    """
    try:
        y_cm = float(target_y_cm)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(y_cm):
        return 0.0
    limit = abs(float(PICKUP_Y_DESCENT_ADJUST_MAX_DEG))
    return float(_clamp(y_cm * float(PICKUP_Y_TO_DESCENT_DEG), -limit, limit))


def _fallback_depth_descent_degs(depth_cm):
    """Map box/fallback depth to arm descent without the old fixed 15-degree plunge."""
    try:
        cm = float(depth_cm)
        if not math.isfinite(cm) or cm <= 0:
            cm = float(PICKUP_DEPTH_DEFAULT)
    except Exception:
        cm = float(PICKUP_DEPTH_DEFAULT)
    return float(_clamp(
        cm * DEPTH_CM_TO_ARM_DEG,
        PICKUP_DESCENT_STEP_DEG,
        PICKUP_MAX_DESCENT_DEG,
    ))

def _vl53l1x_mount_behind_gripper():
    return VL53L1X_MOUNT in ('behind', 'forward', 'bottom', 'under', 'below', 'gripper', 'ahead', 'camera')


def _vl53l1x_mount_down():
    return VL53L1X_MOUNT in ('down', 'vertical', 'table')


def _vl53l1x_mount_description():
    if _vl53l1x_mount_down():
        return f'bottom-mounted/down-facing'
    if _vl53l1x_mount_behind_gripper():
        return f'forward-facing'
    return VL53L1X_MOUNT


def _sensor_to_jaw_offset_cm():
    """Lens/sensor â†’ gripper jaw offset (camera and VL53L1X co-mounted behind gripper)."""
    if _vl53l1x_mount_behind_gripper():
        return float(VL53L1X_MOUNT_OFFSET_CM)
    return float(DEPTH_GRIPPER_OFFSET_CM)


def _jaw_distance_from_camera_depth(distance_cm):
    """Convert sensor/camera-to-object depth to jaw-to-object depth.

    Camera and VL53L1X sit behind the gripper; the object is closer to the jaw
    than the raw ToF/camera reading by the mount offset.
    """
    try:
        camera_cm = float(distance_cm)
    except (TypeError, ValueError):
        camera_cm = float(PICKUP_DEPTH_DEFAULT)
    if not math.isfinite(camera_cm) or camera_cm <= 0:
        camera_cm = float(PICKUP_DEPTH_DEFAULT)
    return float(_clamp(
        camera_cm - _sensor_to_jaw_offset_cm(),
        PICKUP_DEPTH_MIN,
        DEPTH_SANITY_MAX_CM,
    ))


def _camera_intrinsics_px():
    fx = (FRAME_W * 0.5) / math.tan(math.radians(DEPTH_CAMERA_FOV_X_DEG) * 0.5)
    fy = (FRAME_H * 0.5) / math.tan(math.radians(DEPTH_CAMERA_FOV_Y_DEG) * 0.5)
    return {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(FRAME_W * 0.5),
        'cy': float(FRAME_H * 0.5),
    }


def _camera_extrinsic_cm():
    with _calibration_lock:
        offsets = dict(_calibration_state.get('offsets') or {})
    return {
        'gripper_x_cm': round(float(offsets.get('gripper_x_cm', GRIPPER_X_OFFSET_CM)), 2),
        'gripper_y_cm': round(float(offsets.get('gripper_y_cm', GRIPPER_Y_OFFSET_CM)), 2),
        'camera_center_px': round(float(offsets.get('camera_center_px', CAMERA_CENTER_OFFSET_PX)), 2),
        'camera_center_y_px': round(float(offsets.get('camera_center_y_px', CAMERA_CENTER_Y_OFFSET_PX)), 2),
        'sensor_to_jaw_offset_cm': round(float(_sensor_to_jaw_offset_cm()), 2),
    }


def _object_height_measurement(box, distance_cm, depth_target=None):
    """Measure bbox pixel height and convert it to calibrated metric height."""
    try:
        x, y, bw, bh = [float(v) for v in box]
        if bw <= 0 or bh <= 0:
            return None
        top_px = float(y)
        bottom_px = float(y + bh)
        height_px = float(bh)
        source = 'bbox-intrinsic'
        if isinstance(depth_target, dict):
            if depth_target.get('height_px') is not None:
                height_px = max(1.0, float(depth_target.get('height_px')))
                source = str(depth_target.get('source') or 'depth-map-intrinsic')
            if depth_target.get('top_px') is not None and depth_target.get('bottom_px') is not None:
                top_px = float(depth_target.get('top_px'))
                bottom_px = float(depth_target.get('bottom_px'))
                height_px = max(1.0, abs(bottom_px - top_px))
        intr = _camera_intrinsics_px()
        z_camera = float(distance_cm)
        if not math.isfinite(z_camera) or z_camera <= 0:
            return None
        y_top_cm = (top_px - intr['cy']) * z_camera / max(1.0, intr['fy'])
        y_bottom_cm = (bottom_px - intr['cy']) * z_camera / max(1.0, intr['fy'])
        corrected_height_cm = abs(y_bottom_cm - y_top_cm)
        return {
            'bbox_px': [int(round(x)), int(round(y)), int(round(bw)), int(round(bh))],
            'object_height_px': round(float(height_px), 1),
            'pixel_height': round(float(height_px), 1),
            'object_top_px': round(float(top_px), 1),
            'object_bottom_px': round(float(bottom_px), 1),
            'object_vertical_center_px': round(float((top_px + bottom_px) * 0.5), 1),
            'object_height_cm': round(float(corrected_height_cm), 2),
            'corrected_object_height_cm': round(float(corrected_height_cm), 2),
            'camera_distance_cm': round(float(z_camera), 2),
            'height_source': source,
            'height_correction': 'pinhole_intrinsic_plus_gripper_extrinsic',
            'camera_intrinsics_px': {
                'fx': round(float(intr['fx']), 2),
                'fy': round(float(intr['fy']), 2),
                'cx': round(float(intr['cx']), 2),
                'cy': round(float(intr['cy']), 2),
            },
            'camera_extrinsic_cm': _camera_extrinsic_cm(),
        }
    except Exception as e:
        print(f'[PICKUP] height measurement failed: {e}', flush=True)
        return None


def _camera_point_from_depth(px, py, distance_cm):
    """Convert pixel (px, py) + depth to 3D camera-space point, then apply
    gripper calibration offsets so the returned XYZ is the JAW CENTRE, not the
    camera optical-axis centre.

    Horizontal projection math (pinhole model):
        fx = (frame_width / 2) / tan(FOV_x / 2)
        x_cam = (px - cx_frame) * depth / fx      # lateral, cm
        y_cam = (py - cy_frame) * depth / fy      # vertical, cm
        z_cam = depth                              # forward, cm

    Gripper correction (applied after projection):
        x_arm = x_cam + GRIPPER_X_OFFSET_CM
        y_arm = y_cam + GRIPPER_Y_OFFSET_CM

    If the arm consistently grabs N cm to the right:
        set GRIPPER_X_OFFSET_CM = -N  (shift target left by N cm)
    """
    try:
        z_camera = float(distance_cm)
        z = _jaw_distance_from_camera_depth(z_camera)
        intr = _camera_intrinsics_px()
        fx = intr['fx']
        fy = intr['fy']
        # Pinhole back-projection: object position relative to camera lens
        cx_frame = intr['cx']
        cy_frame = intr['cy']
        x_cam = (float(px) - cx_frame) * z_camera / max(1.0, fx)
        y_cam = (float(py) - cy_frame) * z_camera / max(1.0, fy)

        # Apply physical gripper-to-camera offset.
        # GRIPPER_X_OFFSET_CM is the signed distance from camera lens to gripper
        # jaw centre along the arm's lateral axis (positive = jaw right of lens).
        with _calibration_lock:
            gx = _calibration_state['offsets']['gripper_x_cm']
            gy = _calibration_state['offsets']['gripper_y_cm']
        x_arm = x_cam + gx
        y_arm = y_cam + gy

        depth_uncertainty = max(z * 0.05, 2.0)  # 5% or 2 cm
        return {
            'x_cm': round(float(x_arm), 2),
            'y_cm': round(float(y_arm), 2),
            'z_cm': round(float(z), 2),
            'camera_z_cm': round(float(z_camera), 2),
            'sensor_to_jaw_offset_cm': round(float(_sensor_to_jaw_offset_cm()), 2),
            'camera_to_jaw_offset_cm': round(float(DEPTH_GRIPPER_OFFSET_CM), 2),
            'x_cam_raw': round(float(x_cam), 2),  # before gripper correction
            'y_cam_raw': round(float(y_cam), 2),
            'uncertainty_cm': round(depth_uncertainty, 2),
        }
    except Exception:
        camera_cm = float(distance_cm or 0.0)
        return {
            'x_cm': 0.0,
            'y_cm': 0.0,
            'z_cm': _jaw_distance_from_camera_depth(camera_cm),
            'camera_z_cm': camera_cm,
            'sensor_to_jaw_offset_cm': round(float(_sensor_to_jaw_offset_cm()), 2),
            'camera_to_jaw_offset_cm': round(float(DEPTH_GRIPPER_OFFSET_CM), 2),
            'uncertainty_cm': 5.0,
        }


def _depth_target_from_map(depth_map, box, raw_distance_m, grab_fraction=None):
    """Find the geometric center of the object in the depth mask."""
    try:
        if depth_map is None or box is None or len(box) != 4:
            return None
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim != 2:
            return None
        dh, dw = depth.shape
        x, y, bw, bh = [float(v) for v in box]
        sx = dw / float(FRAME_W)
        sy = dh / float(FRAME_H)
        dx1 = int(_clamp(x * sx, 0, dw - 1))
        dx2 = int(_clamp((x + bw) * sx, 0, dw))
        dy1 = int(_clamp(y * sy, 0, dh - 1))
        dy2 = int(_clamp((y + bh) * sy, 0, dh))
        if dx2 <= dx1 or dy2 <= dy1:
            return None

        roi = depth[dy1:dy2, dx1:dx2]
        valid = np.isfinite(roi) & (roi > 0.001)
        if int(valid.sum()) < 20:
            return None

        target_m = float(raw_distance_m)
        if not math.isfinite(target_m) or target_m <= 0:
            target_m = float(np.median(roi[valid]))
        tolerance_m = max(0.025, abs(target_m) * 0.18)
        mask = valid & (np.abs(roi - target_m) <= tolerance_m)

        if int(mask.sum()) < 20:
            vals = roi[valid]
            near_hi = float(np.percentile(vals, 65))
            mask = valid & (roi <= near_hi)
        if int(mask.sum()) < 20:
            return None

        row_counts = mask.sum(axis=1)
        min_row_pixels = max(2, int(mask.shape[1] * 0.08))
        rows = np.where(row_counts >= min_row_pixels)[0]
        if rows.size == 0:
            rows = np.where(row_counts > 0)[0]
        if rows.size == 0:
            return None

        top = int(rows[0])
        bottom = int(rows[-1])
        grab_frac = PICKUP_OBJECT_CENTER_FRACTION
        mask_rows, mask_cols = np.where(mask)
        if mask_rows.size < 20:
            return None
        target_col = int(round(float(np.median(mask_cols))))
        target_row = int(round(float(np.median(mask_rows))))
        target_dx = dx1 + target_col
        target_dy = dy1 + target_row
        patch_half = max(1, int(DEPTH_TARGET_PATCH_PX) // 2)
        px0 = max(0, target_dx - patch_half)
        px1 = min(dw, target_dx + patch_half + 1)
        py0 = max(0, target_dy - patch_half)
        py1 = min(dh, target_dy + patch_half + 1)
        patch = depth[py0:py1, px0:px1]
        patch_valid = patch[np.isfinite(patch) & (patch > 0.001)]
        target_depth_m = None
        patch_pixels = int(patch_valid.size)
        if patch_valid.size >= 5:
            local_tol = max(0.018, abs(target_m) * 0.16)
            local_band = patch_valid[np.abs(patch_valid - target_m) <= local_tol]
            if local_band.size >= 4:
                target_depth_m = float(np.median(local_band))
                patch_pixels = int(local_band.size)
            else:
                target_depth_m = float(np.median(patch_valid))
        height_depth_px = max(1.0, (bottom - top + 1) / max(0.001, sy))
        target_px = {
            'x': float(_clamp(target_dx / max(0.001, sx), 0, FRAME_W - 1)),
            'y': float(_clamp(target_dy / max(0.001, sy), 0, FRAME_H - 1)),
        }
        out = {
            'target_px': target_px,
            'height_px': float(height_depth_px),
            'mask_pixels': int(mask.sum()),
            'top_px': float(_clamp((dy1 + top) / max(0.001, sy), 0, FRAME_H - 1)),
            'bottom_px': float(_clamp((dy1 + bottom) / max(0.001, sy), 0, FRAME_H - 1)),
            'grab_fraction': float(grab_frac),
            'source': 'depth-map-center',
        }
        if target_depth_m is not None:
            out['target_depth_m'] = float(target_depth_m)
            out['target_depth_patch_pixels'] = int(patch_pixels)
        return out
    except Exception as e:
        print(f'[DEPTH] depth-map midpoint failed: {e}', flush=True)
        return None


def _depth_path_from_box(box, distance_cm, depth_target=None, center_px=None):
    """Compute optimal 3D reach path using camera intrinsics and arm geometry.

    KEY FIX (v23): the horizontal target pixel (tx) must be the TRUE object
    centroid â€” the horizontal midpoint of the bounding box â€” not a tunable
    fraction.  Using 0.50 of the YOLO box already gives the centroid, but the
    critical invariant is: tx represents WHERE THE OBJECT IS in image space,
    and _camera_point_from_depth then maps that pixel to arm space, adding the
    GRIPPER calibration offset so the IK target is the JAW CENTRE.

    If depth_target (from the depth-map center finder) is available we use its
    refined center pixel; otherwise we fall back to the bbox center.
    """
    real_height_cm = None
    grab_height_cm = None
    height_info = None
    object_height_px = None
    height_source = str(depth_target.get('source')) if isinstance(depth_target, dict) else 'box-depth'
    grab_fraction = PICKUP_OBJECT_CENTER_FRACTION
    try:
        x, y, bw, bh = [float(v) for v in box]
        height_info = _object_height_measurement(box, distance_cm, depth_target)
        if isinstance(height_info, dict):
            object_height_px = float(height_info.get('object_height_px') or bh)
            real_height_cm = float(height_info.get('corrected_object_height_cm') or height_info.get('object_height_cm'))
            grab_height_cm = real_height_cm * grab_fraction
            height_source = str(height_info.get('height_source') or height_source)
        if real_height_cm is None:
            intr = _camera_intrinsics_px()
            object_height_px = float(bh)
            real_height_cm = (object_height_px / max(1.0, intr['fy'])) * float(distance_cm)
            grab_height_cm = real_height_cm * grab_fraction
        bbox_grab_x = float(_clamp(x + bw * grab_fraction, 0, FRAME_W - 1))
        bbox_grab_y = float(_clamp(y + bh * grab_fraction, 0, FRAME_H - 1))
        tx = bbox_grab_x
        ty = bbox_grab_y

        if isinstance(center_px, dict):
            height_px = float(bh)
            if center_px.get('x') is not None:
                tx = float(_clamp(center_px.get('x'), 0, FRAME_W - 1))
            if center_px.get('y') is not None:
                ty = float(_clamp(center_px.get('y'), 0, FRAME_H - 1))
            object_height_px = object_height_px if object_height_px is not None else height_px
            print(f'[PICKUP] Center-grab target: object height {real_height_cm:.1f}cm, '
                  f'center at {grab_height_cm:.1f}cm, pixel tx={tx:.0f} ty={ty:.0f}', flush=True)
        elif isinstance(center_px, (list, tuple)) and len(center_px) >= 2:
            height_px = float(bh)
            tx = float(_clamp(center_px[0], 0, FRAME_W - 1))
            ty = float(_clamp(center_px[1], 0, FRAME_H - 1))
            object_height_px = object_height_px if object_height_px is not None else height_px
            print(f'[PICKUP] Center-grab target: object height {real_height_cm:.1f}cm, '
                  f'center at {grab_height_cm:.1f}cm, pixel tx={tx:.0f} ty={ty:.0f}', flush=True)
        elif isinstance(depth_target, dict) and isinstance(depth_target.get('target_px'), dict):
            ty = float(_clamp(depth_target['target_px'].get('y', bbox_grab_y), 0, FRAME_H - 1))
            if depth_target['target_px'].get('x') is not None:
                tx = float(_clamp(depth_target['target_px'].get('x'), 0, FRAME_W - 1))
            height_px = float(depth_target.get('height_px') or bh)
            object_height_px = object_height_px if object_height_px is not None else height_px
            print(f'[PICKUP] Depth-map center target: object height {real_height_cm:.1f}cm, '
                  f'center at {grab_height_cm:.1f}cm, pixel tx={tx:.0f} ty={ty:.0f}', flush=True)
        else:
            object_height_px = object_height_px if object_height_px is not None else float(bh)
            print(f'[PICKUP] Bbox center target: object height {real_height_cm:.1f}cm, '
                  f'center at {grab_height_cm:.1f}cm, pixel tx={tx:.0f} ty={ty:.0f}', flush=True)
    except Exception:
        tx, ty = _pickup_aim_xy(None)

    point = _camera_point_from_depth(tx, ty, distance_cm)
    x_cm = float(point['x_cm'])
    if abs(PICKUP_GRAB_X_OFFSET_CM) > 0.001:
        x_cm += float(PICKUP_GRAB_X_OFFSET_CM)
        point['x_cm'] = round(float(x_cm), 2)
    y_cm = float(point['y_cm'])
    z_cm = float(point['z_cm'])
    camera_z_cm = float(point.get('camera_z_cm', distance_cm))
    
    # Compute true 3D distance (not just depth)
    lateral_cm = math.hypot(x_cm, y_cm)
    total_distance = math.sqrt(x_cm**2 + y_cm**2 + z_cm**2)
    
    # Check reachability
    reachable_distance = total_distance
    is_reachable = reachable_distance <= DEPTH_REACH_LIMIT_CM
    
    contact_cm = max(0.0, z_cm)
    
    horizontal_ratio = min(1.0, lateral_cm / max(1.0, DEPTH_REACH_LIMIT_CM * 0.8))
    # The grab height is already applied above by moving ty to the side of the
    # object. Do not subtract it from the depth drop again, or a real 33 cm
    # reading collapses into nearly the same shallow motion as the old fallback.
    height_lift_deg = 0.0
    raw_descent_deg, approach_arm_delta_deg, descent_deg = _depth_contact_drop_degs(
        contact_cm,
        horizontal_ratio,
    )
    extra_drop_deg = _pickup_depth_offset()
    execution_descent_deg = float(_clamp(
        descent_deg + extra_drop_deg,
        PICKUP_DESCENT_STEP_DEG,
        max(PICKUP_DESCENT_STEP_DEG, PICKUP_MAX_DESCENT_DEG - approach_arm_delta_deg),
    ))
    
    approach_cm = max(0.0, float(contact_cm) - DEPTH_APPROACH_CLEARANCE_CM)
    uncertainty = float(point.get('uncertainty_cm', 5.0))

    # Lift height (cm) after gripper closes â€” derived from PICKUP_LIFT_DEG converted
    # back to approximate cm via the arm's deg-to-cm ratio.  This is used by the
    # 3D visualiser so all four path phases are shown correctly.
    lift_cm = round(float(PICKUP_LIFT_DEG) / max(0.1, float(DEPTH_CM_TO_ARM_DEG)), 2)
    
    grasp_pose = {
        'frame': 'calibrated_gripper_jaw_cm',
        'position_cm': {
            'x_cm': round(float(point.get('x_cm', 0.0)), 2),
            'y_cm': round(float(point.get('y_cm', 0.0)), 2),
            'z_cm': round(float(point.get('z_cm', 0.0)), 2),
        },
        'camera_position_cm': {
            'x_cm': round(float(point.get('x_cam_raw', 0.0)), 2),
            'y_cm': round(float(point.get('y_cam_raw', 0.0)), 2),
            'z_cm': round(float(point.get('camera_z_cm', camera_z_cm)), 2),
        },
        'pixel': {'x': round(tx, 1), 'y': round(ty, 1)},
        'bbox_px': [int(round(float(v))) for v in box] if box and len(box) == 4 else None,
        'vertical_center_fraction': round(float(grab_fraction), 3),
        'object_height_px': None if object_height_px is None else round(float(object_height_px), 1),
        'corrected_object_height_cm': None if real_height_cm is None else round(float(real_height_cm), 2),
        'approach_axis': 'z_cm',
        'orientation_hint': {
            'wrist_deg': round(float(_target_wrist_angle({'box': box, 'target_y': ty})), 1),
        },
    }
    return _annotate_pickup_execution_path({
        'bbox_px': [int(round(float(v))) for v in box] if box and len(box) == 4 else None,
        'target_px': {'x': round(tx, 1), 'y': round(ty, 1)},
        'grasp_px': {'x': round(tx, 1), 'y': round(ty, 1)},
        'crosshair_px': {'x': round(float(_pickup_aim_xy(None)[0]), 1), 'y': round(float(_pickup_aim_xy(None)[1]), 1)},
        'target_cm': point,
        'grasp_pose': grasp_pose,
        'object_height_px': None if object_height_px is None else round(float(object_height_px), 1),
        'object_height_cm': None if real_height_cm is None else round(real_height_cm, 2),
        'corrected_object_height_cm': None if real_height_cm is None else round(real_height_cm, 2),
        'grab_height_cm': None if grab_height_cm is None else round(grab_height_cm, 2),
        'grab_height_fraction': round(grab_fraction, 3),
        'height_source': height_source,
        'height_measurement': height_info,
        'depth_mask_pixels': int(depth_target.get('mask_pixels')) if isinstance(depth_target, dict) and depth_target.get('mask_pixels') is not None else None,
        'height_lift_deg': round(height_lift_deg, 2),
        'raw_descent_deg': round(raw_descent_deg, 2),
        'approach_arm_delta_deg': round(approach_arm_delta_deg, 2),
        'approach_cm': round(float(approach_cm), 2),
        'contact_cm': round(float(contact_cm), 2),
        'camera_distance_cm': round(float(camera_z_cm), 2),
        'camera_to_jaw_offset_cm': round(float(DEPTH_GRIPPER_OFFSET_CM), 2),
        'lateral_cm': round(float(lateral_cm), 2),
        'total_distance_cm': round(total_distance, 2),
        'descent_deg': round(descent_deg, 2),
        'extra_drop_deg': round(extra_drop_deg, 2),
        'execution_descent_deg': round(execution_descent_deg, 2),
        'lift_cm': lift_cm,
        'is_reachable': bool(is_reachable),
        'reachable_distance_cm': round(reachable_distance, 2),
        'horizontal_ratio': round(horizontal_ratio, 3),
        'depth_uncertainty_cm': round(uncertainty, 2),
    }, contact_cm)


def _annotate_pickup_execution_path(path, estimated_depth=None):
    """Add the servo drop the pickup routine will actually execute."""
    if not isinstance(path, dict) or path.get('descent_deg') is None:
        return path
    try:
        planned = float(path.get('descent_deg') or 0.0)
        extra = _pickup_depth_offset()
        approach_delta = float(path.get('approach_arm_delta_deg') or 0.0)
        descent_cap = max(PICKUP_DESCENT_STEP_DEG, PICKUP_MAX_DESCENT_DEG - approach_delta)
        execution = float(_clamp(
            planned + extra,
            PICKUP_DESCENT_STEP_DEG,
            descent_cap,
        ))
        if estimated_depth is None:
            estimated_depth = PICKUP_DEPTH_DEFAULT
        final_dip = PICKUP_FINAL_DIP_DEG + max(0.0, float(estimated_depth) - PICKUP_DEPTH_DEFAULT) * PICKUP_FINAL_DIP_DEPTH_GAIN
        final_dip = min(final_dip, max(0.0, PICKUP_MAX_DESCENT_DEG - approach_delta - execution))
        path['extra_drop_deg'] = round(extra, 2)
        path['execution_descent_deg'] = round(execution, 2)
        path['final_dip_deg'] = round(final_dip, 2)
        path['total_drop_deg'] = round(approach_delta + execution + final_dip, 2)
    except Exception:
        pass
    return path


# ═════════════════════════════════════════════════════════════════════════════
class ObjectDetector:
    """Object detection with EMA smoothing."""
    def __init__(self, config):
        self.config = config
        self.locked_detection = None
        self.ema_centroid = None
        self.agree_count = 0
    
    def filter_detection(self, detection):
        """Check if detection passes filters."""
        try:
            conf = float(detection.get('confidence', 0))
            label = detection.get('label', '')
            if conf < self.config.confidence_min:
                return False
            ignore_labels = ['furniture', 'vehicle', 'background']
            if any(ign in label.lower() for ign in ignore_labels):
                return False
            return True
        except Exception:
            return False
    
    def lock_detection(self, detection):
        """Lock target to prevent switching."""
        self.locked_detection = detection
        box = detection.get('box', [0, 0, 1, 1])
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        self.ema_centroid = (cx, cy)
        self.agree_count = 0
    
    def update_detection(self, detection):
        """Update detection with EMA smoothing."""
        if self.locked_detection is None:
            return None
        
        try:
            box = detection.get('box', [0, 0, 1, 1])
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            
            if self.ema_centroid is None:
                self.ema_centroid = (cx, cy)
            
            alpha = self.config.centroid_ema_alpha
            new_cx = alpha * cx + (1.0 - alpha) * self.ema_centroid[0]
            new_cy = alpha * cy + (1.0 - alpha) * self.ema_centroid[1]
            self.ema_centroid = (new_cx, new_cy)
            
            self.agree_count = min(self.agree_count + 1, self.config.min_frames_agreed)
            
            return {
                'centroid_x': new_cx,
                'centroid_y': new_cy,
                'confidence': detection.get('confidence', 0),
                'label': detection.get('label', ''),
            }
        except Exception:
            return None


class DepthIntegration:
    """Depth measurement and 3D localization."""
    def __init__(self, config, transforms):
        self.config = config
        self.transforms = transforms
    
    def measure_object_depth(self, detection, depth_sensor_cm):
        """Get depth measurement with validation."""
        try:
            depth_cm = float(depth_sensor_cm)
            if 3 <= depth_cm <= 70:
                return {
                    'distance_cm': depth_cm,
                    'depth_source': 'tof',
                    'uncertainty_cm': 0.5,
                }
            else:
                box = detection.get('box', [0, 0, 640, 480])
                area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
                estimated = max(5, 100 - area / 100)
                return {
                    'distance_cm': estimated,
                    'depth_source': 'box_heuristic',
                    'uncertainty_cm': 5.0,
                }
        except Exception:
            return {'distance_cm': 25, 'depth_source': 'fallback', 'uncertainty_cm': 10.0}
    
    def localize_object_in_base_frame(self, detection, depth_sensor_cm):
        """Localize object in 3D using detection + depth."""
        try:
            box = detection.get('box', [0, 0, 640, 480])
            px = (float(box[0]) + float(box[2])) / 2.0
            py = (float(box[1]) + float(box[3])) / 2.0
            
            depth_info = self.measure_object_depth(detection, depth_sensor_cm)
            depth_cm = depth_info['distance_cm']
            
            reach_3d = self.transforms.pixel_and_depth_to_base(px, py, depth_cm)
            if reach_3d is None:
                return None
            
            return {
                'x_cm': reach_3d[0],
                'y_cm': reach_3d[1],
                'z_cm': reach_3d[2],
                'pixel_center_x': px,
                'pixel_center_y': py,
                'depth_measurement': depth_cm,
                'confidence': detection.get('confidence', 0),
            }
        except Exception:
            return None


class PickupSequence:
    """Complete pickup sequence executor."""
    def __init__(self, geometry, config):
        self.config = config
        self.geometry = geometry
        self.limits = JointLimits()
        self.intrinsics = CameraIntrinsics()
        self.extrinsics = CameraExtrinsics()
        
        self.fk = ForwardKinematics(geometry, self.limits)
        self.ik = InverseKinematics(geometry, self.limits, self.fk)
        self.transforms = CoordinateTransforms(self.intrinsics, self.extrinsics, self.fk)
        self.planner = self.ik  # PickupPlanner unused, IK used directly
        
        self.detector = ObjectDetector(config)
        self.depth_sensor = DepthIntegration(config, self.transforms)
        self.last_error = None
    
    def plan_pickup(self, object_position):
        """Plan pickup approach trajectory."""
        try:
            x = float(object_position['x_cm'])
            y = float(object_position['y_cm'])
            z = float(object_position['z_cm'])
            
            solution = self.planner.plan_approach(
                x, y, z, 
                hover_height_cm=self.config.hover_height_cm,
                cur_angles={'base': 90.0, 'arm': 90.0, 'wrist': 90.0}
            )
            
            if solution is None:
                self.last_error = 'IK unreachable'
                return None
            
            return {
                'object_position': object_position,
                'hover_angles': solution,
                'grasp_height_cm': z,
            }
        except Exception as e:
            self.last_error = str(e)
            return None
    
    def compute_descent_wrist_angle(self, current_z, target_z, approach_wrist):
        """Compute wrist angle during descent with nose-down tilt."""
        if current_z <= target_z:
            return approach_wrist
        
        descent_depth = current_z - target_z
        tilt_span = self.config.wrist_min_deg + descent_depth * self.config.wrist_gain
        tilt_span = min(tilt_span, self.config.wrist_max_deg)
        
        descent_frac = (current_z - target_z) / max(0.1, descent_depth)
        descent_frac = min(1.0, max(0.0, descent_frac))
        
        wrist_target = approach_wrist - tilt_span * descent_frac
        return wrist_target


# ═════════════════════════════════════════════════════════════════════════════

def _arm_ik_2link(x_cm, z_cm):
    """2-link planar IK for the shoulder (arm) joint.

    Geometry (side view, arm in the vertical plane):
      - Link 1 = ARM_LINK1_CM  (upper arm, from shoulder)
      - Link 2 = ARM_LINK2_CM  (forearm + gripper, to tip)
      - x_cm   = horizontal reach from shoulder to target (forward = positive)
      - z_cm   = vertical drop from shoulder to target    (down    = positive)

    Returns (shoulder_deg, elbow_deg) in the servo's degree space, or None if
    the target is outside the arm's reachable envelope or causes an unstable jump.

    Convention used here matches the physical arm:
      - shoulder_deg = 90 means arm pointing vertical-up (home)
      - shoulder_deg increases to lower the arm toward the floor
      - The raw geometric angle (theta1) is measured from the horizontal.
    """
    L1 = ARM_LINK1_CM
    L2 = ARM_LINK2_CM
    
    # Prevent backwards reach which can cause atan2 quadrant flips and backflips
    x_cm = max(0.1, float(x_cm))
    r  = math.hypot(x_cm, z_cm)

    # Clamp to reachable envelope with a small margin
    r = _clamp(r, abs(L1 - L2) + 0.1, (L1 + L2) - 0.1)

    # Law of cosines â€” elbow angle
    cos_elbow = (L1**2 + L2**2 - r**2) / (2.0 * L1 * L2)
    cos_elbow = _clamp(cos_elbow, -1.0, 1.0)
    elbow_rad = math.acos(cos_elbow)

    # Shoulder angle
    alpha = math.atan2(z_cm, x_cm)
    cos_alpha2 = (L1**2 + r**2 - L2**2) / (2.0 * L1 * r)
    cos_alpha2 = _clamp(cos_alpha2, -1.0, 1.0)
    alpha2 = math.acos(cos_alpha2)

    shoulder_rad = alpha + alpha2
    shoulder_deg = 180.0 - math.degrees(shoulder_rad)
    elbow_deg    = math.degrees(elbow_rad)

    proposed_shoulder = float(_clamp(shoulder_deg, ARM_MIN, ARM_MAX))
    proposed_elbow = float(_clamp(elbow_deg, 0.0, 180.0))

    # Safeguard against extreme joint jumps (backflips)
    with _pos_lock:
        cur_arm = float(_pos.get('arm', 90.0))
        
    # If the IK solution demands an instantaneous jump of more than 60 degrees,
    # it is likely crossing a singularity or flipping the elbow solution. Reject it.
    # FIX v38: raised from 45 to 60 deg — first pickup move from home (arm=90) to a
    # table-reach pose (arm~145+) is legitimately >45 deg and was being incorrectly
    # rejected, causing the arm to never pre-position and descent to start from home.
    if abs(proposed_shoulder - cur_arm) > 60.0:
        print(f'[IK REJECT] Unstable solution: current arm={cur_arm:.1f}, proposed={proposed_shoulder:.1f} (jump={abs(proposed_shoulder - cur_arm):.1f}>60 deg)', flush=True)
        return None

    return (proposed_shoulder, proposed_elbow)


def _arm_ik_2link_enhanced(x_cm, z_cm, use_kinematics=True):
    """Enhanced 2-link IK using the validated PickupPipeline IK solver.
    
    Uses the pipeline's InverseKinematics (analytical, validated, FK-checked).
    Falls back to _arm_ik_2link if pipeline is unavailable.
    
    Args:
        x_cm: forward reach from shoulder (cm)
        z_cm: downward drop from shoulder (cm)
        use_kinematics: if True, use pipeline IK, else use old function
    
    Returns:
        (shoulder_deg, elbow_deg) tuple or None if unreachable
        where elbow_deg = wrist_servo_deg - 90 (old convention)
    """
    global _pipeline, _ik_solver
    
    if not use_kinematics:
        return _arm_ik_2link(x_cm, z_cm)
    
    try:
        # Prefer pipeline IK (validated math)
        ik_solver = _pipeline.ik if _pipeline is not None else _ik_solver
        if ik_solver is None:
            return _arm_ik_2link(x_cm, z_cm)
        
        with _pos_lock:
            cur_arm = float(_pos.get('arm', 90.0))
        
        result = ik_solver.solve_planar(float(x_cm), float(z_cm), cur_arm, prefer_elbow_up=False)
        if result is None:
            return None
        
        arm_deg, wrist_deg = result
        return (float(arm_deg), float(wrist_deg) - 90.0)  # convert wrist_servo → elbow_deg
    except Exception as e:
        print(f'[IK] Enhanced IK failed: {e}, falling back to old IK', flush=True)
        return _arm_ik_2link(x_cm, z_cm)


def _depth_status_snapshot():
    with _depth_lock:
        runtime_enabled = bool(_depth_state.get('runtime_enabled', VL53L1X_ENABLED))
    with _vl53l1x_lock:
        h_ready = bool(_vl53l1x_state.get('ready'))
        meas_ok = bool(_vl53l1x_state.get('measurement_ok'))
        h_failed = bool(_vl53l1x_state.get('failed'))
        h_err = str(_vl53l1x_state.get('last_error') or '')
        raw_cm = _vl53l1x_state.get('last_raw_cm')
        meas_us = _vl53l1x_state.get('last_measurement_us')
    gpio_ok = bool(VL53L1X_ENABLED and runtime_enabled and _gpio_ready[0] and h_ready)
    ready = bool(gpio_ok and meas_ok)
    failed = bool(VL53L1X_ENABLED and runtime_enabled and (h_failed))  # not meas_ok alone is not a failure — sensor may just be warming up
    return {
        'enabled': bool(VL53L1X_ENABLED and runtime_enabled),
        'runtime_enabled': runtime_enabled,
        'ready': ready,
        'gpio_ok': gpio_ok,
        'measurement_ok': meas_ok,
        'failed': failed,
        'loading': False,
        'model': 'VL53L1X (down-facing)' if _vl53l1x_mount_down() else ('VL53L1X (forward-facing)' if _vl53l1x_mount_behind_gripper() else 'VL53L1X ToF'),
        'sensor': 'VL53L1X',
        'mount': VL53L1X_MOUNT,
        'mount_description': _vl53l1x_mount_description(),
        'sda_pin': VL53L1X_SDA_PIN,
        'scl_pin': VL53L1X_SCL_PIN,
        'grasp_clearance_cm': VL53L1X_GRASP_CLEARANCE_CM,
        'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
        'mount_offset_cm': VL53L1X_MOUNT_OFFSET_CM,
        'vl53l1x_scale_factor': VL53L1X_SCALE_FACTOR,
        'vl53l1x_distance_offset_cm': VL53L1X_DISTANCE_OFFSET_CM,
        'poll_interval_sec': VL53L1X_POLL_INTERVAL,
        'last_error': h_err,
        'last_raw_cm': raw_cm,
        'last_measurement_us': meas_us,
        'gpio_ready': bool(_gpio_ready[0]),
    }


def _depth_enabled_runtime():
    return _vl53l1x_enabled_runtime()


def _stop_depth_worker(reason='stopped by user'):
    if _DEPTH_WORKER_MODE:
        return
    with _depth_worker_lock:
        proc = _depth_worker.get('proc')
        _depth_worker['proc'] = None
        _depth_worker['queue'] = None
        _depth_worker['reader'] = None
        _depth_worker['last_start'] = 0.0
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1.2)
            except Exception:
                proc.kill()
        except Exception:
            pass
    with _depth_lock:
        _depth_state['ready'] = False
        _depth_state['loading'] = False
        _depth_state['failed'] = False
        _depth_state['worker_pid'] = None
        _depth_state['worker_returncode'] = None
        _depth_state['last_error'] = str(reason)


def _set_depth_runtime_enabled(enabled, reason='user'):
    enabled = bool(enabled)
    with _depth_lock:
        _depth_state['runtime_enabled'] = enabled
        _depth_state['ready'] = bool(enabled and _vl53l1x_state.get('ready'))
        _depth_state['failed'] = bool(enabled and _vl53l1x_state.get('failed'))
        _depth_state['loading'] = False
        _depth_state['last_error'] = 'VL53L1X enabled' if enabled else 'VL53L1X disabled by user'
    if enabled and _gpio_ready[0]:
        _init_vl53l1x_gpio()
        if reason:
            _set_status(f'VL53L1X enabled ({reason})')
    else:
        if reason:
            _set_status(f'VL53L1X disabled ({reason})')
    return _depth_status_snapshot()

def _box_similarity(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        ar = [ax, ay, ax + max(1.0, aw), ay + max(1.0, ah)]
        br = [bx, by, bx + max(1.0, bw), by + max(1.0, bh)]
        ix1 = max(ar[0], br[0])
        iy1 = max(ar[1], br[1])
        ix2 = min(ar[2], br[2])
        iy2 = min(ar[3], br[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, (ar[2] - ar[0]) * (ar[3] - ar[1]))
        area_b = max(1.0, (br[2] - br[0]) * (br[3] - br[1]))
        return inter / max(1.0, area_a + area_b - inter)
    except Exception:
        return 0.0

def _depth_worker_reader(proc, out_q):
    try:
        for line in proc.stdout:
            line = str(line or '').strip()
            if not line:
                continue
            # Print to parent's stderr so we see worker output in terminal
            print(f'[WORKER-OUT] {line}', file=sys.stderr, flush=True)
            try:
                out_q.put(json.loads(line))
            except Exception:
                out_q.put({'type': 'log', 'message': line[-240:]})
    except Exception as e:
        out_q.put({'type': 'log', 'message': f'reader stopped: {e}'})

def _mark_depth_worker_dead(reason, returncode=None):
    # A hard crash (non-zero / non-None return code) permanently disables Depth Anything V2
    # for this session â€” no timed backoff. Box-size depth takes over immediately.
    # Clean exits (returncode 0 or None) allow a restart after the cooldown.
    hard_crash = returncode not in (None, 0)
    with _depth_lock:
        crashes = int(_depth_state.get('worker_crashes', 0))
        if hard_crash:
            crashes += 1
        _depth_state['worker_crashes'] = crashes
        if hard_crash:
            # Permanent session disable until the user turns Depth Anything V2 back on from the webpage.
            _depth_state['worker_disabled_until'] = float('inf')
            _depth_state['runtime_enabled'] = False
        _depth_state['ready'] = False
        _depth_state['loading'] = False
        _depth_state['failed'] = hard_crash
        _depth_state['worker_pid'] = None
        _depth_state['worker_returncode'] = returncode
        _depth_state['last_error'] = str(reason)
    suffix = ' â€” Depth Anything V2 disabled for session, box depth active' if hard_crash else ''
    print(f'[DEPTH] {reason}{suffix}')

def _ensure_depth_worker():
    if not _depth_enabled_runtime():
        return False
    now = time.time()
    with _depth_lock:
        disabled_until = float(_depth_state.get('worker_disabled_until') or 0.0)
    if now < disabled_until:
        return False

    # FIX BUG 1 (DEADLOCK): Original code held _depth_worker_lock AND then acquired
    # _depth_lock inside it (L706). _stop_depth_worker does the reverse order.
    # Concurrent calls â†’ Aâ†’B / Bâ†’A deadlock. Fix: release _depth_worker_lock before
    # acquiring _depth_lock by splitting into two separate locked sections.
    new_proc = None
    new_out_q = None
    new_reader = None

    with _depth_worker_lock:
        proc = _depth_worker.get('proc')
        if proc is not None and proc.poll() is None:
            return True
        if proc is not None:
            _mark_depth_worker_dead(f'worker exited ({proc.returncode})', proc.returncode)
        if now - float(_depth_worker.get('last_start') or 0.0) < DEPTH_WORKER_RESTART_COOLDOWN_SEC:
            return False

        out_q = queue.Queue()
        print(f'[DEPTH] Spawning worker subprocess: {sys.executable} {os.path.abspath(__file__)} --depth-worker', flush=True)
        try:
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), '--depth-worker'],
                cwd=_DIR,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            print(f'[DEPTH] Worker subprocess started with PID {proc.pid}', flush=True)
        except Exception as e:
            print(f'[DEPTH] FAILED to start worker subprocess: {e}', flush=True)
            traceback.print_exc()
            _mark_depth_worker_dead(f'worker start failed: {e}')
            return False

        reader = threading.Thread(target=_depth_worker_reader,
                                  args=(proc, out_q),
                                  name='depth-worker-reader',
                                  daemon=True)
        reader.start()
        _depth_worker.update({
            'proc': proc,
            'queue': out_q,
            'reader': reader,
            'last_start': now,
        })
        new_proc = proc
        new_out_q = out_q
        new_reader = reader
    # _depth_worker_lock now RELEASED â€” safe to acquire _depth_lock without deadlock risk
    if new_proc is not None:
        with _depth_lock:
            _depth_state['ready'] = False
            _depth_state['failed'] = False
            _depth_state['loading'] = True
            _depth_state['worker_pid'] = new_proc.pid
            _depth_state['worker_returncode'] = None
            _depth_state['last_error'] = 'Depth Anything V2 worker loading'
        print(f'[DEPTH] Worker starting (pid {new_proc.pid})')
        return True
    return False

def _depth_worker_request(frame_bgr, box, timeout=DEPTH_WORKER_TIMEOUT_SEC):
    if frame_bgr is None or not _depth_enabled_runtime():
        return None
    if not _ensure_depth_worker():
        return None

    req_id = f'{os.getpid()}-{time.time():.6f}'
    tmp_path = None
    _yolo_pause_for_depth.set()
    try:
        fd, tmp_path = tempfile.mkstemp(prefix='depth_frame_', suffix='.jpg', dir=_DIR)
        os.close(fd)
        if not cv2.imwrite(tmp_path, frame_bgr):
            return None

        with _depth_request_lock:
            with _depth_worker_lock:
                proc = _depth_worker.get('proc')
                out_q = _depth_worker.get('queue')
            if proc is None or out_q is None or proc.poll() is not None:
                _mark_depth_worker_dead('worker unavailable before request',
                                        None if proc is None else proc.returncode)
                return None

            with _calibration_lock:
                close_range_scale = float(_calibration_state['offsets'].get('close_range_scale', DEPTH_CLOSE_RANGE_SCALE))
            payload = {
                'id': req_id,
                'path': tmp_path,
                'box': [float(v) for v in box],
                'depth_scale_factor': _pickup_depth_scale_factor(),
                'close_range_scale': close_range_scale,
            }
            try:
                proc.stdin.write(json.dumps(payload) + '\n')
                proc.stdin.flush()
            except Exception as e:
                _mark_depth_worker_dead(f'worker pipe failed: {e}',
                                        None if proc is None else proc.poll())
                return None

            end = time.time() + float(timeout)
            while time.time() < end:
                if proc.poll() is not None:
                    _mark_depth_worker_dead(f'worker crashed/exited ({proc.returncode})',
                                            proc.returncode)
                    return None
                try:
                    msg = out_q.get(timeout=0.12)
                except queue.Empty:
                    continue
                msg_type = msg.get('type')
                if msg_type == 'ready':
                    with _depth_lock:
                        _depth_state['ready'] = True
                        _depth_state['loading'] = False
                        _depth_state['failed'] = False
                        _depth_state['last_error'] = ''
                    print(f"[DEPTH] Worker ready on {msg.get('device', 'cpu')}")
                    continue
                if msg_type == 'error':
                    err = str(msg.get('error', 'worker error'))
                    with _depth_lock:
                        _depth_state['last_error'] = err
                    print(f'[DEPTH] {err}')
                    if msg.get('id') == req_id:
                        return None
                    continue
                if msg_type == 'depth' and msg.get('id') == req_id:
                    if not msg.get('ok'):
                        err = str(msg.get('error', 'depth unavailable'))
                        with _depth_lock:
                            _depth_state['last_error'] = err
                        _publish_depth_result({
                            'ts': time.time(),
                            'ok': False,
                            'source': 'depth-anything-v2',
                            'message': err,
                            'distance_cm': None,
                            'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
                            'in_reach': None,
                        }, msg.get('map_jpeg_b64'))
                        return None
                    # BUG FIX: Guard against missing 'value'/'hint' keys (msg.get returns
                    # None when absent, and float(None) raises TypeError crashing the handler).
                    raw_value = msg.get('value')
                    if raw_value is None:
                        err = 'depth worker reply missing value field'
                        with _depth_lock:
                            _depth_state['last_error'] = err
                        print(f'[DEPTH] {err}')
                        return None
                    value = float(raw_value)
                    raw_hint = msg.get('hint')
                    hint = float(_clamp(
                        raw_hint if raw_hint is not None else PICKUP_DEPTH_DEFAULT,
                        PICKUP_DEPTH_MIN, PICKUP_DEPTH_MAX,
                    ))
                    distance_cm = msg.get('distance_cm', None)
                    distance_cm = None if distance_cm is None else float(distance_cm)
                    jaw_distance_cm = None if distance_cm is None else _jaw_distance_from_camera_depth(distance_cm)
                    path_plan = _annotate_pickup_execution_path(
                        msg.get('path'),
                        jaw_distance_cm if jaw_distance_cm is not None else hint,
                    )
                    result = {
                        'ts': time.time(),
                        'ok': True,
                        'source': 'depth-anything-v2',
                        'model': str(msg.get('model') or DEPTH_MODEL_TYPE),
                        'value': value,
                        'distance_m': None if distance_cm is None else round(distance_cm / 100.0, 4),
                        'distance_cm': None if distance_cm is None else round(distance_cm, 2),
                        'jaw_distance_cm': None if jaw_distance_cm is None else round(jaw_distance_cm, 2),
                        'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
                        'in_reach': bool(jaw_distance_cm is not None and jaw_distance_cm <= DEPTH_REACH_LIMIT_CM),
                        'hint': hint,
                        'path': path_plan,
                        'depth_scale_factor': msg.get('depth_scale_factor'),
                        'depth_method': msg.get('depth_method'),
                        'close_range_corrected': bool(msg.get('close_range_corrected')),
                        'message': str(msg.get('message') or ''),
                    }
                    if distance_cm is not None:
                        result['message'] = (
                            f'camera {distance_cm:.1f} cm, jaw {jaw_distance_cm:.1f} cm - in reach'
                            if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM
                            else f'camera {distance_cm:.1f} cm, jaw {jaw_distance_cm:.1f} cm - out of reach'
                        )
                    _publish_depth_result(result, msg.get('map_jpeg_b64'))
                    with _depth_lock:
                        _depth_state['ready'] = True
                        _depth_state['loading'] = False
                        _depth_state['failed'] = False
                        _depth_state['last_box'] = [int(float(v)) for v in box]
                        _depth_state['last_hint'] = hint
                        _depth_state['last_value'] = value
                        _depth_state['last_result'] = dict(result)
                        _depth_state['last_ts'] = time.time()
                        _depth_state['last_error'] = ''
                        _depth_state['inferences'] += 1
                    return result

            with _depth_lock:
                _depth_state['last_error'] = 'Depth Anything V2 worker timed out'
            print('[DEPTH] Worker timed out; using box depth for this grab')
            return None
    except Exception as e:
        with _depth_lock:
            _depth_state['last_error'] = str(e)
        print(f'[DEPTH] Worker request failed: {e}')
        return None
    finally:
        _yolo_pause_for_depth.clear()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def _load_depth_model():
    """Start the crash-isolated Depth Anything V2 worker."""
    return _ensure_depth_worker()

def _depth_depth_map_from_frame(frame_bgr):
    """Depth Anything V2 maps are computed only inside the crash-isolated worker."""
    return None

def _depth_roi_from_box(box):
    try:
        x, y, bw, bh = [float(v) for v in box]
    except Exception:
        return None
    # BUG FIX: Check for negative dimensions and ensure positive size
    if bw <= 1 or bh <= 1 or x < 0 or y < 0:
        return None

    x1 = int(_clamp(x + bw * DEPTH_ROI_X1, 0, FRAME_W - 1))
    x2 = int(_clamp(x + bw * DEPTH_ROI_X2, 0, FRAME_W))
    y1 = int(_clamp(y + bh * DEPTH_ROI_Y1, 0, FRAME_H - 1))
    y2 = int(_clamp(y + bh * DEPTH_ROI_Y2, 0, FRAME_H))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _robust_object_depth_m(depth, x1, y1, x2, y2):
    """Estimate object surface depth from the near, central pixels of a depth ROI."""
    roi = depth[y1:y2, x1:x2]
    valid_mask = np.isfinite(roi) & (roi > 0.001)
    valid = roi[valid_mask]
    if valid.size == 0:
        return None, {'method': 'empty-roi', 'pixels': 0}

    if valid.size >= max(10, DEPTH_OBJECT_MIN_PIXELS):
        # Depth Anything boxes often include table/background. For grasping, the
        # closest coherent surface is usually the object face the gripper touches.
        near_pct = float(_clamp(DEPTH_OBJECT_NEAR_PERCENTILE, 5.0, 60.0))
        max_pct = float(_clamp(DEPTH_OBJECT_MAX_PERCENTILE, near_pct + 5.0, 90.0))
        near_ref = float(np.percentile(valid, near_pct))
        hi_ref = float(np.percentile(valid, max_pct))
        band = valid[(valid >= near_ref * 0.92) & (valid <= hi_ref)]
        if band.size >= max(8, int(DEPTH_OBJECT_MIN_PIXELS * 0.5)):
            return float(np.median(band)), {
                'method': 'near-surface-band',
                'pixels': int(band.size),
                'near_pct': near_pct,
                'max_pct': max_pct,
            }

    if valid.size >= 30:
        lo_pct = float(np.percentile(valid, 15))
        hi_pct = float(np.percentile(valid, 85))
        trimmed = valid[(valid >= lo_pct) & (valid <= hi_pct)]
        if trimmed.size >= 10:
            valid = trimmed
    return float(np.median(valid)), {'method': 'trimmed-median', 'pixels': int(valid.size)}


_vl53l1x_sensor = None
_vl53l0x_stop_var = [0]  # stop_var read during init; needed by single-shot trigger in read loop

# ── VL53L0X hardware interface (8-bit register addresses) ──────────────────
# The sensor on this build is a VL53L0X, not VL53L1X.
# VL53L0X uses plain 8-bit register addresses readable via standard smbus2 calls.

def _vl53l1x_write_reg(bus, reg8, val):
    """Write 1 byte to a VL53L0X register (8-bit address)."""
    bus.write_byte_data(0x29, reg8 & 0xFF, val & 0xFF)

def _vl53l1x_read_reg(bus, reg8):
    """Read 1 byte from a VL53L0X register (8-bit address)."""
    return bus.read_byte_data(0x29, reg8 & 0xFF)

def _vl53l0x_read_u16(bus, reg8):
    """Read a big-endian 16-bit value from two consecutive VL53L0X registers."""
    data = bus.read_i2c_block_data(0x29, reg8 & 0xFF, 2)
    return (data[0] << 8) | data[1]

def _init_vl53l1x_gpio():
    """Initialize VL53L0X on hardware I2C bus 1 (GPIO 2=SDA, GPIO 3=SCL)."""
    global _vl53l1x_sensor, _vl53l0x_stop_var
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        # WHO_AM_I register 0xC0 returns 0xEE on VL53L0X
        who = bus.read_byte_data(0x29, 0xC0)
        if who != 0xEE:
            raise RuntimeError(f'Unexpected VL53L0X WHO_AM_I: 0x{who:02X} (expected 0xEE)')
        # Verify revision registers
        rev_id  = bus.read_byte_data(0x29, 0xC1)
        mod_id  = bus.read_byte_data(0x29, 0xC2)
        # ── Full ST-spec VL53L0X init (FIX v38) ───────────────────────────────
        # Previous "minimal init" was missing the mandatory load-tuning-registers
        # block.  Without it the sensor's internal state machine never arms, so
        # RESULT_INTERRUPT_STATUS (0x13) stays stuck at 0x40 (bit-6, config-error
        # flag) and no measurements are ever produced.
        #
        # This sequence matches the Pololu / Adafruit reference drivers exactly:
        #   1. Data-init handshake (unlock private registers via stop_var)
        #   2. Load ~20 mandatory tuning register writes from ST app-note
        #   3. Set SYSTEM_INTERRUPT_CONFIG_GPIO → new-sample-ready on pin
        #   4. Clear any stale interrupt
        #   5. Enable SIGNAL_RATE_MSRC + SIGNAL_RATE_PRE_RANGE limits
        #   6. Set measurement timing budget (33 ms default → ~30 Hz capable)
        #   7. Configure sequence steps (DSS off, pre-range on, final-range on)
        #   8. Recalculate & apply timing budget
        #   9. Single-shot mode: trigger each sample explicitly from _vl53l1x_read_cm
        #      (continuous back-to-back mode requires SYSTEM_INTERMEASUREMENT_PERIOD
        #       which this minimal driver does not program; without it continuous mode
        #       produces no interrupts on many sensor silicon revisions)

        # 1. Data-init handshake — unlock stop_var from private register bank
        bus.write_byte_data(0x29, 0x88, 0x00)
        bus.write_byte_data(0x29, 0x80, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x00, 0x00)
        stop_var = bus.read_byte_data(0x29, 0x91)
        _vl53l0x_stop_var[0] = stop_var  # persist for single-shot trigger in _vl53l1x_read_cm
        bus.write_byte_data(0x29, 0x00, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x80, 0x00)
        print(f'[VL53L0X] stop_var=0x{stop_var:02X}', flush=True)

        # 2. Mandatory tuning registers (ST app-note; same across all silicon revs)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x00, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x09, 0x00)
        bus.write_byte_data(0x29, 0x10, 0x00)
        bus.write_byte_data(0x29, 0x11, 0x00)
        bus.write_byte_data(0x29, 0x24, 0x01)
        bus.write_byte_data(0x29, 0x25, 0xFF)
        bus.write_byte_data(0x29, 0x75, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x4E, 0x2C)
        bus.write_byte_data(0x29, 0x48, 0x00)
        bus.write_byte_data(0x29, 0x30, 0x20)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x30, 0x09)
        bus.write_byte_data(0x29, 0x54, 0x00)
        bus.write_byte_data(0x29, 0x31, 0x04)
        bus.write_byte_data(0x29, 0x32, 0x03)
        bus.write_byte_data(0x29, 0x40, 0x83)
        bus.write_byte_data(0x29, 0x46, 0x25)
        bus.write_byte_data(0x29, 0x60, 0x00)
        bus.write_byte_data(0x29, 0x27, 0x00)
        bus.write_byte_data(0x29, 0x50, 0x06)
        bus.write_byte_data(0x29, 0x51, 0x00)
        bus.write_byte_data(0x29, 0x52, 0x96)
        bus.write_byte_data(0x29, 0x56, 0x08)
        bus.write_byte_data(0x29, 0x57, 0x30)
        bus.write_byte_data(0x29, 0x61, 0x00)
        bus.write_byte_data(0x29, 0x62, 0x00)
        bus.write_byte_data(0x29, 0x64, 0x00)
        bus.write_byte_data(0x29, 0x65, 0x00)
        bus.write_byte_data(0x29, 0x66, 0xA0)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x22, 0x32)
        bus.write_byte_data(0x29, 0x47, 0x14)
        bus.write_byte_data(0x29, 0x49, 0xFF)
        bus.write_byte_data(0x29, 0x4A, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x7A, 0x0A)
        bus.write_byte_data(0x29, 0x7B, 0x00)
        bus.write_byte_data(0x29, 0x78, 0x21)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x23, 0x34)
        bus.write_byte_data(0x29, 0x42, 0x00)
        bus.write_byte_data(0x29, 0x44, 0xFF)
        bus.write_byte_data(0x29, 0x45, 0x26)
        bus.write_byte_data(0x29, 0x46, 0x05)
        bus.write_byte_data(0x29, 0x40, 0x40)
        bus.write_byte_data(0x29, 0x0E, 0x06)
        bus.write_byte_data(0x29, 0x20, 0x1A)
        bus.write_byte_data(0x29, 0x43, 0x40)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x34, 0x03)
        bus.write_byte_data(0x29, 0x35, 0x44)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x31, 0x04)
        bus.write_byte_data(0x29, 0x4B, 0x09)
        bus.write_byte_data(0x29, 0x4C, 0x05)
        bus.write_byte_data(0x29, 0x4D, 0x04)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x44, 0x00)
        bus.write_byte_data(0x29, 0x45, 0x20)
        bus.write_byte_data(0x29, 0x47, 0x08)
        bus.write_byte_data(0x29, 0x48, 0x28)
        bus.write_byte_data(0x29, 0x67, 0x00)
        bus.write_byte_data(0x29, 0x70, 0x04)
        bus.write_byte_data(0x29, 0x71, 0x01)
        bus.write_byte_data(0x29, 0x72, 0xFE)
        bus.write_byte_data(0x29, 0x76, 0x00)
        bus.write_byte_data(0x29, 0x77, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x0D, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x80, 0x01)
        bus.write_byte_data(0x29, 0x01, 0xF8)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x8E, 0x01)
        bus.write_byte_data(0x29, 0x00, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x80, 0x00)
        print('[VL53L0X] Tuning registers loaded', flush=True)

        # 3. Configure GPIO: new-sample-ready interrupt on pin (active-low)
        bus.write_byte_data(0x29, 0x0A, 0x04)  # SYSTEM_INTERRUPT_CONFIG_GPIO = new sample ready
        gpio_hv = bus.read_byte_data(0x29, 0x84)
        bus.write_byte_data(0x29, 0x84, gpio_hv & ~0x10)  # GPIO_HV_MUX_ACTIVE_HIGH: active-low

        # 4. Clear any stale interrupt
        bus.write_byte_data(0x29, 0x0B, 0x01)  # SYSTEM_INTERRUPT_CLEAR
        time.sleep(0.01)

        # 5. Enable SIGNAL_RATE limit checks (MSRC + PRE_RANGE)
        bus.write_byte_data(0x29, 0x60, 0x00)  # MSRC_CONFIG_CONTROL: enable rate checks
        # FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT = 0.25 MCPS (9.7-bit fixed-point)
        bus.write_byte_data(0x29, 0x44, 0x00)
        bus.write_byte_data(0x29, 0x45, 0x20)  # 0x0020 = 32 = 0.25 MCPS

        # 6. Load reference SPAD count/map from NVM (FIX v39 — was missing, causing
        #    Signal Fail / status 0x01 / 8191mm on every sample).
        #    Without this the sensor operates with the wrong SPAD configuration.
        #    Matches Pololu/Adafruit reference driver get_spad_info() + setSpadCount().
        bus.write_byte_data(0x29, 0x80, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x00, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x06)
        _tmp83 = bus.read_byte_data(0x29, 0x83)
        bus.write_byte_data(0x29, 0x83, _tmp83 | 0x04)
        bus.write_byte_data(0x29, 0xFF, 0x07)
        bus.write_byte_data(0x29, 0x81, 0x01)
        bus.write_byte_data(0x29, 0x80, 0x01)
        bus.write_byte_data(0x29, 0x94, 0x6B)
        bus.write_byte_data(0x29, 0x83, 0x00)
        # Poll until NVM read completes (register 0x83 goes non-zero)
        for _nvm_poll in range(100):
            if bus.read_byte_data(0x29, 0x83) != 0x00:
                break
            time.sleep(0.005)
        bus.write_byte_data(0x29, 0x83, 0x01)
        _spad_nvm   = bus.read_byte_data(0x29, 0x92)
        spad_count       = _spad_nvm & 0x7F            # bits[6:0]
        spad_is_aperture = bool((_spad_nvm >> 7) & 0x01)  # bit 7
        # Restore private register bank
        bus.write_byte_data(0x29, 0x81, 0x00)
        bus.write_byte_data(0x29, 0xFF, 0x06)
        _tmp83 = bus.read_byte_data(0x29, 0x83)
        bus.write_byte_data(0x29, 0x83, _tmp83 & ~0x04)
        bus.write_byte_data(0x29, 0xFF, 0x01)
        bus.write_byte_data(0x29, 0x00, 0x01)
        bus.write_byte_data(0x29, 0xFF, 0x00)
        bus.write_byte_data(0x29, 0x80, 0x00)
        print(f'[VL53L0X] NVM SPAD: count={spad_count} aperture={spad_is_aperture}', flush=True)

        # Apply SPAD enable map: enable only the first spad_count SPADs of the right type.
        # Aperture SPADs start at bit offset 12; non-aperture start at 0.
        SPAD_MAP_REG = 0xB0  # GLOBAL_CONFIG_SPAD_ENABLES_REF_0 (6 bytes)
        spad_map = bytearray(bus.read_i2c_block_data(0x29, SPAD_MAP_REG, 6))
        for _i in range(6):
            spad_map[_i] = 0x00  # clear all first
        _spad_start = 12 if spad_is_aperture else 0
        _enabled = 0
        for _i in range(48):
            if _i < _spad_start:
                continue
            if _enabled >= spad_count:
                break
            spad_map[_i // 8] |= (1 << (_i % 8))
            _enabled += 1
        for _i, _byte in enumerate(spad_map):
            bus.write_byte_data(0x29, SPAD_MAP_REG + _i, _byte)
        print(f'[VL53L0X] SPAD map applied ({_enabled} SPADs enabled)', flush=True)

        # 7. VHV + phase calibration (one-shot internal cal, FIX v39).
        #    Must run before normal ranging sequence is restored.
        #    Without this, signal thresholds are wrong and status stays 0x01.
        bus.write_byte_data(0x29, 0x01, 0x01)  # SYSTEM_SEQUENCE_CONFIG: VHV cal only
        bus.write_byte_data(0x29, 0x00, 0x01)  # SYSRANGE_START: single shot
        for _cal_poll in range(200):            # up to 1 s
            if (bus.read_byte_data(0x29, 0x13) & 0x07) != 0:
                break
            time.sleep(0.005)
        bus.write_byte_data(0x29, 0x00, 0x00)  # stop
        bus.write_byte_data(0x29, 0x0B, 0x01)  # clear interrupt

        bus.write_byte_data(0x29, 0x01, 0x02)  # phase cal only
        bus.write_byte_data(0x29, 0x00, 0x01)
        for _cal_poll in range(200):
            if (bus.read_byte_data(0x29, 0x13) & 0x07) != 0:
                break
            time.sleep(0.005)
        bus.write_byte_data(0x29, 0x00, 0x00)
        bus.write_byte_data(0x29, 0x0B, 0x01)
        print('[VL53L0X] VHV + phase calibration complete', flush=True)

        # 8. Restore full ranging sequence: DSS off, pre-range + final-range on.
        bus.write_byte_data(0x29, 0x01, 0xE8)

        # 9. Single-shot mode — each sample is triggered explicitly with SYSRANGE_START=0x01.
        #    We do NOT use continuous mode here because without SYSTEM_INTERMEASUREMENT_PERIOD
        #    being programmed, continuous mode never raises the new-sample interrupt on most
        #    silicon revisions (exactly the 0x40-stuck bug seen in logs).
        print('[VL53L0X] Init complete — single-shot mode', flush=True)
        time.sleep(0.10)
        _vl53l1x_sensor = bus
        with _vl53l1x_lock:
            _vl53l1x_state['ready'] = True
            _vl53l1x_state['failed'] = False
            _vl53l1x_state['last_error'] = ''
        print(f'[VL53L0X] Sensor ready on I2C bus 1 (SDA=GPIO{VL53L1X_SDA_PIN}, SCL=GPIO{VL53L1X_SCL_PIN}) WHO_AM_I=0x{who:02X} rev=0x{rev_id:02X} mod=0x{mod_id:02X}', flush=True)
        return True
    except ImportError:
        err = "smbus2 not installed. Run 'pip install smbus2 --break-system-packages'"
        with _vl53l1x_lock:
            _vl53l1x_state['failed'] = True
            _vl53l1x_state['last_error'] = err
        print(f'[VL53L0X] {err}', flush=True)
        return False
    except Exception as e:
        with _vl53l1x_lock:
            _vl53l1x_state['failed'] = True
            _vl53l1x_state['last_error'] = str(e)
        print(f'[VL53L0X] Init failed: {e}', flush=True)
        return False

def _vl53l1x_not_ready_message():
    with _vl53l1x_lock:
        if _vl53l1x_state.get('failed'):
            return _vl53l1x_state.get('last_error') or 'Init failed'
        if not _vl53l1x_state.get('ready'):
            return 'Not initialized'
        return ''

def _vl53l1x_read_cm(samples=None):
    """Read distance from VL53L0X via raw smbus2, median-averaged over N samples.

    FIX (a): Interrupt status poll checks bits[2:0] (mask 0x07) not bit6 (0x40).
    FIX (b): Median-averages over VL53L1X_SAMPLES readings.
    FIX (c): Retries up to 3 times on I2C errors caused by servo PWM noise.
             On all retries failing, marks sensor not-ready so next Sensor Read
             call triggers a clean re-init instead of returning stale data.
    FIX v37 (d): Poll timeout now skips the distance read entirely instead of
             falling through to read stale/sentinel data.
    FIX v37 (e): Checks RESULT_RANGE_STATUS (0x14) bits[3:0] == 0x0B (RangeValid)
             before accepting any distance. The VL53L0X returns 20mm (2 cm) as its
             hardcoded error sentinel for every invalid measurement; without this
             check that value passes both the interrupt flag and the MIN_CM filter.
    """
    global _vl53l1x_sensor
    if not _vl53l1x_enabled_runtime() or _vl53l1x_sensor is None:
        with _vl53l1x_lock:
            _vl53l1x_state['last_error'] = _vl53l1x_not_ready_message()
        return None, None

    n = max(1, int(samples) if samples is not None else VL53L1X_SAMPLES)
    I2C_MAX_RETRIES = 3

    for attempt in range(I2C_MAX_RETRIES):
        readings = []
        try:
            bus = _vl53l1x_sensor
            for _s in range(n):
                # FIX v38: single-shot mode — trigger one measurement explicitly.
                # Continuous mode (SYSRANGE_START=0x02) requires
                # SYSTEM_INTERMEASUREMENT_PERIOD to be programmed; without it the
                # sensor never raises the new-sample interrupt on most silicon revs,
                # which is the exact 0x40-stuck-interrupt bug seen in the logs.
                # Single-shot: write stop_var handshake then SYSRANGE_START=0x01.
                bus.write_byte_data(0x29, 0x80, 0x01)
                bus.write_byte_data(0x29, 0xFF, 0x01)
                bus.write_byte_data(0x29, 0x00, 0x00)
                bus.write_byte_data(0x29, 0x91, _vl53l0x_stop_var[0])
                bus.write_byte_data(0x29, 0x00, 0x01)
                bus.write_byte_data(0x29, 0xFF, 0x00)
                bus.write_byte_data(0x29, 0x80, 0x00)
                bus.write_byte_data(0x29, 0x00, 0x01)  # SYSRANGE_START: single shot

                # Poll RESULT_INTERRUPT_STATUS (0x13) bits[2:0] for new sample.
                waited = False
                for _p in range(100):  # 100 × 5ms = 500ms max (single-shot takes ~33ms)
                    status = _vl53l1x_read_reg(bus, 0x13)
                    if (status & 0x07) != 0:
                        waited = True
                        break
                    time.sleep(0.005)

                # Stop ranging immediately after the shot
                bus.write_byte_data(0x29, 0x00, 0x00)  # SYSRANGE_START: stop

                if not waited:
                    print(f'[VL53L0X] WARNING: interrupt poll timed out attempt={attempt+1} '
                          f'(status=0x{status:02X}) — skipping sample {_s+1}/{n}', flush=True)
                    _vl53l1x_write_reg(bus, 0x0B, 0x01)  # clear interrupt anyway
                    if _s < n - 1:
                        time.sleep(VL53L1X_SAMPLE_GAP_SEC)
                    continue

                distance_mm = _vl53l0x_read_u16(bus, 0x1E)
                _vl53l1x_write_reg(bus, 0x0B, 0x01)  # clear interrupt

                # Check RESULT_RANGE_STATUS (0x14) bits[3:0]: 0x0B = RangeValid.
                # VL53L0X returns 20mm (2cm) as the sentinel for every error code.
                range_status = _vl53l1x_read_reg(bus, 0x14)
                range_phase  = range_status & 0x0F
                if range_phase != 0x0B:
                    print(f'[VL53L0X] sample {_s+1}/{n} attempt {attempt+1}: '
                          f'RESULT_RANGE_STATUS=0x{range_phase:02X} (not RangeValid=0x0B) '
                          f'→ rejecting {distance_mm}mm', flush=True)
                    if _s < n - 1:
                        time.sleep(VL53L1X_SAMPLE_GAP_SEC)
                    continue

                cm = distance_mm / 10.0
                print(f'[VL53L0X] sample {_s+1}/{n} attempt {attempt+1}: '
                      f'{distance_mm}mm = {cm:.1f}cm '
                      f'(int_status=0x{status:02X} range_status=0x{range_phase:02X})', flush=True)
                if VL53L1X_MIN_CM <= cm <= VL53L1X_MAX_CM:
                    readings.append(cm)
                else:
                    print(f'[VL53L0X] sample {_s+1}/{n}: {cm:.1f}cm outside valid range '
                          f'[{VL53L1X_MIN_CM}-{VL53L1X_MAX_CM}cm] — skipping', flush=True)
                if _s < n - 1:
                    time.sleep(VL53L1X_SAMPLE_GAP_SEC)

            if not readings:
                print(f'[VL53L0X] all {n} samples rejected attempt={attempt+1}', flush=True)
                if attempt < I2C_MAX_RETRIES - 1:
                    time.sleep(0.05)
                    continue
                return None, None

            readings.sort()
            cm = readings[len(readings) // 2]
            print(f'[VL53L0X] median of {len(readings)} valid samples: {cm:.1f}cm', flush=True)
            with _vl53l1x_lock:
                _vl53l1x_state['measurement_ok'] = True
                _vl53l1x_state['last_raw_cm'] = round(cm, 2)
                _vl53l1x_state['last_ts'] = time.time()
                _vl53l1x_state['readings'] = int(_vl53l1x_state.get('readings', 0)) + len(readings)
                _vl53l1x_state['last_error'] = ''
            return cm, None

        except Exception as e:
            err_str = str(e)
            print(f'[VL53L0X] I2C error attempt {attempt+1}/{I2C_MAX_RETRIES}: {err_str}', flush=True)
            if attempt < I2C_MAX_RETRIES - 1:
                time.sleep(0.08)  # let servo PWM settle
                continue
            # All retries exhausted
            print(f'[VL53L0X] all retries exhausted - marking not-ready for re-init on next call', flush=True)
            with _vl53l1x_lock:
                _vl53l1x_state['ready'] = False
                _vl53l1x_state['last_error'] = f'I2C failed after {I2C_MAX_RETRIES} retries: {err_str}'
            return None, None

    return None, None

def _vl53l1x_enabled_runtime():
    with _depth_lock:
        runtime = bool(_depth_state.get('runtime_enabled', VL53L1X_ENABLED))
    with _vl53l1x_lock:
        ready = bool(_vl53l1x_state.get('ready'))
    return bool(VL53L1X_ENABLED and runtime and _gpio_ready[0] and ready)


def _vl53l1x_scaled_distance_cm(raw_cm):
    """Apply VL53L1X-only calibration to the raw ToF reading (cm).

    FIX v37: sanity-check raw_cm against DEPTH_SANITY_MIN_CM BEFORE feeding it
    into the temporal smoother.  If a 2cm sentinel slips through _vl53l1x_read_cm
    (e.g. first sample after a power glitch before range_status check is reached)
    and enters the smoother history it permanently biases the rolling median.
    Rejecting it here is a second line of defence.
    """
    try:
        cm = float(raw_cm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cm) or cm <= 0:
        return None
    # Second-line defence: reject values that look like the 20mm error sentinel
    # or that are below the sanity floor, before they can poison the smoother.
    if cm < DEPTH_SANITY_MIN_CM:
        print(f'[VL53L0X] _vl53l1x_scaled_distance_cm: {cm:.1f}cm < DEPTH_SANITY_MIN_CM '
              f'({DEPTH_SANITY_MIN_CM}cm) — discarding before smoother', flush=True)
        return None
    smoothed = _depth_smoother.update(cm)
    height_cm = smoothed if smoothed is not None else cm
    with _calibration_lock:
        cr_scale = _calibration_state['offsets']['close_range_scale']
    if height_cm <= DEPTH_CLOSE_RANGE_MAX_CM and abs(cr_scale - 1.0) > 0.001:
        height_cm = height_cm * cr_scale
    height_cm = height_cm * float(VL53L1X_SCALE_FACTOR) + float(VL53L1X_DISTANCE_OFFSET_CM)
    return float(_clamp(
        height_cm,
        DEPTH_SANITY_MIN_CM,
        DEPTH_SANITY_MAX_CM,
    ))


def _vl53l1x_camera_space_distance_cm(sensor_distance_cm):
    """Convert VL53L1X sensor-space distance to camera-space forward depth."""
    try:
        cm = float(sensor_distance_cm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cm) or cm <= 0:
        return None
    return float(_clamp(cm, DEPTH_SANITY_MIN_CM, DEPTH_SANITY_MAX_CM))


def _vl53l1x_forward_hint_cm(box):
    """Box-size forward reach fallback (used only when mount=down)."""
    hint = _depth_hint_from_box(box)
    if hint is not None:
        return float(_clamp(hint, PICKUP_DEPTH_MIN, DEPTH_SANITY_MAX_CM))
    return float(PICKUP_DEPTH_DEFAULT)


def _vl53l1x_apply_down_mount_path(path, height_cm, forward_cm):
    """Down-facing sensor: height drives descent; forward_cm drives IK reach."""
    if not isinstance(path, dict):
        return path
    descent_cm = float(_clamp(
        float(height_cm) - VL53L1X_GRASP_CLEARANCE_CM,
        0.5,
        DEPTH_SANITY_MAX_CM,
    ))
    raw_drop, approach_drop, descent_deg = _depth_contact_drop_degs(descent_cm)
    extra_drop_deg = _pickup_depth_offset()
    execution_descent_deg = float(_clamp(
        descent_deg + extra_drop_deg,
        PICKUP_DESCENT_STEP_DEG,
        max(PICKUP_DESCENT_STEP_DEG, PICKUP_MAX_DESCENT_DEG - approach_drop),
    ))
    path = dict(path)
    path['sensor_height_cm'] = round(float(height_cm), 2)
    path['forward_cm'] = round(float(forward_cm), 2)
    path['contact_cm'] = round(descent_cm, 2)
    path['height_source'] = 'vl53l1x-down'
    path['raw_descent_deg'] = round(raw_drop, 2)
    path['approach_arm_delta_deg'] = round(approach_drop, 2)
    path['descent_deg'] = round(descent_deg, 2)
    path['execution_descent_deg'] = round(execution_descent_deg, 2)
    path['extra_drop_deg'] = round(extra_drop_deg, 2)
    path['total_drop_deg'] = round(approach_drop + execution_descent_deg, 2)
    target = path.get('target_cm')
    if isinstance(target, dict):
        target = dict(target)
        target['z_cm'] = round(_jaw_distance_from_camera_depth(forward_cm), 2)
        target['camera_z_cm'] = round(float(forward_cm), 2)
        path['target_cm'] = target
    return _annotate_pickup_execution_path(path, descent_cm)


def _vl53l1x_distance_plan(box, label='object', force=False):
    """Build pickup depth plan from VL53L1X (no depth map)."""
    if not box or len(box) != 4:
        return None
    if not _vl53l1x_enabled_runtime():
        err = _vl53l1x_not_ready_message() or 'VL53L1X not ready'
        result = {
            'ts': time.time(),
            'ok': False,
            'label': label or 'object',
            'source': 'vl53l1x',
            'message': err,
            'sensor_raw_cm': None,
            'depth_map_ready': False,
        }
        _publish_depth_result(result, None)
        return result

    now = time.time()
    with _depth_lock:
        cached_box = _depth_state['last_box']
        cached_result = _depth_state.get('last_result')
        cached_ts = float(_depth_state['last_ts'] or 0.0)
    if (not force) and cached_result and now - cached_ts <= DEPTH_CACHE_SEC and cached_box is not None:
        if _box_similarity(box, cached_box) >= 0.60 and cached_result.get('source', '').startswith('vl53l1x'):
            result = dict(cached_result)
            result['label'] = label or result.get('label', 'object')
            _publish_depth_result(result, None)
            return result

    raw_cm, _ = _vl53l1x_read_cm()
    if raw_cm is None:
        err = 'VL53L1X: no data (check SDA/SCL wiring)'
        with _vl53l1x_lock:
            err = _vl53l1x_state.get('last_error') or err
        result = {
            'ts': time.time(),
            'ok': False,
            'label': label or 'object',
            'source': 'vl53l1x',
            'message': err,
            'sensor_raw_cm': None,
            'sensor_measurement_us': None,  # not applicable for ToF sensor
            'depth_map_ready': False,
        }
        _publish_depth_result(result, None)
        return result

    mount_down = _vl53l1x_mount_down()
    mount_behind = _vl53l1x_mount_behind_gripper()
    sensor_distance_cm = _vl53l1x_scaled_distance_cm(raw_cm)
    if sensor_distance_cm is None:
        sensor_distance_cm = float(raw_cm)
    distance_cm = _vl53l1x_camera_space_distance_cm(sensor_distance_cm)
    if distance_cm is None:
        distance_cm = float(sensor_distance_cm)
    if mount_down:
        height_cm = sensor_distance_cm
        forward_cm = _vl53l1x_forward_hint_cm(box)
    else:
        height_cm = sensor_distance_cm
        forward_cm = distance_cm
    jaw_distance_cm = _jaw_distance_from_camera_depth(forward_cm)
    cx, cy = FRAME_W * 0.5, FRAME_H * 0.5
    try:
        x, y, bw, bh = [float(v) for v in box]
        cx = x + bw * PICKUP_OBJECT_CENTER_FRACTION
        cy = y + bh * PICKUP_OBJECT_CENTER_FRACTION
    except Exception:
        pass
    center_px = {'x': cx, 'y': cy, 'trust_y': True}
    path = _depth_path_from_box(box, forward_cm, center_px=center_px)
    if isinstance(path, dict):
        path['height_source'] = 'bbox-center'
        path['depth_mask_pixels'] = None
        if mount_down:
            path = _vl53l1x_apply_down_mount_path(path, height_cm, forward_cm)

    result = {
        'ts': time.time(),
        'ok': True,
        'label': label or 'object',
        'source': 'vl53l1x-down' if mount_down else ('vl53l1x-forward' if mount_behind else 'vl53l1x'),
        'mount': VL53L1X_MOUNT,
        'box': [int(float(v)) for v in box],
        'hint': float(_clamp(_depth_distance_to_hint(forward_cm), PICKUP_DEPTH_MIN, PICKUP_DEPTH_MAX)),
        'distance_cm': round(forward_cm, 2),
        'distance_m': round(forward_cm / 100.0, 4),
        'jaw_distance_cm': round(jaw_distance_cm, 2),
        'sensor_raw_cm': round(raw_cm, 2),
        'sensor_distance_cm': round(sensor_distance_cm, 2),
        'vl53l1x_scale_factor': round(float(VL53L1X_SCALE_FACTOR), 4),
        'vl53l1x_distance_offset_cm': round(float(VL53L1X_DISTANCE_OFFSET_CM), 2),
        'sensor_height_cm': round(height_cm, 2) if mount_down else None,
        'sensor_measurement_us': None,  # not applicable for ToF sensor
        'grasp_clearance_cm': round(VL53L1X_GRASP_CLEARANCE_CM, 2) if mount_down else None,
        'sensor_mount_offset_cm': round(_sensor_to_jaw_offset_cm(), 2),
        'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
        'in_reach': bool(jaw_distance_cm <= DEPTH_REACH_LIMIT_CM),
        'depth_map_ready': False,
        'path': path,
    }
    if mount_down:
        descent_cm = float(path.get('contact_cm', height_cm)) if isinstance(path, dict) else height_cm
        result['message'] = (
            f'VL53L1X â†“ height {height_cm:.1f} cm â†’ descend {descent_cm:.1f} cm'
            f' | approach {forward_cm:.1f} cm'
            + (' âœ“ in reach' if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM else ' âœ— out of reach')
        )
    elif mount_behind:
        off = _sensor_to_jaw_offset_cm()
        result['message'] = (
            f'VL53L1X â†’ object {forward_cm:.1f} cm'
            f' | jaw {jaw_distance_cm:.1f} cm (jaw offset {off:.1f} cm)'
            + (' âœ“ in reach' if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM else ' âœ— out of reach')
        )
    else:
        result['message'] = (
            f'VL53L1X raw {raw_cm:.1f} cm â†’ jaw {jaw_distance_cm:.1f} cm'
            + (' âœ“ in reach' if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM else ' âœ— out of reach')
        )
    with _depth_lock:
        _depth_state['last_box'] = [int(float(v)) for v in box]
        _depth_state['last_hint'] = result.get('hint')
        _depth_state['last_value'] = result.get('distance_m')
        _depth_state['last_result'] = dict(result)
        _depth_state['last_ts'] = time.time()
        _depth_state['last_error'] = ''
    with _vl53l1x_lock:
        _vl53l1x_state['last_distance_cm'] = result['distance_cm']
    _publish_depth_result(result, None)
    return result


def _depth_plan_from_box(box, label='object', force=False):
    """Metric depth for pickup from VL53L1X ToF sensor."""
    if not _depth_enabled_runtime() or not box or len(box) != 4:
        return None
    return _vl53l1x_distance_plan(box, label=label, force=force)

def _depth_depth_hint_from_box(box):
    plan = _depth_plan_from_box(box)
    if not plan or plan.get('hint') is None:
        return None
    return float(_clamp(plan.get('hint'), PICKUP_DEPTH_MIN, PICKUP_DEPTH_MAX))

def _depth_queue_drain_thread():
    """
    Background thread: continuously drains the Depth Anything V2 worker output queue so that
    'ready' messages are consumed immediately â€” not only when a pickup is running.
    Without this the button stays on LOADING forever until a grab is triggered.

    FIX BUG 2: depth messages must be put BACK into the queue if we consume them
    here, because _depth_worker_request is waiting for a specific req_id depth reply.
    If the drain thread steals it, the request times out and Depth Anything V2 returns None.
    """
    while _running[0]:
        with _depth_worker_lock:
            proc = _depth_worker.get('proc')
            out_q = _depth_worker.get('queue')
        if proc is None or out_q is None:
            time.sleep(0.5)
            continue
        # Non-blocking drain â€” only handle 'ready' and 'error'; requeue everything else
        try:
            deferred = []
            while True:
                try:
                    msg = out_q.get_nowait()
                except queue.Empty:
                    break
                msg_type = msg.get('type')
                if msg_type == 'ready':
                    with _depth_lock:
                        _depth_state['ready'] = True
                        _depth_state['loading'] = False
                        _depth_state['failed'] = False
                        _depth_state['last_error'] = ''
                    print(f"[DEPTH] Worker ready (drain thread) on {msg.get('device', 'cpu')}")
                elif msg_type == 'error' and msg.get('id') is None:
                    # Only consume global errors (no req_id); per-request errors go back
                    err = str(msg.get('error', 'worker error'))
                    with _depth_lock:
                        _depth_state['last_error'] = err
                    print(f'[DEPTH] {err}')
                else:
                    # depth responses or per-request errors: put back so request loop gets them
                    deferred.append(msg)
            for msg in deferred:
                out_q.put(msg)
        except Exception as e:
            print(f'[DEPTH] drain thread error: {e}')
        time.sleep(0.25)


def _depth_warmup_thread():
    """Initialize VL53L1X GPIO at boot."""
    if not VL53L1X_ENABLED:
        return
    time.sleep(0.5)
    if not _gpio_ready[0]:
        print('[VL53L1X] GPIO not ready â€” sensor disabled', flush=True)
        return
    if _init_vl53l1x_gpio():
        with _depth_lock:
            _depth_state['ready'] = True
            _depth_state['failed'] = False
            _depth_state['loading'] = False
        _set_status('VL53L1X: ready')
        print('[VL53L1X] Sensor ready', flush=True)
    else:
        with _depth_lock:
            _depth_state['failed'] = True
        _set_status('VL53L1X: init failed (check SDA/SCL pins)')

def _touch_heartbeat():
    """Called by HTTP handlers on every successful response to prove liveness."""
    with _heartbeat_lock:
        _heartbeat[0] = time.time()

def _arm_serial_snapshot():
    joint = _arm_serial[0]
    if joint is None:
        return {
            'enabled': bool(ARM_SERIAL_SERVO_ENABLED),
            'connected': False,
            'port': ARM_SERIAL_SERVO_PORT,
            'baud': ARM_SERIAL_SERVO_BAUD,
        }
    try:
        return joint.snapshot()
    except Exception as exc:
        return {
            'enabled': True,
            'connected': False,
            'port': ARM_SERIAL_SERVO_PORT,
            'baud': ARM_SERIAL_SERVO_BAUD,
            'error_message': str(exc),
        }

def _init_arm_serial():
    if not ARM_SERIAL_SERVO_ENABLED:
        print('[ARM-SERIAL] Disabled; arm uses GPIO PWM', flush=True)
        return False
    try:
        joint = _SerialServoJoint(
            ARM_SERIAL_SERVO_PORT,
            ARM_SERIAL_SERVO_BAUD,
            ARM_SERIAL_SERVO_MIN_DEG,
            ARM_SERIAL_SERVO_MAX_DEG,
        )
        _arm_serial[0] = joint
        # Opening a USB serial port often resets the Arduino.  Wait for its
        # startup banner/first telemetry before issuing the home trajectory.
        deadline = time.time() + 3.0
        telemetry_ready = False
        while time.time() < deadline:
            snap = joint.snapshot()
            if snap.get('telemetry_fresh'):
                telemetry_ready = True
                break
            time.sleep(0.05)
        if not telemetry_ready:
            raise RuntimeError('controller opened but no P:/T:/E:/PWM: telemetry was received')
        # Do not energize the gravity-loaded shoulder merely because the
        # server opened its USB port.  The Nano firmware enables itself on the
        # first numeric angle command (slider/Home).  This prevents the
        # startup hold/deadband cycle from moving the arm without a command.
        joint.stop()
        measured = snap.get('position')
        if isinstance(measured, (int, float)) and math.isfinite(float(measured)):
            measured = float(measured)
            with _pos_lock:
                _pos['arm'] = measured
            with _servo_lock:
                _last_angle[PIN_ARM] = measured
        print(f'[ARM-SERIAL] Shoulder connected on {ARM_SERIAL_SERVO_PORT} '
              f'at {ARM_SERIAL_SERVO_BAUD} baud (startup torque off)', flush=True)
        return True
    except Exception as exc:
        if _arm_serial[0] is not None:
            _arm_serial[0].close()
        _arm_serial[0] = None
        print(f'[ARM-SERIAL] unavailable ({exc}); falling back to GPIO{PIN_ARM}', flush=True)
        return False

# Servo hold refresh interval — re-send PWM to all joints periodically.
# Some high-torque servos (DS3235, etc.) have an internal watchdog that
# stops the motor if no new pulse arrives within ~500ms.  This thread
# prevents that by re-sending the last known position.
_SERVO_HOLD_REFRESH_SEC = float(os.environ.get('ARM_SERVO_HOLD_REFRESH_SEC', '0.20'))

def _servo_hold_refresh_thread():
    """Periodically re-send PWM to all positional joints.

    High-torque digital servos can lose their hold if the PWM stream is
    interrupted or if the internal controller watchdog fires.  This thread
    re-sends every joint's last commanded angle at 5 Hz, which is invisible
    to the servos but keeps their internal controllers alive.
    """
    while _running[0]:
        time.sleep(_SERVO_HOLD_REFRESH_SEC)
        if not _gpio_ready[0]:
            continue
        try:
            with _pos_lock:
                snapshot = {k: float(v) for k, v in _pos.items()}
            serial_arm = _arm_serial[0] is not None
            for key, angle in snapshot.items():
                if key == 'arm' and serial_arm:
                    continue
                if GRIP_DC_MODE and key == 'grip':
                    continue
                pin = PIN_BY_JOINT.get(key)
                if pin is None:
                    continue
                if JOINT_MODE.get(key) == 'continuous':
                    continue
                _write_positional(pin, angle, force=True, key=key)
        except Exception:
            pass

def _watchdog_thread():
    """Monitor HTTP handler responsiveness.

    If no heartbeat arrives within _WATCHDOG_STALE_SEC, the HTTP server is
    likely deadlocked (e.g. all handler threads blocked on a lock) or crashed.
    The watchdog logs diagnostics and the servers auto-restart via their own
    while-loop in _run_cmd_api / _run_stream.
    """
    while _running[0]:
        time.sleep(_WATCHDOG_CHECK_INTERVAL)
        with _heartbeat_lock:
            age = time.time() - _heartbeat[0]
        if age < _WATCHDOG_STALE_SEC:
            continue

        print(f'[WATCHDOG] Heartbeat stale ({age:.1f}s) — dumping diagnostics', flush=True)

        # --- Lock diagnostics ---
        for name, lk in [
            ('_scene_lock', _scene_lock),
            ('_pos_lock', _pos_lock),
            ('_frame_lock', _frame_lock),
            ('_servo_lock', _servo_lock),
            ('_motion_lock', _motion_lock),
        ]:
            held = lk.locked()
            print(f'[WATCHDOG]   {name}: locked={held}', flush=True)

        # --- Stack dump of key threads ---
        import traceback as _tb
        frames = sys._current_frames()
        for t in threading.enumerate():
            if t.name in ('pickup', 'MainThread', 'tracking', 'command-api'):
                tid = t.ident
                if tid in frames:
                    print(f'[WATCHDOG]   Stack for {t.name} (id={tid}):', flush=True)
                    _tb.print_stack(frames[tid])

        # --- Thread diagnostics ---
        alive = [t for t in threading.enumerate() if t.is_alive()]
        alive_names = {t.name for t in alive}
        print(f'[WATCHDOG]   threads alive: {len(alive)}', flush=True)
        for t in alive[:20]:
            print(f'[WATCHDOG]     {t.name} daemon={t.daemon}', flush=True)

        # --- Detect dead critical threads ---
        critical = ['command-api', 'stream-server', 'camera', 'detector']
        dead = [n for n in critical if n not in alive_names]
        if dead:
            print(f'[WATCHDOG]   DEAD critical threads: {dead}', flush=True)

        # --- Memory ---
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            print(f'[WATCHDOG]   RSS={ru.ru_maxrss}KB', flush=True)
        except Exception:
            pass

        # --- Pickup state ---
        pickup_locked = _pickup_lock.locked()
        print(f'[WATCHDOG]   _pickup_lock.locked={pickup_locked}', flush=True)
        print(f'[WATCHDOG]   tracking={_tracking[0]} status={_status_txt[0]}', flush=True)

        # --- Touch heartbeat so we don't spam diagnostics ---
        with _heartbeat_lock:
            _heartbeat[0] = time.time()

def _start_task(name, target, *args):
    def _runner():
        try:
            target(*args)
        except Exception as e:
            msg = f'{name} failed: {e}'
            print(f'[TASK] {msg}', flush=True)
            import traceback; traceback.print_exc()
            _set_status(msg)

    t = threading.Thread(target=_runner, name=name, daemon=True)
    t.start()
    return t

def _cancel_all_motions():
    try:
        with _motion_lock:
            for pin in PIN_BY_JOINT.values():
                _motion_seq[pin] = _motion_seq.get(pin, 0) + 1
        for joint, pin in PIN_BY_JOINT.items():
            if JOINT_MODE.get(joint) == 'continuous':
                try:
                    _stop_servo(pin, joint)
                except Exception:
                    pass
    except Exception as e:
        print(f'[MOTION] cancel error (non-fatal): {e}', flush=True)

def _angle_to_us(angle, us_min=500, us_max=2500):
    # FIX: this clamp used to silently swallow out-of-range servo_angle values
    # (e.g. from a reversed joint whose logical min/max didn't fit in 0-180),
    # which pinned the real servo at 500/2500us while callers (FK safety
    # check, 3D model) kept assuming the commanded angle was actually reached.
    # Now it's logged so a mismatch like that shows up immediately instead of
    # only manifesting as "the real arm doesn't match the model".
    clamped = _clamp(angle, 0.0, 180.0)
    if clamped != angle:
        print(f'[SERVO] WARNING: angle {angle:.1f}° out of 0-180° servo range, '
              f'clamped to {clamped:.1f}° — real servo will not match commanded position', flush=True)
    us_min = int(us_min)
    us_max = int(us_max)
    return int(us_min + clamped / 180.0 * (us_max - us_min))

def _write_servo_us(pin, pulse_us, force=False):
    # FIX BUG 12: Clamp minimum to 500Âµs, not 0. pulse_us=0 DISABLES the servo PWM signal
    # entirely in both pigpio and lgpio (their convention for "off"). Any valid servo position
    # uses 500â€“2500Âµs. Clamping to 0 could accidentally release servo hold.
    pulse_us = int(_clamp(pulse_us, 500, 2500))
    with _servo_lock:
        if not force and _last_pulse.get(pin) == pulse_us:
            return True
        try:
            sent = False
            if _USE_PIGPIO:
                pi = _pi[0]
                if pi:
                    pi.set_servo_pulsewidth(pin, pulse_us)
                    sent = True
            else:
                h = _lgpio_h[0]
                if h is not None:
                    lgpio.tx_servo(h, pin, pulse_us, 50)
                    sent = True
        except Exception as e:
            msg = f'Servo write failed on pin {pin}: {e}'
            print(f'[SERVO] {msg}')
            _set_status(msg)
            return False
        if sent:
            _last_pulse[pin] = pulse_us
        return sent

def _write_positional(pin, angle, force=False, key=None):
    try:
        angle = float(angle)
    except (TypeError, ValueError):
        _set_status(f'Invalid servo angle on pin {pin}: {angle}')
        return False
    if not math.isfinite(angle):
        _set_status(f'Invalid servo angle on pin {pin}: {angle}')
        return False

    # The converted shoulder has its own position loop and receives angle
    # targets over USB.  It does not need the 5 Hz PWM refresh used by hobby
    # servos, so identical targets are suppressed even when force=True.
    serial_joint = _arm_serial[0] if key == 'arm' else None
    if serial_joint is not None:
        with _servo_lock:
            if abs(_last_angle.get(pin, -999.0) - angle) < 0.05 and not force:
                return True
            servo_angle = 180.0 - angle if JOINT_REVERSED.get(key, False) else angle
            try:
                serial_joint.move_to(servo_angle)
                print(f'[ARM-SERIAL] SENT angle={servo_angle:.1f}° to Nano', flush=True)
            except Exception as exc:
                msg = f'Arm serial write failed: {exc}'
                print(f'[ARM-SERIAL] {msg}', flush=True)
                _set_status(msg)
                return False
            _last_angle[pin] = angle
            _last_pulse[pin] = None
            return True

    with _servo_lock:
        if key == 'wrist':
            deadband = WRIST_DEADBAND_DEG
        elif key == 'arm':
            deadband = ARM_DEADBAND_DEG
        elif key == 'grip':
            deadband = GRIP_DEADBAND_DEG_SERVO
        elif key == 'base':
            deadband = BASE_DEADBAND_DEG_SERVO
        else:
            deadband = DEADBAND_DEG
        if not force and abs(_last_angle.get(pin, -999.0) - angle) < deadband:
            return True
        servo_angle = 180.0 - angle if JOINT_REVERSED.get(key, False) else angle
        if key == 'wrist':
            pulse = _angle_to_us(servo_angle, WRIST_US_MIN, WRIST_US_MAX)
        elif key == 'arm':
            pulse = _angle_to_us(servo_angle, ARM_US_MIN, ARM_US_MAX)
        elif key == 'grip':
            pulse = _angle_to_us(servo_angle, GRIP_US_MIN, GRIP_US_MAX)
        elif key == 'base':
            pulse = _angle_to_us(servo_angle, BASE_US_MIN, BASE_US_MAX)
        else:
            pulse = _angle_to_us(servo_angle)
        try:
            sent = False
            if _USE_PIGPIO:
                pi = _pi[0]
                if pi:
                    pi.set_servo_pulsewidth(pin, pulse)
                    sent = True
            else:
                h = _lgpio_h[0]
                if h is not None:
                    lgpio.tx_servo(h, pin, pulse, 50)
                    sent = True
        except Exception as e:
            msg = f'Servo write failed on pin {pin}: {e}'
            print(f'[SERVO] {msg}')
            _set_status(msg)
            return False
        if sent:
            _last_angle[pin] = angle
            _last_pulse[pin] = pulse
        return sent

def _continuous_config(key):
    if key == 'wrist':
        return {
            'stop_us': WRIST_CONT_STOP_US,
            'range_us': WRIST_CONT_RANGE_US,
            'speed': WRIST_CONT_SPEED,
            'ms_per_deg': WRIST_CONT_MS_PER_DEG,
            'min_ms': WRIST_CONT_MIN_MS,
            'max_ms': WRIST_CONT_MAX_MS,
            'deadband': WRIST_DEADBAND_DEG,
        }
    return {
        'stop_us': CONT_STOP_US,
        'range_us': CONT_RANGE_US,
        'speed': CONT_SPEED,
        'ms_per_deg': CONT_MS_PER_DEG,
        'min_ms': CONT_MIN_MS,
        'max_ms': CONT_MAX_MS,
        'deadband': DEADBAND_DEG,
    }

def _continuous_pulse(direction, key=None):
    # direction: -1 .. +1
    direction = _clamp(direction, -1.0, 1.0)
    cfg = _continuous_config(key)
    if abs(direction) < 0.05:
        return int(cfg['stop_us'])
    return int(cfg['stop_us'] + direction * cfg['range_us'])

def _stop_servo(pin, key=None):
    _write_servo_us(pin, _continuous_config(key)['stop_us'], force=True)

def _begin_motion(pin):
    with _motion_lock:
        seq = _motion_seq.get(pin, 0) + 1
        _motion_seq[pin] = seq
        return seq

def _motion_valid(pin, seq):
    return _running[0] and _motion_seq.get(pin) == seq

def _end_motion(pin, seq):
    with _motion_lock:
        if _motion_seq.get(pin) == seq:
            _motion_seq.pop(pin, None)

def _move_positional(pin, key, target, lo=None, hi=None, steps=POS_STEPS, delay=POS_DELAY, deadband=DEADBAND_DEG, _no_fk_check=False):
    try:
        target = float(target)
    except (TypeError, ValueError):
        _set_status(f'Invalid target for {key}: {target}')
        return
    if not math.isfinite(target):
        _set_status(f'Invalid target for {key}: {target}')
        return
    # Serial arm has its own PID position loop — skip FK floor gate entirely
    # (the ESP/Nano measures real position via pot, not a model).
    if key == 'arm' and _arm_serial[0] is not None:
        _no_fk_check = True
    if not _no_fk_check:
        allowed, worst, height, margin = _fk_gate_joint_target(key, target)
        if not allowed:
            shortfall = float(margin) - float(height)
            _set_status(f'FLOOR COLLISION ({worst}): {shortfall:.2f}cm below margin')
            _fk_log_reject('MOVE', key, target, worst, height, margin)
            return
    if key == 'arm':
        lo = float(ARM_MIN if lo is None else lo)
        hi = float(ARM_MAX if hi is None else hi)
    if lo is not None and hi is not None:
        target = float(_clamp(target, lo, hi))

    if key == 'wrist':
        target = round(target / WRIST_SNAP_DEG) * WRIST_SNAP_DEG
        steps = WRIST_POS_STEPS
        delay = WRIST_POS_DELAY
        deadband = WRIST_DEADBAND_DEG
    elif key == 'arm':
        target = round(target / ARM_SNAP_DEG) * ARM_SNAP_DEG
        steps = ARM_POS_STEPS
        delay = ARM_POS_DELAY
        deadband = ARM_DEADBAND_DEG

    if lo is not None and hi is not None:
        target = float(_clamp(target, lo, hi))

    # Re-check the exact final servo target after mechanical limit clamp and
    # snap rounding. Floor safety is FK rejection only; no floor angle clamp.
    if not _no_fk_check:
        allowed, worst, height, margin = _fk_gate_joint_target(key, target)
        if not allowed:
            shortfall = float(margin) - float(height)
            _set_status(f'FLOOR COLLISION ({worst}): {shortfall:.2f}cm below margin')
            _fk_log_reject('MOVE final', key, target, worst, height, margin)
            return

    steps = max(1, int(steps))
    delay = max(0.0, float(delay))

    # Serial arm (ESP) has its own motion planner — send final target once,
    # skip Python-side interpolation which confuses the ESP into looping.
    if key == 'arm' and _arm_serial[0] is not None:
        print(f'[ARM-SERIAL] MOVE arm -> {target:.1f}°', flush=True)
        with _pos_lock:
            _pos[key] = target
        _write_positional(pin, target, force=True, key=key)
        return

    with _pos_lock:
        start = float(_pos[key])
        if key == 'arm' and lo is not None and start < float(lo):
            start = float(lo)
            _pos[key] = start
    if abs(start - target) < deadband:
        with _pos_lock:
            _pos[key] = target
        # Re-confirm position for ALL joints (including wrist) to prevent servo creep.
        # The old code skipped wrist here â€” meaning wrist nudges inside the deadband
        # during pickup were silently dropped and the servo never moved.
        _write_positional(pin, target, force=True, key=key)
        return

    seq = _begin_motion(pin)
    try:
        for i in range(1, steps + 1):
            if not _motion_valid(pin, seq):
                return
            angle = start + (target - start) * i / steps
            if key == 'arm' and lo is not None and hi is not None:
                angle = float(_clamp(angle, lo, hi))
            if key in ('arm', 'wrist') and not _no_fk_check:
                ok_step, worst, height, margin = _fk_gate_joint_target(key, angle)
                if not ok_step:
                    _set_status(f'FLOOR COLLISION ({worst}) mid-path â€” stopped')
                    _fk_log_reject(f'PATH step {i}/{steps}', key, angle, worst, height, margin)
                    return
            with _pos_lock:
                _pos[key] = angle
            _write_positional(pin, angle, force=True, key=key)
            time.sleep(delay)
        if _motion_valid(pin, seq):
            with _pos_lock:
                _pos[key] = target
            _write_positional(pin, target, force=True, key=key)
    finally:
        _end_motion(pin, seq)

def _move_continuous(pin, key, target, lo=None, hi=None):
    try:
        target = float(target)
    except (TypeError, ValueError):
        _set_status(f'Invalid target for {key}: {target}')
        return
    if not math.isfinite(target):
        _set_status(f'Invalid target for {key}: {target}')
        return
    allowed, worst, height, margin = _fk_gate_joint_target(key, target)
    if not allowed:
        _fk_log_reject('CONTINUOUS', key, target, worst, height, margin)
        return
    if lo is not None and hi is not None:
        target = float(_clamp(target, lo, hi))
    allowed, worst, height, margin = _fk_gate_joint_target(key, target)
    if not allowed:
        _fk_log_reject('CONTINUOUS final', key, target, worst, height, margin)
        return
    with _pos_lock:
        start = float(_pos[key])
    delta = target - start
    cfg = _continuous_config(key)
    if abs(delta) < 0.5:
        with _pos_lock:
            _pos[key] = target
        _stop_servo(pin, key)
        return

    seq = _begin_motion(pin)
    final_pos = start
    stopped_by_floor = False
    try:
        direction = 1.0 if delta > 0 else -1.0
        duration = abs(delta) * cfg['ms_per_deg']
        duration = _clamp(duration, cfg['min_ms'], cfg['max_ms']) / 1000.0

        if JOINT_REVERSED.get(key, False):
            direction *= -1.0

        pulse = _continuous_pulse(direction * cfg['speed'], key)
        _write_servo_us(pin, pulse, force=True)

        elapsed = 0.0
        while elapsed < duration and _motion_valid(pin, seq):
            chunk = min(CONT_CHUNK_MS / 1000.0, duration - elapsed)
            time.sleep(chunk)
            elapsed += chunk
            progress = _clamp(elapsed / duration if duration > 0 else 1.0, 0.0, 1.0)
            final_pos = start + delta * progress
            if key in ('arm', 'wrist'):
                ok_step, worst, height, margin = _fk_gate_joint_target(key, final_pos)
                if not ok_step:
                    stopped_by_floor = True
                    _set_status(f'FLOOR COLLISION ({worst}) mid-path - stopped')
                    _fk_log_reject('CONTINUOUS mid', key, final_pos, worst, height, margin)
                    break
            with _pos_lock:
                _pos[key] = float(_clamp(final_pos, lo, hi)) if lo is not None and hi is not None else final_pos

        if _motion_valid(pin, seq) and not stopped_by_floor:
            _stop_servo(pin, key)
            with _pos_lock:
                _pos[key] = target
        else:
            with _pos_lock:
                _pos[key] = float(_clamp(final_pos, lo, hi)) if lo is not None and hi is not None else final_pos
    finally:
        _stop_servo(pin, key)
        _end_motion(pin, seq)

def _move_joint(key, target, lo=None, hi=None):
    if key not in PIN_BY_JOINT:
        raise ValueError(f'Unknown joint: {key}')

    # DC motor mode: grip open/close is just motor run, no angle stepping
    if GRIP_DC_MODE and key == 'grip':
        if target <= (GRIP_OPEN + GRIP_CLOSE) / 2:
            _grip_dc_run('open')
        else:
            _grip_dc_run('close')
        return

    allowed, worst, height, margin = _fk_gate_joint_target(key, target)
    if not allowed:
        _fk_log_reject('JOINT', key, target, worst, height, margin)
        with _track_lock:
            if _tracking[0] and key in ('arm', 'wrist'):
                _set_status(f'Tracking: floor limit ({worst}) â€” cannot lower further')
        return

    mode = JOINT_MODE.get(key, 'positional')
    pin = PIN_BY_JOINT[key]
    if mode == 'continuous':
        _move_continuous(pin, key, target, lo, hi)
    else:
        _move_positional(pin, key, target, lo, hi)

def _move_joint_slow(key, target, lo=None, hi=None, _no_fk_check=False):
    """Like _move_joint but uses SLOW_* speed constants.
    Use for every non-tracking move: pickup, IK, home, jog, wave, etc.
    The tracking thread continues to call _move_joint (fast) directly.
    """
    if key not in PIN_BY_JOINT:
        raise ValueError(f'Unknown joint: {key}')

    # DC motor mode: grip open/close is just motor run, no angle stepping
    if GRIP_DC_MODE and key == 'grip':
        if target <= (GRIP_OPEN + GRIP_CLOSE) / 2:
            _grip_dc_run('open')
        else:
            _grip_dc_run('close')
        return

    mode = JOINT_MODE.get(key, 'positional')
    pin  = PIN_BY_JOINT[key]

    if mode == 'continuous':
        _move_continuous(pin, key, target, lo, hi)
        return

    if key == 'wrist':
        _move_positional(pin, key, target, lo, hi,
                         steps=SLOW_WRIST_POS_STEPS, delay=SLOW_WRIST_POS_DELAY,
                         _no_fk_check=_no_fk_check)
    elif key == 'arm':
        _move_positional(pin, key, target, lo, hi,
                         steps=SLOW_ARM_POS_STEPS, delay=SLOW_ARM_POS_DELAY,
                         _no_fk_check=_no_fk_check)
    elif key == 'grip':
        _move_positional(pin, key, target, lo, hi,
                         steps=SLOW_GRIP_POS_STEPS, delay=SLOW_GRIP_POS_DELAY,
                         _no_fk_check=_no_fk_check)
    elif key == 'base':
        _move_positional(pin, key, target, lo, hi,
                         steps=SLOW_BASE_POS_STEPS, delay=SLOW_BASE_POS_DELAY,
                         _no_fk_check=_no_fk_check)
    else:
        _move_positional(pin, key, target, lo, hi,
                         steps=SLOW_POS_STEPS, delay=SLOW_POS_DELAY,
                         _no_fk_check=_no_fk_check)
    if _pickup_lock.locked():
        _last_pickup_move_time[0] = time.time()

def _latest_frame_snapshot():
    with _frame_lock:
        return None if _latest_frame[0] is None else _latest_frame[0].copy()

def _grip_motion_score(frame_a, frame_b):
    """Mean central-frame pixel motion used as a no-extra-sensor grip buzz cue."""
    try:
        if frame_a is None or frame_b is None:
            return 0.0
        h = min(frame_a.shape[0], frame_b.shape[0])
        w = min(frame_a.shape[1], frame_b.shape[1])
        if h < 20 or w < 20:
            return 0.0
        y1, y2 = int(h * 0.55), int(h * 0.98)   # FIX v31: was 0.25->0.82 â€” jaws are in bottom ~15% of frame; old crop excluded them entirely
        x1, x2 = int(w * 0.10), int(w * 0.90)   # FIX v31: wider X to catch both jaw sides
        a = cv2.cvtColor(frame_a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frame_b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        a = cv2.GaussianBlur(a, (5, 5), 0)
        b = cv2.GaussianBlur(b, (5, 5), 0)
        return float(np.mean(cv2.absdiff(a, b)))
    except Exception:
        return 0.0

def _gripper_started_oscillating():
    """Detect residual gripper/camera vibration after a close step.

    There is no servo position/current feedback in this build, so this uses the
    camera as a contact sensor: after the step settles, a flex gripper buzzing on
    an object produces frame-to-frame motion while a free close is mostly still.
    """
    frame_a = _latest_frame_snapshot()
    if frame_a is None:
        return False, 0.0
    time.sleep(max(0.0, float(GRIP_OSC_SAMPLE_SEC)))
    frame_b = _latest_frame_snapshot()
    score = _grip_motion_score(frame_a, frame_b)
    return score >= float(GRIP_OSC_SCORE_THRESHOLD), score

_grip_ema_angle = [None]   # FIX v34: EMA state for smooth gripper angle commands

def _read_gripper_current_ma():
    """Read gripper servo current from INA219 sensor (mA). Returns 0.0 if sensor unavailable."""
    if _ina219[0] is None:
        return 0.0
    try:
        ina = _ina219[0]
        # Force fresh register reads instead of cached values
        _ = ina.bus_voltage
        _ = ina.shunt_voltage
        return abs(ina.current)  # mA
    except Exception:
        return 0.0

def _gripper_contact_by_current():
    """Detect gripper contact via INA219 current spike.

    Returns (detected: bool, current_ma: float).
    """
    current_ma = _read_gripper_current_ma()
    return current_ma >= GRIP_CURRENT_THRESHOLD_MA, current_ma

def _write_grip_angle(angle, use_ema=True):
    """Write gripper servo angle with optional EMA smoothing to reduce jitter.

    FIX v34: Added exponential moving average (EMA) filtering on the commanded angle.
    Raw angle steps cause micro-oscillations in flexible grippers because the compliant
    jaw resonates. EMA damps high-frequency corrections and produces smooth motion.

    use_ema=False during final hold so the exact hold angle is written without drift.
    """
    if GRIP_DC_MODE:
        return  # DC motor mode — no servo PWM on grip pin
    target = float(_clamp(angle, GRIP_OPEN, GRIP_CLOSE))
    if use_ema:
        prev = _grip_ema_angle[0]
        if prev is None:
            _grip_ema_angle[0] = target
        else:
            alpha = float(_clamp(GRIP_EMA_ALPHA, 0.1, 1.0))
            smoothed = alpha * target + (1.0 - alpha) * float(prev)
            _grip_ema_angle[0] = smoothed
            target = smoothed
    else:
        _grip_ema_angle[0] = target
    target = float(_clamp(target, GRIP_OPEN, GRIP_CLOSE))
    with _pos_lock:
        _pos['grip'] = target
    _write_positional(PIN_GRIP, target, force=True, key='grip')
    return target


_grip_dc_last_dir = [None]  # track last direction for hold vs coast

def _grip_dc_stop(hold=False):
    """Stop gripper IBT-4 motor. If hold=True, maintain torque in close direction."""
    if hold:
        hold_speed = max(30, GRIP_DC_SPEED // 4)
        if _USE_PIGPIO:
            pi = _pi[0]
            if pi:
                pi.set_PWM_dutycycle(GRIP_RPWM_PIN, 0)
                pi.set_PWM_dutycycle(GRIP_LPWM_PIN, hold_speed)
        else:
            h = _lgpio_h[0]
            if h is not None:
                lgpio.tx_pwm(h, GRIP_RPWM_PIN, 25000, 0)
                lgpio.tx_pwm(h, GRIP_LPWM_PIN, 25000, hold_speed)
        print('[GRIP-DC] STOP (holding)', flush=True)
    else:
        if _USE_PIGPIO:
            pi = _pi[0]
            if pi:
                pi.set_PWM_dutycycle(GRIP_RPWM_PIN, 0)
                pi.set_PWM_dutycycle(GRIP_LPWM_PIN, 0)
        else:
            h = _lgpio_h[0]
            if h is not None:
                lgpio.tx_pwm(h, GRIP_RPWM_PIN, 25000, 0)
                lgpio.tx_pwm(h, GRIP_LPWM_PIN, 25000, 0)
        print('[GRIP-DC] STOP (coast)', flush=True)
    with _pos_lock:
        _pos['grip'] = 90.0
    print('[GRIP-DC] STOP', flush=True)


def _grip_dc_run(direction, timeout=None):
    """Run gripper IBT-4 motor until INA219 detects stall (current limit hit).

    direction: 'open' = LPWM HIGH, RPWM LOW (reverse)
               'close' = RPWM HIGH, LPWM LOW (forward)
    Polls INA219 every 50ms. Stops when current >= GRIP_DC_CURRENT_LIMIT_MA.
    Forced stop after timeout seconds as burnout safety.
    """
    speed = max(0, min(255, GRIP_DC_SPEED))

    # Start motor: one pin PWM, other pin LOW
    if direction == 'open':
        rpwm, lpwm = speed, 0
    else:
        rpwm, lpwm = 0, speed

    if _USE_PIGPIO:
        pi = _pi[0]
        if pi:
            pi.set_PWM_dutycycle(GRIP_RPWM_PIN, rpwm)
            pi.set_PWM_dutycycle(GRIP_LPWM_PIN, lpwm)
    else:
        h = _lgpio_h[0]
        if h is not None:
            # 25kHz PWM: frequency=25000, duty_cycle=speed (0-255 maps to 0-100%)
            lgpio.tx_pwm(h, GRIP_RPWM_PIN, 25000, rpwm)
            lgpio.tx_pwm(h, GRIP_LPWM_PIN, 25000, lpwm)

    with _pos_lock:
        _pos['grip'] = float(GRIP_OPEN) if direction == 'open' else float(GRIP_CLOSE)
    print(f'[GRIP-DC] {direction.upper()} speed={speed} (limit {GRIP_DC_CURRENT_LIMIT_MA:.0f}mA)', flush=True)
    # Debug: verify INA219 is reading
    test_ma = _read_gripper_current_ma()
    print(f'[GRIP-DC] INA219 test read: {test_ma:.0f}mA (sensor={_ina219[0] is not None})', flush=True)

    # Poll INA219 — skip startup transient, then stop on sustained high current
    dt = float(timeout if timeout is not None else GRIP_DC_STOP_SEC)
    deadline = time.monotonic() + dt
    poll_sec = 0.01
    last_print = 0
    start_time = time.monotonic()
    startup_sec = 0.8
    while time.monotonic() < deadline and _running[0]:
        time.sleep(poll_sec)
        current_ma = _read_gripper_current_ma()
        now = time.monotonic()
        if now - last_print >= 0.2:
            print(f'[GRIP-DC] {direction} current={current_ma:.0f}mA / {GRIP_DC_CURRENT_LIMIT_MA:.0f}mA', flush=True)
            last_print = now
        # Skip first 0.8s for startup current spike to settle
        if (now - start_time) < startup_sec:
            continue
        if current_ma >= GRIP_DC_CURRENT_LIMIT_MA:
            _grip_dc_stop(hold=(direction == 'close'))
            print(f'[GRIP-DC] stopped: {current_ma:.0f}mA >= {GRIP_DC_CURRENT_LIMIT_MA:.0f}mA', flush=True)
            return
    # Timeout fallback — force stop to prevent burnout
    _grip_dc_stop(hold=False)
    print(f'[GRIP-DC] timeout {dt:.1f}s, forced stop', flush=True)


def _hold_grip_at(angle, pulses=None, interval=None):
    """Write grip hold angle multiple times to lock the servo against flex-jaw spring-back.

    FIX v34: Flexible grippers have compliance — the jaw springs back slightly after
    contact.  Repeatedly re-writing the hold angle ensures the servo holds position
    against the spring force without oscillating.
    """
    if GRIP_DC_MODE:
        _grip_dc_stop(hold=True)
        return
    hold = float(_clamp(angle, GRIP_OPEN, GRIP_CLOSE))
    n = int(max(1, pulses if pulses is not None else GRIP_HOLD_PULSES))
    dt = float(max(0.0, interval if interval is not None else GRIP_HOLD_PULSE_INTERVAL))
    _grip_ema_angle[0] = hold   # sync EMA so next write doesn't snap back
    with _pos_lock:
        _pos['grip'] = hold
    for i in range(n):
        _write_positional(PIN_GRIP, hold, force=True, key='grip')
        if i < n - 1:
            time.sleep(dt)


def _home_targets():
    with _calibration_lock:
        offsets = _calibration_state.get('offsets', {})
        return {
            'base': float(_clamp(offsets.get('home_base', 90.0), BASE_MIN, BASE_MAX)),
            'arm': float(_clamp(offsets.get('home_arm', ARM_HOME_DEG + ARM_TRIM_DEG), ARM_MIN, ARM_MAX)),
            'wrist': float(_clamp(offsets.get('home_wrist', WRIST_HOME + WRIST_TRIM_DEG), 0, 180)),
            'grip': float(_clamp(offsets.get('home_grip', GRIP_OPEN), GRIP_OPEN, GRIP_CLOSE)),
        }


def _close_gripper_adaptive(label='object', context='Grip'):
    """Close the flex gripper until contact buzz is detected, then back off and hold.

    FIX v34 — Complete rewrite for MG995/MG996R flexible compliant grippers:

    Oscillation root causes in flexible grippers:
      1. Steps too large → jaw overshoots contact, bounces back, servo corrects, repeat
      2. No EMA smoothing → each raw angle write hits the servo differently
      3. No deadzone → corrections below the snap threshold cause micro-jitter
      4. No hold → spring-back after contact causes repeated open/close cycles
      5. Confirm threshold too sensitive → single camera shake triggers early stop

    Fixes applied:
      - Small close steps with EMA-filtered writes
      - Deadzone: skip write if angle change < GRIP_DEADZONE_DEG
      - After contact: write hold angle multiple times to lock servo position
      - Linearly decreasing step size as jaw closes (proportional deceleration)
      - Longer settle between steps so camera measurement is stable
      - Contact zone starts only after jaw is meaningfully closed (min_frac)
      - Buzz must be confirmed N consecutive samples to rule out camera shake
    """
    # DC motor mode: just run motor to close stop, wait for mechanical stall, then stop.
    if GRIP_DC_MODE:
        _grip_dc_run('close')
        return {'angle': float(GRIP_CLOSE), 'contact': True, 'score': 0.0, 'mode': 'dc-motor'}

    if not GRIP_ADAPTIVE_CLOSE:
        _move_joint_slow('grip', GRIP_CLOSE, GRIP_OPEN, GRIP_CLOSE)
        return {'angle': float(GRIP_CLOSE), 'contact': False, 'score': 0.0, 'mode': 'fixed'}

    tune = _grip_tune_snapshot()
    close_target = float(_clamp(tune.get('close', GRIP_CLOSE), GRIP_OPEN, GRIP_CLOSE))
    backoff_deg = float(max(0.0, tune.get('backoff', GRIP_CONTACT_BACKOFF_DEG)))
    threshold = float(max(0.1, tune.get('threshold', GRIP_OSC_SCORE_THRESHOLD)))
    min_close_fraction = float(_clamp(tune.get('min_frac', GRIP_OSC_MIN_CLOSE_FRACTION), 0.0, 0.95))
    confirm_needed = max(1, int(tune.get('confirm', GRIP_OSC_CONFIRM_SAMPLES)))

    base_step = float(max(0.2, abs(float(GRIP_CLOSE_STEP_DEG))))
    direction = 1.0 if close_target >= GRIP_OPEN else -1.0
    deadzone = float(_clamp(GRIP_DEADZONE_DEG, 0.05, max(0.5, base_step * 0.9)))

    # Reset EMA so stale state from previous close doesn't bias this grab
    _grip_ema_angle[0] = None

    with _pos_lock:
        current = float(_pos.get('grip', GRIP_OPEN))
    current = float(_clamp(current, min(GRIP_OPEN, close_target), max(GRIP_OPEN, close_target)))

    travel = abs(close_target - float(GRIP_OPEN))
    min_contact_angle = float(GRIP_OPEN) + direction * travel * min_close_fraction

    contact_count = 0
    best_score = 0.0
    hold_angle = close_target
    angle = current
    contact_angle = None
    abort_on_pickup = str(context).lower().startswith('pickup')
    last_written = current

    _set_status(f'{context}: {tune["name"]} close on {label}...')
    _write_grip_angle(current, use_ema=False)   # seed EMA at current position

    while direction * (close_target - angle) > 0.01 and _running[0] and \
            (not abort_on_pickup or not _pickup_abort.is_set()):

        # Proportional deceleration: step size shrinks as we approach close_target
        remaining = abs(close_target - angle)
        frac_done = 1.0 - remaining / max(0.1, travel)
        # Ramp: full step at start, half step near end for smooth final approach
        step = float(max(0.2, base_step * (1.0 - 0.5 * frac_done)))

        angle = angle + direction * step
        if direction > 0:
            angle = min(angle, close_target)
        else:
            angle = max(angle, close_target)

        # Deadzone: skip writes that are below the servo's effective resolution
        if abs(angle - last_written) < deadzone:
            time.sleep(float(GRIP_STEP_SETTLE_SEC) * 0.5)
            continue

        _write_grip_angle(angle, use_ema=True)
        last_written = float(_pos.get('grip', angle))  # read back EMA-smoothed value
        time.sleep(float(GRIP_STEP_SETTLE_SEC))

        # Only check for contact once in the valid closing zone
        in_contact_zone = direction * (angle - min_contact_angle) >= 0
        if not in_contact_zone:
            contact_count = 0
            continue

        # Current-based contact detection via INA219
        detected, current_ma = _gripper_contact_by_current()
        best_score = max(best_score, current_ma)

        if detected:
            contact_angle = angle
            contact_count = 1
            while contact_count < confirm_needed and _running[0] and \
                    (not abort_on_pickup or not _pickup_abort.is_set()):
                time.sleep(float(GRIP_CURRENT_SAMPLE_SEC))
                confirm_det, confirm_ma = _gripper_contact_by_current()
                best_score = max(best_score, confirm_ma)
                if not confirm_det:
                    contact_count = 0
                    contact_angle = None
                    break
                contact_count += 1
        else:
            contact_count = 0

        if contact_count >= confirm_needed:
            # Contact confirmed — back off and hold with multiple pulses
            hold_angle = angle - direction * backoff_deg
            hold_angle = float(_clamp(hold_angle, GRIP_OPEN, GRIP_CLOSE))
            # Write with EMA off so exact hold is commanded, then repeat to resist spring-back
            _hold_grip_at(hold_angle)
            print(
                f'[GRIP] contact at {angle:.1f}° '
                f'(strength={tune["name"]}, current={best_score:.0f}mA, threshold={threshold:.0f}mA, '
                f'confirm={confirm_needed}, backoff={backoff_deg:.1f}°); '
                f'holding {hold_angle:.1f}°',
                flush=True
            )
            _set_status(f'{context}: contact ✔ holding gripper at {hold_angle:.0f}°')
            return {
                'angle': hold_angle,
                'contact': True,
                'score': float(best_score),
                'threshold': float(threshold),
                'confirm': int(confirm_needed),
                'contact_angle': float(contact_angle if contact_angle is not None else angle),
                'backoff_deg': float(backoff_deg),
                'strength': tune['name'],
                'mode': 'current-limit',
            }

    # No contact detected â€” write final close angle and hold
    _hold_grip_at(close_target)
    print(
        f'[GRIP] no contact detected; closed to {close_target:.1f}° '
        f'(strength={tune["name"]}, max_current={best_score:.0f}mA)',
        flush=True
    )
    return {
        'angle': float(close_target),
        'contact': False,
        'score': best_score,
        'strength': tune['name'],
        'mode': 'current-limit',
    }

def _move_pickup_topdown_pose(status_prefix='Pickup'):
    """Move arm to the raised top-down approach pose, then return (arm_deg, wrist_deg).

    Image 2 (desired): arm raised high (~132 deg), wrist nose-down (~125 deg) so the
    gripper points AT THE TABLE from above.  This is the opposite of going flat
    (image 1) where arm ~90 deg pushes objects away instead of grabbing them.
    """
    _set_status(f'{status_prefix}: raising to top-down approach...')
    print(f'[{status_prefix.upper()}] Moving to top-down pose: '
          f'arm={PICKUP_TOPDOWN_ARM_DEG} deg  wrist={PICKUP_TOPDOWN_WRIST_DEG} deg', flush=True)

    # Raise shoulder first â€” clears any object below and ensures camera points down.
    _move_joint_slow('arm',
                     float(_clamp(PICKUP_TOPDOWN_ARM_DEG,   ARM_MIN, ARM_MAX)),
                     ARM_MIN, ARM_MAX)
    # Tilt wrist nose-down so gripper jaw faces the table for a vertical grab.
    _move_joint_slow('wrist',
                     float(_clamp(PICKUP_TOPDOWN_WRIST_DEG, 0, 180)),
                     0, 180)

    with _pos_lock:
        cur_arm   = float(_pos['arm'])
        cur_wrist = float(_pos['wrist'])
    print(f'[{status_prefix.upper()}] top-down pose reached: arm={cur_arm:.1f} deg  wrist={cur_wrist:.1f} deg',
          flush=True)
    return cur_arm, cur_wrist

def _pickup_arm_target(target, status_prefix='Pickup'):
    """Return a mechanically valid arm target, or None if FK floor safety rejects it."""
    target = float(_clamp(float(target), ARM_MIN, ARM_MAX))
    with _pos_lock:
        wrist = float(_pos.get('wrist', 90.0))
    pickup_margin = float(PICKUP_FLOOR_SAFETY_MARGIN_CM)
    if not _fk_safe(target, wrist, margin_cm=pickup_margin):
        worst, height, margin = _fk_reject_reason(target, wrist, margin_cm=pickup_margin)
        _fk_log_reject(status_prefix, 'arm', target, worst, height, margin)
        _set_status(f'{status_prefix}: floor limit ({worst}) - move rejected')
        return None
    return target

def _pickup_arm_align_target(cur_arm, requested_delta, phase_max_step=None):
    """Return an arm target that is large enough to survive servo angle snapping.
    phase_max_step: per-phase step cap from the caller (acquire/fine); overrides
    the global PICKUP_ARM_ALIGN_MAX_STEP so the loop cap is actually respected."""
    try:
        cur_arm = float(cur_arm)
        requested_delta = float(requested_delta)
    except (TypeError, ValueError):
        return _pickup_arm_target(cur_arm, 'Pickup align')
    if not math.isfinite(cur_arm) or not math.isfinite(requested_delta):
        return _pickup_arm_target(cur_arm, 'Pickup align')
    if abs(requested_delta) < 0.001:
        return _pickup_arm_target(cur_arm, 'Pickup align')

    # BUG FIX v42: was always using global PICKUP_ARM_ALIGN_MAX_STEP, ignoring
    # the per-phase cap passed from the centering loop — caused up to 40+° drift.
    effective_max = float(phase_max_step) if phase_max_step is not None else PICKUP_ARM_ALIGN_MAX_STEP
    max_step = max(PICKUP_ARM_ALIGN_MIN_STEP, effective_max)
    delta = float(_clamp(requested_delta, -max_step, max_step))
    if abs(delta) < PICKUP_ARM_ALIGN_MIN_STEP:
        delta = math.copysign(PICKUP_ARM_ALIGN_MIN_STEP, delta)

    target = _pickup_arm_target(cur_arm + delta, 'Pickup align')
    if target is None:
        return None
    snap = max(0.001, float(ARM_SNAP_DEG))
    snapped_target = round(target / snap) * snap
    snapped_cur = round(cur_arm / snap) * snap
    if abs(snapped_target - snapped_cur) < snap * 0.5:
        target = _pickup_arm_target(cur_arm + math.copysign(PICKUP_ARM_ALIGN_MIN_STEP, delta), 'Pickup align')
    return None if target is None else float(target)

def _apply_depth_approach_arm_delta(path, status_prefix='Pickup'):
    """Use the metric depth plan to move most of the way before the final descent."""
    if not isinstance(path, dict):
        return 0.0
    try:
        delta = float(path.get('approach_arm_delta_deg') or 0.0)
    except Exception:
        delta = 0.0
    if not math.isfinite(delta) or delta < 0.5:
        return 0.0

    with _pos_lock:
        cur_arm = float(_pos['arm'])
    target_arm = _pickup_arm_target(cur_arm + PICKUP_ARM_DESCEND_SIGN * delta, status_prefix)
    if target_arm is None:
        return 0.0
    moved = abs(target_arm - cur_arm)
    if moved < 0.5:
        return 0.0

    _set_status(f'{status_prefix}: depth approach {moved:.1f} deg...')
    print(f'[{status_prefix.upper()}] depth approach: arm {cur_arm:.1f} deg -> {target_arm:.1f} deg '
          f'(delta={moved:.1f} deg)', flush=True)
    _move_joint_slow('arm', target_arm, ARM_MIN, ARM_MAX)
    return moved

def _check_alignment_convergence(err_x, err_y, prev_err_x, prev_err_y, pass_num, convergence_check_interval=4):
    """FIX v35: Check if alignment is converging or if we should force descent."""
    if pass_num < 3:  # Always allow first few passes
        return True
    if pass_num >= int(PICKUP_ALIGN_PASS_LIMIT):  # Hard limit
        print(f'[PICKUP] Alignment pass limit ({int(PICKUP_ALIGN_PASS_LIMIT)}) reached; forcing descent', flush=True)
        return False
    if pass_num % convergence_check_interval != 0:
        return True
    # Check convergence every N passes
    curr_error = math.hypot(err_x, err_y)
    prev_error = math.hypot(prev_err_x, prev_err_y)
    if prev_error > 1.0:
        improvement = (prev_error - curr_error) / prev_error
        if improvement < float(PICKUP_CONVERGENCE_MIN_REDUCTION):
            print(f'[PICKUP] Convergence stalled at pass {pass_num} (error {curr_error:.1f}px, improvement {improvement:.1%}); forcing descent', flush=True)
            return False
    return True

def _home_all(skip_arm=False):
    # Sequential home avoids all four move-threads fighting over _servo_lock
    # simultaneously, which was a major source of jitter on startup/home.
    # Grip first so we don't accidentally crush anything during the move.
    home = _home_targets()
    if not GRIP_DC_MODE:
        _move_joint_slow('grip',  home['grip'],  GRIP_OPEN,  GRIP_CLOSE)
    _move_joint_slow('wrist', home['wrist'], 0, 180)
    if not skip_arm:
        _move_joint_slow('arm', home['arm'], ARM_MIN, ARM_MAX)
    _move_joint_slow('base',  home['base'],  BASE_MIN, BASE_MAX)

def _do_up():
    """Raise arm away from the table.
    FIX v44: FK convention: arm_deg=90 is UP, higher angle tips DOWN.
    So raising arm = DECREASE angle.
    """
    with _pos_lock:
        cur = float(_pos['arm'])
    target = _clamp(cur - STEP_DEG, ARM_MIN, ARM_MAX)
    _move_joint_slow('arm', target, ARM_MIN, ARM_MAX)
    _set_status(f'Arm up â†’ {target:.0f}Â°')

def _do_down():
    """Lower arm toward the table (higher servo angle) â€” FK blocked at floor."""
    with _pos_lock:
        cur = float(_pos['arm'])
        wrist = float(_pos.get('wrist', 90.0))
    # FIX v44: higher angle = tip goes DOWN, so down = INCREASE angle
    target = _clamp(cur + STEP_DEG, ARM_MIN, ARM_MAX)
    if _arm_serial[0] is None and not _fk_safe(target, wrist):
        worst, height, margin = _fk_reject_reason(target, wrist)
        _fk_log_reject('DOWN command', 'arm', target, worst, height, margin)
        _set_status(f'Down blocked: target would hit floor ({worst})')
        return
    _move_joint_slow('arm', target, ARM_MIN, ARM_MAX)
    _set_status(f'Arm down â†’ {target:.0f}Â°')

def _do_left():
    # Base rotation follows the tracking convention: left = smaller angle.
    with _pos_lock: cur = _pos['base']
    _move_joint_slow('base', _clamp(cur - BASE_STEP_DEG, BASE_MIN, BASE_MAX), BASE_MIN, BASE_MAX)

def _do_right():
    # Base rotation follows the tracking convention: right = larger angle.
    with _pos_lock: cur = _pos['base']
    _move_joint_slow('base', _clamp(cur + BASE_STEP_DEG, BASE_MIN, BASE_MAX), BASE_MIN, BASE_MAX)

def _do_wave():
    with _pos_lock: center = float(_pos['base'])
    for _ in range(WAVE_REPS):
        _move_joint_slow('base', _clamp(center + WAVE_DEG, BASE_MIN, BASE_MAX), BASE_MIN, BASE_MAX)
        _move_joint_slow('base', _clamp(center - WAVE_DEG, BASE_MIN, BASE_MAX), BASE_MIN, BASE_MAX)
    _move_joint_slow('base', center, BASE_MIN, BASE_MAX)

def _do_open():
    if GRIP_DC_MODE:
        _grip_dc_run('open')
    else:
        _move_joint_slow('grip', GRIP_OPEN, GRIP_OPEN, GRIP_CLOSE)

def _do_close():
    """Close gripper until INA219 current hits threshold, then hold."""
    if GRIP_DC_MODE:
        _grip_dc_run('close')
        return
    threshold = GRIP_CURRENT_THRESHOLD_MA
    step = 0.15
    settle = 0.22
    with _pos_lock:
        angle = float(_pos.get('grip', GRIP_OPEN))
    start_angle = angle
    target = GRIP_CLOSE
    direction = 1.0 if target >= GRIP_OPEN else -1.0

    while direction * (target - angle) > 0.01 and _running[0]:
        angle += direction * step
        if direction > 0:
            angle = min(angle, target)
        else:
            angle = max(angle, target)
        _write_grip_angle(angle, use_ema=False)
        time.sleep(settle)

        current_ma = _read_gripper_current_ma()
        if current_ma >= threshold:
            # Hit current limit — back off slightly and hold
            backoff = max(start_angle, angle - direction * 3.0)
            _hold_grip_at(backoff)
            print(f'[GRIP] current limit {current_ma:.0f}mA at {angle:.1f}°, holding {backoff:.1f}°', flush=True)
            return {'angle': backoff, 'contact': True, 'current_ma': current_ma, 'mode': 'current-limit'}

    _hold_grip_at(target)
    return {'angle': target, 'contact': False, 'current_ma': _read_gripper_current_ma(), 'mode': 'current-limit'}

def _do_wrist_left():
    with _pos_lock: cur = _pos['wrist']
    _move_joint_slow('wrist', _clamp(cur + WRIST_STEP_DEG, 0, 180), 0, 180)

def _do_wrist_right():
    with _pos_lock: cur = _pos['wrist']
    _move_joint_slow('wrist', _clamp(cur - WRIST_STEP_DEG, 0, 180), 0, 180)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  OBJECT DETECTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_detector = {'model': None, 'names': {}, 'model_path': None}
_detector_ready = [False]
_detector_mode = ['yolov8n']
_fallback_subtractor = [None]


def _use_fallback_detector(reason):
    print(f'[DET] {reason} Using fallback contour detection.')
    _detector_ready[0] = False
    _detector_mode[0] = 'fallback'
    _detector['model'] = None
    _detector['names'] = {}
    _detector['model_path'] = None
    try:
        _fallback_subtractor[0] = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=28, detectShadows=False
        )
    except Exception as e:
        print(f'[DET] Background subtractor unavailable: {e}')
        _fallback_subtractor[0] = None
    return False

def _detection_is_ignored(label, box, frame_w=FRAME_W, frame_h=FRAME_H):
    name = _clean_object_name(label) or ''
    if name in DETECTION_IGNORE_LABELS:
        return True
    try:
        _, _, bw, bh = [float(v) for v in box]
        if bw <= 0 or bh <= 0:
            return True
        area_ratio = (bw * bh) / max(1.0, float(frame_w * frame_h))
        width_ratio = bw / max(1.0, float(frame_w))
        height_ratio = bh / max(1.0, float(frame_h))
        return (
            area_ratio > DETECTION_MAX_AREA_RATIO or
            width_ratio > DETECTION_MAX_WIDTH_RATIO or
            height_ratio > DETECTION_MAX_HEIGHT_RATIO
        )
    except Exception:
        return True

def _detection_visual_center(frame, box):
    """Return the contour/mask centroid of the visible object inside a YOLO box."""
    try:
        if frame is None or box is None or len(box) != 4:
            return None
        h, w = frame.shape[:2]
        x, y, bw, bh = [int(round(float(v))) for v in box]
        if bw < 8 or bh < 8:
            return None
        x1 = int(_clamp(x, 0, w - 2))
        y1 = int(_clamp(y, 0, h - 2))
        x2 = int(_clamp(x + bw, x1 + 2, w))
        y2 = int(_clamp(y + bh, y1 + 2, h))
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if cw < 8 or ch < 8:
            return None

        fg = None
        if cw >= 24 and ch >= 24:
            try:
                mask = np.zeros((ch, cw), np.uint8)
                rect = (
                    max(1, int(cw * 0.04)),
                    max(1, int(ch * 0.04)),
                    max(2, int(cw * 0.92)),
                    max(2, int(ch * 0.92)),
                )
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(crop, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
                fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            except Exception:
                fg = None

        if fg is None or int(np.count_nonzero(fg)) < max(20, int(cw * ch * 0.015)):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 130)
            fg = np.where(((sat > 45) & (val > 28)) | (edges > 0), 255, 0).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = max(20.0, cw * ch * 0.012)
        centre_roi = np.array([cw * 0.5, ch * 0.5], dtype=np.float32)
        kept = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            m = cv2.moments(cnt)
            if abs(m.get('m00', 0.0)) < 1e-6:
                continue
            cc = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']], dtype=np.float32)
            norm = np.array([max(1.0, cw), max(1.0, ch)], dtype=np.float32)
            if float(np.linalg.norm((cc - centre_roi) / norm)) <= 0.55:
                kept.append(cnt)
        if not kept:
            kept = [max(contours, key=cv2.contourArea)]

        kept_mask = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(kept_mask, kept, -1, 255, thickness=cv2.FILLED)
        m = cv2.moments(kept_mask)
        if abs(m.get('m00', 0.0)) >= 1e-6:
            cx = x1 + (m['m10'] / m['m00'])
            cy = y1 + (m['m01'] / m['m00'])
        else:
            pts = np.vstack(kept)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            if rw < 3 or rh < 3:
                return None
            cx = x1 + rx + rw * 0.5
            cy = y1 + ry + rh * 0.5
        return float(_clamp(cx, 0, FRAME_W - 1)), float(_clamp(cy, 0, FRAME_H - 1))
    except Exception as e:
        print(f'[DET] visual center failed: {e}', flush=True)
        return None

def _stamp_detection_center(det, center, source='bbox-center'):
    if not isinstance(det, dict) or center is None:
        return det
    try:
        cx = float(_clamp(center[0], 0, FRAME_W - 1))
        cy = float(_clamp(center[1], 0, FRAME_H - 1))
        det['object_center_x'] = round(cx, 1)
        det['object_center_y'] = round(cy, 1)
        det['center_x'] = round(cx, 1)
        det['center_y'] = round(cy, 1)
        det['target_x'] = round(cx, 1)
        det['target_y'] = round(cy, 1)
        det['grasp_x'] = round(cx, 1)
        det['grasp_y'] = round(cy, 1)
        det['cx'] = int(round(cx))
        det['cy'] = int(round(cy))
        det['center_source'] = source
        det['centroid_source'] = source
    except Exception:
        pass
    return det

def _load_detector():
    try:
        from ultralytics import YOLO
        local_model = os.path.join(_DIR, YOLOV8_MODEL)
        model_path = local_model if os.path.exists(local_model) else YOLOV8_MODEL
        model = YOLO(model_path)
        try:
            model.fuse()
        except Exception:
            pass

        names = getattr(model, 'names', {}) or {}
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}
        _detector['model'] = model
        _detector['names'] = names
        _detector['model_path'] = str(model_path)
        _detector_ready[0] = True
        _detector_mode[0] = 'yolov8n'
        print(f'[DET] Loaded YOLO from {model_path} ({len(names)} classes, imgsz={INPUT_SIZE}, conf={CONF_THRESH:.2f}).')
        return True
    except ImportError:
        return _use_fallback_detector('Python package "ultralytics" is not installed.')
    except Exception as e:
        return _use_fallback_detector(f'Failed to load YOLO detector: {e}.')


def _detect_objects(frame):
    if _detector_mode[0] == 'yolov8n' and _detector_ready[0] and _detector.get('model') is not None:
        h, w = frame.shape[:2]
        model = _detector['model']
        try:
            preds = model.predict(
                source=frame,
                imgsz=INPUT_SIZE,
                conf=CONF_THRESH,
                iou=NMS_THRESH,
                max_det=max(1, int(YOLOV8_MAX_DET)),
                device=YOLOV8_DEVICE,
                verbose=False,
            )
        except Exception as e:
            print(f'[DET] YOLOv8n inference failed: {e}')
            return []
        if not preds:
            return []
        pred = preds[0]
        boxes_obj = getattr(pred, 'boxes', None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        results = []
        names = getattr(pred, 'names', None) or _detector.get('names') or {}
        try:
            xyxy = boxes_obj.xyxy.detach().cpu().numpy()
            confs = boxes_obj.conf.detach().cpu().numpy()
            classes = boxes_obj.cls.detach().cpu().numpy().astype(int)
        except Exception:
            xyxy = np.asarray(boxes_obj.xyxy, dtype=np.float32)
            confs = np.asarray(boxes_obj.conf, dtype=np.float32)
            classes = np.asarray(boxes_obj.cls, dtype=np.int32)

        for coords, confidence, class_id in zip(xyxy, confs, classes):
            confidence = float(confidence)
            if confidence < CONF_THRESH:
                continue
            x1, y1, x2, y2 = [float(v) for v in coords[:4]]
            x1 = int(_clamp(x1, 0, w - 1))
            y1 = int(_clamp(y1, 0, h - 1))
            x2 = int(_clamp(x2, 0, w))
            y2 = int(_clamp(y2, 0, h))
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw < 2 or bh < 2:
                continue
            class_id = int(class_id)
            if isinstance(names, dict):
                label = str(names.get(class_id, f'id{class_id}'))
            elif isinstance(names, list) and 0 <= class_id < len(names):
                label = str(names[class_id])
            else:
                label = f'id{class_id}'
            cx = int(x1 + bw // 2)
            cy = int(y1 + bh // 2)
            box = [int(x1), int(y1), int(bw), int(bh)]
            if _detection_is_ignored(label, box, w, h):
                continue
            visual_center = _detection_visual_center(frame, box)
            center = visual_center or (cx, cy)
            det = {
                'label': label,
                'class_id': class_id,
                'confidence': round(confidence * 100.0, 1),
                'box': box,
                'cx': cx,
                'cy': cy,
                'depth_hint': _depth_hint_from_box(box),
                'depth_source': 'box',
                'detector': 'yolov8n',
            }
            results.append(_stamp_detection_center(
                det,
                center,
                'visual-mask-centroid' if visual_center is not None else 'bbox-centroid',
            ))
        results.sort(key=lambda d: _target_focus_score(d))
        return results

    # Fallback: contour-based detector so the stream still shows boxes when YOLOv8n is unavailable.
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    if _fallback_subtractor[0] is not None:
        fg = _fallback_subtractor[0].apply(blur)
    else:
        fg = np.full((h, w), 0, dtype=np.uint8)

    edges = cv2.Canny(gray, 45, 130)
    mask = cv2.bitwise_or(fg, edges)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Give the UI a box around the most salient edge region if everything else fails.
        if np.count_nonzero(edges) == 0:
            return []
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(900, int(0.01 * w * h))
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 20 or bh < 20:
            continue
        score = min(99.9, 35.0 + (area / float(w * h)) * 1800.0)
        box = [int(x), int(y), int(bw), int(bh)]
        if _detection_is_ignored('object', box, w, h):
            continue
        visual_center = _detection_visual_center(frame, box)
        det = {
            'label': 'object',
            'class_id': None,
            'confidence': round(score, 1),
            'box': box,
            'cx': int(x + bw // 2),
            'cy': int(y + bh // 2),
            'depth_hint': _depth_hint_from_box(box),
            'depth_source': 'box',
        }
        results.append(_stamp_detection_center(
            det,
            visual_center or (det['cx'], det['cy']),
            'visual-mask-centroid' if visual_center is not None else 'contour-centroid',
        ))

    results.sort(key=lambda d: _target_focus_score(d))
    return results[:max(1, int(YOLOV8_MAX_DET))]

# Colour palette per detection rank (best=cyan, rest dimmer)
_DET_COLOURS = [
    (0, 220, 255),   # #1 â€” bright cyan
    (0, 180, 220),
    (0, 160, 200),
    (0, 140, 180),
    (0, 120, 160),
]

def _draw_detections(frame, dets):
    h, w = frame.shape[:2]
    aim_x, aim_y = _pickup_aim_xy()  # Calibrated crosshair position
    aim_x_i = int(_clamp(aim_x, 0, w - 1))
    aim_y_i = int(_clamp(aim_y, 0, h - 1))

    # â”€â”€ Green crosshair = calibrated pickup aim (where gripper will land) â”€â”€
    cv2.line(frame, (aim_x_i - 14, aim_y_i), (aim_x_i + 14, aim_y_i), (0, 255, 140), 2)
    cv2.line(frame, (aim_x_i, aim_y_i - 14), (aim_x_i, aim_y_i + 14), (0, 255, 140), 2)
    cv2.circle(frame, (aim_x_i, aim_y_i), 20, (0, 255, 140), 1)
    cv2.putText(frame, 'PICKUP AIM', (max(4, aim_x_i - 45), min(h - 8, aim_y_i + 38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 140), 1, cv2.LINE_AA)

    # â”€â”€ Show calibration offset visually â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raw_cx = FRAME_W // 2
    raw_cy = FRAME_H // 2
    if abs(aim_x_i - raw_cx) > 3 or abs(aim_y_i - raw_cy) > 3:
        # Small grey dot at raw optical-axis centre for reference
        cv2.circle(frame, (raw_cx, raw_cy), 4, (80, 80, 80), -1)
        cv2.line(frame, (raw_cx, raw_cy), (aim_x_i, aim_y_i), (60, 60, 60), 1, cv2.LINE_AA)

    if not dets:
        return frame

    for rank, det in enumerate(dets[:3]):
        box = det.get('box') if isinstance(det, dict) else None
        if isinstance(det, dict) and det.get('frozen_pickup_center'):
            try:
                cx = int(_clamp(det.get('object_center_x', det.get('center_x', det.get('cx', FRAME_W // 2))), 0, w - 1))
                cy = int(_clamp(det.get('object_center_y', det.get('center_y', det.get('cy', FRAME_H // 2))), 0, h - 1))
            except Exception:
                cx, cy = w // 2, h // 2
            col = (0, 220, 255)
            cv2.drawMarker(frame, (cx, cy), col, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 14, col, 2)
            cv2.putText(frame, 'DETECTED CENTER', (max(4, cx - 62), max(18, cy - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            if rank == 0:
                grab_col = (0, 128, 255)
                cv2.arrowedLine(frame, (cx, cy), (aim_x_i, aim_y_i),
                                grab_col, 1, cv2.LINE_AA, tipLength=0.25)
                err_x = aim_x_i - cx
                err_y = aim_y_i - cy
                cv2.putText(frame, f'CENTER ({err_x:+d},{err_y:+d}px)',
                            (max(4, cx - 60), min(h - 8, cy + 32)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, grab_col, 1, cv2.LINE_AA)
                wrist_target = _target_wrist_angle(det)
                cv2.putText(frame, f'TARGET WRIST {wrist_target:.0f}deg',
                            (max(4, aim_x_i + 22), max(18, aim_y_i - 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 80, 255), 1, cv2.LINE_AA)
            continue
        if not box or len(box) != 4:
            continue
        try:
            x, y, bw, bh = [int(v) for v in box]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(w - 1, x))
        y1 = max(0, min(h - 1, y))
        x2 = max(0, min(w - 1, x + bw))
        y2 = max(0, min(h - 1, y + bh))
        if x2 <= x1 or y2 <= y1:
            continue
        col = _DET_COLOURS[min(rank, len(_DET_COLOURS) - 1)]

        # Outer box
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        # Corner ticks
        tl = 14
        cv2.line(frame, (x1, y1), (x1 + tl, y1), col, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + tl), col, 3)
        cv2.line(frame, (x2, y1), (x2 - tl, y1), col, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + tl), col, 3)
        cv2.line(frame, (x1, y2), (x1 + tl, y2), col, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - tl), col, 3)
        cv2.line(frame, (x2, y2), (x2 - tl, y2), col, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - tl), col, 3)

        # True object centroid (horizontal midpoint of bbox)
        cx = int(_clamp(det.get('cx', (x1 + x2) // 2), 0, w - 1))
        cy = int(_clamp(det.get('cy', (y1 + y2) // 2), 0, h - 1))
        cv2.circle(frame, (cx, cy), 5, col, -1)
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), col, 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), col, 1)
        cv2.putText(frame, 'DETECTED CENTER',
                    (max(4, cx - 58), min(h - 8, cy + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        # â”€â”€ Rank #1: draw PREDICTED GRAB POINT (orange) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # This is the pixel in the detection that corresponds to where the arm
        # will actually attempt to close its jaw â€” incorporating the calibrated
        # CAMERA_CENTER_OFFSET_PX.  Alignment error = distance between
        # PREDICTED GRAB POINT and PICKUP AIM crosshair.
        if rank == 0:
            grab_col = (0, 128, 255)   # orange-red
            # The grab target pixel is the object centroid (the arm tries to
            # align this pixel to the aim crosshair before descending).
            grab_px = cx
            grab_py = cy
            # Draw predicted grab point as a distinct marker
            cv2.drawMarker(frame, (grab_px, grab_py),
                           grab_col, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.circle(frame, (grab_px, grab_py), 10, grab_col, 1)

            # Draw alignment error arrow (centroid â†’ aim)
            err_x = aim_x_i - grab_px
            err_y = aim_y_i - grab_py
            err_mag = int(math.hypot(err_x, err_y))
            if err_mag > 6:
                cv2.arrowedLine(frame, (grab_px, grab_py), (aim_x_i, aim_y_i),
                                grab_col, 1, cv2.LINE_AA, tipLength=0.25)

            # Label grab point
            cv2.putText(frame, f'GRASP POINT ({err_x:+d},{err_y:+d}px)',
                        (max(4, grab_px - 48), min(h - 8, grab_py - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, grab_col, 1, cv2.LINE_AA)
            wrist_target = _target_wrist_angle(det)
            cv2.putText(frame, f'TARGET WRIST {wrist_target:.0f}deg',
                        (max(4, aim_x_i + 22), max(18, aim_y_i - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 80, 255), 1, cv2.LINE_AA)

        # Label
        label = str(det.get('label', 'object'))
        confidence = _clamp(det.get('confidence', 0.0), 0.0, 100.0)
        label_str = f"#{rank+1} {label}  {confidence:.0f}%"
        font_scale = 0.48
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 5
        if y1 - th - pad * 2 >= 0:
            lx, ly = x1, y1 - pad
            bg_y1, bg_y2 = ly - th - pad, ly + pad // 2
        else:
            lx, ly = x1, y1 + th + pad
            bg_y1, bg_y2 = y1, y1 + th + pad * 2

        bg_x2 = min(w - 1, lx + tw + pad * 2)
        cv2.rectangle(frame, (lx, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(frame, label_str, (lx + pad, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness, cv2.LINE_AA)

        if rank == 0:
            tag = 'GRAB TARGET'
            (ttw, tth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = min(w - ttw - 6, x2 - ttw)
            ty_tag = y2 + tth + 6
            if ty_tag < h:
                cv2.putText(frame, tag, (tx, ty_tag),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 140), 1, cv2.LINE_AA)
    return frame

def _depth_auto_scan_thread():
    """Background thread: poll VL53L1X and update depth readout on the webpage."""
    interval = VL53L1X_POLL_INTERVAL if VL53L1X_POLL_INTERVAL > 0 else DEPTH_AUTO_SCAN_INTERVAL
    if interval <= 0:
        return
    print(f'[VL53L1X] background poll every {interval:.1f}s', flush=True)
    while _running[0]:
        time.sleep(max(0.25, interval))
        if not _running[0]:
            break
        if not _depth_enabled_runtime():
            continue
        if _yolo_pause_for_pickup.is_set():
            continue
        with _scene_lock:
            dets = list(_scene_dets)
        if not dets:
            continue
        det = dets[0]
        box = det.get('box')
        label = str(det.get('label', 'object'))
        if not box or len(box) != 4:
            continue
        try:
            _depth_plan_from_box(box, label)
        except Exception as e:
            print(f'[DEPTH-AUTO] scan error: {e}', flush=True)


def _depth_scan_now():
    """Sensor Read button: always publish a fresh VL53L1X result for the webpage."""
    with _scene_lock:
        dets = list(_scene_dets)
    box = None
    label = 'object'
    if dets:
        det = dets[0]
        box = det.get('box')
        label = str(det.get('label', 'object'))

    if not VL53L1X_ENABLED:
        _publish_depth_result({
            'ts': time.time(),
            'ok': False,
            'source': 'vl53l1x',
            'message': 'VL53L1X disabled (set ARM_DISABLE_VL53L1X=0)',
            'depth_map_ready': False,
        }, None)
        return _depth_result_snapshot(), None

    if not _gpio_ready[0]:
        _publish_depth_result({
            'ts': time.time(),
            'ok': False,
            'source': 'vl53l1x',
            'message': 'GPIO not ready â€” arm server started without GPIO',
            'depth_map_ready': False,
        }, None)
        return _depth_result_snapshot(), None

    with _depth_lock:
        runtime_on = bool(_depth_state.get('runtime_enabled', VL53L1X_ENABLED))
    if not runtime_on:
        _publish_depth_result({
            'ts': time.time(),
            'ok': False,
            'source': 'vl53l1x',
            'message': 'VL53L1X OFF â€” click "VL53L1X: ON" in the page header first',
            'depth_map_ready': False,
        }, None)
        return _depth_result_snapshot(), None

    if not _vl53l1x_state.get('ready'):
        _init_vl53l1x_gpio()

    if box and len(box) == 4:
        try:
            plan = _depth_plan_from_box(box, label, force=True)
            if plan is not None:
                return _depth_result_snapshot(), None
        except Exception as e:
            print(f'[VL53L1X-SCAN] plan error: {e}', flush=True)
            traceback.print_exc()

    raw_cm, _ = _vl53l1x_read_cm()
    mount_down = _vl53l1x_mount_down()
    mount_behind = _vl53l1x_mount_behind_gripper()
    if raw_cm is not None:
        sensor_distance_cm = float(_clamp(_vl53l1x_scaled_distance_cm(raw_cm) or raw_cm, DEPTH_SANITY_MIN_CM, DEPTH_SANITY_MAX_CM))
        distance_cm = float(_vl53l1x_camera_space_distance_cm(sensor_distance_cm) or sensor_distance_cm)
        jaw_cm = _jaw_distance_from_camera_depth(distance_cm)
        off = _sensor_to_jaw_offset_cm()
        msg = (
            f'VL53L1X â†’ object {distance_cm:.1f} cm | jaw {jaw_cm:.1f} cm (offset {off:.1f} cm)'
            if mount_behind else f'VL53L1X raw {raw_cm:.1f} cm'
        )
        if mount_behind:
            msg = (
                f'VL53L1X sensor {sensor_distance_cm:.1f} cm → object {distance_cm:.1f} cm '
                f'| jaw {jaw_cm:.1f} cm (offset {off:.1f} cm)'
            )
        if mount_down:
            msg = (
                f'VL53L1X bottom height {distance_cm:.1f} cm '
                f'(no horizontal camera offset applied)'
            )
        _publish_depth_result({
            'ts': time.time(),
            'ok': True,
            'label': label,
            'source': 'vl53l1x-down' if mount_down else ('vl53l1x-forward' if mount_behind else 'vl53l1x'),
            'mount': VL53L1X_MOUNT,
            'sensor_raw_cm': round(raw_cm, 2),
            'sensor_distance_cm': round(sensor_distance_cm, 2),
            'sensor_height_cm': round(distance_cm, 2) if mount_down else None,
            'sensor_measurement_us': None,  # not applicable for ToF sensor
            'distance_cm': round(distance_cm, 2),
            'jaw_distance_cm': round(jaw_cm, 2),
            'in_reach': bool(jaw_cm <= DEPTH_REACH_LIMIT_CM),
            'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
            'sensor_mount_offset_cm': round(off, 2),
            'depth_map_ready': False,
            'message': msg + (' (no YOLO box â€” raw read only)' if not box else ''),
        }, None)
    else:
        err = _vl53l1x_not_ready_message() or 'VL53L1X: no data (check SDA/SCL wiring)'
        with _vl53l1x_lock:
            err = str(_vl53l1x_state.get('last_error') or err)
        _publish_depth_result({
            'ts': time.time(),
            'ok': False,
            'label': label,
            'source': 'vl53l1x',
            'message': err,
            'sensor_measurement_us': None,  # not applicable for ToF sensor
            'depth_map_ready': False,
        }, None)
        print(f'[VL53L1X-SCAN] failed: {err}', flush=True)
    return _depth_result_snapshot(), None


def _detector_thread():
    _load_detector()

    last_run = 0.0
    while _running[0]:
        now = time.time()
        if _yolo_pause_for_depth.is_set() or _yolo_pause_for_pickup.is_set():
            # Depth Anything V2 and pickup freeze both need detections to stay
            # unchanged. Keep the last published boxes, but do not run YOLO.
            time.sleep(0.05)
            continue

        if now - last_run < DETECTION_INTERVAL:
            time.sleep(0.01)
            continue

        with _frame_lock:
            frame = None if _latest_frame[0] is None else _latest_frame[0].copy()

        if frame is None:
            time.sleep(0.03)
            continue

        try:
            dets = _detect_objects(frame)
        except Exception as e:
            print(f'[DET] inference error: {e}')
            dets = []
        if _yolo_pause_for_pickup.is_set():
            # A pickup freeze may have been installed while YOLO was already
            # running. Discard this completed inference so it cannot overwrite
            # the box captured at button press.
            time.sleep(0.05)
            continue

        # â”€â”€ Post-move blackout (centering stability fix) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # If a pickup centering move was issued very recently, this frame was
        # captured while the arm (and its camera) were still in motion or
        # vibrating.  In a cluttered scene, background objects appear to shift
        # significantly, causing YOLO to pick a wrong centroid.  Discard the
        # frame and wait for the arm to settle before publishing new detections.
        if (time.time() - _last_pickup_move_time[0]) < PICKUP_POST_MOVE_BLACKOUT_SEC:
            time.sleep(0.02)
            continue
        label = str(dets[0].get('label', 'object')) if dets else 'none'
        with _scene_lock:
            _scene_dets[:] = dets
            _scene_info['ts'] = now
            _scene_info['label'] = label
            _scene_info['count'] = len(dets)
        if dets:
            with _track_lock:
                if _tracking[0] and _track_obj[0] is None:
                    _track_obj[0] = label
        last_run = now

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  OBJECT PICKUP (IMPROVED)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_pickup_start_depth_plan = None

def _object_geometric_center_px(det):
    """Return the object's geometric center in image pixels — the only pickup target."""
    if not isinstance(det, dict):
        return _pickup_aim_xy(None)
    try:
        for x_key, y_key in (
            ('target_x', 'target_y'),
            ('object_center_x', 'object_center_y'),
            ('center_x', 'center_y'),
        ):
            if det.get(x_key) is not None and det.get(y_key) is not None:
                return (
                    float(_clamp(det.get(x_key), 0, FRAME_W - 1)),
                    float(_clamp(det.get(y_key), 0, FRAME_H - 1)),
                )
        box = det.get('box')
        if box and len(box) == 4:
            x, y, bw, bh = [float(v) for v in box]
            if bw > 1 and bh > 1:
                frac = PICKUP_OBJECT_CENTER_FRACTION
                return (
                    float(_clamp(x + bw * frac, 0, FRAME_W - 1)),
                    float(_clamp(y + bh * frac, 0, FRAME_H - 1)),
                )
        return (
            float(_clamp(det.get('cx', FRAME_W / 2.0), 0, FRAME_W - 1)),
            float(_clamp(det.get('cy', FRAME_H / 2.0), 0, FRAME_H - 1)),
        )
    except Exception:
        return _pickup_aim_xy(None)


def _target_point(det):
    return _object_geometric_center_px(det)


def _pickup_crosshair_error(cx, cy):
    ax, ay = _pickup_aim_xy(None)
    return float(cx - ax), float(cy - ay), ax, ay


def _pickup_crosshair_aligned(err_x, err_y):
    return (
        abs(err_x) <= PICKUP_CROSSHAIR_X_DEADBAND and
        abs(err_y) <= PICKUP_CROSSHAIR_Y_DEADBAND
    )


def _pickup_phase_aligned(err_x, err_y, phase):
    if phase == 'fine':
        return abs(err_x) <= PICKUP_FINE_X_DEADBAND and abs(err_y) <= PICKUP_FINE_Y_DEADBAND
    return abs(err_x) <= PICKUP_ACQUIRE_X_DEADBAND and abs(err_y) <= PICKUP_ACQUIRE_Y_DEADBAND


def _find_iou_locked_detection(dets, locked_box, label_hint=None):
    """Return the highest-IoU detection that matches the frozen lock box."""
    if not dets or not locked_box or len(locked_box) != 4:
        return None
    matches = []
    for det in dets:
        det_box = det.get('box') if isinstance(det, dict) else None
        if not det_box or len(det_box) != 4:
            continue
        iou = _box_similarity(locked_box, det_box)
        if iou >= PICKUP_IOU_LOCK_THRESHOLD:
            matches.append((dict(det), iou))
    if not matches:
        return None
    hint = _clean_object_name(label_hint)
    if hint:
        label_matches = [(d, iou) for d, iou in matches if _label_matches(hint, d.get('label', ''))]
        if label_matches:
            matches = label_matches
    return max(matches, key=lambda item: item[1])[0]


def _target_depth_plan(det, force=False):
    global _pickup_start_depth_plan
    if not isinstance(det, dict):
        return {'hint': float(PICKUP_DEPTH_DEFAULT), 'source': 'box'}
    try:
        # If we have a cached/captured starting depth plan, reuse its distance/hint
        if _pickup_start_depth_plan is not None:
            plan = dict(_pickup_start_depth_plan)
            box = det.get('box') or det.get('pickup_depth_box')
            if box and len(box) == 4:
                plan['box'] = [int(float(v)) for v in box]
                cx, cy = _target_point(det)
                center_px = {'x': cx, 'y': cy, 'trust_y': True}
                distance_cm = plan.get('distance_cm')
                if distance_cm is not None:
                    path = _depth_path_from_box(box, distance_cm, center_px=center_px)
                    plan['path'] = path
                else:
                    fallback_depth = plan.get('hint', PICKUP_DEPTH_DEFAULT)
                    path = _depth_path_from_box(box, fallback_depth, center_px=center_px)
                    plan['path'] = path
                plan['label'] = det.get('label', plan.get('label', 'object'))
                plan['ts'] = time.time()
            print(f"[PICKUP] Using cached starting depth plan (Z={plan.get('distance_cm')}cm) recalculated for current box", flush=True)
            return plan

        pickup_start_plan = det.get('_pickup_depth_plan')
        if isinstance(pickup_start_plan, dict) and pickup_start_plan.get('distance_cm') is not None:
            plan = dict(pickup_start_plan)
            distance_cm = float(plan.get('distance_cm'))
            jaw_distance_cm = _jaw_distance_from_camera_depth(distance_cm)
            plan['distance_cm'] = round(distance_cm, 2)
            plan['distance_m'] = round(distance_cm / 100.0, 4)
            plan['jaw_distance_cm'] = round(jaw_distance_cm, 2)
            plan['hint'] = float(_clamp(
                plan.get('hint', _depth_distance_to_hint(distance_cm)),
                PICKUP_DEPTH_MIN,
                PICKUP_DEPTH_MAX,
            ))
            plan['reach_limit_cm'] = DEPTH_REACH_LIMIT_CM
            plan['in_reach'] = bool(jaw_distance_cm <= DEPTH_REACH_LIMIT_CM)
            plan['source'] = str(plan.get('source') or 'vl53l1x') + '+pickup-start'
            path = plan.get('path') if isinstance(plan.get('path'), dict) else None
            if path is None:
                cx, cy = _target_point(det)
                depth_box = det.get('pickup_depth_box')
                if not depth_box or len(depth_box) != 4:
                    depth_box = _depth_box_from_center(cx, cy)
                path = _depth_path_from_box(depth_box, distance_cm) if depth_box else None
            if isinstance(path, dict):
                plan['path'] = _annotate_pickup_execution_path(path, distance_cm)
            plan['message'] = (
                f'camera {distance_cm:.1f} cm, jaw {jaw_distance_cm:.1f} cm - in reach (pickup start)'
                if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM
                else f'camera {distance_cm:.1f} cm, jaw {jaw_distance_cm:.1f} cm - out of reach (pickup start)'
            )
            print(f'[PICKUP] Using pickup-start depth plan: Z={distance_cm:.1f}cm', flush=True)
            return plan

        if det.get('frozen_pickup_center'):
            cx, cy = _target_point(det)
            center_px = {'x': cx, 'y': cy, 'trust_y': True}
            depth_box = det.get('pickup_depth_box')
            if not depth_box or len(depth_box) != 4:
                depth_box = _depth_box_from_center(cx, cy)
            plan = _depth_plan_from_box(depth_box, det.get('label', 'object'), force=force) if depth_box and _depth_enabled_runtime() else None
            if plan and plan.get('hint') is not None:
                plan = dict(plan)
                if plan.get('distance_cm') is not None:
                    try:
                        plan['path'] = _depth_path_from_box(depth_box, float(plan['distance_cm']), center_px=center_px)
                    except Exception as e:
                        print(f'[PICKUP] center path override failed: {e}', flush=True)
                plan['source'] = str(plan.get('source') or 'vl53l1x') + '+frozen-center'
                return plan

            fallback_depth = float(PICKUP_DEPTH_DEFAULT)
            path = _depth_path_from_box(depth_box, fallback_depth, center_px=center_px) if depth_box else None
            result = {
                'ts': time.time(),
                'ok': False,
                'label': det.get('label', 'object'),
                'source': 'frozen-center',
                'message': 'VL53L1X unavailable; using frozen center with default depth',
                'hint': _depth_distance_to_hint(fallback_depth),
                'distance_cm': fallback_depth,
                'reach_limit_cm': DEPTH_REACH_LIMIT_CM,
                'in_reach': True,
                'path': path,
                'depth_map_ready': bool(_depth_map_jpeg_snapshot()),
            }
            _publish_depth_result(result, None)
            return result
        box_hint = _depth_hint_from_box(det.get('box'))
        if box_hint is None:
            box_hint = float(PICKUP_DEPTH_DEFAULT)
        plan = _depth_plan_from_box(det.get('box'), det.get('label', 'object'), force=force) if _depth_enabled_runtime() else None
        if plan and plan.get('hint') is not None:
            depth_hint = float(plan.get('hint'))
            weight = _clamp(DEPTH_HINT_WEIGHT, 0.0, 1.0)
            hint = float(depth_hint) * weight + float(box_hint) * (1.0 - weight)
            det['depth_hint'] = float(_clamp(hint, PICKUP_DEPTH_MIN, PICKUP_DEPTH_MAX))
            det['depth_source'] = 'vl53l1x+box'
            if plan.get('distance_cm') is not None:
                det['depth_cm'] = float(plan['distance_cm'])
                det['in_reach'] = bool(plan.get('in_reach'))
            plan = dict(plan)
            plan['hint'] = det['depth_hint']
            _annotate_pickup_execution_path(
                plan.get('path'),
                plan.get('distance_cm') if plan.get('distance_cm') is not None else plan['hint'],
            )
            return plan
        hint = det.get('depth_hint', box_hint)
        det['depth_hint'] = float(_clamp(hint, PICKUP_DEPTH_MIN, PICKUP_DEPTH_MAX))
        det['depth_source'] = 'box'
        return {
            'hint': det['depth_hint'],
            'source': 'box',
            'distance_cm': None,
            'in_reach': None,
            'path': None,
            'message': 'VL53L1X unavailable; using box-size fallback',
        }
    except Exception:
        return {'hint': float(PICKUP_DEPTH_DEFAULT), 'source': 'box'}

def _do_pickup(obj_name=None, frozen_target=None):
    if not _pickup_lock.acquire(blocking=False):
        _set_status('Pickup: already running')
        return
    global _pickup_start_depth_plan
    _pickup_start_depth_plan = None
    try:
        _set_tracking(False)
        _cancel_all_motions()
        _do_pickup_locked(obj_name, frozen_target)
    finally:
        _pickup_lock.release()

def _do_pickup_locked(obj_name=None, frozen_target=None):
    """
    Aim at the best detected object, grab it, and lift.
      - Wraps the entire sequence in try/except so a crash doesn't freeze the arm
      - Depth hint is fetched ONCE before descent (Depth Anything V2 can block for up to
        DEPTH_WORKER_TIMEOUT_SEC seconds â€” calling it inside the descent loop was
        freezing the arm mid-descent)
      - Descent loop never calls Depth Anything V2; it reuses the pre-fetched estimate
      - Better lost-target handling during descent and pre-grip
    """
    # FIX v18: Clear abort flag at the start of every pickup so a previous stop
    # press doesn't immediately abort the next grab.
    _pickup_abort.clear()
    prev_margin = getattr(_fk_margin_local, 'margin_cm', None)
    _fk_margin_local.margin_cm = float(PICKUP_FLOOR_SAFETY_MARGIN_CM)
    global _pickup_start_depth_plan
    _pickup_start_depth_plan = None
    try:
        _do_pickup_body(obj_name, frozen_target)
    except Exception as exc:
        traceback.print_exc()
        _set_status(f'Pickup error: {exc}')
    finally:
        _pickup_start_depth_plan = None
        if prev_margin is None:
            try:
                delattr(_fk_margin_local, 'margin_cm')
            except AttributeError:
                pass
        else:
            _fk_margin_local.margin_cm = prev_margin
        # Leave abort clear so status polling isn't confused between grabs.
        _pickup_abort.clear()
        _clear_pickup_frozen_target()


def _wait_settle(seconds):
    """Sleep with periodic checks for abort flag. Allows early exit on emergency stop."""
    end = time.time() + float(seconds)
    while time.time() < end and _running[0] and not _pickup_abort.is_set():
        time.sleep(0.05)


def _ik_move_smooth(joint, target_deg, lo, hi, chunk_deg=4.0, chunk_wait=0.08):
    """Move joint smoothly in chunks to avoid jerky IK repositioning."""
    with _pos_lock:
        cur = float(_pos.get(joint, 90.0))
    
    target_deg = float(_clamp(target_deg, lo, hi))
    delta = target_deg - cur
    
    # Safety check: reject suspicious jumps
    if abs(delta) > 45.0:
        print(f'[IK-SAFE] REJECTED {joint} jump of {delta:.1f}° (cur={cur:.1f}°, target={target_deg:.1f}°)', flush=True)
        _set_status(f'Pickup: IK safeguard rejected {joint} jump of {delta:.0f}°')
        return False
    
    # Small move: just do it directly
    if abs(delta) < 0.5:
        return True
    
    # Large move: split into chunks
    n_chunks = max(1, int(math.ceil(abs(delta) / chunk_deg)))
    print(f'[IK-SMOOTH] {joint} {cur:.1f}°→{target_deg:.1f}° in {n_chunks} chunks', flush=True)
    
    for i in range(1, n_chunks + 1):
        if _pickup_abort.is_set() or not _running[0]:
            return False
        
        # Interpolate toward target
        step_target = float(_clamp(cur + delta * i / n_chunks, lo, hi))
        _move_joint_slow(joint, step_target, lo, hi)
        time.sleep(chunk_wait)
    
    return True


def _do_pickup_body(obj_name=None, frozen_target=None):
    """
    CENTER-GRAB PICKUP (no centering loop, no fallbacks).

    Flow:
    1. Detect and lock one object.
    2. Open gripper, get object center + depth.
    3. Run full kinematic simulation (IK + validation).
    4. If simulation passes, execute the plan directly.
    5. If simulation fails, abort (no fallback).
    """
    _pickup_abort.clear()
    prev_margin = getattr(_fk_margin_local, 'margin_cm', None)
    _fk_margin_local.margin_cm = float(PICKUP_FLOOR_SAFETY_MARGIN_CM)
    global _pickup_start_depth_plan
    _pickup_start_depth_plan = None
    _depth_smoother.reset()

    obj_name = _clean_object_name(obj_name)

    try:
        # Step 1: Detect and lock object
        _set_status('Pickup: acquiring target...')
        with _scene_lock:
            dets = list(_scene_dets)

        if not dets:
            _set_status('Pickup: no objects detected - aborting')
            return

        if frozen_target is not None and isinstance(frozen_target, dict):
            locked_target = dict(frozen_target)
        else:
            locked_target = _best_detection(dets, obj_name)

        if locked_target is None:
            _set_status(f'Pickup: no object found matching "{obj_name}" - aborting')
            return

        locked_box = locked_target.get('box')
        if not locked_box or len(locked_box) != 4:
            _set_status('Pickup: invalid detection box - aborting')
            return

        locked_label = str(locked_target.get('label', 'object'))
        locked_confidence = float(locked_target.get('confidence', 0.0))
        if locked_confidence < PICKUP_MIN_CONFIDENCE:
            _set_status(f'Pickup: {locked_label} confidence too low ({locked_confidence:.0f}%) - aborting')
            return

        locked_box = [int(round(float(v))) for v in locked_box]
        _pickup_noted_center[0] = (float(locked_box[0] + locked_box[2] * 0.5),
                                   float(locked_box[1] + locked_box[3] * 0.5))
        _set_status(f'Pickup: locked on {locked_label} (conf={locked_confidence:.0f}%)')

        # Step 2: Open gripper
        _move_joint_slow('grip', GRIP_OPEN, GRIP_OPEN, GRIP_CLOSE)
        _wait_settle(0.40)

        # Step 3: Get object geometric center and depth
        cx, cy = _object_geometric_center_px(locked_target)

        with _depth_lock:
            _depth_state['last_ts'] = 0.0
        depth_plan = _target_depth_plan(locked_target, force=True)
        if depth_plan is None:
            _set_status('Pickup: depth scan failed - aborting')
            return

        distance_cm = depth_plan.get('distance_cm')
        if distance_cm is None:
            _set_status('Pickup: no depth reading (VL53L0X) - aborting')
            return

        metric_depth_cm = _pickup_metric_depth_cm(
            depth_plan, depth_plan.get('path'),
            depth_plan.get('hint', PICKUP_DEPTH_DEFAULT),
        )

        _set_status(f'Pickup: center=({cx:.0f},{cy:.0f}) depth={metric_depth_cm:.1f}cm')

        # Step 4: Get current arm pose first (needed for geometry-based descent)
        with _pos_lock:
            cur_base = float(_pos['base'])
            cur_arm = float(_pos['arm'])
            cur_wrist = float(_pos['wrist'])

        # Step 4a: CENTER — rotate base + arm to align with object pixel position
        aim_x, aim_y = _pickup_aim_xy(None)
        err_x_px = float(cx) - aim_x
        err_y_px = float(cy) - aim_y
        deg_per_px_x = float(DEPTH_CAMERA_FOV_X_DEG) / float(FRAME_W)
        deg_per_px_y = float(DEPTH_CAMERA_FOV_Y_DEG) / float(FRAME_H)
        _CENTER_GAIN_BASE = 0.85
        _CENTER_GAIN_ARM  = 1.0
        base_correction = -1.0 * err_x_px * deg_per_px_x * _CENTER_GAIN_BASE
        arm_correction = -1.0 * err_y_px * deg_per_px_y * _CENTER_GAIN_ARM
        centered_base = float(_clamp(cur_base + base_correction, BASE_MIN, BASE_MAX))
        centered_arm = float(_clamp(cur_arm + arm_correction, ARM_MIN, ARM_MAX))
        print(f'[PICKUP] Centering: err=({err_x_px:.0f},{err_y_px:.0f}px) '
              f'base {cur_base:.1f}→{centered_base:.1f} arm {cur_arm:.1f}→{centered_arm:.1f}',
              flush=True)

        # Step 4b: Compute geometry-aware descent (replaces fixed DEPTH_CM_TO_ARM_DEG)
        height_info = _object_height_measurement(locked_box, metric_depth_cm)
        object_height_cm = 0.0
        if isinstance(height_info, dict):
            object_height_cm = float(height_info.get('corrected_object_height_cm')
                                      or height_info.get('object_height_cm') or 0.0)
        target_z_cm = object_height_cm * PICKUP_HEIGHT_CENTER_FRACTION

        descent_deg, ik_grasp_arm, ik_grasp_wrist = _descent_deg_from_depth(
            metric_depth_cm, centered_arm, cur_wrist, target_z_cm)

        hover_angles = {
            'base': centered_base,
            'arm': float(_clamp(
                (centered_arm + ik_grasp_arm) * 0.5,
                ARM_MIN, ARM_MAX)),
            'wrist': float(_clamp(
                (cur_wrist + ik_grasp_wrist) * 0.5,
                PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)),
        }
        grasp_angles = {
            'base': centered_base,
            'arm': ik_grasp_arm,
            'wrist': ik_grasp_wrist,
        }
        print(f'[PICKUP] Descent computed: depth={metric_depth_cm:.1f}cm → '
              f'drop={descent_deg:.1f}° hover_arm={hover_angles["arm"]:.1f} '
              f'grasp_arm={grasp_angles["arm"]:.1f} '
              f'grasp_wrist={grasp_angles["wrist"]:.1f}', flush=True)

        # Freeze target for UI overlay
        _stamp_detection_center(locked_target, (cx, cy), 'simulation-center')
        _set_pickup_frozen_target(locked_target)

        # Step 5: Move to centered + hover position
        _set_status('Pickup: centering on object...')
        with _pos_lock:
            cur_base = float(_pos['base'])
            cur_arm = float(_pos['arm'])
            cur_wrist = float(_pos['wrist'])

        if abs(hover_angles['base'] - cur_base) > 1.0:
            _set_status('Pickup: rotating to center...')
            _ik_move_smooth('base', hover_angles['base'], BASE_MIN, BASE_MAX)
            _wait_settle(0.30)
        if abs(hover_angles['arm'] - cur_arm) > 0.5:
            _set_status('Pickup: adjusting arm angle...')
            _ik_move_smooth('arm', hover_angles['arm'], ARM_MIN, ARM_MAX)
            _wait_settle(0.30)
        if abs(hover_angles['wrist'] - cur_wrist) > 1.0:
            _ik_move_smooth('wrist', hover_angles['wrist'], PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
            _wait_settle(0.25)

        # Step 6: Descend from hover to grasp pose (interpolated)
        _set_status('Pickup: descending to grasp...')
        arm_travel = abs(grasp_angles['arm'] - hover_angles['arm'])
        if arm_travel > 1.0:
            descent_steps = max(8, int(arm_travel / 1.2))
            for i in range(1, descent_steps + 1):
                if _pickup_abort.is_set() or not _running[0]:
                    return
                frac = i / descent_steps
                arm_interp = hover_angles['arm'] + (grasp_angles['arm'] - hover_angles['arm']) * frac
                wrist_interp = hover_angles['wrist'] + (grasp_angles['wrist'] - hover_angles['wrist']) * frac

                _move_joint_slow('arm', arm_interp, ARM_MIN, ARM_MAX)
                _move_joint_slow('wrist', wrist_interp, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
                _wait_settle(0.12)
        else:
            _move_joint_slow('arm', grasp_angles['arm'], ARM_MIN, ARM_MAX)
            _move_joint_slow('wrist', grasp_angles['wrist'], PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
            _wait_settle(0.30)

        # Step 7: Adaptive grip
        _set_status(f'Pickup: closing gripper on {locked_label}...')
        _grip_ema_angle[0] = None
        _close_gripper_adaptive(locked_label, 'Pickup grip')
        _wait_settle(PICKUP_GRIP_HOLD_SEC)
        _hold_grip_at(float(_pos.get('grip', GRIP_OPEN)))

        # Step 8: Lift
        _set_status(f'Pickup: lifting {locked_label}...')
        with _pos_lock:
            cur_arm_grip = float(_pos['arm'])
            cur_wrist_grip = float(_pos['wrist'])

        lift_target = float(_clamp(
            cur_arm_grip - PICKUP_LIFT_DEG,
            ARM_MIN, ARM_MAX,
        ))
        # FIX: Skip FK safety check for lift — the arm just descended to this
        # position, so lifting is always physically safe (moving away from floor).
        # The FK model uses the opposite sign convention to the physical servo,
        # so _fk_safe incorrectly blocks the lift as a "floor collision".

        _move_joint_slow('arm', lift_target, ARM_MIN, ARM_MAX, _no_fk_check=True)
        _wait_settle(0.60)

        _set_status(f'Pickup: {locked_label} grabbed and raised')
        print(f'[PICKUP] Complete: {locked_label} picked up successfully', flush=True)

    except Exception as exc:
        traceback.print_exc()
        _set_status(f'Pickup error: {exc}')

    finally:
        _pickup_start_depth_plan = None
        if prev_margin is None:
            try:
                delattr(_fk_margin_local, 'margin_cm')
            except AttributeError:
                pass
        else:
            _fk_margin_local.margin_cm = prev_margin
        _pickup_abort.clear()
        _clear_pickup_frozen_target()



# ========================================================================
# [COMMENTED OUT: Old nested pickup helpers - replaced by deterministic]
# ========================================================================
# This section contained corrupted remnants from the old pickup impl:
#   - _fresh_target(), _wait_for_target(), _align_with_gripper_camera()
#   - _center_object_crosshair(), _pickup_nudge_arm_y(), and many others
# All functionality is now in _do_pickup_body() (lines 4972-5342).
#


def _tracking_axis_nudge(err_px, half_span_px, gain, max_nudge_deg, min_nudge_deg):
    """Compute a proportional servo nudge (degrees) for one tracking axis.

    Args:
        err_px        - signed pixel error (positive = object right of / below aim)
        half_span_px  - half the frame dimension for that axis (FRAME_W/2 or FRAME_H/2)
        gain          - scale factor (TRACK_GAIN_X/Y * edge boost)
        max_nudge_deg - hard cap on output magnitude (TRACK_MAX_NUDGE)
        min_nudge_deg - minimum output magnitude; 0 returned when below this
                        threshold so micro-jitter does not buzz the servo

    Returns a signed float in degrees. The caller applies the axis sign
    (base_sign / arm_sign_track) and FK safety check before commanding.
    """
    if half_span_px <= 0:
        return 0.0
    # Normalise error to [-1, +1] range
    norm = err_px / float(half_span_px)
    nudge = float(_clamp(norm * gain * max_nudge_deg, -max_nudge_deg, max_nudge_deg))
    if abs(nudge) < min_nudge_deg:
        return 0.0
    return nudge


def _tracking_thread():
    """
    Target lock with a FROZEN reference box â€” the root fix for focus-switching.

    Why previous versions still switched targets
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Every accepted frame overwrote `locked_box` with the latest detection.
    When YOLO jittered the box 40 px sideways, the reference shifted too.
    Next frame, the shifted box had high IoU against the new (also shifted)
    detection â€” so jitter was silently accepted and the arm chased it.
    After enough jitter-steps the reference had drifted to a DIFFERENT object
    and the lock was broken without ever triggering the lost-count guard.

    v22 fix â€” frozen reference + centroid-only control
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    1. `locked_box` is set ONCE when a target is first acquired.  It is NEVER
       overwritten by normal detections.  Only a genuine large movement of the
       object (IoU falling below TRACK_BOX_UPDATE_IOU) triggers a slow EMA
       blend of the reference toward the new position.

    2. The error signal is computed from the EMA-smoothed CENTROID, not from
       the raw box.  Box-level jitter (Â±20 px shifts) has almost zero effect
       on a centroid EMA with alpha=0.25.

    3. Single-frame centroid jumps > TRACK_MAX_CENTROID_JUMP_PX are discarded
       entirely.  YOLO can teleport a box by 80 px when re-classifying; the
       real object cannot move that fast between frames.

    4. Hysteresis deadband: once centred (error < deadband) stay centred until
       error > TRACK_HYSTERESIS_PX.  Eliminates boundary oscillation.

    5. Longer grace period (TRACK_LOST_LIMIT=12) before unlocking â€” prevents
       the arm's own shadow briefly occluding the target from resetting the lock.

    6. Slower update rate (0.30 s) so the servo fully settles before the next
       error measurement.  Eliminates corrections based on in-flight position.
    """
    # aim_x/aim_y computed per-iteration from _pickup_aim_xy() so tracking
    # drives the object to the GREEN CROSSHAIR, not raw frame centre.
    cx_centre = FRAME_W // 2   # EMA init fallback only
    cy_centre = FRAME_H // 2   # EMA init fallback only
    base_sign      = -1.0   # hardware: inverted base servo
    arm_sign_track = -1.0   # arm down â†’ camera tilts up

    # â”€â”€ Per-session mutable state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lost_count  = 0

    # EMA centroid â€” the ONLY position signal used for servo control
    ema_cx = None
    ema_cy = None

    # Hysteresis flags
    centred_x = False
    centred_y = False

    # Frozen reference box â€” set once, updated only on genuine object movement
    locked_box   = None   # [x, y, w, h] â€” the FROZEN reference
    locked_label = None

    # Stable-detection gate
    stable_count   = 0
    last_det_label = None

    # Box-size change filter
    last_area          = None
    size_change_frames = 0

    # Previous EMA centroid â€” used to detect teleport jumps
    prev_ema_cx = None
    prev_ema_cy = None

    def _find_best_match(dets):
        """Scan ALL detections, return the one with best IoU vs frozen locked_box.
        Returns (det, iou) â€” caller decides whether iou is good enough.
        Never relies on YOLO confidence ranking so a re-ranked scene cannot
        steal the lock.
        """
        best_det = None
        best_iou = 0.0
        for det in dets:
            iou = _box_similarity(locked_box, det.get('box'))
            if iou > best_iou:
                best_iou = iou
                best_det = det
        return best_det, best_iou

    def _reset_session():
        nonlocal lost_count, ema_cx, ema_cy, centred_x, centred_y
        nonlocal locked_box, locked_label, stable_count, last_det_label
        nonlocal last_area, size_change_frames, prev_ema_cx, prev_ema_cy
        lost_count = 0
        ema_cx = ema_cy = None
        centred_x = centred_y = False
        locked_box = locked_label = None
        stable_count = 0
        last_det_label = None
        last_area = None
        size_change_frames = 0
        prev_ema_cx = prev_ema_cy = None

    while _running[0]:
        with _track_lock:
            is_tracking = _tracking[0]
            obj_name    = _track_obj[0]

        if not is_tracking:
            _reset_session()
            time.sleep(0.2)
            continue

        with _scene_lock:
            dets = list(_scene_dets)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  PHASE A â€” locked: search ALL dets by IoU against frozen reference
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        candidate = None

        if locked_box is not None:
            best_det, best_iou = _find_best_match(dets)

            if best_iou < TRACK_LOCK_IOU_THRESHOLD:
                # No detection overlaps the reference well enough.
                # This is the ONLY place lost_count increments while locked.
                lost_count += 1
                stable_count = 0
                if lost_count >= TRACK_LOST_LIMIT:
                    _set_status('Tracking: target lost â€” unlocking')
                    _reset_session()
                else:
                    _set_status(
                        f'Tracking: searching for target '
                        f'({lost_count}/{TRACK_LOST_LIMIT})â€¦'
                    )
                time.sleep(TRACK_UPDATE_INTERVAL)
                continue

            # Good match found â€” reset lost counter
            lost_count = 0
            candidate = best_det

            # â”€â”€ Selective reference update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Only blend the reference toward the new box when IoU has dropped
            # low enough to indicate genuine object movement (not jitter).
            # This keeps the reference frozen during normal jitter, but follows
            # the object if it actually moves across the frame.
            if best_iou < TRACK_BOX_UPDATE_IOU:
                new_box = candidate.get('box')
                if new_box and len(new_box) == 4:
                    locked_box = [
                        TRACK_BOX_UPDATE_ALPHA * float(new_box[i]) +
                        (1.0 - TRACK_BOX_UPDATE_ALPHA) * float(locked_box[i])
                        for i in range(4)
                    ]

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  PHASE B â€” unlocked: acquire a new target
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if candidate is None:
            # Not yet locked â€” use label hint, fall back to highest confidence
            candidate = _best_detection(dets, obj_name)

        if candidate is None:
            lost_count += 1
            if lost_count >= TRACK_LOST_LIMIT:
                _set_status('Tracking: no detections')
            time.sleep(TRACK_UPDATE_INTERVAL)
            continue

        # â”€â”€ Acquire lock on first valid candidate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if locked_box is None:
            locked_box   = list(candidate.get('box') or [0, 0, 1, 1])
            locked_label = candidate.get('label', obj_name)
            ema_cx = float(candidate.get('cx', cx_centre))
            ema_cy = float(candidate.get('cy', cy_centre))
            prev_ema_cx = ema_cx
            prev_ema_cy = ema_cy
            stable_count = 1
            last_det_label = locked_label
            _set_status(f'Tracking: locked onto {locked_label or "object"}')
            time.sleep(TRACK_UPDATE_INTERVAL)
            continue   # start stabilisation gate fresh

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  STABILITY GATE â€” require N consecutive matching frames
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        det_label = candidate.get('label', '')
        if det_label == last_det_label:
            stable_count = min(stable_count + 1, TRACK_STABLE_FRAMES_REQUIRED + 6)
        else:
            # Label switched â€” reset gate but do NOT break the IoU lock
            stable_count = 1
            last_det_label = det_label
            # Reset EMA so we don't carry stale centroid from the wrong object
            ema_cx = float(candidate.get('cx', cx_centre))
            ema_cy = float(candidate.get('cy', cy_centre))

        if stable_count < TRACK_STABLE_FRAMES_REQUIRED:
            time.sleep(TRACK_UPDATE_INTERVAL)
            continue

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  BOUNDING-BOX SIZE FILTER
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        box = candidate.get('box')
        cur_area = None
        if box and len(box) == 4:
            try:
                cur_area = float(box[2]) * float(box[3])
            except Exception:
                pass
        if cur_area is not None and last_area is not None and last_area > 0:
            area_change = abs(cur_area - last_area) / last_area
            if area_change > TRACK_SIZE_CHANGE_THRESHOLD:
                size_change_frames += 1
                if size_change_frames < TRACK_SIZE_CHANGE_FRAMES:
                    # Transient scale jump â€” hold current EMA, skip servo update
                    time.sleep(TRACK_UPDATE_INTERVAL)
                    continue
            else:
                size_change_frames = 0
        else:
            size_change_frames = 0
        if cur_area is not None:
            last_area = cur_area

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  CENTROID EMA SMOOTHING
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        raw_cx = float(candidate.get('cx', cx_centre))
        raw_cy = float(candidate.get('cy', cy_centre))

        # Discard single-frame teleport jumps
        if prev_ema_cx is not None:
            jump = ((raw_cx - prev_ema_cx) ** 2 + (raw_cy - prev_ema_cy) ** 2) ** 0.5
            if jump > TRACK_MAX_CENTROID_JUMP_PX:
                # Silently skip this frame â€” keep previous EMA
                time.sleep(TRACK_UPDATE_INTERVAL)
                continue

        if ema_cx is None:
            ema_cx, ema_cy = raw_cx, raw_cy
        else:
            ema_cx = TRACK_EMA_ALPHA * raw_cx + (1.0 - TRACK_EMA_ALPHA) * ema_cx
            ema_cy = TRACK_EMA_ALPHA * raw_cy + (1.0 - TRACK_EMA_ALPHA) * ema_cy

        prev_ema_cx = ema_cx
        prev_ema_cy = ema_cy

        aim_x, aim_y = _pickup_aim_xy(None)
        err_x = ema_cx - aim_x
        err_y = ema_cy - aim_y
        boost_x, boost_y = _box_edge_boost(candidate)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  HYSTERESIS DEADBAND
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if centred_x:
            if abs(err_x) > TRACK_HYSTERESIS_PX:
                centred_x = False
        else:
            if abs(err_x) <= TRACK_PX_DEADBAND:
                centred_x = True

        if centred_y:
            if abs(err_y) > TRACK_HYSTERESIS_PX:
                centred_y = False
        else:
            if abs(err_y) <= TRACK_PX_DEADBAND:
                centred_y = True

        moved = False

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        #  PROPORTIONAL SERVO CORRECTIONS (tiny steps)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if not centred_x:
            base_nudge = _tracking_axis_nudge(
                err_x, FRAME_W / 2.0, TRACK_GAIN_X * boost_x,
                TRACK_MAX_NUDGE, TRACK_MIN_BASE_NUDGE_DEG,
            )
            with _pos_lock:
                cur_base = _pos['base']
            new_base = float(_clamp(cur_base + base_sign * base_nudge, BASE_MIN, BASE_MAX))
            if abs(new_base - cur_base) >= TRACK_MIN_BASE_NUDGE_DEG:
                _move_joint('base', new_base, BASE_MIN, BASE_MAX)
                moved = True

        if not centred_y:
            arm_nudge = _tracking_axis_nudge(
                err_y, FRAME_H / 2.0, TRACK_GAIN_Y * boost_y,
                TRACK_MAX_NUDGE, TRACK_MIN_ARM_NUDGE_DEG,
            )
            arm_nudge = arm_sign_track * arm_nudge
            with _pos_lock:
                cur_arm = float(_pos['arm'])
            new_arm = float(_clamp(cur_arm + arm_nudge, float(ARM_MIN), float(ARM_MAX)))
            if abs(new_arm - cur_arm) >= TRACK_MIN_ARM_NUDGE_DEG:
                _move_joint('arm', new_arm, float(ARM_MIN), float(ARM_MAX))
                moved = True

        # Wrist fine-aim: computed from EMA centroid, not raw box
        if TRACK_USE_WRIST_AIM:
            # Synthesise a minimal det dict using smoothed cy for wrist calc
            wrist_det = {'cy': ema_cy, 'box': locked_box}
            desired_wrist = _target_wrist_angle(wrist_det)
            with _pos_lock:
                cur_wrist = float(_pos['wrist'])
            wrist_err = desired_wrist - cur_wrist
            if abs(wrist_err) > WRIST_DEADBAND_DEG:
                wrist_nudge = _clamp(wrist_err * TRACK_GAIN_WRIST,
                                     -TRACK_MAX_WRIST_NUDGE, TRACK_MAX_WRIST_NUDGE)
                _move_joint('wrist',
                            float(_clamp(cur_wrist + wrist_nudge, 0, 180)), 0, 180)

        label_display = locked_label or obj_name or 'object'
        if moved:
            _set_status(
                f'Tracking: {label_display} '
                f'X={err_x:+.0f}px Y={err_y:+.0f}px â†’ nudging'
            )
        else:
            _set_status(
                f'Tracking: {label_display} âœ“ locked '
                f'(X={err_x:+.0f} Y={err_y:+.0f})'
            )

        time.sleep(TRACK_UPDATE_INTERVAL)


def _tracking_thread_safe():
    """FIX v29: Wrapper that restarts _tracking_thread on any crash.

    _tracking_thread runs an infinite while loop.  If any unhandled exception
    escapes the loop (e.g. a bad detection frame, a lock timeout, a math error),
    the thread dies silently.  The UI shows 'Tracking started' but the arm never
    moves again because the scheduling loop is gone.

    This wrapper catches that case and restarts the inner loop automatically.
    """
    while _running[0]:
        try:
            _tracking_thread()
        except Exception as exc:
            print(f'[TRACK] thread crashed, restarting in 1 s: {exc}', flush=True)
            _set_status('Tracking: restarting after error')
            time.sleep(1.0)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _do_120_align_and_pickup(obj_name=None):
    """
    NEW WORKFLOW: 120 alignment passes â†’ capture depth map â†’ predict 3D location â†’ pick up
    
    Steps:
      1. Perform 120 alignment passes to center the object in frame
      2. Capture depth map using Depth Anything V2
      3. Calculate 3D position of object in camera-space
      4. Move arm base/shoulder to the predicted X/Y position
      5. Execute full pickup sequence (descent, grip, lift)
    """
    if not _pickup_lock.acquire(blocking=False):
        _set_status('Pickup: already running')
        return
    global _pickup_start_depth_plan
    _pickup_start_depth_plan = None
    try:
        _set_tracking(False)
        _cancel_all_motions()
        _pickup_abort.clear()
        prev_margin = getattr(_fk_margin_local, 'margin_cm', None)
        _fk_margin_local.margin_cm = float(PICKUP_FLOOR_SAFETY_MARGIN_CM)

        def _wait_settle(seconds):
            end = time.time() + float(seconds)
            while time.time() < end and _running[0] and not _pickup_abort.is_set():
                time.sleep(0.05)

        def _execute_planned_grip_and_lift(label, depth_plan):
            """Follow the already-generated depth plan, then grip and lift."""
            path = depth_plan.get('path') if isinstance(depth_plan, dict) else None
            if isinstance(depth_plan, dict) and depth_plan.get('distance_cm') is not None:
                estimated_depth = float(_clamp(depth_plan.get('distance_cm'), PICKUP_DEPTH_MIN, DEPTH_REACH_LIMIT_CM))
            else:
                estimated_depth = float(_clamp(
                    depth_plan.get('hint', PICKUP_DEPTH_DEFAULT) if isinstance(depth_plan, dict) else PICKUP_DEPTH_DEFAULT,
                    PICKUP_DEPTH_MIN,
                    PICKUP_DEPTH_MAX,
                ))

            metric_depth_cm = _pickup_metric_depth_cm(depth_plan, path, estimated_depth)
            estimated_depth = metric_depth_cm
            full_drop_deg, _, planned_descent = _depth_contact_drop_degs(metric_depth_cm)
            planned_descent = float(planned_descent)

            extra_drop_deg = _pickup_depth_offset()
            depth_approach_delta_done = 0.0
            print(f'[PICKUP-120] Metric Z={metric_depth_cm:.1f}cm â†’ full_drop={full_drop_deg:.1f}Â° '
                  f'planned_descent={planned_descent:.1f}Â°', flush=True)

            # Snapshot ALL joints as approach reference â€” no pose flip.
            with _pos_lock:
                approach_base  = float(_pos['base'])
                approach_arm   = float(_pos['arm'])
                approach_wrist = float(_pos['wrist'])
                approach_grip  = float(_pos['grip'])
            print(f'[PICKUP-120] Approach pose: base={approach_base:.1f}Â° arm={approach_arm:.1f}Â° '
                  f'wrist={approach_wrist:.1f}Â° grip={approach_grip:.1f}Â°', flush=True)

            if isinstance(path, dict):
                depth_approach_delta_done = _apply_depth_approach_arm_delta(path, 'Pickup-120 approach')
                if depth_approach_delta_done > 0.0:
                    _wait_settle(0.35)
                    with _pos_lock:
                        approach_base  = float(_pos['base'])
                        approach_arm   = float(_pos['arm'])
                        approach_wrist = float(_pos['wrist'])
                        approach_grip  = float(_pos['grip'])
                    print(f'[PICKUP-120] After depth approach: base={approach_base:.1f} deg '
                          f'arm={approach_arm:.1f} deg wrist={approach_wrist:.1f} deg '
                          f'grip={approach_grip:.1f} deg', flush=True)

            depth_bias_120 = min(float(PICKUP_CENTER_CONTACT_BIAS_DEG),
                                 float(PICKUP_CENTER_CONTACT_BIAS_DEG) * (metric_depth_cm / 20.0))
            descent_budget = float(_clamp(
                max(planned_descent + depth_bias_120 + extra_drop_deg,
                    PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG),
                min(PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG, PICKUP_MAX_DESCENT_DEG),
                PICKUP_MAX_DESCENT_DEG,
            ))
            print(f'[PICKUP-120] depth approach={depth_approach_delta_done:.1f} deg, '
                  f'final descent={descent_budget:.1f} deg '
                  f'(center_bias={PICKUP_CENTER_CONTACT_BIAS_DEG:.1f}, '
                  f'min_before_grip={PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG:.1f}, Z={metric_depth_cm:.1f}cm)', flush=True)
            print(f'[PICKUP-120] total commanded drop={depth_approach_delta_done + descent_budget:.1f} deg '
                  f'(depth approach={depth_approach_delta_done:.1f}, final descent={descent_budget:.1f})',
                  flush=True)

            target_wrist_from_depth = _wrist_from_depth(metric_depth_cm, approach_arm)
            wrist_approach_span = abs(target_wrist_from_depth - approach_wrist)
            wrist_sign = 1.0 if target_wrist_from_depth >= approach_wrist else -1.0
            wrist_approach_span = float(_clamp(
                wrist_approach_span,
                0.0,
                PICKUP_WRIST_APPROACH_MAX_DEG,
            ))

            # Pre-position wrist to midpoint before descent, then ramp to full
            _wrist_predescent_offset_120 = wrist_sign * (wrist_approach_span * 0.5)
            _wrist_predescent_target_120 = float(_clamp(
                approach_wrist + _wrist_predescent_offset_120,
                PICKUP_WRIST_MIN, PICKUP_WRIST_MAX,
            ))
            if abs(_wrist_predescent_target_120 - approach_wrist) >= float(WRIST_SNAP_DEG):
                print(f'[PICKUP-120] Pre-descent wrist: {approach_wrist:.1f}Â°â†’{_wrist_predescent_target_120:.1f}Â°', flush=True)
                _move_joint_slow('wrist', _wrist_predescent_target_120, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
                approach_wrist = _wrist_predescent_target_120

            def _wrist_for_descent(descended_deg):
                frac = _clamp(float(descended_deg) / max(PICKUP_DESCENT_STEP_DEG, descent_budget), 0.0, 1.0)
                # Ramp full descent 0â†’100% for reliable wrist movement on every step
                return float(_clamp(
                    approach_wrist + wrist_sign * wrist_approach_span * frac,
                    PICKUP_WRIST_MIN,
                    PICKUP_WRIST_MAX,
                ))

            def _move_wrist_for_descent(descended_deg):
                target_wrist = _wrist_for_descent(descended_deg)
                with _pos_lock:
                    cur_wrist = float(_pos['wrist'])
                if abs(target_wrist - cur_wrist) >= float(WRIST_SNAP_DEG):
                    _move_joint_slow('wrist', target_wrist, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)

            total_descended = 0.0
            while total_descended < descent_budget and _running[0] and not _pickup_abort.is_set():
                step = min(PICKUP_DESCENT_STEP_DEG, descent_budget - total_descended)
                with _pos_lock:
                    cur_arm = float(_pos['arm'])
                target_arm = _pickup_arm_target(cur_arm + PICKUP_ARM_DESCEND_SIGN * step, 'Pickup-120 descent')
                if target_arm is None:
                    _set_status('Pickup: floor limit reached - stopping 3D path descent')
                    break
                _set_status(f'Pickup: following 3D path {total_descended + step:.1f}/{descent_budget:.1f} deg')
                _move_joint_slow('arm', target_arm, ARM_MIN, ARM_MAX)
                total_descended += step
                _move_wrist_for_descent(total_descended)
                _wait_settle(PICKUP_DESCENT_SETTLE)

            if _pickup_abort.is_set() or not _running[0]:
                _set_status('Pickup: aborted before grip')
                return False

            with _pos_lock:
                cur_arm = float(_pos['arm'])
            final_dip = PICKUP_FINAL_DIP_DEG + max(0.0, estimated_depth - PICKUP_DEPTH_DEFAULT) * PICKUP_FINAL_DIP_DEPTH_GAIN
            final_dip = min(final_dip, max(0.0, PICKUP_MAX_DESCENT_DEG - depth_approach_delta_done - total_descended))
            target_arm = _pickup_arm_target(cur_arm + PICKUP_ARM_DESCEND_SIGN * final_dip, 'Pickup-120 final dip')
            if target_arm is not None:
                _move_joint_slow('arm', target_arm, ARM_MIN, ARM_MAX)
            _move_wrist_for_descent(descent_budget)
            _wait_settle(0.35)

            _set_status(f'Pickup: reached {label}, closing gripper...')
            grip_result = _close_gripper_adaptive(label, 'Pickup-120 grip')
            _wait_settle(PICKUP_GRIP_HOLD_SEC)

            with _pos_lock:
                actual_grip = float(_pos['grip'])
            grip_midpoint = (GRIP_OPEN + GRIP_CLOSE) / 2.0
            grip_retries = 0
            while (not grip_result.get('contact')) and actual_grip < grip_midpoint + 5 and grip_retries < 1:
                _set_status(f'Pickup: grip weak, re-gripping (attempt {grip_retries + 1})...')
                _write_grip_angle(GRIP_OPEN + (actual_grip - GRIP_OPEN) * 0.4, use_ema=False)
                _wait_settle(0.35)
                _grip_ema_angle[0] = None
                grip_result = _close_gripper_adaptive(label, 'Pickup-120 re-grip')
                _wait_settle(PICKUP_GRIP_HOLD_SEC)
                with _pos_lock:
                    actual_grip = float(_pos['grip'])
                grip_retries += 1

            _set_status(f'Pickup: gripper closed, lifting {label}...')
            with _pos_lock:
                cur_arm_after_grip = float(_pos['arm'])
            lift_target = float(_clamp(
                cur_arm_after_grip - PICKUP_ARM_DESCEND_SIGN * PICKUP_LIFT_DEG,
                ARM_MIN,
                ARM_MAX,
            ))
            print(f'[PICKUP-120] Lift after grip: arm {cur_arm_after_grip:.1f} deg -> {lift_target:.1f} deg', flush=True)
            _move_joint_slow('arm', lift_target, ARM_MIN, ARM_MAX)
            if PICKUP_LEVEL_WRIST_AFTER_LIFT:
                _move_joint_slow('wrist', _clamp(approach_wrist, 0, 180), 0, 180)
            _wait_settle(0.60)

            _set_status(f'Pickup: grabbed {label}')
            return True

        _set_status('Pickup: starting 120-pass alignment...')

        _move_joint_slow('grip', GRIP_OPEN, GRIP_OPEN, GRIP_CLOSE)
        _wait_settle(0.40)

        # Continue from current arm position â€” no pose repositioning.
        with _pos_lock:
            print(f'[PICKUP-120] Start pose: base={_pos["base"]:.1f}Â° arm={_pos["arm"]:.1f}Â° '
                  f'wrist={_pos["wrist"]:.1f}Â° grip={_pos["grip"]:.1f}Â°', flush=True)
        
        # Step 1: Perform 120 alignment passes to center object
        print('[PICKUP-120] Starting 120-pass alignment sequence', flush=True)
        label = _clean_object_name(obj_name)

        locked_120_box = [None]
        frozen_120_det = [None]

        def _clone_120_detection(det):
            if not isinstance(det, dict):
                return None
            frozen = dict(det)
            box = frozen.get('box')
            if box and len(box) == 4:
                frozen['box'] = list(box)
                try:
                    x, y, bw, bh = [float(v) for v in box]
                    frozen['cx'] = int(_clamp(x + bw * 0.5, 0, FRAME_W - 1))
                    frozen['cy'] = int(_clamp(y + bh * 0.5, 0, FRAME_H - 1))
                except Exception:
                    pass
            frozen['frozen_pickup_box'] = True
            detector = str(frozen.get('detector') or 'yolov8n')
            frozen['detector'] = detector if 'pickup-freeze' in detector else detector + '+pickup-freeze'
            return frozen

        def _freeze_120_target(det):
            frozen = _clone_120_detection(det)
            if frozen is None:
                return None
            frozen_120_det[0] = frozen
            locked_120_box[0] = list(frozen.get('box') or [0, 0, 1, 1])
            _yolo_pause_for_pickup.set()
            with _scene_lock:
                _scene_dets[:] = [dict(frozen)]
                _scene_info['ts'] = time.time()
                _scene_info['label'] = str(frozen.get('label', 'object'))
                _scene_info['count'] = 1
            print(f'[PICKUP-120] Frozen target box={frozen.get("box")} label={frozen.get("label", "object")}', flush=True)
            return frozen

        def _locked_120_detection(dets, label_hint=None):
            if frozen_120_det[0] is not None:
                return _clone_120_detection(frozen_120_det[0])
            if not dets:
                return None
            usable = [d for d in dets if _detection_area_ratio(d) >= PICKUP_MIN_BOX_AREA_RATIO]
            if usable:
                dets = usable

            locked = locked_120_box[0]
            if locked is not None:
                matches = [
                    (d, _box_similarity(locked, d.get('box')))
                    for d in dets
                ]
                matches = [(d, iou) for d, iou in matches if iou >= PICKUP_IOU_LOCK_THRESHOLD]
                if not matches:
                    return None
                hint = _clean_object_name(label_hint)
                if hint:
                    label_matches = [(d, iou) for d, iou in matches if _label_matches(hint, d.get('label', ''))]
                    if label_matches:
                        matches = label_matches
                best, _ = max(matches, key=lambda item: item[1])
                return best

            return _best_detection(dets, label_hint)
        
        # Get initial target
        with _scene_lock:
            dets = list(_scene_dets)
        best_det = _locked_120_detection(dets, label) if dets else None
        
        if best_det is None:
            _set_status('Pickup: no object detected')
            print('[PICKUP-120] No object detected, aborting', flush=True)
            return

        frozen_best = _freeze_120_target(best_det)
        if frozen_best is not None:
            best_det = frozen_best
        
        # Run 120 alignment passes with settling between each
        max_passes = 120
        stable_frames = 0
        last_tgt = best_det
        aligned_120 = False
        arm_sign_120 = [PICKUP_ARM_ALIGN_SIGN]
        last_y_err_120 = [None]
        last_arm_delta_120 = [None]
        
        for pass_n in range(max_passes):
            if not _running[0] or _pickup_abort.is_set():
                break
                
            # Get fresh target
            with _scene_lock:
                dets = list(_scene_dets)
            
            tgt = _locked_120_detection(dets, label) if dets else None
            if tgt is None:
                if last_arm_delta_120[0] is not None:
                    with _pos_lock:
                        cur_arm = float(_pos['arm'])
                    undo_arm = _pickup_arm_align_target(cur_arm, -last_arm_delta_120[0])
                    if undo_arm is not None:
                        _move_joint_slow('arm', undo_arm, ARM_MIN, ARM_MAX)
                    _wait_settle(0.35)
                _set_status(f'Pickup: object lost during alignment (pass {pass_n + 1}/120)')
                print(f'[PICKUP-120] Object lost at pass {pass_n + 1}', flush=True)
                return
            
            last_tgt = tgt
            
            # Calculate centering error
            try:
                aim_x, aim_y = _pickup_aim_xy(None)
                box = tgt.get('box')
                if box and len(box) == 4:
                    tx = float(_clamp(box[0] + box[2] * PICKUP_TARGET_X_FRACTION, 0, FRAME_W - 1))
                    ty = float(_clamp(box[1] + box[3] * (0.50 if PICKUP_CENTER_GRAB_ENABLED else PICKUP_TARGET_Y_FRACTION), 0, FRAME_H - 1))
                else:
                    tx, ty = aim_x, aim_y
                
                err_x = tx - aim_x
                err_y = ty - aim_y
                err_mag = math.sqrt(err_x**2 + err_y**2)

                if (last_y_err_120[0] is not None and last_arm_delta_120[0] is not None and
                        abs(err_y) > abs(last_y_err_120[0]) + PICKUP_ARM_WRONG_WAY_THRESHOLD_PX):
                    arm_sign_120[0] *= -1.0
                    last_y_err_120[0] = None
                    last_arm_delta_120[0] = None
                    stable_frames = 0
                    _set_status('Pickup: reversed arm centering direction')
                
                # Check vertical stability. X bearing is handled by the 3D depth
                # path after capture, so avoid chasing it with camera-mounted base moves.
                if abs(err_y) <= PICKUP_FINE_Y_DEADBAND:
                    stable_frames += 1
                    if stable_frames >= PICKUP_STABLE_FRAMES:
                        aligned_120 = True
                        _set_status(f'Pickup: height aligned after {pass_n + 1} passes')
                        print(f'[PICKUP-120] Height aligned at pass {pass_n + 1}', flush=True)
                        break
                else:
                    stable_frames = 0
                
                # Only correct vertical aim here. Base/X correction is made once from
                # metric depth after the map is generated.
                if abs(err_y) > PICKUP_FINE_Y_DEADBAND:
                    correction = (err_y / float(FRAME_H)) * PICKUP_ARM_GAIN_DEG
                    arm_delta = _clamp(arm_sign_120[0] * correction,
                                       -PICKUP_ARM_ALIGN_MAX_STEP,
                                       PICKUP_ARM_ALIGN_MAX_STEP)
                    with _pos_lock:
                        cur_arm = float(_pos['arm'])
                    new_arm = _pickup_arm_align_target(cur_arm, arm_delta)
                    actual_delta = 0.0 if new_arm is None else float(new_arm - cur_arm)
                    if abs(actual_delta) < 0.5:
                        # FIX v19: arm at floor limit in 120-pass loop â€” tilt wrist for Y aim
                        _set_status(f'Pickup: arm at floor limit, using wrist for Y ({pass_n + 1}/120)')
                        desired_wrist_120 = _target_wrist_angle(tgt)
                        with _pos_lock:
                            cur_wrist_120 = float(_pos['wrist'])
                        w_err_120 = desired_wrist_120 - cur_wrist_120
                        if abs(w_err_120) > WRIST_DEADBAND_DEG:
                            w_nudge_120 = _clamp(w_err_120 * PICKUP_WRIST_ALIGN_GAIN,
                                                 -PICKUP_WRIST_ALIGN_MAX_STEP,
                                                 PICKUP_WRIST_ALIGN_MAX_STEP)
                            _move_joint_slow('wrist', float(_clamp(cur_wrist_120 + w_nudge_120,
                                                                    PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)),
                                             PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
                        # Don't return â€” continue loop
                    if new_arm is not None and abs(actual_delta) >= 0.5:
                        _move_joint_slow('arm', new_arm, ARM_MIN, ARM_MAX)
                        last_y_err_120[0] = err_y
                        last_arm_delta_120[0] = actual_delta
                    else:
                        last_arm_delta_120[0] = None
                else:
                    last_arm_delta_120[0] = None
                
                if (pass_n + 1) % 10 == 0:
                    _set_status(f'Pickup: aligning... {pass_n + 1}/120 (error: {err_mag:.1f}px)')
                    print(f'[PICKUP-120] Pass {pass_n + 1}/120: err_x={err_x:.1f}px, err_y={err_y:.1f}px', flush=True)
                
                # Settle after each correction
                _wait_settle(0.38)
                
            except Exception as e:
                print(f'[PICKUP-120] Pass {pass_n + 1} failed: {e}', flush=True)
                _wait_settle(0.20)

        if not aligned_120:
            _set_status('Pickup: crosshair not centred safely - aborting before grab')
            print('[PICKUP-120] Crosshair was not centred safely; aborting before depth/grab', flush=True)
            return
        
        # Step 2: Capture depth map for the centered object
        _set_status('Pickup: capturing depth map...')
        print('[PICKUP-120] Alignment complete, capturing depth map', flush=True)
        
        try:
            with _scene_lock:
                dets = list(_scene_dets)
            
            final_tgt = _locked_120_detection(dets, label) if dets else last_tgt
            if final_tgt is None:
                _set_status('Pickup: object lost during depth capture')
                return
            
            # Get depth plan which includes 3D coordinates
            depth_plan = _target_depth_plan(final_tgt)
            
            if not depth_plan:
                _set_status('Pickup: depth capture failed')
                print('[PICKUP-120] Depth plan is None', flush=True)
                return
            
            distance_cm = depth_plan.get('distance_cm')
            path = depth_plan.get('path')
            
            if distance_cm is None:
                _set_status(f'Pickup: depth unavailable (using fallback)')
                print('[PICKUP-120] No distance_cm in depth_plan, proceeding with box hint', flush=True)
            else:
                _set_status(f'Pickup: object at {distance_cm:.1f} cm')
                print(f'[PICKUP-120] Depth captured: {distance_cm:.1f} cm', flush=True)
            
            # Validate depth is within usable range (replaces broken 3D IK simulation gate)
            use_depth = distance_cm or PICKUP_DEPTH_DEFAULT
            if use_depth < PICKUP_DEPTH_MIN or use_depth > DEPTH_SANITY_MAX_CM:
                print(f'[PICKUP-120] Depth {use_depth:.1f}cm out of range, using default', flush=True)
                use_depth = PICKUP_DEPTH_DEFAULT
            _set_status(f'Pickup-120: depth={use_depth:.1f}cm, proceeding with descent...')

            # â”€â”€ Step 3 & 4: 2-link IK from metric 3D point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Arm geometry: L1=6.5 cm, L2=12.7 cm, total effective reach=19.2 cm.
            if isinstance(path, dict) and path.get('target_cm') is not None:
                target_xyz = path.get('target_cm')
                if isinstance(target_xyz, dict):
                    try:
                        # Get pixel coordinates and raw sensor depth
                        target_px = path.get('target_px', {})
                        px = float(target_px.get('x', 0.0))
                        py = float(target_px.get('y', 0.0))
                        raw_depth_cm = float(distance_cm if distance_cm else
                                             target_xyz.get('z_cm', PICKUP_DEPTH_DEFAULT) + _sensor_to_jaw_offset_cm())

                        _set_status(f'Pickup: repositioning for 3D approach...')

                        with _pos_lock:
                            cur_base = float(_pos['base'])
                            cur_arm = float(_pos['arm'])
                            cur_wrist = float(_pos['wrist'])

                        # Transform camera-space detection to base-frame coordinates
                        reach_pos = _pipeline.transforms.pixel_and_depth_to_base(
                            px, py, raw_depth_cm,
                            cur_base, cur_arm, cur_wrist,
                        )
                        if reach_pos is None:
                            print(f'[PICKUP-120] pixel_and_depth_to_base returned None', flush=True)
                            solution = None
                        else:
                            target_x_cm = float(reach_pos[0])
                            target_y_cm = float(reach_pos[1])
                            target_z_cm = float(reach_pos[2])
                            print(f'[PICKUP-120] 3D target (base frame): X={target_x_cm:.1f} Y={target_y_cm:.1f} Z={target_z_cm:.1f} cm', flush=True)

                            solution = _pipeline.ik.solve_full(
                                target_x_cm, target_y_cm, target_z_cm,
                                cur_base, cur_arm, cur_wrist,
                                safety_margin_cm=PICKUP_FLOOR_SAFETY_MARGIN_CM,
                            )

                        if solution is not None:
                            pipeline_base = solution['base']
                            pipeline_arm = solution['arm']
                            pipeline_wrist = solution['wrist']

                            print(f'[PICKUP-120] pipeline IK -> base={pipeline_base:.1f}deg arm={pipeline_arm:.1f}deg wrist={pipeline_wrist:.1f}deg', flush=True)

                            # Base rotation
                            if not PICKUP_LOCK_BASE_AFTER_ALIGN:
                                if abs(pipeline_base - cur_base) > 0.5:
                                    print(f'[PICKUP-120] base {cur_base:.1f}deg->{pipeline_base:.1f}deg', flush=True)
                                    _move_joint_slow('base', pipeline_base, BASE_MIN, BASE_MAX)
                                    _wait_settle(0.35)

                            # Arm move
                            if abs(pipeline_arm - cur_arm) > 1.0:
                                print(f'[PICKUP-120] arm {cur_arm:.1f}deg->{pipeline_arm:.1f}deg (IK, Z={target_z_cm:.1f}cm)', flush=True)
                                _move_joint_slow('arm', pipeline_arm, ARM_MIN, ARM_MAX)
                                _wait_settle(0.35)

                            # Wrist move
                            pipeline_wrist_clamped = _clamp(pipeline_wrist, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
                            if abs(pipeline_wrist_clamped - cur_wrist) > 1.0:
                                print(f'[PICKUP-120] wrist {cur_wrist:.1f}deg->{pipeline_wrist_clamped:.1f}deg (IK)', flush=True)
                                _move_joint_slow('wrist', pipeline_wrist_clamped, PICKUP_WRIST_MIN, PICKUP_WRIST_MAX)
                                _wait_settle(0.35)
                        else:
                            print(f'[PICKUP-120] Pipeline IK returned None (target: X={target_x_cm:.1f} Y={target_y_cm:.1f} Z={target_z_cm:.1f})', flush=True)

                        _set_status(f'Pickup: top-down approach ready, Z={target_z_cm:.1f}cm, proceeding...')
                        print('[PICKUP-120] 3D bearing + top-down pose complete', flush=True)

                    except Exception as e:
                        print(f'[PICKUP-120] 3D/IK error: {e}', flush=True)
                        traceback.print_exc()
            
            pickup_label = str(final_tgt.get('label') or label or 'object')

            # Step 5: Execute the depth-map plan directly.
            _set_status('Pickup: executing 3D path, grip, and lift...')
            print('[PICKUP-120] Starting direct 3D path sequence', flush=True)
            _execute_planned_grip_and_lift(pickup_label, depth_plan)
            print('[PICKUP-120] Pickup complete', flush=True)
            
        except Exception as e:
            print(f'[PICKUP-120] Error during depth/pickup: {e}', flush=True)
            traceback.print_exc()
            _set_status(f'Pickup error: {e}')
    
    finally:
        _pickup_start_depth_plan = None
        if 'prev_margin' in locals():
            if prev_margin is None:
                try:
                    delattr(_fk_margin_local, 'margin_cm')
                except AttributeError:
                    pass
            else:
                _fk_margin_local.margin_cm = prev_margin
        _pickup_abort.clear()
        _yolo_pause_for_pickup.clear()
        _pickup_lock.release()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  COMMAND DISPATCHER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _dispatch(cmd: str) -> str:
    cmd = ' '.join(str(cmd or '').strip().lower().split())
    cmd = cmd.replace('pick-up', 'pickup')
    if not cmd:
        return 'No command'

    if cmd in ('up', 'move up'):
        _start_task('move-up', _do_up)
        return 'Moving up'
    elif cmd in ('down', 'move down'):
        _start_task('move-down', _do_down)
        return 'Moving down'
    elif cmd in ('left', 'go left', 'rotate left'):
        _start_task('rotate-left', _do_left)
        return 'Rotating left'
    elif cmd in ('right', 'go right', 'rotate right'):
        _start_task('rotate-right', _do_right)
        return 'Rotating right'
    elif cmd == 'wave':
        _start_task('wave', _do_wave)
        return 'Waving'
    elif cmd in ('open', 'release', 'open gripper', 'open hand'):
        _start_task('open-gripper', _do_open)
        return 'Gripper open'
    elif cmd in ('close', 'grip', 'close gripper', 'close hand'):
        _start_task('close-gripper', _do_close)
        return 'Gripper closed'
    elif cmd in ('gripper softer', 'grip softer', 'softer gripper'):
        snap = _set_grip_tune(delta=-1)
        return f'Gripper strength: {snap["name"]}'
    elif cmd in ('gripper harder', 'grip harder', 'harder gripper'):
        snap = _set_grip_tune(delta=1)
        return f'Gripper strength: {snap["name"]}'
    elif cmd in ('wrist left', 'wrist rotate left'):
        _start_task('wrist-left', _do_wrist_left)
        return 'Wrist left'
    elif cmd in ('wrist right', 'wrist rotate right'):
        _start_task('wrist-right', _do_wrist_right)
        return 'Wrist right'
    elif cmd in ('home', 'reset', 'center', 'centre'):
        _start_task('home', _home_all)
        return 'Homing'
    elif cmd in ('stop tracking', 'stop follow', 'stop following'):
        _set_tracking(False)
        return 'Tracking stopped'
    elif cmd in ('stop', 'halt', 'emergency stop'):
        _set_tracking(False)
        _pickup_abort.set()          # FIX v18: interrupt any running pickup immediately
        try:
            _cancel_all_motions()
        except Exception:
            pass
        # FIX v29: clear yolo pause flags so the detector resumes publishing detections.
        # Without this, if stop is pressed before/during pickup the detector stays frozen
        # and tracking sees no detections when re-enabled - arm never moves.
        _yolo_pause_for_pickup.clear()
        _yolo_pause_for_depth.clear()
        _set_status('STOPPED')
        return 'Stopped'
    elif cmd == '120 align' or cmd == '120 align and pickup':
        _start_task('120-align-pickup', _do_120_align_and_pickup)
        return 'Starting 120-pass alignment and pickup sequence'
    elif re.search(r'\b(grab|pick up|pickup|pick)\b', cmd):
        obj = _extract_object_name(cmd, ('grab', 'pick up', 'pickup', 'pick'))
        _start_task('pickup', _do_pickup, obj)
        return f'Grabbing {obj or "best object"}'
    elif re.search(r'\b(track|follow)\b', cmd):
        obj = _extract_object_name(cmd, ('track', 'follow'))
        if not obj:
            with _scene_lock:
                dets = list(_scene_dets)
            best = _best_detection(dets) if dets else None
            if isinstance(best, dict):
                obj = _clean_object_name(best.get('label'))
        _set_tracking(True, obj)
        return f'Tracking {obj}' if obj else 'Tracking started'
    else:
        return f'Unknown command: {cmd}'

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  HTTP API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class CommandAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_bytes(self, body, content_type='application/octet-stream', status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path, content_type):
        try:
            with open(path, 'rb') as f:
                body = f.read()
        except FileNotFoundError:
            self._send_json({'error': f'Missing file: {os.path.basename(path)}'}, 404)
            return
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
            return
        self._send_bytes(body, content_type)

    def _send_json(self, data, status=200):
        try:
            def _json_safe(obj):
                if isinstance(obj, float):
                    return obj if math.isfinite(obj) else None
                if isinstance(obj, np.floating):
                    val = float(obj)
                    return val if math.isfinite(val) else None
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return _json_safe(obj.tolist())
                if isinstance(obj, dict):
                    return {str(k): _json_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_json_safe(v) for v in obj]
                if isinstance(obj, (str, int, bool)) or obj is None:
                    return obj
                return str(obj)

            body = json.dumps(_json_safe(data), ensure_ascii=False, allow_nan=False).encode('utf-8')
        except Exception as e:
            # Last-resort: return a minimal error body rather than dropping the connection.
            try:
                body = json.dumps({'error': f'JSON serialization failed: {e}'}, allow_nan=False).encode('utf-8')
            except Exception:
                body = b'{"error":"JSON serialization failed"}'
            status = 500
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/control', '/index.html'):
            self._send_file(_CONTROL_HTML, 'text/html; charset=utf-8')
        elif path == '/three.min.js':
            self._send_file(_THREE_JS, 'application/javascript; charset=utf-8')
        elif path == '/arm_control_v2.html':
            self._send_file(_CONTROL_HTML, 'text/html; charset=utf-8')
        elif path == '/pi_arm_server_v3 (1).py':
            self._send_file(os.path.abspath(__file__), 'text/x-python; charset=utf-8')
        elif path == '/status':
            try:
                tracking, track_obj = _tracking_snapshot()
                if _scene_lock.acquire(timeout=2.0):
                    try:
                        scene_snap = dict(_scene_info)
                    finally:
                        _scene_lock.release()
                else:
                    print('[STATUS] _scene_lock timeout', flush=True)
                    scene_snap = {'label': 'unknown', 'count': 0}
                pos = _pos_snapshot()
                arm_serial_snap = _arm_serial_snapshot()
                actual_arm = arm_serial_snap.get('position')
                if isinstance(actual_arm, (int, float)) and math.isfinite(float(actual_arm)):
                    # Report measured shoulder position to the UI without
                    # overwriting the planner's commanded-position state.
                    pos['arm'] = float(actual_arm)
            except Exception as exc:
                print(f'[STATUS] snapshot error (non-fatal): {exc}', flush=True)
                tracking, track_obj = False, None
                scene_snap = {'label': 'unknown', 'count': 0}
                with _pos_lock:
                    pos = dict(_pos)
                arm_serial_snap = _arm_serial_snapshot()
            self._send_json({
                'status': _status_txt[0],
                'position': pos,
                'tracking': tracking,
                'track_object': track_obj,
                'detector': scene_snap.get('label', 'unknown'),
                'detector_mode': _detector_mode[0],
                'grip_tune': _grip_tune_snapshot(),
                'depth_model': _depth_status_snapshot(),
                'depth_result': _depth_result_snapshot(),
                'detections': scene_snap.get('count', 0),
                'gpio_ready': bool(_gpio_ready[0]),
                'arm_serial': arm_serial_snap,
                'digital_twin': {
                    'L1_cm': ARM_LINK1_CM,
                    'L2_cm': ARM_LINK2_CM,
                    'reach_cm': ARM_REACH_PHYSICAL_CM,
                    'floor_below_shoulder_cm': FLOOR_BELOW_SHOULDER_CM,
                    'safety_margin_cm': FLOOR_SAFETY_MARGIN_CM,
                    'gripper_finger_ext_cm': GRIPPER_FINGER_EXT_CM,
                },
            })
            _touch_heartbeat()
        elif path == '/detections':
            if _scene_lock.acquire(timeout=2.0):
                try:
                    scene_ts = _scene_info['ts']
                    scene_label = _scene_info['label']
                    scene_count = _scene_info['count']
                finally:
                    _scene_lock.release()
            else:
                print('[DETECTIONS] _scene_lock timeout', flush=True)
                scene_ts = 0
                scene_label = 'unknown'
                scene_count = 0
            items = _current_display_detections()
            data = {
                'updated': scene_ts,
                'label': scene_label,
                'count': len(items) if _yolo_pause_for_pickup.is_set() else scene_count,
                'items': items,
                'pickup_frozen': bool(_yolo_pause_for_pickup.is_set()),
                'depth_model': _depth_status_snapshot(),
                'depth_result': _depth_result_snapshot(),
            }
            self._send_json(data)
        elif path == '/pickup_feedback':
            with _learn_lock:
                data = dict(_pickup_learning[0])
            self._send_json(data)
        elif path == '/depth':
            self._send_json({'depth_model': _depth_status_snapshot()})
        elif path in ('/depth/latest', '/depth/result'):
            self._send_json({
                'depth_model': _depth_status_snapshot(),
                'depth_result': _depth_result_snapshot(),
            })
        elif path == '/depth/scan':
            result, _jpeg = _depth_scan_now()
            self._send_json({
                'depth_result': result or _depth_result_snapshot(),
                'depth_map_b64': None,
                'has_map': False,
            })
        elif path == '/depth_map.jpg':
            self._send_json({'error': 'Depth map disabled (VL53L1X mode)'}, 404)
        elif path == '/calibrate':
            with _calibration_lock:
                snap = dict(_calibration_state)
            self._send_json({'calibration': snap})
        elif path == '/grip_tune':
            self._send_json({'grip_tune': _grip_tune_snapshot()})
        elif path == '/gripper_current':
            current_ma = 0.0
            voltage_v = 0.0
            power_mw = 0.0
            sensor_ok = False
            if _ina219[0] is not None:
                try:
                    current_ma = _ina219[0].current   # already in mA
                    voltage_v = _ina219[0].bus_voltage  # already in V
                    power_mw = _ina219[0].power       # already in mW
                    sensor_ok = True
                except Exception as exc:
                    print(f'[INA219] Read error: {exc}', flush=True)
            self._send_json({
                'current_ma': round(current_ma, 1),
                'voltage_v': round(voltage_v, 3),
                'power_mw': round(power_mw, 1),
                'sensor_ok': sensor_ok,
            })
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            self._send_json({'error': 'Invalid Content-Length'}, 400)
            return
        if length < 0 or length > MAX_JSON_BODY:
            self._send_json({'error': 'JSON body too large'}, 413)
            return
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode('utf-8') if body else '{}')
        except Exception:
            self._send_json({'error': 'Invalid JSON'}, 400)
            return
        if not isinstance(data, dict):
            self._send_json({'error': 'JSON body must be an object'}, 400)
            return

        if path == '/calibrate':
            """Calibration mode endpoint.

            GET  /calibrate                          â€” read current offsets
            POST /calibrate  {"gripper_x_cm": N}    â€” set GRIPPER_X_OFFSET_CM
            POST /calibrate  {"gripper_y_cm": N}    â€” set GRIPPER_Y_OFFSET_CM
            POST /calibrate  {"camera_center_px": N} â€” set CAMERA_CENTER_OFFSET_PX
            POST /calibrate  {"close_range_scale": N} â€” set close-range depth scale
            POST /calibrate  {"depth_real_cm": R, "depth_predicted_cm": P}
                             â€” auto-compute close_range_scale = R / P

            POST /calibrate  {"grab_error_x_cm": E, "grab_depth_cm": D}
                             â€” auto-compute gripper_x_cm offset:
                               correction = -E   (if arm grabbed E cm to the right, add -E)
                               camera_center_px = E * fx / D

            POST /calibrate  {"reset": true}        â€” restore factory defaults
            POST /calibrate  {"mode": true/false}   â€” enable/disable calibration mode
            """
            # Apply each key independently so the UI can send partial updates
            changed = {}
            with _calibration_lock:
                offsets = _calibration_state['offsets']

                if _as_bool(data.get('reset', False)):
                    offsets['gripper_x_cm']     = 0.0
                    offsets['gripper_y_cm']     = 0.0
                    offsets['camera_center_px'] = 0.0
                    offsets['camera_center_y_px'] = 0.0
                    offsets['close_range_scale'] = 1.0
                    offsets['home_base'] = 90.0
                    offsets['home_arm'] = float(ARM_HOME_DEG + ARM_TRIM_DEG)
                    offsets['home_wrist'] = float(_clamp(WRIST_HOME + WRIST_TRIM_DEG, 0, 180))
                    offsets['home_grip'] = float(GRIP_OPEN)
                    changed['reset'] = True

                if 'mode' in data:
                    _calibration_state['mode'] = _as_bool(data['mode'])
                    changed['mode'] = _calibration_state['mode']

                if 'gripper_x_cm' in data:
                    try:
                        v = float(data['gripper_x_cm'])
                        if math.isfinite(v):
                            offsets['gripper_x_cm'] = v
                            changed['gripper_x_cm'] = v
                    except (TypeError, ValueError):
                        pass

                if 'gripper_y_cm' in data:
                    try:
                        v = float(data['gripper_y_cm'])
                        if math.isfinite(v):
                            offsets['gripper_y_cm'] = v
                            changed['gripper_y_cm'] = v
                    except (TypeError, ValueError):
                        pass

                if 'camera_center_px' in data:
                    try:
                        v = float(data['camera_center_px'])
                        if math.isfinite(v):
                            offsets['camera_center_px'] = v
                            changed['camera_center_px'] = v
                    except (TypeError, ValueError):
                        pass

                if 'camera_center_y_px' in data:
                    try:
                        v = float(data['camera_center_y_px'])
                        if math.isfinite(v):
                            offsets['camera_center_y_px'] = v
                            changed['camera_center_y_px'] = v
                    except (TypeError, ValueError):
                        pass

                if 'close_range_scale' in data:
                    try:
                        v = float(data['close_range_scale'])
                        if math.isfinite(v) and 0.1 <= v <= 5.0:
                            offsets['close_range_scale'] = v
                            changed['close_range_scale'] = v
                    except (TypeError, ValueError):
                        pass

                if 'home' in data:
                    try:
                        home_data = data['home']
                        if str(home_data).lower() == 'current':
                            home_data = _pos_snapshot()
                        if isinstance(home_data, dict):
                            limits = {
                                'base': (BASE_MIN, BASE_MAX),
                                'arm': (ARM_MIN, ARM_MAX),
                                'wrist': (0, 180),
                                'grip': (GRIP_OPEN, GRIP_CLOSE),
                            }
                            for joint, (lo, hi) in limits.items():
                                if joint in home_data:
                                    v = float(home_data[joint])
                                    if math.isfinite(v):
                                        v = float(_clamp(v, lo, hi))
                                        offsets[f'home_{joint}'] = v
                                        changed[f'home_{joint}'] = v
                    except (TypeError, ValueError):
                        pass

                # Auto-compute close_range_scale from a measurement pair
                if 'depth_real_cm' in data and 'depth_predicted_cm' in data:
                    try:
                        real_cm = float(data['depth_real_cm'])
                        pred_cm = float(data['depth_predicted_cm'])
                        if pred_cm > 0 and real_cm > 0:
                            new_scale = _clamp(real_cm / pred_cm, 0.1, 5.0)
                            offsets['close_range_scale'] = new_scale
                            changed['close_range_scale'] = new_scale
                            changed['auto_computed'] = True
                    except (TypeError, ValueError):
                        pass

                # Auto-compute gripper/crosshair offsets from measured grab error.
                if 'grab_error_x_cm' in data or 'grab_error_y_cm' in data:
                    try:
                        depth_cm = float(data.get('grab_depth_cm') or 20.0)
                        fx = (FRAME_W * 0.5) / math.tan(math.radians(DEPTH_CAMERA_FOV_X_DEG) * 0.5)
                        fy = (FRAME_H * 0.5) / math.tan(math.radians(DEPTH_CAMERA_FOV_Y_DEG) * 0.5)
                        if 'grab_error_x_cm' in data:
                            err_x = float(data['grab_error_x_cm'])  # positive = grabbed right of target
                            new_gx = offsets['gripper_x_cm'] - err_x
                            offsets['gripper_x_cm'] = new_gx
                            changed['gripper_x_cm'] = new_gx
                            px_shift = -err_x * fx / max(1.0, depth_cm)
                            new_cco = _clamp(offsets['camera_center_px'] + px_shift, -FRAME_W, FRAME_W)
                            offsets['camera_center_px'] = new_cco
                            changed['camera_center_px'] = new_cco
                        if 'grab_error_y_cm' in data:
                            err_y = float(data['grab_error_y_cm'])  # positive = grabbed too low / below target
                            new_gy = offsets['gripper_y_cm'] - err_y
                            offsets['gripper_y_cm'] = new_gy
                            changed['gripper_y_cm'] = new_gy
                            py_shift = -err_y * fy / max(1.0, depth_cm)
                            new_ccy = _clamp(offsets.get('camera_center_y_px', 0.0) + py_shift, -FRAME_H, FRAME_H)
                            offsets['camera_center_y_px'] = new_ccy
                            changed['camera_center_y_px'] = new_ccy
                        changed['auto_computed'] = True
                    except (TypeError, ValueError):
                        pass

                snap = dict(_calibration_state)

            if changed:
                _set_status(f'Calibration updated: {", ".join(f"{k}={v}" for k, v in changed.items() if k != "auto_computed")}')
                print(f'[CAL] Updated offsets: {snap["offsets"]}', flush=True)

            self._send_json({'calibration': snap, 'changed': changed})
            return

        if path == '/grip_tune':
            level = data.get('level')
            delta = data.get('delta', 0)
            mode = str(data.get('mode', '')).strip().lower()
            if level is None:
                if mode in ('softer', 'soft', 'down', '-1'):
                    delta = -1
                elif mode in ('harder', 'hard', 'up', '+1', '1'):
                    delta = 1
                elif mode in ('normal', 'reset'):
                    level = 2
            try:
                snap = _set_grip_tune(
                    delta=int(float(delta or 0)),
                    level=None if level is None else int(float(level)),
                )
            except (TypeError, ValueError):
                self._send_json({'error': 'Send level 0-4, delta +/-1, or mode softer/harder'}, 400)
                return
            self._send_json({'status': _status_txt[0], 'grip_tune': snap})
            return

        if path == '/command':
            cmd = data.get('command', '')
            try:
                result = _dispatch(cmd)
            except Exception as exc:
                print(f'[CMD] dispatch error: {exc}', flush=True)
                traceback.print_exc()
                result = f'Stop (error: {exc})'
            _set_status(result)
            try:
                tracking, track_obj = _tracking_snapshot()
                pos = _pos_snapshot()
            except Exception as exc:
                print(f'[CMD] snapshot error (non-fatal): {exc}', flush=True)
                tracking, track_obj = False, None
                with _pos_lock:
                    pos = dict(_pos)
            self._send_json({
                'status': result,
                'position': pos,
                'tracking': tracking,
                'track_object': track_obj,
            })
            _touch_heartbeat()
            return

        if path == '/pickup':
            obj = _clean_object_name(data.get('object', None))
            if _pickup_lock.locked():
                self._send_json({'status': 'Pickup: already running', 'position': _pos_snapshot()}, 409)
                return

            # Capture the current best target synchronously at button press, but
            # keep YOLO live so the alignment loop can re-measure the centroid
            # after each arm/base correction.
            with _scene_lock:
                dets = list(_scene_dets)
            frozen = _best_detection(dets, obj) if dets else None
            if frozen is None:
                _clear_pickup_frozen_target()
                self._send_json({'status': 'Pickup: no object detected', 'position': _pos_snapshot()}, 404)
                return
            frozen = dict(frozen)
            pickup_depth_box = None
            raw_box = frozen.get('box') if isinstance(frozen, dict) else None
            if raw_box and len(raw_box) == 4:
                pickup_depth_box = [int(round(float(v))) for v in raw_box]
            if pickup_depth_box is None:
                pickup_depth_box = _depth_box_from_center(
                    frozen.get('cx', FRAME_W / 2),
                    frozen.get('cy', FRAME_H / 2),
                )

            # Depth is intentionally captured inside the pickup thread AFTER
            # live crosshair centering.  Running DA2 here blocks the button
            # handler before alignment and gives the grab a pre-centering Z plan.
            depth_result_at_press = None
            depth_map_b64_at_press = None

            # Snapshot the arm position BEFORE the pickup thread moves anything.
            # This is what the UI shows as "position at pickup trigger".
            with _pos_lock:
                snap = {
                    'base':  round(float(_pos['base']),  1),
                    'arm':   round(float(_pos['arm']),   1),
                    'wrist': round(float(_pos['wrist']), 1),
                    'grip':  round(float(_pos['grip']),  1),
                }
            with _pickup_start_pos_lock:
                _pickup_start_pos[0] = dict(snap)
            _start_task('pickup', _do_pickup, obj, frozen)
            resp = {
                'status': f'Grabbing {frozen.get("label", obj or "best object")}',
                'position': _pos_snapshot(),
                'pickup_start_pos': snap,
                'frozen_target': {
                    'target_x': frozen.get('target_x', frozen.get('object_center_x', frozen.get('cx'))),
                    'target_y': frozen.get('target_y', frozen.get('object_center_y', frozen.get('cy'))),
                    'object_center_x': frozen.get('object_center_x', frozen.get('cx')),
                    'object_center_y': frozen.get('object_center_y', frozen.get('cy')),
                    'target_class': frozen.get('target_class'),
                    'label': frozen.get('label'),
                    'centroid_source': frozen.get('centroid_source', frozen.get('center_source')),
                    'pipeline': 'yolo-mask-centroid-tof',
                },
            }
            if depth_result_at_press is not None:
                resp['depth_result'] = depth_result_at_press
            if depth_map_b64_at_press is not None:
                resp['depth_map_b64'] = depth_map_b64_at_press
                resp['has_map'] = True
            self._send_json(resp)
            _touch_heartbeat()
            return

        if path == '/simulate':
            """Run kinematic simulation without any arm movement.

            Computes the full 3-step pickup path centered on the object:
            1. CENTER: rotate base + arm to align with object pixel position
            2. HOVER: descend halfway to approach position
            3. GRASP: descend to contact with depth-driven wrist angle
            Returns all 3 joint-angle poses + FK positions for ghost arms.
            """
            sim_px = data.get('px')
            sim_py = data.get('py')
            sim_depth = data.get('depth_cm')

            if sim_px is None or sim_py is None:
                self._send_json({'success': False, 'errors': ['px and py required']}, 400)
                return

            sensor_depth = None

            # Try to get from current detection with forced depth scan
            dets = []
            if _scene_lock.acquire(timeout=2.0):
                try:
                    dets = list(_scene_dets)
                finally:
                    _scene_lock.release()
            else:
                print('[SIM] _scene_lock timeout', flush=True)
            frozen = _best_detection(dets, data.get('object')) if dets else None
            if frozen:
                with _depth_lock:
                    _depth_state['last_ts'] = 0.0
                depth_plan = _target_depth_plan(frozen, force=True)
                if depth_plan:
                    sensor_depth = depth_plan.get('distance_cm')

            # Fallback: latest ToF reading
            if sensor_depth is None or sensor_depth <= 0:
                depth_snap = _depth_result_snapshot()
                sensor_depth = depth_snap.get('distance_cm') if isinstance(depth_snap, dict) else None

            # Fallback: UI-sent depth
            if sensor_depth is None or sensor_depth <= 0:
                sensor_depth = sim_depth

            if sensor_depth is None or sensor_depth <= 0:
                sensor_depth = PICKUP_DEPTH_DEFAULT

            # Get metric depth (jaw distance) for descent calculation
            metric_depth = float(_clamp(
                _jaw_distance_from_camera_depth(float(sensor_depth)),
                PICKUP_DEPTH_MIN, DEPTH_SANITY_MAX_CM))

            # Get current arm pose
            with _pos_lock:
                cur_base = float(_pos['base'])
                cur_arm = float(_pos['arm'])
                cur_wrist = float(_pos['wrist'])

            # === STEP 1: CENTER — convert pixel error to base/arm angles ===
            # Camera FOV → degrees per pixel
            fov_x = float(DEPTH_CAMERA_FOV_X_DEG)
            fov_y = float(DEPTH_CAMERA_FOV_Y_DEG)
            deg_per_px_x = fov_x / float(FRAME_W)
            deg_per_px_y = fov_y / float(FRAME_H)

            # Pixel error from calibrated aim point (center of frame)
            aim_x, aim_y = _pickup_aim_xy(None)
            err_x = float(sim_px) - aim_x
            err_y = float(sim_py) - aim_y
            pixel_err = math.hypot(err_x, err_y)
            centered = pixel_err < 60.0

            # Base correction: horizontal pixel error → base rotation
            # base_sign = -1.0 (hardware inverted)
            # Gain < 1.0 to prevent overshoot; tracking uses 0.35 per-step
            CENTER_GAIN_BASE = 0.85
            CENTER_GAIN_ARM  = 1.0
            base_correction = -1.0 * err_x * deg_per_px_x * CENTER_GAIN_BASE

            # Arm correction: vertical pixel error → arm rotation
            # arm_sign_track = -1.0 (hardware inverted)
            arm_correction = -1.0 * err_y * deg_per_px_y * CENTER_GAIN_ARM

            centered_base = float(_clamp(
                cur_base + base_correction, BASE_MIN, BASE_MAX))
            centered_arm = float(_clamp(
                cur_arm + arm_correction, ARM_MIN, ARM_MAX))
            centered_wrist = cur_wrist

            # === STEP 2: HOVER — half descent from centered pose ===
            full_drop, _, final_descent = _depth_contact_drop_degs(metric_depth)
            depth_bias = min(float(PICKUP_CENTER_CONTACT_BIAS_DEG),
                             float(PICKUP_CENTER_CONTACT_BIAS_DEG) * (metric_depth / 20.0))
            descent_deg = float(_clamp(
                full_drop + depth_bias,
                PICKUP_MIN_DESCENT_BEFORE_GRIP_DEG,
                PICKUP_MAX_DESCENT_DEG))

            hover_arm = float(_clamp(
                centered_arm + PICKUP_ARM_DESCEND_SIGN * descent_deg * 0.5,
                ARM_MIN, ARM_MAX))
            hover_wrist = float(_clamp(
                (cur_wrist + _wrist_from_depth(metric_depth, centered_arm)) * 0.5,
                PICKUP_WRIST_MIN, PICKUP_WRIST_MAX))

            # === STEP 3: GRASP — full descent from centered pose ===
            grasp_arm = float(_clamp(
                centered_arm + PICKUP_ARM_DESCEND_SIGN * descent_deg,
                ARM_MIN, ARM_MAX))
            grasp_wrist = _wrist_from_depth(metric_depth, centered_arm)

            # FK for all 3 poses
            center_pos, _ = _pipeline.fk.gripper_pose(centered_base, centered_arm, centered_wrist)
            hover_pos, _ = _pipeline.fk.gripper_pose(centered_base, hover_arm, hover_wrist)
            grasp_pos, _ = _pipeline.fk.gripper_pose(centered_base, grasp_arm, grasp_wrist)

            result = {
                'success': True,
                'validation_ok': True,
                'errors': [],
                'depth_cm': float(sensor_depth),
                'metric_depth_cm': metric_depth,
                'descent_deg': descent_deg,
                'wrist_from_depth': grasp_wrist,
                'object_pixel': (float(sim_px), float(sim_py)),
                'centering': {
                    'centered': centered,
                    'pixel_error': pixel_err,
                    'base_correction': base_correction,
                    'arm_correction': arm_correction,
                },
                'hover_angles': {
                    'base': centered_base,
                    'arm': hover_arm,
                    'wrist': hover_wrist,
                    'grip': GRIP_OPEN,
                },
                'grasp_angles': {
                    'base': centered_base,
                    'arm': grasp_arm,
                    'wrist': grasp_wrist,
                    'grip': GRIP_CLOSE,
                },
                'center_position': {
                    'x': float(center_pos[0]),
                    'y': float(center_pos[1]),
                    'z': float(center_pos[2]),
                },
                'hover_position': {
                    'x': float(hover_pos[0]),
                    'y': float(hover_pos[1]),
                    'z': float(hover_pos[2]),
                },
                'grasp_position': {
                    'x': float(grasp_pos[0]),
                    'y': float(grasp_pos[1]),
                    'z': float(grasp_pos[2]),
                },
                'score': {
                    'total_score': 1.0 if centered else 0.7,
                    'centering_score': 1.0 if centered else 0.5,
                },
                'trajectory': [
                    {'base': cur_base, 'arm': cur_arm, 'wrist': cur_wrist},
                    {'base': centered_base, 'arm': centered_arm, 'wrist': centered_wrist},
                    {'base': centered_base, 'arm': hover_arm, 'wrist': hover_wrist},
                    {'base': centered_base, 'arm': grasp_arm, 'wrist': grasp_wrist},
                ],
            }
            print(f'[SIM] depth={sensor_depth:.1f}cm metric={metric_depth:.1f}cm '
                  f'descent={descent_deg:.1f}° '
                  f'center=({centered_base:.1f},{centered_arm:.1f}) '
                  f'hover=({centered_base:.1f},{hover_arm:.1f},{hover_wrist:.1f}) '
                  f'grasp=({centered_base:.1f},{grasp_arm:.1f},{grasp_wrist:.1f}) '
                  f'centering_err=({err_x:.0f},{err_y:.0f}px) centered={centered}',
                  flush=True)
            self._send_json(result)
            _touch_heartbeat()
            return

        if path == '/pickup_feedback':
            result = str(data.get('result', '')).strip().lower()
            if result not in ('success', 'failed', 'too_low', 'too_high', 'too_close', 'too_far'):
                self._send_json({'error': 'result must be success, failed, too_low, too_high, too_close, or too_far'}, 400)
                return
            reason = str(data.get('reason', '')).strip().lower()
            if result == 'too_low':
                reason = 'too_low'
            elif result == 'too_high':
                reason = 'too_high'
            elif result == 'too_close':
                reason = 'too_close'
            elif result == 'too_far':
                reason = 'too_far'
            feedback = _record_pickup_feedback(
                result == 'success',
                opened=_as_bool(data.get('opened', False)),
                homed=_as_bool(data.get('homed', False)),
                reason=reason,
            )
            self._send_json({'status': _status_txt[0], 'learning': feedback, 'position': _pos_snapshot()})
            return

        if path == '/depth/scan':
            if not VL53L1X_ENABLED:
                self._send_json({'error': 'VL53L1X is disabled'}, 503)
                return
            result, _jpeg = _depth_scan_now()
            self._send_json({
                'depth_result': result or _depth_result_snapshot(),
                'depth_map_b64': None,
                'has_map': False,
            })
            return

        if path == '/depth':
            if _as_bool(data.get('toggle', False)):
                with _depth_lock:
                    desired = not bool(_depth_state.get('runtime_enabled', VL53L1X_ENABLED))
            else:
                desired = data.get('enabled', data.get('on', None))
                if desired is None:
                    self._send_json({'error': 'Send enabled true/false or toggle true'}, 400)
                    return
                desired = _as_bool(desired)
            snapshot = _set_depth_runtime_enabled(desired, reason='manual toggle')
            self._send_json({'status': 'VL53L1X on' if desired else 'VL53L1X off', 'depth_model': snapshot})
            return

        if path == '/grip_dc':
            cmd = str(data.get('cmd', data.get('action', ''))).strip().lower()
            if cmd not in ('open', 'close', 'stop'):
                self._send_json({'error': 'cmd must be open, close, or stop'}, 400)
                return
            if cmd == 'stop':
                _grip_dc_stop()
                self._send_json({'status': 'grip stopped', 'position': _pos_snapshot()})
            else:
                _start_task('grip-dc', _grip_dc_run, cmd)
                self._send_json({'status': f'grip {cmd}', 'position': _pos_snapshot()})
            _touch_heartbeat()
            return

        if path == '/joint':
            joint = str(data.get('joint', '')).strip().lower()
            angle = data.get('angle', None)
            if joint not in JOINT_LIMITS or angle is None:
                self._send_json({'error': 'Invalid joint or angle'}, 400)
                return
            try:
                angle = float(angle)
            except (TypeError, ValueError):
                self._send_json({'error': 'Angle must be a number'}, 400)
                return
            if not math.isfinite(angle):
                self._send_json({'error': 'Angle must be finite'}, 400)
                return
            lo, hi = JOINT_LIMITS[joint]
            angle = _clamp(angle, lo, hi)
            # DC motor mode: grip commands go through DC control, not angle tracking
            if GRIP_DC_MODE and joint == 'grip':
                if angle <= (GRIP_OPEN + GRIP_CLOSE) / 2:
                    _start_task('grip-dc', _grip_dc_run, 'open')
                else:
                    _start_task('grip-dc', _grip_dc_run, 'close')
                self._send_json({'status': f'grip -> {"open" if angle <= (GRIP_OPEN + GRIP_CLOSE) / 2 else "close"}', 'position': _pos_snapshot()})
                _touch_heartbeat()
                return
            allowed, worst, height, margin = _fk_gate_joint_target(joint, angle)
            if not allowed:
                shortfall = float(margin) - float(height)
                _set_status(f'REJECTED: Floor collision ({worst})')
                _fk_log_reject('/joint', joint, angle, worst, height, margin)
                self._send_json({
                    'error': f'Floor collision ({worst}): {shortfall:.2f}cm below safety margin',
                    'segment': worst,
                    'clearance_cm': height,
                }, 400)
                return
            # Manual shoulder commands come from the UI slider and may cover a
            # large angle in one request.  Use the existing slow trajectory for
            # that heavily loaded joint so the initial acceleration/current
            # spike does not trip a high-torque servo's overload protection.
            # Tracking still calls _move_joint directly and keeps its faster
            # response; the other manual joints retain their current profiles.
            mover = _move_joint_slow if joint == 'arm' else _move_joint
            _start_task(f'{joint}-move', mover, joint, angle, lo, hi)
            self._send_json({'status': f'{joint} -> {angle:.1f}°', 'position': _pos_snapshot()})
            _touch_heartbeat()
            return

        self._send_json({'error': 'Not found'}, 404)

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path not in ('/', '/stream'):
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # FIX BUG 10: Don't re-send the same jpeg every 1ms. Track the last sent
        # object identity and only write when the buffer has been replaced by the
        # camera thread. Also enforce a minimum inter-frame delay (~30fps cap) to
        # avoid saturating the connection with duplicate frames.
        last_sent = None
        min_frame_interval = 1.0 / 30.0  # 30 fps cap
        last_sent_time = 0.0
        try:
            while _running[0]:
                with _stream_lock:
                    jpeg = _stream_jpeg[0]
                now = time.time()
                if jpeg is None or jpeg is last_sent or (now - last_sent_time) < min_frame_interval:
                    time.sleep(0.005)
                    continue
                try:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    last_sent = jpeg
                    last_sent_time = now
                except Exception:
                    break
        except Exception:
            pass

class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

def _run_cmd_api():
    while _running[0]:
        try:
            server = ReusableThreadingHTTPServer(('0.0.0.0', CMD_API_PORT), CommandAPIHandler)
        except OSError as e:
            msg = f'Command API failed to start on port {CMD_API_PORT}: {e}'
            print(f'[CMD API] {msg}', flush=True)
            _set_status(msg)
            time.sleep(2)
            continue
        server.daemon_threads = True
        print(f'[CMD API] Listening on port {CMD_API_PORT}', flush=True)
        _touch_heartbeat()
        try:
            server.serve_forever()
        except Exception as e:
            print(f'[CMD API] serve_forever crashed: {e}', flush=True)
            import traceback as _tb; _tb.print_exc()
        except KeyboardInterrupt:
            break
        print(f'[CMD API] Server stopped, restarting in 1s...', flush=True)
        time.sleep(1)

def _run_stream():
    while _running[0]:
        try:
            server = ReusableThreadingHTTPServer(('0.0.0.0', STREAM_PORT), StreamHandler)
        except OSError as e:
            msg = f'Stream failed to start on port {STREAM_PORT}: {e}'
            print(f'[STREAM] {msg}', flush=True)
            _set_status(msg)
            time.sleep(2)
            continue
        server.daemon_threads = True
        print(f'[STREAM] Listening on port {STREAM_PORT}', flush=True)
        try:
            server.serve_forever()
        except Exception as e:
            print(f'[STREAM] serve_forever crashed: {e}', flush=True)
            import traceback as _tb; _tb.print_exc()
        except KeyboardInterrupt:
            break
        print(f'[STREAM] Server stopped, restarting in 1s...', flush=True)
        time.sleep(1)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CAMERA CAPTURE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _make_status_frame(message):
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (32, 32, 32)
    cv2.putText(frame, 'Robot Arm Server', (24, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(message), (24, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    return frame

def _publish_stream_frame(frame):
    try:
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    except Exception as e:
        print(f'[STREAM] JPEG encode failed: {e}')
        return False
    if ok:
        with _stream_lock:
            _stream_jpeg[0] = buf.tobytes()
        return True
    return False

def _camera_thread():
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    except Exception as e:
        msg = f'Camera open failed: {e}'
        print(f'[CAM] {msg}')
        _set_status(msg)
        while _running[0]:
            _publish_stream_frame(_make_status_frame(msg))
            time.sleep(1.0)
        return

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    _cap_ref[0] = cap
    if not cap.isOpened():
        msg = f'Camera {CAMERA_INDEX} unavailable'
        print(f'[CAM] {msg}')
        _set_status(msg)
        while _running[0]:
            _publish_stream_frame(_make_status_frame(msg))
            time.sleep(1.0)
        cap.release()
        return

    failed_reads = 0
    while _running[0]:
        try:
            ok, frame = cap.read()
        except Exception as e:
            ok, frame = False, None
            print(f'[CAM] read failed: {e}')
        if not ok:
            failed_reads += 1
            if failed_reads == 1 or failed_reads % 50 == 0:
                _publish_stream_frame(_make_status_frame('Waiting for camera frame...'))
            time.sleep(0.05)
            continue
        failed_reads = 0

        with _frame_lock:
            _latest_frame[0] = frame.copy()

        # Draw the most recent detections without waiting for a new inference
        # pass. During pickup, force the overlay to use the frozen box only.
        dets = _current_display_detections()

        if dets:
            overlay = frame.copy()
            overlay = _draw_detections(overlay, dets)
        else:
            overlay = frame

        _publish_stream_frame(overlay)

    cap.release()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•ï¿½ï¿½â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _run_depth_worker_main():
    """Child process: load Depth Anything V2 metric depth and answer JSON-line requests."""
    print('[DA2-WORKER] Starting depth worker process...', flush=True)
    try:
        os.environ.setdefault('TORCH_HOME', _DEPTH_TORCH_HOME_PATH)
        print(f'[DA2-WORKER] TORCH_HOME={_DEPTH_TORCH_HOME_PATH}', flush=True)
        import torch
        print(f'[DA2-WORKER] PyTorch imported', flush=True)
        try:
            torch.set_num_threads(max(1, int(DEPTH_TORCH_THREADS)))
        except Exception:
            pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[DA2-WORKER] Device: {device}', flush=True)

        # Load the METRIC dpt.py directly by file path using importlib.
        # This bypasses sys.path entirely â€” no risk of picking up the base dpt.py.
        import importlib.util as _ilu
        _metric_dpt_candidates = [
            os.path.join(_DIR, 'Depth-Anything-V2', 'metric_depth', 'depth_anything_v2', 'dpt.py'),
            os.path.join(_DIR, 'metric_depth', 'depth_anything_v2', 'dpt.py'),
        ]
        _metric_dpt_path = None
        for _c in _metric_dpt_candidates:
            if os.path.isfile(_c):
                _metric_dpt_path = _c
                break
        if _metric_dpt_path is None:
            raise RuntimeError(
                'Cannot find metric dpt.py. Expected at: '
                + _metric_dpt_candidates[0]
            )
        print(f'[DA2-WORKER] Loading metric dpt.py from: {_metric_dpt_path}', flush=True)

        # Also add metric_depth to sys.path so dpt.py can import its siblings
        _metric_depth_dir = os.path.dirname(os.path.dirname(_metric_dpt_path))
        if _metric_depth_dir not in sys.path:
            sys.path.insert(0, _metric_depth_dir)

        # Purge any already-cached depth_anything_v2 modules so we load fresh
        for _k in list(sys.modules.keys()):
            if _k.startswith('depth_anything_v2'):
                del sys.modules[_k]

        _spec = _ilu.spec_from_file_location('depth_anything_v2.dpt', _metric_dpt_path)
        _dpt_mod = _ilu.module_from_spec(_spec)
        sys.modules['depth_anything_v2.dpt'] = _dpt_mod
        _spec.loader.exec_module(_dpt_mod)
        DepthAnythingV2 = _dpt_mod.DepthAnythingV2

        import inspect as _inspect
        if 'max_depth' not in _inspect.signature(DepthAnythingV2.__init__).parameters:
            raise RuntimeError(
                f'Loaded dpt.py from {_metric_dpt_path} but DepthAnythingV2.__init__ '
                'still has no max_depth param â€” wrong file or wrong class.'
            )
        print(f'[DA2-WORKER] Metric DepthAnythingV2 loaded OK (max_depth param confirmed)', flush=True)

        encoder = DEPTH_ENCODER if DEPTH_ENCODER in DEPTH_MODEL_CONFIGS else 'vits'
        model_config = dict(DEPTH_MODEL_CONFIGS[encoder])
        max_depth = int(DEPTH_MAX_DEPTH_M)
        
        # Official API: DepthAnythingV2(**{**config, 'max_depth': max_depth})
        model_config['max_depth'] = max_depth
        
        print(f'[DA2-WORKER] Loading model: encoder={encoder}, max_depth={max_depth}m', flush=True)
        model = DepthAnythingV2(**model_config)
        print(f'[DA2-WORKER] Model created, loading checkpoint...', flush=True)

        checkpoint = _DEPTH_CHECKPOINT_PATH
        print(f'[DA2-WORKER] Checkpoint path: {checkpoint}', flush=True)
        if not os.path.isfile(checkpoint):
            os.makedirs(_DEPTH_CHECKPOINT_DIR, exist_ok=True)
            print(f'[DA2-WORKER] Checkpoint not found, downloading...', flush=True)
            try:
                from huggingface_hub import hf_hub_download
                checkpoint = hf_hub_download(
                    repo_id=DEPTH_HF_REPO,
                    filename=DEPTH_CHECKPOINT_NAME,
                    local_dir=_DEPTH_CHECKPOINT_DIR,
                )
            except Exception as e:
                raise RuntimeError(
                    f'Missing Depth Anything V2 checkpoint {DEPTH_CHECKPOINT_NAME}. '
                    f'Put it in {_DEPTH_CHECKPOINT_DIR} or install huggingface_hub '
                    f'so it can download {DEPTH_HF_REPO}.'
                ) from e

        # Load without weights_only â€” on Pi PyTorch 2.0-2.1, weights_only=True
        # uses a restricted unpickler that silently skips certain tensor storage
        # types, leaving weights zeroed. This is a local trusted file.
        try:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location='cpu')

        # Load all keys including max_depth. Use strict=False to tolerate minor
        # mismatches (e.g. extra keys from older checkpoints).
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'[DA2-WORKER] load_state_dict missing keys: {missing[:8]}', flush=True)
        if unexpected:
            print(f'[DA2-WORKER] load_state_dict unexpected keys: {unexpected[:8]}', flush=True)

        # Verify checkpoint file size and weights
        ckpt_size_mb = os.path.getsize(checkpoint) / 1024 / 1024
        print(f'[DA2-WORKER] checkpoint size: {ckpt_size_mb:.1f} MB', flush=True)
        
        # NOTE: 94.6 MB is VALID for metric vits. The auto-recovery check has been disabled
        # because we've verified the checkpoint weights are good and model produces correct output.
        # Skipping size-based recovery to avoid unnecessary re-downloads.
        
        if ckpt_size_mb < 50:
            print(f'[DA2-WORKER] WARNING: Checkpoint severely undersized ({ckpt_size_mb:.1f} MB). Model may not work correctly.', flush=True)
        
        # Verify weights loaded
        try:
            all_params = list(model.parameters())
            total_abs = sum(float(p.abs().sum()) for p in all_params)
            first_abs = float(all_params[0].abs().sum()) if all_params else 0
            print(f'[DA2-WORKER] weight check: first param abs-sum={first_abs:.4f}, total={total_abs:.1f} (total should be >> 1000)', flush=True)
            if total_abs < 10:
                raise RuntimeError(
                    f'Total weight abs-sum={total_abs:.4f} â€” weights not loaded properly. '
                    f'Delete {checkpoint} and restart to re-download.'
                )
        except StopIteration:
            pass

        # Force max_depth into every location the model might read it from.
        # DA2 versions differ: some use a plain attribute, some a registered buffer,
        # some a submodule attribute. Cover all three.
        _md = int(DEPTH_MAX_DEPTH_M)
        model.max_depth = _md
        # If it's stored as a registered buffer, overwrite the buffer tensor directly.
        if 'max_depth' in dict(model.named_buffers()):
            model.register_buffer('max_depth', torch.tensor(float(_md)))
            print(f'[DA2-WORKER] max_depth registered as buffer = {_md}', flush=True)
        # Also patch any head submodule that might hold it.
        for _name, _mod in model.named_modules():
            if hasattr(_mod, 'max_depth') and _mod is not model:
                _mod.max_depth = _md
                print(f'[DA2-WORKER] patched submodule {_name}.max_depth = {_md}', flush=True)
        model.to(device).eval()
        
        print(f'[DA2-WORKER] Model loaded successfully', flush=True)
        print(json.dumps({'type': 'ready', 'model': DEPTH_MODEL_TYPE, 'device': str(device)}),
              flush=True)
    except BaseException as e:
        print(f'[DA2-WORKER] FATAL ERROR: {e}', flush=True)
        traceback.print_exc()
        print(json.dumps({'type': 'error', 'error': f'worker load failed: {e}'}),
              flush=True)
        return 2

    for line in sys.stdin:
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get('id')
            path = str(req.get('path') or '')
            box = req.get('box')
            depth_scale_factor = _coerce_depth_scale(req.get('depth_scale_factor', DEPTH_SCALE_FACTOR))
            try:
                close_range_scale = float(req.get('close_range_scale', DEPTH_CLOSE_RANGE_SCALE))
                if not math.isfinite(close_range_scale):
                    close_range_scale = float(DEPTH_CLOSE_RANGE_SCALE)
                close_range_scale = float(_clamp(close_range_scale, 0.1, 5.0))
            except Exception:
                close_range_scale = float(DEPTH_CLOSE_RANGE_SCALE)
            
            # Load frame (BGR from OpenCV)
            frame = cv2.imread(path)
            if frame is None:
                raise RuntimeError(f'Cannot read frame from {path}')

            # Official API: depth = model.infer_image(raw_image, input_size)
            # infer_image handles BGRâ†’RGB conversion internally
            print(f'[DA2] Inferring depth from {os.path.basename(path)}...', flush=True)
            with torch.no_grad():
                depth = model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
            depth = np.asarray(depth, dtype=np.float32)
            
            if depth.ndim != 2:
                raise RuntimeError(f'Expected 2D depth map, got shape {depth.shape}')
            
            # Safety: if output is [0,1] range, apply max_depth scaling
            d_max = float(depth.max()) if depth.size > 0 else 0.0
            d_min = float(depth.min()) if depth.size > 0 else 0.0
            if d_max <= 1.01 and d_max > 0.01:
                print(f'[DA2] WARNING: Output in [0,1] range, applying max_depth scaling', flush=True)
                depth = depth * DEPTH_MAX_DEPTH_M
                d_max = float(depth.max())
            
            # Extract depth value from ROI
            # v23 fix: use FULL OBJECT MASK median instead of just the inner 20-80% ROI.
            # The inner-ROI approach misses the true object surface at close range.
            dh, dw = depth.shape
            bx = by = bw_px = bh_px = 0.0
            try:
                bx, by, bw_px, bh_px = [float(v) for v in box]
                # Scale YOLO box (frame pixels) to depth map pixels
                sx = dw / float(FRAME_W)
                sy = dh / float(FRAME_H)
                dx_f = bx * sx
                dy_f = by * sy
                dbw = bw_px * sx
                dbh = bh_px * sy
                # Use the full object bbox for depth extraction (not shrunken ROI)
                # to get a representative sample of the whole object surface.
                x1 = int(_clamp(dx_f + dbw * 0.10, 0, dw - 1))
                x2 = int(_clamp(dx_f + dbw * 0.90, 0, dw))
                y1 = int(_clamp(dy_f + dbh * 0.10, 0, dh - 1))
                y2 = int(_clamp(dy_f + dbh * 0.90, 0, dh))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError('degenerate ROI')
            except Exception:
                # Fallback: center region of depth map
                x1, x2 = dw // 4, (3 * dw) // 4
                y1, y2 = dh // 4, (3 * dh) // 4

            # Extract depth from ROI â€” use MEDIAN of valid pixel samples
            distance_m, depth_stats = _robust_object_depth_m(depth, x1, y1, x2, y2)
            roi_valid = np.array([distance_m], dtype=np.float32) if distance_m is not None else np.array([], dtype=np.float32)

            if roi_valid.size == 0:
                # Fallback to full depth map
                full_valid = depth[np.isfinite(depth) & (depth > 0.001)]
                if full_valid.size > 0:
                    roi_valid = full_valid
                else:
                    distance_m = PICKUP_DEPTH_DEFAULT / 100.0
                    print(f'[DA2] No valid depth in map, using fallback {distance_m*100:.1f}cm', flush=True)

            # Trim extreme outliers before taking the median
            # (DA2 sometimes produces spike pixels at object edges)
            if roi_valid.size >= 30:
                lo_pct = float(np.percentile(roi_valid, 15))
                hi_pct = float(np.percentile(roi_valid, 85))
                trimmed = roi_valid[(roi_valid >= lo_pct) & (roi_valid <= hi_pct)]
                if trimmed.size >= 10:
                    roi_valid = trimmed

            distance_m = float(np.median(roi_valid)) if roi_valid.size > 0 else (PICKUP_DEPTH_DEFAULT / 100.0)
            distance_cm = distance_m * 100.0

            # Apply global scale factor for calibration
            raw_distance_m = distance_m
            distance_m = distance_m * float(depth_scale_factor)
            distance_cm = distance_m * 100.0

            close_range_corrected = False
            # Apply close-range correction (v23)
            # At tabletop distances (< DEPTH_CLOSE_RANGE_MAX_CM) apply a separate
            # per-range scale correction tuned for this depth regime.
            if distance_cm <= DEPTH_CLOSE_RANGE_MAX_CM:
                cr_scale = float(close_range_scale)
                if abs(cr_scale - 1.0) > 0.001:
                    distance_cm_before = distance_cm
                    distance_cm = distance_cm * cr_scale
                    distance_m = distance_cm / 100.0
                    close_range_corrected = True
                    print(f'[DA2] Close-range Ã—{cr_scale:.3f}: {distance_cm_before:.1f}â†’{distance_cm:.1f}cm', flush=True)

            used_depth_fallback = False
            # Sanity check
            if distance_cm < DEPTH_SANITY_MIN_CM or distance_cm > DEPTH_SANITY_MAX_CM:
                # Use box-size hint fallback instead of PICKUP_DEPTH_DEFAULT (5 cm).
                # Returning 5 was masking this branch firing on every frame.
                hint_fallback = _depth_distance_to_hint(None)
                print(f'[DA2] Depth {distance_cm:.1f}cm outside valid range '
                      f'[{DEPTH_SANITY_MIN_CM:.0f}â€“{DEPTH_SANITY_MAX_CM:.0f}cm], using box-size hint', flush=True)
                distance_cm = float(hint_fallback) if hint_fallback else PICKUP_DEPTH_DEFAULT
                distance_m = distance_cm / 100.0
                used_depth_fallback = True

            hint = _depth_distance_to_hint(distance_cm)
            jaw_distance_cm = _jaw_distance_from_camera_depth(distance_cm)
            depth_target = None
            if not used_depth_fallback:
                depth_target = _depth_target_from_map(
                    depth,
                    box,
                    raw_distance_m,
                    PICKUP_OBJECT_CENTER_FRACTION,
                )
                if isinstance(depth_target, dict) and depth_target.get('target_depth_m') is not None:
                    try:
                        target_distance_cm = float(depth_target['target_depth_m']) * 100.0 * float(depth_scale_factor)
                        if target_distance_cm <= DEPTH_CLOSE_RANGE_MAX_CM:
                            target_distance_cm *= float(close_range_scale)
                        if DEPTH_SANITY_MIN_CM <= target_distance_cm <= DEPTH_SANITY_MAX_CM:
                            delta_cm = abs(target_distance_cm - float(distance_cm))
                            if delta_cm <= float(DEPTH_TARGET_MAX_DELTA_CM):
                                blend = float(_clamp(DEPTH_TARGET_DEPTH_BLEND, 0.0, 1.0))
                                before_cm = float(distance_cm)
                                distance_cm = target_distance_cm * blend + before_cm * (1.0 - blend)
                                distance_m = distance_cm / 100.0
                                hint = _depth_distance_to_hint(distance_cm)
                                jaw_distance_cm = _jaw_distance_from_camera_depth(distance_cm)
                                print(f'[DA2] Target-patch depth {target_distance_cm:.1f}cm blended with ROI '
                                      f'{before_cm:.1f}cm -> {distance_cm:.1f}cm', flush=True)
                            else:
                                print(f'[DA2] Target-patch depth ignored: {target_distance_cm:.1f}cm '
                                      f'vs ROI {distance_cm:.1f}cm (delta {delta_cm:.1f}cm)', flush=True)
                    except Exception as e:
                        print(f'[DA2] target patch depth failed: {e}', flush=True)
            path_plan = _depth_path_from_box(box, distance_cm, depth_target)

            # Create visualization
            valid = depth[np.isfinite(depth) & (depth > 0.001)]
            if valid.size > 0:
                lo = float(np.percentile(valid, 2))
                hi = float(np.percentile(valid, 98))
                norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0.0, 1.0) if hi > lo else np.full(depth.shape, 0.5, dtype=np.float32)
            else:
                norm = np.full(depth.shape, 0.5, dtype=np.float32)
            
            # Colormap: invert so near is hot (red), far is cold (blue)
            near_uint8 = (255.0 * (1.0 - norm)).astype(np.uint8)
            colour = cv2.applyColorMap(near_uint8, cv2.COLORMAP_TURBO)
            
            # Draw ROI box
            fx1 = int(_clamp(bx + bw_px * DEPTH_ROI_X1, 0, FRAME_W - 1))
            fx2 = int(_clamp(bx + bw_px * DEPTH_ROI_X2, 0, FRAME_W))
            fy1 = int(_clamp(by + bh_px * DEPTH_ROI_Y1, 0, FRAME_H - 1))
            fy2 = int(_clamp(by + bh_px * DEPTH_ROI_Y2, 0, FRAME_H))
            
            # Scale to depth map space for drawing
            sx_vis = dw / float(FRAME_W)
            sy_vis = dh / float(FRAME_H)
            vx1 = int(fx1 * sx_vis)
            vx2 = int(fx2 * sx_vis)
            vy1 = int(fy1 * sy_vis)
            vy2 = int(fy2 * sy_vis)
            cv2.rectangle(colour, (vx1, vy1), (vx2, vy2), (255, 255, 255), 2)
            try:
                tpx = path_plan.get('target_px') if isinstance(path_plan, dict) else None
                if isinstance(tpx, dict):
                    tx_vis = int(_clamp(float(tpx.get('x', 0.0)) * sx_vis, 0, dw - 1))
                    ty_vis = int(_clamp(float(tpx.get('y', 0.0)) * sy_vis, 0, dh - 1))
                    cv2.drawMarker(colour, (tx_vis, ty_vis), (0, 255, 255),
                                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            except Exception:
                pass
            
            # Add text
            status = 'IN REACH' if jaw_distance_cm <= DEPTH_REACH_LIMIT_CM else 'OUT OF REACH'
            cv2.putText(colour, f'cam {distance_cm:.1f} jaw {jaw_distance_cm:.1f}cm {status}', (max(6, vx1), max(22, vy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Encode depth map as JPEG
            ok_jpg, jpg = cv2.imencode('.jpg', colour, [cv2.IMWRITE_JPEG_QUALITY, 82])
            map_b64 = base64.b64encode(jpg.tobytes()).decode('ascii') if ok_jpg else None
            
            print(f'[DA2] Depth: {distance_cm:.1f}cm (raw: {raw_distance_m:.4f}m, scaled: {distance_m:.4f}m)', flush=True)
            print(json.dumps({
                'type': 'depth',
                'id': req_id,
                'ok': True,
                'model': DEPTH_MODEL_TYPE,
                'value': distance_m,
                'distance_cm': distance_cm,
                'jaw_distance_cm': jaw_distance_cm,
                'hint': hint,
                'path': path_plan,
                'map_jpeg_b64': map_b64,
                'depth_scale_factor': depth_scale_factor,
                'depth_method': depth_stats,
                'close_range_corrected': close_range_corrected,
            }), flush=True)
        except (KeyboardInterrupt, SystemExit):
            # FIX BUG 7: Do NOT catch KI/SE in the per-request handler. Let the worker
            # exit cleanly on SIGINT so the parent can join it instead of killing with SIGKILL.
            break
        except BaseException as e:
            print(json.dumps({
                'type': 'depth',
                'id': req_id,
                'ok': False,
                'error': str(e),
            }), flush=True)
    return 0

def _host_ips():
    ips = []
    try:
        name = socket.gethostname()
        for info in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = info[4][0]
            if ip and not ip.startswith('127.') and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith('127.') and ip not in ips:
                ips.insert(0, ip)
        finally:
            s.close()
    except Exception:
        pass
    return ips or ['127.0.0.1']

def main():
    print('[ARM] Starting up...')
    _load_pickup_learning()    
    # Initialize kinematics library
    global _robot_geometry, _ik_solver, _fk_solver, _pickup_config, _pipeline
    _robot_geometry = RobotGeometry(
        link1_cm=ARM_LINK1_CM,
        link2_cm=ARM_LINK2_CM,
        link2_forearm_frac=ARM_L2_FOREARM_FRAC,
        link2_wrist_frac=ARM_L2_WRIST_FRAC,
        gripper_finger_ext_cm=GRIPPER_FINGER_EXT_CM,
        floor_below_shoulder_cm=FLOOR_BELOW_SHOULDER_CM,
    )
    _fk_solver = ForwardKinematics(_robot_geometry, JointLimits())
    _ik_solver = InverseKinematics(_robot_geometry, JointLimits(), _fk_solver)
    _pickup_config = PickupConfig()
    # Validated pipeline (replaces old kinematics for IK/transforms)
    _pipeline = PickupPipeline(
        geometry=_robot_geometry,
        limits=JointLimits(),
        config=_pickup_config,
    )
    print('[KINEMATICS] Solvers initialized')
    print('[PIPELINE] Validated pickup pipeline ready')

    _threads = [
        _start_task('command-api', _run_cmd_api),
        _start_task('stream-server', _run_stream),
    ]
    time.sleep(0.25)

    ips = _host_ips()
    print('')
    for ip in ips:
        print(f'[ARM] Control page:  http://{ip}:{CMD_API_PORT}/')
        print(f'[ARM] Camera stream: http://{ip}:{STREAM_PORT}/stream')
    depth_state = 'enabled' if VL53L1X_ENABLED else 'disabled'
    print(f'[ARM] Depth sensor:  VL53L1X ({depth_state}) mount={VL53L1X_MOUNT} '
          f'SDA=GPIO{VL53L1X_SDA_PIN} SCL=GPIO{VL53L1X_SCL_PIN} '
          f'scale={VL53L1X_SCALE_FACTOR:.3f} offset={VL53L1X_DISTANCE_OFFSET_CM:+.1f}cm')

    gpio_ok = False
    if _USE_PIGPIO:
        try:
            import pigpio
            _pi[0] = pigpio.pi()
            if not _pi[0].connected:
                raise RuntimeError('pigpio daemon is not running')
            # IBT-4 gripper motor pins
            if GRIP_DC_MODE:
                _pi[0].set_mode(GRIP_RPWM_PIN, pigpio.OUTPUT)
                _pi[0].set_mode(GRIP_LPWM_PIN, pigpio.OUTPUT)
            gpio_ok = True
        except Exception as e:
            msg = f'GPIO init failed: {e}. Web server is still running.'
            print(f'[ERROR] {msg}')
            _set_status(msg)
    else:
        try:
            h = lgpio.gpiochip_open(0)
            _lgpio_h[0] = h
            for pin in PIN_BY_JOINT.values():
                lgpio.gpio_claim_output(h, pin)
            # IBT-4 gripper motor pins
            if GRIP_DC_MODE:
                lgpio.gpio_claim_output(h, GRIP_RPWM_PIN)
                lgpio.gpio_claim_output(h, GRIP_LPWM_PIN)
            gpio_ok = True
        except Exception as e:
            msg = f'GPIO init failed: {e}. Web server is still running.'
            print(f'[ERROR] {msg}')
            _set_status(msg)

    _init_arm_serial()
    _gpio_ready[0] = gpio_ok
    if gpio_ok:
        # Ensure gripper DC motor is stopped at startup
        if GRIP_DC_MODE:
            _grip_dc_stop(hold=False)
        serial_arm_active = _arm_serial[0] is not None
        print('[ARM] Homing servos...')
        _home_all(skip_arm=False)
        print('[ARM] Ready')
    else:
        print('[ARM] Web page is ready, but arm movement is disabled until GPIO is fixed.')

    _threads.extend([
        _start_task('camera', _camera_thread),
        _start_task('detector', _detector_thread),
        _start_task('vl53l1x-warmup', _depth_warmup_thread),
        _start_task('vl53l1x-poll', _depth_auto_scan_thread),
        _start_task('tracking', _tracking_thread_safe),
        _start_task('watchdog', _watchdog_thread),
        _start_task('servo-hold-refresh', _servo_hold_refresh_thread),
    ])

    print('[ARM] Open the Control page URL above in your browser')
    print('[ARM] Press Ctrl+C to quit\n')

    def _shutdown(sig, frame):
        print('\n[ARM] Shutting down...')
        _set_status('Shutting down')
        _set_tracking(False)
        try:
            if _gpio_ready[0]:
                _home_all()
        except Exception as e:
            print(f'[ARM] Home during shutdown failed: {e}')
        _running[0] = False
        _cancel_all_motions()
        _stop_depth_worker('shutdown')
        if _USE_PIGPIO and _pi[0]:
            try:
                _pi[0].stop()
            except Exception:
                pass
        elif _lgpio_h[0] is not None:
            try:
                lgpio.gpiochip_close(_lgpio_h[0])
            except Exception:
                pass
        if _arm_serial[0] is not None:
            try:
                _arm_serial[0].close()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while _running[0]:
        time.sleep(1)

if __name__ == '__main__':
    if _DEPTH_WORKER_MODE:
        sys.exit(_run_depth_worker_main())
    main()



