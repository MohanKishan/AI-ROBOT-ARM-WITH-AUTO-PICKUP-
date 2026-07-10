4-DOF Robot Arm with Real-Time 3D Visualization & Auto Pickup
A Raspberry Pi 4B-controlled 4-DOF robot arm with a web-based 3D interface built on Three.js.

Hardware
Base — DEAO 20kg digital servo on GPIO 16.
Arm — Pro-Range DS5160 60kgcm servo on GPIO 18.
Wrist — GS5508MG 55kg servo on GPIO 23.
Gripper — MG996R DC motor (pot removed) on an IBT-4 (BTS7960) H-bridge. RPWM on GPIO 25, LPWM on GPIO 26. INA219 current sensor on I2C (0x40) for stall detection.
Depth sensor — VL53L1X on I2C (0x29).
Camera — USB webcam for YOLOv8 vision.

Software Stack
Server — Python HTTP on port 8081. Serves a single HTML page with Three.js 3D visualization.
Detection — YOLOv8n for object identification.
IK — 2-link inverse kinematics with full path planning (hover → descend → grip → lift).
FK — Calibrated 6-point interpolated lookup table mapping arm angle to shoulder position.
Gripper control — INA219 monitors current in real time. Motor runs until stall is detected (500mA threshold), then holds torque at 25% duty cycle. Open command coasts freely. 0.8s startup delay ignores inrush current spike.

Key Features
Autonomous pickup — Voice or web UI triggers YOLO detection, IK path planning, and center-grab alignment. Objects beyond 34cm are rejected. FOV-based centering corrects base and arm angles for accurate grabs.
3D digital twin — Live Three.js model mirrors all joints in real time. Ghost arms show planned path during pickup sequences.
Voice commands — Chrome Web Speech API with fuzzy Levenshtein matching filters background noise. Supports "pick up [object]", "home", "wave", "open/close gripper".
Safety — Floor collision avoidance, joint limit enforcement, emergency stop button, servo hold refresh thread.

Architecture
The Pi runs a lightweight HTTP server hosting a single-page app with Three.js 3D rendering, voice recognition, and WebSocket communication. GPIO PWM drives servos directly. The IBT-4 H-bridge provides raw DC motor control for the gripper. Arduino Nano handles PID feedback for the arm joint via USB serial.
