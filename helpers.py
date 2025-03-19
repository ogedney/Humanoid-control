import pybullet as pb
import math
import pybullet_data
import numpy as np

def setup_camera_controls():
    """
    Creates and returns a camera control function for the PyBullet debug visualizer.
    
    The returned function enables interactive camera control with the following keys:
    - Arrow keys: Rotate camera view (left/right for yaw, up/down for pitch)
    - +/-: Zoom in/out
    - WASD: Pan camera position horizontally relative to view direction
    - Q/E: Pan camera position vertically
    - 1: Toggle wireframe mode
    
    Returns:
        function: An update_camera function that should be called in the simulation loop
                 to process camera control inputs.
    
    Example:
        update_camera = setup_camera_controls()
        while simulation_running:
            update_camera()
            # ... rest of simulation loop
    """
    # Initial camera settings
    camera_distance = 6.0  # Increased for better initial view
    camera_yaw = 45
    camera_pitch = -30  # Adjusted for better viewing angle
    target = [0, 0, 3]  # [x, y, z] - Set z to 1.0 to look at approximate COM height

    def update_camera():
        keys = pb.getKeyboardEvents()
        nonlocal camera_distance, camera_yaw, camera_pitch, target
        
        # Zoom with +/-
        if ord('=') in keys and keys[ord('=')] & pb.KEY_WAS_TRIGGERED:  # '+' key
            camera_distance = max(0.1, camera_distance - 0.4)
        if ord('-') in keys and keys[ord('-')] & pb.KEY_WAS_TRIGGERED:
            camera_distance += 0.4
            
        # Rotate with arrow keys
        if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
            camera_yaw -= 1
        if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
            camera_yaw += 1
        if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
            camera_pitch -= 1
        if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
            camera_pitch += 1

        # Pan with WASD relative to camera view direction
        pan_speed = 0.1
        yaw_rad = camera_yaw * math.pi / 180  # Convert to radians
        
        # Calculate forward and right vectors (parallel to ground)
        forward = np.array([-math.cos(yaw_rad), math.sin(yaw_rad), 0])
        right = np.array([math.sin(yaw_rad), math.cos(yaw_rad), 0])
        
        # Toggle wireframe with '1' key
        if ord('1') in keys and keys[ord('1')] & pb.KEY_WAS_TRIGGERED:
            current_mode = pb.getDebugVisualizerConfigParameters()[pb.COV_ENABLE_WIREFRAME]
            pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME, 1 - current_mode)
        
        if ord('w') in keys and keys[ord('w')] & pb.KEY_IS_DOWN:  # Forward
            target += forward * pan_speed
        if ord('s') in keys and keys[ord('s')] & pb.KEY_IS_DOWN:  # Backward
            target -= forward * pan_speed
        if ord('a') in keys and keys[ord('a')] & pb.KEY_IS_DOWN:  # Left
            target -= right * pan_speed
        if ord('d') in keys and keys[ord('d')] & pb.KEY_IS_DOWN:  # Right
            target += right * pan_speed
        
        # Up/Down panning with Q/E
        if ord('q') in keys and keys[ord('q')] & pb.KEY_IS_DOWN:  # Up
            target[2] += pan_speed
        if ord('e') in keys and keys[ord('e')] & pb.KEY_IS_DOWN:  # Down
            target[2] -= pan_speed
            
        # Update camera
        pb.resetDebugVisualizerCamera(
            camera_distance,
            camera_yaw,
            camera_pitch,
            target
        )
    
    return update_camera 

def get_atlas_initial_pose():
    """
    Returns a dictionary of initial joint positions for the humanoid robot
    to achieve a stable standing pose.
    """
    return {
        "waist": 0.0,
        "chest": 0.0,
        "neck": 0.0,
        "right_hip_x": 0.0,
        "right_hip_z": 0.0,
        "right_hip_y": 0.0,
        "right_knee": 0.0,
        "right_ankle": 0.0,
        "right_shoulder1": 0.0,
        "right_shoulder2": 0.0,
        "right_elbow": 0.0,
        "left_hip_x": 0.0,
        "left_hip_z": 0.0,
        "left_hip_y": 0.0,
        "left_knee": 0.0,
        "left_ankle": 0.0,
        "left_shoulder1": 0.0,
        "left_shoulder2": 0.0,
        "left_elbow": 0.0
    }

def setup_simulation_environment():
    """
    Sets up the PyBullet simulation environment with a plane and humanoid.
    Returns the IDs for the created objects and applies initial pose to the humanoid.
    """
    physicsClient = pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)  # Standard gravity
    
    # Setup plane
    planeId = pb.loadURDF("plane.urdf")
    pb.changeDynamics(planeId, -1, lateralFriction=1.0, restitution=0.0)
    
    # Setup humanoid
    startPos = [0, 0, 3.5]  # Lower starting position
    startOrientation = pb.getQuaternionFromEuler([math.pi/2, 0, 0])  # Rotate 90 degrees around X to make it stand
    
    # Load humanoid model
    humanoidId = pb.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation)
    
    # Setup joint dynamics for humanoid
    num_joints = pb.getNumJoints(humanoidId)
    for joint in range(num_joints):
        info = pb.getJointInfo(humanoidId, joint)
        joint_type = info[2]
        
        # Only adjust controllable joints
        if joint_type != pb.JOINT_FIXED:
            pb.changeDynamics(humanoidId, 
                             joint,
                             jointDamping=0.5,
                             angularDamping=0.5,
                             lateralFriction=0.2)
    
    # Apply initial pose to humanoid
    initial_pose = get_atlas_initial_pose()
    for joint in range(num_joints):
        info = pb.getJointInfo(humanoidId, joint)
        joint_name = info[1].decode('utf-8')
        if joint_name in initial_pose:
            pb.resetJointState(humanoidId, joint, initial_pose[joint_name])
    
    # Let the robot settle in the initial pose
    for _ in range(50):
        pb.stepSimulation()
    
    return physicsClient, planeId, humanoidId 