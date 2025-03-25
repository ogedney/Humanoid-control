import pybullet as pb
import math
import pybullet_data
import numpy as np
import time

def setup_camera_controls():
    """
    Creates and returns a camera control function for the PyBullet debug visualizer.
    
    The returned function enables interactive camera control with the following keys:
    - Arrow keys: Rotate camera view (left/right for yaw, up/down for pitch)
    - +/-: Zoom in/out
    - WASD: Pan camera position horizontally relative to view direction
    - Q/E: Pan camera position vertically
    
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
    camera_distance = 7.0
    camera_yaw = 45
    camera_pitch = -30
    target = [0, 0, 2.0]

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
        
        # WASD controls are ONLY for camera movement (no wireframe toggle)
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

def add_coordinate_frame():
    """
    Add a coordinate frame visualization to the simulation.
    
    Creates three colored lines at the origin (0,0,0):
    - Red line: X-axis (forward/backward direction)
    - Green line: Y-axis (left/right direction)
    - Blue line: Z-axis (up/down direction)
    """
    # Create a small coordinate frame (1 meter long)
    length = 1.0
    thickness = 0.02
    
    # X-axis (red)
    pb.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[length, 0, 0],
        lineColorRGB=[1, 0, 0],
        lineWidth=thickness
    )
    
    # Y-axis (green)
    pb.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, length, 0],
        lineColorRGB=[0, 1, 0],
        lineWidth=thickness
    )
    
    # Z-axis (blue)
    pb.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, 0, length],
        lineColorRGB=[0, 0, 1],
        lineWidth=thickness
    )

def setup_humanoid_for_control(use_gui=True):
    """
    Sets up the PyBullet simulation and prepares a humanoid robot specifically for position control.
    
    This function:
    1. Initializes PyBullet with or without GUI
    2. Configures visualization settings
    3. Sets up physics parameters for stability
    4. Loads the ground plane and humanoid robot
    5. Sets up camera view and controls (if GUI is enabled)
    6. Prepares joints with appropriate damping for stable control
    7. Lets the robot settle initially
    
    Args:
        use_gui (bool): Whether to run with GUI visualization (default: True)
    
    Returns:
        tuple: (physicsClient, robotId, joint_indices, joint_names, update_camera)
            - physicsClient: PyBullet physics client ID
            - robotId: ID of the loaded humanoid robot
            - joint_indices: List of indices for controllable joints
            - joint_names: List of names for controllable joints
            - update_camera: Function to update camera based on user input (None if GUI is disabled)
    """
    # Connect to PyBullet with or without GUI
    if use_gui:
        physicsClient = pb.connect(pb.GUI)
        print("Connected to PyBullet with GUI")
    else:
        physicsClient = pb.connect(pb.DIRECT)
        print("Connected to PyBullet without GUI")
    
    # Configure visualization
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1 if use_gui else 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1 if use_gui else 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    
    # Configure physics for stability
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)
    pb.setPhysicsEngineParameter(fixedTimeStep=1/240.0, numSolverIterations=10, numSubSteps=1)
    
    # Load ground plane
    planeId = pb.loadURDF("plane.urdf")
    print("Loaded ground plane")
    
    # Add coordinate frame visualization only if GUI is enabled
    if use_gui:
        add_coordinate_frame()
        print("Added coordinate frame visualization")
    
    # Load humanoid robot
    startPos = [0, 0, 3.5]
    startOrientation = pb.getQuaternionFromEuler([np.pi/2, 0, 0])
    robotId = pb.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation,
                         flags=pb.URDF_MAINTAIN_LINK_ORDER)
    print(f"Loaded humanoid robot, ID: {robotId}")
    
    # Set initial camera view and setup camera controls only if GUI is enabled
    update_camera = None
    if use_gui:
        pb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1.0]
        )
        update_camera = setup_camera_controls()
        print("Camera controls enabled: Arrow keys to rotate, +/- to zoom, WASD to pan, Q/E for up/down")
        print("NOTE: W key is for camera movement only, wireframe toggle disabled")
    
    # Get joint information
    joint_indices = []
    joint_names = []
    
    for i in range(pb.getNumJoints(robotId)):
        info = pb.getJointInfo(robotId, i)
        joint_type = info[2]
        
        if joint_type != pb.JOINT_FIXED:
            joint_indices.append(i)
            joint_names.append(info[1].decode('utf-8'))
            print(f"Joint {i}: {info[1].decode('utf-8')}")
            
            # Add damping for stability
            pb.changeDynamics(
                robotId, 
                i, 
                jointDamping=5.0,
                linearDamping=0.9,
                angularDamping=0.9,
                maxJointVelocity=10.0
            )
            
            # Reset joint state to zeros with zero velocity
            pb.resetJointState(robotId, i, 0, 0)
    
    # Wait for visualization to initialize if GUI is enabled
    if use_gui:
        time.sleep(1.0)
    
    print("Environment setup complete")
    print("Setup complete - robot ready for control")
    return physicsClient, robotId, joint_indices, joint_names, update_camera

def reset_humanoid(robot_id):
    """
    Resets the humanoid robot to its initial position and orientation.
    
    Args:
        robot_id: PyBullet ID of the humanoid robot
        
    Returns:
        None
    """
    # Initial position and orientation
    start_pos = [0, 0, 3.5]
    start_orientation = pb.getQuaternionFromEuler([np.pi/2, 0, 0])
    
    # Reset base position and orientation
    pb.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)
    
    # Reset joint states to zero position and velocity
    for i in range(pb.getNumJoints(robot_id)):
        joint_info = pb.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        
        if joint_type != pb.JOINT_FIXED:
            pb.resetJointState(robot_id, i, 0, 0)
    
    # Allow a short stabilization period
    for _ in range(10):
        pb.stepSimulation()
        
    print("Robot reset to initial position")
    
    return