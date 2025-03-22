import pybullet as pb
import time
import numpy as np
from helpers import setup_camera_controls, setup_simulation_environment, get_atlas_initial_pose
from control import HumanoidController

# Setup simulation and get object IDs
physicsClient, planeId, humanoidId = setup_simulation_environment()

# Give the GUI time to initialize properly 
time.sleep(0.5)

# Initialize the controller
controller = HumanoidController(humanoidId)

# Initialize lists to store state history
states_history = []

# Set initial standing pose for Atlas
target_positions = np.zeros(controller.num_controlled_joints)
joint_name_to_target = get_atlas_initial_pose()

# Apply the target positions
for i, name in enumerate(controller.joint_names):
    if name in joint_name_to_target:
        target_positions[i] = joint_name_to_target[name]

camera_updater = setup_camera_controls()

# Control parameters - more gentle for position control
kp = 400.0  # Position gain
kd = 40.0   # Velocity gain

# Let the simulation run
for i in range(10000):
    # Update camera and controller state
    camera_updater()
    controller.update_state()
    
    # Use position control to maintain pose
    controller.set_positions(target_positions, kp=kp, kd=kd)
    
    # Record state history
    states_history.append(controller.get_all_states())
    
    # Step simulation
    pb.stepSimulation()
    time.sleep(1./240.)

print("Simulation complete. Final state:")
print(f"Position: {states_history[-1]['base_position']}")
print(f"Number of timesteps recorded: {len(states_history)}")

pb.disconnect()
