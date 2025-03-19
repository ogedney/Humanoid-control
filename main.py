import pybullet as pb
import time
import numpy as np
from helpers import setup_camera_controls, setup_simulation_environment, get_atlas_initial_pose
from control import HumanoidController

# Setup simulation and get object IDs
physicsClient, planeId, humanoidId = setup_simulation_environment()

# Initialize the controller
controller = HumanoidController(humanoidId)

# Initialize lists to store state history
states_history = []

# Set initial standing pose for Atlas
# Get the initial joint positions from helpers for target positions
target_positions = np.zeros(controller.num_controlled_joints)
joint_name_to_target = get_atlas_initial_pose()

# Apply the target positions
for i, name in enumerate(controller.joint_names):
    if name in joint_name_to_target:
        target_positions[i] = joint_name_to_target[name]

camera_updater = setup_camera_controls()

# PD control parameters for Atlas
kp = 500.0  # Higher proportional gain for Atlas
kd = 50.0   # Higher derivative gain for Atlas

for i in range(10000):
    camera_updater()
    
    # Update state
    controller.update_state()
    
    # Compute and apply torques using PD control with gravity compensation
    torques = controller.compute_pd_torques(target_positions, kp=kp, kd=kd)
    controller.apply_torques(torques)
    
    # Store the state
    states_history.append(controller.get_all_states())
    
    pb.stepSimulation()
    time.sleep(1./240.)

print("Simulation complete. Final state:")
print(f"Position: {states_history[-1]['base_position']}")
print(f"Number of timesteps recorded: {len(states_history)}")

pb.disconnect()
