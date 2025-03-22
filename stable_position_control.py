import pybullet as pb
import time
import numpy as np
from helpers import setup_humanoid_for_control

def main():
    try:
        # Setup environment and get necessary components
        physicsClient, robotId, joint_indices, joint_names, update_camera = setup_humanoid_for_control()
        
        # Position control parameters
        kp = 0.3  # Very low gains for stability
        kd = 0.5  # Higher damping than position gain
        max_force = 20.0  # Low force to avoid instability
        
        # Set target positions (all zeros)
        target_positions = np.zeros(len(joint_indices))
        
        print("Starting simulation with position control...")
        
        # Main simulation loop
        for i in range(10000):
            # Update camera based on user input
            update_camera()
            
            # Check if robot is still stable
            try:
                pos, _ = pb.getBasePositionAndOrientation(robotId)
                if i % 100 == 0:
                    print(f"Step {i}: Robot at position {pos}")
                    
                    # If position is NaN, early stop
                    if np.isnan(pos[0]) or np.isnan(pos[1]) or np.isnan(pos[2]):
                        print("ERROR: Robot position is NaN, stopping")
                        break
            except Exception as e:
                print(f"ERROR: Could not get robot position: {e}")
                break
            
            # Apply position control to all joints with very conservative parameters
            for j, joint_idx in enumerate(joint_indices):
                try:
                    # Get current joint state
                    joint_state = pb.getJointState(robotId, joint_idx)
                    current_pos = joint_state[0]
                    current_vel = joint_state[1]
                    
                    # Very small step towards target
                    if abs(current_pos - target_positions[j]) > 0.01:
                        # Only move a tiny bit toward target each step
                        direction = 1 if target_positions[j] > current_pos else -1
                        target_this_step = current_pos + direction * 0.001
                        
                        pb.setJointMotorControl2(
                            bodyUniqueId=robotId,
                            jointIndex=joint_idx,
                            controlMode=pb.POSITION_CONTROL,
                            targetPosition=target_this_step,  # Incremental movement
                            positionGain=kp,
                            velocityGain=kd,
                            force=max_force
                        )
                except Exception as e:
                    print(f"ERROR at joint {joint_idx}: {e}")
                    continue
            
            # Step simulation 
            pb.stepSimulation()
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Check for quit key - use 'x' instead of 'q' to avoid conflicts
            keys = pb.getKeyboardEvents()
            if ord('x') in keys and keys[ord('x')] & pb.KEY_WAS_TRIGGERED:
                print("X key pressed, exiting")
                break
        
        print("Simulation completed successfully")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        try:
            pb.disconnect()
            print("Disconnected from PyBullet")
        except:
            pass

if __name__ == "__main__":
    main() 