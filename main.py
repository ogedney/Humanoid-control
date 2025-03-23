import pybullet as pb
import time
import numpy as np
from helpers import setup_humanoid_for_control
from controllers import create_controller

def main():
    try:
        # Setup environment and get necessary components
        physicsClient, robotId, joint_indices, joint_names, update_camera = setup_humanoid_for_control()
        
        # Create a controller 
        controller_type = 'torque'
        controller = create_controller(
            controller_type=controller_type,
            robot_id=robotId,
            joint_indices=joint_indices,
            joint_names=joint_names,
            max_force=20.0
        )
        
        # Set target positions (all zeros)
        controller.set_target_positions(np.zeros(len(joint_indices)))
        
        print(f"Starting simulation with {controller_type} control...")
        
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
            
            # Update controller
            controller.update()
            
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