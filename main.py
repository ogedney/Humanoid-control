import pybullet as pb
import time
import numpy as np
from helpers import setup_humanoid_for_control
from controllers import create_controller

def main():
    try:
        # Setup environment and get necessary components
        physicsClient, robotId, joint_indices, joint_names, update_camera = setup_humanoid_for_control()
        
        # Create a PPO controller for humanoid walking
        controller_type = 'ppo'
        controller = create_controller(
            controller_type=controller_type,
            robot_id=robotId,
            joint_indices=joint_indices,
            joint_names=joint_names,
            max_force=20.0,
            # PPO specific parameters
            hidden_dim=64,        # Size of hidden layers
            learning_rate=3e-4,   # Learning rate
            batch_size=64,        # Batch size for updates
            clip_param=0.2,       # PPO clipping parameter
            gamma=0.99,           # Discount factor
            lambd=0.95,           # GAE lambda
            train_interval=2000,  # Steps between training
        )
        
        print(f"Starting simulation with {controller_type} control...")
        
        # Main simulation loop
        for i in range(100000):  # Extended number of steps for learning
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
            
            # Sleep to make it real-time (or faster/slower if needed)
            # Reduce sleep time for faster training
            time.sleep(1./480.)  # Faster than real-time for training
            
            # Check for quit key - use 'x' instead of 'q' to avoid conflicts
            keys = pb.getKeyboardEvents()
            if ord('x') in keys and keys[ord('x')] & pb.KEY_WAS_TRIGGERED:
                print("X key pressed, exiting")
                break
            
            # Save model explicitly on q press
            if ord('s') in keys and keys[ord('s')] & pb.KEY_WAS_TRIGGERED:
                controller.save_model("manual_save")
                print("Model saved manually")
        
        print("Simulation completed successfully")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        try:
            # Save model on exit
            if 'controller' in locals() and hasattr(controller, 'save_model'):
                controller.save_model("final")
                print("Final model saved")
                
            pb.disconnect()
            print("Disconnected from PyBullet")
        except:
            pass

if __name__ == "__main__":
    main() 