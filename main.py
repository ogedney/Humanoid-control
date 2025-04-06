import pybullet as pb
import time
import numpy as np
import os
from helpers import setup_humanoid_for_control
from controllers import create_controller
import argparse

def get_env_float(name, default):
    """Get a float value from environment variables with a default fallback."""
    value = os.environ.get(f"PPO_{name.upper()}")
    return float(value) if value is not None else default

def get_env_int(name, default):
    """Get an integer value from environment variables with a default fallback."""
    value = os.environ.get(f"PPO_{name.upper()}")
    return int(value) if value is not None else default

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a humanoid robot using PPO')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI visualization')
    parser.add_argument('--new-model', action='store_true', help='Start with a fresh model instead of loading existing one')
    parser.add_argument('--experiment-dir', type=str, help='Directory for experiment results and models')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible training')
    args = parser.parse_args()
    
    try:
        # Setup environment and get necessary components
        physicsClient, robotId, joint_indices, joint_names, update_camera = setup_humanoid_for_control(use_gui=not args.no_gui)
        
        # Create a PPO controller for humanoid walking
        controller_type = 'ppo'
        
        # Set different max forces for different joints
        joint_max_forces = []
        for name in joint_names:
            if 'hip' in name.lower():
                joint_max_forces.append(1000)  # Higher force for hip joints
            else:
                joint_max_forces.append(100)   # Standard force for all other joints
        
        # Read hyperparameters from environment variables with defaults
        hidden_dim = get_env_int("hidden_dim", 256)
        learning_rate = get_env_float("learning_rate", 3e-5)
        batch_size = get_env_int("batch_size", 128)
        clip_param = get_env_float("clip_param", 0.1)
        gamma = get_env_float("gamma", 0.99)
        lambd = get_env_float("lambd", 0.95)
        max_buffer_size = get_env_int("max_buffer_size", 8000)
        value_coef = get_env_float("value_coef", 0.5)
        entropy_coef = get_env_float("entropy_coef", 0.01)
        train_interval = get_env_int("train_interval", 4000)
        
        # Get seed from environment variable if not provided via command line
        seed = args.seed
        if seed is None:
            seed_env = os.environ.get("PPO_SEED")
            if seed_env is not None:
                seed = int(seed_env)
        
        # Use experiment directory if provided
        model_dir = os.path.join(args.experiment_dir, "ppo_models") if args.experiment_dir else "ppo_models"
        
        # Log the hyperparameters being used
        print("=== Training with hyperparameters ===")
        print(f"hidden_dim: {hidden_dim}")
        print(f"learning_rate: {learning_rate}")
        print(f"batch_size: {batch_size}")
        print(f"clip_param: {clip_param}")
        print(f"gamma: {gamma}")
        print(f"lambd: {lambd}")
        print(f"max_buffer_size: {max_buffer_size}")
        print(f"value_coef: {value_coef}")
        print(f"entropy_coef: {entropy_coef}")
        print(f"train_interval: {train_interval}")
        print(f"model_dir: {model_dir}")
        print(f"seed: {seed}")
        print("====================================")
        
        controller = create_controller(
            controller_type=controller_type,
            robot_id=robotId,
            joint_indices=joint_indices,
            joint_names=joint_names,
            max_force=100,         # Default max force (used as fallback)
            joint_max_forces=joint_max_forces,  # Joint-specific max forces
            # PPO specific parameters
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            clip_param=clip_param,
            gamma=gamma,
            lambd=lambd,
            skip_load=args.new_model,
            max_buffer_size=max_buffer_size,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            model_dir=model_dir,
            train_interval=train_interval,  # Pass training interval to controller
            seed=seed  # Pass the seed to the controller
        )
        
        print(f"Starting simulation with {controller_type} control...")
        print("Press 'x' to exit, 's' to save model")
        
        # Main simulation loop
        max_steps = 10_000_000  # 10 million steps
        for i in range(max_steps):
            # Update camera based on user input (only if GUI is enabled)
            if update_camera is not None:
                update_camera()
            
            # Check if robot is still stable
            try:
                pos, _ = pb.getBasePositionAndOrientation(robotId)
                if i % 1000 == 0:  # Print less frequently to reduce output
                    print(f"Step {i}/{max_steps}: Robot at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    print(f"Episode rewards - Total: {controller.episode_reward:.2f}")
                    print(f"  Forward: {controller.episode_forward_reward:.2f}")
                    print(f"  Height: {controller.episode_height_penalty:.2f}")
                    print(f"  Energy: {controller.episode_energy_penalty:.2f}")
                    print(f"  Velocity: {controller.episode_velocity_reward:.2f}")
                    print(f"  Orientation: {controller.episode_orientation_penalty:.2f}")
                    print("----------------------------------------")
                    
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
            
            # Sleep to make it real-time (only if GUI is enabled)
            if update_camera is not None:
                time.sleep(1./480.)  # Faster than real-time for training
            
            # Check for quit key (only if GUI is enabled)
            if update_camera is not None:
                keys = pb.getKeyboardEvents()
                if ord('x') in keys and keys[ord('x')] & pb.KEY_WAS_TRIGGERED:
                    print("X key pressed, exiting")
                    break
                
                # Save model explicitly on s press
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