"""
Humanoid Robot Control Simulation using PPO

This script allows training a Proximal Policy Optimization (PPO) agent to control a humanoid robot
model in the PyBullet physics simulator. It also allows loading and running pre-trained models.

Usage Examples:

1.  **Train a new model:**
    ```bash
    python main.py --experiment-dir my_training_run --no-gui
    ```
    - Creates a new model or loads the latest checkpoint from `my_training_run/ppo_models`.
    - Runs the simulation without GUI for faster training.
    - Saves checkpoints and best model to `my_training_run/ppo_models`.
    - Hyperparameters can be set via environment variables (e.g., `PPO_LEARNING_RATE=0.0001 python main.py ...`).

2.  **Start training from scratch (ignore existing models):**
    ```bash
    python main.py --experiment-dir fresh_run --new-model --no-gui
    ```
    - Ignores any existing models in `fresh_run/ppo_models`.
    - Starts training a completely new model.

3.  **Load and run a specific pre-trained model (no training/saving):**
    ```bash
    python main.py --load-model-path path/to/your/ppo_model_best.pt --run-only --gui
    ```
    - Loads the specified `.pt` file.
    - Runs the simulation with the GUI enabled.
    - Disables training and saving of any new models.

4.  **Load a specific model and continue training:**
    ```bash
    python main.py --load-model-path path/to/your/ppo_model_checkpoint.pt --experiment-dir continue_run --no-gui
    ```
    - Loads the specified `.pt` file (including optimizer state if possible).
    - Continues training, saving new checkpoints/best models to `continue_run/ppo_models`.

Command-line Arguments:
    --no-gui:             Run simulation without the graphical interface (faster for training).
                          GUI is enabled by default if this flag is absent.
    --new-model:          Force start of training with a new, randomly initialized model,
                          ignoring any existing models in the experiment directory.
                          Cannot be used with --load-model-path.
    --experiment-dir DIR: Directory to save/load training checkpoints, best models, and logs.
                          Defaults to 'ppo_models' if not specified during training.
                          Ignored if --run-only is used.
                          Required if loading a specific model (--load-model-path) but *not* using --run-only.
    --seed SEED:          Integer random seed for reproducibility.
    --load-model-path PATH: Path to a specific `.pt` model file to load.
                          If provided, this specific model is loaded, overriding the default
                          behavior of loading from the experiment directory.
                          Cannot be used with --new-model.
    --run-only:           Load a model (requires --load-model-path) and run it in the simulation
                          without performing any training updates or saving any new models.
                          Useful for evaluating or demonstrating a trained agent.

Hyperparameters:
    Hyperparameters (e.g., learning rate, batch size) are primarily controlled via environment
    variables prefixed with `PPO_` (e.g., `PPO_LEARNING_RATE`, `PPO_BATCH_SIZE`).
    See `get_env_float` and `get_env_int` functions for details and default values.
"""
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
    parser = argparse.ArgumentParser(description='Train or run a humanoid robot using PPO')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI visualization')
    parser.add_argument('--new-model', action='store_true', help='Start with a fresh model instead of loading existing one')
    parser.add_argument('--experiment-dir', type=str, help='Directory for experiment results and models (used for training/saving)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible training')
    parser.add_argument('--load-model-path', type=str, help='Path to a specific .pt model file to load and run')
    parser.add_argument('--run-only', action='store_true', help='Load model and run without training or saving further models')
    args = parser.parse_args()

    # Validate arguments
    if args.load_model_path and args.new_model:
        parser.error("Cannot use --load-model-path and --new-model together.")
    if args.run_only and not args.load_model_path:
        parser.error("--run-only requires --load-model-path to be specified.")
    if args.run_only and args.experiment_dir:
        print("Warning: --experiment-dir is ignored when using --run-only.")
        args.experiment_dir = None # Nullify experiment dir in run-only mode

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
        
        # Determine model directory - used only if not loading a specific path or if saving is enabled
        model_dir = None
        if not args.run_only:
             # Use experiment directory if provided and not in run-only mode
             model_dir = os.path.join(args.experiment_dir, "ppo_models") if args.experiment_dir else "ppo_models"
        elif args.load_model_path and not args.run_only:
             # If loading a specific model but not run_only, save checkpoints relative to the loaded model's dir?
             # For now, let's require experiment_dir if saving is intended when loading a specific model.
             if not args.experiment_dir:
                   parser.error("--experiment-dir must be specified to save checkpoints when loading a specific model without --run-only.")
             model_dir = os.path.join(args.experiment_dir, "ppo_models")

        # Log the hyperparameters being used
        print("=== Configuration ===")
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
        if model_dir:
             print(f"model_dir (for saving): {model_dir}")
        if args.load_model_path:
             print(f"load_model_path: {args.load_model_path}")
        if args.run_only:
             print("run_only: True (Training and saving disabled)")
        print("=====================")
        
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
            skip_load=args.new_model, # skip_load is handled internally based on load_model_path now
            max_buffer_size=max_buffer_size,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            model_dir=model_dir, # Pass potentially None model_dir
            train_interval=train_interval,
            seed=seed,
            load_model_path=args.load_model_path, # Pass the specific path
            run_only=args.run_only # Pass the run_only flag
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
                
                # Save model explicitly on s press (only if not in run_only mode)
                if ord('s') in keys and keys[ord('s')] & pb.KEY_WAS_TRIGGERED:
                    if not args.run_only:
                        controller.save_model("manual_save")
                        print("Model saved manually")
                    else:
                        print("Manual saving disabled in --run-only mode.")
        
        print("Simulation completed successfully")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        try:
            # Save model on exit (only if not in run_only mode)
            if not args.run_only and 'controller' in locals() and hasattr(controller, 'save_model'):
                 # Check if controller thinks saving is enabled (it should respect run_only flag internally)
                 if hasattr(controller, 'run_only') and not controller.run_only:
                       controller.save_model("final")
                       print("Final model saved")

            pb.disconnect()
            print("Disconnected from PyBullet")
        except Exception as disconnect_error: # Catch specific errors if needed
             print(f"Error during cleanup: {disconnect_error}")

if __name__ == "__main__":
    main() 