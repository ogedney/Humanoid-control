#!/usr/bin/env python3
import subprocess
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

"""
PPO Hyperparameter Tuning
========================

Systematically tune PPO hyperparameters by running experiments with different parameter values.

Basic Usage:
----------
- List parameters: `python tune_hyperparams.py`
- Tune parameter: `python tune_hyperparams.py --param learning_rate`
- Analyze results: `python tune_hyperparams.py --analyze --param learning_rate`
- Custom directory: `python tune_hyperparams.py --param batch_size --dir custom_results`

Run Organization:
---------------
Experiments are organized in timestamp-based runs:
- New run: `python tune_hyperparams.py --param learning_rate`
  Creates: tune_results/learning_rate/run_YYYYMMDD_HHMMSS/
- Analyze specific run: `python tune_hyperparams.py --analyze --param learning_rate --run run_20250406_102045`
- Analyze all runs: `python tune_hyperparams.py --analyze --param learning_rate`

Experiments automatically stop when performance stabilizes, generating comparative analysis
plots and summary statistics to identify optimal parameter values.
"""

# Define hyperparameter ranges to test
HYPERPARAMS = {
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
    "hidden_dim": [64, 128, 256],
    "batch_size": [64, 128, 256], 
    "clip_param": [0.1, 0.2, 0.3],
    "gamma": [0.97, 0.99, 0.995],
    "lambd": [0.9, 0.95, 0.98],
    "value_coef": [0.25, 0.5, 1.0],
    "entropy_coef": [0.005, 0.01, 0.02, 0.05],
    "max_buffer_size": [4000, 8000, 16000],
    "train_interval": [2000, 4000, 8000]
}

# Minimum number of episodes to run for each configuration
MIN_EPISODES = 20

# Minimum number of steps to run for each configuration
MIN_STEPS = 1000000

# Set up reward stability check - we consider training "long enough" when
# the average reward over the last N episodes has stabilized
STABILITY_WINDOW = 5  # Number of episodes to check for stability
STABILITY_THRESHOLD = 0.1  # Maximum relative change to consider stable

# Fixed seed for reproducibility across runs
# Using a large prime number for good randomness initialization
DEFAULT_SEED = 42

def run_experiment(param_name, param_value, base_dir="tune_results", 
                   min_episodes=MIN_EPISODES, min_steps=MIN_STEPS, use_gui=False,
                   value_index=None, total_values=None, seed=DEFAULT_SEED, run_dir=None):
    """
    Run a single hyperparameter tuning experiment with a specific parameter value.
    
    This function:
    1. Creates a dedicated directory for the experiment
    2. Launches a subprocess running main.py with the specified hyperparameter
    3. Monitors the training process by parsing stdout in real-time
    4. Collects episode rewards and lengths
    5. Automatically terminates training when specific stability criteria are met
    6. Generates plots and summary statistics of the training results
    
    The function determines that training has run "long enough" when:
    - At least min_episodes have completed
    - At least min_steps total environment steps have been taken
    - The coefficient of variation of rewards over the last STABILITY_WINDOW 
      episodes is below STABILITY_THRESHOLD, indicating reward stabilization
    
    Args:
        param_name (str): Name of the hyperparameter to vary (must match those in main.py)
        param_value (float or int): Value to set for the hyperparameter
        base_dir (str): Base directory to store experiment results
        min_episodes (int): Minimum number of episodes to run before checking stability
        min_steps (int): Minimum number of environment steps to run before checking stability
        use_gui (bool): Whether to enable the GUI during the simulation
        value_index (int, optional): Index of the current value being tested
        total_values (int, optional): Total number of values to test
        seed (int): Random seed for reproducible experiments
        run_dir (str, optional): Specific run directory to use within the parameter directory
        
    Returns:
        dict: Results dictionary containing:
            - param_name: Name of the hyperparameter
            - param_value: Value of the hyperparameter
            - episodes: Number of episodes completed
            - total_steps: Total environment steps taken
            - mean_reward: Mean episode reward
            - std_reward: Standard deviation of episode rewards
            - max_reward: Maximum episode reward achieved
            - mean_episode_length: Average episode length in steps
            - loss_data: Dictionary containing neural network loss curves
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided run_dir if specified, otherwise use timestamp directly
    if run_dir:
        # When run_dir is provided, it means we're inside a run directory already
        # So we should create param_value_timestamp inside that run directory
        experiment_dir = os.path.join(base_dir, f"{param_value}_{timestamp}")
    else:
        # Without run_dir, use the old format - this case shouldn't happen with the updated code
        experiment_dir = os.path.join(base_dir, f"{param_name}_{param_value}_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    # CRITICAL: Create a simplified experiment directory for main.py
    # main.py might have issues with deeply nested directories
    main_experiment_dir = os.path.join(experiment_dir, "experiment_data")
    os.makedirs(main_experiment_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "param_name": param_name,
        "param_value": param_value,
        "timestamp": timestamp,
        "min_episodes": min_episodes,
        "min_steps": min_steps,
        "seed": seed
    }
    
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Create model directory within main experiment directory
    model_dir = os.path.join(main_experiment_dir, "ppo_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a marker file to check if the process started correctly
    marker_file = os.path.join(experiment_dir, "process_started.txt")
    with open(marker_file, "w") as f:
        f.write("Process started\n")
    
    # Set up stats tracking file
    stats_file = os.path.join(experiment_dir, "training_stats.txt")
    with open(stats_file, "w") as f:
        f.write("episode,reward,length,steps\n")
    
    # Set up loss tracking file
    loss_file = os.path.join(experiment_dir, "loss_stats.txt")
    with open(loss_file, "w") as f:
        f.write("training_iter,total_steps,policy_loss,value_loss,entropy,total_loss,kl_divergence\n")
    
    # Set up environment variables for hyperparameter
    env = os.environ.copy()
    env["PPO_" + param_name.upper()] = str(param_value)
    
    # Start training process
    progress_info = ""
    if value_index is not None and total_values is not None:
        progress_info = f" ({value_index}/{total_values})"
    
    print(f"Starting experiment with {param_name}={param_value}{progress_info}")
    
    # Set up command with --new-model to start fresh, using the simplified experiment directory
    cmd = ["python", "main.py", "--new-model", "--experiment-dir", main_experiment_dir, "--seed", str(seed)]
    
    # Add --no-gui flag if GUI is disabled
    if not use_gui:
        cmd.append("--no-gui")
    
    # Start the process and timer
    start_time = time.time()
    last_update_time = start_time
    update_interval = 15  # Show update every 15 seconds
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except Exception as e:
        print(f"Error starting process: {e}")
        return {
            "param_name": param_name,
            "param_value": param_value,
            "episodes": 0,
            "total_steps": 0,
            "mean_reward": 0,
            "std_reward": 0,
            "max_reward": 0,
            "mean_episode_length": 0,
            "error": str(e)
        }
    
    # Track stats
    episodes = 0
    steps = 0
    rewards = []
    episode_lengths = []
    
    # Track loss data
    loss_data = {
        'training_iter': [],
        'total_steps': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'total_loss': [],
        'kl_divergence': []  # Track KL divergence between old and new policies
    }
    training_iter = 0
    
    # Create a debug log file to capture all output
    debug_log_path = os.path.join(experiment_dir, "debug_output.log")
    with open(debug_log_path, "w") as debug_log:
        debug_log.write(f"Command: {' '.join(cmd)}\n\n")
        debug_log.write("Output:\n")
    
    # Monitor the process output
    try:
        start_check_time = time.time()
        process_started = False
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            # Log all output to the debug file
            with open(debug_log_path, "a") as debug_log:
                debug_log.write(line)
            
            # Only parse the output, don't print everything to terminal
            
            # Check if process started correctly
            if not process_started:
                # Display first few lines to help with debugging
                print(f"Output: {line.strip()}")
                
                # If we've waited more than 30 seconds and no output, check if the process is still running
                if time.time() - start_check_time > 30:
                    if process.poll() is not None:
                        print(f"Process exited with code {process.poll()} before producing output")
                        break
                    else:
                        # Process is still running but no output - mark as started anyway
                        process_started = True
                        print("Process is running but not producing output - continuing")
            
            # Format periodic status updates
            current_time = time.time()
            elapsed_seconds = int(current_time - start_time)
            if current_time - last_update_time >= update_interval or "Episode" in line and "finished with reward" in line:
                minutes, seconds = divmod(elapsed_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                
                if hours > 0:
                    time_str = f"{hours}h {minutes}m {seconds}s"
                else:
                    time_str = f"{minutes}m {seconds}s"
                
                progress_str = ""
                if value_index is not None and total_values is not None:
                    progress_str = f" ({value_index}/{total_values})"
                    
                print(f"{param_name} being tuned: testing {param_value}{progress_str}, has run for {time_str}, {episodes} episodes, {steps} steps")
                last_update_time = current_time
            
            # Parse episode completion lines
            if "Episode" in line and "finished with reward" in line:
                try:
                    # Extract episode number, reward, and steps using regex for more robustness
                    import re
                    number_pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
                    
                    # Extract numbers using regex instead of specific positions (use raw strings for regex)
                    episode_match = re.search(r'Episode\s+(\d+)', line)
                    reward_match = re.search(r'reward\s+({})'.format(number_pattern), line)
                    steps_match = re.search(r'after\s+(\d+)', line)
                    
                    if episode_match and reward_match and steps_match:
                        episode_num = int(episode_match.group(1))
                        reward = float(reward_match.group(1))
                        length = int(steps_match.group(1))
                        
                        process_started = True  # Mark as started if we've processed an episode
                        
                        episodes = max(episodes, episode_num + 1)  # +1 because episodes are 0-indexed
                        steps += length
                        rewards.append(reward)
                        episode_lengths.append(length)
                        
                        # Log to stats file
                        with open(stats_file, "a") as f:
                            f.write(f"{episode_num},{reward},{length},{steps}\n")
                        
                        # Check if we've run long enough
                        if episodes >= min_episodes and steps >= min_steps:
                            # Check for stability in rewards
                            if len(rewards) >= STABILITY_WINDOW:
                                recent_rewards = rewards[-STABILITY_WINDOW:]
                                mean_reward = np.mean(recent_rewards)
                                if mean_reward != 0:  # Avoid division by zero
                                    std_reward = np.std(recent_rewards)
                                    cv = std_reward / abs(mean_reward)  # Coefficient of variation
                                    
                                    if cv < STABILITY_THRESHOLD:
                                        print(f"Training stabilized after {episodes} episodes, stopping")
                                        break
                    else:
                        print(f"Warning: Could not parse episode line format: {line.strip()}")
                        print(f"Matches found: episode={bool(episode_match)}, reward={bool(reward_match)}, steps={bool(steps_match)}")
                except Exception as e:
                    print(f"Error parsing episode output: {e}")
                    print(f"Raw line: {line.strip()}")
            
            # Parse loss data lines
            if "LOSS_DATA:" in line:
                try:
                    # Extract loss values using regex with more robust number pattern
                    # Pattern matches any number format including scientific notation and negative numbers
                    import re
                    number_pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
                    
                    process_started = True  # Mark as started if we've processed loss data
                    
                    # Use string format instead of f-strings with raw patterns
                    policy_loss = float(re.search(r'policy_loss=({})'.format(number_pattern), line).group(1))
                    value_loss = float(re.search(r'value_loss=({})'.format(number_pattern), line).group(1))
                    entropy = float(re.search(r'entropy=({})'.format(number_pattern), line).group(1))
                    total_loss = float(re.search(r'total_loss=({})'.format(number_pattern), line).group(1))
                    kl_divergence = float(re.search(r'kl_divergence=({})'.format(number_pattern), line).group(1))
                    
                    training_iter += 1
                    
                    # Store loss data
                    loss_data['training_iter'].append(training_iter)
                    loss_data['total_steps'].append(steps)
                    loss_data['policy_loss'].append(policy_loss)
                    loss_data['value_loss'].append(value_loss)
                    loss_data['entropy'].append(entropy)
                    loss_data['total_loss'].append(total_loss)
                    loss_data['kl_divergence'].append(kl_divergence)
                    
                    # Log to loss file
                    with open(loss_file, "a") as f:
                        f.write(f"{training_iter},{steps},{policy_loss},{value_loss},{entropy},{total_loss},{kl_divergence}\n")
                    
                except Exception as e:
                    print(f"Error parsing loss data: {e}")
                    print(f"Raw line: {line.strip()}")
                    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Terminate the process if it's still running
        if process.poll() is None:
            print("Terminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate gracefully, force killing...")
                process.kill()
    
    # Check for empty results (process didn't run properly)
    if not process_started and not rewards:
        print(f"Warning: Process didn't produce any valid output. Check if main.py is running correctly.")
        print(f"Command used: {' '.join(cmd)}")
        print(f"Debug log saved to: {debug_log_path}")
    
    # Copy model files from the simplified experiment directory back to our experiment directory
    try:
        import shutil
        model_files = os.listdir(os.path.join(main_experiment_dir, "ppo_models"))
        for model_file in model_files:
            src = os.path.join(main_experiment_dir, "ppo_models", model_file)
            dst = os.path.join(experiment_dir, "ppo_models", model_file)
            # Make sure the destination directory exists
            os.makedirs(os.path.join(experiment_dir, "ppo_models"), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"Copied {len(model_files)} model files from {main_experiment_dir}/ppo_models to {experiment_dir}/ppo_models")
    except Exception as e:
        print(f"Error copying model files: {e}")
    
    # Show final stats
    elapsed_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    else:
        time_str = f"{minutes}m {seconds}s"
    
    mean_reward = np.mean(rewards) if rewards else 0
    print(f"Completed experiment: {param_name}={param_value}, total time: {time_str}")
    print(f"  Episodes: {episodes}, Steps: {steps}, Mean reward: {mean_reward:.2f}")
    
    # Generate summary plots
    if rewards:
        plt.figure(figsize=(12, 16))
        
        # Plot rewards
        plt.subplot(3, 1, 1)
        plt.plot(rewards)
        plt.title(f"Rewards for {param_name}={param_value}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(3, 1, 2)
        plt.plot(episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        
        # Plot loss curves if available
        if loss_data['training_iter']:
            plt.subplot(3, 1, 3)
            plt.plot(loss_data['training_iter'], loss_data['policy_loss'], label='Policy Loss')
            plt.plot(loss_data['training_iter'], loss_data['value_loss'], label='Value Loss')
            plt.plot(loss_data['training_iter'], loss_data['total_loss'], label='Total Loss')
            plt.title("Training Losses")
            plt.xlabel("Training Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "training_curve.png"))
        
        # Generate additional plots for entropy and KL divergence
        if loss_data['training_iter']:
            plt.figure(figsize=(12, 10))
            
            # Entropy plot
            plt.subplot(2, 1, 1)
            plt.plot(loss_data['training_iter'], loss_data['entropy'])
            plt.title("Entropy")
            plt.xlabel("Training Iteration")
            plt.ylabel("Entropy")
            plt.grid(True)
            
            # KL divergence plot
            plt.subplot(2, 1, 2)
            plt.plot(loss_data['training_iter'], loss_data['kl_divergence'])
            plt.title("KL Divergence (Policy Update Magnitude)")
            plt.xlabel("Training Iteration")
            plt.ylabel("KL Divergence")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, "policy_metrics.png"))
            plt.close()
    
    # Calculate average loss values
    avg_policy_loss = np.mean(loss_data['policy_loss']) if loss_data['policy_loss'] else 0
    avg_value_loss = np.mean(loss_data['value_loss']) if loss_data['value_loss'] else 0
    avg_entropy = np.mean(loss_data['entropy']) if loss_data['entropy'] else 0
    avg_total_loss = np.mean(loss_data['total_loss']) if loss_data['total_loss'] else 0
    avg_kl_divergence = np.mean(loss_data['kl_divergence']) if loss_data['kl_divergence'] else 0
    
    # Save final results summary
    results = {
        "param_name": param_name,
        "param_value": param_value,
        "episodes": episodes,
        "total_steps": steps,
        "mean_reward": np.mean(rewards) if rewards else 0,
        "std_reward": np.std(rewards) if rewards else 0,
        "max_reward": np.max(rewards) if rewards else 0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        "loss_metrics": {
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_entropy": avg_entropy,
            "avg_total_loss": avg_total_loss,
            "avg_kl_divergence": avg_kl_divergence
        }
    }
    
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_results(base_dir="tune_results", param_name=None, run_dir=None):
    """
    Analyze and compare results across multiple completed experiments.
    
    This function:
    1. Recursively searches the results directory for experiment results
    2. Loads and processes all results.json files found
    3. Groups results by parameter name
    4. Generates comparative bar charts showing mean and max rewards for each parameter value
    5. Prints performance summary tables sorted by mean reward (best first)
    
    The analysis focuses on comparing different values of the same hyperparameter
    to identify which values led to the best performance.
    
    Args:
        base_dir (str): Base directory containing experiment result folders
        param_name (str, optional): If provided, only analyze experiments for this specific parameter.
            If None, analyze all experiments found in the directory.
        run_dir (str, optional): If provided, only analyze experiments within this run directory.
            
    Returns:
        None, but produces:
        - Comparative bar charts saved as {param_name}_comparison.png in the base_dir
        - Printed summary tables showing performance statistics for each parameter value
    """
    results = []
    
    # If we're analyzing a specific parameter with a run directory
    if param_name and run_dir:
        search_path = os.path.join(base_dir, param_name, run_dir)
    # If we're analyzing just a specific parameter
    elif param_name:
        search_path = os.path.join(base_dir, param_name)
    # If we're analyzing everything
    else:
        search_path = base_dir
    
    # Function to recursively search for result files
    def find_result_files(directory):
        found_results = []
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isdir(item_path):
                # Check if this directory contains a results.json file
                results_file = os.path.join(item_path, "results.json")
                if os.path.exists(results_file):
                    # If it does, load the results
                    with open(results_file, "r") as f:
                        result = json.load(f)
                    # Filter by param_name if specified
                    if param_name is None or result.get("param_name") == param_name:
                        found_results.append(result)
                else:
                    # If not, search its subdirectories
                    found_results.extend(find_result_files(item_path))
        
        return found_results
    
    if os.path.exists(search_path):
        results = find_result_files(search_path)
    
    if not results:
        print(f"No results found to analyze in {search_path}")
        return
    
    # Group results by parameter name
    param_results = {}
    for result in results:
        p_name = result["param_name"]
        if p_name not in param_results:
            param_results[p_name] = []
        param_results[p_name].append(result)
    
    # Generate comparative plots for each parameter
    for p_name, p_results in param_results.items():
        # Determine the output directory for plots
        if run_dir:
            output_dir = os.path.join(base_dir, p_name, run_dir)
        else:
            output_dir = os.path.join(base_dir, p_name)
        
        # Sort by parameter value for plotting
        p_results.sort(key=lambda x: x["param_value"])
        
        values = [str(r["param_value"]) for r in p_results]
        mean_rewards = [r["mean_reward"] for r in p_results]
        max_rewards = [r["max_reward"] for r in p_results]
        
        # Reward plot
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(values))
        width = 0.35
        
        plt.bar(x - width/2, mean_rewards, width, label='Mean Reward')
        plt.bar(x + width/2, max_rewards, width, label='Max Reward')
        
        plt.xlabel(f'{p_name} Value')
        plt.ylabel('Reward')
        plt.title(f'Performance by {p_name}')
        plt.xticks(x, values)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{p_name}_reward_comparison.png"))
        plt.close()
        
        # Loss metrics plots if available
        if all('loss_metrics' in r for r in p_results):
            # Policy loss
            policy_losses = [r.get('loss_metrics', {}).get('avg_policy_loss', 0) for r in p_results]
            value_losses = [r.get('loss_metrics', {}).get('avg_value_loss', 0) for r in p_results]
            total_losses = [r.get('loss_metrics', {}).get('avg_total_loss', 0) for r in p_results]
            entropy_values = [r.get('loss_metrics', {}).get('avg_entropy', 0) for r in p_results]
            kl_divergences = [r.get('loss_metrics', {}).get('avg_kl_divergence', 0) for r in p_results]
            
            # Loss comparison plot
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, policy_losses, width, label='Policy Loss')
            plt.bar(x + width/2, value_losses, width, label='Value Loss')
            plt.xlabel(f'{p_name} Value')
            plt.ylabel('Average Loss')
            plt.title(f'Loss Metrics by {p_name}')
            plt.xticks(x, values)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{p_name}_loss_comparison.png"))
            plt.close()
            
            # Entropy comparison plot
            plt.figure(figsize=(10, 6))
            plt.bar(x, entropy_values, width, label='Entropy')
            plt.xlabel(f'{p_name} Value')
            plt.ylabel('Average Entropy')
            plt.title(f'Entropy by {p_name}')
            plt.xticks(x, values)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{p_name}_entropy_comparison.png"))
            plt.close()
            
            # KL divergence comparison plot
            plt.figure(figsize=(10, 6))
            plt.bar(x, kl_divergences, width, label='KL Divergence')
            plt.xlabel(f'{p_name} Value')
            plt.ylabel('Average KL Divergence')
            plt.title(f'Policy Update Magnitude by {p_name}')
            plt.xticks(x, values)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{p_name}_kl_divergence_comparison.png"))
            plt.close()
        
    # Print summary table
    print("\nPerformance Summary:")
    for p_name, p_results in param_results.items():
        print(f"\n{p_name}:")
        print(f"{'Value':<10} {'Mean Reward':<15} {'Max Reward':<15} {'Episodes':<10} {'Steps':<10}")
        print("-" * 60)
        
        # Sort by mean reward (best first)
        p_results.sort(key=lambda x: x["mean_reward"], reverse=True)
        
        for r in p_results:
            print(f"{str(r['param_value']):<10} {r['mean_reward']:<15.2f} {r['max_reward']:<15.2f} {r['episodes']:<10} {r['total_steps']:<10}")
        
        # Print loss metrics if available
        if all('loss_metrics' in r for r in p_results):
            print("\nLoss Metrics:")
            print(f"{'Value':<10} {'Policy Loss':<15} {'Value Loss':<15} {'Total Loss':<15} {'Entropy':<10} {'KL Div':<10}")
            print("-" * 75)
            
            # Same order as reward table (already sorted)
            for r in p_results:
                metrics = r.get('loss_metrics', {})
                policy_loss = metrics.get('avg_policy_loss', 0)
                value_loss = metrics.get('avg_value_loss', 0)
                total_loss = metrics.get('avg_total_loss', 0)
                entropy = metrics.get('avg_entropy', 0)
                kl_divergence = metrics.get('avg_kl_divergence', 0)
                
                print(f"{str(r['param_value']):<10} {policy_loss:<15.6f} {value_loss:<15.6f} {total_loss:<15.6f} {entropy:<10.6f} {kl_divergence:<10.6f}")

def move_existing_files(param_dir):
    """
    Move existing experiment files in a parameter directory to a subfolder.
    
    This function:
    1. Creates a 'run1' subfolder if it doesn't exist
    2. Moves all experiment files and folders to this subfolder
    3. Preserves any analysis files at the parameter directory level
    
    Args:
        param_dir (str): Path to the parameter directory
    """
    if not os.path.exists(param_dir):
        return
        
    # Create run1 directory if it doesn't exist
    run1_dir = os.path.join(param_dir, "run1")
    os.makedirs(run1_dir, exist_ok=True)
    
    # Get list of items to move (all folders named like param_value_timestamp)
    items_to_move = []
    for item in os.listdir(param_dir):
        item_path = os.path.join(param_dir, item)
        # Only move directories that match the experiment naming pattern 
        # and comparison PNG files
        if (os.path.isdir(item_path) and not item.startswith("run")) or \
           (item.endswith(".png") and "comparison" in item):
            items_to_move.append(item)
    
    # Move items to run1 directory
    for item in items_to_move:
        src_path = os.path.join(param_dir, item)
        dst_path = os.path.join(run1_dir, item)
        
        try:
            os.rename(src_path, dst_path)
            print(f"Moved {item} to {run1_dir}")
        except Exception as e:
            print(f"Error moving {item}: {e}")

def tune_single_param(param_name, base_dir="tune_results", use_gui=False, seed=DEFAULT_SEED):
    """
    Systematically tune a single hyperparameter by running experiments for all configured values.
    
    This function:
    1. Validates that the requested parameter exists in the HYPERPARAMS dictionary
    2. Creates a dedicated directory for this parameter's experiments
    3. Creates a new run directory for this series of experiments
    4. Sequentially runs experiments for each value defined in HYPERPARAMS
    5. After all values are tested, analyzes the results to identify the best value
    
    The function automates the entire process of testing multiple values of a single
    hyperparameter while keeping all other parameters at their default values.
    
    Args:
        param_name (str): Name of the hyperparameter to tune, must exist in HYPERPARAMS dict
        base_dir (str): Base directory to store all results
        use_gui (bool): Whether to enable the GUI during the simulation
        seed (int): Random seed for reproducible experiments
            
    Returns:
        list: List of result dictionaries from all experiments run for this parameter,
              each containing performance metrics as described in run_experiment()
              
    Raises:
        Prints an error message if the parameter name is not found in HYPERPARAMS
    """
    if param_name not in HYPERPARAMS:
        print(f"Unknown hyperparameter: {param_name}")
        print(f"Available hyperparameters: {list(HYPERPARAMS.keys())}")
        return
    
    values = HYPERPARAMS[param_name]
    total_values = len(values)
    print(f"Tuning {param_name} with values: {values}")
    
    # Create parameter directory
    param_dir = os.path.join(base_dir, param_name)
    os.makedirs(param_dir, exist_ok=True)
    
    # Check if there are existing experiment files and move them to run1 if needed
    move_existing_files(param_dir)
    
    # Create a new run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{timestamp}"
    run_path = os.path.join(param_dir, run_dir)
    os.makedirs(run_path, exist_ok=True)
    
    print(f"Created new run directory: {run_path}")
    
    # Run experiments for each value
    results = []
    for i, value in enumerate(values):
        # Pass run_path as the base_dir so experiments are created inside the run directory
        result = run_experiment(param_name, value, base_dir=run_path, use_gui=use_gui, 
                               value_index=i+1, total_values=total_values, seed=seed)
        results.append(result)
        
    # Analyze results just for this run
    analyze_results(base_dir, param_name, run_dir)
    
    return results

def run_multiple_values(param_name, param_values, base_dir="tune_results", 
                        min_episodes=MIN_EPISODES, min_steps=MIN_STEPS, use_gui=False,
                        run_index=None):
    """
    Run experiments for multiple parameter values and compare results.
    
    Args:
        param_name: Name of the parameter to tune
        param_values: List of values to test for the parameter
        base_dir: Base directory for saving results
        min_episodes: Minimum number of episodes to run for each value
        min_steps: Minimum number of steps to run for each value
        use_gui: Whether to show the GUI during simulation
        run_index: Optional run index number to use (instead of timestamp)
    
    Returns:
        Dictionary containing results for each parameter value
    """
    if run_index is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"run_{timestamp}"
    else:
        run_dir = f"run{run_index}"
    
    # Create parameter-specific results directory with run subdirectory
    param_dir = os.path.join(base_dir, param_name)
    run_path = os.path.join(param_dir, run_dir)
    os.makedirs(run_path, exist_ok=True)
    
    # Move existing files to run1 if they exist and this is a new run
    if run_index is None:
        move_existing_files(param_dir)
    
    # Write experiment config
    config = {
        "param_name": param_name,
        "param_values": param_values,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "min_episodes": min_episodes,
        "min_steps": min_steps,
        "use_gui": use_gui
    }
    
    with open(os.path.join(run_path, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate seeds for each experiment for better comparison
    seeds = [random.randint(1, 10000) for _ in range(len(param_values))]
    
    # Run experiments for each parameter value
    results = []
    for i, value in enumerate(param_values):
        # Use the same random seed for all experiments for fair comparison
        seed = seeds[0]  # Use the same seed for all values
        
        # Set value index for progress reporting
        value_index = i + 1
        total_values = len(param_values)
        
        # Run experiment in the run directory
        result = run_experiment(
            param_name=param_name,
            param_value=value,
            base_dir=run_path,  # Use run_path as the base directory
            min_episodes=min_episodes,
            min_steps=min_steps,
            use_gui=use_gui,
            value_index=value_index,
            total_values=total_values,
            seed=seed,
            run_dir=run_path  # Pass run_dir to indicate it's part of a multi-experiment run
        )
        
        results.append(result)
    
    # Save combined results
    combined_results = {
        "param_name": param_name,
        "results": results
    }
    
    with open(os.path.join(run_path, "combined_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Create comparative visualizations
    create_comparison_visualizations(results, param_name, run_path)
    
    # Print summary table
    print("\nResults Summary:")
    print("-" * 80)
    print(f"Parameter: {param_name}")
    print("-" * 80)
    headers = ["Value", "Episodes", "Steps", "Mean Reward", "Max Reward", "Entropy", "KL Div"]
    rows = []
    
    for result in results:
        value = result["param_value"]
        episodes = result["episodes"]
        steps = result["total_steps"]
        mean_reward = result["mean_reward"]
        max_reward = result["max_reward"]
        
        # Check if we have loss data
        if "avg_entropy" in result:
            entropy = f"{result['avg_entropy']:.4f}"
        else:
            entropy = "N/A"
            
        if "avg_kl_divergence" in result:
            kl_div = f"{result['avg_kl_divergence']:.6f}"
        else:
            kl_div = "N/A"
        
        rows.append([value, episodes, steps, f"{mean_reward:.2f}", f"{max_reward:.2f}", entropy, kl_div])
    
    # Format and print table
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    # Print headers
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in rows:
        formatted_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(formatted_row)
    
    print("-" * 80)
    
    return combined_results

def main():
    """
    Main entry point for hyperparameter tuning script.
    
    This function:
    1. Parses command-line arguments to determine the operation mode
    2. Creates the results directory if it doesn't exist
    3. Dispatches to the appropriate function based on the requested operation:
       - With --param: Tunes a specific parameter by running experiments for all its values
       - With --analyze: Analyzes existing experiment results without running new ones
       - With no arguments: Lists available parameters that can be tuned
    
    Command-line Arguments:
        --param (str): Parameter name to tune
        --analyze (flag): Analyze existing results
        --dir (str): Results directory (default: "tune_results")
        --gui (flag): Enable GUI visualization (disabled by default for faster training)
        --no-gui (flag): Disable GUI visualization (default)
        --seed (int): Random seed for reproducible experiments (default: 42)
        --run (str): Specific run directory to analyze (for --analyze)
        --test (flag): Run in quick test mode with minimal episodes/steps
        --value (float): Single value to test (used with --test)
    
    Using from Python code:
        # Tune a specific parameter
        import tune_hyperparams
        results = tune_hyperparams.tune_single_param("learning_rate")
        
        # Analyze existing results
        tune_hyperparams.analyze_results()
        
        # Run an individual experiment with custom stability settings
        result = tune_hyperparams.run_experiment("entropy_coef", 0.03, 
                                               min_episodes=30, min_steps=150000)
        
        # Analyze results from a specific run
        tune_hyperparams.analyze_results(
            base_dir="tune_results", 
            param_name="learning_rate", 
            run_dir="run_20250406_102045"
        )
    
    Examples:
        # Tune learning rate (creates a new run directory)
        python tune_hyperparams.py --param learning_rate
        
        # Analyze all learning rate experiments across all runs
        python tune_hyperparams.py --analyze --param learning_rate
        
        # Analyze only the experiments from a specific run
        python tune_hyperparams.py --analyze --param learning_rate --run run_20250406_102045
        
        # Run with visualization enabled (useful for debugging)
        python tune_hyperparams.py --param hidden_dim --gui
    
    Notes:
        Parameters are tuned sequentially, not in parallel. Each value of a parameter
        is tested completely before moving to the next one. This is intentional to
        avoid resource contention and ensure consistent test conditions.
        
    Returns:
        None, but initiates parameter tuning or result analysis based on arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PPO")
    parser.add_argument("--param", type=str, help="Parameter to tune")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--dir", type=str, default="tune_results", help="Results directory")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI visualization (default)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible experiments")
    parser.add_argument("--run", type=str, help="Specific run directory to analyze (for --analyze)")
    parser.add_argument("--test", action="store_true", help="Run in quick test mode with minimal episodes/steps")
    parser.add_argument("--value", type=float, help="Single value to test (used with --test)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gui and args.no_gui:
        parser.error("Cannot specify both --gui and --no-gui")
        
    # Determine GUI setting (default is no GUI)
    use_gui = args.gui and not args.no_gui
    
    # Create base results directory
    os.makedirs(args.dir, exist_ok=True)
    
    if args.analyze:
        # Just analyze existing results
        analyze_results(args.dir, args.param, args.run)
    elif args.param and args.test:
        # Test mode for quick testing of a single value
        if args.value is None:
            # Use first value in the list if not specified
            value = HYPERPARAMS[args.param][0]
            print(f"No value specified, using first value in list: {value}")
        else:
            value = args.value
        
        print(f"TESTING MODE: Running quick test for {args.param}={value}")
        result = run_experiment(
            param_name=args.param,
            param_value=value,
            base_dir=os.path.join(args.dir, "test"),
            min_episodes=1,  # Very minimal for testing
            min_steps=100,   # Very minimal for testing
            use_gui=use_gui,  # Use the determined GUI setting
            seed=args.seed
        )
        print("\nTest Results:")
        print(f"Parameter: {args.param}, Value: {value}")
        print(f"Episodes: {result['episodes']}, Steps: {result['total_steps']}")
        print(f"Mean Reward: {result['mean_reward']:.2f}, Max Reward: {result['max_reward']:.2f}")
    elif args.param:
        # Tune a specific parameter
        tune_single_param(args.param, args.dir, use_gui=use_gui, seed=args.seed)
    else:
        # Show available parameters
        print("Available hyperparameters:")
        for param, values in HYPERPARAMS.items():
            print(f"  {param}: {values}")
        print("\nUse --param to specify which one to tune")
        print("Add --gui to enable visualization (slower but helpful for debugging)")
        print("Add --no-gui to disable visualization (default)")
        print("Add --test for a quick test run with minimal episodes/steps")
        print("Add --value to specify a single value to test (requires --test)")
        print(f"Default seed value: {DEFAULT_SEED} (use --seed to change)")
        print("Use --run to specify a specific run directory when analyzing results")

if __name__ == "__main__":
    main() 