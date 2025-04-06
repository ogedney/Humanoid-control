#!/usr/bin/env python3
import subprocess
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
Hyperparameter Tuning Script for PPO Controller
===============================================

This script provides tools to systematically tune hyperparameters for the PPO reinforcement
learning algorithm by running multiple experiments with different parameter values and
analyzing the results.

Command Line Usage Examples:
---------------------------

1. List available hyperparameters:
   ```
   python tune_hyperparams.py
   ```

2. Tune a specific hyperparameter (e.g., learning_rate):
   ```
   python tune_hyperparams.py --param learning_rate
   ```
   This will run experiments for all values of learning_rate defined in HYPERPARAMS.

3. Analyze results from previous tuning runs:
   ```
   python tune_hyperparams.py --analyze
   ```

4. Analyze results for a specific parameter only:
   ```
   python tune_hyperparams.py --analyze --param learning_rate
   ```

5. Use a custom directory for results:
   ```
   python tune_hyperparams.py --param batch_size --dir custom_results
   ```

Each experiment will:
- Run until reaching stability criteria or minimum requirements
- Generate plots of rewards and episode lengths
- Save detailed statistics for later analysis
- Automatically stop when the performance stabilizes

After all experiments complete, a comparative analysis showing the best parameter
values will be generated.
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
MIN_STEPS = 100000

# Set up reward stability check - we consider training "long enough" when
# the average reward over the last N episodes has stabilized
STABILITY_WINDOW = 5  # Number of episodes to check for stability
STABILITY_THRESHOLD = 0.1  # Maximum relative change to consider stable

# Fixed seed for reproducibility across runs
# Using a large prime number for good randomness initialization
DEFAULT_SEED = 42

def run_experiment(param_name, param_value, base_dir="tune_results", 
                   min_episodes=MIN_EPISODES, min_steps=MIN_STEPS, use_gui=False,
                   value_index=None, total_values=None, seed=DEFAULT_SEED):
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
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{param_name}_{param_value}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
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
    
    # Create model directory within experiment directory
    model_dir = os.path.join(experiment_dir, "ppo_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up stats tracking file
    stats_file = os.path.join(experiment_dir, "training_stats.txt")
    with open(stats_file, "w") as f:
        f.write("episode,reward,length,steps\n")
    
    # Set up environment variables for hyperparameter
    env = os.environ.copy()
    env["PPO_" + param_name.upper()] = str(param_value)
    
    # Start training process
    progress_info = ""
    if value_index is not None and total_values is not None:
        progress_info = f" ({value_index}/{total_values})"
    
    print(f"Starting experiment with {param_name}={param_value}{progress_info}")
    
    # Set up command with --new-model to start fresh and include seed
    cmd = ["python", "main.py", "--new-model", "--experiment-dir", experiment_dir, "--seed", str(seed)]
    
    # Add --no-gui flag if GUI is disabled
    if not use_gui:
        cmd.append("--no-gui")
    
    # Start the process and timer
    start_time = time.time()
    last_update_time = start_time
    update_interval = 15  # Show update every 15 seconds
    
    process = subprocess.Popen(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Track stats
    episodes = 0
    steps = 0
    rewards = []
    episode_lengths = []
    
    # Monitor the process output
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            # Only parse the output, don't print everything to terminal
            
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
                    # Extract episode number, reward, and steps
                    parts = line.split()
                    episode_num = int(parts[1])
                    reward = float(parts[5])
                    length = int(parts[7])
                    
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
                except Exception as e:
                    print(f"Error parsing output: {e}")
                    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Terminate the process if it's still running
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
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
    
    # Generate summary plot
    if rewards:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(rewards)
        plt.title(f"Rewards for {param_name}={param_value}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "training_curve.png"))
    
    # Save final results summary
    results = {
        "param_name": param_name,
        "param_value": param_value,
        "episodes": episodes,
        "total_steps": steps,
        "mean_reward": np.mean(rewards) if rewards else 0,
        "std_reward": np.std(rewards) if rewards else 0,
        "max_reward": np.max(rewards) if rewards else 0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0
    }
    
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_results(base_dir="tune_results", param_name=None):
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
            
    Returns:
        None, but produces:
        - Comparative bar charts saved as {param_name}_comparison.png in the base_dir
        - Printed summary tables showing performance statistics for each parameter value
    """
    results = []
    
    # Iterate through experiment directories
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        # Check if results file exists
        results_file = os.path.join(exp_path, "results.json")
        if not os.path.exists(results_file):
            continue
            
        # Load results
        with open(results_file, "r") as f:
            result = json.load(f)
            
        # Filter by param_name if specified
        if param_name is None or result.get("param_name") == param_name:
            results.append(result)
    
    if not results:
        print("No results found to analyze")
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
        # Sort by parameter value for plotting
        p_results.sort(key=lambda x: x["param_value"])
        
        values = [str(r["param_value"]) for r in p_results]
        mean_rewards = [r["mean_reward"] for r in p_results]
        max_rewards = [r["max_reward"] for r in p_results]
        
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
        plt.savefig(os.path.join(base_dir, f"{p_name}_comparison.png"))
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

def tune_single_param(param_name, base_dir="tune_results", use_gui=False, seed=DEFAULT_SEED):
    """
    Systematically tune a single hyperparameter by running experiments for all configured values.
    
    This function:
    1. Validates that the requested parameter exists in the HYPERPARAMS dictionary
    2. Creates a dedicated directory for this parameter's experiments
    3. Sequentially runs experiments for each value defined in HYPERPARAMS
    4. After all values are tested, analyzes the results to identify the best value
    
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
    
    # Create result directory
    param_dir = os.path.join(base_dir, param_name)
    os.makedirs(param_dir, exist_ok=True)
    
    # Run experiments for each value
    results = []
    for i, value in enumerate(values):
        result = run_experiment(param_name, value, param_dir, use_gui=use_gui, 
                               value_index=i+1, total_values=total_values, seed=seed)
        results.append(result)
        
    # Analyze results
    analyze_results(param_dir, param_name)
    
    return results

def main():
    """
    Main entry point for the hyperparameter tuning script.
    
    This function:
    1. Parses command-line arguments to determine the operation mode
    2. Creates the results directory if it doesn't exist
    3. Dispatches to the appropriate function based on the requested operation:
       - With --param: Tunes a specific parameter by running experiments for all its values
       - With --analyze: Analyzes existing experiment results without running new ones
       - With no arguments: Lists available parameters that can be tuned
    
    Command-line Arguments:
        --param (str): Parameter name to tune
        --analyze (flag): Analyze existing results without running new experiments
        --dir (str): Results directory (default: "tune_results")
        --gui (flag): Enable GUI visualization (disabled by default for faster training)
        --seed (int): Random seed for reproducible experiments (default: 42)
    
    Using from Python code:
        # Tune a specific parameter
        import tune_hyperparams
        results = tune_hyperparams.tune_single_param("learning_rate")
        
        # Analyze existing results
        tune_hyperparams.analyze_results()
        
        # Run an individual experiment with custom stability settings
        result = tune_hyperparams.run_experiment("entropy_coef", 0.03, 
                                               min_episodes=30, min_steps=150000)
    
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible experiments")
    
    args = parser.parse_args()
    
    # Create base results directory
    os.makedirs(args.dir, exist_ok=True)
    
    if args.analyze:
        # Just analyze existing results
        analyze_results(args.dir, args.param)
    elif args.param:
        # Tune a specific parameter
        tune_single_param(args.param, args.dir, use_gui=args.gui, seed=args.seed)
    else:
        # Show available parameters
        print("Available hyperparameters:")
        for param, values in HYPERPARAMS.items():
            print(f"  {param}: {values}")
        print("\nUse --param to specify which one to tune")
        print("Add --gui to enable visualization (slower but helpful for debugging)")
        print(f"Default seed value: {DEFAULT_SEED} (use --seed to change)")

if __name__ == "__main__":
    main() 