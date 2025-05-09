import pybullet as pb
import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
from collections import deque
import time
from helpers import reset_humanoid

class Controller(ABC):
    """
    Abstract base class for humanoid robot controllers.
    """
    def __init__(self, robot_id, joint_indices, joint_names):
        """
        Initialize the controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
        """
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.joint_names = joint_names
        self.target_positions = np.zeros(len(joint_indices))
        
    @abstractmethod
    def update(self):
        """
        Update the controller. This method should be called in the simulation loop.
        """
        pass
    
    def set_target_positions(self, positions):
        """
        Set target positions for joints.
        
        Args:
            positions: Array of target positions for joints
        """
        if len(positions) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} positions, got {len(positions)}")
        self.target_positions = positions

class PositionController(Controller):
    """
    Position control for humanoid joints.
    """
    def __init__(self, robot_id, joint_indices, joint_names, kp=0.3, kd=0.5, max_force=20.0):
        """
        Initialize the position controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
            kp: Position gain
            kd: Velocity gain
            max_force: Maximum force applied to reach target position
        """
        super().__init__(robot_id, joint_indices, joint_names)
        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        
    def update(self):
        """
        Apply position control to all joints with conservative parameters.
        """
        for j, joint_idx in enumerate(self.joint_indices):
            try:
                # Get current joint state
                joint_state = pb.getJointState(self.robot_id, joint_idx)
                current_pos = joint_state[0]
                current_vel = joint_state[1]
                
                # Very small step towards target
                if abs(current_pos - self.target_positions[j]) > 0.01:
                    # Only move a tiny bit toward target each step
                    direction = 1 if self.target_positions[j] > current_pos else -1
                    target_this_step = current_pos + direction * 0.001
                    
                    pb.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=pb.POSITION_CONTROL,
                        targetPosition=target_this_step,  # Incremental movement
                        positionGain=self.kp,
                        velocityGain=self.kd,
                        force=self.max_force
                    )
            except Exception as e:
                print(f"ERROR at joint {joint_idx}: {e}")
                continue

class VelocityController(Controller):
    """
    Velocity control for humanoid joints.
    """
    def __init__(self, robot_id, joint_indices, joint_names, kd=0.5, max_force=20.0):
        """
        Initialize the velocity controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
            kd: Velocity gain
            max_force: Maximum force applied to reach target velocity
        """
        super().__init__(robot_id, joint_indices, joint_names)
        self.kd = kd
        self.max_force = max_force
        self.target_velocities = np.zeros(len(joint_indices))
        
    def set_target_velocities(self, velocities):
        """
        Set target velocities for joints.
        
        Args:
            velocities: Array of target velocities for joints
        """
        if len(velocities) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} velocities, got {len(velocities)}")
        self.target_velocities = velocities
        
    def update(self):
        """
        Apply velocity control to all joints.
        """
        for j, joint_idx in enumerate(self.joint_indices):
            try:
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.VELOCITY_CONTROL,
                    targetVelocity=self.target_velocities[j],
                    velocityGain=self.kd,
                    force=self.max_force
                )
            except Exception as e:
                print(f"ERROR at joint {joint_idx}: {e}")
                continue

class TorqueController(Controller):
    """
    Torque control for humanoid joints.
    """
    def __init__(self, robot_id, joint_indices, joint_names, max_force=20.0):
        """
        Initialize the torque controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
            max_force: Maximum force applied
        """
        super().__init__(robot_id, joint_indices, joint_names)
        self.max_force = max_force
        self.target_torques = np.zeros(len(joint_indices))
        
    def set_target_torques(self, torques):
        """
        Set target torques for joints.
        
        Args:
            torques: Array of target torques for joints
        """
        if len(torques) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} torques, got {len(torques)}")
        self.target_torques = torques
        
    def update(self):
        """
        Apply torque control to all joints.
        """
        for j, joint_idx in enumerate(self.joint_indices):
            try:
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.TORQUE_CONTROL,
                    force=self.target_torques[j]
                )
            except Exception as e:
                print(f"ERROR at joint {joint_idx}: {e}")
                continue

class PDController(Controller):
    """
    PD control for humanoid joints.
    """
    def __init__(self, robot_id, joint_indices, joint_names, kp=0.3, kd=0.5, max_force=20.0):
        """
        Initialize the PD controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
            kp: Position gain
            kd: Velocity gain
            max_force: Maximum force applied
        """
        super().__init__(robot_id, joint_indices, joint_names)
        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        self.target_velocities = np.zeros(len(joint_indices))
        
    def set_target_velocities(self, velocities):
        """
        Set target velocities for joints.
        
        Args:
            velocities: Array of target velocities for joints
        """
        if len(velocities) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} velocities, got {len(velocities)}")
        self.target_velocities = velocities
        
    def update(self):
        """
        Apply PD control to all joints.
        """
        for j, joint_idx in enumerate(self.joint_indices):
            try:
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.PD_CONTROL,
                    targetPosition=self.target_positions[j],
                    targetVelocity=self.target_velocities[j],
                    positionGain=self.kp,
                    velocityGain=self.kd,
                    force=self.max_force
                )
            except Exception as e:
                print(f"ERROR at joint {joint_idx}: {e}")
                continue

class PPONetwork(nn.Module):
    """
    Neural network for PPO policy and value functions with separate networks.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # Policy network (completely separate) - reduced to 2 hidden layers
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy output layer
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value network (completely separate)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Value output layer
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize with larger weights to encourage more movement
            nn.init.orthogonal_(module.weight, gain=100)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the separate policy and value networks."""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        # Policy path
        policy_features = self.policy_network(x)
        action_mean = self.policy_mean(policy_features)
        # Initialize with larger standard deviation to encourage exploration
        action_log_std = torch.ones_like(action_mean) * 0.5
        action_std = torch.exp(action_log_std)
        
        # Value path (completely separate)
        value_features = self.value_network(x)
        value = self.value(value_features)
        
        return action_mean, action_std, value

class PPOController(Controller):
    """
    PPO-based controller for humanoid walking using torque control.
    """
    def __init__(self, robot_id, joint_indices, joint_names, max_force=20.0,
                 hidden_dim=256, learning_rate=3e-4, batch_size=64,
                 clip_param=0.2, gamma=0.99, lambd=0.95,
                 value_coef=0.5, entropy_coef=0.01, max_buffer_size=4000,
                 model_dir="ppo_models", skip_load=False, joint_max_forces=None,
                 train_interval=4000, seed=None,
                 load_model_path=None, run_only=False):
        """
        Initialize the PPO controller.
        
        Args:
            robot_id: PyBullet ID of the robot
            joint_indices: List of joint indices to control
            joint_names: List of joint names corresponding to joint_indices
            max_force: Maximum torque applied to joints (used if joint_max_forces is None)
            hidden_dim: Hidden dimension of the neural networks
            learning_rate: Learning rate for optimization
            batch_size: Batch size for PPO updates
            clip_param: PPO clipping parameter
            gamma: Discount factor
            lambd: GAE lambda parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_buffer_size: Maximum size of the experience buffer
            model_dir: Directory to save models
            skip_load: If True, start with a fresh model instead of loading existing one.
                     Ignored if load_model_path is provided.
            joint_max_forces: List of maximum forces for each joint. If None, max_force is used for all joints.
            train_interval: Number of steps between training updates.
            seed: Random seed for reproducible network initialization (None for random).
            load_model_path (str, optional): Path to a specific model file (.pt) to load.
                                            Overrides model_dir and skip_load for loading.
            run_only (bool, optional): If True, disable training and saving. Requires load_model_path.
        """
        super().__init__(robot_id, joint_indices, joint_names)
        # Increase max force significantly for more movement
        self.max_force = max_force * 5.0
        
        # Set random seed if provided for reproducibility
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            # Extra determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize joint-specific max forces if provided
        if joint_max_forces is not None:
            if len(joint_max_forces) != len(joint_indices):
                raise ValueError(f"Expected {len(joint_indices)} joint max forces, got {len(joint_max_forces)}")
            self.joint_max_forces = torch.tensor(joint_max_forces).float() * 5.0  # Apply same scaling
        else:
            self.joint_max_forces = None
        
        # PPO parameters
        self.clip_param = clip_param
        self.gamma = gamma
        self.lambd = lambd
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        
        # State and action dimensions
        self.joint_dim = len(joint_indices)
        # Find torso link index
        self.torso_link_index = -1 # Default to base link (-1 means use base orientation)
        for i in range(pb.getNumJoints(self.robot_id)):
            info = pb.getJointInfo(self.robot_id, i)
            # Link name is at index 12
            link_name = info[12].decode('utf-8').lower()
            # Check if common torso/body/chest names are in the link name
            if 'torso' in link_name or 'body' in link_name or 'chest' in link_name:
                 # The link index is the same as the joint index for the link connected by that joint
                 # In PyBullet, joint index `i` connects the parent link to the child link `i`
                self.torso_link_index = i 
                print(f"---> Identified torso/chest link: index={self.torso_link_index}, name='{info[12].decode('utf-8')}'")
                break # Stop searching once found
        
        if self.torso_link_index == -1:
             print("---> Warning: Could not find link containing 'torso', 'body' or 'chest'. Will use base orientation for angle checks.")


        self.state_dim = self.joint_dim * 2 + 7  # joint pos/vel + base pos (3) + base ori (4)
        self.action_dim = self.joint_dim
        
        # Create network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer for training (only needed if not run_only)
        self.buffers = None
        if not run_only:
             self.buffers = {
                  'states': deque(maxlen=max_buffer_size),
                  'actions': deque(maxlen=max_buffer_size),
                  'rewards': deque(maxlen=max_buffer_size),
                  'values': deque(maxlen=max_buffer_size),
                  'log_probs': deque(maxlen=max_buffer_size),
                  'dones': deque(maxlen=max_buffer_size),
             }
        
        # Training and episode tracking
        self.episodes = 0
        self.steps = 0
        self.train_interval = train_interval  # Steps between training updates
        self.model_dir = model_dir # Directory for saving (if not run_only)
        self.last_train_time = time.time()
        self.training_time = 0
        self.training_iterations = 0
        self.run_only = run_only # Store the run_only flag
        
        # Create model directory if it doesn't exist and saving is enabled
        if self.model_dir and not self.run_only:
             os.makedirs(model_dir, exist_ok=True)
        elif not self.model_dir and not self.run_only and not load_model_path:
             # Default model dir if none specified and not run_only/loading specific path
             self.model_dir = "ppo_models"
             os.makedirs(self.model_dir, exist_ok=True)

        
        # For reward calculation
        self.prev_base_pos = None
        self.latest_actions = torch.zeros(self.action_dim).to(self.device)
        
        # Initialize episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.best_reward = -float("inf")
        
        # Track reward components
        self.episode_forward_reward = 0
        self.episode_height_penalty = 0
        self.episode_energy_penalty = 0
        self.episode_velocity_reward = 0
        self.episode_orientation_penalty = 0  # Changed from orientation_reward to orientation_penalty to match main.py
        
        # For orientation tracking
        self.prev_torso_angle_rad = None
        
        # Load model logic
        if load_model_path:
            print(f"Attempting to load specific model from: {load_model_path}")
            self.load_model(specific_path=load_model_path)
        elif not skip_load:
            print(f"Attempting to load model from directory: {self.model_dir}")
            self.load_model() # Load from default directory (best or checkpoint)
        else:
            print("Starting with fresh model")
            
        # If run_only, switch network to evaluation mode
        if self.run_only:
             print("Run-only mode enabled: Setting network to evaluation mode.")
             self.network.eval()
        
    def get_state(self):
        """
        Get the current state of the robot.
        
        Returns:
            Numpy array of state features
        """
        # Get joint positions and velocities
        joint_states = []
        for joint_idx in self.joint_indices:
            joint_state = pb.getJointState(self.robot_id, joint_idx)
            joint_states.append((joint_state[0], joint_state[1]))  # pos, vel
        
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        # Get base position and orientation
        base_pos, base_orn = pb.getBasePositionAndOrientation(self.robot_id)
        
        # Combine all state components
        state = np.concatenate([
            joint_positions, 
            joint_velocities,
            base_pos,
            base_orn
        ])
        
        return state
    
    def _get_torso_angle(self):
        """Calculates the angle of the torso/chest relative to the world vertical axis."""
        try:
            if self.torso_link_index == -1: # Use base orientation if link not found
                _, orn_quat = pb.getBasePositionAndOrientation(self.robot_id)
            else: # Use link orientation
                 # Fetch the state of the specific link
                 link_state = pb.getLinkState(self.robot_id, self.torso_link_index)
                 orn_quat = link_state[1] # World orientation quaternion of the link

            # World vertical axis (positive Z)
            world_z = np.array([0, 0, 1])
            
            # Get the rotation matrix from the quaternion
            rot_matrix = np.array(pb.getMatrixFromQuaternion(orn_quat)).reshape(3, 3)
            
            # Extract the link's local Y-axis vector in world coordinates
            # Assuming the link's local Y-axis points 'up' after initial rotation
            link_up_axis_world = rot_matrix[:, 1] # Use Y-axis (second column)
            
            # Calculate the dot product between the link's up-axis and the world's Z-axis
            # Normalize vectors to ensure dot product is cosine of the angle
            # Handle potential zero vector
            norm_link_up = np.linalg.norm(link_up_axis_world)
            if norm_link_up < 1e-6:
                # If link vector is zero, assume upright to avoid errors
                return 0.0 
                
            link_up_axis_world_norm = link_up_axis_world / norm_link_up
            dot_product = np.dot(link_up_axis_world_norm, world_z)
            
            # Clip the dot product to avoid numerical errors with arccos (should be between -1 and 1)
            dot_product = np.clip(dot_product, -1.0, 1.0) 
            
            # Calculate the angle in radians using arccos
            # This angle represents the deviation from the world vertical axis
            angle_rad = np.arccos(dot_product)
            
            return angle_rad # Return angle in radians
            
        except Exception as e:
            # Print specific PyBullet errors if available
            if isinstance(e, pb.error):
                 print(f"PyBullet Error calculating torso angle: {e}")
            else:
                print(f"General Error calculating torso angle: {e}")
            # Return a neutral angle (0 = upright) in case of error
            return 0.0 

    def compute_reward(self):
        """
        Compute the reward for the current state.
        
        Returns:
            float: The reward value
        """
        # Get current base position and orientation
        current_base_pos, current_base_orn = pb.getBasePositionAndOrientation(self.robot_id)
        current_base_pos = np.array(current_base_pos)
        
        # Get current torso angle
        current_torso_angle_rad = self._get_torso_angle()
        
        # Initialize previous position and angle if needed
        if self.prev_base_pos is None:
            self.prev_base_pos = current_base_pos
            self.prev_torso_angle_rad = current_torso_angle_rad
            return 0.0
        
        # Forward movement reward (x-axis) - significantly increased reward
        forward_reward = (current_base_pos[0] - self.prev_base_pos[0]) * 200.0  # Increased from 50 to 200
        self.episode_forward_reward += forward_reward
        
        # Height penalty if too low (fallen)
        height_penalty = -50.0 if current_base_pos[2] < 1.5 else 0.0
        self.episode_height_penalty += height_penalty
        
        # Energy efficiency penalty (very small penalty)
        energy_penalty = -0.000001 * torch.sum(torch.abs(self.latest_actions)).item()
        self.episode_energy_penalty += energy_penalty
        
        # Add reward for forward velocity
        forward_velocity = (current_base_pos[0] - self.prev_base_pos[0]) / (1/240.0)  # velocity in m/s
        velocity_reward = 0.1 * abs(forward_velocity)  # Reward for any forward velocity
        self.episode_velocity_reward += velocity_reward

        # Orientation penalty based on deviation from the world vertical axis
        # Calculate the squared differences (higher value = worse orientation)
        prev_squared_angle = self.prev_torso_angle_rad ** 2
        current_squared_angle = current_torso_angle_rad ** 2
        
        # Penalty for deviation (increased squared angle)
        # If current_squared_angle is larger than prev_squared_angle, the difference is positive
        orientation_deviation = current_squared_angle - prev_squared_angle
        
        # Scale the penalty - negative when deviating, positive when improving
        orientation_penalty = - 1600 * orientation_deviation
        self.episode_orientation_penalty += orientation_penalty
        
        # Update previous position and angle
        self.prev_base_pos = current_base_pos
        self.prev_torso_angle_rad = current_torso_angle_rad
        
        # Combine rewards and penalties
        reward = forward_reward + height_penalty + energy_penalty + velocity_reward + orientation_penalty
        
        return reward
    
    def has_fallen(self):
        """
        Check if the humanoid has fallen based on torso angle or base height.
        
        Returns:
            bool: True if fallen, False otherwise
        """
        # Check torso angle
        torso_angle_rad = self._get_torso_angle()
        # Fall if angle > 30 degrees (pi/6 radians) from vertical
        fall_threshold_rad = np.pi / 6.0 
        angle_fallen = abs(torso_angle_rad) > fall_threshold_rad

        # Check base height as a fallback
        base_pos, _ = pb.getBasePositionAndOrientation(self.robot_id)
        height_fallen = base_pos[2] < 1.0 # Use a slightly lower threshold if angle is primary check

        # Episode ends if either condition is met
        return angle_fallen or height_fallen
        
    def select_action(self, state):
        """
        Select an action using the policy network.
        
        Args:
            state: Current state
            
        Returns:
            torch.Tensor: Selected action
            float: Log probability of the action
            float: Value estimate
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Get policy and value outputs
        with torch.no_grad():
            action_mean, action_std, value = self.network(state)
        
        # Create normal distribution with larger standard deviation for exploration
        exploration_scale = max(0.5, 3.0 * (1 - self.steps / 1_000_000))  # Decay from 3.0 to 0.5
        dist = Normal(action_mean, action_std * exploration_scale)
        
        # Sample action and get log probability
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        # Clip actions to valid range with joint-specific limits if available
        if self.joint_max_forces is not None:
            # Use joint-specific max forces
            joint_max_forces = self.joint_max_forces.to(self.device)
            joint_min_forces = -joint_max_forces
            action_clipped = torch.max(torch.min(action, joint_max_forces), joint_min_forces)
        else:
            # Use uniform max force for all joints
            action_clipped = torch.clamp(action, -self.max_force, self.max_force)
        
        return action_clipped, log_prob, value.item()
    
    def update(self):
        """
        Apply the policy. If not run_only, also update the experience buffer and potentially train.
        """
        # Get current state
        state = self.get_state()
        
        # Select action
        action, log_prob, value = self.select_action(state)
        self.latest_actions = action.clone()
        
        # Apply torques to joints
        for j, joint_idx in enumerate(self.joint_indices):
            try:
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.TORQUE_CONTROL,
                    force=action[j].item()
                )
            except Exception as e:
                print(f"ERROR at joint {joint_idx}: {e}")
                continue
        
        # Compute reward (still useful for tracking/logging even in run_only)
        reward = self.compute_reward()
        self.episode_reward += reward
        self.episode_length += 1
        
        # Check if episode is done
        done = self.has_fallen()
        
        # Store experience in buffer only if training is enabled
        if not self.run_only and self.buffers is not None:
            self.buffers['states'].append(state)
            self.buffers['actions'].append(action.cpu().numpy())
            self.buffers['rewards'].append(reward)
            self.buffers['values'].append(value)
            self.buffers['log_probs'].append(log_prob.item())
            self.buffers['dones'].append(done)
        
        self.steps += 1
        
        # If episode is done, reset and log
        if done:
            # WARNING: DO NOT MODIFY THE FORMAT OF THE FOLLOWING PRINT STATEMENT
            # The hyperparameter tuning script (tune_hyperparams.py) parses this specific output format 
            # to extract episode statistics. It expects:
            # - "Episode" keyword followed by episode number at position 1
            # - "finished with reward" followed by the reward value at position 4
            # - "after" followed by episode length (steps) at position 6
            # Changing this format will break the automated hyperparameter tuning process.
            print(f"Episode {self.episodes} finished with reward {self.episode_reward:.2f} after {self.episode_length} steps")
            # Print reward components
            print(f"  Reward Breakdown: Fwd={self.episode_forward_reward:.2f}, Hgt={self.episode_height_penalty:.2f}, Eng={self.episode_energy_penalty:.2f}, Vel={self.episode_velocity_reward:.2f}, Ori={self.episode_orientation_penalty:.2f} \n")

            # Save best model only if training
            if not self.run_only:
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward
                    self.save_model("best") # save_model internally checks if model_dir exists
            
            # Reset episode tracking
            self.episodes += 1
            self.episode_reward = 0
            self.episode_length = 0
            self.prev_base_pos = None
            
            # Reset reward components
            self.episode_forward_reward = 0
            self.episode_height_penalty = 0
            self.episode_energy_penalty = 0
            self.episode_velocity_reward = 0
            self.episode_orientation_penalty = 0  # Reset orientation penalty tracker
            self.prev_torso_angle_rad = None  # Reset previous angle
            
            # Reset humanoid position for the next episode
            reset_humanoid(self.robot_id)
        
        # Periodically train only if not run_only and buffer has enough samples
        if not self.run_only and self.buffers is not None and self.steps % self.train_interval == 0 and len(self.buffers['states']) >= self.batch_size:
            self.train()
            
            # Log training frequency
            current_time = time.time()
            elapsed = current_time - self.last_train_time
            self.last_train_time = current_time
            print(f"Training: {elapsed:.2f} seconds since last train, avg training time: {self.training_time/max(1, self.training_iterations):.3f}s")
            
            # Save checkpoint model (only if training)
            self.save_model("checkpoint") # save_model internally checks if model_dir exists
    
    def train(self, num_epochs=10):
        """
        Train the policy using PPO. No changes needed here, will only be called if not run_only.
        """
        if self.run_only or self.buffers is None:
             print("Warning: train() called but run_only is True or buffer is None. Skipping training.")
             return # Should not happen based on update() logic, but safe guard

        if len(self.buffers['states']) < self.batch_size:
            return
            
        start_time = time.time()
        
        # Convert buffers to numpy arrays
        states = np.array(self.buffers['states'])
        actions = np.array(self.buffers['actions'])
        rewards = np.array(self.buffers['rewards'])
        values = np.array(self.buffers['values'])
        log_probs = np.array(self.buffers['log_probs'])
        dones = np.array(self.buffers['dones'])
        
        # Calculate advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute returns and advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # For last step
            else:
                next_value = values[t + 1]
                
            # If the episode is done, there is no next state value
            if dones[t]:
                next_value = 0
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.lambd * last_gae * (1 - dones[t])
            advantages[t] = last_gae
            
        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Get old policy distribution parameters for all states (for KL calculation)
        with torch.no_grad():
            old_action_mean, old_action_std, _ = self.network(states_tensor)
            old_policy = Normal(old_action_mean, old_action_std)
        
        # Track losses for logging
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        avg_total_loss = 0
        avg_kl_divergence = 0  # Track KL divergence between old and new policies
        update_count = 0
        
        # Mini-batch training
        for _ in range(num_epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Train in mini-batches
            for start_idx in range(0, len(states), self.batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Mini-batch tensors
                mb_states = states_tensor[idx]
                mb_actions = actions_tensor[idx]
                mb_old_log_probs = old_log_probs_tensor[idx]
                mb_advantages = advantages_tensor[idx]
                mb_returns = returns_tensor[idx]
                
                # Forward pass
                action_mean, action_std, values = self.network(mb_states)
                
                # Create distribution
                dist = Normal(action_mean, action_std)
                
                # Calculate new log probabilities
                new_log_probs = dist.log_prob(mb_actions).sum(1)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                # Calculate PPO loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = 0.5 * ((values.squeeze() - mb_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                
                self.optimizer.step()
                
                # Accumulate losses for logging
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy.item()
                avg_total_loss += loss.item()
                
                # Calculate KL divergence between old and new policy for this batch
                with torch.no_grad():
                    old_mb_mean = old_action_mean[idx]
                    old_mb_std = old_action_std[idx]
                    old_mb_policy = Normal(old_mb_mean, old_mb_std)
                    new_mb_policy = Normal(action_mean, action_std)
                    
                    # KL divergence for multivariate Gaussian (analytically calculated)
                    # KL(p||q) = 0.5 * (log(det(Σq)/det(Σp)) - d + tr(Σq^-1 Σp) + (μq-μp)^T Σq^-1 (μq-μp))
                    # For diagonal covariance matrices, this simplifies to:
                    kl_per_dim = 0.5 * (
                        2 * torch.log(action_std) - 2 * torch.log(old_mb_std) 
                        + (old_mb_std.pow(2) + (old_mb_mean - action_mean).pow(2)) / action_std.pow(2) 
                        - 1
                    )
                    # Sum across action dimensions, then average across batch
                    kl_divergence = kl_per_dim.sum(dim=1).mean().item()
                    avg_kl_divergence += kl_divergence
                
                update_count += 1
        
        # Calculate average losses
        if update_count > 0:
            avg_policy_loss /= update_count
            avg_value_loss /= update_count
            avg_entropy /= update_count
            avg_total_loss /= update_count
            avg_kl_divergence /= update_count
        
        # Only clear buffers if they're full
        if len(self.buffers['states']) >= self.buffers['states'].maxlen:
            for key in self.buffers:
                self.buffers[key].clear()
            
        # Track training time
        training_time = time.time() - start_time
        self.training_time += training_time
        self.training_iterations += 1
        
        # Log losses to be captured by hyperparameter tuning script
        print(f"LOSS_DATA: policy_loss={avg_policy_loss:.6f}, value_loss={avg_value_loss:.6f}, entropy={avg_entropy:.6f}, total_loss={avg_total_loss:.6f}, kl_divergence={avg_kl_divergence:.6f}")
    
    def save_model(self, tag="checkpoint"):
        """
        Save the model. Only saves if not in run_only mode and model_dir is set.
        """
        if self.run_only:
            # print("Save skipped: run_only mode is active.")
            return # Silently skip saving in run_only mode
        if not self.model_dir:
             print("Save skipped: model_dir not specified.")
             return

        save_path = os.path.join(self.model_dir, f"ppo_model_{tag}.pt")
        try:
            # Ensure the directory exists just before saving
            os.makedirs(self.model_dir, exist_ok=True)
            torch.save({
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'episodes': self.episodes,
                'steps': self.steps,
                'best_reward': self.best_reward
            }, save_path)
            # print(f"Model saved to {save_path}") # Reduce print frequency
        except Exception as e:
             print(f"Error saving model to {save_path}: {e}")
        
    def load_model(self, specific_path=None):
        """
        Load the model. Prioritizes specific_path if provided.
        Otherwise, tries loading from best or checkpoint in model_dir.
        """
        path_to_load = None
        if specific_path:
            if os.path.exists(specific_path):
                 path_to_load = specific_path
            else:
                 print(f"Error: Specified model path does not exist: {specific_path}")
                 # Decide whether to raise an error or just warn and continue with fresh model
                 # For now, warn and continue fresh.
                 print("Warning: Continuing with a fresh model.")
                 self.episodes = 0
                 self.steps = 0
                 self.best_reward = -float('inf')
                 return
        elif self.model_dir:
            checkpoint_path = os.path.join(self.model_dir, "ppo_model_checkpoint.pt")
            best_path = os.path.join(self.model_dir, "ppo_model_best.pt")
            # Prioritize best model if it exists
            path_to_load = best_path if os.path.exists(best_path) else checkpoint_path
        
        if path_to_load and os.path.exists(path_to_load):
            try:
                print(f"Loading model from {path_to_load}...")
                checkpoint = torch.load(path_to_load, map_location=self.device)
                self.network.load_state_dict(checkpoint['network'])
                
                # Load optimizer state only if we are going to train further
                if not self.run_only and 'optimizer' in checkpoint:
                     try:
                          self.optimizer.load_state_dict(checkpoint['optimizer'])
                     except ValueError as e:
                          print(f"Warning: Could not load optimizer state, possibly due to changed model structure or learning rate. Optimizer state reset. Error: {e}")
                     except Exception as e:
                          print(f"Warning: An unexpected error occurred loading optimizer state: {e}. Optimizer state reset.")

                # Load training progress only if useful (i.e., if continuing training)
                if not self.run_only:
                     self.episodes = checkpoint.get('episodes', 0)
                     self.steps = checkpoint.get('steps', 0)
                     self.best_reward = checkpoint.get('best_reward', -float('inf'))
                     print(f"Loaded model from {path_to_load}")
                     print(f"Continuing from episode {self.episodes}, steps {self.steps}, best reward {self.best_reward:.2f}")
                else:
                     # Don't load training progress if run_only
                     print(f"Loaded network weights from {path_to_load} for run-only mode.")
                     self.episodes = 0
                     self.steps = 0
                     self.best_reward = -float('inf') # Reset reward tracking for run_only

            except KeyError as e:
                 print(f"Error loading model: Missing key {e} in checkpoint file {path_to_load}. Check if the model file is complete or compatible.")
                 print("Warning: Continuing with potentially partially loaded or fresh model.")
            except Exception as e:
                print(f"Error loading model from {path_to_load}: {e}")
                print("Warning: Continuing with a fresh model.")
                # Reset state if loading failed critically
                self.episodes = 0
                self.steps = 0
                self.best_reward = -float('inf')
        else:
             if not specific_path: # Only print if we weren't given a specific (non-existent) path
                  print(f"No existing model found in {self.model_dir} to load.")
             # No warning needed if specific_path was given but didn't exist (handled above)


def create_controller(controller_type, robot_id, joint_indices, joint_names, **kwargs):
    """
    Factory function to create a controller of the specified type.
    
    Args:
        controller_type: Type of controller to create ('position', 'velocity', 'torque', 'pd', 'ppo')
        robot_id: PyBullet ID of the robot
        joint_indices: List of joint indices to control
        joint_names: List of joint names corresponding to joint_indices
        **kwargs: Additional arguments to pass to the controller constructor
        
    Returns:
        Controller instance of the specified type
    """
    controllers = {
        'position': PositionController,
        'velocity': VelocityController,
        'torque': TorqueController,
        'pd': PDController,
        'ppo': PPOController
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}. Valid types: {list(controllers.keys())}")
    
    return controllers[controller_type](robot_id, joint_indices, joint_names, **kwargs) 