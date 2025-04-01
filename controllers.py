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
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Policy network (completely separate)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
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
                 hidden_dim=64, learning_rate=3e-4, batch_size=64, 
                 clip_param=0.2, gamma=0.99, lambd=0.95, 
                 value_coef=0.5, entropy_coef=0.01, max_buffer_size=4000,
                 model_dir="ppo_models", skip_load=False, joint_max_forces=None):
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
            skip_load: If True, start with a fresh model instead of loading existing one
            joint_max_forces: List of maximum forces for each joint. If None, max_force is used for all joints.
        """
        super().__init__(robot_id, joint_indices, joint_names)
        # Increase max force significantly for more movement
        self.max_force = max_force * 5.0
        
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
        
        # Experience buffer for training
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
        self.train_interval = 1000  # Steps between training updates
        self.model_dir = model_dir
        self.last_train_time = time.time()
        self.training_time = 0
        self.training_iterations = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
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
        self.episode_orientation_penalty = 0 # Add tracker for orientation penalty
        
        # Load model if it exists and skip_load is False
        if not skip_load:
            self.load_model()
        else:
            print("Starting with fresh model")
        
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
        
        # Initialize previous position if needed
        if self.prev_base_pos is None:
            self.prev_base_pos = current_base_pos
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

        # Orientation penalty based on torso angle
        torso_angle_rad = self._get_torso_angle()
        # Penalize deviation from upright (0 radians). Scale penalty quadratically.
        # Using a coefficient of -20 to make the penalty significant.
        orientation_penalty = -5.0 * (torso_angle_rad ** 2) 
        self.episode_orientation_penalty += orientation_penalty
        
        # Update previous position
        self.prev_base_pos = current_base_pos
        
        # Combine rewards
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
        Apply the policy and update the experience buffer.
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
        
        # Compute reward
        reward = self.compute_reward()
        self.episode_reward += reward
        self.episode_length += 1
        
        # --- Debug: Print torso angle ---
        # current_torso_angle_rad = self._get_torso_angle()
        # current_torso_angle_deg = np.degrees(current_torso_angle_rad) # Use np.degrees for clarity
        # print(f"Step {self.steps}: Torso Angle (Deg): {current_torso_angle_deg:.2f}")
        # --- End Debug ---
        
        # Check if episode is done
        done = self.has_fallen()
        
        # Store experience in buffer
        self.buffers['states'].append(state)
        self.buffers['actions'].append(action.cpu().numpy())
        self.buffers['rewards'].append(reward)
        self.buffers['values'].append(value)
        self.buffers['log_probs'].append(log_prob.item())
        self.buffers['dones'].append(done)
        
        self.steps += 1
        
        # If episode is done, reset and log
        if done:
            print(f"Episode {self.episodes} finished with reward {self.episode_reward:.2f} after {self.episode_length} steps")
            # Print reward components
            print(f"  Reward Breakdown: Fwd={self.episode_forward_reward:.2f}, Hgt={self.episode_height_penalty:.2f}, Eng={self.episode_energy_penalty:.2f}, Vel={self.episode_velocity_reward:.2f}, Ori={self.episode_orientation_penalty:.2f} \n")

            # Save best model
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                self.save_model("best")
            
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
            self.episode_orientation_penalty = 0 # Reset orientation penalty tracker
            
            # Reset humanoid position for the next episode
            reset_humanoid(self.robot_id)
        
        # Periodically train if buffer has enough samples
        if self.steps % self.train_interval == 0 and len(self.buffers['states']) >= self.batch_size:
            self.train()
            
            # Log training frequency
            current_time = time.time()
            elapsed = current_time - self.last_train_time
            self.last_train_time = current_time
            print(f"Training: {elapsed:.2f} seconds since last train, avg training time: {self.training_time/max(1, self.training_iterations):.3f}s")
            
            # Save checkpoint model
            self.save_model("checkpoint")
    
    def train(self, num_epochs=10):
        """
        Train the policy using PPO.
        
        Args:
            num_epochs: Number of epochs to train for
        """
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
        
        # Clear buffers after training
        for key in self.buffers:
            self.buffers[key].clear()
            
        # Track training time
        training_time = time.time() - start_time
        self.training_time += training_time
        self.training_iterations += 1
    
    def save_model(self, tag="checkpoint"):
        """
        Save the model.
        
        Args:
            tag: Tag to add to the filename
        """
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self.episodes,
            'steps': self.steps,
            'best_reward': self.best_reward
        }, os.path.join(self.model_dir, f"ppo_model_{tag}.pt"))
        
    def load_model(self):
        """
        Load the model if it exists.
        """
        checkpoint_path = os.path.join(self.model_dir, "ppo_model_checkpoint.pt")
        best_path = os.path.join(self.model_dir, "ppo_model_best.pt")
        
        path_to_load = best_path if os.path.exists(best_path) else checkpoint_path
        
        if os.path.exists(path_to_load):
            try:
                checkpoint = torch.load(path_to_load, map_location=self.device)
                self.network.load_state_dict(checkpoint['network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.episodes = checkpoint['episodes']
                self.steps = checkpoint['steps']
                self.best_reward = checkpoint['best_reward']
                print(f"Loaded model from {path_to_load}")
                print(f"Continuing from episode {self.episodes}, steps {self.steps}, best reward {self.best_reward:.2f}")
            except Exception as e:
                print(f"Error loading model: {e}")

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