import numpy as np
import pybullet as pb

class HumanoidController:
    def __init__(self, humanoid_id):
        self.humanoid_id = humanoid_id
        self.num_joints = pb.getNumJoints(humanoid_id)
        
        # Initialize state tracking arrays
        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.joint_torques = np.zeros(self.num_joints)
        
        # Store joint information
        self.joint_names = []
        self.joint_limits = []
        self.max_forces = []
        self.joint_indices = []  # Store controllable joint indices
        
        # Get information about each joint
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.humanoid_id, i)
            joint_type = info[2]  # Get joint type
            
            # Only consider joints we can control (exclude fixed joints)
            if joint_type != pb.JOINT_FIXED:
                joint_name = info[1].decode('utf-8')
                self.joint_names.append(joint_name)
                self.joint_limits.append((info[8], info[9]))  # Lower and upper limits
                
                # Atlas has higher torque limits than the default humanoid
                # Use the info from URDF if available, otherwise set reasonable defaults
                max_force = info[10]
                if max_force == 0.0:  # If not specified in URDF
                    if "leg" in joint_name.lower():
                        max_force = 300.0  # Higher torque for leg joints
                    elif "arm" in joint_name.lower():
                        max_force = 150.0  # Medium torque for arm joints
                    elif "back" in joint_name.lower():
                        max_force = 200.0  # Higher torque for back joints
                    else:
                        max_force = 100.0  # Default torque
                
                self.max_forces.append(max_force)
                self.joint_indices.append(i)
        
        # Update number of controllable joints
        self.num_controlled_joints = len(self.joint_indices)
        
        # Print joint information for debugging
        print(f"Atlas model loaded with {self.num_controlled_joints} controllable joints:")
        for i, name in enumerate(self.joint_names):
            print(f"  {i}: {name} (limits: {self.joint_limits[i]}, max force: {self.max_forces[i]})")
        
        # Disable default motor control
        for joint_idx in self.joint_indices:
            pb.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=joint_idx,
                controlMode=pb.VELOCITY_CONTROL,
                force=0
            )
    
    def update_state(self):
        """Update the current state of all joints"""
        joint_states = pb.getJointStates(self.humanoid_id, self.joint_indices)
        self.joint_positions = np.array([state[0] for state in joint_states])
        self.joint_velocities = np.array([state[1] for state in joint_states])
        self.joint_torques = np.array([state[3] for state in joint_states])
        
        pos, orn = pb.getBasePositionAndOrientation(self.humanoid_id)
        self.base_position = np.array(pos)
        self.base_orientation = np.array(orn)
        
        # Get base velocity
        linear_vel, angular_vel = pb.getBaseVelocity(self.humanoid_id)
        self.base_linear_velocity = np.array(linear_vel)
        self.base_angular_velocity = np.array(angular_vel)
    
    def get_mass_matrix(self):
        """Get the mass matrix (M) of the humanoid"""
        return np.array(pb.calculateMassMatrix(
            bodyUniqueId=self.humanoid_id,
            objPositions=self.joint_positions.tolist()
        ))
    
    def get_gravity_compensation(self):
        """
        Calculate gravity compensation torques for a humanoid with incomplete URDF.
        This method uses available PyBullet functions and physics principles to estimate
        gravity compensation without requiring complete axis definitions.
        """
        # Get the mass matrix if available (may work even with incomplete URDF)
        try:
            mass_matrix = np.array(pb.calculateMassMatrix(
                bodyUniqueId=self.humanoid_id,
                objPositions=self.joint_positions.tolist()
            ))
        except:
            # If mass matrix calculation fails, use identity matrix as fallback
            mass_matrix = np.eye(self.num_controlled_joints)
        
        # Initialize gravity compensation torques
        gravity_torques = np.zeros(self.num_controlled_joints)
        
        # Use default gravity value (PyBullet typically uses -9.81 in z direction)
        gravity_vec = np.array([0, 0, -10])
        gravity_magnitude = np.linalg.norm(gravity_vec)
        
        # Get link information for each joint
        for i, joint_idx in enumerate(self.joint_indices):
            # Get joint and link information
            joint_info = pb.getJointInfo(self.humanoid_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            
            try:
                # Get link state and mass properties
                link_state = pb.getLinkState(self.humanoid_id, joint_idx, computeLinkVelocity=0)
                dynamics_info = pb.getDynamicsInfo(self.humanoid_id, joint_idx)
                link_mass = dynamics_info[0]
                
                if link_mass <= 0:
                    # If mass is not defined, estimate based on joint name
                    if "leg" in joint_name.lower():
                        link_mass = 2.0
                    elif "arm" in joint_name.lower():
                        link_mass = 1.0
                    elif "head" in joint_name.lower():
                        link_mass = 1.5
                    else:
                        link_mass = 0.5
                        
                # Get center of mass position in world frame
                com_pos_world = link_state[0]
                
                # Get joint axis if available, otherwise use a default
                joint_axis = np.array(joint_info[13])
                if np.all(joint_axis == 0):
                    # If axis is not defined, use a default based on joint name
                    if "x" in joint_name.lower():
                        joint_axis = np.array([1, 0, 0])
                    elif "y" in joint_name.lower():
                        joint_axis = np.array([0, 1, 0])
                    else:
                        joint_axis = np.array([0, 0, 1])
                
                # Normalize the axis
                if np.linalg.norm(joint_axis) > 0:
                    joint_axis = joint_axis / np.linalg.norm(joint_axis)
                
                # Calculate the moment arm from joint to COM
                parent_index = joint_info[16]
                if parent_index >= 0:
                    try:
                        parent_link_state = pb.getLinkState(self.humanoid_id, parent_index)
                        parent_com = parent_link_state[0]
                        moment_arm = np.array(com_pos_world) - np.array(parent_com)
                    except:
                        # If parent link info fails, use a default moment arm
                        moment_arm = np.array([0, 0, 0.2])
                else:
                    # For base joints, use the distance from base
                    base_pos, _ = pb.getBasePositionAndOrientation(self.humanoid_id)
                    moment_arm = np.array(com_pos_world) - np.array(base_pos)
                
                # Cross product of joint axis and gravitational force gives torque direction
                # (force = mass * gravity, along negative z-axis typically)
                force = link_mass * gravity_magnitude * np.array([0, 0, -1])
                torque_vec = np.cross(moment_arm, force)
                
                # Project the torque onto the joint axis
                gravity_torques[i] = np.dot(torque_vec, joint_axis)
                
            except Exception as e:
                # If calculating for this joint fails, use a simple approximation
                if "leg" in joint_name.lower():
                    gravity_torques[i] = 2.0 * gravity_magnitude * 0.2
                elif "arm" in joint_name.lower():
                    gravity_torques[i] = 1.0 * gravity_magnitude * 0.15
                else:
                    gravity_torques[i] = 0.5 * gravity_magnitude * 0.1
        
        return gravity_torques
    
    def apply_torques(self, torques):
        """Apply torques to the joints"""
        if len(torques) != self.num_controlled_joints:
            raise ValueError(f"Expected {self.num_controlled_joints} torques, got {len(torques)}")
        
        # Apply torque control to each joint
        for i, joint_idx in enumerate(self.joint_indices):
            # Clip torque to max force
            torque = np.clip(torques[i], -self.max_forces[i], self.max_forces[i])
            pb.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=joint_idx,
                controlMode=pb.TORQUE_CONTROL,
                force=torque
            )
    
    def compute_pd_torques(self, target_positions, target_velocities=None, kp=500, kd=50):
        """Compute PD control torques with gravity compensation - adjusted for Atlas"""
        if target_velocities is None:
            target_velocities = np.zeros(self.num_controlled_joints)
            
        # Compute position and velocity errors
        pos_error = target_positions - self.joint_positions
        vel_error = target_velocities - self.joint_velocities
        
        # PD control with gravity compensation
        gravity = self.get_gravity_compensation()
        tau = kp * pos_error + kd * vel_error + gravity
        
        return tau
    
    def get_joint_state(self, joint_name):
        """Get the state of a specific joint by name"""
        if joint_name in self.joint_names:
            idx = self.joint_names.index(joint_name)
            return {
                'position': self.joint_positions[idx],
                'velocity': self.joint_velocities[idx],
                'torque': self.joint_torques[idx],
                'limits': self.joint_limits[idx],
                'max_force': self.max_forces[idx]
            }
        else:
            raise ValueError(f"Joint {joint_name} not found")
    
    def get_all_states(self):
        """Return a dictionary containing all current states"""
        return {
            'base_position': self.base_position,
            'base_orientation': self.base_orientation,
            'base_linear_velocity': self.base_linear_velocity,
            'base_angular_velocity': self.base_angular_velocity,
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities,
            'joint_torques': self.joint_torques,
            'joint_names': self.joint_names
        } 