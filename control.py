import numpy as np
import pybullet as pb

class HumanoidController:
    def __init__(self, humanoid_id):
        self.humanoid_id = humanoid_id
        self.joint_indices = []
        self.joint_names = []
        self.joint_limits = []
        self.max_forces = []
        
        # Get controllable joints
        for i in range(pb.getNumJoints(humanoid_id)):
            info = pb.getJointInfo(humanoid_id, i)
            if info[2] != pb.JOINT_FIXED:
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
                self.joint_limits.append((info[8], info[9]))  # Lower and upper limits
                
                # Set appropriate force limits based on joint type
                max_force = info[10]
                if max_force == 0.0:  # If not specified in URDF
                    if "leg" in info[1].decode('utf-8').lower():
                        max_force = 200.0  # Higher force for leg joints
                    elif "arm" in info[1].decode('utf-8').lower():
                        max_force = 100.0  # Medium force for arm joints
                    elif "back" in info[1].decode('utf-8').lower():
                        max_force = 150.0  # Higher force for back joints
                    else:
                        max_force = 50.0  # Default force
                self.max_forces.append(max_force)
        
        self.num_controlled_joints = len(self.joint_indices)
        print(f"Humanoid model loaded with {self.num_controlled_joints} controllable joints")
    
    def update_state(self):
        """Update the current state of all joints"""
        joint_states = pb.getJointStates(self.humanoid_id, self.joint_indices)
        self.joint_positions = np.array([state[0] for state in joint_states])
        self.joint_velocities = np.array([state[1] for state in joint_states])
        
        # Get base state
        pos, orn = pb.getBasePositionAndOrientation(self.humanoid_id)
        lin_vel, ang_vel = pb.getBaseVelocity(self.humanoid_id)
        self.base_position = np.array(pos)
        self.base_orientation = np.array(orn)
        self.base_linear_velocity = np.array(lin_vel)
        self.base_angular_velocity = np.array(ang_vel)
    
    def set_positions(self, positions, velocities=None, kp=400, kd=40):
        """Set target joint positions with optional target velocities"""
        if velocities is None:
            velocities = np.zeros(self.num_controlled_joints)
            
        for i, joint_idx in enumerate(self.joint_indices):
            pb.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=joint_idx,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=positions[i],
                targetVelocity=velocities[i],
                positionGain=kp,
                velocityGain=kd,
                force=self.max_forces[i],
                maxVelocity=2.0
            )
    
    def get_joint_state(self, joint_name):
        """Get state of a specific joint"""
        if joint_name in self.joint_names:
            idx = self.joint_names.index(joint_name)
            return {
                'position': self.joint_positions[idx],
                'velocity': self.joint_velocities[idx],
                'limits': self.joint_limits[idx]
            }
        raise ValueError(f"Joint {joint_name} not found")
    
    def get_all_states(self):
        """Get all robot states"""
        return {
            'base_position': self.base_position,
            'base_orientation': self.base_orientation,
            'base_linear_velocity': self.base_linear_velocity,
            'base_angular_velocity': self.base_angular_velocity,
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities,
            'joint_names': self.joint_names
        } 