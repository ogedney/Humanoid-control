import pybullet as pb
import numpy as np
from abc import ABC, abstractmethod

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

def create_controller(controller_type, robot_id, joint_indices, joint_names, **kwargs):
    """
    Factory function to create a controller of the specified type.
    
    Args:
        controller_type: Type of controller to create ('position', 'velocity', 'torque', 'pd')
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
        'pd': PDController
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}. Valid types: {list(controllers.keys())}")
    
    return controllers[controller_type](robot_id, joint_indices, joint_names, **kwargs) 