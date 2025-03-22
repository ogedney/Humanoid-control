import pybullet as pb

import pybullet_data
import time
import numpy as np

class HumanoidController:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.joint_indices = []
        self.joint_names = []
        self.joint_limits = []
        self.max_forces = []
        
        # Get controllable joints
        for i in range(pb.getNumJoints(robot_id)):
            info = pb.getJointInfo(robot_id, i)
            if info[2] != pb.JOINT_FIXED:
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
                self.joint_limits.append((info[8], info[9]))  # Lower and upper limits
                
                # Set appropriate force limits based on joint type
                max_force = info[10]
                if max_force == 0.0:  # If not specified in URDF
                    if "leg" in info[1].decode('utf-8').lower():
                        max_force = 100.0  # Higher force for leg joints
                    elif "arm" in info[1].decode('utf-8').lower():
                        max_force = 50.0   # Medium force for arm joints
                    elif "back" in info[1].decode('utf-8').lower():
                        max_force = 75.0   # Medium-high force for back joints
                    else:
                        max_force = 30.0   # Default force
                self.max_forces.append(max_force)
                
                # Add damping for stability
                pb.changeDynamics(
                    robot_id, 
                    i, 
                    jointDamping=2.0,  # Reduced damping
                    linearDamping=0.5,
                    angularDamping=0.5
                )
                
                # Reset joint state to zeros with zero velocity
                pb.resetJointState(robot_id, i, 0, 0)
        
        self.num_controlled_joints = len(self.joint_indices)
        print(f"Humanoid loaded with {self.num_controlled_joints} controllable joints")
        
        # Initialize state tracking arrays
        self.joint_positions = np.zeros(self.num_controlled_joints)
        self.joint_velocities = np.zeros(self.num_controlled_joints)
        
        # Let the robot settle with light position control
        self._initial_settling()
    
    def _initial_settling(self):
        """Allow the robot to settle in a stable position before full simulation"""
        target_positions = np.zeros(self.num_controlled_joints)
        for _ in range(100):  # Increased settling time
            # Apply light position control
            for i, joint_idx in enumerate(self.joint_indices):
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=target_positions[i],
                    force=self.max_forces[i] * 0.5  # Reduced force for settling
                )
            pb.stepSimulation()
    
    def update_state(self):
        """Update the current state of all joints"""
        try:
            joint_states = pb.getJointStates(self.robot_id, self.joint_indices)
            self.joint_positions = np.array([state[0] for state in joint_states])
            self.joint_velocities = np.array([state[1] for state in joint_states])
            
            # Get base state
            pos, orn = pb.getBasePositionAndOrientation(self.robot_id)
            lin_vel, ang_vel = pb.getBaseVelocity(self.robot_id)
            self.base_position = np.array(pos)
            self.base_orientation = np.array(orn)
            self.base_linear_velocity = np.array(lin_vel)
            self.base_angular_velocity = np.array(ang_vel)
            return True
        except Exception as e:
            print(f"Error updating state: {e}")
            return False
    
    def set_positions(self, positions, velocities=None, kp=0.3, kd=0.5):
        """Set target joint positions with optional target velocities"""
        if velocities is None:
            velocities = np.zeros(self.num_controlled_joints)
        
        try:
            for i, joint_idx in enumerate(self.joint_indices):
                # Get joint info
                joint_info = pb.getJointInfo(self.robot_id, joint_idx)
                lower_limit, upper_limit = joint_info[8], joint_info[9]
                
                # Get current position
                current_pos = self.joint_positions[i]
                
                # Calculate a safe target within joint limits
                target_pos = positions[i]
                if lower_limit < upper_limit:  # If joint has limits
                    target_pos = max(lower_limit, min(upper_limit, target_pos))
                
                # Use smaller increments for stability (0.001 instead of 0.005)
                if abs(current_pos - target_pos) > 0.01:
                    direction = 1 if target_pos > current_pos else -1
                    incremental_target = current_pos + direction * 0.001
                else:
                    incremental_target = target_pos
                
                # Apply position control
                pb.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=incremental_target,
                    targetVelocity=velocities[i],
                    positionGain=kp,
                    velocityGain=kd,
                    force=self.max_forces[i] * 0.8  # Slightly reduced force
                )
            return True
        except Exception as e:
            print(f"Error setting joint positions: {e}")
            return False
    
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

def setup_simulation_environment():
    """Sets up the PyBullet simulation environment with a plane and humanoid"""
    # Connect to PyBullet with GUI
    physicsClient = pb.connect(pb.GUI)
    
    # Configure visualization
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)  # Disable shadows for performance
    
    # Configure physics for stability
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)
    pb.setPhysicsEngineParameter(
        fixedTimeStep=1/240.0, 
        numSolverIterations=50,  # Increased solver iterations for stability
        numSubSteps=4  # Add substeps for more accurate simulation
    )
    
    # Load ground plane
    planeId = pb.loadURDF("plane.urdf")
    
    # Load humanoid robot
    startPos = [0, 0, 0.9]  # Lower starting position for stability
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
    robotId = pb.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation,
                         flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_USE_SELF_COLLISION)
    
    # Set camera view
    pb.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    return physicsClient, planeId, robotId

def main():
    try:
        # Setup simulation
        physicsClient, planeId, robotId = setup_simulation_environment()
        print("Simulation environment set up")
        
        # Initialize controller
        controller = HumanoidController(robotId)
        
        # State history
        states_history = []
        
        # Get initial standing pose
        target_positions = np.zeros(controller.num_controlled_joints)
        
        # Main simulation loop
        print("Starting simulation with position control...")
        step = 0
        max_steps = 10000
        while step < max_steps:
            # Update state - break if update fails
            if not controller.update_state():
                print("State update failed, terminating simulation")
                break
            
            # Set target positions - break if setting positions fails
            if not controller.set_positions(target_positions):
                print("Setting positions failed, terminating simulation")
                break
            
            # Record state
            current_state = controller.get_all_states()
            states_history.append(current_state)
            
            # Step simulation
            pb.stepSimulation()
            
            # Print info occasionally
            if step % 500 == 0:
                print(f"Step {step}: Base position = {controller.base_position}")
                # Check if robot has fallen
                if controller.base_position[2] < 0.3:
                    print("Robot has fallen, terminating simulation")
                    break
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Check for quit key
            keys = pb.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & pb.KEY_WAS_TRIGGERED:
                print("Q key pressed, exiting")
                break
                
            step += 1
        
        print("Simulation completed successfully")
        if states_history:
            print(f"Final position: {states_history[-1]['base_position']}")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        try:
            pb.disconnect()
            print("Disconnected from PyBullet")
        except:
            pass

if __name__ == "__main__":
    main() 