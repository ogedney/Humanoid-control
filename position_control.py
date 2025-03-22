import pybullet as pb
import pybullet_data
import time
import numpy as np

def main():
    try:
        # Connect to PyBullet with GUI
        physicsClient = pb.connect(pb.GUI)
        print("Connected to PyBullet")
        
        # Configure visualization - minimal settings
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        
        # Add data path and set gravity
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)
        pb.setTimeStep(1/240.0)
        
        # Load ground plane
        planeId = pb.loadURDF("plane.urdf")
        print("Loaded ground plane")
        
        # Load humanoid robot
        startPos = [0, 0, 1.1]
        startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
        robotId = pb.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation)
        print(f"Loaded humanoid robot, ID: {robotId}")
        
        # Set camera view
        pb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1.0]
        )
        
        # Get joint information
        joint_indices = []
        joint_names = []
        
        for i in range(pb.getNumJoints(robotId)):
            info = pb.getJointInfo(robotId, i)
            joint_type = info[2]
            
            if joint_type != pb.JOINT_FIXED:
                joint_indices.append(i)
                joint_names.append(info[1].decode('utf-8'))
                print(f"Joint {i}: {info[1].decode('utf-8')}")
                
                # Set joint damping for stability
                pb.changeDynamics(
                    robotId, 
                    i, 
                    jointDamping=1.0,
                    angularDamping=1.0
                )
        
        # Wait for visualization to initialize
        time.sleep(0.5)
        
        # Define a simple target pose (all zeros for standing straight)
        num_joints = len(joint_indices)
        target_positions = np.zeros(num_joints)
        
        # Simple position control parameters
        kp = 400.0  # Position gain
        kd = 40.0   # Velocity gain
        max_force = 200.0  # Maximum force
        
        # Simulation loop
        print("Starting simulation with position control...")
        for i in range(10000):
            # Apply position control to each controllable joint
            for j, joint_idx in enumerate(joint_indices):
                pb.setJointMotorControl2(
                    bodyUniqueId=robotId,
                    jointIndex=joint_idx,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=target_positions[j],
                    positionGain=kp,
                    velocityGain=kd,
                    force=max_force
                )
            
            # Step simulation
            pb.stepSimulation()
            
            # Print base position occasionally
            if i % 500 == 0:
                pos, _ = pb.getBasePositionAndOrientation(robotId)
                print(f"Step {i}: Base position = {pos}")
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Check for quit key
            keys = pb.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & pb.KEY_WAS_TRIGGERED:
                print("Q key pressed, exiting")
                break
        
        print("Simulation completed successfully")
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