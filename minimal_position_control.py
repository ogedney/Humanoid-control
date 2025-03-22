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
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
        
        # Add data path and set gravity
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)
        
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
        
        # Wait for visualization to initialize
        time.sleep(1.0)
        print("Starting simulation...")
        
        # Set very minimal position control parameters
        kp = 100.0  # Lower position gain
        kd = 10.0   # Lower velocity gain
        max_force = 50.0  # Lower max force
        
        # Simulation loop with single-joint control
        for i in range(10000):
            # Debug: check if robot still exists
            try:
                pos, _ = pb.getBasePositionAndOrientation(robotId)
                if i % 100 == 0:
                    print(f"Step {i}: Robot at position {pos}")
            except Exception as e:
                print(f"ERROR: Could not get robot position: {e}")
                break
            
            # Apply position control only to the first joint with minimal force
            if len(joint_indices) > 0:
                joint_idx = joint_indices[0]
                try:
                    pb.setJointMotorControl2(
                        bodyUniqueId=robotId,
                        jointIndex=joint_idx,
                        controlMode=pb.POSITION_CONTROL,
                        targetPosition=0.0,
                        positionGain=kp,
                        velocityGain=kd,
                        force=max_force
                    )
                    if i % 100 == 0:
                        print(f"Applied control to joint {joint_idx}")
                except Exception as e:
                    print(f"ERROR: Failed to apply position control: {e}")
                    break
            
            # Step simulation
            try:
                pb.stepSimulation()
                if i % 1000 == 0:
                    print("Stepped simulation")
            except Exception as e:
                print(f"ERROR: Failed to step simulation: {e}")
                break
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Break after 1000 steps for testing
            if i == 999:
                print("First 1000 steps completed successfully")
        
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