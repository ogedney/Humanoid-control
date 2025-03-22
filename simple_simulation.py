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
        
        # Load ground plane
        planeId = pb.loadURDF("plane.urdf")
        print("Loaded ground plane")
        
        # Load humanoid robot - standard position, no rotation
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
        
        # Add a small delay to ensure visualization is ready
        time.sleep(0.5)
        
        # Simple simulation loop
        should_exit = False
        for i in range(10000):
            if should_exit:
                break
                
            # Step simulation
            pb.stepSimulation()
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Optional: exit with Q key
            try:
                keys = pb.getKeyboardEvents()
                if ord('q') in keys and keys[ord('q')] & pb.KEY_WAS_TRIGGERED:
                    should_exit = True
                    print("Q key pressed, exiting")
            except Exception as e:
                print(f"Error getting keyboard events: {e}")
                break
        
        print("Simulation completed successfully")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Ensure disconnect happens even if there are errors
        try:
            pb.disconnect()
            print("Disconnected from PyBullet")
        except:
            pass

if __name__ == "__main__":
    main() 