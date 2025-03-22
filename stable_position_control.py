import pybullet as pb
import pybullet_data
import time
import numpy as np
from helpers import setup_camera_controls

def main():
    try:
        # Connect to PyBullet with GUI
        physicsClient = pb.connect(pb.GUI)
        print("Connected to PyBullet")
        
        # Configure visualization
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
        # Explicitly disable wireframe mode to avoid visualization issues
        pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME, 0)
        # Disable keyboard shortcuts to prevent wireframe toggle
        pb.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        
        # Configure physics for stability
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)
        pb.setPhysicsEngineParameter(fixedTimeStep=1/240.0, numSolverIterations=10, numSubSteps=1)
        
        # Load ground plane
        planeId = pb.loadURDF("plane.urdf")
        print("Loaded ground plane")
        
        # Load humanoid robot
        startPos = [0, 0, 0.9]  # Lower starting position to avoid falling
        startOrientation = pb.getQuaternionFromEuler([0, 0, 0])
        robotId = pb.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation,
                              flags=pb.URDF_MAINTAIN_LINK_ORDER)
        print(f"Loaded humanoid robot, ID: {robotId}")
        
        # Set initial camera view
        pb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1.0]
        )
        
        # Setup interactive camera controls from helpers.py
        # The modified version has W key for forward movement only, no wireframe toggle
        update_camera = setup_camera_controls()
        print("Camera controls enabled: Arrow keys to rotate, +/- to zoom, WASD to pan, Q/E for up/down")
        print("NOTE: W key is for camera movement only, wireframe toggle disabled")
        
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
                
                # Add damping for stability
                pb.changeDynamics(
                    robotId, 
                    i, 
                    jointDamping=5.0,  # Strong damping
                    linearDamping=0.9,
                    angularDamping=0.9,
                    maxJointVelocity=10.0  # Limit velocity
                )
                
                # Reset joint state to zeros with zero velocity
                pb.resetJointState(robotId, i, 0, 0)
        
        # Wait for visualization to initialize
        time.sleep(1.0)
        print("Starting simulation...")
        
        # Small early stabilization period without control to let it settle
        for i in range(100):
            pb.stepSimulation()
            time.sleep(1/240.0)
        
        # Position control parameters
        kp = 0.3  # Very low gains for stability
        kd = 0.5  # Higher damping than position gain
        max_force = 20.0  # Low force to avoid instability
        
        # Set target positions (all zeros)
        target_positions = np.zeros(len(joint_indices))
        
        # Main simulation loop
        for i in range(10000):
            # Update camera based on user input
            update_camera()
            
            # Check if robot is still stable
            try:
                pos, _ = pb.getBasePositionAndOrientation(robotId)
                if i % 100 == 0:
                    print(f"Step {i}: Robot at position {pos}")
                    
                    # If position is NaN, early stop
                    if np.isnan(pos[0]) or np.isnan(pos[1]) or np.isnan(pos[2]):
                        print("ERROR: Robot position is NaN, stopping")
                        break
            except Exception as e:
                print(f"ERROR: Could not get robot position: {e}")
                break
            
            # Apply position control to all joints with very conservative parameters
            for j, joint_idx in enumerate(joint_indices):
                try:
                    # Get current joint state
                    joint_state = pb.getJointState(robotId, joint_idx)
                    current_pos = joint_state[0]
                    current_vel = joint_state[1]
                    
                    # Very small step towards target
                    if abs(current_pos - target_positions[j]) > 0.01:
                        # Only move a tiny bit toward target each step
                        direction = 1 if target_positions[j] > current_pos else -1
                        target_this_step = current_pos + direction * 0.001
                        
                        pb.setJointMotorControl2(
                            bodyUniqueId=robotId,
                            jointIndex=joint_idx,
                            controlMode=pb.POSITION_CONTROL,
                            targetPosition=target_this_step,  # Incremental movement
                            positionGain=kp,
                            velocityGain=kd,
                            force=max_force
                        )
                except Exception as e:
                    print(f"ERROR at joint {joint_idx}: {e}")
                    continue
            
            # Step simulation 
            pb.stepSimulation()
            
            # Sleep to make it real-time
            time.sleep(1./240.)
            
            # Check for quit key - use 'x' instead of 'q' to avoid conflicts
            keys = pb.getKeyboardEvents()
            if ord('x') in keys and keys[ord('x')] & pb.KEY_WAS_TRIGGERED:
                print("X key pressed, exiting")
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