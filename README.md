# Humanoid Control System

A PyBullet-based system for humanoid robot control with multiple control strategies.

## Overview

This project provides a modular framework for controlling a humanoid robot in PyBullet simulation. It allows easy switching between different control strategies like position control, velocity control, torque control, and PD control.

## Controllers

The system uses a controller hierarchy with:

- `Controller`: Abstract base class that all controllers inherit from
- `PositionController`: Controls joints to reach specific positions
- `VelocityController`: Controls joints to maintain specific velocities  
- `TorqueController`: Directly applies torques to joints
- `PDController`: Position-derivative control for more complex motions

## Usage

To run the simulation with the default position controller:

```bash
python main.py
```

### Camera Controls

- Arrow keys: Rotate camera view
- +/-: Zoom in/out
- WASD: Pan camera horizontally
- Q/E: Pan camera vertically
- X: Exit simulation

## Implementing Custom Controllers

To create a new controller:

1. Extend the `Controller` base class in `controllers.py`
2. Implement the required `update()` method
3. Add your controller to the controller dictionary in `create_controller()`

Example:

```python
class MyCustomController(Controller):
    def __init__(self, robot_id, joint_indices, joint_names, **kwargs):
        super().__init__(robot_id, joint_indices, joint_names)
        # Initialize controller-specific parameters
        
    def update(self):
        # Implement your control strategy here
```

Then update the controller factory:

```python
controllers = {
    'position': PositionController,
    'velocity': VelocityController,
    'torque': TorqueController,
    'pd': PDController,
    'custom': MyCustomController  # Add your controller
}
```

## Switching Controllers

To switch controllers, update the `controller_type` variable in `main.py`:

```python
controller_type = 'velocity'  # Change to velocity control
controller = create_controller(
    controller_type=controller_type,
    robot_id=robotId,
    joint_indices=joint_indices,
    joint_names=joint_names,
    # Additional parameters specific to the controller
)
``` 