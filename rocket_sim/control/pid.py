"""
PID Controller for rocket control.
"""
import numpy as np
from ..config import DT, MASS, GRAVITY


class PID:
    """Generic PID controller."""
    
    def __init__(self, kp, ki, kd, limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = limits  # (min, max)
        
        self.integrator = 0
        self.last_error = 0
        
    def update(self, error, dt=None):
        """Compute PID output for given error."""
        if dt is None:
            dt = DT
            
        self.integrator += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        output = self.kp * error + self.ki * self.integrator + self.kd * derivative
        
        if self.limits:
            output = np.clip(output, self.limits[0], self.limits[1])
            
        return output
        
    def reset(self):
        """Reset integrator and derivative state."""
        self.integrator = 0
        self.last_error = 0


class RocketPIDController:
    """
    Cascaded PID controller for rocket hover control.
    Altitude Loop (Y) -> Thrust
    Lateral Loop (X) -> Target Angle -> Attitude Loop (Theta) -> Torque
    """
    
    def __init__(self):
        # Altitude Loop (Y) -> Thrust
        # Increased D gain for damping vertical oscillations
        self.pid_alt = PID(kp=8.0, ki=0.5, kd=12.0, limits=(-MASS*GRAVITY, 30.0))
        
        # Lateral Position Loop (X) -> Target Angle
        # Added small integral term to eliminate steady-state error
        self.pid_x = PID(kp=0.2, ki=0.02, kd=0.5, limits=(-0.25, 0.25))
        
        # Attitude Loop (Theta) -> Torque
        # Increased D for angular damping to prevent oscillation
        self.pid_theta = PID(kp=40.0, ki=0.05, kd=15.0, limits=(-5, 5))

    def compute_control(self, state, target, gravity=None):
        """
        Compute control input [Thrust, Torque].
        state: [x, y, vx, vy, theta, omega]
        target: [x, y, ...] target state
        gravity: local gravity value (optional, for variable gravity)
        """
        if gravity is None:
            gravity = GRAVITY
            
        x, y, vx, vy, theta, omega = state
        tx, ty = target[0], target[1]
        
        # Altitude Control with velocity damping
        err_y = ty - y
        # Add velocity damping term directly (feedforward)
        thrust_cmd = MASS * gravity + self.pid_alt.update(err_y) - 2.0 * vy
        
        # Lateral Control -> Target Angle
        err_x = tx - x
        theta_target = -self.pid_x.update(err_x)
        
        # Attitude Control with angular rate damping
        err_theta = theta_target - theta
        # Add angular rate damping directly
        torque_cmd = self.pid_theta.update(err_theta) - 3.0 * omega
        
        return np.array([thrust_cmd, torque_cmd])
    
    def reset(self):
        """Reset all PID controllers."""
        self.pid_alt.reset()
        self.pid_x.reset()
        self.pid_theta.reset()
