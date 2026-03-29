"""
PID Controller for rocket control.
"""
import numpy as np
from ..config import DT, MASS, GRAVITY, EARTH_G


class PID:
    """Generic PID controller with anti-windup."""
    
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
        
        # Simple anti-windup: limit integrator range
        if self.limits:
            int_limit = self.limits[1] * 2.0 # Allow some integration but not too much
            self.integrator = np.clip(self.integrator, -int_limit, int_limit)
            
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
        # Use more D for aggressive damping
        self.pid_alt = PID(kp=8.0, ki=0.5, kd=20.0, limits=(-MASS*EARTH_G, 30.0))
        
        # Lateral Position Loop (X) -> Target Angle
        self.pid_x = PID(kp=0.2, ki=0.02, kd=0.5, limits=(-0.4, 0.4))
        
        # Attitude Loop (Theta) -> Torque
        self.pid_theta = PID(kp=40.0, ki=0.05, kd=15.0, limits=(-5, 5))

    def compute_control(self, state, target, gravity=None):
        """
        Compute control input [Thrust, Torque].
        state: [x, y, vx, vy, theta, omega]
        target: [x, y, vx, vy, theta, omega] target state
        gravity: local gravity value (optional, for variable gravity)
        """
        if gravity is None:
            gravity = GRAVITY
            
        x, y, vx, vy, theta, omega = state
        tx, ty = target[0], target[1]
        tvx, tvy = target[2], target[3]
        
        # Altitude Control with velocity tracking
        err_y = ty - y
        err_vy = tvy - vy
        # Add position and velocity correction to base weight
        thrust_base = MASS * gravity + self.pid_alt.update(err_y) + 5.0 * err_vy
        # COMPENSATE FOR TILT: ay = (T*cos(theta) - mg)/m. To maintain ay target, need T = (mg + corrections)/cos(theta)
        # Limit the compensation to avoid extreme thrust at very high tilt
        cos_theta = np.cos(theta)
        cos_theta_clamped = max(0.2, abs(cos_theta)) # Max ~5x thrust mult
        thrust_cmd = thrust_base / cos_theta_clamped
        
        # Lateral Control -> Target Angle Delta
        # Use target[4] (if set) as base, then add PID correction for position & velocity
        base_theta = target[4] if len(target) > 4 else 0.0
        err_x = tx - x
        err_vx = tvx - vx
        
        # Combined position and velocity control
        # Increased err_vx damping coefficient for better braking
        theta_target = base_theta - self.pid_x.update(err_x) - 0.2 * err_vx
        
        # Safeguard: Limit tilt to prevent falling (e.g., max 80 degrees)
        max_tilt = 1.4  # ~80 degrees
        theta_target = np.clip(theta_target, -max_tilt, max_tilt)
        
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
