"""
Rocket dynamics and environment physics.
"""
import numpy as np
from .config import (
    DT, MASS, M_INERTIA, GRAVITY, EARTH_G, MARS_G,
    R_EARTH, R_MARS, SIGMA_PROCESS_VEL, SIGMA_PROCESS_RATE
)


def get_gravity(x, y):
    """
    Returns vertical gravity magnitude based on position.
    Simulates separate gravity wells for Earth/Mars with a continuous blend.
    """
    from .config import EARTH_X, MARS_X, R_EARTH, R_MARS, EARTH_G, MARS_G
    
    h = max(0, y)
    
    # Midpoint for gravity transition
    midpoint = (EARTH_X + MARS_X) / 2.0
    transition_width = abs(MARS_X - EARTH_X) * 0.4
    
    # Earth Component
    g_earth_val = 0.0
    dist_to_earth_x = abs(x - EARTH_X)
    if dist_to_earth_x < midpoint + transition_width:
        base = EARTH_G * (R_EARTH / (R_EARTH + h))**2
        # Transition factor
        factor = np.clip((midpoint + transition_width - x) / (2 * transition_width), 0.0, 1.0)
        g_earth_val = base * factor
        
    # Mars Component
    g_mars_val = 0.0
    dist_to_mars_x = abs(x - MARS_X)
    if dist_to_mars_x < midpoint + transition_width:
        base = MARS_G * (R_MARS / (R_MARS + h))**2
        # Transition factor
        factor = np.clip((x - (midpoint - transition_width)) / (2 * transition_width), 0.0, 1.0)
        g_mars_val = base * factor
        
    # Minimum Control Floor (Deep Space)
    return max(0.1, g_earth_val, g_mars_val)


class Rocket:
    """
    Simulates 2D rocket dynamics with 6-DOF state.
    State: [x, y, vx, vy, theta, omega]
    Input: [Thrust, Torque]
    """
    
    def __init__(self, initial_x=0.0, initial_y=0.0, use_variable_gravity=False):
        self.state = np.zeros(6)
        self.state[0] = initial_x
        self.state[1] = initial_y
        self.use_variable_gravity = use_variable_gravity
        self.history = []

    def dynamics(self, state, u):
        """Compute state derivatives (continuous dynamics)."""
        x, y, vx, vy, theta, omega = state
        F_thrust, Torque = u
        
        if self.use_variable_gravity:
            g_local = get_gravity(x, y)
        else:
            g_local = GRAVITY

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        ax = -(F_thrust / MASS) * sin_t
        ay = (F_thrust / MASS) * cos_t - g_local
        alpha = Torque / M_INERTIA

        return np.array([vx, vy, ax, ay, omega, alpha])

    def step(self, u):
        """RK4 Integration step."""
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * DT * k1, u)
        k3 = self.dynamics(self.state + 0.5 * DT * k2, u)
        k4 = self.dynamics(self.state + DT * k3, u)
        
        self.state += (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ground constraint
        if self.state[1] < 0:
            self.state[1] = 0
            self.state[3] = max(0, self.state[3])

        # Add process noise
        noise = np.random.normal(
            0, 
            [0, 0, SIGMA_PROCESS_VEL, SIGMA_PROCESS_VEL, 0, SIGMA_PROCESS_RATE], 
            6
        )
        self.state += noise
        
        self.history.append(self.state.copy())
        return self.state

    def get_measurement(self):
        """Returns noisy measurement [x, y, theta]."""
        from .config import SIGMA_MEASURE_POS, SIGMA_MEASURE_ANGLE
        meas = np.array([self.state[0], self.state[1], self.state[4]])
        noise = np.random.normal(
            0, 
            [SIGMA_MEASURE_POS, SIGMA_MEASURE_POS, SIGMA_MEASURE_ANGLE], 
            3
        )
        return meas + noise
