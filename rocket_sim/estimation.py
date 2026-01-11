"""
Extended Kalman Filter for rocket state estimation.
"""
import numpy as np
from .config import (
    DT, MASS, M_INERTIA, GRAVITY,
    SIGMA_PROCESS_POS, SIGMA_PROCESS_VEL, SIGMA_PROCESS_ANGLE, SIGMA_PROCESS_RATE,
    SIGMA_MEASURE_POS, SIGMA_MEASURE_ANGLE
)
from .dynamics import get_gravity


class EKF:
    """
    Extended Kalman Filter for 6-DOF rocket state estimation.
    State: [x, y, vx, vy, theta, omega]
    Measurement: [x, y, theta]
    """
    
    def __init__(self, initial_x=0.0, use_variable_gravity=False):
        self.x = np.zeros(6)
        self.x[0] = initial_x
        self.use_variable_gravity = use_variable_gravity
        self.P = np.eye(6) * 0.1
        
        self.Q = np.diag([
            SIGMA_PROCESS_POS**2, SIGMA_PROCESS_POS**2,
            SIGMA_PROCESS_VEL**2, SIGMA_PROCESS_VEL**2,
            SIGMA_PROCESS_ANGLE**2, SIGMA_PROCESS_RATE**2
        ])
        
        self.R = np.diag([
            SIGMA_MEASURE_POS**2, SIGMA_MEASURE_POS**2, SIGMA_MEASURE_ANGLE**2
        ])

    def predict(self, u):
        """Predict step using control input u = [Thrust, Torque]."""
        F_thrust, Torque = u
        theta = self.x[4]
        
        if self.use_variable_gravity:
            g_local = get_gravity(self.x[0], self.x[1])
        else:
            g_local = GRAVITY
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        acc_x = -(F_thrust / MASS) * sin_t
        acc_y = (F_thrust / MASS) * cos_t - g_local
        alpha = Torque / M_INERTIA
        
        # Euler predict
        self.x[0] += self.x[2] * DT
        self.x[1] += self.x[3] * DT
        self.x[2] += acc_x * DT
        self.x[3] += acc_y * DT
        self.x[4] += self.x[5] * DT
        self.x[5] += alpha * DT
        
        # Jacobian F
        F_jac = np.eye(6)
        F_jac[0, 2] = DT
        F_jac[1, 3] = DT
        F_jac[4, 5] = DT
        
        d_vx_dtheta = DT * (-(F_thrust / MASS) * cos_t)
        d_vy_dtheta = DT * (-(F_thrust / MASS) * sin_t)
        
        F_jac[2, 4] = d_vx_dtheta
        F_jac[3, 4] = d_vy_dtheta
        
        self.P = F_jac @ self.P @ F_jac.T + self.Q

    def update(self, z):
        """Update step with measurement z = [x, y, theta]."""
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 4] = 1
        
        y = z - H @ self.x
        # Wrap angle residual
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
