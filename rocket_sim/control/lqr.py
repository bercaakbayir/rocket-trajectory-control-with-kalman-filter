"""
LQR Controller for rocket control.
"""
import numpy as np
import scipy.linalg
from ..config import MASS, M_INERTIA, GRAVITY


def compute_lqr_gain(gravity_val=None):
    """
    Compute LQR gain matrix for rocket hover control.
    Linearized around hover state (theta=0, F=mg).
    
    Args:
        gravity_val: Local gravity value (default: GRAVITY constant)
        
    Returns:
        K: 2x6 gain matrix
    """
    if gravity_val is None:
        gravity_val = GRAVITY
    
    # State: [x, y, vx, vy, theta, omega]
    # Input: [F, Tau]
    
    # A Matrix (linearized dynamics)
    A = np.zeros((6, 6))
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 4] = -gravity_val  # Linearization of -F/m sin(theta) with F=mg
    A[3, 4] = 0
    A[4, 5] = 1
    
    # B Matrix (input coupling)
    B = np.zeros((6, 2))
    B[3, 0] = 1.0 / MASS
    B[5, 1] = 1.0 / M_INERTIA
    
    # Cost matrices
    Q_cost = np.diag([2.0, 5.0, 5.0, 5.0, 50.0, 5.0])
    R_cost = np.diag([0.01, 1.0])
    
    # Solve Riccati equation
    P_sol = scipy.linalg.solve_continuous_are(A, B, Q_cost, R_cost)
    K = np.linalg.inv(R_cost) @ B.T @ P_sol
    
    return K


def compute_lqr_control(K, state, target, gravity_val=None):
    """
    Compute LQR control input.
    
    Args:
        K: LQR gain matrix
        state: Current state [x, y, vx, vy, theta, omega]
        target: Target state [x, y, vx, vy, theta, omega]
        gravity_val: Local gravity for feedforward
        
    Returns:
        u: Control input [Thrust, Torque]
    """
    if gravity_val is None:
        gravity_val = GRAVITY
        
    error = state - target
    u_optimal = -K @ error
    
    # Add equilibrium feedforward
    F_eq = MASS * gravity_val
    u = u_optimal + np.array([F_eq, 0.0])
    
    return u
