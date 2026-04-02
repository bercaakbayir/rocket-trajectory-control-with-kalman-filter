#!/usr/bin/env python3
"""
Rocket Hover Test Simulation
Tests EKF + PID control on a single planet, matching Mars mission methodology.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rocket_sim.config import (
    DT, MASS, EARTH_G, WIDTH, HEIGHT,
    EARTH_X, R_EARTH
)
from rocket_sim.dynamics import Rocket, get_gravity
from rocket_sim.estimation import EKF
from rocket_sim.control import RocketPIDController


def main():
    # Configuration
    SIM_TIME = 20.0
    INITIAL_X = EARTH_X
    HOVER_HEIGHT = 15.0
    
    # Initialize
    rocket = Rocket(initial_x=INITIAL_X, initial_y=0.0, use_variable_gravity=True)
    ekf = EKF(initial_x=INITIAL_X, use_variable_gravity=True)
    
    pid_controller = RocketPIDController()
    
    print(f"Simulation Start. Controller: PID + EKF (Variable Gravity Mode)")
    
    # Target State (Hover at defined height)
    x_target = np.array([INITIAL_X, HOVER_HEIGHT, 0, 0, 0, 0])
    
    # Histories
    true_states = []
    est_states = []
    inputs = []
    
    steps = int(SIM_TIME / DT)
    
    # Simulation loop
    for i in range(steps):
        # Measurement & EKF Update
        z = rocket.get_measurement()
        ekf.update(z)
        
        # PID Control with variable gravity logic
        g_current = get_gravity(ekf.x[0], ekf.x[1])
        u = pid_controller.compute_control(ekf.x, x_target, gravity=g_current)
        
        # Input saturation
        u[0] = np.clip(u[0], 0, 80.0)  # Standard thrust limit
        u[1] = np.clip(u[1], -30.0, 30.0) # Standard torque limit
        
        # Predict & Step
        ekf.predict(u)
        rocket.step(u)
        
        true_states.append(rocket.state.copy())
        est_states.append(ekf.x.copy())
        inputs.append(u)

    # Convert to arrays
    true_states = np.array(true_states)
    est_states = np.array(est_states)
    inputs = np.array(inputs)
    
    # --- Animation ---
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_facecolor('black')
    ax_anim.set_xlim(INITIAL_X - 50, INITIAL_X + 50)
    ax_anim.set_ylim(-10, 50)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, color='gray', alpha=0.3, linestyle='--')
    ax_anim.set_title("Rocket Hover Test (PID + EKF)", color='white')
    
    # Rocket
    rocket_body = Polygon([[0,0]]*4, color='cyan', alpha=0.9, label='True State')
    rocket_est = Polygon([[0,0]]*4, color='white', alpha=0.4, linestyle='--')
    flame = Polygon([[0,0]]*3, color='orange')
    
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.add_patch(flame)
    
    ax_anim.plot([INITIAL_X], [HOVER_HEIGHT], 'gx', markersize=10, label='Target')
    ax_anim.legend(loc='upper right')
    
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, color='white')
    
    # Input plots
    ax_thrust = fig.add_subplot(gs[1, 0])
    ax_thrust.set_facecolor('black')
    line_thrust, = ax_thrust.plot([], [], 'r', label='Thrust')
    ax_thrust.set_title("Thrust (N)", color='white')
    ax_thrust.tick_params(colors='white')
    ax_thrust.set_ylim(0, 100)
    
    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.set_facecolor('black')
    line_y, = ax_y.plot([], [], 'b', label='Altitude')
    ax_y.set_title("Altitude (m)", color='white')
    ax_y.tick_params(colors='white')
    ax_y.set_ylim(-1, 20)

    def animate(i):
        idx = min(i, len(true_states) - 1)
        x, y, vx, vy, theta, omega = true_states[idx]
        xe, ye, vxe, vye, theta_e, omega_e = est_states[idx]
        
        w, h = WIDTH, HEIGHT
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_corners = corners @ R.T + np.array([x, y])
        rocket_body.set_xy(rot_corners)
        
        ce, se = np.cos(theta_e), np.sin(theta_e)
        Re = np.array([[ce, -se], [se, ce]])
        rot_corners_e = corners @ Re.T + np.array([xe, ye])
        rocket_est.set_xy(rot_corners_e)
        
        thrust = inputs[idx][0]
        flame_len = thrust / 20.0
        flame_pts = np.array([[-w/4, -h/2], [w/4, -h/2], [0, -h/2 - flame_len]])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        flame.set_xy(rot_flame)
        
        time_text.set_text(f"Time: {idx*DT:.1f}s")
        
        start = max(0, idx-400)
        times = np.arange(start, idx+1) * DT
        line_thrust.set_data(times, inputs[start:idx+1, 0])
        ax_thrust.set_xlim(times[0], times[-1]+1)
        
        line_y.set_data(times, true_states[start:idx+1, 1])
        ax_y.set_xlim(times[0], times[-1]+5)
        
        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(true_states), interval=20, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
