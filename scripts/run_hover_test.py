#!/usr/bin/env python3
"""
Rocket Hover Test Simulation
Demonstrates LQR/PID control with EKF state estimation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rocket_sim.config import DT, MASS, GRAVITY, WIDTH, HEIGHT
from rocket_sim.dynamics import Rocket
from rocket_sim.estimation import EKF
from rocket_sim.control import RocketPIDController, compute_lqr_gain
from rocket_sim.visualization import RocketVisualizer, create_input_plots


def main():
    # Configuration
    SIM_TIME = 20.0
    USE_PID = True  # Toggle between PID and LQR
    
    # Initialize
    rocket = Rocket(initial_x=5.0, initial_y=0.0)
    ekf = EKF(initial_x=5.0)
    
    lqr_K = compute_lqr_gain()
    pid_controller = RocketPIDController()
    
    print(f"Simulation Start. Controller: {'PID' if USE_PID else 'LQR'}")
    
    # Target State (Hover at 10m)
    x_target = np.array([0.0, 10.0, 0, 0, 0, 0])
    
    # Histories
    true_states = []
    est_states = []
    inputs = []
    
    steps = int(SIM_TIME / DT)
    
    # Simulation loop
    for i in range(steps):
        z = rocket.get_measurement()
        ekf.update(z)
        
        state_est = ekf.x
        
        if USE_PID:
            u = pid_controller.compute_control(state_est, x_target)
        else:
            error = state_est - x_target
            u_optimal = -lqr_K @ error
            F_eq = MASS * GRAVITY
            u = u_optimal + np.array([F_eq, 0.0])
        
        # Input saturation
        u[0] = np.clip(u[0], 0, 3 * MASS * GRAVITY)
        u[1] = np.clip(u[1], -5.0, 5.0)
        
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
    fig = plt.figure(figsize=(16, 9))  # Wider, 16:9 aspect
    fig.canvas.manager.full_screen_toggle()  # Fullscreen
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_xlim(-25, 25)
    ax_anim.set_ylim(-15, 25)  # Show less Earth, focus on rocket area
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title("Rocket Hover Test (LQR/PID + EKF)")
    
    ax_thrust, ax_torque, line_thrust, line_torque = create_input_plots(fig, gs)
    
    # Target marker
    ax_anim.plot(0, 10, 'gx', markersize=10, label='Target')
    
    # Visualizer
    viz = RocketVisualizer(ax_anim, show_earth=True, show_rcs=True)
    ax_anim.legend()
    
    def animate(i):
        idx = min(i, len(true_states) - 1)
        
        viz.update(true_states[idx], est_states[idx], inputs[idx], idx * DT)
        
        # Update plots
        start = max(0, idx - 100)
        times = np.arange(start, idx + 1) * DT
        
        line_thrust.set_data(times, inputs[start:idx+1, 0])
        ax_thrust.set_xlim(times[0], times[-1] + 1)
        
        line_torque.set_data(times, inputs[start:idx+1, 1])
        ax_torque.set_xlim(times[0], times[-1] + 1)
        
        return []
    
    ani = animation.FuncAnimation(fig, animate, frames=len(true_states), interval=50, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
