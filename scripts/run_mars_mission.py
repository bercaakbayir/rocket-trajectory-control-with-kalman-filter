#!/usr/bin/env python3
"""
Mars Mission Simulation
Earth -> Mars -> Earth mission with variable gravity and adaptive control.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rocket_sim.config import (
    DT, MASS, EARTH_G, MARS_G, WIDTH, HEIGHT,
    EARTH_X, MARS_X, R_EARTH, R_MARS, R_GRAVITY_EARTH, R_GRAVITY_MARS,
    TRANSIT_SPEED, LANDING_SPEED
)
from rocket_sim.dynamics import Rocket, get_gravity
from rocket_sim.estimation import EKF
from rocket_sim.control import compute_lqr_gain, RocketPIDController


def main():
    SIM_TIME = 250.0
    USE_PID = True  # Toggle between PID and LQR
    
    rocket = Rocket(initial_x=EARTH_X, initial_y=0.0, use_variable_gravity=True)
    ekf = EKF(initial_x=EARTH_X, use_variable_gravity=True)

    # Pre-compute LQR gains for different environments
    K_EARTH = compute_lqr_gain(EARTH_G)
    K_MARS = compute_lqr_gain(MARS_G)
    K_SPACE = compute_lqr_gain(1.0)
    
    # PID Controller
    pid_controller = RocketPIDController()
    
    mission_stage = 0
    wait_start_time = 0
    current_target = np.array([EARTH_X, 0.0, 0, 0, 0, 0])
    
    steps = int(SIM_TIME / DT)
    
    true_states = []
    est_states = []
    inputs = []
    targets_hist = []
    
    print(f"Mission Start: Earth -> Mars -> Earth (Controller: {'PID' if USE_PID else 'LQR'})")
    
    for i in range(steps):
        current_time = i * DT
        x_est, y_est = ekf.x[0], ekf.x[1]
        
        stage_goal = current_target.copy()
        current_stage_speed = LANDING_SPEED
        transit_mode = False  # Flag for increased thrust during transit
        
        # Orbit height outside atmosphere (above gravity radius)
        ORBIT_HEIGHT = 20.0
        
        # State machine logic
        if mission_stage == 0:  # Launch from Earth (vertical ascent)
            stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            if abs(y_est - ORBIT_HEIGHT) < 1.0 and abs(ekf.x[3]) < 0.5:
                mission_stage = 1
                print(f"[{current_time:.2f}s] Orbit Reached. Beginning Horizontal Transit to Mars.")
                
        elif mission_stage == 1:  # Transit to Mars (horizontal flight)
            dist_to_mars = abs(x_est - MARS_X)
            vx = ekf.x[2]
            # Brake earlier and more aggressively
            stopping_dist = max(100.0, abs(vx * vx) / 1.5)
            
            if dist_to_mars > stopping_dist:
                # Accelerating phase - thrust towards Mars
                target_theta = -1.2
                target_vx = TRANSIT_SPEED
                current_stage_speed = TRANSIT_SPEED
                stage_goal = np.array([MARS_X, ORBIT_HEIGHT, target_vx, 0, target_theta, 0])
                transit_mode = True
            else:
                # Decelerating phase: let velocity-tracking PID handle braking
                target_theta = 0.0
                target_vx = 0.0
                # Clamp ghost target to destination during braking
                stage_goal = np.array([MARS_X, ORBIT_HEIGHT, target_vx, 0, target_theta, 0])
                current_target[0] = MARS_X
                transit_mode = True
            
            if dist_to_mars < 10.0 and abs(vx) < 2.0:
                mission_stage = 7
                print(f"[{current_time:.2f}s] Arrived at Mars Area. Stabilizing.")

        elif mission_stage == 7:  # Mars Stabilize
            stage_goal = np.array([MARS_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            transit_mode = False # Back to normal precision
            if abs(x_est - MARS_X) < 1.0 and abs(y_est - ORBIT_HEIGHT) < 1.0 and abs(ekf.x[2]) < 0.5:
                mission_stage = 2
                print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 2:  # Land Mars
            stage_goal = np.array([MARS_X, 0.0, 0, -LANDING_SPEED, 0, 0])
            current_stage_speed = LANDING_SPEED
            if abs(y_est - 0.0) < 0.5 and abs(ekf.x[3]) < 0.5:
                mission_stage = 3
                wait_start_time = current_time
                print(f"[{current_time:.2f}s] Touchdown Mars. Waiting 5s.")
                
        elif mission_stage == 3:  # Wait on Mars
            stage_goal = np.array([MARS_X, 0.0, 0, 0, 0, 0])
            if (current_time - wait_start_time) > 5.0:
                mission_stage = 4
                print(f"[{current_time:.2f}s] Launching from Mars.")
                
        elif mission_stage == 4:  # Launch Mars (vertical ascent)
            stage_goal = np.array([MARS_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            if abs(y_est - ORBIT_HEIGHT) < 1.0:
                mission_stage = 5
                print(f"[{current_time:.2f}s] Mars Orbit Reached. Beginning Horizontal Transit to Earth.")
        
        elif mission_stage == 5:  # Transit to Earth (horizontal flight)
            dist_to_earth = abs(x_est - EARTH_X)
            vx = ekf.x[2]
            stopping_dist = max(70.0, abs(vx * vx) / 1.5)
            
            if dist_to_earth > stopping_dist:
                target_theta = 1.2
                target_vx = -TRANSIT_SPEED
                current_stage_speed = TRANSIT_SPEED
                stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, target_vx, 0, target_theta, 0])
                transit_mode = True
            else:
                target_theta = 0.0 # Let PID handle braking
                target_vx = 0.0
                stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, target_vx, 0, target_theta, 0])
                current_target[0] = EARTH_X
                transit_mode = True
            
            if dist_to_earth < 10.0 and abs(vx) < 2.0:
                mission_stage = 8
                print(f"[{current_time:.2f}s] Earth Orbit Reached. Stabilizing.")

        elif mission_stage == 8:  # Earth Stabilize
            stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            current_stage_speed = 5.0
            pos_ok = abs(x_est - EARTH_X) < 1.0 and abs(y_est - ORBIT_HEIGHT) < 1.0
            vel_ok = abs(ekf.x[2]) < 0.5 and abs(ekf.x[3]) < 0.5
            if pos_ok and vel_ok:
                mission_stage = 6
                print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 6:  # Land Earth
            stage_goal = np.array([EARTH_X, 0.0, 0, -LANDING_SPEED, 0, 0])
            current_stage_speed = LANDING_SPEED
            if abs(y_est - 0.0) < 0.5 and abs(ekf.x[3]) < 0.5:
                mission_stage = 9
                print(f"[{current_time:.2f}s] Touchdown Earth. Mission Complete!")
        
        elif mission_stage == 9: # Post-landing wait
            stage_goal = np.array([EARTH_X, 0.0, 0, 0, 0, 0])
            current_stage_speed = 0.0

        # Smooth trajectory update
        move_speed = current_stage_speed * DT
        
        dx = stage_goal[0] - current_target[0]
        if abs(dx) > move_speed:
            current_target[0] += np.sign(dx) * move_speed
        else:
            current_target[0] = stage_goal[0]
            
        dy = stage_goal[1] - current_target[1]
        if abs(dy) > move_speed:
            current_target[1] += np.sign(dy) * move_speed
        else:
            current_target[1] = stage_goal[1]
            
        # Update target orientation and velocities directly
        current_target[2] = stage_goal[2]
        current_target[3] = stage_goal[3]
        current_target[4] = stage_goal[4]

        targets_hist.append(current_target.copy())

        # Measurement & Update
        z = rocket.get_measurement()
        ekf.update(z)
        
        # Control
        g_current = get_gravity(ekf.x[0], ekf.x[1])
        
        if USE_PID:
            # PID Control with variable gravity
            u = pid_controller.compute_control(ekf.x, current_target, gravity=g_current)
        else:
            # LQR Control
            if g_current < 0.1:
                K_current = K_SPACE
            elif abs(ekf.x[0] - EARTH_X) < abs(ekf.x[0] - MARS_X):
                K_current = K_EARTH
            else:
                K_current = K_MARS
            
            error = ekf.x - current_target
            u_optimal = -K_current @ error
            
            tilt_factor = max(0.2, np.cos(ekf.x[4]))
            F_eq = (MASS * g_current) / tilt_factor
            u = u_optimal + np.array([F_eq, 0.0])
        # Input saturation - higher limits during transit
        if transit_mode:
            u[0] = np.clip(u[0], 0, 80.0)  # Higher thrust for transit
            u[1] = np.clip(u[1], -50.0, 50.0)  # More torque authority
        else:
            u[0] = np.clip(u[0], 0, 50.0)
            u[1] = np.clip(u[1], -30.0, 30.0)
        
        ekf.predict(u)
        rocket.step(u)
        
        true_states.append(rocket.state.copy())
        est_states.append(ekf.x.copy())
        inputs.append(u)

    # Convert arrays
    true_states = np.array(true_states)
    est_states = np.array(est_states)
    inputs = np.array(inputs)
    targets_hist = np.array(targets_hist)
    
    # --- Animation ---
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_xlim(EARTH_X - 70, MARS_X + 70)
    ax_anim.set_ylim(-70, 70)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title("Earth-Mars Mission (LQR + EKF)")
    
    # Planets
    g_earth_patch = Circle((EARTH_X, -R_EARTH), R_GRAVITY_EARTH, color='skyblue', alpha=0.15)
    ax_anim.add_patch(g_earth_patch)
    earth_patch = Circle((EARTH_X, -R_EARTH), R_EARTH, color='blue', alpha=0.6, label='Earth')
    ax_anim.add_patch(earth_patch)
    
    g_mars_patch = Circle((MARS_X, -R_MARS), R_GRAVITY_MARS, color='skyblue', alpha=0.15)
    ax_anim.add_patch(g_mars_patch)
    mars_patch = Circle((MARS_X, -R_MARS), R_MARS, color='red', alpha=0.6, label='Mars')
    ax_anim.add_patch(mars_patch)
    
    # Rocket
    rocket_body = Polygon([[0,0]]*4, color='blue', alpha=0.8, label='Rocket')
    rocket_est = Polygon([[0,0]]*4, color='gray', alpha=0.5, linestyle='--')
    flame = Polygon([[0,0]]*3, color='orange')
    rcs_left = Polygon([[0,0]]*3, color='white')
    rcs_right = Polygon([[0,0]]*3, color='white')
    
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.add_patch(flame)
    ax_anim.add_patch(rcs_left)
    ax_anim.add_patch(rcs_right)
    
    target_marker, = ax_anim.plot([], [], 'gx', markersize=10, label='Target')
    ax_anim.legend(loc='upper right')
    
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    
    # Input plots
    ax_thrust = fig.add_subplot(gs[1, 0])
    line_thrust, = ax_thrust.plot([], [], 'r')
    ax_thrust.set_title("Thrust")
    ax_thrust.set_ylim(0, 55)
    
    ax_x = fig.add_subplot(gs[1, 1])
    line_x, = ax_x.plot([], [], 'b')
    ax_x.set_title("X Position")
    ax_x.set_ylim(-5, 350)

    def animate(i):
        idx = min(i, len(true_states) - 1)
        x, y, vx, vy, theta, omega = true_states[idx]
        xe, ye, vxe, vye, theta_e, omega_e = est_states[idx]
        tx, ty = targets_hist[idx][0], targets_hist[idx][1]
        
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
        torque = inputs[idx][1]
        
        flame_len = thrust / 20.0
        flame_pts = np.array([[-w/4, -h/2], [w/4, -h/2], [0, -h/2 - flame_len]])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        flame.set_xy(rot_flame)
        
        rcs_size = 0.3
        rcs_top = h/2 - 0.2
        
        if torque < -1.0:
            pts = np.array([[-w/2, rcs_top], [-w/2 - rcs_size, rcs_top], [-w/2, rcs_top - rcs_size]])
            rcs_left.set_xy(pts @ R.T + np.array([x, y]))
            rcs_left.set_visible(True)
        else:
            rcs_left.set_visible(False)
            
        if torque > 1.0:
            pts = np.array([[w/2, rcs_top], [w/2 + rcs_size, rcs_top], [w/2, rcs_top - rcs_size]])
            rcs_right.set_xy(pts @ R.T + np.array([x, y]))
            rcs_right.set_visible(True)
        else:
            rcs_right.set_visible(False)
            
        target_marker.set_data([tx], [ty])
        time_text.set_text(f"Time: {idx*DT:.1f}s")
        
        start = max(0, idx-200)
        times = np.arange(start, idx+1) * DT
        line_thrust.set_data(times, inputs[start:idx+1, 0])
        ax_thrust.set_xlim(times[0], times[-1]+5)
        
        line_x.set_data(times, true_states[start:idx+1, 0])
        ax_x.set_xlim(times[0], times[-1]+5)
        
        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(true_states), interval=20, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
