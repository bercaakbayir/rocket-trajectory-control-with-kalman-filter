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
    TRANSIT_SPEED, LANDING_SPEED, ORBIT_HEIGHT
)
from rocket_sim.dynamics import Rocket, get_gravity
from rocket_sim.estimation import EKF
from rocket_sim.control import compute_lqr_gain, RocketPIDController


from matplotlib.widgets import Button

def main():
    SIM_TIME = 5000.0
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
    wait_start_time = 0.0
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
        
        # State machine logic
        if mission_stage == 0:  # Launch from Earth (vertical ascent)
            stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            current_stage_speed = 5000.0 # Faster ascent
            if abs(y_est - ORBIT_HEIGHT) < 10000.0:
                mission_stage = 1
                print(f"[{current_time:.2f}s] Orbit Reached. Beginning Horizontal Transit to Mars.")
                
        elif mission_stage == 1:  # Transit to Mars (horizontal flight)
            dist_to_mars = abs(x_est - MARS_X)
            vx = ekf.x[2]
            # Brake earlier and more aggressively
            stopping_dist = max(10000.0, abs(vx * vx) / 1.5)
            
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
            
            if dist_to_mars < 10000.0 and abs(vx) < 5.0:
                mission_stage = 7
                print(f"[{current_time:.2f}s] Arrived at Mars Area. Stabilizing.")

        elif mission_stage == 7:  # Mars Stabilize
            stage_goal = np.array([MARS_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            transit_mode = False 
            if abs(x_est - MARS_X) < 10000.0 and abs(y_est - ORBIT_HEIGHT) < 10000.0:
                mission_stage = 2
                print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 2:  # Land Mars
            stage_goal = np.array([MARS_X, 0.0, 0, -LANDING_SPEED, 0, 0])
            current_stage_speed = 5000.0
            if abs(y_est - 0.0) < 50.0:
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
            current_stage_speed = 5000.0
            if abs(y_est - ORBIT_HEIGHT) < 10000.0:
                mission_stage = 5
                print(f"[{current_time:.2f}s] Mars Orbit Reached. Beginning Horizontal Transit to Earth.")
        
        elif mission_stage == 5:  # Transit to Earth (horizontal flight)
            dist_to_earth = abs(x_est - EARTH_X)
            vx = ekf.x[2]
            stopping_dist = max(10000.0, abs(vx * vx) / 1.5)
            
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
            
            if dist_to_earth < 10000.0 and abs(vx) < 5.0:
                mission_stage = 8
                print(f"[{current_time:.2f}s] Earth Orbit Reached. Stabilizing.")

        elif mission_stage == 8:  # Earth Stabilize
            stage_goal = np.array([EARTH_X, ORBIT_HEIGHT, 0, 0, 0, 0])
            current_stage_speed = 5000.0
            pos_ok = abs(x_est - EARTH_X) < 10000.0 and abs(y_est - ORBIT_HEIGHT) < 10000.0
            if pos_ok:
                mission_stage = 6
                print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 6:  # Land Earth
            stage_goal = np.array([EARTH_X, 0.0, 0, -LANDING_SPEED, 0, 0])
            current_stage_speed = 5000.0
            if abs(y_est - 0.0) < 50.0:
                mission_stage = 9
                print(f"[{current_time:.2f}s] Touchdown Earth. Mission Complete!")
        
        elif mission_stage == 9: # Post-landing wait
            stage_goal = np.array([EARTH_X, 0.0, 0, 0, 0, 0])
            current_stage_speed = 0.0

        # Snap target instantly to stage goal — the target IS the waypoint,
        # and the rocket's PID handles the actual smooth movement toward it.
        current_target[:] = stage_goal[:]

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
    true_states_arr = np.array(true_states)
    est_states_arr = np.array(est_states)
    inputs_arr = np.array(inputs)
    targets_hist_arr = np.array(targets_hist)
    
    # --- Animation State ---
    class AnimState:
        def __init__(self):
            self.speed = 1
            self.curr_idx = 0
    state = AnimState()
    
    # --- Animation ---
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('black')
    fig.suptitle("INTERPLANETARY MISSION: EARTH TO MARS", color='#39FF14', fontsize=22, fontweight='bold', y=0.97)
    
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 1, 0.2], hspace=0.35)
    
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_facecolor('black')
    
    # Realistic limits for 200M scale with 5M orbit maneuvers
    ax_anim.set_xlim(EARTH_X - R_EARTH * 5.0, MARS_X + R_MARS * 5.0)
    ax_anim.set_ylim(-20000000.0, 40000000.0)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, color='gray', alpha=0.1, linestyle='--')
    
    # Path: Neon Green Dotted Line (matching new orbit altitude)
    path_x = [EARTH_X, EARTH_X, MARS_X, MARS_X]
    path_y = [0, ORBIT_HEIGHT, ORBIT_HEIGHT, 0]
    ax_anim.plot(path_x, path_y, color='#39FF14', linestyle=':', linewidth=2, alpha=0.8)

    # Planets
    # Visual Gravity "Atmospheres" (Glows) - Physics uses R_GRAVITY, but for display we use a smaller, brighter glow
    ax_anim.add_patch(Circle((EARTH_X, -R_EARTH), R_EARTH * 3.0, color='skyblue', alpha=0.15))
    earth_patch = Circle((EARTH_X, -R_EARTH), R_EARTH, color='#1a73e8', alpha=0.9, label='Earth')
    ax_anim.add_patch(earth_patch)
    
    ax_anim.add_patch(Circle((MARS_X, -R_MARS), R_MARS * 3.0, color='#ff3300', alpha=0.15))
    mars_patch = Circle((MARS_X, -R_MARS), R_MARS, color='#c1440e', alpha=0.9, label='Mars')
    ax_anim.add_patch(mars_patch)
    
    # Rocket
    rocket_body = Polygon([[0,0]]*4, color='#00f2ff', alpha=1.0, label='Rocket')
    rocket_est = Polygon([[0,0]]*4, color='white', alpha=0.4, linestyle='--')
    flame = Polygon([[0,0]]*3, color='#ffea00')
    rcs_left = Polygon([[0,0]]*3, color='white')
    rcs_right = Polygon([[0,0]]*3, color='white')
    
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.add_patch(flame)
    ax_anim.add_patch(rcs_left)
    ax_anim.add_patch(rcs_right)
    
    target_marker, = ax_anim.plot([], [], 'rx', markersize=12, markeredgewidth=2, label='Target')
    ax_anim.tick_params(colors='white', labelsize=8)
    
    # Legend at Top-Right, slightly above the plot to avoid overlapping planets
    ax_anim.legend(loc='upper right', bbox_to_anchor=(1.0, 1.08), facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)
    
    # Input plots
    ax_thrust = fig.add_subplot(gs[1, 0])
    ax_thrust.set_facecolor('#0a0a0a')
    line_thrust, = ax_thrust.plot([], [], '#ff3300', linewidth=1.5)
    ax_thrust.set_title("ENGINE THRUST (N)", color='white', fontsize=11, fontweight='bold')
    ax_thrust.tick_params(colors='white', labelsize=9)
    ax_thrust.set_ylim(0, 100)
    ax_thrust.grid(True, alpha=0.1)
    
    ax_x = fig.add_subplot(gs[1, 1])
    ax_x.set_facecolor('#0a0a0a')
    line_x, = ax_x.plot([], [], '#00f2ff', linewidth=1.5)
    ax_x.set_title("MISSION PROGRESS (X-POS)", color='white', fontsize=11, fontweight='bold')
    ax_x.tick_params(colors='white', labelsize=9)
    ax_x.set_ylim(-R_EARTH, MARS_X + R_MARS)
    ax_x.grid(True, alpha=0.1)

    # Speed Toggle Button
    ax_speed = fig.add_subplot(gs[2, :])
    btn_speed = Button(ax_speed, 'Speed: 1x', color='#222', hovercolor='#444')
    btn_speed.label.set_color('white')
    btn_speed.label.set_fontweight('bold')
    btn_speed.label.set_fontsize(12)

    class Controls:
        def cycle_speed(self, event):
            speeds = [1, 2, 3, 5, 10]
            current_idx = speeds.index(state.speed)
            state.speed = speeds[(current_idx + 1) % len(speeds)]
            btn_speed.label.set_text(f"Speed: {state.speed}x")
            fig.canvas.draw_idle()
    
    controls = Controls()
    btn_speed.on_clicked(controls.cycle_speed)

    def animate(i):
        idx = state.curr_idx
        if idx >= len(true_states_arr):
            state.curr_idx = 0
            idx = 0
        
        state.curr_idx += state.speed
        
        x, y, vx, vy, theta, omega = true_states_arr[idx]
        xe, ye, vxe, vye, theta_e, omega_e = est_states_arr[idx]
        tx, ty = targets_hist_arr[idx][0], targets_hist_arr[idx][1]
        
        v_scale = 1000000.0 
        w, h = WIDTH * v_scale, HEIGHT * v_scale
        
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_corners = corners @ R.T + np.array([x, y])
        rocket_body.set_xy(rot_corners)
        
        ce, se = np.cos(theta_e), np.sin(theta_e)
        Re = np.array([[ce, -se], [se, ce]])
        rot_corners_e = corners @ Re.T + np.array([xe, ye])
        rocket_est.set_xy(rot_corners_e)
        
        thrust = inputs_arr[idx][0]
        torque = inputs_arr[idx][1]
        
        # Reduced flame length scaling
        fl_scale = (thrust / 10.0) * v_scale
        flame_pts = np.array([[-w/4, -h/2], [w/4, -h/2], [0, -h/2 - fl_scale]])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        flame.set_xy(rot_flame)
        
        rcs_left.set_visible(torque < -1.0)
        rcs_right.set_visible(torque > 1.0)
        if torque < -1.0:
            rcs_left.set_xy(np.array([[-w/2, h/2], [-w/2 - w, h/2], [-w/2, h/2 - w]]) @ R.T + [x, y])
        if torque > 1.0:
            rcs_right.set_xy(np.array([[w/2, h/2], [w/2 + w, h/2], [w/2, h/2 - w]]) @ R.T + [x, y])
            
        target_marker.set_data([tx], [ty])
        
        start = max(0, idx-500)
        times = np.arange(start, idx+1) * DT
        line_thrust.set_data(times, inputs_arr[start:idx+1, 0])
        ax_thrust.set_xlim(times[0], times[-1]+5)
        
        line_x.set_data(times, true_states_arr[start:idx+1, 0])
        ax_x.set_xlim(times[0], times[-1]+5)
        
        return []

    fig.btn_speed = btn_speed
    fig.controls = controls

    ani = animation.FuncAnimation(fig, animate, frames=len(true_states_arr), interval=10, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
