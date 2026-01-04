import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
import scipy.linalg

# --- Constants ---
DT = 0.01  # Time step 
SIM_TIME = 80.0  # Increased for longer distance
EARTH_G = 9.81
MARS_G = 3.73 # Updated as requested
EARTH_X = 0.0
MARS_X = 300.0  # 300m away (Separated)
TRANSIT_SPEED = 15.0 # m/s (Slower for better control)
LANDING_SPEED = 2.0 # m/s (Target walker speed)

# Planet Visuals
R_EARTH = 50.0  
R_MARS = 25.0
R_GRAVITY_EARTH = 60.0 # R_EARTH + 10
R_GRAVITY_MARS = 35.0 # R_MARS + 10

# Rocket Parameters
MASS = 1.0  # kg
WIDTH = 0.5 # m
HEIGHT = 2.0 # m
M_INERTIA = (1/12) * MASS * (WIDTH**2 + HEIGHT**2)
LENGTH_COG_TO_THRUSTER = 1.0

# Noise Parameters (Standard Deviations)
SIGMA_PROCESS_POS = 0.001
SIGMA_PROCESS_VEL = 0.001
SIGMA_PROCESS_ANGLE = 0.0001
SIGMA_PROCESS_RATE = 0.001
SIGMA_MEASURE_POS = 0.1
SIGMA_MEASURE_ANGLE = 0.01

def get_gravity(x, y):
    """
    Returns vertical gravity magnitude based on position (Altitude falloff).
    Simulates separate gravity wells with a continuous blend and floor.
    """
    h = max(0, y)
    
    # Earth Component
    # Fade out linearly from x=50 to x=150
    g_earth_val = 0.0
    if x < 150.0:
        base = EARTH_G * (R_EARTH / (R_EARTH + h))**2
        # Full strength until 50, then fade to 0 at 150
        factor = 1.0
        if x > 50.0:
            factor = np.clip((150.0 - x) / 100.0, 0.0, 1.0)
        g_earth_val = base * factor
        
    # Mars Component
    # Fade in linearly from x=150 to x=250
    g_mars_val = 0.0
    if x > 150.0:
        base = MARS_G * (R_MARS / (R_MARS + h))**2
        # Fade in from 0 at 150 to Full at 250
        factor = np.clip((x - 150.0) / 100.0, 0.0, 1.0)
        g_mars_val = base * factor
        
    # Minimum Control Floor
    # Ensure at least 0.5 m/s^2 so engines stay lit for LQR authority
    return max(0.5, g_earth_val, g_mars_val)

# --- Rocket Dynamics ---
class Rocket:
    def __init__(self):
        # State: [x, y, vx, vy, theta, omega]
        self.state = np.zeros(6)
        self.state[0] = EARTH_X # Start at Earth
        self.state[1] = 0.0 # Start on ground
        
        # State History for plotting
        self.history = []

    def dynamics(self, state, u):
        x, y, vx, vy, theta, omega = state
        F_thrust, Torque = u
        
        g_local = get_gravity(x, y)

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        ax = -(F_thrust / MASS) * sin_t
        ay = (F_thrust / MASS) * cos_t - g_local
        alpha = Torque / M_INERTIA

        return np.array([vx, vy, ax, ay, omega, alpha])

    def step(self, u):
        """RK4 Integration"""
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * DT * k1, u)
        k3 = self.dynamics(self.state + 0.5 * DT * k2, u)
        k4 = self.dynamics(self.state + DT * k3, u)
        
        self.state += (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ground constraint
        # Only valid if within planet radius (approx flat top constraint)
        # Earth Constraint
        if abs(self.state[0] - EARTH_X) < 50.0: # Pad area
             if self.state[1] < 0:
                self.state[1] = 0
                self.state[3] = max(0, self.state[3])
        
        # Mars Constraint
        if abs(self.state[0] - MARS_X) < 50.0:
             if self.state[1] < 0:
                self.state[1] = 0
                self.state[3] = max(0, self.state[3])
        
        # If in between, no ground constraint (Gap) - Fall to doom if too low?
        # Let's say bottom of screen is limit
        if self.state[1] < -50:
            pass # Fallen into abyss

        # Add process noise
        noise = np.random.normal(0, [0, 0, SIGMA_PROCESS_VEL, SIGMA_PROCESS_VEL, 0, SIGMA_PROCESS_RATE], 6)
        self.state += noise
        
        self.history.append(self.state.copy())
        return self.state

    def get_measurement(self):
        meas = np.array([self.state[0], self.state[1], self.state[4]])
        noise = np.random.normal(0, [SIGMA_MEASURE_POS, SIGMA_MEASURE_POS, SIGMA_MEASURE_ANGLE], 3)
        return meas + noise

# --- Extended Kalman Filter ---
class EKF:
    def __init__(self):
        self.x = np.zeros(6)
        self.x[0] = EARTH_X
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
        F_thrust, Torque = u
        theta = self.x[4]
        g_local = get_gravity(self.x[0], self.x[1]) # Use estimated position
        
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
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 4] = 1
        
        y = z - H @ self.x
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

# --- LQR Controller ---
def compute_lqr_gain(gravity_val):
    # A Matrix around Hover (theta=0, F=mg)
    # x_dot = vx 
    # vx_dot ~ -g * theta
    
    A = np.zeros((6, 6))
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 4] = -gravity_val # Linearization depends on local gravity
    A[3, 4] = 0
    A[4, 5] = 1
    
    B = np.zeros((6, 2))
    B[3, 0] = 1.0 / MASS
    B[5, 1] = 1.0 / M_INERTIA
    
    # Costs
    # State Penalty: [x, y, vx, vy, theta, omega]
    # Relaxed for smoothness: X(20->8), Theta(100->50). 
    # Increase Y cost (5->30) to prevent crashing on Earth.
    Q_cost = np.diag([8.0, 30.0, 5.0, 5.0, 50.0, 20.0]) 
    
    # R Cost: Expensive Torque (RCS) to prevent snapping
    # Thrust 0.05, Torque 0.5 (was 0.01)
    R_cost = np.diag([0.05, 0.5]) 
    
    P_sol = scipy.linalg.solve_continuous_are(A, B, Q_cost, R_cost)
    K = np.linalg.inv(R_cost) @ B.T @ P_sol
    return K

# --- Simulation and Animation ---
def main():
    rocket = Rocket()
    ekf = EKF()

    # Pre-compute Gains for Earth, Mars, and Space (0G)
    K_EARTH = compute_lqr_gain(EARTH_G)
    K_MARS = compute_lqr_gain(MARS_G)
    # Use small fictional gravity for Space LQR to ensure controllability in linearization
    K_SPACE = compute_lqr_gain(1.0)
    
    # Mission State Machine
    # Stages: 
    # 0:Launch, 1:Transit->Mars, 7:Mars Stabilize, 2:Land Mars, 
    # 3:Wait, 4:Launch Mars, 5:Transit->Earth, 8:Earth Stabilize, 6:Land Earth
    mission_stage = 0
    wait_start_time = 0
    
    # Trajectory Smoothing - "Ghost Target"
    # Continuous Sliding Target (not discrete)
    current_target = np.array([EARTH_X, 0.0, 0, 0, 0, 0])
    
    steps = int(SIM_TIME / DT)
    
    true_states = []
    est_states = []
    inputs = []
    targets_hist = []
    
    print("Mission Start: Earth -> Mars -> Earth")
    
    # Internal flags
    landed_mars = False
    return_journey_started = False
    
    for i in range(steps):
        current_time = i * DT
        x_est, y_est = ekf.x[0], ekf.x[1]
        
        # Determine Final Goal for this Stage
        stage_goal = current_target.copy()
        current_stage_speed = LANDING_SPEED
        
        # Default Logic
        if mission_stage == 0: # Launch & Hover Earth
            stage_goal = np.array([EARTH_X, 12.0, 0, 0, 0, 0]) 
            current_stage_speed = 5.0 # Launch slowly
            
            if abs(y_est - 12.0) < 1.0 and abs(ekf.x[3]) < 0.5:
                mission_stage = 1
                print(f"[{current_time:.2f}s] Orbit Reached. Accelerating to Mars.")
                
        elif mission_stage == 1: # Transit to Mars
            stage_goal = np.array([MARS_X, 12.0, 0, 0, 0, 0])
            
            # --- Adaptive Speed Profile ---
            dist_to_mars = abs(x_est - MARS_X)
            
            target_vel = 0.25 * dist_to_mars
            current_stage_speed = np.clip(target_vel, 5.0, TRANSIT_SPEED)
            
            # Widen arrival radius (5.0 -> 20.0) to catch the orbit easier
            if dist_to_mars < 20.0: 
                mission_stage = 7 # Go to Stabilize
                print(f"[{current_time:.2f}s] Arrived at Mars Area. Stabilizing.")

        elif mission_stage == 7: # Mars Stabilize
            stage_goal = np.array([MARS_X, 12.0, 0, 0, 0, 0])
            current_stage_speed = 5.0
            
            # Check for stability (Position and Velocity)
            pos_ok = abs(x_est - MARS_X) < 1.0 and abs(y_est - 12.0) < 1.0
            vel_ok = abs(ekf.x[2]) < 0.5 and abs(ekf.x[3]) < 0.5
            
            if pos_ok and vel_ok:
                mission_stage = 2
                print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 2: # Land Mars
            stage_goal = np.array([MARS_X, 0.0, 0, 0, 0, 0])
            current_stage_speed = 2.0 # Very slow precise landing
            if abs(y_est - 0.0) < 0.2 and abs(ekf.x[3]) < 0.1:
                mission_stage = 3
                wait_start_time = current_time
                print(f"[{current_time:.2f}s] Touchdown Mars. Waiting 5s.")
                
        elif mission_stage == 3: # Wait on Mars
            stage_goal = np.array([MARS_X, 0.0, 0, 0, 0, 0])
            current_stage_speed = 0.0
            if (current_time - wait_start_time) > 5.0:
                mission_stage = 4
                print(f"[{current_time:.2f}s] Launching from Mars.")
                
        elif mission_stage == 4: # Launch Mars
            stage_goal = np.array([MARS_X, 12.0, 0, 0, 0, 0]) # Target 12m
            current_stage_speed = 5.0
            if abs(y_est - 12.0) < 1.0:
                mission_stage = 5
                print(f"[{current_time:.2f}s] Mars Orbit Reached. Accelerating to Earth.")
        
        elif mission_stage == 5: # Transit to Earth
            stage_goal = np.array([EARTH_X, 12.0, 0, 0, 0, 0])
             
            # --- Adaptive Speed Profile ---
            dist_to_earth = abs(x_est - EARTH_X)
            
            target_vel = 0.25 * dist_to_earth
            current_stage_speed = np.clip(target_vel, 5.0, TRANSIT_SPEED)
            
            if dist_to_earth < 20.0:
                mission_stage = 8
                print(f"[{current_time:.2f}s] Earth Orbit Reached. Stabilizing.")

        elif mission_stage == 8: # Earth Stabilize
             stage_goal = np.array([EARTH_X, 12.0, 0, 0, 0, 0])
             current_stage_speed = 5.0
             
             pos_ok = abs(x_est - EARTH_X) < 1.0 and abs(y_est - 12.0) < 1.0
             vel_ok = abs(ekf.x[2]) < 0.5 and abs(ekf.x[3]) < 0.5
             
             if pos_ok and vel_ok:
                 mission_stage = 6
                 print(f"[{current_time:.2f}s] Stabilized. Beginning Descent.")
                
        elif mission_stage == 6: # Land Earth
             stage_goal = np.array([EARTH_X, 0.0, 0, 0, 0, 0])
             current_stage_speed = 2.0
             if abs(y_est - 0.0) < 0.1:
                 # Done
                 pass

        # Smooth Trajectory Update (Continuous Sliding)
        move_speed = current_stage_speed * DT 
        
        # Update X
        dx = stage_goal[0] - current_target[0]
        if abs(dx) > move_speed:
            current_target[0] += np.sign(dx) * move_speed
        else:
            current_target[0] = stage_goal[0]
            
        # Update Y
        dy = stage_goal[1] - current_target[1]
        if abs(dy) > move_speed:
            current_target[1] += np.sign(dy) * move_speed
        else:
            current_target[1] = stage_goal[1]

        targets_hist.append(current_target.copy())

        # 1. Measurement
        z = rocket.get_measurement()
        ekf.update(z)
        
        # 2. Control
        # Select Gravity and Gain based on Estimated Position
        g_current = get_gravity(ekf.x[0], ekf.x[1])
        
        # Select Base Gain
        if g_current < 0.1:
            K_current = K_SPACE
        elif abs(ekf.x[0] - EARTH_X) < abs(ekf.x[0] - MARS_X):
            K_current = K_EARTH
        else:
            K_current = K_MARS
        
        error = ekf.x - current_target 
        u_optimal = -K_current @ error
        
        # Feedforward Gravity Compensation with Tilt Correction
        # Boost thrust when tilted to maintain vertical lift component
        tilt_factor = max(0.2, np.cos(ekf.x[4])) # Avoid div/0
        F_eq = (MASS * g_current) / tilt_factor
        u = u_optimal + np.array([F_eq, 0.0])
        
        # RCS Thrusters: Higher Limits
        u[0] = np.clip(u[0], 0, 50.0) # Main Engine
        u[1] = np.clip(u[1], -30.0, 30.0) # More Powerful RCS for Authority
        
        # 3. Predict & Step
        ekf.predict(u)
        rocket.step(u)
        
        true_states.append(rocket.state.copy())
        est_states.append(ekf.x.copy())
        inputs.append(u)
        
        # Special Logic for "Waiting" state (Cut engines if stable on ground?)
        if mission_stage == 3:
             # Just keep it stable or maybe reduce cost? 
             # LQR is fine, it will hover at y=0.
             pass
        



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
    ax_anim.set_title("Earth-Mars Mission (LQR + EKF) - High Speed Transit")
    
    # Draw Planets and Gravity Fields
    
    # Earth Gravity Field
    g_earth_patch = Circle((EARTH_X, -R_EARTH), R_GRAVITY_EARTH, color='skyblue', alpha=0.15, label='Earth Gravity')
    ax_anim.add_patch(g_earth_patch)

    # Earth Planet
    earth_center = (EARTH_X, -R_EARTH)
    earth_patch = Circle(earth_center, R_EARTH, color='blue', alpha=0.6, label='Earth')
    ax_anim.add_patch(earth_patch)
    
    # Mars Gravity Field
    g_mars_patch = Circle((MARS_X, -R_MARS), R_GRAVITY_MARS, color='skyblue', alpha=0.15, label='Mars Gravity')
    ax_anim.add_patch(g_mars_patch)

    # Mars Planet
    mars_center = (MARS_X, -R_MARS) 
    mars_patch = Circle(mars_center, R_MARS, color='red', alpha=0.6, label='Mars')
    ax_anim.add_patch(mars_patch)
    
    # Objects
    rocket_body = Polygon([[0,0], [0,0], [0,0], [0,0]], color='blue', alpha=0.8, label='Rocket')
    rocket_est = Polygon([[0,0], [0,0], [0,0], [0,0]], color='gray', alpha=0.5, linestyle='--', label='Est')
    flame = Polygon([[0,0], [0,0], [0,0]], color='orange')
    
    # RCS Visuals
    rcs_left_poly = Polygon([[0,0], [0,0], [0,0]], color='white')
    rcs_right_poly = Polygon([[0,0], [0,0], [0,0]], color='white')
    
    target_marker, = ax_anim.plot([], [], 'gx', markersize=10, label='Target')
    
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.add_patch(flame)
    ax_anim.add_patch(rcs_left_poly)
    ax_anim.add_patch(rcs_right_poly)
    ax_anim.legend(loc='upper right')
    
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    
    # Info Plots
    ax_thrust = fig.add_subplot(gs[1, 0])
    line_thrust, = ax_thrust.plot([], [], 'r')
    ax_thrust.set_title("Thrust")
    ax_thrust.set_ylim(0, 55) # Higher limit
    
    ax_x = fig.add_subplot(gs[1, 1])
    line_x, = ax_x.plot([], [], 'b')
    ax_x.set_title("X Position")
    ax_x.set_ylim(-5, 350)

    def animate(i):
        idx = i if i < len(true_states) else len(true_states) - 1
        x, y, vx, vy, theta, omega = true_states[idx]
        xe, ye, vxe, vye, theta_e, omega_e = est_states[idx]
        tx, ty = targets_hist[idx][0], targets_hist[idx][1]
        
        # Rocket Body
        w, h = WIDTH, HEIGHT
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_corners = corners @ R.T + np.array([x, y])
        rocket_body.set_xy(rot_corners)
        
        # Estimate
        ce, se = np.cos(theta_e), np.sin(theta_e)
        Re = np.array([[ce, -se], [se, ce]])
        rot_corners_e = corners @ Re.T + np.array([xe, ye])
        rocket_est.set_xy(rot_corners_e)
        
        # Flame
        thrust = inputs[idx][0]
        torque = inputs[idx][1]
        
        flame_len = thrust / 20.0
        flame_pts = np.array([[-w/4, -h/2], [w/4, -h/2], [0, -h/2 - flame_len]])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        flame.set_xy(rot_flame)
        
        # RCS Visual Update
        # Torque > 0 (CCW) -> Left Thruster fires down? No, Torque CCW means Force on Right Side Up, or Left Side Down.
        # Simple Visual: 
        # +Torque -> Fire Right Thruster (Pushing Left? No)
        # Torque = F*r. +Torque = CounterClockwise. 
        # Top-Right firing Left -> CCW. Top-Left firing Right -> CW.
        # Let's show "puffs" near the top.
        
        rcs_size = 0.3
        rcs_top = h/2 - 0.2
        
        # Left Side Thruster (Fires when needs CW torque? or CCW?)
        # Let's say we have thrusters firing outward.
        # Firing Right Thruster (on right side, pushing Left) -> CCW Torque (+)
        # Firing Left Thruster (on left side, pushing Right) -> CW Torque (-)
        
        if torque < -1.0: # CW -> Fire Left Side
            # Triangle pointing Left
            pts = np.array([[-w/2, rcs_top], [-w/2 - rcs_size, rcs_top], [-w/2, rcs_top - rcs_size]])
            rcs_left_poly.set_xy(pts @ R.T + np.array([x, y]))
            rcs_left_poly.set_visible(True)
        else:
            rcs_left_poly.set_visible(False)
            
        if torque > 1.0: # CCW -> Fire Right Side
            pts = np.array([[w/2, rcs_top], [w/2 + rcs_size, rcs_top], [w/2, rcs_top - rcs_size]])
            rcs_right_poly.set_xy(pts @ R.T + np.array([x, y]))
            rcs_right_poly.set_visible(True)
        else:
            rcs_right_poly.set_visible(False)
            
        
        target_marker.set_data([tx], [ty])
        time_text.set_text(f"Time: {idx*DT:.1f}s")
        
        # Plots
        start = max(0, idx-200)
        times = np.arange(start, idx+1) * DT
        line_thrust.set_data(times, inputs[start:idx+1, 0])
        ax_thrust.set_xlim(times[0], times[-1]+5)
        
        line_x.set_data(times, true_states[start:idx+1, 0])
        ax_x.set_xlim(times[0], times[-1]+5)
        
        return rocket_body, flame, target_marker, time_text, line_thrust, line_x, rcs_left_poly, rcs_right_poly

    ani = animation.FuncAnimation(fig, animate, frames=len(true_states), interval=20, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
