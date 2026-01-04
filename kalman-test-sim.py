import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
import scipy.linalg

# --- Constants ---
DT = 0.01  # Time step (Reduced for stability)
SIM_TIME = 20.0  # Total simulation time
GRAVITY = 9.81

# Rocket Parameters
MASS = 1.0  # kg
WIDTH = 0.5 # m
HEIGHT = 2.0 # m
M_INERTIA = (1/12) * MASS * (WIDTH**2 + HEIGHT**2)  # Moment of inertia
LENGTH_COG_TO_THRUSTER = 1.0 # Distance from CoG to bottom thruster

# Noise Parameters (Standard Deviations)
SIGMA_PROCESS_POS = 0.001
SIGMA_PROCESS_VEL = 0.001
SIGMA_PROCESS_ANGLE = 0.0001
SIGMA_PROCESS_RATE = 0.001
SIGMA_MEASURE_POS = 0.1  # Reduced from 0.5
SIGMA_MEASURE_ANGLE = 0.01 # Reduced from 0.05

# --- Rocket Dynamics ---
class Rocket:
    def __init__(self):
        # State: [x, y, vx, vy, theta, omega]
        self.state = np.zeros(6)
        self.state[1] = 0.0 # Start on ground
        
        # State History for plotting
        self.history = []

    def dynamics(self, state, u):
        """
        Nonlinear dynamics of the rocket.
        state: [x, y, vx, vy, theta, omega]
        u: [Thrust, Torque]
        """
        x, y, vx, vy, theta, omega = state
        F_thrust, Torque = u

        # Equations of motion
        # Fx = F * sin(theta) (Assuming theta=0 is UP)
        # Fy = F * cos(theta) - mg
        
        # Careful with coordinate system:
        # Standard math: 0 is Right (X-axis).
        # Rocket sim: 0 usually Up (Y-axis).
        # Let's align with Standard Math representation for simplicity in sin/cos?
        # No, let's Stick to visual intuition: Theta=0 is UP (Y-axis).
        # x-axis is Right.
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        ax = (F_thrust / MASS) * (-sin_t) # tilts left -> +x? No. positive theta (CCW) -> tilts left -> F vector points Left.
        # Wait. If 0 is UP. +Theta is CCW (Left tilt). Rocket top points Left. 
        # Then Thrust vector points Up-Left. 
        # Force X component is -sin(theta) * F
        # Force Y component is cos(theta) * F
        
        ax = -(F_thrust / MASS) * sin_t
        ay = (F_thrust / MASS) * cos_t - GRAVITY
        alpha = Torque / M_INERTIA

        return np.array([vx, vy, ax, ay, omega, alpha])

    def step(self, u):
        """RK4 Integration"""
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * DT * k1, u)
        k3 = self.dynamics(self.state + 0.5 * DT * k2, u)
        k4 = self.dynamics(self.state + DT * k3, u)
        
        self.state += (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ground constraint (simple bounce or stick)
        if self.state[1] < 0:
            self.state[1] = 0
            self.state[3] = max(0, self.state[3]) # No downward velocity

        # Add process noise
        noise = np.random.normal(0, [0, 0, SIGMA_PROCESS_VEL, SIGMA_PROCESS_VEL, 0, SIGMA_PROCESS_RATE], 6)
        self.state += noise
        
        self.history.append(self.state.copy())
        return self.state

    def get_measurement(self):
        """Returns noisy measurement [x, y, theta]"""
        # We measure Position and Orientation
        # H matrix extracts [x, y, theta]
        meas = np.array([self.state[0], self.state[1], self.state[4]])
        noise = np.random.normal(0, [SIGMA_MEASURE_POS, SIGMA_MEASURE_POS, SIGMA_MEASURE_ANGLE], 3)
        return meas + noise

# --- Extended Kalman Filter ---
class EKF:
    def __init__(self):
        self.x = np.zeros(6) # Estimate
        self.P = np.eye(6) * 0.1 # Covariance
        
        # Process Noise Covariance Q
        # Assumes noise enters through acceleration/velocity
        self.Q = np.diag([
            SIGMA_PROCESS_POS**2, SIGMA_PROCESS_POS**2,
            SIGMA_PROCESS_VEL**2, SIGMA_PROCESS_VEL**2,
            SIGMA_PROCESS_ANGLE**2, SIGMA_PROCESS_RATE**2
        ])
        
        # Measurement Noise Covariance R
        self.R = np.diag([
            SIGMA_MEASURE_POS**2, SIGMA_MEASURE_POS**2, SIGMA_MEASURE_ANGLE**2
        ])

    def predict(self, u):
        """
        Predict step using Jacobians directly or linearization
        x_pred = f(x, u)
        P_pred = F * P * F.T + Q
        """
        # 1. State Prediction (Nonlinear)
        # We can reuse the Rocket dynamics logic or simple Euler for estimation
        F_thrust, Torque = u
        theta = self.x[4]
        
        # Simple Euler for prediction step
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        acc_x = -(F_thrust / MASS) * sin_t
        acc_y = (F_thrust / MASS) * cos_t - GRAVITY
        alpha = Torque / M_INERTIA
        
        # x, y, vx, vy, theta, omega
        # x_new = x + vx * dt
        # vx_new = vx + ax * dt ...
        
        # Explicit update
        self.x[0] += self.x[2] * DT
        self.x[1] += self.x[3] * DT
        self.x[2] += acc_x * DT
        self.x[3] += acc_y * DT
        self.x[4] += self.x[5] * DT
        self.x[5] += alpha * DT
        
        # 2. Jacobian F computation (df/dx)
        # State: x, y, vx, vy, theta, omega
        # dx_dot/d... -> 1 at vx
        # dvx_dot/dtheta -> (-F/m) * cos(theta)
        
        F_jac = np.eye(6)
        F_jac[0, 2] = DT
        F_jac[1, 3] = DT
        F_jac[4, 5] = DT
        
        # Partial derivatives of velocity/accel terms
        # d(vx_new)/d(theta) = d(vx + ax*dt)/dtheta = dt * d(ax)/dtheta
        # ax = -F/m * sin(theta) -> dax/dtheta = -F/m * cos(theta)
        d_vx_dtheta = DT * (-(F_thrust / MASS) * cos_t)
        
        # d(vy_new)/d(theta) = dt * d(ay)/dtheta
        # ay = F/m * cos(theta) - g -> day/dtheta = -F/m * sin(theta)
        d_vy_dtheta = DT * (-(F_thrust / MASS) * sin_t)
        
        F_jac[2, 4] = d_vx_dtheta
        F_jac[3, 4] = d_vy_dtheta
        
        self.P = F_jac @ self.P @ F_jac.T + self.Q

    def update(self, z):
        """
        Update step
        z: Measurement [x, y, theta]
        """
        # Measurement Function H: Linear extraction of [x, y, theta]
        # z = Hx
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 4] = 1
        
        # Residue
        y = z - H @ self.x
        # Wrap angle residual if necessary (not needed for small angles but good practice)
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

# --- LQR Controller ---
def compute_lqr_gain():
    """
    Linearize system around Hover state and compute K.
    Hover state: x=target, y=target, vx=0, vy=0, theta=0, omega=0
    Input: F = mg, Tau = 0
    """
    # State: [x, y, vx, vy, theta, omega]
    # Input: [F, Tau]
    
    # A Matrix (df/dx)
    # x_dot = vx ...
    # vx_dot = -F/m sin(theta) -> d/dtheta = -F/m cos(0) = -g (since F=mg)
    # vy_dot = F/m cos(theta) - g -> d/dtheta = -F/m sin(0) = 0
    # Note: d(vy_dot)/dF = 1/m * cos(0) = 1/m
    
    A = np.zeros((6, 6))
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 4] = -GRAVITY # From linearization of -F/m sin(theta) with F=mg, theta=0
    A[3, 4] = 0        # From linearization of F/m cos(theta) - g -> -F/m sin(0) = 0
    A[4, 5] = 1
    
    # B Matrix (df/du)
    # u = [F, Tau]
    # vx_dot dependence on F: -1/m sin(0) = 0
    # vy_dot dependence on F: 1/m cos(0) = 1/m
    # omega_dot dependence on Tau: 1/I
    
    B = np.zeros((6, 2))
    B[3, 0] = 1.0 / MASS
    B[5, 1] = 1.0 / M_INERTIA
    
    # Costs
    # State Penalty: [x, y, vx, vy, theta, omega]
    # Tuning for Damping:
    # - Increase vx, vy penalties (Damping) to prevent overshoot.
    # - Increase omega penalty to prevent angular oscillation.
    Q_cost = np.diag([2.0, 5.0, 5.0, 5.0, 50.0, 5.0]) 
    R_cost = np.diag([0.01, 1.0]) # Penalize Torque usage more
    
    # Solve Riccati
    P_sol = scipy.linalg.solve_continuous_are(A, B, Q_cost, R_cost)
    K = np.linalg.inv(R_cost) @ B.T @ P_sol
    return K

# --- Simulation and Animation ---
def main():
    rocket = Rocket()
    ekf = EKF()
    K = compute_lqr_gain()
    
    # Target State (Hover at 10m)
    x_target = np.array([0.0, 10.0, 0, 0, 0, 0])
    
    # Histories
    true_states = []
    est_states = []
    measurements = []
    inputs = []
    
    steps = int(SIM_TIME / DT)
    
    # Initial Push (optional, to make it work for it)
    rocket.state[0] = 5.0 # Start off-center
    ekf.x[0] = 5.0
    
    for i in range(steps):
        # 1. State Estimation (EKF)
        # Get sensor data
        z = rocket.get_measurement()
        # EKF Update
        ekf.update(z)
        
        # 2. Control (LQR)
        state_est = ekf.x
        error = state_est - x_target
        u_optimal = -K @ error
        
        # Add Equilibrium input
        F_eq = MASS * GRAVITY
        u = u_optimal + np.array([F_eq, 0.0])
        
        # Input Saturation (Realism)
        u[0] = np.clip(u[0], 0, 3 * MASS * GRAVITY) # Thrust limits
        u[1] = np.clip(u[1], -5.0, 5.0)       # Torque limits
        
        # 3. Predict Step (EKF)
        ekf.predict(u)
        
        # 4. Simulation Step
        rocket.step(u)
        
        # Store data
        true_states.append(rocket.state.copy())
        est_states.append(ekf.x.copy())
        measurements.append(z)
        inputs.append(u)

    # Convert to arrays
    true_states = np.array(true_states)
    est_states = np.array(est_states)
    measurements = np.array(measurements)
    inputs = np.array(inputs)
    
    # --- Animation ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Main Animation Axis
    ax_anim = fig.add_subplot(gs[0, :])
    ax_anim.set_xlim(-15, 15)
    ax_anim.set_ylim(-2, 18)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title("Rocket LQR Hover with EKF Estimation")
    
    # Inputs Axis
    ax_input = fig.add_subplot(gs[1, 0])
    ax_input.set_title("Thrust (N)")
    ax_input.set_ylim(0, 30)
    line_thrust, = ax_input.plot([], [], 'r-')
    
    ax_torque = fig.add_subplot(gs[1, 1])
    ax_torque.set_title("Torque (Nm)")
    ax_torque.set_ylim(-6, 6)
    line_torque, = ax_torque.plot([], [], 'b-')

    # Visual Objects
    rocket_body = Rectangle((-WIDTH/2, 0), WIDTH, HEIGHT, color='blue', alpha=0.8)
    rocket_est = Rectangle((-WIDTH/2, 0), WIDTH, HEIGHT, color='red', alpha=0.3, linestyle='--')
    
    flame = Polygon([[0,0], [0,0], [0,0]], color='orange')
    
    target_marker = ax_anim.plot(0, 10, 'gx', markersize=10, label='Target')[0]
    
    # Add patches
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.add_patch(flame)
    
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    
    def animate(i):
        # Index
        idx = i if i < len(true_states) else len(true_states) - 1
        
        # True State
        x, y, vx, vy, theta, omega = true_states[idx]
        
        # Est State
        xe, ye, vxe, vye, theta_e, omega_e = est_states[idx]
        
        # Update Rocket Body (True)
        # Transform rectangle: Translate + Rotate
        t_start = ax_anim.transData
        coords = np.array([[-WIDTH/2, -HEIGHT/2], [WIDTH/2, -HEIGHT/2], [WIDTH/2, HEIGHT/2], [-WIDTH/2, HEIGHT/2]])
        
        # Rotation Matrix (visual rotation is negative of math analysis usually, but let's check)
        # Here theta=0 is UP. Rectangle definition is axis aligned.
        # We need to rotate coordinates by -theta (CCW is positive in our math)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Center of rocket is (x, y). 
        # Our rectangle patch is defined from bottom-left corner relative to anchor.
        # Easier to just calculate polygon points manually.
        
        # Rocket Body Points (Centered at CoG approx)
        # Let's say CoG is in middle of rect
        w, h = WIDTH, HEIGHT
        corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])
        
        # Dynamic Rotation
        rotated_corners = corners @ R.T
        
        # Translation
        final_corners = rotated_corners + np.array([x, y])
        rocket_body.set_xy(final_corners) # Error: Rectangle doesn't support generic xy points like Polygon. 
        # Wait, Rectangle is axis aligned unless transformed.
        # Let's switch rocket_body to Polygon for easy rotation.
        
        # Flame
        thrust = inputs[idx][0]
        flame_len = thrust / 20.0 # Scale
        flame_pts = np.array([
            [-w/4, -h/2],
            [w/4, -h/2],
            [0, -h/2 - flame_len]
        ])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        flame.set_xy(rot_flame)
        
        # Estimated Ghost
        ce, se = np.cos(theta_e), np.sin(theta_e)
        Re = np.array([[ce, -se], [se, ce]])
        rot_corners_est = corners @ Re.T + np.array([xe, ye])
        # Need to re-init rocket_est as Polygon in init if I do this.
        
        time_text.set_text(f"Time: {idx*DT:.1f}s")
        
        # Update Plots
        start_plot = max(0, idx - 100)
        times = np.arange(start_plot, idx+1) * DT
        
        line_thrust.set_data(times, inputs[start_plot:idx+1, 0])
        ax_input.set_xlim(times[0], times[-1] + 1)
        
        line_torque.set_data(times, inputs[start_plot:idx+1, 1])
        ax_torque.set_xlim(times[0], times[-1] + 1)
        
        return rocket_body, flame, time_text, line_thrust, line_torque

    # Fixup objects for animation function
    rocket_body.remove()
    rocket_est.remove()
    rocket_body = Polygon([[0,0], [0,0], [0,0], [0,0]], color='blue', alpha=0.8, label='True')
    rocket_est = Polygon([[0,0], [0,0], [0,0], [0,0]], color='red', alpha=0.3, linestyle='--', label='EKF Est')
    ax_anim.add_patch(rocket_body)
    ax_anim.add_patch(rocket_est)
    ax_anim.legend()

    ani = animation.FuncAnimation(fig, animate, frames=len(true_states), interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
