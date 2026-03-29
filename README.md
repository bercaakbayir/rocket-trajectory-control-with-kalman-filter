# Rocket Trajectory Control with Kalman Filter

This project simulates 2D rocket flight using advanced control theory (LQR) and state estimation (Extended Kalman Filter). It demonstrates how to control an unstable system (a rocket) in the presence of noisy sensor data.

## 1. Kalman Filter (EKF)

The project uses an **Extended Kalman Filter (EKF)** to estimate the true state of the rocket.

### Why do we need it?
In the real world, sensors are noisy. A GPS might say you are at 10.1m, then 9.9m, then 10.2m, even if you are standing still. If we fed this raw noisy data directly into a controller, the rocket engines would jitter violently, wasting fuel and causing instability.

### How it works:
The EKF fuses information from two sources:
1.  **Prediction (Physics Model):** It uses the rocket's equations of motion to predict where the rocket *should* be in the next instant, given the current thrust and torque.
2.  **Update (Measurement):** When a new sensor reading comes in (Position `x, y` and Angle `theta`), it compares it to the prediction.

The filter produces a weighted average of these two, trusting the one with less uncertainty (covariance).

**State Vector:**
The filter estimates 6 variables:
*   `x, y`: Position
*   `vx, vy`: Velocity
*   `theta`: Orientation (Angle)
*   `omega`: Angular Velocity

Even though we only measure position and angle directly, the Kalman Filter can infer (estimate) the hidden states like **velocity** and **angular rate** by observing how position and angle change over time.

### Mathematical Formulation
The EKF uses the following sequence of equations at each time step $k$:

#### 1. Predict Step (Time Update)
We predict the next state using the non-linear physics equations $f(x, u)$ and propogate the error covariance $P$.

$$
\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u_k)
$$

$$
P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k
$$

Where:
*   $\hat{x}_{k|k-1}$ is the predicted state estimate.
*   $f(\cdot)$ is the non-linear rocket dynamics model.
*   $F_k$ is the **Jacobian Matrix** of the dynamics, describing how the state changes with respect to itself ($\partial f / \partial x$).
*   $P$ is the state covariance matrix (uncertainty).
*   $Q$ is the process noise covariance (uncertainty in our model).

#### 2. Update Step (Measurement Update)
We correct the predicted state using the new sensor measurement $z_k$.

$$
\tilde{y}_k = z_k - H \hat{x}_{k|k-1}
$$

$$
S_k = H P_{k|k-1} H^T + R
$$

$$
K_k = P_{k|k-1} H^T S_k^{-1}
$$

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k
$$

$$
P_{k|k} = (I - K_k H) P_{k|k-1}
$$

Where:
*   $z_k$ is the actual measurement from sensors ($x, y, \theta$).
*   $H$ is the measurement matrix (mapping state to sensors).
*   $\tilde{y}_k$ is the measurement residual (error between expected and actual).
*   $K_k$ is the **Kalman Gain**. A high $K$ means we trust the sensor more; a low $K$ means we trust the model more.
*   $R$ is the measurement noise covariance (sensor noise).

---

## 2. Linear Quadratic Regulator (LQR)

**LQR** is an optimal control strategy used to stabilize the rocket.

### How it works:
LQR calculates the best possible engine `Thrust` and `Torque` to bring the rocket's state to zero error (the target) with minimum cost.
It minimizes a cost function `J` that balances:
*   **State Error:** How far we are from the target (Position, Velocity, Angle).
*   **Control Effort:** How much fuel/thrust we use.

The result is a simple gain matrix `K`. The optimal control input `u` is calculated as:
```
u = -K * (Estimated_State - Target_State)
```

### Interaction with Kalman Filter
This project relies on the **Separation Principle**:
1.  **EKF** cleans the noisy sensor data to produce a smooth, accurate estimate of the *Current State*.
2.  **LQR** assumes it has perfect knowledge of the state and calculates the optimal control action based on the EKF's estimate.

Without the Kalman Filter, the LQR controller would react to every bit of sensor noise, destabilizing the rocket.

---

## 3. PID Controller (Proportional-Integral-Derivative)

The **Mars Mission** script uses a robust, cascaded **PID Controller** as an alternative to LQR for complex trajectory tracking.

### Cascaded Architecture
The controller manages three nested loops to ensure stability and precision:
1.  **Altitude Loop (Y):** Controls `Thrust` to maintain a target altitude or vertical velocity.
2.  **Lateral Loop (X):** Computes the required **tilt angle** ($\theta$) to move horizontally towards a target coordinate or maintain a horizontal velocity.
3.  **Attitude Loop ($\theta$):** Controls `Torque` to achieve the target tilt angle with precision and damping.

### Advanced Features
*   **Tilt-Compensated Thrust:** To maintain constant altitude while tilting for horizontal transit, the controller automatically increases thrust magnitude ($T_{req} = T_{base} / \cos(\theta)$), compensating for the loss of vertical lift.
*   **Velocity-Aware Tracking:** The controller minimizes both position and velocity errors simultaneously. This allows the mission logic to command a specific transit speed (e.g., 15 m/s) and perform precise "braking" burns by targeting zero velocity at the destination.
*   **Anti-Windup Logic:** Prevents the integrator from accumulating excessive error during long-duration thrust phases (like launch), ensuring zero overshoot during orbit insertion.

---

## 4. Mission Simulation (`scripts/run_mars_mission.py`)

This script implements a complete **Earth-Mars-Earth** mission using the EKF for state estimation and the Cascaded PID for control.

**Mission Lifecycle:**
*   **Launch from Earth:** Vertical ascent to orbit altitude.
*   **Earth-Mars Transit:** High-speed horizontal flight (15 m/s) with aerodynamic-like tilting and precision braking at destination.
*   **Mars Landing:** Soft descent to the surface with touchdown detection.
*   **Mars Launch:** Return to Mars orbit after a scientific stay.
*   **Mars-Earth Transit:** High-speed return to Earth.
*   **Earth Landing:** Final touchdown at the original launch coordinates.

**Environmental Simulation:**
The physics engine simulates variable gravity fields ($9.81$ m/s² on Earth, $3.73$ m/s² on Mars) and blends them continuously as the rocket transits through space.

---

## 5. Development Testbed (`kalman-test-sim.py`)

A simplified single-stage simulation used to verify the basic EKF and LQR/PID loop stability. 

**Visuals:**
*   **Blue Box:** True physical state of the rocket.
*   **Red Dashed Box:** EKF's real-time estimate.
*   **Graphs:** Live telemetry of thrust, torque, and positional errors.
