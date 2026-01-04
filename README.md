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

## 3. kalman-test-sim.py

This is a simplified testbed simulation designed to verify the control architecture.

**Key Features:**
*   **Single Stage:** The rocket simply tries to hover at a target altitude (10m) and cancel out any initial velocity or displacement.
*   **Visuals:**
    *   **Blue Box:** The true physical rocket.
    *   **Red Dashed Box:** The EKF's *estimate* of where the rocket is. You can see how the estimate converges to the true position.
    *   **Graphs:** Real-time plots of Thrust and Torque inputs.
*   **Code Structure:**
    *   Defined `Rocket` class with non-linear dynamics.
    *   `EKF` class for estimation.
    *   `compute_lqr_gain()` calculates the optimal gains for hovering.
    *   Main loop runs the simulation path: `Dynamics -> Sensors -> EKF -> LQR -> Actuators`.

Run this file to see a basic demonstration of the LQR+EKF loop working to stabilize the rocket.

---

## 4. mars-launcher.py

This is a comprehensive mission simulation that builds upon the testbed.

**Key Features:**
*   **Full Mission Profile:** Simulates a complete Earth-to-Mars-to-Earth journey.
    *   **Launch:** Ascend from Earth.
    *   **Transit:** High-speed transfer to Mars.
    *   **Landing:** Stabilize and land softly on Mars.
    *   **Return:** Launch from Mars and return to Earth.
*   **Variable Gravity:** The physics model blends Earth gravity (9.81 m/s²) and Mars gravity (3.73 m/s²) based on the rocket's position.
*   **Adaptive Control:**
    *   The LQR controller automatically switches its gain matrices (`K_EARTH`, `K_MARS`, `K_SPACE`) depending on the local gravity environment to maintain stability.
*   **Trajectory Smoothing:** Instead of jumping instantly between targets, a "ghost target" slides smoothly along the path, guiding the rocket gently.
*   **Visuals:** Detailed animation showing Earth, Mars, gravity fields, and RCS (Reaction Control System) thruster puffs firing to orient the rocket.
