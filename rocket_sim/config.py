"""
Configuration constants for the rocket simulation.
"""
import numpy as np

# --- Time ---
DT = 0.01  # Time step (seconds)

# --- Gravity ---
EARTH_G = 9.81  # m/s^2
MARS_G = 3.71   # m/s^2 (Standard Mars gravity)
GRAVITY = EARTH_G  # Default for single-planet sims

# --- Rocket Parameters ---
MASS = 1.0  # kg
WIDTH = 0.5  # m
HEIGHT = 2.0  # m
M_INERTIA = (1/12) * MASS * (WIDTH**2 + HEIGHT**2)  # Moment of inertia (kg·m²)
LENGTH_COG_TO_THRUSTER = 1.0  # Distance from CoG to bottom thruster (m)

# --- Process Noise (Standard Deviations) ---
SIGMA_PROCESS_POS = 0.001
SIGMA_PROCESS_VEL = 0.001
SIGMA_PROCESS_ANGLE = 0.0001
SIGMA_PROCESS_RATE = 0.001

# --- Measurement Noise (Standard Deviations) ---
SIGMA_MEASURE_POS = 0.1
SIGMA_MEASURE_ANGLE = 0.01

# --- Planet Visuals (Real Size in meters) ---
R_EARTH = 6371000.0  
R_MARS = 3389500.0
R_GRAVITY_EARTH = R_EARTH + 100000.0  # Atmosphere/Gravity influence radius
R_GRAVITY_MARS = R_MARS + 50000.0

# --- Mission Parameters ---
EARTH_X = 0.0
# ---# Mission scale (Final balanced realism: significant gap + 10x speed accessibility)
MARS_X = 200000000.0 # 200,000 km (Earth is at 0)
TRANSIT_SPEED = 500000.0 # m/s for interplanetary flight
LANDING_SPEED = 10.0 # m/s for descent phase
ORBIT_HEIGHT = 5000000.0 # 5,000 km target orbit
MAX_THRUST = 40.0 # Increased for snappier response at scale
