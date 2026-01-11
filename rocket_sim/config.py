"""
Configuration constants for the rocket simulation.
"""
import numpy as np

# --- Time ---
DT = 0.01  # Time step (seconds)

# --- Gravity ---
EARTH_G = 9.81  # m/s^2
MARS_G = 3.73   # m/s^2
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

# --- Planet Visuals ---
R_EARTH = 40.0  # Earth is larger than Mars
R_MARS = 20.0
R_GRAVITY_EARTH = 55.0  # Atmosphere radius
R_GRAVITY_MARS = 30.0

# --- Mission Parameters ---
EARTH_X = 0.0
MARS_X = 300.0
TRANSIT_SPEED = 15.0  # m/s
LANDING_SPEED = 2.0   # m/s
