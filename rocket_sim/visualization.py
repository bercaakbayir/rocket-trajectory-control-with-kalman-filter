"""
Visualization helpers for rocket simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle

from .config import WIDTH, HEIGHT, DT, R_EARTH, R_GRAVITY_EARTH


class RocketVisualizer:
    """
    Handles animation setup and updates for rocket simulation.
    """
    
    def __init__(self, ax, show_earth=True, show_rcs=True):
        """
        Initialize visualizer on given axes.
        
        Args:
            ax: Matplotlib axes for animation
            show_earth: Whether to show Earth/ground patches
            show_rcs: Whether to show RCS thruster polygons
        """
        self.ax = ax
        self.show_rcs = show_rcs
        
        # Earth visuals
        if show_earth:
            g_patch = Circle((0, -R_EARTH), R_GRAVITY_EARTH, color='skyblue', alpha=0.15)
            ax.add_patch(g_patch)
            earth_patch = Circle((0, -R_EARTH), R_EARTH, color='blue', alpha=0.6)
            ax.add_patch(earth_patch)
        
        # Rocket body (will be replaced with Polygon)
        self.rocket_body = Polygon([[0,0]]*4, color='blue', alpha=0.8, label='Rocket')
        self.rocket_est = Polygon([[0,0]]*4, color='red', alpha=0.3, linestyle='--', label='EKF Est')
        ax.add_patch(self.rocket_body)
        ax.add_patch(self.rocket_est)
        
        # Flame
        self.flame = Polygon([[0,0]]*3, color='orange')
        ax.add_patch(self.flame)
        
        # RCS
        if show_rcs:
            self.rcs_left = Polygon([[0,0]]*3, color='gray')
            self.rcs_right = Polygon([[0,0]]*3, color='gray')
            ax.add_patch(self.rcs_left)
            ax.add_patch(self.rcs_right)
        
        # Time text
        self.time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
    def update(self, true_state, est_state, inputs, time):
        """
        Update all visual elements.
        
        Args:
            true_state: True rocket state [x, y, vx, vy, theta, omega]
            est_state: Estimated rocket state
            inputs: Control inputs [Thrust, Torque]
            time: Current simulation time
        """
        x, y, vx, vy, theta, omega = true_state
        xe, ye, vxe, vye, theta_e, omega_e = est_state
        thrust, torque = inputs
        
        w, h = WIDTH, HEIGHT
        
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rocket body corners
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        rot_corners = corners @ R.T + np.array([x, y])
        self.rocket_body.set_xy(rot_corners)
        
        # Estimated body
        ce, se = np.cos(theta_e), np.sin(theta_e)
        Re = np.array([[ce, -se], [se, ce]])
        rot_corners_e = corners @ Re.T + np.array([xe, ye])
        self.rocket_est.set_xy(rot_corners_e)
        
        # Flame
        flame_len = thrust / 20.0
        flame_pts = np.array([[-w/4, -h/2], [w/4, -h/2], [0, -h/2 - flame_len]])
        rot_flame = flame_pts @ R.T + np.array([x, y])
        self.flame.set_xy(rot_flame)
        
        # RCS
        if self.show_rcs:
            rcs_size = 0.3
            rcs_top = h/2 - 0.2
            
            if torque < -0.5:
                pts = np.array([[w/2, rcs_top], [w/2 + rcs_size, rcs_top], [w/2, rcs_top - rcs_size]])
                self.rcs_right.set_xy(pts @ R.T + np.array([x, y]))
                self.rcs_right.set_visible(True)
            else:
                self.rcs_right.set_visible(False)
                
            if torque > 0.5:
                pts = np.array([[-w/2, rcs_top], [-w/2 - rcs_size, rcs_top], [-w/2, rcs_top - rcs_size]])
                self.rcs_left.set_xy(pts @ R.T + np.array([x, y]))
                self.rcs_left.set_visible(True)
            else:
                self.rcs_left.set_visible(False)
        
        self.time_text.set_text(f"Time: {time:.1f}s")


def create_input_plots(fig, gs):
    """Create thrust and torque subplots."""
    ax_thrust = fig.add_subplot(gs[1, 0])
    ax_thrust.set_title("Thrust (N)")
    ax_thrust.set_ylim(0, 35)
    line_thrust, = ax_thrust.plot([], [], 'r-')
    
    ax_torque = fig.add_subplot(gs[1, 1])
    ax_torque.set_title("Torque (Nm)")
    ax_torque.set_ylim(-6, 6)
    line_torque, = ax_torque.plot([], [], 'b-')
    
    return ax_thrust, ax_torque, line_thrust, line_torque
