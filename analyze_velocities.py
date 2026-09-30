#!/usr/bin/env python3
"""
Script for comparing estimated velocities from CSV with computed velocities from poses.

Usage:
    python3 analyze_velocities.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class PoseAnalyzer:
    def __init__(self, estimate_file, velocity_csv_file, gt_file=None):
        """
        Initialize analyzer with pose file and velocity CSV.
        
        Args:
            estimate_file: Path to estimate.txt (12 values per line - 3x4 transformation matrix)
            velocity_csv_file: Path to estimated_velocities.csv
            gt_file: Path to gt_kitti_stairs.txt (ground truth poses)
        """
        self.estimate_file = Path(estimate_file)
        self.velocity_csv_file = Path(velocity_csv_file)
        self.gt_file = Path(gt_file) if gt_file else None
        self.poses = []
        self.gt_poses = []
        self.estimated_velocities = None
        self.computed_velocities = []
        self.gt_velocities = []
        
    def load_poses(self):                                                          # file delle pose dell'output di mad-icp
        """Load poses from estimate.txt file."""
        print(f"Loading poses from {self.estimate_file}...")
        
        with open(self.estimate_file, 'r') as f:
            for line in f:
                values = np.array([float(x) for x in line.split()])
                # Reshape 12 values to 3x4 transformation matrix
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
                self.poses.append(T)
        
        print(f"Loaded {len(self.poses)} poses")
        return self.poses
    
    def load_estimated_velocities(self):                                    # file delle velocità stimate dal vel_estimator
        """Load estimated velocities from CSV."""
        print(f"Loading estimated velocities from {self.velocity_csv_file}...")
        self.estimated_velocities = pd.read_csv(self.velocity_csv_file)
        print(f"Loaded {len(self.estimated_velocities)} velocity estimates")
        return self.estimated_velocities
    
    def load_gt_poses(self):                                                                             # legge il file GT
        """Load ground truth poses from gt_kitti_stairs.txt file."""
        if self.gt_file is None or not self.gt_file.exists():
            print("Ground truth file not found or not specified")
            return []
        
        print(f"Loading GT poses from {self.gt_file}...")
        
        with open(self.gt_file, 'r') as f:
            for line in f:
                values = np.array([float(x) for x in line.split()])
                # Reshape 12 values to 3x4 transformation matrix
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
                self.gt_poses.append(T)
        
        print(f"Loaded {len(self.gt_poses)} GT poses")
        return self.gt_poses
    
    def rotation_to_euler(self, R):
        """
        Extract Euler angles (roll, pitch, yaw) from rotation matrix.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            (roll, pitch, yaw) in radians
        """
        # Roll
        roll = np.arctan2(-R[1, 2], R[2, 2])
        # Pitch
        pitch = np.arcsin(R[0, 2])
        # Yaw
        yaw = np.arctan2(-R[0, 1], R[0, 0])
        
        return roll, pitch, yaw
    
    def compute_angular_velocity_from_matrix_log(self, R_rel, dt):
        """
        Compute angular velocity using matrix logarithm (SO(3) geometry).
        More robust than Euler angles - recommended by thesis supervisor.
        
        Args:
            R_rel: 3x3 relative rotation matrix
            dt: time step in seconds
            
        Returns:
            omega: angular velocity vector [ωx, ωy, ωz]
            
        Formula: omega = log(R_rel) / dt
        """
        # Trace of rotation matrix
        trace = np.trace(R_rel)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))  # Clip to avoid numerical errors
        
        # If rotation is very small (identity), return zero angular velocity
        if theta < 1e-6:
            return np.zeros(3)
        
        # Extract skew-symmetric matrix from SO(3) logarithm
        K = (R_rel - R_rel.T) / (2 * np.sin(theta))
        
        # Extract omega vector from skew-symmetric matrix
        # For skew-symmetric K = [0, -w_z, w_y]
        #                       [w_z, 0, -w_x]
        #                       [-w_y, w_x, 0]
        omega = np.array([K[2, 1], K[0, 2], K[1, 0]]) * theta / dt
        
        return omega
    
    def compute_velocities(self):
        """
        Compute velocities from consecutive poses.
        
        Assumes constant velocity model: v = Δt / Δt_seconds
        """
        print("\nComputing velocities from consecutive poses...")
        
        # Assume constant timestep based on CSV if available
        dt = 0.1  # Default 10 Hz
        
        if self.estimated_velocities is not None:
            # Try to get timestep from CSV if timestamp column exists
            if 'timestamp' in self.estimated_velocities.columns:
                timestamps = self.estimated_velocities['timestamp'].values
                if len(timestamps) > 1:
                    dt = np.mean(np.diff(timestamps))
                    print(f"Extracted timestep from CSV: Δt = {dt:.4f} seconds (≈ {1/dt:.1f} Hz)")
            else:
                print(f"No timestamp column in CSV, using default: Δt = {dt:.4f} seconds (≈ {1/dt:.1f} Hz)")
        else:
            print(f"Using default timestep Δt = {dt:.4f} seconds (≈ {1/dt:.1f} Hz)")
        
        for i in range(len(self.poses) - 1):
            T_prev = self.poses[i]
            T_now = self.poses[i + 1]
            
            # Relative transformation: T_rel = T_prev^-1 * T_now
            T_rel = np.linalg.inv(T_prev) @ T_now
            
            # Extract translation
            t_rel = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            
            # Linear velocity
            v_lin = t_rel / dt
            
            # Angular velocity using Matrix Logarithm (SO(3) geometry) - MORE ROBUST
            w_ang = self.compute_angular_velocity_from_matrix_log(R_rel, dt)
            
            # OLD METHOD - Commented out: Angular velocity from Euler angles (has gimbal lock issues)
            # roll, pitch, yaw = self.rotation_to_euler(R_rel)
            # w_ang = np.array([roll, pitch, yaw]) / dt
            
            vel_dict = {
                'frame_id': i + 1,  # +1 to align with CSV (frame of arrival, not departure)
                'vx': v_lin[0],
                'vy': v_lin[1],
                'vz': v_lin[2],
                'ωx': w_ang[0],
                'ωy': w_ang[1],
                'ωz': w_ang[2]
            }
            
            self.computed_velocities.append(vel_dict)
        
        print(f"Computed {len(self.computed_velocities)} velocity pairs")
        return pd.DataFrame(self.computed_velocities)
    
    def compute_gt_velocities(self):
        """
        Compute ground truth velocities from consecutive GT poses.
        Uses same method as compute_velocities().
        """
        if len(self.gt_poses) == 0:
            print("No GT poses loaded")
            return pd.DataFrame()
        
        print("\nComputing GT velocities from consecutive poses...")
        
        dt = 0.1  # Default 10 Hz
        
        if self.estimated_velocities is not None:
            if 'timestamp' in self.estimated_velocities.columns:
                timestamps = self.estimated_velocities['timestamp'].values
                if len(timestamps) > 1:
                    dt = np.mean(np.diff(timestamps))
                    print(f"Using timestep from CSV: Δt = {dt:.4f} seconds (≈ {1/dt:.1f} Hz)")
        
        for i in range(len(self.gt_poses) - 1):
            T_prev = self.gt_poses[i]
            T_now = self.gt_poses[i + 1]
            
            # Relative transformation: T_rel = T_prev^-1 * T_now
            T_rel = np.linalg.inv(T_prev) @ T_now
            
            # Extract translation
            t_rel = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            
            # Linear velocity
            v_lin = t_rel / dt
            
            # Angular velocity using Matrix Logarithm (SO(3) geometry) - MORE ROBUST
            w_ang = self.compute_angular_velocity_from_matrix_log(R_rel, dt)
            
            # OLD METHOD - Commented out: Angular velocity from Euler angles (has gimbal lock issues)
            # roll, pitch, yaw = self.rotation_to_euler(R_rel)
            # w_ang = np.array([roll, pitch, yaw]) / dt
            
            vel_dict = {
                'frame_id': i + 1,
                'vx': v_lin[0],
                'vy': v_lin[1],
                'vz': v_lin[2],
                'ωx': w_ang[0],
                'ωy': w_ang[1],
                'ωz': w_ang[2]
            }
            
            self.gt_velocities.append(vel_dict)
        
        print(f"Computed {len(self.gt_velocities)} GT velocity pairs")
        return pd.DataFrame(self.gt_velocities)
    
    def compare_velocities(self):
        """Compare estimated vs computed velocities."""
        print("\n" + "="*80)
        print("VELOCITY COMPARISON ANALYSIS")
        print("="*80)
        
        computed_df = pd.DataFrame(self.computed_velocities)
        
        # DEBUG: Check alignment of first 5 frames
        print("\n" + "-"*80)
        print("ALIGNMENT CHECK (First 10 frames):")
        print("-"*80)
        print(f"{'CSV frame_id':<15} {'CSV vx':<15} {'Script frame_id':<15} {'Script vx':<15} {'Match?':<15}")
        print("-"*80)
        
        estimated_df_temp = self.estimated_velocities.copy()
        for idx in range(min(10, len(estimated_df_temp), len(computed_df))):
            csv_frame_id = estimated_df_temp.iloc[idx]['frame_id']
            csv_vx = estimated_df_temp.iloc[idx]['vx']
            script_frame_id = computed_df.iloc[idx]['frame_id']
            script_vx = computed_df.iloc[idx]['vx']
            match = "✓" if csv_frame_id == script_frame_id else "✗ OFFSET!"
            
            print(f"{csv_frame_id:<15} {csv_vx:<15.6f} {script_frame_id:<15} {script_vx:<15.6f} {match:<15}")
        
        print("-"*80 + "\n")
        
        # Align by frame_id (drop header row from CSV if needed)
        estimated_df = self.estimated_velocities.copy()
        
        # Get common frame IDs
        common_frames = np.intersect1d(
            estimated_df['frame_id'].values,
            computed_df['frame_id'].values
        )
        
        print(f"\nCommon frames for comparison: {len(common_frames)}")
        
        # Extract data for common frames
        est_data = estimated_df[estimated_df['frame_id'].isin(common_frames)].reset_index(drop=True)
        comp_data = computed_df[computed_df['frame_id'].isin(common_frames)].reset_index(drop=True)
        
        if len(est_data) != len(comp_data):
            print(f"Warning: Dataset size mismatch ({len(est_data)} vs {len(comp_data)})")
        
        # Calculate differences
        vel_components = ['vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz']
        
        print("\n" + "-"*80)
        print("LINEAR VELOCITIES (m/s):")
        print("-"*80)
        print(f"{'Component':<15} {'Estimated':<15} {'Computed':<15} {'Difference':<15} {'Error %':<15}")
        print("-"*80)
        
        for comp in ['vx', 'vy', 'vz']:
            est_mean = est_data[comp].mean()
            comp_mean = comp_data[comp].mean()
            diff = est_mean - comp_mean
            error_pct = (diff / (comp_mean + 1e-10)) * 100  # Avoid division by zero
            
            print(f"{comp:<15} {est_mean:<15.6f} {comp_mean:<15.6f} {diff:<15.6f} {error_pct:<15.2f}%")
        
        print("\n" + "-"*80)
        print("ANGULAR VELOCITIES (rad/s):")
        print("-"*80)
        print(f"{'Component':<15} {'Estimated':<15} {'Computed':<15} {'Difference':<15} {'Error %':<15}")
        print("-"*80)
        
        for comp in ['ωx', 'ωy', 'ωz']:
            est_mean = est_data[comp].mean()
            comp_mean = comp_data[comp].mean()
            diff = est_mean - comp_mean
            error_pct = (diff / (comp_mean + 1e-10)) * 100  # Avoid division by zero
            
            print(f"{comp:<15} {est_mean:<15.6f} {comp_mean:<15.6f} {diff:<15.6f} {error_pct:<15.2f}%")
        
        # Calculate RMS errors
        print("\n" + "-"*80)
        print("ROOT MEAN SQUARE (RMS) ERRORS:")
        print("-"*80)
        
        for comp in vel_components:
            rms_error = np.sqrt(np.mean((est_data[comp].values - comp_data[comp].values)**2))
            print(f"{comp:<15}: {rms_error:<15.6e}")
        
        return est_data, comp_data
    
    def save_comparison_csv(self, estimated_df, computed_df, gt_df=None, timestamp=None):
        """Save detailed comparison to CSV file."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.velocity_csv_file).parent / f"velocity_comparison_{timestamp}.csv"
        
        print(f"\nSaving detailed comparison to {output_file}...")
        
        vel_components = ['vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz']
        
        comparison_data = []
        for idx in range(len(estimated_df)):
            row = {'frame_id': estimated_df.iloc[idx]['frame_id']}
            
            for comp in vel_components:
                row[f'{comp}_velestimator'] = estimated_df.iloc[idx][comp]
                row[f'{comp}_computed'] = computed_df.iloc[idx][comp]
                # row[f'{comp}_difference'] = estimated_df.iloc[idx][comp] - computed_df.iloc[idx][comp]
                if gt_df is not None and idx < len(gt_df):
                    row[f'{comp}_gt'] = gt_df.iloc[idx][comp]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        return comparison_df
    
    def plot_comparison(self, estimated_df, computed_df, timestamp=None):
        """Create visualization plots."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(6, 1, figsize=(30, 25))
        # fig.suptitle('Velocity Comparison: Estimated vs Computed', fontsize=16)
        
        vel_components = ['vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz']
        
        for idx, (ax, comp) in enumerate(zip(axes.flat, vel_components)):
            frames = range(len(estimated_df))
            
            # Plot computed first (background), then estimated (foreground)
            #ax.plot(frames, computed_df[comp].values, 'r--', label='Computed from odometry output', linewidth=1.5, alpha=0.7)
            ax.plot(frames, computed_df[comp].values, 'r-', label='Computed from odometry output', linewidth=2)
            #ax.plot(frames, estimated_df[comp].values, 'b-', label='Estimated from MAD-ICP algorithm', linewidth=2)
            
            ax.set_xlabel('Frame ID')
            if comp.startswith('v'):
                ax.set_ylabel(f'{comp} (m/s)')
            else:
                ax.set_ylabel(f'{comp} (rad/s)')
            
            ax.set_title(f'{comp} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_plot = Path(self.velocity_csv_file).parent / f"velocity_comparison_{timestamp}.png"
        plt.savefig(output_plot, dpi=150)
        print(f"Saved plot to {output_plot}")
        plt.close()
    
    def plot_gt_vs_computed(self, gt_df, computed_df, timestamp=None, limit=None):         # plot GT vs Computed Velocities
        """Create visualization plots comparing GT vs Computed velocities.
    
        Args:
            limit: Maximum number of frames to plot (None = plot all)
        """
        if gt_df is None or len(gt_df) == 0:
            print("GT data not available for plotting")
            return
    
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\nGenerating GT vs Computed plots...")
    
        fig, axes = plt.subplots(6, 1, figsize=(30, 25))                                    # per modificare i riquadri
        fig.suptitle('Velocity Comparison: GT vs Computed', fontsize=16)
    
        vel_components = ['vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz']
    
        # Align data by frame_id for GT vs Computed
        common_frames = np.intersect1d(
            gt_df['frame_id'].values,
            computed_df['frame_id'].values
        )
    
        gt_aligned = gt_df[gt_df['frame_id'].isin(common_frames)].reset_index(drop=True)
        comp_aligned = computed_df[computed_df['frame_id'].isin(common_frames)].reset_index(drop=True)
    
        # Limit to first X rows if specified
        if limit is not None:
            gt_aligned = gt_aligned.head(limit)
            comp_aligned = comp_aligned.head(limit)
    
        for idx, (ax, comp) in enumerate(zip(axes.flat, vel_components)):
            frames = range(len(gt_aligned))
            
            # Plot computed first (background), then GT (foreground) ------------------------------------------------------
            ax.plot(frames, gt_aligned[comp].values, 'b-', label='GT', linewidth=2)  
            ax.plot(frames, comp_aligned[comp].values, 'r--', label='Computed', linewidth=1.5, alpha=0.7)
              
            
            ax.set_xlabel('Frame ID')
            if comp.startswith('v'):
                ax.set_ylabel(f'{comp} (m/s)')
            else:
                ax.set_ylabel(f'{comp} (rad/s)')
            
            ax.set_title(f'{comp} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)          # per evidenziare valore 0
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_plot = Path(self.velocity_csv_file).parent / f"velocity_comparison_gt_vs_computed_{timestamp}.png"
        plt.savefig(output_plot, dpi=150)
        print(f"Saved plot to {output_plot}")
        plt.close()
    
    def run(self):
        """Run complete analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.load_poses()
        self.load_gt_poses()
        self.load_estimated_velocities()
        computed_df = self.compute_velocities()
        gt_df = self.compute_gt_velocities() if len(self.gt_poses) > 0 else None
        estimated_df, comp_df = self.compare_velocities()
        self.save_comparison_csv(estimated_df, comp_df, gt_df, timestamp)
        self.plot_comparison(estimated_df, comp_df, timestamp)
        if gt_df is not None and len(gt_df) > 0:
            self.plot_gt_vs_computed(gt_df, comp_df, timestamp, limit=None)  # Limit to first X frames for GT vs Computed plot
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)


if __name__ == "__main__":
    # Input paths
    base_path = Path("") # percorso base dei file delle successive 3 righe
    estimate_file = base_path / "estimate.txt"              # file delle pose dell'output di mad-icp
    velocity_csv = base_path / "estimated_velocities.csv"   # file delle velocità stimate dal vel_estimator
    gt_file = base_path / "gt_kitti_stairs.txt"             # Ground truth file
    
    # Check files exist
    if not estimate_file.exists():
        print(f"Error: {estimate_file} not found!")
        exit(1)
    
    if not velocity_csv.exists():
        print(f"Error: {velocity_csv} not found!")
        exit(1)
    
    # Run analyzer
    analyzer = PoseAnalyzer(estimate_file, velocity_csv, gt_file)
    analyzer.run()
