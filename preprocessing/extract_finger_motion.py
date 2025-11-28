"""
Extract finger motion features from webcam recordings.
Processes JSON from MediaPipe hand tracking → clean (x, y) trajectories.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
import argparse


def load_webcam_data(json_path: str) -> dict:
    """Load recorded webcam tracking data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_trajectories(data: list) -> np.ndarray:
    """
    Extract (x, y) finger trajectories from tracking data.
    
    Args:
        data: List of recorded frames with timestamps and positions
        
    Returns:
        Array of shape (num_frames, 3) with [timestamp, x, y]
    """
    trajectories = []
    
    for frame in data:
        timestamp = frame['timestamp']
        x = frame['x']
        y = frame['y']
        
        trajectories.append([timestamp, x, y])
    
    return np.array(trajectories)


def smooth_trajectory(trajectory: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Smooth trajectory using Savitzky-Golay filter.
    
    Args:
        trajectory: Array of shape (num_frames, 3) [timestamp, x, y]
        window_length: Filter window size (must be odd)
        polyorder: Polynomial order
        
    Returns:
        Smoothed trajectory
    """
    smoothed = trajectory.copy()
    
    # Smooth x and y separately (keep timestamps unchanged)
    if len(trajectory) > window_length:
        smoothed[:, 1] = savgol_filter(trajectory[:, 1], window_length, polyorder)
        smoothed[:, 2] = savgol_filter(trajectory[:, 2], window_length, polyorder)
    
    return smoothed


def normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    Normalize x, y coordinates to [0, 1] range.
    
    Args:
        trajectory: Array of shape (num_frames, 3) [timestamp, x, y]
        
    Returns:
        Normalized trajectory
    """
    normalized = trajectory.copy()
    
    # Normalize x
    x_min, x_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    if x_max > x_min:
        normalized[:, 1] = (trajectory[:, 1] - x_min) / (x_max - x_min)
    
    # Normalize y
    y_min, y_max = trajectory[:, 2].min(), trajectory[:, 2].max()
    if y_max > y_min:
        normalized[:, 2] = (trajectory[:, 2] - y_min) / (y_max - y_min)
    
    return normalized


def resample_trajectory(trajectory: np.ndarray, target_fps: int = 30) -> np.ndarray:
    """
    Resample trajectory to target frame rate.
    
    Args:
        trajectory: Array of shape (num_frames, 3) [timestamp, x, y]
        target_fps: Target frames per second
        
    Returns:
        Resampled trajectory
    """
    if len(trajectory) < 2:
        return trajectory
    
    # Create uniform time grid
    t_start = trajectory[0, 0]
    t_end = trajectory[-1, 0]
    duration = t_end - t_start
    
    num_samples = int(duration * target_fps)
    t_uniform = np.linspace(t_start, t_end, num_samples)
    
    # Interpolate x and y
    x_interp = np.interp(t_uniform, trajectory[:, 0], trajectory[:, 1])
    y_interp = np.interp(t_uniform, trajectory[:, 0], trajectory[:, 2])
    
    # Combine
    resampled = np.column_stack([t_uniform, x_interp, y_interp])
    
    return resampled


def visualize_trajectory(trajectory: np.ndarray, title: str = "Finger Trajectory", save_path: str = None):
    """Visualize finger trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot x and y over time
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], label='X position', linewidth=2)
    axes[0].plot(trajectory[:, 0], trajectory[:, 2], label='Y position', linewidth=2)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Position (normalized)')
    axes[0].set_title(f'{title} - Position vs Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2D trajectory
    axes[1].plot(trajectory[:, 1], trajectory[:, 2], linewidth=2, alpha=0.6)
    axes[1].scatter(trajectory[0, 1], trajectory[0, 2], c='green', s=100, label='Start', zorder=5)
    axes[1].scatter(trajectory[-1, 1], trajectory[-1, 2], c='red', s=100, label='End', zorder=5)
    axes[1].set_xlabel('X position (normalized)')
    axes[1].set_ylabel('Y position (normalized)')
    axes[1].set_title(f'{title} - 2D Path')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Process webcam finger tracking data."""
    parser = argparse.ArgumentParser(description='Extract finger motion from webcam data')
    parser.add_argument('--input', default='data/recordings/finger_tracking.json',
                       help='Input JSON file')
    parser.add_argument('--output', default='data/features/finger_xy.npy',
                       help='Output numpy file')
    parser.add_argument('--target-fps', type=int, default=30,
                       help='Target frame rate')
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='Smoothing window size')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Extracting Finger Motion from Webcam Data")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target FPS: {args.target_fps}")
    print()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found!")
        print("Please record webcam data first using record_webcam.html")
        return
    
    # Load data
    print("Loading webcam data...")
    data = load_webcam_data(args.input)
    print(f"✓ Loaded {len(data)} frames")
    
    # Extract trajectories
    print("\nExtracting trajectories...")
    trajectory = extract_trajectories(data)
    print(f"✓ Extracted trajectory: {trajectory.shape}")
    print(f"  Duration: {trajectory[-1, 0] - trajectory[0, 0]:.2f} seconds")
    print(f"  Original FPS: {len(trajectory) / (trajectory[-1, 0] - trajectory[0, 0]):.1f}")
    
    # Smooth trajectory
    print("\nSmoothing trajectory...")
    smoothed = smooth_trajectory(trajectory, window_length=args.smooth_window)
    print(f"✓ Applied Savitzky-Golay filter (window={args.smooth_window})")
    
    # Normalize
    print("\nNormalizing coordinates...")
    normalized = normalize_trajectory(smoothed)
    print(f"✓ Normalized to [0, 1] range")
    
    # Resample
    print(f"\nResampling to {args.target_fps} FPS...")
    resampled = resample_trajectory(normalized, target_fps=args.target_fps)
    print(f"✓ Resampled: {resampled.shape}")
    print(f"  Final FPS: {len(resampled) / (resampled[-1, 0] - resampled[0, 0]):.1f}")
    
    # Save
    np.save(args.output, resampled)
    print(f"\n✓ Saved to {args.output}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total frames: {len(resampled)}")
    print(f"  Duration: {resampled[-1, 0] - resampled[0, 0]:.2f} seconds")
    print(f"  X range: [{resampled[:, 1].min():.3f}, {resampled[:, 1].max():.3f}]")
    print(f"  Y range: [{resampled[:, 2].min():.3f}, {resampled[:, 2].max():.3f}]")
    
    # Visualize
    if args.visualize:
        print("\nCreating visualization...")
        viz_path = output_path.parent / 'finger_trajectory_visualization.png'
        visualize_trajectory(resampled, save_path=str(viz_path))
    
    print("\n✓ Finger motion extraction complete!")
    print(f"\nNext step: python preprocessing/process_imu.py")


if __name__ == "__main__":
    main()