import numpy as np
import matplotlib.pyplot as plt

def generate_racetrack(track_width=20, radius=100, num_segments=8, save_path="../tracks/racetrack_test.png"):
    # Generate angles for segments
    angles = np.linspace(0, 2 * np.pi, num_segments + 1)

    # Inner boundary coordinates
    inner_x = (radius * np.cos(angles))
    inner_y = (radius * np.sin(angles))

    # Outer boundary coordinates
    outer_radius = radius + track_width
    outer_x = (outer_radius * np.cos(angles))
    outer_y = (outer_radius * np.sin(angles))

    # Plot the racetrack
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(inner_x, inner_y)
    ax.plot(outer_x, outer_y)
    plt.savefig(save_path)

generate_racetrack()