import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "2d_wave.csv"
GIF_FILE = "wave_simulation.gif"

SIDE = 256                  # Must match the C code's SIDE definition
MAX_FRAMES = 64            # Truncate to the first 120 frames to keep file size small
FPS = 2                    # Adjust animation speed here
STRIDE = 4                  # Skip pixels for rendering speed (1 = slow/high detail, 4 = fast)

print(f"Reading up to {MAX_FRAMES} frames from {CSV_FILE}...")

try:
    # usecols=range(SIDE*SIDE) ignores the trailing "10" written by fputs in the C code
    data = np.loadtxt(CSV_FILE, delimiter=',', max_rows=MAX_FRAMES, usecols=range(SIDE * SIDE))
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Ensure the C code has run successfully.")
    sys.exit()

# Reshape the flat 1D rows into a 3D array: (Time, Y, X)
num_loaded_frames = data.shape[0]
data = data.reshape(num_loaded_frames, SIDE, SIDE)
print(f"Successfully loaded {num_loaded_frames} frames.")

# ==========================================
# 3D VISUALIZATION SETUP
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create coordinate grids using the STRIDE
X, Y = np.meshgrid(np.arange(0, SIDE, STRIDE), np.arange(0, SIDE, STRIDE))

# Pre-compute global min and max so the Z-axis doesn't bounce around during animation
z_min, z_max = np.min(data), np.max(data)

# We store the surface object in a list so we can update/replace it inside the loop
surf = [None]

def update_frame(frame_idx):
    if surf[0] is not None:
        surf[0].remove()  # Clear the old surface

    # Get the Z values for this time step, applying the same spatial STRIDE
    Z = data[frame_idx, ::STRIDE, ::STRIDE]

    # Plot the new surface. cmap='viridis' gives it a clean, scientific topographic look.
    surf[0] = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none',
                              vmin=z_min, vmax=z_max)

    # Lock the axes
    ax.set_zlim(z_min, z_max)
    ax.set_title(f"Wave Simulation (Frame {frame_idx + 1}/{num_loaded_frames})")

    # Simple progress tracker
    if frame_idx % 10 == 0:
        print(f"Rendering frame {frame_idx + 1}...")

    return surf[0],

print(f"Generating 3D animation at {FPS} FPS...")
ani = animation.FuncAnimation(fig, update_frame, frames=num_loaded_frames, blit=False)

# Save as an animated GIF using Pillow (built into Matplotlib)
writer = animation.PillowWriter(fps=FPS)
ani.save(GIF_FILE, writer=writer)

print(f"Done! Animation saved as '{GIF_FILE}'")