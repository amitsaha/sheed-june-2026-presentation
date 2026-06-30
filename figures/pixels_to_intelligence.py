import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

IMAGE_PATH = "eye.jpg"

# Creating custom colormaps for maximum presentation visibility
# Format: [Low values, Mid values, High values]
cmap_red = LinearSegmentedColormap.from_list(
    "VisRed", ["#ffeeee", "#ff3333", "#660000"]
)
cmap_green = LinearSegmentedColormap.from_list(
    "VisGreen", ["#eeffee", "#00cc00", "#003300"]
)

# 🌟 HIGH-VISIBILITY BLUE: Ice Blue -> Electric Neon Blue -> Dark Navy
cmap_blue = LinearSegmentedColormap.from_list(
    "VisBlue", ["#e8f4f8", "#0088ff", "#001155"]
)


# Load the real image
full_image = np.array(Image.open(IMAGE_PATH))

# Handle RGBA (remove transparency channel if it exists)
if full_image.shape[-1] == 4:
    full_image = full_image[:, :, :3]

# ==========================================
# 2. EXTRACT A "MICROSCOPIC" 5x5 PATCH
# ==========================================
# We zoom into the exact center of the image to grab a 5x5 pixel grid
h, w, _ = full_image.shape
cy, cx = h // 2, w // 2
patch_size = 5
half = patch_size // 2

# The 5x5x3 matrix
patch = full_image[cy - half : cy + half + 1, cx - half : cx + half + 1, :]

# Extract individual R, G, B matrices from the 5x5 patch
R = patch[:, :, 0]
G = patch[:, :, 1]
B = patch[:, :, 2]

# ==========================================
# 3. SET UP THE PRESENTATION DASHBOARD
# ==========================================
fig = plt.figure(figsize=(18, 9))
plt.rcParams["font.family"] = "sans-serif"
# fig.suptitle(
#     "From Anatomy to Algorithms: How AI Parses Real Images",
#     fontsize=26,
#     fontweight="bold",
#     y=0.98,
# )

# Create layout: Top row for images/matrices, Bottom row for the ML vector
gs = fig.add_gridspec(
    2, 5, height_ratios=[3, 1], width_ratios=[1.5, 1.2, 1, 1, 1], wspace=0.3
)

ax_full = fig.add_subplot(gs[0, 0])
ax_patch = fig.add_subplot(gs[0, 1])
ax_r = fig.add_subplot(gs[0, 2])
ax_g = fig.add_subplot(gs[0, 3])
ax_b = fig.add_subplot(gs[0, 4])
ax_ml = fig.add_subplot(gs[1, :])


def format_axis(ax, title):
    ax.set_title(title, fontsize=15, pad=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# --- Plot 1: Full Original Image ---
ax_full.imshow(full_image)
format_axis(ax_full, "1. Full Medical Image\n(Millions of Pixels)")

# Draw a red box in the center to show where we are zooming in
rect = patches.Rectangle(
    (cx - half - 5, cy - half - 5),
    patch_size + 10,
    patch_size + 10,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
ax_full.add_patch(rect)

# --- Plot 2: Zoomed 5x5 Patch ---
ax_patch.imshow(patch)
format_axis(ax_patch, f"2. Zoomed Patch\n({patch_size}x{patch_size} Pixels)")


# --- Function to overlay matrix numbers ---
def plot_matrix_with_text(ax, matrix, cmap, title):
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=255)
    format_axis(ax, title)

    # Overlay the exact numerical values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Dynamic text color for readability
            color = "white" if matrix[i, j] < 128 else "black"
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )


# --- Plot 3, 4, 5: R, G, B Matrices ---
plot_matrix_with_text(ax_r, R, cmap_red, "3. Red Channel\nMatrix (R)")
plot_matrix_with_text(ax_g, G, cmap_green, "4. Green Channel\nMatrix (G)")
plot_matrix_with_text(ax_b, B, cmap_blue, "5. Blue Channel\nMatrix (B)")

# ==========================================
# 4. SHOW MACHINE LEARNING INPUT (FLATTENED)
# ==========================================
# AI algorithms consume 1D arrays. We flatten the 5x5x3 matrix into 75 features.
ml_vector = patch.flatten()

ax_ml.set_title(
    f"6. Machine Learning Input: The 'Feature Vector' ({len(ml_vector)} Data Points)",
    fontsize=18,
    fontweight="bold",
    pad=10,
)
ax_ml.axis("off")

# Format the array for presentation
vector_text = (
    "[ "
    + ", ".join(map(str, ml_vector[:15]))
    + f", ... ({len(ml_vector) - 15} more numbers) ]"
)
ax_ml.text(
    0.5,
    0.5,
    vector_text,
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#2c3e50",
    bbox=dict(facecolor="#ecf0f1", edgecolor="#bdc3c7", boxstyle="round,pad=1"),
)

plt.tight_layout()

plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)
plt.show()
