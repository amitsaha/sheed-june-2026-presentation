import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. SETUP INFOGRAPHIC CANVAS
# ==========================================
fig = plt.figure(figsize=(20, 9))
plt.rcParams["font.family"] = "sans-serif"
fig.patch.set_facecolor("#f8f9fa")  # Light medical/clean background

fig.suptitle(
    'How AI "Learns": From Numbers to Diagnosis (Deep Learning)',
    fontsize=28,
    fontweight="bold",
    y=0.95,
    color="#2c3e50",
)

# Create a single large axis to draw our diagram freely
ax = fig.add_axes([0, 0, 1, 0.85])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def draw_text(
    x,
    y,
    text,
    size=14,
    weight="normal",
    color="black",
    ha="center",
    va="center",
    bg=None,
):
    if bg:
        ax.text(
            x,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha=ha,
            va=va,
            bbox=dict(facecolor=bg, edgecolor="none", boxstyle="round,pad=0.5"),
        )
    else:
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


# ==========================================
# 2. STAGE 1: THE INPUT (Raw Numbers)
# ==========================================
# Draw the RGB matrices as stacked 2D boxes
for i, color in enumerate(["#0088ff", "#00cc00", "#ff3333"]):
    offset = i * 2
    rect = patches.Rectangle(
        (5 + offset, 45 + offset),
        12,
        12,
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
        alpha=0.7,
    )
    ax.add_patch(rect)

draw_text(11, 38, "INPUT DATA", size=16, weight="bold", color="#2c3e50")
draw_text(
    11,
    33,
    "The numerical grids\n(R, G, B channels)\nwe just extracted.",
    size=13,
    color="#555555",
)

# ==========================================
# 3. STAGE 2: LAYER 1 (Low-Level Features)
# ==========================================
# Draw connecting lines (representing mathematical weights)
for y in [45, 51, 57]:
    ax.plot([20, 28], [y, 65], color="#bdc3c7", linewidth=1.5, zorder=1)
    ax.plot([20, 28], [y, 35], color="#bdc3c7", linewidth=1.5, zorder=1)

# Draw "Feature Maps" (AI looking for basic shapes/edges)
for i in range(4):
    y_pos = 65 - (i * 10)
    rect = patches.Rectangle(
        (28, y_pos), 8, 8, linewidth=1, edgecolor="#34495e", facecolor="#ecf0f1"
    )
    ax.add_patch(rect)
    # Add a little "pattern" inside to simulate edge detection
    ax.plot([29, 35], [y_pos + 4, y_pos + 4], color="#34495e", linewidth=2)

draw_text(32, 23, "LAYER 1", size=16, weight="bold", color="#2c3e50")
draw_text(
    32,
    18,
    "Math filters detect\nbasic edges, lines,\nand color gradients.",
    size=13,
    color="#555555",
)

# Learning Annotation
draw_text(
    24,
    75,
    "LEARNING HAPPENS HERE:\nAI adjusts the 'math filters'\nbased on millions of past patients.",
    size=11,
    weight="bold",
    color="#c0392b",
    bg="#fadbd8",
)

# ==========================================
# 4. STAGE 3: N-LAYERS (The "Deep" in Deep Learning)
# ==========================================
# Draw a stylized "Black Box" / Hidden Layers area
box = patches.FancyBboxPatch(
    (45, 30),
    15,
    40,
    boxstyle="round,pad=1",
    linewidth=2,
    edgecolor="#3498db",
    facecolor="#ebf5fb",
    linestyle="--",
)
ax.add_patch(box)

draw_text(52.5, 50, "N - LAYERS\n(Deep CNN)", size=16, weight="bold", color="#2980b9")
draw_text(
    52.5,
    38,
    "Combining basic\nedges into complex\nbiological shapes\n(e.g., cell nuclei,\ntissue structures).",
    size=13,
    color="#34495e",
)

# Draw arrows flowing in and out
ax.arrow(
    38, 50, 4, 0, head_width=2, head_length=2, fc="#7f8c8d", ec="#7f8c8d", linewidth=2
)
ax.arrow(
    63, 50, 4, 0, head_width=2, head_length=2, fc="#7f8c8d", ec="#7f8c8d", linewidth=2
)

# ==========================================
# 5. STAGE 4: FINAL LAYER (Decision Nodes)
# ==========================================
# Draw fully connected neurons (circles)
node_y_positions = [65, 55, 45, 35, 25]
node_colors = [
    "#2ecc71",
    "#e74c3c",
    "#95a5a6",
    "#e74c3c",
    "#2ecc71",
]  # Simulate activations

for i, y in enumerate(node_y_positions):
    circle = patches.Circle(
        (75, y), 3, linewidth=2, edgecolor="#2c3e50", facecolor=node_colors[i]
    )
    ax.add_patch(circle)
    # Connect N-layers to final nodes
    ax.plot([67, 72], [50, y], color="#bdc3c7", linewidth=1, zorder=0)

draw_text(75, 15, "FINAL LAYER", size=16, weight="bold", color="#2c3e50")
draw_text(
    75,
    8,
    "Distills all visual data\ninto final clinical\nbiomarkers.",
    size=13,
    color="#555555",
)

# ==========================================
# 6. STAGE 5: THE OUTPUT (Clinical Diagnosis)
# ==========================================
# Connect final nodes to output
for y in node_y_positions:
    ax.plot([78, 85], [y, 55], color="#bdc3c7", linewidth=1, zorder=0)
    ax.plot([78, 85], [y, 45], color="#bdc3c7", linewidth=1, zorder=0)

# Draw Output Bars
draw_text(92, 60, "AI PREDICTION", size=18, weight="bold", color="#2c3e50")

# Benign Bar
ax.add_patch(
    patches.Rectangle((85, 52), 2, 4, facecolor="#ecf0f1", edgecolor="black")
)  # Base bar
ax.add_patch(patches.Rectangle((85, 52), 0.24, 4, facecolor="#2ecc71"))  # Fill
draw_text(
    85, 54, " Benign Tissue (12%)", size=14, weight="bold", color="#27ae60", ha="left"
)

# Malignant / Abnormal Bar
ax.add_patch(
    patches.Rectangle((85, 42), 12, 4, facecolor="#ecf0f1", edgecolor="black")
)  # Base bar
ax.add_patch(patches.Rectangle((85, 42), 10.5, 4, facecolor="#e74c3c"))  # Fill
draw_text(
    85,
    44,
    " Abnormal Pathology (88%)",
    size=14,
    weight="bold",
    color="#c0392b",
    ha="left",
)

draw_text(
    91,
    32,
    "The final math calculation\nresults in a statistical\nprobability.",
    size=13,
    color="#555555",
)

# Save high-resolution version
plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)

plt.show()
