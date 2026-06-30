import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.ticker import MultipleLocator

# Adjust default font to make sure it looks modern and clean
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Initialize Figure
fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

milestones = [
    {
        "year": 1963,
        "title": "3D Object Recognition",
        "desc": "Early machine perception:\n3D structure from 2D line drawings",
        "offset": 1.3,
        "border_color": "#34495e",
    },
    {
        "year": 1986,
        "title": "Canny Edge Detector",
        "desc": "Formal criteria for robust\nedge detection and localization",
        "offset": 1.4,
        "border_color": "#27ae60",
    },
    # {
    #     "year": 2001,
    #     "title": "Viola–Jones Face Detector",
    #     "desc": "Real-time face detection\nusing Haar features and AdaBoost",
    #     "offset": -1.1,
    #     "border_color": "#27ae60",
    # },
    # {
    #     "year": 2004,
    #     "title": "SIFT Features",
    #     "desc": "Scale- and rotation-invariant\nlocal image descriptors",
    #     "offset": 1.5,
    #     "border_color": "#27ae60",
    # },
    {
        "year": 2009,
        "title": "ImageNet Dataset",
        "desc": "Large-scale labelled image dataset",
        "offset": -0.7,
        "border_color": "#27ae60",
    },
    {
        "year": 2012,
        "title": "AlexNet",
        "desc": "GPU-trained CNN marks the\nlarge-scale deep learning shift",
        "offset": 0.8,
        "border_color": "#d35400",
    },
    {
        "year": 2015,
        "title": "ResNet",
        "desc": "Very deep residual networks\nand real-time object detection",
        "offset": -1.8,
        "border_color": "#d35400",
    },
    {
        "year": 2016,
        "title": "DenseNet",
        "desc": "More efficient during deployment\nBuilds on ResNet",
        "offset": 1.5,
        "border_color": "#d35400",
    },
    {
        "year": 2020,
        "title": "Vision Transformer",
        "desc": "Self-attention applied to\nimage patches at scale",
        "offset": 2.4,
        "border_color": "#8e44ad",
    },
    {
        "year": 2021,
        "title": "CLIP",
        "desc": "Natural-language supervision\nfor transferable vision models",
        "offset": -3.0,
        "border_color": "#8e44ad",
    },
    {
        "year": 2024,
        "title": "MedSAM",
        "desc": "Segment anything in medical images",
        "offset": 0.9,
        "border_color": "#8e44ad",
    },
]


eras = [
    {
        "start": 1958,
        "end": 1982,
        "color": "#f0f4f8",
        "label": "FOUNDATIONAL ERA\n(1960s–early 1980s)",
        "text_color": "#475b62",
    },
    {
        "start": 1982,
        "end": 2012,
        "color": "#f4f9f4",
        "label": "HANDCRAFTED FEATURES\n+ CLASSICAL ML\n(1980s–2000s)",
        "text_color": "#2e5a36",
    },
    {
        "start": 2012,
        "end": 2027,
        "color": "#fef5f0",
        "label": "DEEP LEARNING\nAND\nFOUNDATION MODELS\n(2010-)",
        "text_color": "#a0522d",
    },
]
# Draw background bands representing the Eras
for era in eras:
    ax.axvspan(era["start"], era["end"], color=era["color"], zorder=1)

    # Calculate midpoint for placing the Era Labels
    midpoint = (era["start"] + era["end"]) / 2
    ax.text(
        midpoint,
        2.9,
        era["label"],
        color=era["text_color"],
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.85,
        style="italic",
        zorder=2,
    )

# ----------------------------------------------------
# 2. Add Axis Styling & Central Timeline Baseline
# ----------------------------------------------------
ax.set_xlim(1958, 2027)
ax.set_ylim(-3.5, 3.5)

# Solid timeline line
ax.plot([1958, 2026.2], [0, 0], color="#2c3e50", linewidth=3, zorder=3)

# Arrowhead indicating future growth
ax.annotate(
    "",
    xy=(2027, 0),
    xytext=(2026.2, 0),
    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=3, mutation_scale=20),
    zorder=3,
)

# Hide y-axis completely as it has no physical unit scale here
ax.get_yaxis().set_visible(False)

# Clean up spines (borders)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

# Format X-axis Ticks
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.tick_params(axis="x", colors="#555555", labelsize=11, length=6, width=1.5)
ax.set_xlabel("Year", fontsize=10, fontweight="bold", color="#2c3e50", labelpad=15)
ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#bdc3c7")

# ----------------------------------------------------
# 4. Plot Stems, Nodes, and Annotation Boxes
# ----------------------------------------------------
for m in milestones:
    x = m["year"]
    y = m["offset"]

    # Draw timeline node (white circle with dark blue border)
    ax.scatter(
        x, 0, facecolor="white", edgecolor="#2c3e50", linewidth=2, s=120, zorder=5
    )

    # Draw dashed vertical stem connecting node to the label box
    ax.plot([x, x], [0, y], color="#7f8c8d", linestyle="--", linewidth=1.2, zorder=2)

    # Create text area for Title and Description
    t_title = TextArea(
        m["title"],
        textprops=dict(color="#1a252f", size=9.5, weight="bold", ha="center"),
    )
    t_desc = TextArea(m["desc"], textprops=dict(color="#555555", size=6.2, ha="center"))

    # Stack title above description
    packer = VPacker(children=[t_title, t_desc], align="center", pad=2, sep=2)

    # Put packed text into an AnnotationBbox positioned at (x, y)
    ab = AnnotationBbox(
        packer,
        (x, y),
        xybox=(0, 0),
        xycoords="data",
        boxcoords="offset points",
        bboxprops=dict(
            boxstyle="round,pad=0.6",
            fc="#ffffff",
            ec=m["border_color"],
            lw=1.5,
            alpha=0.98,
        ),
        arrowprops=None,
        zorder=10,
    )
    ax.add_artist(ab)

# ----------------------------------------------------
# 5. Title, Subtitle, and Layout Tweaks
# ----------------------------------------------------
# ax.set_title(
#     "The Evolution of Computer Vision",
#     fontsize=22,
#     fontweight="bold",
#     pad=30,
#     color="#2c3e50",
# )
# fig.text(
#     0.5,
#     0.91,
#     "From geometric reconstructions to foundation models and generative AI",
#     ha="center",
#     fontsize=12,
#     color="#7f8c8d",
#     style="italic",
# )

# Adjust layout to make sure text is not cut off
plt.subplots_adjust(top=0.85, bottom=0.1)

# Save high-resolution version
plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)

# Display the plot
plt.show()
