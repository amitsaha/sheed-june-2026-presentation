import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

steps = [
    ("Digital images", "Images become computable\nas pixel matrices"),
    ("Image processing", "Edges, filters, shapes,\ntexture, morphology"),
    # ("Handcrafted features", "SIFT, HOG, LBP;\nengineered descriptors"),
    # ("Classical machine learning", "SVMs, boosting,\nrandom forests"),
    ("....", "Handcrafted features, classicial machine learning"),
    ("Deep learning / CNNs", "Visual features learned\nautomatically from data"),
    ("Computer vision", "Detection, segmentation,\ntracking, pose estimation"),
    ("Medical computer vision", "X-ray, CT, MRI, retina,\npathology, ultrasound"),
    ("Clinical AI applications", "Screening, diagnosis,\ntriage, workflow support"),
]

box_w = 6.8
box_h = 1.05
x = 5
top_y = 12.6
gap = 1.55

colors = [
    "#eef3f8",
    "#eef3f8",
    "#f2f8f2",
    "#f2f8f2",
    "#fff1e8",
    "#fff1e8",
    "#f5effa",
    "#f5effa",
]

edge_colors = [
    "#34495e",
    "#34495e",
    "#2e7d32",
    "#2e7d32",
    "#d35400",
    "#d35400",
    "#6c3483",
    "#6c3483",
]

for i, ((title, subtitle), fc, ec) in enumerate(zip(steps, colors, edge_colors)):
    y = top_y - i * gap

    box = FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.22,rounding_size=0.16",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y + 0.18,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a252f",
    )

    ax.text(
        x,
        y - 0.22,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
        linespacing=1.2,
    )

    if i < len(steps) - 1:
        y_next = top_y - (i + 1) * gap
        arrow = FancyArrowPatch(
            (x, y - box_h / 2 - 0.08),
            (x, y_next + box_h / 2 + 0.08),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color="#2c3e50",
        )
        ax.add_patch(arrow)

# ax.text(
#     5,
#     13.7,
#     "Evolution of Computer Vision",
#     ha="center",
#     va="center",
#     fontsize=22,
#     fontweight="bold",
#     color="#17202a",
# )

ax.text(
    5,
    13.7,
    "From pixels to clinical decision support",
    ha="center",
    va="center",
    fontsize=14,
    color="#17202a",
)

plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)
plt.show()
