import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP PRESENTATION CANVAS
# ==========================================
fig = plt.figure(figsize=(18, 10))
plt.rcParams["font.family"] = "sans-serif"
fig.patch.set_facecolor("#ffffff")

# fig.suptitle(
#     'Demystifying AI "Learning": Adjusting the Weights',
#     fontsize=28,
#     fontweight="bold",
#     y=0.95,
#     color="#2c3e50",
# )

# Text explaining the concept at the top
plt.figtext(
    0.5,
    0.88,
    "AI applies a 'Weight Matrix' (a mathematical filter) to patient data.\nIf the resulting score doesn't match the known diagnosis, the AI adjusts the weights. This is 'Learning'.",
    ha="center",
    fontsize=16,
    color="#555555",
)

# Use GridSpec for layout: 2 Rows (Attempt 1 vs Attempt 2), 7 Columns
gs = fig.add_gridspec(
    2, 7, width_ratios=[1, 0.2, 1, 0.2, 1, 0.2, 1.5], wspace=0.1, hspace=0.4, top=0.8
)


def format_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# Helper function to draw 2x2 matrices
def draw_matrix(ax, data, title, color_theme, is_weight=False):
    ax.imshow(data, cmap=color_theme, vmin=0, vmax=5)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Draw gridlines and numbers
    rows, cols = data.shape
    for x in range(cols + 1):
        ax.axvline(x - 0.5, color="black", linewidth=2)
    for y in range(rows + 1):
        ax.axhline(y - 0.5, color="black", linewidth=2)

    for i in range(rows):
        for j in range(cols):
            # If it's a weight matrix, add a little "x" to show it's a multiplier
            prefix = "x " if is_weight else ""
            ax.text(
                j,
                i,
                f"{prefix}{data[i, j]}",
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color="black",
            )
    format_ax(ax)


# Helper for math symbols
def add_symbol(ax, symbol):
    ax.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=40, fontweight="bold")
    format_ax(ax)


# ==========================================
# 2. ATTEMPT 1: UNTRAINED AI (The First Guess)
# ==========================================
# Data matrices
input_data = np.array([[2, 1], [0, 4]])  # Patient Data (fixed)
weights_1 = np.array([[1, 1], [1, 1]])  # Untrained Weights (Just guessing x1)
output_1 = input_data * weights_1  # Element-wise multiplication
sum_1 = np.sum(output_1)  # Final score

# Plot Row 1
ax_in1 = fig.add_subplot(gs[0, 0])
ax_sym1 = fig.add_subplot(gs[0, 1])
ax_w1 = fig.add_subplot(gs[0, 2])
ax_sym2 = fig.add_subplot(gs[0, 3])
ax_out1 = fig.add_subplot(gs[0, 4])
ax_result1 = fig.add_subplot(gs[0, 6])

draw_matrix(ax_in1, input_data, "Patient Data\n(Pixels)", "Blues")
add_symbol(ax_sym1, "$\\times$")
draw_matrix(ax_w1, weights_1, "AI Weights\n(First Guess)", "Oranges", is_weight=True)
add_symbol(ax_sym2, "$=$")
draw_matrix(ax_out1, output_1, "Resulting\nSignal", "Greys")

# Output Evaluation Box 1
format_ax(ax_result1)
ax_result1.text(0.1, 0.8, "Signal Sum:", fontsize=18, color="#555555")
ax_result1.text(0.1, 0.5, f"{sum_1}", fontsize=36, fontweight="bold", color="#e74c3c")
ax_result1.text(
    0.1,
    0.2,
    "Target for Disease X = 10\nERROR: AI is wrong.",
    fontsize=16,
    fontweight="bold",
    color="#c0392b",
)

# Add a big curved arrow to show "Backpropagation" (Learning)
style = "Simple, tail_width=2, head_width=8, head_length=8"
kw = dict(arrowstyle=style, color="#9b59b6")
arrow = patches.FancyArrowPatch(
    (0.8, 0.45),
    (0.4, 0.45),
    connectionstyle="arc3,rad=-0.1",
    transform=fig.transFigure,
    **kw,
)
fig.patches.append(arrow)
fig.text(
    0.6,
    0.46,
    "Learning Algorithm Adjusts Weights",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#8e44ad",
)

# ==========================================
# 3. ATTEMPT 2: TRAINED AI (The Correction)
# ==========================================
# Data matrices
weights_2 = np.array(
    [[2.5, 1], [0, 1]]
)  # Adjusted Weights (Learned to focus on top-left!)
output_2 = input_data * weights_2
sum_2 = np.sum(output_2)

# Plot Row 2
ax_in2 = fig.add_subplot(gs[1, 0])
ax_sym3 = fig.add_subplot(gs[1, 1])
ax_w2 = fig.add_subplot(gs[1, 2])
ax_sym4 = fig.add_subplot(gs[1, 3])
ax_out2 = fig.add_subplot(gs[1, 4])
ax_result2 = fig.add_subplot(gs[1, 6])

draw_matrix(ax_in2, input_data, "Patient Data\n(Unchanged)", "Blues")
add_symbol(ax_sym3, "$\\times$")
draw_matrix(ax_w2, weights_2, "AI Weights\n(Adjusted!)", "Wistia", is_weight=True)
add_symbol(ax_sym4, "$=$")
draw_matrix(ax_out2, output_2, "Resulting\nSignal", "Greys")

# Output Evaluation Box 2
format_ax(ax_result2)
ax_result2.text(0.1, 0.8, "Signal Sum:", fontsize=18, color="#555555")
ax_result2.text(
    0.1, 0.5, f"{int(sum_2)}", fontsize=36, fontweight="bold", color="#27ae60"
)
ax_result2.text(
    0.1,
    0.2,
    "Target for Disease X = 10\nSUCCESS: AI has 'Learned'.",
    fontsize=16,
    fontweight="bold",
    color="#27ae60",
)

plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)
plt.show()
