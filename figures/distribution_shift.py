import matplotlib.pyplot as plt
import numpy as np

# 1. Generate synthetic data
np.random.seed(42)

# Training data distribution (e.g., historical clinic data)
train_features = np.random.normal(loc=2.0, scale=1.0, size=1000)

# Production data distribution (e.g., deployed in a new clinic with a shift)
prod_features = np.random.normal(loc=5.0, scale=1.2, size=1000)

# Ground truth decision boundary (e.g., any value > 3.5 is high risk)
boundary = 3.5

# 2. Plotting the distributions
plt.figure(figsize=(10, 6))

# Plot Training Distribution Histogram
plt.hist(
    train_features,
    bins=30,
    alpha=0.6,
    color="#1f77b4",
    label="Training Data ($P_{train}(X)$)",
    edgecolor="black",
    linewidth=0.5,
)

# Plot Production Distribution Histogram
plt.hist(
    prod_features,
    bins=30,
    alpha=0.6,
    color="#ff7f0e",
    label="Production Data ($P_{test}(X)$)",
    edgecolor="black",
    linewidth=0.5,
)

# Add True Decision Boundary Line
plt.axvline(
    x=boundary,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"True Decision Boundary (X = {boundary})",
)

# 3. Formatting the plot
# plt.title("Visualization of Distribution Shift (Covariate Shift)", fontsize=14, pad=15)
plt.xlabel("Feature Value (X)", fontsize=12)
plt.ylabel("Frequency / Count", fontsize=12)
plt.xlim(-2, 10)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(fontsize=11, loc="upper right")

# Add annotations to highlight the problem area
plt.text(
    0.5,
    80,
    "Model performs\nwell here",
    color="#1f77b4",
    fontweight="bold",
    ha="center",
)
plt.text(
    6.5,
    80,
    "Unseen data territory:\nModel uncertainty / failure",
    color="#ff7f0e",
    fontweight="bold",
    ha="center",
)

plt.tight_layout()


# Save high-resolution version
plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)

plt.show()
