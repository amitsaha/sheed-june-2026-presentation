import matplotlib.pyplot as plt
import numpy as np

# Adjust default font to make sure it looks clean and professional
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Initialize 3-panel figure
fig, axs = plt.subplots(1, 3, figsize=(18, 9), facecolor="#f8fafc")

# ------------------------------------------------------------------
# PANEL 1: DATA COHORT UNDERREPRESENTATION (Input Disparity)
# ------------------------------------------------------------------
# Data representing the historic Fitzpatrick Skin Type (FST) distribution in dermatology archives (like pre-2023 ISIC)
skin_types = [
    "Type I\n(Pale)",
    "Type II\n(Fair)",
    "Type III\n(Light Brown)",
    "Type IV\n(Moderate)",
    "Type V\n(Dark Brown)",
    "Type VI\n(Deep Black)",
]
percentages = [25.0, 36.5, 23.5, 10.2, 3.8, 1.0]

# Fitzpatrick Scale-inspired color palette for bars
fitz_colors = ["#fdf2e9", "#f5cba7", "#eb984e", "#a04000", "#5e3300", "#2c1a04"]

axs[0].set_facecolor("#ffffff")
bars = axs[0].bar(
    skin_types,
    percentages,
    color=fitz_colors,
    edgecolor="#94a3b8",
    linewidth=1.2,
    zorder=3,
)

# Add exact percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    axs[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#334155",
    )

# Styling Panel 1
axs[0].set_title(
    "1. THE TRAINING DATA GAP", fontsize=13, fontweight="bold", color="#1e293b", pad=15
)
axs[0].set_ylabel(
    "Percentage of Training Images (%)",
    fontsize=10.5,
    fontweight="bold",
    color="#475569",
)
axs[0].set_ylim(0, 45)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["left"].set_color("#cbd5e1")
axs[0].spines["bottom"].set_color("#cbd5e1")
axs[0].tick_params(axis="both", colors="#475569", labelsize=9.5)
axs[0].grid(axis="y", linestyle=":", alpha=0.5, color="#cbd5e1", zorder=0)

# Explanatory annotation box for Panel 1
axs[0].text(
    2.5,
    -7.5,
    "Historical benchmarks represent lighter skins (I-III) at over 85%,\n"
    "creating severe performance degradation in darker-toned patients (IV-VI).",
    ha="center",
    fontsize=9.5,
    color="#475569",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", fc="#f1f5f9", ec="#e2e8f0", lw=1),
)


# ------------------------------------------------------------------
# PANEL 2: HOW THE MODEL LEARNS (Algorithmic Shortcuts)
# ------------------------------------------------------------------
# We will use this middle subplot as a custom drawing canvas to explain "Shortcut Learning"
axs[1].set_facecolor("#ffffff")
axs[1].set_xlim(0, 10)
axs[1].set_ylim(0, 10)
axs[1].axis("off")

# Title for Panel 2
axs[1].text(
    5,
    10.3,
    "2. ALGORITHMIC SHORTCUT PATHWAY",
    fontsize=13,
    fontweight="bold",
    ha="center",
    color="#1e293b",
)

# Box 1: Input Image
axs[1].text(
    5,
    9.0,
    "INPUT CLINICAL IMAGE\n(e.g., Chest X-Ray / Skin Photograph)",
    ha="center",
    va="center",
    weight="bold",
    size=9.5,
    color="#1e293b",
    bbox=dict(boxstyle="round,pad=0.6", fc="#f1f5f9", ec="#cbd5e1", lw=1.5),
)

# Box 2A (Left): True Clinical Signal
axs[1].text(
    2.3,
    5.2,
    "True Pathology Signal\n(Target Clinical Intent)\n\n"
    "• Actual cellular morphology\n"
    "• Textural tissue changes\n"
    "• True anatomical disease",
    ha="center",
    va="center",
    size=8.5,
    color="#047857",
    bbox=dict(boxstyle="round,pad=0.6", fc="#ecfdf5", ec="#10b981", lw=1.2),
)

# Box 2B (Right): Spurious Shortcuts
axs[1].text(
    7.7,
    5.2,
    "Confounding 'Shortcuts'\n(Exploited Covariates)\n\n"
    "• Surgical skin markings\n"
    "• Hospital site metadata\n"
    "• Scanner hardware noise",
    ha="center",
    va="center",
    size=8.5,
    color="#b91c1c",
    bbox=dict(boxstyle="round,pad=0.6", fc="#fef2f2", ec="#ef4444", lw=1.2),
)

# Box 3: Biased Outcome
axs[1].text(
    5,
    1.2,
    "BIASED MODEL DECISION\nFails on underserved patient populations with different\nsurgical markers, scanner types, or imaging protocols",
    ha="center",
    va="center",
    weight="bold",
    size=9,
    color="#ffffff",
    bbox=dict(boxstyle="round,pad=0.6", fc="#1e293b", ec="#0f172a", lw=1.5),
)

# Connective Flow Arrows (Annotate)
# Input -> Pathology (Green Arrow, thinner)
axs[1].annotate(
    "",
    xy=(2.3, 6.7),
    xytext=(4.2, 8.4),
    arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.5, ls=":"),
)
axs[1].text(
    2.9, 7.7, "Ignored Signal", size=8, color="#047857", ha="center", rotation=35
)

# Input -> Confounders (Red Arrow, thick)
axs[1].annotate(
    "",
    xy=(7.7, 6.7),
    xytext=(5.8, 8.4),
    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.5),
)
axs[1].text(
    7.1,
    7.7,
    "Exploited Bias",
    size=8.5,
    weight="bold",
    color="#b91c1c",
    ha="center",
    rotation=-35,
)

# Pathology -> Output (Weak weight)
axs[1].annotate(
    "",
    xy=(4.0, 1.8),
    xytext=(2.3, 3.7),
    arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5),
)

# Confounders -> Output (Heavy weight)
axs[1].annotate(
    "",
    xy=(6.0, 1.8),
    xytext=(7.7, 3.7),
    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.5),
)


# ------------------------------------------------------------------
# PANEL 3: CLINICAL CONSEQUENCES (Diagnostic Underdiagnosis Gap)
# ------------------------------------------------------------------
# Data directly based on Nature Medicine (2021) chest X-ray underdiagnosis study by Seyyed-Kalantari et al.
cohorts = [
    "White (Reference)",
    "Black",
    "Hispanic",
    "Medicaid (Low SES)",
    "Hispanic Female + Medicaid",
]
underdiagnosis_rates = [15.2, 21.5, 23.8, 26.4, 31.1]

# Cascading warning-red color palette representing cumulative/intersectional bias
disparity_colors = ["#94a3b8", "#fda4af", "#f43f5e", "#db2777", "#9f1239"]

axs[2].set_facecolor("#ffffff")
y_pos = np.arange(len(cohorts))
bars_h = axs[2].barh(
    y_pos,
    underdiagnosis_rates,
    color=disparity_colors,
    edgecolor="#475569",
    linewidth=1.2,
    zorder=3,
)

# Add value markers at the end of each bar
for bar in bars_h:
    width = bar.get_width()
    axs[2].text(
        width + 1.0,
        bar.get_y() + bar.get_height() / 2.0,
        f"{width:.1f}%",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1e293b",
    )

# Draw baseline reference line (White cohort rate) to visually anchor the gap
axs[2].axvline(
    15.2, color="#475569", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2
)
axs[2].text(
    15.2,
    -0.6,
    "Reference",
    ha="center",
    va="top",
    fontsize=8,
    color="#475569",
    weight="bold",
)

# Styling Panel 3
axs[2].set_title(
    "3. THE UNDERDIAGNOSIS GAP", fontsize=13, fontweight="bold", color="#1e293b", pad=15
)
axs[2].set_xlabel(
    "AI Underdiagnosis Rate (False Negative Rate, %)",
    fontsize=10.5,
    fontweight="bold",
    color="#475569",
)
axs[2].set_yticks(y_pos)
axs[2].set_yticklabels(cohorts, fontsize=9.5, fontweight="bold", color="#334155")
axs[2].set_xlim(0, 40)
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].spines["left"].set_color("#cbd5e1")
axs[2].spines["bottom"].set_color("#cbd5e1")
axs[2].tick_params(axis="x", colors="#475569", labelsize=9.5)
axs[2].grid(axis="x", linestyle=":", alpha=0.5, color="#cbd5e1", zorder=0)

# Explanatory annotation box for Panel 3
axs[2].text(
    20,
    -1.8,
    "The model mistakenly classifies sick patients as 'healthy' at significantly\n"
    "higher rates in marginalized populations, directly delaying critical treatment.",
    ha="center",
    fontsize=9.5,
    color="#475569",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", fc="#f1f5f9", ec="#e2e8f0", lw=1),
)


# ------------------------------------------------------------------
# GLOBAL FIGURE TITLES & LAYOUT TWEAKS
# ------------------------------------------------------------------
# fig.suptitle(
#     "The Lifecycle of Demographic Bias in Medical Computer Vision",
#     fontsize=21,
#     fontweight="bold",
#     color="#1e293b",
#     y=0.96,
# )
# fig.text(
#     0.5,
#     0.90,
#     "How initial population exclusion scales into structural disparities in clinical diagnostics",
#     ha="center",
#     va="center",
#     fontsize=12,
#     color="#64748b",
#     style="italic",
# )

# # Footnote / Sources attribution
# fig.text(
#     0.5,
#     0.03,
#     "Sources & Reference Cases: (1) Fitzpatrick skin tone representation in standard ISIC dermatology archives[<vertex-ai-rich-citation-chip>1</vertex-ai-rich-citation-chip>]. "
#     "(2) Seyyed-Kalantari et al., Nature Medicine (2021)[<vertex-ai-rich-citation-chip>2</vertex-ai-rich-citation-chip>].",
#     ha="center",
#     va="center",
#     fontsize=9,
#     color="#94a3b8",
# )

# Clean layout spacing
plt.subplots_adjust(top=0.81, bottom=0.17, left=0.06, right=0.94, wspace=0.32)

# Save high-resolution version
plt.savefig(f"{__file__}.png", bbox_inches="tight", dpi=300)

# Display the plot
plt.show()
