"""
Analysis and plots for the Color Crush dataset.

=========================================================================
OUTCOME MEASURES
=========================================================================

Three measures of the same underlying response are available for every
sublevel. All three are carried through every results table so the choice
is a documented robustness check rather than a hidden decision.

    mean_deltaE2000         PRIMARY. Perceptually uniform (CIEDE2000), so
                            comparisons ACROSS the eight base colors are
                            legitimate. Compresses large errors on
                            saturated colors.
    mean_deltaE76           Euclidean CIELAB distance. No compression,
                            less perceptually uniform.
    mean_response_magnitude The game's own logged response, in game units.
                            No hex round-trip, exactly quantised, but a
                            unit buys a different amount of visible
                            difference on each axis and base color, so it
                            must NOT be compared across colors.

=========================================================================
HYPOTHESIS FAMILIES
=========================================================================

Family A -- Mean comparison. Does average accuracy differ between groups?
    Saved to: Delta2000_analysis.xlsx

Family B -- Spread comparison. Is one group more variable/inconsistent
    than another, regardless of average?
    Saved to: spread_analysis.xlsx

Family C -- Axis-specific tendency. Is a group specifically imprecise on
    one axis (axis_error_L/a/b), or systematically biased toward one end
    of an axis (bias_L/a/b, e.g. tolerating lighter more than darker)?
    Saved to: axis_analysis.xlsx

Family D -- Non-response. Does the rate of skipped sublevels differ
    between groups? A skipped sublevel is scored by the game as maximum
    error, so this is both a data-quality check and a result in its own
    right.
    Saved to: nonresponse_analysis.xlsx

Age is additionally analysed as a CONTINUOUS predictor (Spearman and
Pearson per level and overall), because binning it into five groups
throws away most of the information and creates small groups.

=========================================================================
STATISTICAL APPROACH
=========================================================================

Two groups, following the reference paper's decision tree:
    Shapiro-Wilk for normality, Levene for equality of variances, then
    (1) Student's t-test + Cohen's d      normal, equal variances
    (2) Welch's t-test + Hedges' g        normal, unequal variances
    (3) Mann-Whitney U + rank-biserial    non-normal, or n < 15 in a group

Three or more groups:
    Kruskal-Wallis + epsilon squared, with Dunn's post-hoc (BH corrected),
    because Kruskal-Wallis alone does not say WHICH groups differ.

Because selecting a test conditionally on a pre-test has a known effect on
the true Type I error rate, and because Shapiro-Wilk has very little power
at these group sizes, the unconditional alternatives are ALSO computed and
written to every row (student_*, welch_*, mannwhitney_*, anova_*,
kruskal_*). The column test_selected_by records which branch fired and
why, so the choice is auditable.

Multiple comparisons: a Benjamini-Hochberg FDR correction is applied
within the PRIMARY comparison set only (see PRIMARY_* settings below).
Per-level and secondary-outcome comparisons are exploratory and reported
with raw p-values, clearly labelled. A correction across the whole family
is also reported for transparency.

Every comparison reports the minimum detectable effect size at the given
group sizes, so a null result can be read as "probably no effect" or "not
enough participants yet" rather than being left ambiguous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 20260727
np.random.seed(RANDOM_SEED)

ALPHA = 0.05
POWER_TARGET = 0.80

# A group needs at least this many participants to enter a comparison.
# TODO for the final analysis: 3 is enough to run a test but far too few
# to interpret one. The minimum detectable effect column will show how
# little such a comparison can see.
MIN_GROUP_SIZE = 3

# The reference paper's rule: below this n in either group, use the
# non-parametric test regardless of what Shapiro says.
SMALL_SAMPLE_THRESHOLD = 15

SUBLEVELS_PER_LEVEL = 6
MAIN_COLOR_LEVELS = 8

# Outcomes
PRIMARY_OUTCOME = "mean_deltaE2000"
SECONDARY_OUTCOMES = ["mean_deltaE76", "mean_response_magnitude"]

AXIS_ERROR_OUTCOMES = ["axis_error_L", "axis_error_a", "axis_error_b"]
AXIS_BIAS_OUTCOMES = ["bias_L", "bias_a", "bias_b"]
AXIS_OUTCOMES = AXIS_ERROR_OUTCOMES + AXIS_BIAS_OUTCOMES

GROUP_VARIABLES = [
    "biologicalSex",
    "eyeColor",
    "colorBlindness",
    "age_group",
    "Nationality",
]

# The confirmatory set. Everything else is exploratory.
PRIMARY_GROUP_VARIABLES = ["colorBlindness", "biologicalSex"]
PRIMARY_SCOPES = ["overall"]
PRIMARY_OUTCOMES = [PRIMARY_OUTCOME]

# Non-answers, excluded rather than treated as their own group. Values are
# compared case-insensitively after stripping whitespace.
EXCLUDED_GROUP_VALUES: dict[str, list[str]] = {
    "colorBlindness": ["don't know", "dont know", "unknown", ""],
    "eyeColor": ["select eye color", "nan", "none", ""],
    "biologicalSex": ["prefer not to say", "nan", ""],
    "Nationality": ["nan", ""],
}

AGE_BINS = [0, 19, 29, 39, 49, 120]
AGE_LABELS = ["<=19", "20-29", "30-39", "40-49", "50+"]

# Per-participant diagnostic plots: chosen-colour heatmaps, sublevel distance
# plots, compass spiderwebs and six-axis radars. One PNG per participant or per
# participant/level/attempt, so this is a few hundred files at full sample size.
# Set to False to skip them all when you only want the group-level results.
MAKE_PARTICIPANT_DIAGNOSTIC_PLOTS = True #TODO
DIAGNOSTIC_PLOT_DPI = 150
FIGURE_DPI = 300

LEVEL_BY_BASE_COLOR = {
    "DE3B62": 1,
    "52DE48": 2,
    "5246E8": 3,
    "E048E0": 4,
    "D8DE4D": 5,
    "5ADED6": 6,
    "C5917D": 7,
    "704C3C": 8,
}

LEVEL_COLORS = {n: f"#{h}" for h, n in LEVEL_BY_BASE_COLOR.items()}

# The six axes, in the order the sublevels are played.
AXIS_LABELS_IN_ORDER = ["+a", "+L", "+b", "-a", "-L", "-b"]


# ============================================================
# PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent
excel_file = project_folder / "excel_files" / "combined_final_results.xlsx"
plot_folder = project_folder / "plots"
plot_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SMALL HELPERS
# ============================================================

def save_current_plot(filename: str, subfolder: str | None = None) -> None:
    """Save the current figure to the plot folder."""
    folder = plot_folder if subfolder is None else plot_folder / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()

    print(f"Saved: {output_path}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check that all needed columns exist."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        print(f"Skipping because these columns are missing: {missing}")
        return False

    return True


def safe_filename(text: str) -> str:
    """Make text safe to use in a filename."""
    text = str(text)

    for bad in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", " ", "+"]:
        text = text.replace(bad, "_")

    return text[:120]


def boxplot_compat(data: list[np.ndarray], labels: list[str], **kwargs: Any) -> dict:
    """
    plt.boxplot renamed 'labels' to 'tick_labels' in matplotlib 3.9.
    Try the new name and fall back so the script runs on either version.
    """
    try:
        return plt.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return plt.boxplot(data, labels=labels, **kwargs)


MISSING_LABEL = "No answer"


def canonicalise_text_column(series: pd.Series) -> pd.Series:
    """
    Merge case and whitespace variants of a free-text answer without
    rewriting how it is spelled.

    Title-casing everything looked tidy but mangled real answers:
    str.title() splits on apostrophes, so "Don't know" became "Don'T Know",
    and it would turn "USA" into "Usa". Instead values are matched
    case-insensitively and each group is displayed using its most common
    original spelling.
    """
    text = series.apply(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    text = text.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})

    counts = text.dropna().value_counts()
    canonical_by_key: dict[str, str] = {}

    for value in counts.index:
        key = str(value).casefold()

        # value_counts is sorted descending, so the first spelling seen for
        # a key is the most common one.
        if key not in canonical_by_key:
            canonical_by_key[key] = str(value)

    return text.apply(
        lambda v: canonical_by_key.get(str(v).casefold(), v) if pd.notna(v) else np.nan
    )


def labels_from_index(index: Any) -> list[str]:
    """
    Turn an index into plain strings for a categorical matplotlib axis.

    Index.astype(str) is not safe here: recent pandas preserves missing
    values as NA instead of converting them to the string "nan", and
    matplotlib then raises TypeError on a float where it expects a string.
    """
    return [MISSING_LABEL if pd.isna(value) else str(value) for value in index]


def is_excluded_value(group_column: str, value: Any) -> bool:
    """Is this a non-answer that the comparisons drop?"""
    if pd.isna(value):
        return True

    excluded = [v.strip().lower() for v in EXCLUDED_GROUP_VALUES.get(group_column, [])]

    return str(value).strip().lower() in excluded


# ============================================================
# LOAD DATA
# ============================================================

def load_long_data() -> pd.DataFrame:
    """Load the all_final_colors_long sheet: one row per sublevel."""
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file:\n{excel_file}\n\n"
            "Run the data handler script first."
        )

    df = pd.read_excel(excel_file, sheet_name="all_final_colors_long")

    numeric_columns = [
        "age",
        "level_number",
        "attempt_number",
        "sublevel_index",
        "final_index",
        "payload_slot",
        "axis_sign",
        "deltaE76",
        "deltaE2000",
        "response_magnitude",
        "direction_magnitude",
        "sublevel_duration_ms",
        "whole_level_duration_ms",
        "sublevel_chosen_color_count",
        "sublevel_final_selected_color_count",
        "sublevel_total_color_action_count",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "sublevel_is_nonresponse" in df.columns:
        df["sublevel_is_nonresponse"] = (
            df["sublevel_is_nonresponse"].astype(str).str.lower().isin(["true", "1"])
        )

    return normalise_demographics(df)


def load_short_data() -> pd.DataFrame:
    """Load the all_levels_short sheet: one row per level attempt."""
    df = pd.read_excel(excel_file, sheet_name="all_levels_short")

    for column in df.columns:
        if column.startswith(("mean_", "median_", "min_", "max_", "std_")) or column in (
            "age",
            "level_number",
            "attempt_number",
            "whole_level_duration_ms",
            "n_sublevels",
            "n_nonresponse_sublevels",
            "total_final_selected_colors",
            "total_color_actions",
        ):
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return normalise_demographics(df)


def normalise_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy the free-text demographic answers and build the participant id
    and age group.

    Whitespace and capitalisation variants ('blue', 'Blue ') would
    otherwise become separate groups as more participants arrive.
    """
    df = df.copy()

    for column in ["biologicalSex", "eyeColor", "colorBlindness", "Nationality"]:
        if column in df.columns:
            df[column] = canonicalise_text_column(df[column])

    # A participant is identified by uuid AND source file: the same person
    # can have more than one file, and grouping on source_file alone would
    # count them twice.
    if "participant_uuid" in df.columns and "source_file" in df.columns:
        df["participant_id"] = (
            df["participant_uuid"].astype(str) + "__" + df["source_file"].astype(str)
        )
    elif "source_file" in df.columns:
        df["participant_id"] = df["source_file"].astype(str)

    # age_group was referenced by the analysis but never created, so every
    # age comparison silently returned nothing. Build it here.
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
        ).astype(object)

    return df


# ============================================================
# AGGREGATION
# ============================================================

DEMOGRAPHIC_COLUMNS = [
    "participant_uuid",
    "source_file",
    "age",
    "age_group",
    "biologicalSex",
    "eyeColor",
    "colorBlindness",
    "Nationality",
    "Device_Model",
    "Operating_System",
]


def build_attempt_level_df(
    long_df: pd.DataFrame,
    exclude_nonresponse: bool = False,
) -> pd.DataFrame:
    """
    One row per participant / level / attempt, with all three outcome
    measures plus the per-axis outcomes.

    Because each sublevel probes a known axis, the axis outcomes are read
    straight off the axis label rather than reconstructed from Lab values.
    """
    needed = ["participant_id", "level_number", "attempt_number", "axis_label"]

    if not require_columns(long_df, needed):
        return pd.DataFrame()

    df = long_df.copy()

    if exclude_nonresponse and "sublevel_is_nonresponse" in df.columns:
        df = df[~df["sublevel_is_nonresponse"]]

    if df.empty:
        return pd.DataFrame()

    group_cols = ["participant_id", "level_number", "attempt_number"]

    agg_rules: dict[str, Any] = {
        "mean_deltaE2000": ("deltaE2000", "mean"),
        "mean_deltaE76": ("deltaE76", "mean"),
        "mean_response_magnitude": ("response_magnitude", "mean"),
        "n_sublevels_used": ("deltaE2000", "count"),
        "whole_level_duration_ms": ("whole_level_duration_ms", "first"),
        "base_color": ("base_color", "first"),
    }

    if "sublevel_is_nonresponse" in df.columns:
        agg_rules["n_nonresponse_sublevels"] = ("sublevel_is_nonresponse", "sum")

    for column in DEMOGRAPHIC_COLUMNS:
        if column in df.columns:
            agg_rules[column] = (column, "first")

    attempt_df = df.groupby(group_cols, as_index=False, dropna=False).agg(**agg_rules)

    # Per-axis error: mean deltaE2000 on each signed axis.
    axis_pivot = df.pivot_table(
        index=group_cols,
        columns="axis_label",
        values="deltaE2000",
        aggfunc="mean",
    )

    for axis in AXIS_LABELS_IN_ORDER:
        if axis not in axis_pivot.columns:
            axis_pivot[axis] = np.nan

    # Imprecision on an axis, ignoring direction.
    for axis in ["L", "a", "b"]:
        axis_pivot[f"axis_error_{axis}"] = axis_pivot[[f"+{axis}", f"-{axis}"]].mean(
            axis=1
        )

    # Signed bias: positive means the participant tolerated more error on
    # the + end of the axis than the - end.
    for axis in ["L", "a", "b"]:
        axis_pivot[f"bias_{axis}"] = (
            axis_pivot[f"+{axis}"] - axis_pivot[f"-{axis}"]
        )

    keep = AXIS_OUTCOMES + AXIS_LABELS_IN_ORDER
    axis_pivot = axis_pivot[[c for c in keep if c in axis_pivot.columns]]
    axis_pivot.columns = [
        c if c in AXIS_OUTCOMES else f"axis_raw_{c}" for c in axis_pivot.columns
    ]

    return attempt_df.merge(axis_pivot.reset_index(), on=group_cols, how="left")


OUTCOME_COLUMNS_FOR_AGGREGATION = (
    [PRIMARY_OUTCOME]
    + SECONDARY_OUTCOMES
    + AXIS_OUTCOMES
    + [f"axis_raw_{a}" for a in AXIS_LABELS_IN_ORDER]
)


def build_participant_level_df(attempt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average repeated attempts so each participant has one row per level.

    This stops a participant who replayed a level from carrying more
    weight than one who played it once.
    """
    if attempt_df.empty:
        return pd.DataFrame()

    group_cols = ["participant_id", "level_number"]

    agg_rules: dict[str, Any] = {
        c: "mean" for c in OUTCOME_COLUMNS_FOR_AGGREGATION if c in attempt_df.columns
    }

    for column in ["whole_level_duration_ms", "n_nonresponse_sublevels", "n_sublevels_used"]:
        if column in attempt_df.columns:
            agg_rules[column] = "mean"

    agg_rules["attempt_number"] = "count"

    for column in DEMOGRAPHIC_COLUMNS + ["base_color"]:
        if column in attempt_df.columns:
            agg_rules[column] = "first"

    out = attempt_df.groupby(group_cols, as_index=False, dropna=False).agg(agg_rules)

    return out.rename(columns={"attempt_number": "n_attempts"})


def build_overall_df(participant_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per participant, averaging across every level that
    participant completed, so each participant counts exactly once.

    NOTE for Family B: participants who completed more levels have a less
    noisy mean, so between-participant variance is partly a function of how
    many levels each played. n_levels_completed is carried through so this
    can be checked or controlled.
    """
    if participant_level_df.empty:
        return pd.DataFrame()

    agg_rules: dict[str, Any] = {
        c: "mean"
        for c in OUTCOME_COLUMNS_FOR_AGGREGATION
        if c in participant_level_df.columns
    }

    # Counts are summed across levels; durations are averaged.
    if "whole_level_duration_ms" in participant_level_df.columns:
        agg_rules["whole_level_duration_ms"] = "mean"

    for column in ["n_nonresponse_sublevels", "n_sublevels_used"]:
        if column in participant_level_df.columns:
            agg_rules[column] = "sum"

    agg_rules["level_number"] = "count"

    for column in DEMOGRAPHIC_COLUMNS:
        if column in participant_level_df.columns:
            agg_rules[column] = "first"

    out = participant_level_df.groupby(
        "participant_id", as_index=False, dropna=False
    ).agg(agg_rules)

    out = out.rename(columns={"level_number": "n_levels_completed"})

    if {"n_nonresponse_sublevels", "n_sublevels_used"} <= set(out.columns):
        total = out["n_nonresponse_sublevels"] + out["n_sublevels_used"]
        out["nonresponse_rate"] = np.where(
            total > 0, out["n_nonresponse_sublevels"] / total, np.nan
        )

    return out


# ============================================================
# EFFECT SIZES
# ============================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d with a pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return np.nan

    return float((np.mean(group1) - np.mean(group2)) / pooled_sd)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d corrected for small-sample bias."""
    d = cohens_d(group1, group2)

    if np.isnan(d):
        return np.nan

    n1, n2 = len(group1), len(group2)
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))

    return float(d * correction)


def rank_biserial_from_mannwhitney(
    u_statistic: float,
    n1: int,
    n2: int,
) -> float:
    """
    Rank-biserial correlation, on a -1 to 1 scale.

    Signed so that a POSITIVE value means group 1 tends to score higher,
    matching the sign convention of Cohen's d and Hedges' g. The opposite
    convention makes the parametric and non-parametric effect sizes in the
    same row appear to contradict each other.
    """
    if n1 == 0 or n2 == 0:
        return np.nan

    return float(2 * u_statistic / (n1 * n2) - 1)


def eta_and_omega_squared(groups: list[np.ndarray]) -> tuple[float, float]:
    """Eta squared and omega squared, the effect sizes paired with ANOVA."""
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    n_total = len(all_values)
    k = len(groups)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    ss_total = ss_between + ss_within

    if ss_total == 0:
        return np.nan, np.nan

    eta_sq = ss_between / ss_total

    df_between = k - 1
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    if np.isnan(ms_within) or (ss_total + ms_within) == 0:
        omega_sq = np.nan
    else:
        omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    return float(eta_sq), float(omega_sq)


def epsilon_squared_from_kruskal(h_statistic: float, n_total: int) -> float:
    """
    Epsilon squared, the effect size paired with Kruskal-Wallis:
        epsilon^2 = H / (n - 1)

    The previous implementation used (H - k + 1) / (n - k), which is
    eta-squared-based-on-H, a different statistic under the same name.
    """
    if n_total <= 1:
        return np.nan

    return float(h_statistic / (n_total - 1))


def welch_mean_difference_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Mean difference (group1 - group2) with a Welch confidence interval."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    diff = float(m1 - m2)
    se = np.sqrt(v1 / n1 + v2 / n2)

    if se == 0:
        return diff, np.nan, np.nan

    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)

    return diff, float(diff - t_crit * se), float(diff + t_crit * se)


def dunn_posthoc(
    groups: list[np.ndarray],
    names: list[str],
) -> str:
    """
    Dunn's post-hoc test for Kruskal-Wallis, with Benjamini-Hochberg
    correction across the pairwise comparisons.

    Kruskal-Wallis only says that the groups are not all alike; without a
    post-hoc test a significant result cannot be attributed to any pair.
    """
    all_values = np.concatenate(groups)
    n_total = len(all_values)

    if n_total < 3 or len(groups) < 3:
        return ""

    ranks = scipy_stats.rankdata(all_values)

    mean_ranks, sizes, offset = [], [], 0

    for g in groups:
        mean_ranks.append(float(np.mean(ranks[offset : offset + len(g)])))
        sizes.append(len(g))
        offset += len(g)

    # Tie correction.
    _, counts = np.unique(all_values, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    tie_term = tie_sum / (12 * (n_total - 1)) if n_total > 1 else 0.0

    pairs, p_values = [], []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sigma = np.sqrt(
                ((n_total * (n_total + 1) / 12) - tie_term)
                * (1 / sizes[i] + 1 / sizes[j])
            )

            if sigma == 0:
                continue

            z = (mean_ranks[i] - mean_ranks[j]) / sigma
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

            pairs.append((names[i], names[j], float(z)))
            p_values.append(float(p))

    if not pairs:
        return ""

    corrected = benjamini_hochberg(p_values)

    return "; ".join(
        f"{a} vs {b}: z={z:.2f}, p_adj={p:.4f}"
        for (a, b, z), p in zip(pairs, corrected)
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.

    NaN p-values are passed through as NaN rather than being ranked, since
    sorting them would silently corrupt the ordering of the real ones.
    """
    n_real = sum(1 for p in p_values if p is not None and not np.isnan(p))

    if n_real == 0:
        return [np.nan] * len(p_values)

    indexed = [
        (i, p)
        for i, p in enumerate(p_values)
        if p is not None and not np.isnan(p)
    ]
    ranked = sorted(indexed, key=lambda pair: pair[1])

    corrected: list[float] = [np.nan] * len(p_values)
    running_min = 1.0

    for rank in range(n_real, 0, -1):
        idx, p = ranked[rank - 1]
        value = min(p * n_real / rank, 1.0)
        running_min = min(running_min, value)
        corrected[idx] = running_min

    return corrected


def minimum_detectable_effect_two_groups(n1: int, n2: int) -> float | None:
    """Smallest Cohen's d detectable at POWER_TARGET given these group sizes."""
    if not STATSMODELS_AVAILABLE or n1 < 2 or n2 < 2:
        return None

    try:
        return float(
            TTestIndPower().solve_power(
                effect_size=None,
                nobs1=n1,
                ratio=n2 / n1,
                alpha=ALPHA,
                power=POWER_TARGET,
                alternative="two-sided",
            )
        )
    except Exception:
        return None


def minimum_detectable_effect_anova(n_total: int, k_groups: int) -> float | None:
    """Smallest Cohen's f detectable at POWER_TARGET given this total n."""
    if not STATSMODELS_AVAILABLE or n_total <= k_groups:
        return None

    try:
        return float(
            FTestAnovaPower().solve_power(
                effect_size=None,
                nobs=n_total,
                alpha=ALPHA,
                power=POWER_TARGET,
                k_groups=k_groups,
            )
        )
    except Exception:
        return None


# ============================================================
# GROUP COMPARISON
# ============================================================

def prepare_groups(
    df: pd.DataFrame,
    group_column: str,
    outcome_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], int, str] | None:
    """
    Drop non-answers, drop groups below MIN_GROUP_SIZE and return what is
    left. Shared by every family so exclusion rules cannot drift apart.
    """
    if group_column not in df.columns or outcome_column not in df.columns:
        return None

    working_df = df[[group_column, outcome_column]].dropna()

    if working_df.empty:
        return None

    excluded_values = [
        v.strip().lower() for v in EXCLUDED_GROUP_VALUES.get(group_column, [])
    ]

    nonanswer_mask = (
        working_df[group_column].astype(str).str.strip().str.lower().isin(excluded_values)
    )
    n_excluded = int(nonanswer_mask.sum())
    working_df = working_df[~nonanswer_mask]

    if working_df.empty:
        return None

    group_sizes = working_df.groupby(group_column, dropna=False)[outcome_column].count()

    usable = group_sizes[group_sizes >= MIN_GROUP_SIZE].index.tolist()
    small = group_sizes[group_sizes < MIN_GROUP_SIZE].index.tolist()

    groups_summary = ", ".join(f"{n} (n={c})" for n, c in group_sizes.items())

    return working_df, usable, small, n_excluded, groups_summary


def describe_groups(
    filtered_df: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    usable_groups: list[str],
) -> dict[str, Any]:
    """
    Per-group descriptives.

    Without these, a results table gives a p-value and an effect size
    whose SIGN depends on the alphabetical order of the category names, so
    it is impossible to say which group actually did better.
    """
    out: dict[str, Any] = {}
    parts = []

    for name in usable_groups:
        values = filtered_df.loc[
            filtered_df[group_column] == name, outcome_column
        ].values
        parts.append(
            f"{name}: n={len(values)}, M={np.mean(values):.3f}, "
            f"SD={np.std(values, ddof=1):.3f}, Mdn={np.median(values):.3f}"
        )

    out["group_descriptives"] = " | ".join(parts)

    if len(usable_groups) == 2:
        g1 = filtered_df.loc[
            filtered_df[group_column] == usable_groups[0], outcome_column
        ].values
        g2 = filtered_df.loc[
            filtered_df[group_column] == usable_groups[1], outcome_column
        ].values

        out.update(
            {
                "group1_name": usable_groups[0],
                "group2_name": usable_groups[1],
                "group1_n": len(g1),
                "group2_n": len(g2),
                "group1_mean": float(np.mean(g1)),
                "group2_mean": float(np.mean(g2)),
                "group1_sd": float(np.std(g1, ddof=1)),
                "group2_sd": float(np.std(g2, ddof=1)),
                "group1_median": float(np.median(g1)),
                "group2_median": float(np.median(g2)),
            }
        )

        diff, lo, hi = welch_mean_difference_ci(g1, g2)
        out["mean_difference_group1_minus_group2"] = diff
        out["mean_difference_ci_low"] = lo
        out["mean_difference_ci_high"] = hi
        out["direction"] = (
            f"{usable_groups[0]} > {usable_groups[1]}"
            if diff > 0
            else f"{usable_groups[1]} > {usable_groups[0]}"
        )

    return out


def compare_groups(
    df: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    scope_label: str,
) -> dict[str, Any] | None:
    """
    Compare one grouping variable against one outcome within one scope.

    Two groups follow the reference paper's decision tree; three or more
    use Kruskal-Wallis with Dunn's post-hoc. The alternatives are computed
    regardless and reported alongside.
    """
    prepared = prepare_groups(df, group_column, outcome_column)

    if prepared is None:
        return None

    working_df, usable_groups, small_groups, n_excluded, groups_summary = prepared

    is_primary = (
        scope_label in PRIMARY_SCOPES
        and group_column in PRIMARY_GROUP_VARIABLES
        and outcome_column in PRIMARY_OUTCOMES
    )

    base: dict[str, Any] = {
        "scope": scope_label,
        "group_variable": group_column,
        "outcome": outcome_column,
        "analysis_role": "primary" if is_primary else "exploratory",
        "groups": groups_summary,
        "n_groups_total": len(working_df[group_column].unique()),
        "n_groups_usable": len(usable_groups),
        "n_excluded_nonanswer_rows": n_excluded,
        "excluded_small_groups": (
            ", ".join(str(g) for g in small_groups) if small_groups else "none"
        ),
    }

    if len(usable_groups) < 2:
        base["status"] = "skipped_not_enough_groups"
        base["note"] = (
            f"Fewer than 2 groups with at least {MIN_GROUP_SIZE} participants."
        )
        return base

    filtered_df = working_df[working_df[group_column].isin(usable_groups)]

    group_arrays = [
        filtered_df.loc[filtered_df[group_column] == name, outcome_column].values
        for name in usable_groups
    ]
    n_total = sum(len(a) for a in group_arrays)

    base["n_total_usable"] = n_total
    base["status"] = "tested"
    base.update(describe_groups(filtered_df, group_column, outcome_column, usable_groups))

    # ----- assumption checks -----
    shapiro_p_values = []

    for arr in group_arrays:
        if len(arr) >= 3 and np.std(arr) > 0:
            try:
                shapiro_p_values.append(float(scipy_stats.shapiro(arr)[1]))
            except Exception:
                pass

    # bool() is not decoration: scipy returns numpy bools, and
    # np.bool_(False) is False evaluates to False, so an "is False" test
    # against a numpy bool never fires and the decision tree would
    # silently fall through to the wrong branch.
    normality_ok = (
        bool(all(p > ALPHA for p in shapiro_p_values)) if shapiro_p_values else None
    )

    try:
        levene_stat, levene_p = scipy_stats.levene(*group_arrays)
        equal_variance_ok = bool(levene_p > ALPHA)
    except Exception:
        levene_stat, levene_p, equal_variance_ok = np.nan, np.nan, None

    base.update(
        {
            "shapiro_min_p_value": min(shapiro_p_values) if shapiro_p_values else np.nan,
            "normality_ok": normality_ok,
            "levene_statistic": levene_stat,
            "levene_p_value": levene_p,
            "equal_variance_ok": equal_variance_ok,
        }
    )

    # --------------------------------------------------------
    # Two groups
    # --------------------------------------------------------
    if len(usable_groups) == 2:
        g1, g2 = group_arrays
        n1, n2 = len(g1), len(g2)

        student_t, student_p = scipy_stats.ttest_ind(g1, g2, equal_var=True)
        welch_t, welch_p = scipy_stats.ttest_ind(g1, g2, equal_var=False)

        # method='asymptotic' with tie correction: the outcome is heavily
        # quantised, so the exact test is not valid.
        u_stat, u_p = scipy_stats.mannwhitneyu(
            g1, g2, alternative="two-sided", method="asymptotic"
        )

        d = cohens_d(g1, g2)
        g = hedges_g(g1, g2)
        rbc = rank_biserial_from_mannwhitney(u_stat, n1, n2)

        base.update(
            {
                "test_type": "two_group",
                "student_t_statistic": float(student_t),
                "student_p_value": float(student_p),
                "cohens_d": d,
                "welch_t_statistic": float(welch_t),
                "welch_p_value": float(welch_p),
                "hedges_g": g,
                "mannwhitney_u_statistic": float(u_stat),
                "mannwhitney_p_value": float(u_p),
                "rank_biserial_r": rbc,
            }
        )

        # The reference paper's decision tree.
        if min(n1, n2) < SMALL_SAMPLE_THRESHOLD:
            selected, reason = (
                "Mann-Whitney U",
                f"n={min(n1, n2)} < {SMALL_SAMPLE_THRESHOLD} in one group",
            )
        elif normality_ok is False:
            selected, reason = "Mann-Whitney U", "Shapiro rejected normality"
        elif equal_variance_ok is False:
            selected, reason = "Welch's t-test", "normal, Levene rejected equal variances"
        else:
            selected, reason = "Independent t-test", "normal with equal variances"

        if selected == "Mann-Whitney U":
            primary_stat, primary_p = float(u_stat), float(u_p)
            effect_name, effect = "rank-biserial r", rbc
        elif selected == "Welch's t-test":
            primary_stat, primary_p = float(welch_t), float(welch_p)
            effect_name, effect = "Hedges' g", g
        else:
            primary_stat, primary_p = float(student_t), float(student_p)
            effect_name, effect = "Cohen's d", d

        mde = minimum_detectable_effect_two_groups(n1, n2)

        base.update(
            {
                "primary_test": selected,
                "test_selected_by": reason,
                "primary_statistic": primary_stat,
                "primary_p_value": primary_p,
                "primary_effect_size_name": effect_name,
                "primary_effect_size": effect,
                "min_detectable_effect_name": "Cohen's d" if mde else None,
                "min_detectable_effect": mde,
            }
        )

    # --------------------------------------------------------
    # Three or more groups
    # --------------------------------------------------------
    else:
        try:
            f_stat, f_p = scipy_stats.f_oneway(*group_arrays)
        except Exception:
            f_stat, f_p = np.nan, np.nan

        eta_sq, omega_sq = eta_and_omega_squared(group_arrays)

        try:
            h_stat, h_p = scipy_stats.kruskal(*group_arrays)
        except Exception:
            h_stat, h_p = np.nan, np.nan

        epsilon_sq = epsilon_squared_from_kruskal(h_stat, n_total)
        mde = minimum_detectable_effect_anova(n_total, len(usable_groups))

        base.update(
            {
                "test_type": "multi_group",
                "anova_f_statistic": float(f_stat),
                "anova_p_value": float(f_p),
                "omega_squared": omega_sq,
                "eta_squared": eta_sq,
                "kruskal_h_statistic": float(h_stat),
                "kruskal_p_value": float(h_p),
                "epsilon_squared": epsilon_sq,
                "primary_test": "Kruskal-Wallis",
                "test_selected_by": "more than two groups",
                "primary_statistic": float(h_stat),
                "primary_p_value": float(h_p),
                "primary_effect_size_name": "epsilon squared",
                "primary_effect_size": epsilon_sq,
                "dunn_posthoc_bh": dunn_posthoc(
                    group_arrays, [str(g) for g in usable_groups]
                ),
                "min_detectable_effect_name": "Cohen's f" if mde else None,
                "min_detectable_effect": mde,
            }
        )

    return base


def compare_spread(
    df: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    scope_label: str,
) -> dict[str, Any] | None:
    """
    Test whether groups differ in SPREAD rather than in average.

    Levene's test (centred on the median, i.e. Brown-Forsythe) is the
    primary test because Bartlett's is very sensitive to non-normality at
    these sample sizes. Bartlett and the two-group F-test are reported
    alongside.
    """
    prepared = prepare_groups(df, group_column, outcome_column)

    if prepared is None:
        return None

    working_df, usable_groups, small_groups, n_excluded, groups_summary = prepared

    result: dict[str, Any] = {
        "scope": scope_label,
        "group_variable": group_column,
        "outcome": outcome_column,
        "analysis_role": "exploratory",
        "groups": groups_summary,
        "n_groups_usable": len(usable_groups),
        "n_excluded_nonanswer_rows": n_excluded,
        "excluded_small_groups": (
            ", ".join(str(g) for g in small_groups) if small_groups else "none"
        ),
    }

    if len(usable_groups) < 2:
        result["status"] = "skipped_not_enough_groups"
        return result

    filtered_df = working_df[working_df[group_column].isin(usable_groups)]

    group_arrays = [
        filtered_df.loc[filtered_df[group_column] == g, outcome_column].values
        for g in usable_groups
    ]

    result["status"] = "tested"
    result.update(
        describe_groups(filtered_df, group_column, outcome_column, usable_groups)
    )

    try:
        levene_stat, levene_p = scipy_stats.levene(*group_arrays, center="median")
    except Exception:
        levene_stat, levene_p = np.nan, np.nan

    try:
        bartlett_stat, bartlett_p = scipy_stats.bartlett(*group_arrays)
    except Exception:
        bartlett_stat, bartlett_p = np.nan, np.nan

    result.update(
        {
            "primary_test": "Levene's test (median-centred)",
            "primary_statistic": levene_stat,
            "primary_p_value": levene_p,
            "bartlett_statistic": bartlett_stat,
            "bartlett_p_value": bartlett_p,
            "group_sds": ", ".join(
                f"{n}: {np.std(a, ddof=1):.3f}"
                for n, a in zip(usable_groups, group_arrays)
            ),
        }
    )

    if len(usable_groups) == 2:
        var1 = np.var(group_arrays[0], ddof=1)
        var2 = np.var(group_arrays[1], ddof=1)
        n1, n2 = len(group_arrays[0]), len(group_arrays[1])

        if var2 > 0 and var1 > 0:
            if var1 >= var2:
                f_stat, df1, df2 = var1 / var2, n1 - 1, n2 - 1
            else:
                f_stat, df1, df2 = var2 / var1, n2 - 1, n1 - 1

            f_p = min(2 * (1 - scipy_stats.f.cdf(f_stat, df1, df2)), 1.0)
            log_ratio = float(np.log(var1 / var2))
        else:
            f_stat, f_p, log_ratio = np.nan, np.nan, np.nan

        result.update(
            {
                "test_type": "two_group",
                "variance_f_statistic": f_stat,
                "variance_f_p_value": f_p,
                "effect_size_name": "log variance ratio (group1/group2)",
                "effect_size": log_ratio,
            }
        )
    else:
        result["test_type"] = "multi_group"

    return result


# ============================================================
# FAMILY RUNNERS
# ============================================================

def run_family(
    participant_level_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    outcomes: list[str],
    comparison_function: Any,
    family_name: str,
    include_per_level: bool = True,
) -> pd.DataFrame:
    """
    Run one hypothesis family across scopes, outcomes and group variables,
    then apply FDR correction within the primary set and, separately,
    across the whole family.
    """
    print("\n" + "=" * 78)
    print(family_name.upper())
    print("=" * 78)

    results: list[dict[str, Any]] = []

    for outcome in outcomes:
        for group_column in GROUP_VARIABLES:
            result = comparison_function(overall_df, group_column, outcome, "overall")

            if result is not None:
                results.append(result)

    if include_per_level and "level_number" in participant_level_df.columns:
        for level in sorted(participant_level_df["level_number"].dropna().unique()):
            level_df = participant_level_df[
                participant_level_df["level_number"] == level
            ]

            for outcome in outcomes:
                for group_column in GROUP_VARIABLES:
                    result = comparison_function(
                        level_df, group_column, outcome, f"level_{int(level)}"
                    )

                    if result is not None:
                        results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("  No comparisons could be run.")
        return results_df

    tested = results_df["status"] == "tested"

    if tested.any():
        # Correction across the whole family, for transparency.
        family_p = results_df.loc[tested, "primary_p_value"].tolist()
        results_df.loc[tested, "p_fdr_within_family"] = benjamini_hochberg(family_p)

        # Correction within the confirmatory set only. This is the one to
        # report: the exploratory rows would otherwise drag the threshold
        # down for the comparisons the study was designed to make.
        primary_mask = tested & (results_df["analysis_role"] == "primary")

        if primary_mask.any():
            primary_p = results_df.loc[primary_mask, "primary_p_value"].tolist()
            results_df.loc[primary_mask, "p_fdr_within_primary"] = (
                benjamini_hochberg(primary_p)
            )

        results_df["significant_raw"] = results_df["primary_p_value"] < ALPHA
        results_df["significant_after_fdr_primary"] = (
            results_df.get("p_fdr_within_primary", pd.Series(np.nan, index=results_df.index))
            < ALPHA
        )
        results_df["significant_after_fdr_family"] = (
            results_df["p_fdr_within_family"] < ALPHA
        )

    n_tested = int(tested.sum())
    n_primary = int((tested & (results_df["analysis_role"] == "primary")).sum())

    print(f"  Comparisons run:     {n_tested}  (confirmatory: {n_primary})")
    print(f"  Comparisons skipped: {int((~tested).sum())}")

    if n_tested:
        print(
            f"  Significant at raw p < {ALPHA}: "
            f"{int(results_df.loc[tested, 'significant_raw'].sum())} / {n_tested}"
        )

        if n_primary:
            print(
                "  Significant after FDR within confirmatory set: "
                f"{int(results_df['significant_after_fdr_primary'].sum())} / {n_primary}"
            )

    return results_df


def run_age_correlations(
    participant_level_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    """
    Age as a continuous predictor. Binning age into five groups discards
    most of its information and manufactures small groups, so the
    correlation is the better-powered analysis; age_group is kept only for
    comparability with the group-based tests.
    """
    print("\n" + "=" * 78)
    print("AGE AS A CONTINUOUS PREDICTOR")
    print("=" * 78)

    rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, pd.DataFrame]] = [("overall", overall_df)]

    if "level_number" in participant_level_df.columns:
        for level in sorted(participant_level_df["level_number"].dropna().unique()):
            scopes.append(
                (
                    f"level_{int(level)}",
                    participant_level_df[
                        participant_level_df["level_number"] == level
                    ],
                )
            )

    for scope_label, scope_df in scopes:
        if "age" not in scope_df.columns:
            continue

        for outcome in outcomes:
            if outcome not in scope_df.columns:
                continue

            sub = scope_df[["age", outcome]].dropna()

            if len(sub) < 4 or sub["age"].nunique() < 3:
                rows.append(
                    {
                        "scope": scope_label,
                        "outcome": outcome,
                        "n": len(sub),
                        "status": "skipped_not_enough_data",
                    }
                )
                continue

            rho, rho_p = scipy_stats.spearmanr(sub["age"], sub[outcome])
            r, r_p = scipy_stats.pearsonr(sub["age"], sub[outcome])
            slope, intercept = np.polyfit(sub["age"], sub[outcome], 1)

            rows.append(
                {
                    "scope": scope_label,
                    "outcome": outcome,
                    "n": len(sub),
                    "status": "tested",
                    "analysis_role": (
                        "primary"
                        if scope_label == "overall" and outcome == PRIMARY_OUTCOME
                        else "exploratory"
                    ),
                    "spearman_rho": float(rho),
                    "spearman_p_value": float(rho_p),
                    "pearson_r": float(r),
                    "pearson_p_value": float(r_p),
                    "linear_slope_per_year": float(slope),
                    "linear_intercept": float(intercept),
                    "age_min": float(sub["age"].min()),
                    "age_max": float(sub["age"].max()),
                }
            )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        return results_df

    tested = results_df["status"] == "tested"

    if tested.any():
        results_df.loc[tested, "spearman_p_fdr"] = benjamini_hochberg(
            results_df.loc[tested, "spearman_p_value"].tolist()
        )

    print(f"  Correlations run: {int(tested.sum())}")

    return results_df


# ============================================================
# SAVING
# ============================================================

def save_results(
    results_df: pd.DataFrame,
    sheet_name: str,
    filename: str,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Write one results table (plus any extra sheets) to Excel."""
    if results_df.empty and not extra_sheets:
        print(f"\nNo results to save for {filename}.")
        return

    output_folder = project_folder / "excel_files"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / filename

    # Put the columns a reader needs first.
    preferred = [
        "analysis_role",
        "scope",
        "group_variable",
        "outcome",
        "status",
        "primary_test",
        "test_selected_by",
        "primary_p_value",
        "p_fdr_within_primary",
        "p_fdr_within_family",
        "significant_raw",
        "significant_after_fdr_primary",
        "primary_effect_size_name",
        "primary_effect_size",
        "direction",
        "group_descriptives",
        "min_detectable_effect_name",
        "min_detectable_effect",
    ]

    if not results_df.empty:
        ordered = [c for c in preferred if c in results_df.columns]
        ordered += [c for c in results_df.columns if c not in ordered]
        results_df = results_df[ordered]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not results_df.empty:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for name, frame in (extra_sheets or {}).items():
            if not frame.empty:
                frame.to_excel(writer, sheet_name=name, index=False)

        for ws_name in writer.book.sheetnames:
            ws = writer.book[ws_name]
            ws.freeze_panes = "A2"

            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            for col in ws.columns:
                max_len = max(
                    (len(str(c.value)) for c in col[:200] if c.value is not None),
                    default=0,
                )
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(max_len + 2, 10), 45
                )

    print(f"\nSaved results:\n  {output_path}")


# ============================================================
# PLOTS
# ============================================================

def plot_outcome_by_level(
    participant_level_df: pd.DataFrame,
    outcome: str,
    ylabel: str,
    filename: str,
    force_zero_bottom: bool = False,
) -> None:
    """Box plot of one outcome per color level, with participant points."""
    if not require_columns(participant_level_df, ["level_number", outcome]):
        return

    plot_df = participant_level_df[["level_number", outcome]].dropna()

    if plot_df.empty:
        print(f"Skipping {filename}: no usable data.")
        return

    levels_sorted = sorted(plot_df["level_number"].unique())
    data_by_level = [
        plot_df.loc[plot_df["level_number"] == lvl, outcome].values
        for lvl in levels_sorted
    ]

    if all(len(v) == 0 for v in data_by_level):
        print(f"Skipping {filename}: every level is empty.")
        return

    plt.figure(figsize=(10, 6))
    box_parts = boxplot_compat(
        data_by_level,
        [str(int(lvl)) for lvl in levels_sorted],
        showmeans=True,
        zorder=2,
    )

    # scatter_handle was previously referenced in the legend even when no
    # points had been drawn, which raised NameError on a level with no data.
    scatter_handle = None

    for i, lvl in enumerate(levels_sorted, start=1):
        values = data_by_level[i - 1]

        if len(values) == 0:
            continue

        scatter_handle = plt.scatter(
            np.full(len(values), i),
            values,
            alpha=0.5,
            s=20,
            color=LEVEL_COLORS.get(int(lvl), "grey"),
            zorder=1,
        )

    handles = [
        box_parts["medians"][0],
        box_parts["means"][0],
        box_parts["boxes"][0],
        box_parts["fliers"][0],
    ]
    labels = [
        "Median",
        "Mean",
        "IQR (25th-75th percentile)",
        "Outliers (beyond 1.5xIQR)",
    ]

    if scatter_handle is not None:
        handles.append(scatter_handle)
        labels.append("Individual participants")

    plt.legend(handles, labels, loc="best")

    if force_zero_bottom:
        plt.ylim(bottom=0)

    plt.title(f"{ylabel} by color level")
    plt.xlabel("Level number")
    plt.ylabel(ylabel)

    save_current_plot(filename)


def plot_outcome_by_group(
    overall_df: pd.DataFrame,
    participant_level_df: pd.DataFrame,
    outcome: str = PRIMARY_OUTCOME,
) -> None:
    """
    Box plots of the primary outcome by each demographic group.

    The analysis previously produced p-values for group differences with
    no figure showing them, so a reader had no way to see what a
    significant result looked like.
    """
    for group_column in GROUP_VARIABLES:
        prepared = prepare_groups(overall_df, group_column, outcome)

        if prepared is None:
            continue

        working_df, usable_groups, _, _, _ = prepared

        if len(usable_groups) < 2:
            continue

        data = [
            working_df.loc[working_df[group_column] == g, outcome].values
            for g in usable_groups
        ]

        plt.figure(figsize=(max(7, 1.6 * len(usable_groups)), 6))
        boxplot_compat(
            data, [f"{g}\n(n={len(d)})" for g, d in zip(usable_groups, data)],
            showmeans=True, zorder=2,
        )

        for i, values in enumerate(data, start=1):
            plt.scatter(
                np.random.normal(i, 0.04, size=len(values)),
                values,
                alpha=0.6,
                s=22,
                color="grey",
                zorder=1,
            )

        plt.title(f"{outcome} by {group_column} (pooled across levels)")
        plt.xlabel(group_column)
        plt.ylabel(outcome)
        plt.xticks(rotation=0)

        save_current_plot(
            f"08_{outcome}_by_{safe_filename(group_column)}.png",
            subfolder="group_comparisons",
        )

        # Per level, as a small multiple grid.
        if "level_number" not in participant_level_df.columns:
            continue

        levels = sorted(participant_level_df["level_number"].dropna().unique())

        if not levels:
            continue

        n_cols = 4
        n_rows = int(np.ceil(len(levels) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows), squeeze=False
        )

        for ax, level in zip(axes.flat, levels):
            level_df = participant_level_df[
                participant_level_df["level_number"] == level
            ]
            level_prepared = prepare_groups(level_df, group_column, outcome)

            if level_prepared is None:
                ax.set_visible(False)
                continue

            lvl_df, lvl_groups, _, _, _ = level_prepared

            if len(lvl_groups) < 2:
                ax.set_visible(False)
                continue

            lvl_data = [
                lvl_df.loc[lvl_df[group_column] == g, outcome].values
                for g in lvl_groups
            ]
            ax.boxplot(lvl_data, showmeans=True)
            ax.set_xticks(range(1, len(lvl_groups) + 1))
            ax.set_xticklabels(
                [str(g)[:10] for g in lvl_groups], rotation=30, ha="right", fontsize=8
            )
            ax.set_title(
                f"Level {int(level)}",
                fontsize=10,
                color=LEVEL_COLORS.get(int(level), "black"),
            )
            ax.set_ylabel(outcome, fontsize=8)

        for ax in axes.flat[len(levels) :]:
            ax.set_visible(False)

        fig.suptitle(f"{outcome} by {group_column}, per level", fontsize=13)
        fig.tight_layout()

        output_folder = plot_folder / "group_comparisons"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_folder
            / f"08_{outcome}_by_{safe_filename(group_column)}_per_level.png"
        )
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"Saved: {output_path}")


def plot_axis_profile(long_df: pd.DataFrame) -> None:
    """
    Mean error on each of the six axes, overall and per level.

    This is the figure the axis family is about: it shows directly whether
    lightness is harder than hue, and whether errors are asymmetric.
    """
    if not require_columns(long_df, ["axis_label", "deltaE2000", "level_number"]):
        return

    df = long_df.dropna(subset=["axis_label", "deltaE2000"])

    if df.empty:
        return

    order = [a for a in AXIS_LABELS_IN_ORDER if a in df["axis_label"].unique()]

    plt.figure(figsize=(9, 6))
    data = [df.loc[df["axis_label"] == a, "deltaE2000"].values for a in order]
    boxplot_compat(data, order, showmeans=True)
    plt.title("Delta E 2000 by manipulated axis (all levels pooled)")
    plt.xlabel("Axis (game Lab space: L lightness, a and b chromatic)")
    plt.ylabel("Delta E 2000")
    save_current_plot("09_deltaE2000_by_axis.png")

    pivot = df.pivot_table(
        index="level_number", columns="axis_label", values="deltaE2000", aggfunc="mean"
    )
    pivot = pivot[[a for a in order if a in pivot.columns]]

    plt.figure(figsize=(10, 6))

    for axis in pivot.columns:
        plt.plot(pivot.index, pivot[axis], marker="o", label=axis)

    plt.title("Mean Delta E 2000 per axis, by color level")
    plt.xlabel("Level number")
    plt.ylabel("Mean Delta E 2000")
    plt.legend(title="Axis")
    save_current_plot("09_deltaE2000_by_axis_and_level.png")


def plot_nonresponse(long_df: pd.DataFrame, overall_df: pd.DataFrame) -> None:
    """Where non-responses happen, and who produces them."""
    if "sublevel_is_nonresponse" not in long_df.columns:
        return

    by_axis = long_df.groupby("axis_label")["sublevel_is_nonresponse"].mean()
    by_axis = by_axis.reindex(
        [a for a in AXIS_LABELS_IN_ORDER if a in by_axis.index]
    )

    by_level = long_df.groupby("level_number")["sublevel_is_nonresponse"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(labels_from_index(by_axis.index), 100 * by_axis.values, color="indianred")
    axes[0].set_title("Non-response rate by axis")
    axes[0].set_xlabel("Axis")
    axes[0].set_ylabel("Sublevels with no selection [%]")

    axes[1].bar(
        [str(int(v)) for v in by_level.index],
        100 * by_level.values,
        color="indianred",
    )
    axes[1].set_title("Non-response rate by level")
    axes[1].set_xlabel("Level number")
    axes[1].set_ylabel("Sublevels with no selection [%]")

    fig.tight_layout()
    output_path = plot_folder / "10_nonresponse_rate.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_participant_counts_by_group(overall_df: pd.DataFrame) -> None:
    """Participant counts per demographic group."""
    for group_column in ["eyeColor", "age_group", "biologicalSex", "colorBlindness", "Nationality"]:
        if group_column not in overall_df.columns:
            print(f"Skipping {group_column} count plot: column not found.")
            continue

        counts = overall_df[group_column].value_counts(dropna=False)

        if counts.empty:
            continue

        labels = labels_from_index(counts.index)
        colours = [
            "lightgrey" if is_excluded_value(group_column, v) else "steelblue"
            for v in counts.index
        ]

        plt.figure(figsize=(max(8, 1.2 * len(labels)), 6))
        bars = plt.bar(labels, counts.values, color=colours)

        for bar, value in zip(bars, counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.title(
            f"Participant count by {group_column}\n"
            "grey = non-answer, excluded from comparisons"
        )
        plt.xlabel(group_column)
        plt.ylabel("Number of participants")
        plt.xticks(rotation=45, ha="right")

        save_current_plot(
            f"07_participant_count_by_{safe_filename(group_column)}.png",
            subfolder="group_overview",
        )


def plot_age_vs_performance_per_color(participant_level_df: pd.DataFrame) -> None:
    """Age against the primary outcome, one panel per color level."""
    needed = ["age", "level_number", PRIMARY_OUTCOME]

    if not require_columns(participant_level_df, needed):
        return

    plot_df = participant_level_df[needed].dropna()

    if plot_df.empty:
        print("Skipping age plots: no usable age values.")
        return

    levels = sorted(plot_df["level_number"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(levels) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows), squeeze=False
    )

    for ax, level in zip(axes.flat, levels):
        level_df = plot_df[plot_df["level_number"] == level]
        colour = LEVEL_COLORS.get(int(level), "grey")

        ax.scatter(level_df["age"], level_df[PRIMARY_OUTCOME], alpha=0.7, color=colour)

        if level_df["age"].nunique() >= 2 and len(level_df) >= 3:
            slope, intercept = np.polyfit(
                level_df["age"], level_df[PRIMARY_OUTCOME], 1
            )
            x = np.linspace(level_df["age"].min(), level_df["age"].max(), 100)
            ax.plot(x, slope * x + intercept, linewidth=2, color=colour)
            rho, p = scipy_stats.spearmanr(level_df["age"], level_df[PRIMARY_OUTCOME])
            ax.set_title(
                f"Level {int(level)}: rho={rho:.2f}, p={p:.3f}", fontsize=10
            )
        else:
            ax.set_title(f"Level {int(level)}: too little variation", fontsize=10)

        ax.set_xlabel("Age")
        ax.set_ylabel(PRIMARY_OUTCOME, fontsize=8)

    for ax in axes.flat[len(levels) :]:
        ax.set_visible(False)

    fig.suptitle(f"Age vs {PRIMARY_OUTCOME}, per color level", fontsize=13)
    fig.tight_layout()

    output_path = plot_folder / "04_age_vs_performance_per_level.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_attempt_count_by_level(short_df: pd.DataFrame) -> None:
    """Unique, repeated and missing-but-expected level attempts per level."""
    needed = ["participant_id", "level_number"]

    if not require_columns(short_df, needed):
        return

    df = short_df[needed].dropna().copy()
    df["level_number"] = df["level_number"].astype(int)

    levels = list(range(1, MAIN_COLOR_LEVELS + 1))

    unique_df = df.drop_duplicates(["participant_id", "level_number"])

    found = unique_df["level_number"].value_counts().reindex(levels, fill_value=0)
    all_attempts = df["level_number"].value_counts().reindex(levels, fill_value=0)

    missing = pd.Series(0, index=levels)

    for _, person in unique_df.groupby("participant_id"):
        levels_found = sorted(person["level_number"].unique())

        if not levels_found:
            continue

        for level in set(range(1, max(levels_found) + 1)) - set(levels_found):
            if level in missing.index:
                missing.loc[level] += 1

    x = np.arange(len(levels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, found.values, width=0.6, color="darkgreen", label="Unique participants")
    plt.bar(
        x,
        all_attempts.values - found.values,
        width=0.6,
        bottom=found.values,
        color="limegreen",
        label="Repeated attempts",
    )
    plt.bar(
        x,
        missing.values,
        width=0.6,
        bottom=all_attempts.values,
        color="gold",
        label="Missing but expected",
    )

    plt.title("Level attempts per level")
    plt.xlabel("Level number")
    plt.ylabel("Count")
    plt.xticks(x, [str(l) for l in levels])
    plt.legend()

    save_current_plot("01_attempts_by_level.png")


def plot_axis_radar_per_attempt(short_df: pd.DataFrame) -> None:
    """
    Six-axis radar of the response magnitudes for one level attempt.

    The old version drew eight compass points, but two of the eight
    payload slots duplicate two others, so the shape implied eight
    independent directions where there are only six.
    """
    magnitude_columns = [f"magnitude_axis_{a}" for a in AXIS_LABELS_IN_ORDER]

    if not require_columns(short_df, magnitude_columns):
        return

    folder = plot_folder / "axis_radars"
    folder.mkdir(parents=True, exist_ok=True)

    angles = np.linspace(0, 2 * np.pi, len(AXIS_LABELS_IN_ORDER), endpoint=False)

    print(f"\nCreating axis radar plots for {len(short_df)} attempts...")

    for _, row in short_df.iterrows():
        values = [row.get(c) for c in magnitude_columns]

        if any(pd.isna(v) for v in values):
            continue

        closed_angles = np.concatenate([angles, angles[:1]])
        closed_values = np.array(list(values) + [values[0]])

        level_number = int(row["level_number"])
        colour = LEVEL_COLORS.get(level_number, "grey")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.plot(closed_angles, closed_values, color=colour, linewidth=2)
        ax.fill(closed_angles, closed_values, color=colour, alpha=0.3)
        ax.set_xticks(angles)
        ax.set_xticklabels(AXIS_LABELS_IN_ORDER)
        ax.set_ylim(0, max(11.0, float(np.nanmax(closed_values)) * 1.1))

        fig.suptitle("Response magnitude per axis", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Level {level_number}, attempt {int(row['attempt_number'])}", fontsize=10
        )

        filename = (
            f"{safe_filename(str(row.get('participant_uuid')))}"
            f"__level{level_number}_attempt{int(row['attempt_number'])}_axis_radar.png"
        )
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=DIAGNOSTIC_PLOT_DPI)
        plt.close(fig)

    print(f"Saved axis radars in:\n{folder}")


def plot_chosen_colors_heatmap_by_participant(long_df: pd.DataFrame) -> None:
    """
    One heatmap per participant/source file.

    x-axis: color level / attempt
    y-axis: sublevel
    cell:   number of colors chosen in that sublevel

    Shows whether a participant clicked through a level without choosing
    colors. Cells now also carry the axis that sublevel probed.
    """
    needed = [
        "source_file",
        "level_number",
        "attempt_number",
        "sublevel_index",
        "sublevel_chosen_color_count",
    ]

    if not require_columns(long_df, needed):
        return

    plot_df = long_df[needed + ["axis_label"]].copy()
    plot_df["participant_uuid"] = long_df.get("participant_uuid", "missing_uuid")

    for column in ["level_number", "attempt_number", "sublevel_index",
                   "sublevel_chosen_color_count"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    plot_df = plot_df.dropna(
        subset=["source_file", "level_number", "attempt_number", "sublevel_index"]
    )

    for column in ["level_number", "attempt_number", "sublevel_index"]:
        plot_df[column] = plot_df[column].astype(int)

    folder = plot_folder / "chosen_colors_heatmaps"
    folder.mkdir(parents=True, exist_ok=True)

    groups = plot_df.groupby(["participant_uuid", "source_file"], dropna=False)

    print(f"\nCreating chosen-color heatmaps for {len(groups)} participants...")

    for (participant_uuid, source_file), person_df in groups:
        person_df = person_df.copy()
        person_df["level_attempt_label"] = (
            "L" + person_df["level_number"].astype(str)
            + " A" + person_df["attempt_number"].astype(str)
        )

        attempt_labels = (
            person_df[["level_number", "attempt_number", "level_attempt_label"]]
            .drop_duplicates()
            .sort_values(["level_number", "attempt_number"])["level_attempt_label"]
            .tolist()
        )

        heatmap_df = (
            person_df.pivot_table(
                index="sublevel_index",
                columns="level_attempt_label",
                values="sublevel_chosen_color_count",
                aggfunc="first",
            )
            .reindex(
                index=list(range(SUBLEVELS_PER_LEVEL)),
                columns=attempt_labels,
            )
        )

        if heatmap_df.empty:
            continue

        plt.figure(figsize=(max(SUBLEVELS_PER_LEVEL, len(attempt_labels) * 0.8), 6))
        plt.imshow(heatmap_df.values, aspect="auto")
        plt.colorbar(label="Number of colors chosen")

        plt.title(
            "Number of colors chosen per sublevel and color level\n"
            f"Participant: {participant_uuid}"
        )
        plt.xlabel("Color level / attempt")
        plt.ylabel("Sublevel (axis probed)")

        plt.xticks(
            ticks=range(len(attempt_labels)),
            labels=attempt_labels,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            ticks=range(SUBLEVELS_PER_LEVEL),
            labels=[
                f"{i} ({AXIS_LABELS_IN_ORDER[i]})"
                for i in range(SUBLEVELS_PER_LEVEL)
            ],
        )

        for y in range(heatmap_df.shape[0]):
            for x in range(heatmap_df.shape[1]):
                value = heatmap_df.iloc[y, x]
                plt.text(
                    x,
                    y,
                    str(int(value)) if pd.notna(value) else "?",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()
        filename = (
            f"participant_{safe_filename(str(participant_uuid))}"
            f"__{safe_filename(str(source_file))}__chosen_colors_heatmap.png"
        )
        plt.savefig(folder / filename, dpi=DIAGNOSTIC_PLOT_DPI)
        plt.close()

    print(f"Saved chosen-color heatmaps in:\n{folder}")


def plot_each_participant_sublevel_distances(long_df: pd.DataFrame) -> None:
    """
    One plot per participant/source file showing Delta E 2000 for every
    sublevel across all levels. One line per completed level attempt, so
    repeated attempts appear as separate lines.
    """
    needed = [
        "source_file",
        "level_number",
        "attempt_number",
        "sublevel_index",
        "deltaE2000",
    ]

    if not require_columns(long_df, needed):
        return

    plot_df = long_df[needed].copy()
    plot_df["participant_uuid"] = long_df.get("participant_uuid", "missing_uuid")

    if "sublevel_is_nonresponse" in long_df.columns:
        plot_df["sublevel_is_nonresponse"] = long_df["sublevel_is_nonresponse"]
    else:
        plot_df["sublevel_is_nonresponse"] = False

    for column in ["level_number", "attempt_number", "sublevel_index", "deltaE2000"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    plot_df = plot_df.dropna(
        subset=["source_file", "level_number", "attempt_number", "sublevel_index"]
    )

    for column in ["level_number", "attempt_number", "sublevel_index"]:
        plot_df[column] = plot_df[column].astype(int)

    folder = plot_folder / "participant_sublevel_plots"
    folder.mkdir(parents=True, exist_ok=True)

    groups = plot_df.groupby(["participant_uuid", "source_file"], dropna=False)

    print(f"\nCreating participant sublevel plots for {len(groups)} participants...")

    for (participant_uuid, source_file), person_df in groups:
        person_df = person_df.sort_values(
            ["level_number", "attempt_number", "sublevel_index"]
        )

        plt.figure(figsize=(14, 6))

        for (level_number, attempt_number), attempt_df in person_df.groupby(
            ["level_number", "attempt_number"], dropna=False
        ):
            attempt_df = attempt_df.sort_values("sublevel_index")

            complete = pd.DataFrame(
                {"sublevel_index": list(range(SUBLEVELS_PER_LEVEL))}
            ).merge(attempt_df, on="sublevel_index", how="left")

            x = (
                (level_number - 1) * SUBLEVELS_PER_LEVEL
                + complete["sublevel_index"]
                + 1
            )

            line = plt.plot(
                x,
                complete["deltaE2000"],
                marker="o",
                linewidth=1.5,
                label=f"Level {level_number}, attempt {attempt_number}",
            )[0]

            # Mark sublevels where nothing was submitted: the game scores
            # these as maximum error, so they are not real judgements.
            nonresponse = complete["sublevel_is_nonresponse"].fillna(False).astype(bool)

            if nonresponse.any():
                plt.scatter(
                    x[nonresponse],
                    complete.loc[nonresponse, "deltaE2000"],
                    marker="x",
                    s=90,
                    color=line.get_color(),
                    zorder=5,
                )

        for boundary in range(
            SUBLEVELS_PER_LEVEL,
            MAIN_COLOR_LEVELS * SUBLEVELS_PER_LEVEL,
            SUBLEVELS_PER_LEVEL,
        ):
            plt.axvline(boundary + 0.5, linestyle="--", linewidth=0.8, alpha=0.4)

        midpoints = [
            ((level - 1) * SUBLEVELS_PER_LEVEL) + ((SUBLEVELS_PER_LEVEL + 1) / 2)
            for level in range(1, MAIN_COLOR_LEVELS + 1)
        ]

        plt.xticks(midpoints, [f"Level {l}" for l in range(1, MAIN_COLOR_LEVELS + 1)])
        plt.title(
            "Participant sublevel distances  (x = no selection submitted)\n"
            f"Participant: {participant_uuid}"
        )
        plt.xlabel("Color level")
        plt.ylabel("Delta E 2000")
        plt.legend(
            title="Completed level attempts",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
        )
        plt.tight_layout()

        filename = (
            f"participant_{safe_filename(str(participant_uuid))}"
            f"__{safe_filename(str(source_file))}_sublevel_deltaE2000.png"
        )
        plt.savefig(folder / filename, dpi=DIAGNOSTIC_PLOT_DPI, bbox_inches="tight")
        plt.close()

    print(f"Saved participant sublevel plots in:\n{folder}")


# Fixed compass angles, in degrees clockwise from the top.
DIRECTION_AXIS_ANGLES_DEG = {
    "U+": 90,
    "L1+": 45,
    "V+": 0,
    "L2+": 315,
    "U-": 270,
    "L1-": 225,
    "V-": 180,
    "L2-": 135,
}


def plot_compass_spiderweb_by_participant_level(short_df: pd.DataFrame) -> None:
    """
    One spiderweb (radar) plot per participant/level/attempt, showing the
    eight compass-direction magnitudes logged in the finalcolors payload.

    Note when reading these: compass positions L2+ and L2- duplicate L1+
    and L1-, because both diagonals map to the lightness axis. The shape
    therefore has eight points but only six independent values. The
    six-axis version in plot_axis_radar_per_attempt shows the same data
    without the duplication.
    """
    magnitude_columns = [
        c for c in short_df.columns
        if c.startswith("dir_") and c.endswith("_magnitude")
    ]

    if not magnitude_columns:
        print(
            "Skipping compass spiderwebs: no dir_*_magnitude columns found. "
            "Re-run the data handler."
        )
        return

    needed = ["source_file", "level_number", "attempt_number"]

    if not require_columns(short_df, needed):
        return

    plot_df = short_df[needed + magnitude_columns].copy()
    plot_df["participant_uuid"] = short_df.get("participant_uuid", "missing_uuid")

    for column in ["level_number", "attempt_number"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    plot_df = plot_df.dropna(subset=needed)
    plot_df["level_number"] = plot_df["level_number"].astype(int)
    plot_df["attempt_number"] = plot_df["attempt_number"].astype(int)

    # "dir_2_V+_magnitude" -> "V+"
    axis_by_column = {
        col: "_".join(col.split("_")[2:-1]) for col in magnitude_columns
    }

    ordered_columns = sorted(
        magnitude_columns,
        key=lambda col: -DIRECTION_AXIS_ANGLES_DEG.get(axis_by_column[col], 0),
    )
    axis_labels = [axis_by_column[c] for c in ordered_columns]
    axis_angles_rad = [
        np.deg2rad(DIRECTION_AXIS_ANGLES_DEG.get(label, 0)) for label in axis_labels
    ]

    folder = plot_folder / "compass_spiderwebs"
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating compass spiderweb plots for {len(plot_df)} attempts...")

    for _, row in plot_df.iterrows():
        magnitudes = [row.get(c) if pd.notna(row.get(c)) else 0 for c in ordered_columns]

        angles_plot = axis_angles_rad + [axis_angles_rad[0]]
        magnitudes_plot = list(magnitudes) + [magnitudes[0]]

        level_number = int(row["level_number"])
        level_color = LEVEL_COLORS.get(level_number, "grey")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, max(1.0, float(np.nanmax(magnitudes_plot)) * 1.1))
        ax.set_rlabel_position(90)

        ax.plot(angles_plot, magnitudes_plot, color=level_color, linewidth=2)
        ax.fill(angles_plot, magnitudes_plot, color=level_color, alpha=0.3)
        ax.set_xticks(axis_angles_rad)
        ax.set_xticklabels(axis_labels)

        fig.suptitle("Color Magnitudes", fontsize=16, fontweight="bold")
        ax.set_title(
            f"Level {level_number}, attempt {int(row['attempt_number'])}", fontsize=10
        )

        participant_folder = (
            folder / f"participant_{safe_filename(str(row['participant_uuid']))}"
        )
        participant_folder.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{safe_filename(str(row['source_file']))}"
            f"__level{level_number}_attempt{int(row['attempt_number'])}"
            f"__compass_spiderweb.png"
        )

        fig.tight_layout()
        fig.savefig(participant_folder / filename, dpi=DIAGNOSTIC_PLOT_DPI)
        plt.close(fig)

    print(f"Saved compass spiderwebs in:\n{folder}")


# ============================================================
# DATA OVERVIEW
# ============================================================

def create_data_overview(
    long_df: pd.DataFrame,
    participant_level_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    """Print a descriptive overview of the cleaned dataset."""
    print("\n" + "=" * 78)
    print("DATA OVERVIEW")
    print("=" * 78)

    print("\nDataset size:")
    print(f"  Participants (uuid + file):   {overall_df['participant_id'].nunique()}")
    print(f"  Participant x level rows:     {len(participant_level_df)}")
    print(f"  Sublevel rows:                {len(long_df)}")

    if "n_attempts" in participant_level_df.columns:
        repeated = int((participant_level_df["n_attempts"] > 1).sum())
        print(f"  Participant/level with repeats: {repeated}")

    if "sublevel_is_nonresponse" in long_df.columns:
        n = int(long_df["sublevel_is_nonresponse"].sum())
        print(
            f"  Non-response sublevels:       {n} "
            f"({100 * n / max(len(long_df), 1):.1f}%)"
        )

    print("\n" + "-" * 78)
    print("Overview by color level")
    print("-" * 78)

    overview = (
        participant_level_df.groupby(["level_number", "base_color"], dropna=False)
        .agg(
            participants=("participant_id", "nunique"),
            mean_deltaE2000=(PRIMARY_OUTCOME, "mean"),
            median_deltaE2000=(PRIMARY_OUTCOME, "median"),
            sd_deltaE2000=(PRIMARY_OUTCOME, "std"),
            mean_deltaE76=("mean_deltaE76", "mean"),
            mean_magnitude=("mean_response_magnitude", "mean"),
        )
        .reset_index()
        .sort_values("level_number")
    )

    print(overview.to_string(index=False))

    print("\nRanked hardest first, by each measure (they can disagree):")

    for measure in ["mean_deltaE2000", "mean_deltaE76", "mean_magnitude"]:
        order = overview.sort_values(measure, ascending=False)["level_number"]
        print(f"  {measure:18s}: {[int(l) for l in order]}")

    print("\n" + "-" * 78)
    print("Group sizes (one count per participant)")
    print("-" * 78)

    for group_column in GROUP_VARIABLES:
        if group_column not in overall_df.columns:
            continue

        print(f"\n{group_column}:")

        counts = overall_df[group_column].value_counts(dropna=False)

        for value, count in counts.items():
            label = MISSING_LABEL if pd.isna(value) else str(value)
            marker = (
                "   <- non-answer, excluded from comparisons"
                if is_excluded_value(group_column, value)
                else ""
            )
            print(f"  {label:<24s} {count:>4d}{marker}")

        n_usable = sum(
            c
            for v, c in counts.items()
            if not is_excluded_value(group_column, v)
        )
        print(f"  {'-> usable for comparisons':<24s} {n_usable:>4d}")

    if "age" in overall_df.columns and overall_df["age"].notna().any():
        print("\n" + "-" * 78)
        print("Age")
        print("-" * 78)
        print(overall_df["age"].describe().to_string())


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    long_df = load_long_data()
    short_df = load_short_data()

    attempt_df = build_attempt_level_df(long_df, exclude_nonresponse=False)
    participant_level_df = build_participant_level_df(attempt_df)
    overall_df = build_overall_df(participant_level_df)

    # Sensitivity version: same pipeline with non-response sublevels removed.
    attempt_df_sens = build_attempt_level_df(long_df, exclude_nonresponse=True)
    participant_level_sens = build_participant_level_df(attempt_df_sens)
    overall_sens = build_overall_df(participant_level_sens)

    print("\nLoaded data:")
    print(f"  Sublevel rows:                {len(long_df)}")
    print(f"  Level attempts:               {len(attempt_df)}")
    print(f"  Participant x level rows:     {len(participant_level_df)}")
    print(f"  Participants:                 {len(overall_df)}")

    create_data_overview(long_df, participant_level_df, overall_df)

    # ----- plots -----
    plot_attempt_count_by_level(short_df)
    plot_outcome_by_level(
        participant_level_df, PRIMARY_OUTCOME, "Mean Delta E 2000", "05_deltaE2000_by_level.png"
    )
    plot_outcome_by_level(
        participant_level_df,
        "mean_response_magnitude",
        "Mean response magnitude (game units)",
        "05_magnitude_by_level.png",
    )

    if "whole_level_duration_ms" in participant_level_df.columns:
        duration_df = participant_level_df.copy()
        duration_df["whole_level_duration_s"] = (
            duration_df["whole_level_duration_ms"] / 1000
        )
        plot_outcome_by_level(
            duration_df,
            "whole_level_duration_s",
            "Whole-level duration [s]",
            "06_duration_by_level.png",
            force_zero_bottom=True,
        )

    plot_participant_counts_by_group(overall_df)
    plot_age_vs_performance_per_color(participant_level_df)
    plot_outcome_by_group(overall_df, participant_level_df, PRIMARY_OUTCOME)
    plot_axis_profile(long_df)
    plot_nonresponse(long_df, overall_df)

    if MAKE_PARTICIPANT_DIAGNOSTIC_PLOTS:
        plot_chosen_colors_heatmap_by_participant(long_df)
        plot_each_participant_sublevel_distances(long_df)
        plot_compass_spiderweb_by_participant_level(short_df)
        plot_axis_radar_per_attempt(short_df)
    else:
        print(
            "\nSkipping per-participant diagnostic plots "
            "(MAKE_PARTICIPANT_DIAGNOSTIC_PLOTS is False)."
        )

    # ----- Family A: means -----
    family_a = run_family(
        participant_level_df,
        overall_df,
        [PRIMARY_OUTCOME] + SECONDARY_OUTCOMES,
        compare_groups,
        "Family A: mean comparison",
    )

    family_a_sens = run_family(
        participant_level_sens,
        overall_sens,
        [PRIMARY_OUTCOME],
        compare_groups,
        "Family A sensitivity: non-response sublevels excluded",
    )

    age_results = run_age_correlations(
        participant_level_df, overall_df, [PRIMARY_OUTCOME] + SECONDARY_OUTCOMES
    )

    save_results(
        family_a,
        "group_comparisons",
        "Delta2000_analysis.xlsx",
        extra_sheets={
            "sensitivity_no_nonresponse": family_a_sens,
            "age_continuous": age_results,
        },
    )

    # ----- Family B: spread -----
    family_b = run_family(
        participant_level_df,
        overall_df,
        [PRIMARY_OUTCOME],
        compare_spread,
        "Family B: spread comparison",
    )
    save_results(family_b, "spread_comparisons", "spread_analysis.xlsx")

    # ----- Family C: axes -----
    family_c = run_family(
        participant_level_df,
        overall_df,
        AXIS_OUTCOMES,
        compare_groups,
        "Family C: axis-specific error and bias",
    )

    axis_descriptives = (
        long_df.groupby(["level_number", "axis_label"], dropna=False)
        .agg(
            n=("deltaE2000", "count"),
            mean_deltaE2000=("deltaE2000", "mean"),
            sd_deltaE2000=("deltaE2000", "std"),
            mean_magnitude=("response_magnitude", "mean"),
            nonresponse_rate=("sublevel_is_nonresponse", "mean"),
        )
        .reset_index()
    )

    save_results(
        family_c,
        "axis_comparisons",
        "axis_analysis.xlsx",
        extra_sheets={"axis_descriptives": axis_descriptives},
    )

    # ----- Family D: non-response -----
    if "nonresponse_rate" in overall_df.columns:
        family_d = run_family(
            participant_level_df,
            overall_df,
            ["nonresponse_rate"],
            compare_groups,
            "Family D: non-response rate",
            include_per_level=False,
        )
        save_results(family_d, "nonresponse_comparisons", "nonresponse_analysis.xlsx")

    print("\nDone.")
    print(f"Plots were saved in:\n{plot_folder}")


if __name__ == "__main__":
    main()