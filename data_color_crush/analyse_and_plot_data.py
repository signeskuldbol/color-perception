from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats as scipy_stats


try:
    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


SUBLEVELS_PER_LEVEL = 6
MAIN_COLOR_LEVELS = 8

LEVEL_BY_BASE_COLOR = {
    "DE3B62": 1,  # red / pink-red
    "52DE48": 2,  # green
    "5246E8": 3,  # blue-violet
    "E048E0": 4,  # magenta / purple
    "D8DE4D": 5,  # yellow-green
    "5ADED6": 6,  # cyan / turquoise
    "C5917D": 7,  # peach / tan
    "704C3C": 8,  # brown
}

LEVEL_COLORS = {
    level_number: f"#{base_hex}"
    for base_hex, level_number in LEVEL_BY_BASE_COLOR.items()
}
# ============================================================
# STATISTICAL ANALYSIS SETTINGS
# ============================================================

# A group (e.g. "Male", "Brown eyes") needs at least this many participants
# to be included in a comparison. Groups smaller than this are excluded,
# and a comparison is skipped entirely if fewer than 2 groups remain.
MIN_GROUP_SIZE = 3

ALPHA = 0.05
POWER_TARGET = 0.80

GROUP_VARIABLES = [
    "biologicalSex",
    "eyeColor",
    "colorBlindness",
    "Nationality",
]

OUTCOME_COLUMN = "mean_deltaE2000"

# Categories that represent a non-answer rather than a real demographic
# group, and should be excluded from comparisons entirely (not treated
# as their own group).
EXCLUDED_GROUP_VALUES: dict[str, list[str]] = {
    "colorBlindness": ["Don't know"],
    "eyeColor": ["Select Eye Color"],
}

# ============================================================
# PATH SETUP
# ============================================================

project_folder = Path(__file__).resolve().parent

excel_file = project_folder / "excel_files" / "combined_final_results.xlsx"
plot_folder = project_folder / "plots"

plot_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_current_plot(filename: str) -> None:
    """
    Save the current matplotlib figure to the plot folder.
    """
    output_path = plot_folder / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """
    Check that all needed columns exist before creating a plot or overview.
    """
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        print(f"Skipping because these columns are missing: {missing}")
        return False

    return True


def safe_filename(text: str) -> str:
    """
    Make text safe to use in a filename.
    """
    text = str(text)

    replacements = {
        "\\": "_",
        "/": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text[:120]

# ============================================================
# LOAD DATA
# ============================================================

def load_short_data() -> pd.DataFrame:
    """
    Load the all_levels_short sheet from the combined Excel file.

    This sheet contains:
        one row per participant/source file per completed level attempt.
    """
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file:\n{excel_file}\n\n"
            "Run your Excel creation script first."
        )

    df = pd.read_excel(
        excel_file,
        sheet_name="all_levels_short",
    )

    numeric_columns = [
        "age",
        "level_number",
        "attempt_number",
        "n_final_colors",
        "mean_deltaE76",
        "median_deltaE76",
        "min_deltaE76",
        "max_deltaE76",
        "std_deltaE76",
        "mean_deltaE2000",
        "median_deltaE2000",
        "min_deltaE2000",
        "max_deltaE2000",
        "std_deltaE2000",
        "whole_level_duration_ms",
        "mean_sublevel_duration_ms",
        "total_final_selected_colors",
        "total_selected_actions",
        "total_deselected_actions",
        "total_color_actions",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def load_long_data() -> pd.DataFrame:
    """
    Load the all_final_colors_long sheet from the combined Excel file.

    This sheet contains:
        one row per participant/source file per completed level attempt
        per final color/sublevel.
    """
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file:\n{excel_file}\n\n"
            "Run your Excel creation script first."
        )

    df = pd.read_excel(
        excel_file,
        sheet_name="all_final_colors_long",
    )

    numeric_columns = [
        "level_number",
        "attempt_number",
        "final_index",
        "deltaE76",
        "deltaE2000",
        "sublevel_chosen_color_count",
        "sublevel_final_selected_color_count",
        "sublevel_duration_ms",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# ANALYSIS DATA PREPARATION
# ============================================================

def average_repeated_attempts_per_participant_level(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average repeated attempts so each participant/source file has only
    one row per color level.

    This prevents participants with repeated attempts from contributing
    more weight than participants with only one attempt.

    Example:
        If one participant completed level 3 three times,
        those three attempts become one averaged level-3 row.
    """
    needed = [
        "source_file",
        "level_number",
        "mean_deltaE2000",
    ]

    if not require_columns(df, needed):
        return pd.DataFrame()

    analysis_df = df.copy()

    numeric_columns = [
        "age",
        "level_number",
        "mean_deltaE2000",
        "mean_deltaE76",
        "median_deltaE2000",
        "median_deltaE76",
        "whole_level_duration_ms",
        "mean_sublevel_duration_ms",
        "total_final_selected_colors",
        "total_selected_actions",
        "total_deselected_actions",
        "total_color_actions",
    ]

    for column in numeric_columns:
        if column in analysis_df.columns:
            analysis_df[column] = pd.to_numeric(
                analysis_df[column],
                errors="coerce",
            )

    group_columns = [
        "source_file",
        "level_number",
    ]

    aggregation_rules = {}

    # Numeric performance / behavior columns should be averaged.
    numeric_columns_to_average = [
        "mean_deltaE2000",
        "mean_deltaE76",
        "median_deltaE2000",
        "median_deltaE76",
        "whole_level_duration_ms",
        "mean_sublevel_duration_ms",
        "total_final_selected_colors",
        "total_selected_actions",
        "total_deselected_actions",
        "total_color_actions",
    ]

    for column in numeric_columns_to_average:
        if column in analysis_df.columns:
            aggregation_rules[column] = "mean"

    # These columns describe the participant or the level.
    # They should be copied from the first row in the repeated-attempt group.
    columns_to_keep_first = [
        "participant_uuid",
        "base_color",
        "age",
        "biologicalSex",
        "eyeColor",
        "colorBlindness",
        "Nationality",
        "Device_Model",
        "Operating_System",
    ]

    for column in columns_to_keep_first:
        if column in analysis_df.columns:
            aggregation_rules[column] = "first"

    participant_level_df = (
        analysis_df
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(aggregation_rules)
    )

    return participant_level_df


# ============================================================
# DATA OVERVIEW
# ============================================================

def create_data_overview(
    short_df: pd.DataFrame,
    long_df: pd.DataFrame | None = None,
    min_group_participants: int = 5,
) -> None:
    """
    Print a first overview of the cleaned Color Crush dataset.

    This is descriptive analysis, not formal statistics.

    Use this to understand:
        - dataset size
        - participants per level
        - hardest levels
        - group sizes
        - age distribution
        - sublevel overview
    """
    print("\n" + "=" * 80)
    print("DATA OVERVIEW")
    print("=" * 80)

    needed_short_columns = [
        "source_file",
        "level_number",
        "base_color",
        "mean_deltaE2000",
    ]

    if not require_columns(short_df, needed_short_columns):
        return

    df = short_df.copy()

    numeric_columns = [
        "age",
        "level_number",
        "mean_deltaE2000",
        "median_deltaE2000",
        "whole_level_duration_ms",
        "mean_sublevel_duration_ms",
        "total_final_selected_colors",
        "total_color_actions",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "participant_uuid" in df.columns:
        df["participant_file_id"] = (
            df["participant_uuid"].astype(str)
            + "__"
            + df["source_file"].astype(str)
        )
    else:
        df["participant_file_id"] = df["source_file"].astype(str)

    total_participant_files = df["participant_file_id"].nunique()
    total_level_rows = len(df)

    print("\nBasic dataset size:")
    print(f"  Total attempts with repetitions being meaned: {total_participant_files}")
    print(f"  Total completed sublevels:          {total_level_rows}")

    if long_df is not None:
        print(f"  Final-color/sublevel rows:       {len(long_df)}")

    # --------------------------------------------------------
    # Overview by color level
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("Overview by color level")
    print("-" * 80)

    aggregation_rules = {
        "participants": ("participant_file_id", "nunique"),
        "mean_deltaE2000": ("mean_deltaE2000", "mean"),
        "median_deltaE2000": ("mean_deltaE2000", "median"),
        "std_deltaE2000": ("mean_deltaE2000", "std"),
    }

    if "total_final_selected_colors" in df.columns:
        aggregation_rules["mean_total_final_selected_colors"] = (
            "total_final_selected_colors",
            "mean",
        )

    if "whole_level_duration_ms" in df.columns:
        aggregation_rules["mean_whole_level_duration_s"] = (
            "whole_level_duration_ms",
            lambda x: x.mean() / 1000,
        )

    level_overview = (
        df
        .groupby(
            ["level_number", "base_color"],
            dropna=False,
        )
        .agg(**aggregation_rules)
        .reset_index()
        .sort_values("level_number")
    )

    print(level_overview.to_string(index=False))

    # --------------------------------------------------------
    # Hardest levels overall
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("Color levels ranked, with hardest first(based on mean Delta E 2000)")
    print("-" * 80)

    hardest_overall = level_overview.sort_values(
        "mean_deltaE2000",
        ascending=False,
    )

    columns_to_show = [
        "level_number",
        "base_color",
        "participants",
        "mean_deltaE2000",
        "median_deltaE2000",
    ]

    print(
        hardest_overall[
            [
                column for column in columns_to_show
                if column in hardest_overall.columns
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Group sizes
    # --------------------------------------------------------
    group_columns = [
        "biologicalSex",
        "eyeColor",
        "colorBlindness",
        #"Nationality",
        #"Device_Model",
        #"Operating_System",
    ]

    existing_group_columns = [
        column for column in group_columns
        if column in df.columns
    ]

    print("\n" + "-" * 80)
    print("Group sizes")
    print("-" * 80)

    if not existing_group_columns:
        print("No demographic group columns found.")

    for group_column in existing_group_columns:
        print(f"\nGroup column: {group_column}")

        group_sizes = (
            df[
                [
                    "participant_file_id",
                    group_column,
                ]
            ]
            .drop_duplicates()
            .groupby(
                group_column,
                dropna=False,
            )
            .agg(
                participants=(
                    "participant_file_id",
                    "nunique",
                )
            )
            .reset_index()
            .sort_values(
                "participants",
                ascending=False,
            )
        )

        print(group_sizes.to_string(index=False))

    # --------------------------------------------------------
    # Age overview
    # --------------------------------------------------------
    if "age" in df.columns:
        print("\n" + "-" * 80)
        print("Age overview")
        print("-" * 80)

        age_df = (
            df[
                [
                    "participant_file_id",
                    "age",
                ]
            ]
            .drop_duplicates()
            .dropna(subset=["age"])
            .copy()
        )

        if age_df.empty:
            print("No usable age values found.")
        else:
            age_df["age_group"] = pd.cut(
                age_df["age"],
                bins=[0, 19, 29, 39, 49, 120],
                labels=[
                    "<=19",
                    "20-29",
                    "30-39",
                    "40-49",
                    "50+",
                ],
                right=True,
            )

            age_group_sizes = (
                age_df
                .groupby(
                    "age_group",
                    dropna=False,
                )
                .agg(
                    participants=(
                        "participant_file_id",
                        "nunique",
                    )
                )
                .reset_index()
            )

            print("\nAge groups:")
            print(age_group_sizes.to_string(index=False))

            print(age_df["age"].describe().to_string())

    print("\n" + "=" * 80)
    print("END OF DATA OVERVIEW")
    print("=" * 80)


# ============================================================
# PLOT 1: FOUND, MISSING, AND REPEATED LEVELS
# ============================================================

def plot_attempt_count_by_level(df: pd.DataFrame) -> None:
    """
    Plot found, missing, and repeated level data by level.

    This plot should use the attempt-level dataframe, not the averaged
    participant-level dataframe, because repeated attempts are part of
    what this plot is trying to show.
    """
    needed = [
        "source_file",
        "level_number",
    ]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].copy()

    if "participant_uuid" in df.columns:
        plot_df["participant_uuid"] = df["participant_uuid"]
    else:
        plot_df["participant_uuid"] = "missing_uuid"

    plot_df = plot_df.dropna(
        subset=[
            "source_file",
            "level_number",
        ]
    )

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["level_number"])
    plot_df["level_number"] = plot_df["level_number"].astype(int)

    levels = list(range(1, MAIN_COLOR_LEVELS + 1))

    unique_level_df = plot_df.drop_duplicates(
        [
            "participant_uuid",
            "source_file",
            "level_number",
        ]
    )

    found_counts = (
        unique_level_df["level_number"]
        .value_counts()
        .sort_index()
        .reindex(levels, fill_value=0)
    )

    all_attempt_counts = (
        plot_df["level_number"]
        .value_counts()
        .sort_index()
        .reindex(levels, fill_value=0)
    )

    participant_levels = (
        unique_level_df
        .groupby(
            [
                "participant_uuid",
                "source_file",
            ]
        )["level_number"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="levels_found")
    )

    missing_counts = pd.Series(0, index=levels)

    for _, row in participant_levels.iterrows():
        levels_found = row["levels_found"]

        if not levels_found:
            continue

        max_level = max(levels_found)
        expected_levels = set(range(1, max_level + 1))
        actual_levels = set(levels_found)

        missing_levels = expected_levels - actual_levels

        for missing_level in missing_levels:
            if missing_level in missing_counts.index:
                missing_counts.loc[missing_level] += 1

    x_positions = list(range(len(levels)))
    bar_width = 0.6

    plt.figure(figsize=(10, 6))

    plt.bar(
        x_positions,
        found_counts.values,
        width=bar_width,
        color="darkgreen",
        label="Unique participant attempts",
    )

    plt.bar(
        x_positions,
        all_attempt_counts.values-found_counts.values,
        width=bar_width,
        bottom=found_counts.values,
        color="limegreen",
        label="Repeated attempts",
    )

    plt.bar(
        x_positions,
        missing_counts.values,
        width=bar_width,
        bottom=found_counts.values+all_attempt_counts.values-found_counts.values,
        color="yellow",
        label="Missing but expected",
    )

    plt.title("Level attempts pr. level")
    plt.xlabel("Level number")
    plt.ylabel("Count")
    plt.xticks(
        x_positions,
        [
            str(level)
            for level in levels
        ],
    )
    plt.legend()

    save_current_plot("01_found_missing_and_repeated_levels_by_level.png")


# ============================================================
# PLOT 2: CHOSEN COLORS HEATMAPS BY PARTICIPANT
# ============================================================

def plot_chosen_colors_heatmap_by_participant(
    long_df: pd.DataFrame,
) -> None:
    """
    Create one heatmap per participant/source file.

    x-axis:
        color level / attempt

    y-axis:
        sublevel / final_index

    cell value:
        number of colors chosen in that sublevel

    This helps show whether a participant clicked through a level
    without choosing colors.
    """
    needed = [
        "source_file",
        "level_number",
        "attempt_number",
        "final_index",
        "sublevel_chosen_color_count",
    ]

    if not require_columns(long_df, needed):
        return

    plot_df = long_df[needed].copy()

    if "participant_uuid" in long_df.columns:
        plot_df["participant_uuid"] = long_df["participant_uuid"]
    else:
        plot_df["participant_uuid"] = "missing_uuid"

    numeric_columns = [
        "level_number",
        "attempt_number",
        "final_index",
        "sublevel_chosen_color_count",
    ]

    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    plot_df = plot_df.dropna(
        subset=[
            "source_file",
            "level_number",
            "attempt_number",
            "final_index",
        ]
    )

    plot_df["level_number"] = plot_df["level_number"].astype(int)
    plot_df["attempt_number"] = plot_df["attempt_number"].astype(int)
    plot_df["final_index"] = plot_df["final_index"].astype(int)

    participant_heatmap_folder = plot_folder / "chosen_colors_heatmaps"
    participant_heatmap_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_groups = plot_df.groupby(
        [
            "participant_uuid",
            "source_file",
        ],
        dropna=False,
    )

    print(
        f"\nCreating chosen-color heatmaps for "
        f"{len(participant_groups)} participant/source file groups..."
    )

    for (participant_uuid, source_file), person_df in participant_groups:
        person_df = person_df.copy()

        person_df["level_attempt_label"] = (
            "L"
            + person_df["level_number"].astype(str)
            + " A"
            + person_df["attempt_number"].astype(str)
        )

        attempt_order_df = (
            person_df[
                [
                    "level_number",
                    "attempt_number",
                    "level_attempt_label",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "level_number",
                    "attempt_number",
                ]
            )
        )

        attempt_labels = (
            attempt_order_df["level_attempt_label"]
            .tolist()
        )

        heatmap_df = (
            person_df
            .pivot_table(
                index="final_index",
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

        figure_width = max(
            SUBLEVELS_PER_LEVEL,
            len(attempt_labels) * 0.8,
        )

        plt.figure(figsize=(figure_width, 6))

        plt.imshow(
            heatmap_df.values,
            aspect="auto",
        )

        plt.colorbar(label="Number of colors chosen")

        plt.title(
            "Number of colors chosen per sublevel and color level\n"
            f"Participant/source file: {participant_uuid}"
        )

        plt.xlabel("Color level / attempt")
        plt.ylabel("Sublevel / final color index")

        plt.xticks(
            ticks=range(len(attempt_labels)),
            labels=attempt_labels,
            rotation=45,
            ha="right",
        )

        plt.yticks(
            ticks=range(SUBLEVELS_PER_LEVEL),
            labels=[
                str(index)
                for index in range(SUBLEVELS_PER_LEVEL)
            ],
        )

        for y_index in range(heatmap_df.shape[0]):
            for x_index in range(heatmap_df.shape[1]):
                value = heatmap_df.iloc[y_index, x_index]

                if pd.notna(value):
                    label = str(int(value))
                else:
                    label = "?"

                plt.text(
                    x_index,
                    y_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()

        filename = (
            f"participant_{safe_filename(participant_uuid)}"
            f"__{safe_filename(source_file)}"
            f"__chosen_colors_heatmap.png"
        )

        output_path = participant_heatmap_folder / filename

        plt.savefig(
            output_path,
            dpi=300,
        )
        plt.close()

    print(f"Saved chosen-color heatmaps in:\n{participant_heatmap_folder}")


# ============================================================
# PLOT 3: PARTICIPANT SUBLEVEL DISTANCE PLOTS
# ============================================================

def plot_each_participant_sublevel_distances(
    long_df: pd.DataFrame,
) -> None:
    """
    Create one plot per participant/source file.

    Each plot shows Delta E 2000 for every sublevel/final color.

    One line = one completed level attempt.
    If a participant repeated the same level, that repeated attempt gets
    a separate line.

    This plot intentionally uses the long dataframe with repeated attempts,
    because it is meant to inspect participant-level behavior.
    """
    needed = [
        "source_file",
        "level_number",
        "attempt_number",
        "final_index",
        "deltaE2000",
    ]

    if not require_columns(long_df, needed):
        return

    participant_plot_folder = plot_folder / "participant_sublevel_plots"
    participant_plot_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = long_df[needed].copy()

    if "participant_uuid" in long_df.columns:
        plot_df["participant_uuid"] = long_df["participant_uuid"]
    else:
        plot_df["participant_uuid"] = "missing_uuid"

    numeric_columns = [
        "level_number",
        "attempt_number",
        "final_index",
        "deltaE2000",
    ]

    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    plot_df = plot_df.dropna(
        subset=[
            "source_file",
            "level_number",
            "attempt_number",
            "final_index",
        ]
    )

    plot_df["level_number"] = plot_df["level_number"].astype(int)
    plot_df["attempt_number"] = plot_df["attempt_number"].astype(int)
    plot_df["final_index"] = plot_df["final_index"].astype(int)

    plot_df["x_position"] = (
        (plot_df["level_number"] - 1) * SUBLEVELS_PER_LEVEL
        + plot_df["final_index"]
        + 1
    )

    participant_groups = plot_df.groupby(
        [
            "participant_uuid",
            "source_file",
        ],
        dropna=False,
    )

    print(
        f"\nCreating participant sublevel plots for "
        f"{len(participant_groups)} participant/source file groups..."
    )

    for (participant_uuid, source_file), person_df in participant_groups:
        person_df = person_df.sort_values(
            [
                "level_number",
                "attempt_number",
                "final_index",
            ]
        )

        plt.figure(figsize=(14, 6))

        for (level_number, attempt_number), attempt_df in person_df.groupby(
            [
                "level_number",
                "attempt_number",
            ],
            dropna=False,
        ):
            attempt_df = attempt_df.sort_values("final_index")

            full_index = pd.DataFrame(
                {
                    "final_index": list(range(SUBLEVELS_PER_LEVEL))
                }
            )

            attempt_complete = full_index.merge(
                attempt_df,
                on="final_index",
                how="left",
            )

            attempt_complete["level_number"] = level_number
            attempt_complete["attempt_number"] = attempt_number
            attempt_complete["x_position"] = (
                (level_number - 1) * SUBLEVELS_PER_LEVEL
                + attempt_complete["final_index"]
                + 1
            )

            label = f"Level {level_number}, attempt {attempt_number}"

            plt.plot(
                attempt_complete["x_position"],
                attempt_complete["deltaE2000"],
                marker="o",
                linewidth=1.5,
                label=label,
            )

        for boundary in range(
            SUBLEVELS_PER_LEVEL,
            MAIN_COLOR_LEVELS * SUBLEVELS_PER_LEVEL,
            SUBLEVELS_PER_LEVEL,
        ):
            plt.axvline(
                boundary + 0.5,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

        level_midpoints = [
            ((level - 1) * SUBLEVELS_PER_LEVEL)
            + ((SUBLEVELS_PER_LEVEL + 1) / 2)
            for level in range(1, MAIN_COLOR_LEVELS + 1)
        ]

        plt.xticks(
            level_midpoints,
            [
                f"Level {level}"
                for level in range(1, MAIN_COLOR_LEVELS + 1)
            ],
            rotation=0,
        )

        plt.title(
            f"Participant sublevel distances\n"
            f"Participant/source file: {participant_uuid}"
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
            f"participant_{safe_filename(participant_uuid)}"
            f"__{safe_filename(source_file)}"
            f"_sublevel_deltaE2000.png"
        )

        output_path = participant_plot_folder / filename

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Saved participant plots in:\n{participant_plot_folder}")


# ============================================================
# PLOT 4: AGE VS PERFORMANCE, ONE PLOT PER COLOR LEVEL
# ============================================================

def plot_age_vs_performance_per_color(
    df: pd.DataFrame,
) -> None:
    """
    Create one plot per color level.

    x-axis:
        participant age

    y-axis:
        mean Delta E 2000

    Each plot includes a simple linear regression line.

    This function should use participant_level_df, where repeated attempts
    have already been averaged.
    """
    needed = [
        "age",
        "level_number",
        "mean_deltaE2000",
    ]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].copy()

    plot_df["age"] = pd.to_numeric(
        plot_df["age"],
        errors="coerce",
    )

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df["mean_deltaE2000"] = pd.to_numeric(
        plot_df["mean_deltaE2000"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "age",
            "level_number",
            "mean_deltaE2000",
        ]
    )

    if plot_df.empty:
        print("Skipping age plot because there are no usable age values.")
        return

    age_plot_folder = plot_folder / "age_vs_performance_by_level"
    age_plot_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for level in range(1, MAIN_COLOR_LEVELS + 1):
        level_df = plot_df[
            plot_df["level_number"] == level
        ].copy()

        if level_df.empty:
            print(f"Skipping level {level} age plot: no data.")
            continue

        level_color = LEVEL_COLORS.get(level, "grey")

        plt.figure(figsize=(8, 6))

        plt.scatter(
            level_df["age"],
            level_df["mean_deltaE2000"],
            alpha=0.7,
            color=level_color,
        )

        if level_df["age"].nunique() >= 2 and len(level_df) >= 3:
            slope, intercept = np.polyfit(
                level_df["age"],
                level_df["mean_deltaE2000"],
                deg=1,
            )

            x_values = np.linspace(
                level_df["age"].min(),
                level_df["age"].max(),
                100,
            )

            y_values = slope * x_values + intercept

            plt.plot(
                x_values,
                y_values,
                linewidth=2,
                color=level_color,
            )

            plt.title(
                f"Age vs. mean Delta E 2000 - Level {level}\n"
                f"Linear regression slope = {slope:.3f}"
            )
        else:
            plt.title(
                f"Age vs. mean Delta E 2000 - Level {level}\n"
                "Not enough age variation for regression line"
            )

        plt.xlabel("Age")
        plt.ylabel("Mean Delta E 2000")

        output_path = (
            age_plot_folder
            / f"04_age_vs_mean_deltaE2000_level_{level}.png"
        )

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
        )
        plt.close()

        print(f"Saved: {output_path}")


# ============================================================
# PLOT 5: MEAN DELTA E 2000 BY LEVEL
# ============================================================

def plot_mean_deltaE2000_by_level(
    df: pd.DataFrame,
) -> None:
    """
    Plot average mean Delta E 2000 for each color level.

    This function should use participant_level_df, where repeated attempts
    have already been averaged.
    """
    needed = [
        "level_number",
        "mean_deltaE2000",
    ]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].copy()

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df["mean_deltaE2000"] = pd.to_numeric(
        plot_df["mean_deltaE2000"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "level_number",
            "mean_deltaE2000",
        ]
    )

    if plot_df.empty:
        print("Skipping mean Delta E plot: no usable data.")
        return

    levels_sorted = sorted(plot_df["level_number"].unique())

    data_by_level = [
        plot_df.loc[
            plot_df["level_number"] == level,
            "mean_deltaE2000",
        ].values
        for level in levels_sorted
    ]

    plt.figure(figsize=(10, 6))

    box_parts = plt.boxplot(
        data_by_level,
        tick_labels=[str(level) for level in levels_sorted],
        showmeans=True,
        zorder=2,
    )

    jitter_strength = 0.00

    for i, level in enumerate(levels_sorted, start=1):
        values = data_by_level[i - 1]

        if len(values) == 0:
            continue

        x_jitter = i + np.random.uniform(
            -jitter_strength,
            jitter_strength,
            size=len(values),
        )

        scatter_handle = plt.scatter(
            x_jitter,
            values,
            alpha=0.5,
            s=20,
            color=LEVEL_COLORS.get(level, "grey"),
            zorder=1,
        )

    legend_handles = [
        box_parts["medians"][0],
        box_parts["means"][0],
        box_parts["boxes"][0],
        box_parts["fliers"][0],
        scatter_handle,
    ]

    legend_labels = [
        "Median",
        "Mean",
        "IQR (25th-75th percentile)",
        "Outliers (beyond 1.5xIQR)",
        "Individual participants",
    ]

    plt.legend(legend_handles, legend_labels, loc="best")

    plt.title("Spread of mean Delta E 2000 by color level")
    plt.xlabel("Level number")
    plt.ylabel("Mean Delta E 2000 (per participant)")

    save_current_plot("05_mean_deltaE2000_by_level.png")


# ============================================================
# PLOT 6: WHOLE LEVEL DURATION BY LEVEL
# ============================================================

def plot_duration_by_level(
    df: pd.DataFrame,
) -> None:
    """
    Plot average whole-level duration for each color level.

    Duration is converted from milliseconds to seconds for the plot.

    This function should use participant_level_df, where repeated attempts
    have already been averaged.
    """
    needed = [
        "level_number",
        "whole_level_duration_ms",
    ]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].copy()

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df["whole_level_duration_ms"] = pd.to_numeric(
        plot_df["whole_level_duration_ms"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "level_number",
            "whole_level_duration_ms",
        ]
    )

    if plot_df.empty:
        print("Skipping duration plot: no usable data.")
        return

    plot_df["whole_level_duration_s"] = (
        plot_df["whole_level_duration_ms"] / 1000
    )

    levels_sorted = sorted(plot_df["level_number"].unique())

    data_by_level = [
        plot_df.loc[
            plot_df["level_number"] == level,
            "whole_level_duration_s",
        ].values
        for level in levels_sorted
    ]

    plt.figure(figsize=(10, 6))

    box_parts = plt.boxplot(
        data_by_level,
        tick_labels=[str(level) for level in levels_sorted],
        showmeans=True,
        zorder=2,
    )

    jitter_strength = 0.00

    for i, level in enumerate(levels_sorted, start=1):
        values = data_by_level[i - 1]

        if len(values) == 0:
            continue

        x_jitter = i + np.random.uniform(
            -jitter_strength,
            jitter_strength,
            size=len(values),
        )

        scatter_handle = plt.scatter(
            x_jitter,
            values,
            alpha=0.5,
            s=20,
            color=LEVEL_COLORS.get(level, "grey"),
            zorder=1,
        )

    legend_handles = [
        box_parts["medians"][0],
        box_parts["means"][0],
        box_parts["boxes"][0],
        box_parts["fliers"][0],
        scatter_handle,
    ]

    legend_labels = [
        "Median",
        "Mean",
        "IQR (25th-75th percentile)",
        "Outliers (beyond 1.5xIQR)",
        "Individual participants",
    ]

    plt.legend(legend_handles, legend_labels, loc="best")
    plt.ylim(ymin=0)

    plt.title("Spread of whole-level duration by color level")
    plt.xlabel("Level number")
    plt.ylabel("Duration [s] (per participant)")

    save_current_plot("06_mean_duration_by_level.png")

# ============================================================
# PLOT 7: group overview plots (demographics, etc.)
# ============================================================

def plot_participant_counts_by_group(
    df: pd.DataFrame,
) -> None:
    """
    Plot the number of participants in each demographic group:
        - eye color
        - age bin
        - sex
        - color blindness
        - Nationality

    This function should use participant_level_df (repeated attempts
    already averaged). Participants are de-duplicated by
    participant_uuid + source_file, since participant_level_df has one
    row per participant per level, not one row per participant overall.
    """
    needed = ["source_file"]

    if not require_columns(df, needed):
        return

    df = df.copy()

    if "participant_uuid" in df.columns:
        df["participant_file_id"] = (
            df["participant_uuid"].astype(str)
            + "__"
            + df["source_file"].astype(str)
        )
    else:
        df["participant_file_id"] = df["source_file"].astype(str)

    group_plot_folder = plot_folder / "group_overview"
    group_plot_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Eye color
    # ------------------------------------------------------
    if "eyeColor" in df.columns:
        eye_color_df = (
            df[["participant_file_id", "eyeColor"]]
            .drop_duplicates()
            .dropna(subset=["eyeColor"])
        )

        if not eye_color_df.empty:
            counts = (
                eye_color_df
                .groupby("eyeColor", dropna=False)["participant_file_id"]
                .nunique()
                .sort_index()
            )

            plt.figure(figsize=(8, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Participant count by eye color")
            plt.xlabel("Eye color")
            plt.ylabel("Number of participants")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            output_path = (
                group_plot_folder / "07_participant_count_by_eye_color.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print("Skipping eye color plot: no usable data.")
    else:
        print("Skipping eye color plot: column not found.")

    # ------------------------------------------------------
    # Age bins
    # ------------------------------------------------------
    if "age" in df.columns:
        age_df = (
            df[["participant_file_id", "age"]]
            .drop_duplicates()
            .copy()
        )
        age_df["age"] = pd.to_numeric(age_df["age"], errors="coerce")
        age_df = age_df.dropna(subset=["age"])

        if not age_df.empty:
            age_bins = [0, 19, 29, 39, 49, 120]
            age_labels = ["<=19", "20-29", "30-39", "40-49", "50+"]

            age_df["age_bin"] = pd.cut(
                age_df["age"],
                bins=age_bins,
                labels=age_labels,
                right=True,
            )

            counts = (
                age_df
                .groupby("age_bin", dropna=False)["participant_file_id"]
                .nunique()
                .reindex(age_labels, fill_value=0)
            )

            plt.figure(figsize=(8, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Participant count by age group")
            plt.xlabel("Age group")
            plt.ylabel("Number of participants")
            plt.tight_layout()

            output_path = (
                group_plot_folder / "07_participant_count_by_age_group.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print("Skipping age group plot: no usable age data.")
    else:
        print("Skipping age group plot: column not found.")

    # ------------------------------------------------------
    # Sex
    # ------------------------------------------------------
    if "biologicalSex" in df.columns:
        sex_df = (
            df[["participant_file_id", "biologicalSex"]]
            .drop_duplicates()
            .dropna(subset=["biologicalSex"])
        )

        if not sex_df.empty:
            counts = (
                sex_df
                .groupby("biologicalSex", dropna=False)["participant_file_id"]
                .nunique()
                .sort_index()
            )

            plt.figure(figsize=(8, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Participant count by sex")
            plt.xlabel("Sex")
            plt.ylabel("Number of participants")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            output_path = (
                group_plot_folder / "07_participant_count_by_sex.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print("Skipping sex plot: no usable data.")
    else:
        print("Skipping sex plot: column not found.")

    # ------------------------------------------------------
    # Color blindness
    # ------------------------------------------------------
    if "colorBlindness" in df.columns:
        color_blindness_df = (
            df[["participant_file_id", "colorBlindness"]]
            .drop_duplicates()
            .dropna(subset=["colorBlindness"])
        )

        if not color_blindness_df.empty:
            counts = (
                color_blindness_df
                .groupby("colorBlindness", dropna=False)["participant_file_id"]
                .nunique()
                .sort_index()
            )

            plt.figure(figsize=(8, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Participant count by color blindness status")
            plt.xlabel("Color blindness status")
            plt.ylabel("Number of participants")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            output_path = (
                group_plot_folder / "07_participant_count_by_color_blindness.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print("Skipping color blindness plot: no usable data.")
    else:
        print("Skipping color blindness plot: column not found.")

    # ------------------------------------------------------
    # Nationality
    # ------------------------------------------------------
    if "Nationality" in df.columns:
        nationality_df = (
            df[["participant_file_id", "Nationality"]]
            .drop_duplicates()
            .dropna(subset=["Nationality"])
        )

        if not nationality_df.empty:
            counts = (
                nationality_df
                .groupby("Nationality", dropna=False)["participant_file_id"]
                .nunique()
                .sort_index()
            )

            plt.figure(figsize=(8, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Participant count by nationality")
            plt.xlabel("Nationality")
            plt.ylabel("Number of participants")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            output_path = (
                group_plot_folder / "07_participant_count_by_nationality.png"
            )
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved: {output_path}")
        else:
            print("Skipping nationality plot: no usable data.")
    else:
        print("Skipping nationality plot: column not found.")


# ============================================================
# PLOT 8: spiderwebs pr participant, pr color level
# ============================================================

# Fixed compass angle (in the same order as DIRECTION_AXIS_LABELS),
# in degrees, measured clockwise from the top 
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


def plot_compass_spiderweb_by_participant_level(
    df: pd.DataFrame,
) -> None:
    """
    Create one spiderweb (radar) plot per participant/source file per
    color level per attempt, showing the 8 compass-direction magnitudes
    logged in the finalcolors payload.

    This function should use the attempt-level dataframe (all_levels_short),
    not the averaged participant-level dataframe, so repeated attempts
    are shown individually rather than blurred together.
    """
    magnitude_columns = [
        col for col in df.columns
        if col.startswith("dir_") and col.endswith("_magnitude")
    ]

    if not magnitude_columns:
        print(
            "Skipping compass spiderweb plots: no dir_*_magnitude "
            "columns found. Re-run the Excel export script with the "
            "compass-direction parsing added."
        )
        return

    needed = [
        "source_file",
        "level_number",
        "attempt_number",
    ] + magnitude_columns

    if not require_columns(df, needed):
        return

    plot_df = df[needed].copy()

    if "participant_uuid" in df.columns:
        plot_df["participant_uuid"] = df["participant_uuid"]
    else:
        plot_df["participant_uuid"] = "missing_uuid"

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )
    plot_df["attempt_number"] = pd.to_numeric(
        plot_df["attempt_number"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "source_file",
            "level_number",
            "attempt_number",
        ]
    )

    plot_df["level_number"] = plot_df["level_number"].astype(int)
    plot_df["attempt_number"] = plot_df["attempt_number"].astype(int)

    # Extract axis label from each column name, e.g.
    # "dir_2_V+_magnitude" -> "V+"
    axis_by_column = {}

    for col in magnitude_columns:
        parts = col.split("_")
        # dir, <index>, <label...>, magnitude
        axis_label = "_".join(parts[2:-1])
        axis_by_column[col] = axis_label

    # Order axes by their fixed compass angle so the spider is drawn
    # in visual clockwise order, not payload order.
    ordered_columns = sorted(
        magnitude_columns,
        key=lambda col: -DIRECTION_AXIS_ANGLES_DEG.get(
            axis_by_column[col], 0
        ),
    )

    axis_labels = [axis_by_column[col] for col in ordered_columns]
    axis_angles_rad = [
        np.deg2rad(DIRECTION_AXIS_ANGLES_DEG.get(label, 0))
        for label in axis_labels
    ]

    spiderweb_folder = plot_folder / "compass_spiderwebs"
    spiderweb_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_groups = plot_df.groupby(
        [
            "participant_uuid",
            "source_file",
            "level_number",
            "attempt_number",
        ],
        dropna=False,
    )

    print(
        f"\nCreating compass spiderweb plots for "
        f"{len(participant_groups)} participant/level/attempt groups..."
    )

    for (
        participant_uuid,
        source_file,
        level_number,
        attempt_number,
    ), attempt_df in participant_groups:
        row = attempt_df.iloc[0]

        magnitudes = [
            row.get(col, 0) or 0
            for col in ordered_columns
        ]

        # Close the loop for the plot.
        angles_plot = axis_angles_rad + [axis_angles_rad[0]]
        magnitudes_plot = magnitudes + [magnitudes[0]]

        level_color = LEVEL_COLORS.get(level_number, "grey")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.0)
        ax.set_rlabel_position(90)

        ax.plot(
            angles_plot,
            magnitudes_plot,
            color=level_color,
            linewidth=2,
        )
        ax.fill(
            angles_plot,
            magnitudes_plot,
            color=level_color,
            alpha=0.3,
        )

        ax.set_xticks(axis_angles_rad)
        ax.set_xticklabels(axis_labels)

        fig.suptitle(
            "Color Magnitudes",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_title(
            f"Level {level_number}, attempt {attempt_number}",
            fontsize=10,
        )

        participant_folder = (
            spiderweb_folder
            / f"participant_{safe_filename(str(participant_uuid))}"
        )
        participant_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{safe_filename(str(source_file))}"
            f"__level{level_number}_attempt{attempt_number}"
            f"__compass_spiderweb.png"
        )

        output_path = participant_folder / filename

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
        )
        plt.close()

    print(f"Saved compass spiderweb plots in:\n{spiderweb_folder}")

# ============================================================
# Analysis Functions
# ============================================================

def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Hedges' g: Cohen's d corrected for small-sample bias.
    Preferred over plain Cohen's d when group sizes are small (roughly
    under 20-30 per group).
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_sd = np.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    cohens_d = (mean1 - mean2) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))

    return cohens_d * correction


def rank_biserial_from_mannwhitney(
    u_statistic: float,
    n1: int,
    n2: int,
) -> float:
    """
    Rank-biserial correlation: the effect size that pairs with the
    Mann-Whitney U test, on a -1 to 1 scale.
    """
    return 1 - (2 * u_statistic) / (n1 * n2)


def eta_and_omega_squared(
    groups: list[np.ndarray],
) -> tuple[float, float]:
    """
    Eta squared and omega squared: the effect sizes that pair with a
    one-way ANOVA. Omega squared is less biased for small samples and
    should be treated as the primary number; eta squared is reported
    alongside it.
    """
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    n_total = len(all_values)
    k = len(groups)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in groups
    )

    ss_within = sum(
        np.sum((g - np.mean(g)) ** 2)
        for g in groups
    )

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
        omega_sq = (
            (ss_between - df_between * ms_within)
            / (ss_total + ms_within)
        )

    return eta_sq, omega_sq


def epsilon_squared_from_kruskal(
    h_statistic: float,
    n_total: int,
    k_groups: int,
) -> float:
    """
    Epsilon squared: the effect size that pairs with the Kruskal-Wallis
    test (the non-parametric counterpart to omega squared).
    """
    denominator = n_total - k_groups

    if denominator <= 0:
        return np.nan

    return (h_statistic - k_groups + 1) / denominator


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.

    Controls the expected proportion of false positives among the
    comparisons flagged as significant, across all comparisons run.
    This is applied once, at the end, across every comparison in the
    whole results table -- not separately per group variable -- since
    that is the actual family of tests being run.
    """
    n = len(p_values)

    if n == 0:
        return []

    indexed = list(enumerate(p_values))
    ranked = sorted(indexed, key=lambda pair: pair[1])

    corrected = [1.0] * n
    running_min = 1.0

    for rank in range(n, 0, -1):
        idx, p = ranked[rank - 1]
        value = min(p * n / rank, 1.0)
        running_min = min(running_min, value)
        corrected[idx] = running_min

    return corrected


def minimum_detectable_effect_two_groups(
    n1: int,
    n2: int,
) -> float | None:
    """
    Given the actual sizes of two groups, what is the smallest Cohen's d
    this comparison could reliably detect at 80% power?

    This reframes a non-significant result honestly: it tells you
    whether "no difference found" means "probably no real difference"
    or "not enough participants yet to tell."
    """
    if not STATSMODELS_AVAILABLE:
        return None

    try:
        analysis = TTestIndPower()

        return float(
            analysis.solve_power(
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


def minimum_detectable_effect_anova(
    n_total: int,
    k_groups: int,
) -> float | None:
    """
    Same idea as minimum_detectable_effect_two_groups, but for the
    multi-group ANOVA case. Returns the smallest Cohen's f detectable
    at 80% power given the current total sample size.
    """
    if not STATSMODELS_AVAILABLE:
        return None

    try:
        analysis = FTestAnovaPower()

        return float(
            analysis.solve_power(
                effect_size=None,
                nobs=n_total,
                alpha=ALPHA,
                power=POWER_TARGET,
                k_groups=k_groups,
            )
        )
    except Exception:
        return None

def compare_groups(
    df: pd.DataFrame,
    group_column: str,
    outcome_column: str,
    scope_label: str,
    ) -> dict[str, Any] | None:
    """
    Run a statistical group comparison for one grouping variable against
    one outcome, within one analysis scope (e.g. "overall" or "level_3").

    Automatically chooses:
        - Welch's t-test + Mann-Whitney U, if there are exactly 2 usable
          groups.
        - One-way ANOVA + Kruskal-Wallis, if there are 3+ usable groups.

    A group is only "usable" if it has at least MIN_GROUP_SIZE
    participants. If fewer than 2 usable groups remain, the comparison
    is skipped (not silently dropped -- a row is still returned
    explaining why).

    Returns a dict describing the test(s) run and their results, or
    None if the required columns are entirely missing.
    """
    if group_column not in df.columns or outcome_column not in df.columns:
        return None

    working_df = df[[group_column, outcome_column]].dropna()

    excluded_values = EXCLUDED_GROUP_VALUES.get(group_column, [])
    nonanswer_mask = working_df[group_column].isin(excluded_values)
    n_excluded_as_nonanswer = int(nonanswer_mask.sum())

    working_df = working_df[~nonanswer_mask]

    if working_df.empty:
        return None

    if working_df.empty:
        return None

    group_sizes = (
        working_df
        .groupby(group_column, dropna=False)[outcome_column]
        .count()
    )

    small_groups = group_sizes[group_sizes < MIN_GROUP_SIZE].index.tolist()
    usable_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index.tolist()

    groups_summary = ", ".join(
        f"{name} (n={count})"
        for name, count in group_sizes.items()
    )

    if len(usable_groups) < 2:
        return {
            "scope": scope_label,
            "group_variable": group_column,
            "outcome": outcome_column,
            "groups": groups_summary,
            "n_groups_total": len(group_sizes),
            "n_groups_usable": len(usable_groups),
            "status": "skipped_not_enough_groups",
            "note": (
                f"Excluded {n_excluded_as_nonanswer} non-answer rows "
                f"({', '.join(excluded_values) if excluded_values else 'none'}). "
                f"Fewer than 2 groups with at least {MIN_GROUP_SIZE} "
                "participants remain. Excluded small groups: "
                + (
                    ", ".join(str(g) for g in small_groups)
                    if small_groups
                    else "none"
                )
            ),
        }

    filtered_df = working_df[working_df[group_column].isin(usable_groups)]

    group_arrays = [
        filtered_df.loc[
            filtered_df[group_column] == group_name,
            outcome_column,
        ].values
        for group_name in usable_groups
    ]

    n_total = sum(len(arr) for arr in group_arrays)

    # --------------------------------------------------------
    # Assumption checks.
    # --------------------------------------------------------
    normality_p_values = []

    for arr in group_arrays:
        if len(arr) >= 3:
            try:
                _, p = scipy_stats.shapiro(arr)
                normality_p_values.append(p)
            except Exception:
                pass

    normality_ok = (
        all(p > ALPHA for p in normality_p_values)
        if normality_p_values
        else None
    )

    try:
        _, levene_p = scipy_stats.levene(*group_arrays)
        equal_variance_ok = levene_p > ALPHA
    except Exception:
        levene_p = None
        equal_variance_ok = None

    result: dict[str, Any] = {
        "scope": scope_label,
        "group_variable": group_column,
        "outcome": outcome_column,
        "groups": groups_summary,
        "n_groups_total": len(group_sizes),
        "n_groups_usable": len(usable_groups),
        "n_total_usable": n_total,
        "status": "tested",
        "note": (
            f"Excluded {n_excluded_as_nonanswer} non-answer rows "
            f"({', '.join(excluded_values) if excluded_values else 'none'}). "
            + (
                "Excluded small groups: "
                + ", ".join(str(g) for g in small_groups)
                if small_groups
                else ""
            )
        ),
        "normality_ok": normality_ok,
        "levene_p_value": levene_p,
        "equal_variance_ok": equal_variance_ok,
    }

    # --------------------------------------------------------
    # Two groups: Welch's t-test + Mann-Whitney U.
    # --------------------------------------------------------
    if len(usable_groups) == 2:
        group1, group2 = group_arrays
        n1, n2 = len(group1), len(group2)

        t_stat, t_p = scipy_stats.ttest_ind(
            group1,
            group2,
            equal_var=False,
        )

        g_effect = hedges_g(group1, group2)

        u_stat, u_p = scipy_stats.mannwhitneyu(
            group1,
            group2,
            alternative="two-sided",
        )

        rank_biserial = rank_biserial_from_mannwhitney(u_stat, n1, n2)

        mde = minimum_detectable_effect_two_groups(n1, n2)

        result.update(
            {
                "test_type": "two_group",
                "primary_test": "Welch's t-test",
                "primary_statistic": t_stat,
                "primary_p_value": t_p,
                "primary_effect_size_name": "Hedges' g",
                "primary_effect_size": g_effect,
                "robust_test": "Mann-Whitney U",
                "robust_statistic": u_stat,
                "robust_p_value": u_p,
                "robust_effect_size_name": "rank-biserial r",
                "robust_effect_size": rank_biserial,
                "min_detectable_effect_name": (
                    "Cohen's d" if mde is not None else None
                ),
                "min_detectable_effect": mde,
            }
        )

    # --------------------------------------------------------
    # Three or more groups: one-way ANOVA + Kruskal-Wallis.
    # --------------------------------------------------------
    else:
        f_stat, f_p = scipy_stats.f_oneway(*group_arrays)
        eta_sq, omega_sq = eta_and_omega_squared(group_arrays)

        h_stat, h_p = scipy_stats.kruskal(*group_arrays)
        epsilon_sq = epsilon_squared_from_kruskal(
            h_stat,
            n_total,
            len(usable_groups),
        )

        mde = minimum_detectable_effect_anova(n_total, len(usable_groups))

        result.update(
            {
                "test_type": "multi_group",
                "primary_test": "One-way ANOVA",
                "primary_statistic": f_stat,
                "primary_p_value": f_p,
                "primary_effect_size_name": "omega squared",
                "primary_effect_size": omega_sq,
                "primary_effect_size_secondary_name": "eta squared",
                "primary_effect_size_secondary": eta_sq,
                "robust_test": "Kruskal-Wallis",
                "robust_statistic": h_stat,
                "robust_p_value": h_p,
                "robust_effect_size_name": "epsilon squared",
                "robust_effect_size": epsilon_sq,
                "min_detectable_effect_name": (
                    "Cohen's f" if mde is not None else None
                ),
                "min_detectable_effect": mde,
            }
        )

    return result

def build_overall_participant_df(
    participant_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse participant_level_df (one row per participant per level) down
    to one row per participant, averaging the outcome across every level
    that participant completed.

    This is what the "overall" (pooled across levels) group comparisons
    use, so a participant who completed more levels does not carry more
    statistical weight than one who completed fewer -- each participant
    counts once, as independence assumptions require.
    """
    needed = ["source_file", OUTCOME_COLUMN]

    if not require_columns(participant_level_df, needed):
        return pd.DataFrame()

    group_columns = ["source_file"]

    aggregation_rules: dict[str, Any] = {OUTCOME_COLUMN: "mean"}

    demographic_columns = [
        "participant_uuid",
        "age",
        "biologicalSex",
        "eyeColor",
        "colorBlindness",
        "Nationality",
    ]

    for column in demographic_columns:
        if column in participant_level_df.columns:
            aggregation_rules[column] = "first"

    overall_df = (
        participant_level_df
        .groupby(group_columns, as_index=False, dropna=False)
        .agg(aggregation_rules)
    )

    return overall_df

def run_statistical_analysis(
    participant_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run group comparisons (sex, eye color, colorblindness, nationality)
    against mean Delta E 2000, both pooled across all levels ("overall")
    and separately for each color level.

    Returns one results DataFrame -- one row per (scope x group variable)
    comparison -- with test statistics, effect sizes, assumption checks,
    and a minimum detectable effect size given the actual group sizes.

    A Benjamini-Hochberg FDR correction is applied once, across every
    comparison that was actually run (not per group variable), since
    that full set is the real "family" of tests being performed.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    if not STATSMODELS_AVAILABLE:
        print(
            "\nNote: statsmodels is not installed, so minimum detectable "
            "effect sizes will not be calculated. Install it with:\n"
            "  pip install statsmodels"
        )

    results: list[dict[str, Any]] = []

    overall_df = build_overall_participant_df(participant_level_df)

    print(
        f"\nOverall analysis uses {len(overall_df)} participants "
        "(averaged across all levels each participant completed)."
    )

    for group_column in GROUP_VARIABLES:
        result = compare_groups(
            overall_df,
            group_column=group_column,
            outcome_column=OUTCOME_COLUMN,
            scope_label="overall",
        )

        if result is not None:
            results.append(result)

    if "level_number" in participant_level_df.columns:
        levels_present = sorted(
            participant_level_df["level_number"].dropna().unique()
        )

        for level in levels_present:
            level_df = participant_level_df[
                participant_level_df["level_number"] == level
            ]

            for group_column in GROUP_VARIABLES:
                result = compare_groups(
                    level_df,
                    group_column=group_column,
                    outcome_column=OUTCOME_COLUMN,
                    scope_label=f"level_{int(level)}",
                )

                if result is not None:
                    results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo comparisons could be run -- not enough data yet.")
        return results_df

    testable_mask = results_df["status"] == "tested"

    if testable_mask.any():
        p_values = results_df.loc[testable_mask, "primary_p_value"].tolist()
        corrected_p_values = benjamini_hochberg(p_values)

        results_df.loc[testable_mask, "primary_p_value_fdr_corrected"] = (
            corrected_p_values
        )

        results_df["significant_raw"] = (
            results_df["primary_p_value"] < ALPHA
        )
        results_df["significant_after_fdr_correction"] = (
            results_df["primary_p_value_fdr_corrected"] < ALPHA
        )

    n_tested = int(testable_mask.sum())
    n_skipped = int((~testable_mask).sum())

    print(f"\nComparisons run: {n_tested}")
    print(f"Comparisons skipped (not enough data): {n_skipped}")

    if n_tested > 0:
        n_significant_raw = int(
            results_df.loc[testable_mask, "significant_raw"].sum()
        )
        n_significant_corrected = int(
            results_df.loc[
                testable_mask, "significant_after_fdr_correction"
            ].sum()
        )

        print(
            f"Significant at raw p < {ALPHA}: "
            f"{n_significant_raw} / {n_tested}"
        )
        print(
            f"Significant after FDR correction: "
            f"{n_significant_corrected} / {n_tested}"
        )

    return results_df


def save_statistical_results(results_df: pd.DataFrame) -> None:
    """
    Save the group comparison results table to its own Excel workbook,
    with the same readability formatting used elsewhere in this project.
    """
    if results_df.empty:
        print("\nNo statistical results to save.")
        return

    output_folder = project_folder / "excel_files"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "statistical_analysis.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(
            writer,
            sheet_name="group_comparisons",
            index=False,
        )

        workbook = writer.book
        ws = workbook["group_comparisons"]
        ws.freeze_panes = "A2"

        if ws.max_row > 1 and ws.max_column > 1:
            ws.auto_filter.ref = ws.dimensions

        for col in ws.columns:
            col_letter = col[0].column_letter

            max_len = 0
            for cell in col[:200]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))

            ws.column_dimensions[col_letter].width = min(
                max(max_len + 2, 10),
                45,
            )

    print(f"\nSaved statistical analysis results:\n  {output_path}")
    

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    attempt_level_df = load_short_data()
    long_df = load_long_data()

    participant_level_df = average_repeated_attempts_per_participant_level(
        attempt_level_df
    )

    print("\nLoaded data:")
    print(f"  Rows from all_levels_short:       {len(attempt_level_df)}")
    print(f"  Rows after averaging repeats:     {len(participant_level_df)}")
    print(f"  Rows from all_final_colors_long:  {len(long_df)}")

    print("\nUnique participant/source files by level and base color:")
    if require_columns(
        attempt_level_df,
        [
            "source_file",
            "level_number",
            "base_color",
        ],
    ):
        temp_df = attempt_level_df.copy()

        if "participant_uuid" in temp_df.columns:
            unique_columns = [
                "participant_uuid",
                "source_file",
                "level_number",
                "base_color",
            ]
        else:
            unique_columns = [
                "source_file",
                "level_number",
                "base_color",
            ]

        print(
            temp_df
            .drop_duplicates(unique_columns)
            .groupby(
                [
                    "level_number",
                    "base_color",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="n_unique_participant_files")
            .sort_values("level_number")
            .to_string(index=False)
        )

    # Descriptive overview.
    # Use participant_level_df so repeated attempts do not over-weight the overview.
    create_data_overview(
        participant_level_df,
        long_df=long_df,
        min_group_participants=1,
    )

    # Quality-control / repeat overview.
    # Use attempt_level_df because this plot should show repeated attempts.
    plot_attempt_count_by_level(attempt_level_df)

    # Participant/sublevel diagnostic plots.
    # Use long_df because these plots inspect individual attempts/sublevels.
    plot_chosen_colors_heatmap_by_participant(long_df)
    plot_each_participant_sublevel_distances(long_df)

    # General performance plots.
    # Use participant_level_df because repeated attempts have been averaged.
    plot_age_vs_performance_per_color(participant_level_df)
    plot_mean_deltaE2000_by_level(participant_level_df)
    plot_duration_by_level(participant_level_df)
    plot_participant_counts_by_group(participant_level_df)
    plot_compass_spiderweb_by_participant_level(attempt_level_df)

    # Statistical analysis.
    results_df = run_statistical_analysis(participant_level_df)
    save_statistical_results(results_df)

    print("\nDone.")
    print(f"Plots were saved in:\n{plot_folder}")


if __name__ == "__main__":
    main()