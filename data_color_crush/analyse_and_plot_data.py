from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    bar_width = 0.4

    found_bar_positions = [
        x - bar_width / 2
        for x in x_positions
    ]

    all_bar_positions = [
        x + bar_width / 2
        for x in x_positions
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        found_bar_positions,
        found_counts.values,
        width=bar_width,
        label="Level found",
    )

    plt.bar(
        found_bar_positions,
        missing_counts.values,
        width=bar_width,
        bottom=found_counts.values,
        label="Missing but expected",
    )

    plt.bar(
        all_bar_positions,
        all_attempt_counts.values,
        width=bar_width,
        label="All attempts including repeats",
    )

    plt.bar(
        all_bar_positions,
        missing_counts.values,
        width=bar_width,
        bottom=all_attempt_counts.values,
    )

    plt.title("Found, missing, and repeated level data by level")
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

        plt.figure(figsize=(8, 6))

        plt.scatter(
            level_df["age"],
            level_df["mean_deltaE2000"],
            alpha=0.7,
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

    level_summary = (
        plot_df
        .groupby("level_number", dropna=False)
        .agg(
            mean_deltaE2000=(
                "mean_deltaE2000",
                "mean",
            ),
            median_deltaE2000=(
                "mean_deltaE2000",
                "median",
            ),
            std_deltaE2000=(
                "mean_deltaE2000",
                "std",
            ),
            n_participant_files=(
                "mean_deltaE2000",
                "count",
            ),
        )
        .reset_index()
        .sort_values("level_number")
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        level_summary["level_number"].astype(str),
        level_summary["mean_deltaE2000"],
    )

    plt.title("Mean Delta E 2000 by color level")
    plt.xlabel("Level number")
    plt.ylabel("Mean Delta E 2000")

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

    level_summary = (
        plot_df
        .groupby("level_number", dropna=False)
        .agg(
            mean_whole_level_duration_s=(
                "whole_level_duration_s",
                "mean",
            ),
            median_whole_level_duration_s=(
                "whole_level_duration_s",
                "median",
            ),
            n_participant_files=(
                "whole_level_duration_s",
                "count",
            ),
        )
        .reset_index()
        .sort_values("level_number")
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        level_summary["level_number"].astype(str),
        level_summary["mean_whole_level_duration_s"],
    )

    plt.title("Mean whole-level duration by color level")
    plt.xlabel("Level number")
    plt.ylabel("Mean duration [s]")

    save_current_plot("06_mean_duration_by_level.png")


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

    print("\nDone.")
    print(f"Plots were saved in:\n{plot_folder}")


if __name__ == "__main__":
    main()