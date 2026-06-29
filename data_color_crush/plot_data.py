from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

project_folder = Path(__file__).resolve().parent

excel_file = project_folder / "excel_files" / "combined_final_results.xlsx"
plot_folder = project_folder / "plots"

plot_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_short_data() -> pd.DataFrame:
    """
    Load the all_levels_short sheet from the combined Excel file.

    This sheet should contain:
        one row per participant × completed level attempt
    """
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file:\n{excel_file}\n\n"
            "Run your Excel creation script first."
        )

    df = pd.read_excel(excel_file, sheet_name="all_levels_short")

    # Convert important columns to numeric.
    numeric_cols = [
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
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_long_data() -> pd.DataFrame:
    """
    Load the all_final_colors_long sheet from the combined Excel file.

    This sheet contains:
        one row per participant × level attempt × final color/sublevel
    """
    if not excel_file.exists():
        raise FileNotFoundError(
            f"Could not find Excel file:\n{excel_file}\n\n"
            "Run your Excel creation script first."
        )

    df = pd.read_excel(excel_file, sheet_name="all_final_colors_long")

    numeric_cols = [
        "level_number",
        "attempt_number",
        "final_index",
        "deltaE76",
        "deltaE2000",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_current_plot(filename: str) -> None:
    """
    Save the current matplotlib figure.
    """
    output_path = plot_folder / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """
    Check that all needed columns exist before making a plot.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        print(f"Skipping plot because these columns are missing: {missing}")
        return False

    return True

def safe_filename(text: str) -> str:
    """
    Make text safe to use in a filename.
    """
    text = str(text)
    text = text.replace("\\", "_")
    text = text.replace("/", "_")
    text = text.replace(":", "_")
    text = text.replace("*", "_")
    text = text.replace("?", "_")
    text = text.replace('"', "_")
    text = text.replace("<", "_")
    text = text.replace(">", "_")
    text = text.replace("|", "_")
    text = text.replace(" ", "_")

    return text[:120]

# ============================================================
# PLOT 1: COMPLETED LEVEL ATTEMPTS BY LEVEL
# ============================================================

def plot_attempt_count_by_level(df: pd.DataFrame) -> None:
    """
    Shows completed and missing level data by level.

    Blue bars:
        number of unique participant/files where the level was found

    Orange stacked on blue:
        number of participant/files where this level is missing,
        but should have existed because a later level was found

    Red bars:
        total number of completed level attempts, including repeats

    Orange stacked on red:
        same missing expected count, shown on top of the total attempts bar too
    """
    needed = ["participant_uuid", "source_file", "level_number"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["level_number"])
    plot_df["level_number"] = plot_df["level_number"].astype(int)

    levels = list(range(1, 9))

    # One row per participant/file/level found.
    unique_level_df = plot_df.drop_duplicates(
        ["participant_uuid", "source_file", "level_number"]
    )

    # Blue: count unique participant/files where each level was found.
    found_counts = (
        unique_level_df["level_number"]
        .value_counts()
        .sort_index()
        .reindex(levels, fill_value=0)
    )

    # Red: count all completed level attempts, including repeats.
    all_attempt_counts = (
        plot_df["level_number"]
        .value_counts()
        .sort_index()
        .reindex(levels, fill_value=0)
    )

    # Orange: levels that are missing, but expected because a later level exists.
    participant_levels = (
        unique_level_df
        .groupby(["participant_uuid", "source_file"])["level_number"]
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

    # Bar positions.
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

    # Blue: found unique participant/files.
    plt.bar(
        found_bar_positions,
        found_counts.values,
        width=bar_width,
        color="blue",
        label="Level found",
    )

    # Orange on blue: missing expected levels.
    plt.bar(
        found_bar_positions,
        missing_counts.values,
        width=bar_width,
        bottom=found_counts.values,
        color="orange",
        label="Missing but expected",
    )

    # Red: all attempts including repetitions.
    plt.bar(
        all_bar_positions,
        all_attempt_counts.values,
        width=bar_width,
        color="red",
        label="All attempts including repeats",
    )

    # Orange on red: missing expected levels, shown on top of all attempts too.
    plt.bar(
        all_bar_positions,
        missing_counts.values,
        width=bar_width,
        bottom=all_attempt_counts.values,
        color="orange",
    )

    plt.title("Found, missing, and repeated level data by level")
    plt.xlabel("Level number")
    plt.ylabel("Count")
    plt.xticks(x_positions, [str(level) for level in levels])
    plt.legend()

    save_current_plot("01_found_missing_and_repeated_levels_by_level.png")
# ============================================================
# PLOT 2: MEAN DELTA E 2000 BY LEVEL
# ============================================================

def plot_mean_deltaE2000_by_level_boxplot(df: pd.DataFrame) -> None:
    """
    Shows the distribution of mean Delta E 2000 for each level.

    This helps you see whether some color levels generally have
    larger or smaller color distances.
    """
    needed = ["level_number", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()
    plot_df["level_number"] = plot_df["level_number"].astype(int)

    levels = sorted(plot_df["level_number"].unique())

    data = [
        plot_df.loc[
            plot_df["level_number"] == level,
            "mean_deltaE2000",
        ].values
        for level in levels
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=[str(level) for level in levels], showmeans=True)

    plt.title("Mean Delta E 2000 by level")
    plt.xlabel("Level number")
    plt.ylabel("Mean Delta E 2000")

    save_current_plot("02_mean_deltaE2000_by_level_boxplot.png")


# ============================================================
# PLOT 3: MEAN DELTA E 76 BY LEVEL
# ============================================================

def plot_mean_deltaE76_by_level_boxplot(df: pd.DataFrame) -> None:
    """
    Same as the Delta E 2000 plot, but using Delta E 76.
    """
    needed = ["level_number", "mean_deltaE76"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()
    plot_df["level_number"] = plot_df["level_number"].astype(int)

    levels = sorted(plot_df["level_number"].unique())

    data = [
        plot_df.loc[
            plot_df["level_number"] == level,
            "mean_deltaE76",
        ].values
        for level in levels
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=[str(level) for level in levels], showmeans=True)

    plt.title("Mean Delta E 76 by level")
    plt.xlabel("Level number")
    plt.ylabel("Mean Delta E 76")

    save_current_plot("03_mean_deltaE76_by_level_boxplot.png")


# ============================================================
# PLOT 4: REPEATED ATTEMPTS
# ============================================================

def plot_attempt_number_distribution(df: pd.DataFrame) -> None:
    """
    Shows how many rows are first attempts, second attempts, third attempts, etc.

    This helps you understand how common repeated level attempts are.
    """
    needed = ["attempt_number"]

    if not require_columns(df, needed):
        return

    counts = (
        df["attempt_number"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)

    plt.title("Distribution of repeated level attempts")
    plt.xlabel("Attempt number")
    plt.ylabel("Number of completed level attempts")

    save_current_plot("04_attempt_number_distribution.png")


# ============================================================
# PLOT 5: MEAN DELTA E 2000 BY COLOR VISION STATUS
# ============================================================

def plot_mean_deltaE2000_by_colorblindness(df: pd.DataFrame) -> None:
    """
    Compares mean Delta E 2000 by color vision status.

    This is useful for comparing NCV and CVD groups if both are present.
    """
    needed = ["colorBlindness", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()

    groups = sorted(plot_df["colorBlindness"].astype(str).unique())

    if len(groups) < 2:
        print("Skipping colorBlindness plot because fewer than 2 groups were found.")
        return

    data = [
        plot_df.loc[
            plot_df["colorBlindness"].astype(str) == group,
            "mean_deltaE2000",
        ].values
        for group in groups
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=groups, showmeans=True)

    plt.title("Mean Delta E 2000 by color vision status")
    plt.xlabel("Color vision status")
    plt.ylabel("Mean Delta E 2000")
    plt.xticks(rotation=30, ha="right")

    save_current_plot("05_mean_deltaE2000_by_colorBlindness_boxplot.png")


# ============================================================
# PLOT 6: MEAN DELTA E 2000 BY EYE COLOR
# ============================================================

def plot_mean_deltaE2000_by_eye_color(df: pd.DataFrame) -> None:
    """
    Compares mean Delta E 2000 by eye color.

    This is useful for checking whether eye-color groups look different.
    """
    needed = ["eyeColor", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()

    group_counts = plot_df["eyeColor"].astype(str).value_counts()

    # Keep groups with at least 2 observations.
    valid_groups = sorted(group_counts[group_counts >= 2].index)

    if len(valid_groups) < 2:
        print("Skipping eyeColor plot because fewer than 2 usable groups were found.")
        return

    data = [
        plot_df.loc[
            plot_df["eyeColor"].astype(str) == group,
            "mean_deltaE2000",
        ].values
        for group in valid_groups
    ]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=valid_groups, showmeans=True)

    plt.title("Mean Delta E 2000 by eye color")
    plt.xlabel("Eye color")
    plt.ylabel("Mean Delta E 2000")
    plt.xticks(rotation=30, ha="right")

    save_current_plot("06_mean_deltaE2000_by_eyeColor_boxplot.png")


# ============================================================
# PLOT 7: MEAN DELTA E 2000 BY BIOLOGICAL SEX
# ============================================================

def plot_mean_deltaE2000_by_biological_sex(df: pd.DataFrame) -> None:
    """
    Compares mean Delta E 2000 by biological sex.
    """
    needed = ["biologicalSex", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()

    groups = sorted(plot_df["biologicalSex"].astype(str).unique())

    if len(groups) < 2:
        print("Skipping biologicalSex plot because fewer than 2 groups were found.")
        return

    data = [
        plot_df.loc[
            plot_df["biologicalSex"].astype(str) == group,
            "mean_deltaE2000",
        ].values
        for group in groups
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=groups, showmeans=True)

    plt.title("Mean Delta E 2000 by biological sex")
    plt.xlabel("Biological sex")
    plt.ylabel("Mean Delta E 2000")
    plt.xticks(rotation=30, ha="right")

    save_current_plot("07_mean_deltaE2000_by_biologicalSex_boxplot.png")


# ============================================================
# PLOT 8: AGE VS MEAN DELTA E 2000
# ============================================================

def plot_age_vs_mean_deltaE2000(df: pd.DataFrame) -> None:
    """
    Scatterplot of age against mean Delta E 2000.

    This helps you see whether color distance changes with age.
    """
    needed = ["age", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()

    if plot_df.empty:
        print("Skipping age plot because there are no valid age values.")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["age"], plot_df["mean_deltaE2000"], alpha=0.7)

    plt.title("Age vs mean Delta E 2000")
    plt.xlabel("Age")
    plt.ylabel("Mean Delta E 2000")

    save_current_plot("08_age_vs_mean_deltaE2000_scatter.png")


# ============================================================
# PLOT 9: PARTICIPANT AVERAGE DELTA E 2000
# ============================================================

def plot_participant_average_deltaE2000_distribution(df: pd.DataFrame) -> None:
    """
    Creates a histogram of each participant's average Delta E 2000.

    This helps you check whether some participants are extreme outliers.
    """
    needed = ["participant_uuid", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    participant_df = (
        df[needed]
        .dropna()
        .groupby("participant_uuid", dropna=False)
        .agg(
            participant_mean_deltaE2000=("mean_deltaE2000", "mean"),
            n_level_attempts=("mean_deltaE2000", "count"),
        )
        .reset_index()
    )

    if participant_df.empty:
        print("Skipping participant histogram because there are no valid values.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(participant_df["participant_mean_deltaE2000"], bins=15)

    plt.title("Participant average mean Delta E 2000")
    plt.xlabel("Participant average mean Delta E 2000")
    plt.ylabel("Number of participants")

    save_current_plot("09_participant_average_deltaE2000_histogram.png")


# ============================================================
# PLOT 10: MEAN DELTA E 2000 BY LEVEL AND COLOR VISION STATUS
# ============================================================

def plot_level_means_by_colorblindness(df: pd.DataFrame) -> None:
    """
    Line plot showing average mean Delta E 2000 across levels,
    separated by color vision status.

    This can show whether NCV/CVD patterns differ across color levels.
    """
    needed = ["level_number", "colorBlindness", "mean_deltaE2000"]

    if not require_columns(df, needed):
        return

    plot_df = df[needed].dropna().copy()
    plot_df["level_number"] = plot_df["level_number"].astype(int)

    summary = (
        plot_df
        .groupby(["level_number", "colorBlindness"], dropna=False)
        .agg(group_mean_deltaE2000=("mean_deltaE2000", "mean"))
        .reset_index()
    )

    groups = sorted(summary["colorBlindness"].astype(str).unique())

    if len(groups) < 2:
        print("Skipping level/colorBlindness line plot because fewer than 2 groups were found.")
        return

    plt.figure(figsize=(9, 5))

    for group in groups:
        group_df = summary[summary["colorBlindness"].astype(str) == group]
        group_df = group_df.sort_values("level_number")

        plt.plot(
            group_df["level_number"],
            group_df["group_mean_deltaE2000"],
            marker="o",
            label=group,
        )

    plt.title("Mean Delta E 2000 by level and color vision status")
    plt.xlabel("Level number")
    plt.ylabel("Group mean Delta E 2000")
    plt.legend(title="Color vision status")

    save_current_plot("10_mean_deltaE2000_by_level_and_colorBlindness.png")

# ============================================================
# PLOT 11: PR PERSON PLOTS
# ============================================================
def plot_each_participant_sublevel_distances(long_df: pd.DataFrame) -> None:
    """
    Create one plot per participant/file.

    Each plot shows Delta E 2000 for every sublevel/final color.

    One line = one completed level attempt.
    If a participant repeated the same level, that repeated attempt gets
    a separate line.

    The plot is saved in:
        plots/participant_sublevel_plots/
    """
    needed = [
        "participant_uuid",
        "source_file",
        "level_number",
        "attempt_number",
        "final_index",
        "deltaE2000",
    ]

    if not require_columns(long_df, needed):
        return

    participant_plot_folder = plot_folder / "participant_sublevel_plots"
    participant_plot_folder.mkdir(parents=True, exist_ok=True)

    plot_df = long_df[needed].copy()

    plot_df["level_number"] = pd.to_numeric(
        plot_df["level_number"],
        errors="coerce",
    )

    plot_df["attempt_number"] = pd.to_numeric(
        plot_df["attempt_number"],
        errors="coerce",
    )

    plot_df["final_index"] = pd.to_numeric(
        plot_df["final_index"],
        errors="coerce",
    )

    plot_df["deltaE2000"] = pd.to_numeric(
        plot_df["deltaE2000"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "participant_uuid",
            "source_file",
            "level_number",
            "attempt_number",
            "final_index",
        ]
    )

    plot_df["level_number"] = plot_df["level_number"].astype(int)
    plot_df["attempt_number"] = plot_df["attempt_number"].astype(int)
    plot_df["final_index"] = plot_df["final_index"].astype(int)

    # x_position places the 8 sublevels of each level next to each other.
    # Level 1 gets x = 1 to 8
    # Level 2 gets x = 9 to 16
    # ...
    # Level 8 gets x = 57 to 64
    plot_df["x_position"] = (
        (plot_df["level_number"] - 1) * 8
        + plot_df["final_index"]
        + 1
    )

    participant_groups = plot_df.groupby(
        ["participant_uuid", "source_file"],
        dropna=False,
    )

    print(
        f"\nCreating participant sublevel plots for "
        f"{len(participant_groups)} participant/file groups..."
    )

    for (participant_uuid, source_file), person_df in participant_groups:
        person_df = person_df.sort_values(
            ["level_number", "attempt_number", "final_index"]
        )

        plt.figure(figsize=(14, 6))

        # One line per level attempt.
        for (level_number, attempt_number), attempt_df in person_df.groupby(
            ["level_number", "attempt_number"],
            dropna=False,
        ):
            attempt_df = attempt_df.sort_values("final_index")

            # Reindex so missing final_index values show as gaps.
            full_index = pd.DataFrame(
                {
                    "final_index": list(range(8))
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
                (level_number - 1) * 8
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

        # Add vertical lines between levels.
        for boundary in range(8, 64, 8):
            plt.axvline(
                boundary + 0.5,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

        # Label the middle of each level group.
        level_midpoints = [
            ((level - 1) * 8) + 4.5
            for level in range(1, 9)
        ]

        plt.xticks(
            level_midpoints,
            [f"Level {level}" for level in range(1, 9)],
            rotation=0,
        )

        plt.title(
            f"Participant sublevel distances\n"
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
            f"participant_{safe_filename(participant_uuid)}"
            f"__{safe_filename(source_file)}"
            f"_sublevel_deltaE2000.png"
        )

        output_path = participant_plot_folder / filename

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved participant plots in:\n{participant_plot_folder}")


# ============================================================
# OPTIONAL: CREATE SIMPLE SUMMARY EXCEL FILE
# ============================================================

def create_plot_summary_excel(df: pd.DataFrame) -> None:
    """
    Creates a small Excel file with group summaries used for checking plots.
    This is optional, but useful.
    """
    summary_file = plot_folder / "plot_summary_tables.xlsx"

    with pd.ExcelWriter(summary_file, engine="openpyxl") as writer:
        if require_columns(df, ["level_number", "mean_deltaE2000", "mean_deltaE76"]):
            by_level = (
                df
                .groupby("level_number", dropna=False)
                .agg(
                    n_level_attempts=("mean_deltaE2000", "count"),
                    mean_deltaE2000=("mean_deltaE2000", "mean"),
                    std_deltaE2000=("mean_deltaE2000", "std"),
                    mean_deltaE76=("mean_deltaE76", "mean"),
                    std_deltaE76=("mean_deltaE76", "std"),
                )
                .reset_index()
            )

            by_level.to_excel(writer, sheet_name="by_level", index=False)

        if require_columns(df, ["colorBlindness", "mean_deltaE2000", "mean_deltaE76"]):
            by_colorblindness = (
                df
                .groupby("colorBlindness", dropna=False)
                .agg(
                    n_level_attempts=("mean_deltaE2000", "count"),
                    n_participants=("participant_uuid", "nunique"),
                    mean_deltaE2000=("mean_deltaE2000", "mean"),
                    std_deltaE2000=("mean_deltaE2000", "std"),
                    mean_deltaE76=("mean_deltaE76", "mean"),
                    std_deltaE76=("mean_deltaE76", "std"),
                )
                .reset_index()
            )

            by_colorblindness.to_excel(
                writer,
                sheet_name="by_colorBlindness",
                index=False,
            )

        if require_columns(df, ["eyeColor", "mean_deltaE2000", "mean_deltaE76"]):
            by_eye_color = (
                df
                .groupby("eyeColor", dropna=False)
                .agg(
                    n_level_attempts=("mean_deltaE2000", "count"),
                    n_participants=("participant_uuid", "nunique"),
                    mean_deltaE2000=("mean_deltaE2000", "mean"),
                    std_deltaE2000=("mean_deltaE2000", "std"),
                    mean_deltaE76=("mean_deltaE76", "mean"),
                    std_deltaE76=("mean_deltaE76", "std"),
                )
                .reset_index()
            )

            by_eye_color.to_excel(writer, sheet_name="by_eyeColor", index=False)

        if require_columns(df, ["biologicalSex", "mean_deltaE2000", "mean_deltaE76"]):
            by_biological_sex = (
                df
                .groupby("biologicalSex", dropna=False)
                .agg(
                    n_level_attempts=("mean_deltaE2000", "count"),
                    n_participants=("participant_uuid", "nunique"),
                    mean_deltaE2000=("mean_deltaE2000", "mean"),
                    std_deltaE2000=("mean_deltaE2000", "std"),
                    mean_deltaE76=("mean_deltaE76", "mean"),
                    std_deltaE76=("mean_deltaE76", "std"),
                )
                .reset_index()
            )

            by_biological_sex.to_excel(
                writer,
                sheet_name="by_biologicalSex",
                index=False,
            )

    print(f"Saved summary tables: {summary_file}")

### debugging:

def print_missing_previous_levels(df: pd.DataFrame) -> None:
    """
    Finds participant files that have a later level but are missing earlier levels.

    Example:
        participant has level 8 but does not have level 7.
    """
    needed = ["participant_uuid", "source_file", "level_number", "base_color"]

    if not require_columns(df, needed):
        return

    check_df = df[needed].dropna().copy()

    check_df["level_number"] = pd.to_numeric(
        check_df["level_number"],
        errors="coerce",
    )

    check_df = check_df.dropna(subset=["level_number"])
    check_df["level_number"] = check_df["level_number"].astype(int)

    participant_levels = (
        check_df
        .drop_duplicates(["participant_uuid", "source_file", "level_number"])
        .groupby(["participant_uuid", "source_file"])["level_number"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="levels_found")
    )

    problem_rows = []

    for _, row in participant_levels.iterrows():
        levels_found = row["levels_found"]

        if not levels_found:
            continue

        max_level = max(levels_found)
        expected_levels = set(range(1, max_level + 1))
        actual_levels = set(levels_found)
        missing_levels = sorted(expected_levels - actual_levels)

        if missing_levels:
            problem_rows.append(
                {
                    "participant_uuid": row["participant_uuid"],
                    "source_file": row["source_file"],
                    "levels_found": str(levels_found),
                    "max_level": max_level,
                    "missing_before_max": str(missing_levels),
                }
            )

    problem_df = pd.DataFrame(problem_rows)

    print("\nParticipants/files with later levels but missing earlier levels:")

    if problem_df.empty:
        print("None found. Level progression looks consistent.")
    else:
        print(problem_df.to_string(index=False))

        output_path = plot_folder / "participants_missing_previous_levels.xlsx"
        problem_df.to_excel(output_path, index=False)
        print(f"\nSaved diagnostic file: {output_path}")
# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_short_data()
    long_df = load_long_data()

    print("\nUnique participants/files by level and base color:")
    print(
        df
        .drop_duplicates(["participant_uuid", "source_file", "level_number", "base_color"])
        .groupby(["level_number", "base_color"])
        .size()
        .reset_index(name="n_unique_participant_files")
        .sort_values("level_number")
    )
    # debugging: check for missing previous levels
    print_missing_previous_levels(df)

    print(f"Loaded rows from all_levels_short: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    plot_attempt_count_by_level(df)
    plot_mean_deltaE2000_by_level_boxplot(df)
    plot_mean_deltaE76_by_level_boxplot(df)
    plot_attempt_number_distribution(df)
    plot_mean_deltaE2000_by_colorblindness(df)
    plot_mean_deltaE2000_by_eye_color(df)
    plot_mean_deltaE2000_by_biological_sex(df)
    plot_age_vs_mean_deltaE2000(df)
    plot_participant_average_deltaE2000_distribution(df)
    plot_level_means_by_colorblindness(df)
    plot_each_participant_sublevel_distances(long_df)

    create_plot_summary_excel(df)

    print("\nDone.")
    print(f"Plots were saved in:\n{plot_folder}")


if __name__ == "__main__":
    main()