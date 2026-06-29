"""
Create one combined Excel file from Color Crush participant .txt exports.

The output workbook contains:
    - participant_summary:
        one row per participant
    - all_levels_long:
        one row per participant per level per final color index
    - level_1 ... level_9:
        one row per participant, containing the final result for that level

This script focuses only on finalcolors results, not the full selection/deselection process.

Expected project structure:

color-perception/
├── create_combined_final_results.py
└── data_color_crush/
    ├── users_22_juni_2026/
    │   ├── participant_file_1.txt
    │   ├── participant_file_2.txt
    │   └── ...
    └── excel_files/
        └── combined_final_results.xlsx

"""

from __future__ import annotations

from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76
import json
import numpy as np
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

project_folder = Path(__file__).resolve().parent

input_folder = project_folder / "users_22_juni_2026"
output_folder = project_folder / "excel_files"

output_file = output_folder / "combined_final_results.xlsx"


# ============================================================
# REGEX PATTERNS
# ============================================================

USER_RE = re.compile(r"^USER:\s*(.+?)\s*$", re.MULTILINE)

DEMOGRAPHICS_RE = re.compile(
    r"Subcollection:\s*demographics.*?Data:\s*(\{.*?\})\s*(?=\n\s*Subcollection:|\Z)",
    re.DOTALL,
)

COLOR_BLOCK_RE = re.compile(
    r"Color ID:\s*([A-Fa-f0-9]{6})\s*\n\s*Data:\s*(\{.*?\})\s*(?=\n\s*Color ID:|\Z)",
    re.DOTALL,
)

# ============================================================
# LEVEL MAPPING
# ============================================================

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

TUTORIAL_COLORS = {
    "000000",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json_object(text: str) -> dict[str, Any] | None:
    """Load a JSON object safely."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def safe_sheet_name(name: str) -> str:
    """
    Excel sheet names cannot contain some characters and must be <= 31 chars.
    """
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return cleaned[:31]


def extract_levels_completed_from_filename(path: Path) -> int | None:
    """
    Extract the final number from filenames like:
        20260420_094350_uuid_colors_processed_9.txt

    Returns:
        9
    """
    match = re.search(r"_colors_processed_(\d+)$", path.stem)
    if match:
        return int(match.group(1))  
    return None


def split_log_line(log: str) -> tuple[int | None, str | None, str]:
    """
    Split log lines like:
        41242,gamelevelbegun,DE3B62
        82343,colorsgenerated,0 DF3A62

    into:
        timestamp_ms, event_type, payload
    """
    parts = str(log).split(",", 2)

    if len(parts) == 1:
        return None, None, parts[0]

    if len(parts) == 2:
        ts, event_type = parts
        payload = ""
    else:
        ts, event_type, payload = parts

    try:
        timestamp_ms = int(ts)
    except ValueError:
        timestamp_ms = None

    return timestamp_ms, event_type.strip(), payload.strip()


def hex_to_rgb01(hex_color: str | None) -> np.ndarray | None:
    """
    Convert hex color like '52DE48' to RGB values in range 0-1.

    Returns:
        numpy array shaped like (1, 1, 3), which skimage expects.
    """
    if not hex_color:
        return None

    hex_color = hex_color.strip().replace("#", "")

    if not re.fullmatch(r"[A-Fa-f0-9]{6}", hex_color):
        return None

    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    return np.array([[[r, g, b]]], dtype=float)

def hex_to_rgb255(
    hex_color: str | None,
) -> tuple[int | None, int | None, int | None]:
    """
    Convert hex color like '52DE48' to ordinary RGB values from 0 to 255.

    Example:
        '52DE48' -> 82, 222, 72
    """
    if not hex_color:
        return None, None, None

    hex_color = hex_color.strip().replace("#", "")

    if not re.fullmatch(r"[A-Fa-f0-9]{6}", hex_color):
        return None, None, None

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return r, g, b


def hex_to_lab_skimage(
    hex_color: str | None,
) -> tuple[float | None, float | None, float | None]:
    """
    Convert hex color to Lab using scikit-image.
    """
    rgb = hex_to_rgb01(hex_color)

    if rgb is None:
        return None, None, None

    lab = rgb2lab(rgb)[0, 0]

    return float(lab[0]), float(lab[1]), float(lab[2])

def deltaE_from_hex(
    base_hex: str | None,
    final_hex: str | None,
) -> tuple[float | None, float | None]:
    """
    Calculate both Delta E 76 and Delta E 2000 between two hex colors.

    Both colors are converted:
        hex -> RGB 0-1 -> Lab

    Returns:
        deltaE76, deltaE2000
    """
    base_rgb = hex_to_rgb01(base_hex)
    final_rgb = hex_to_rgb01(final_hex)

    if base_rgb is None or final_rgb is None:
        return None, None

    base_lab = rgb2lab(base_rgb)
    final_lab = rgb2lab(final_rgb)

    delta_e_76 = deltaE_cie76(base_lab, final_lab)
    delta_e_2000 = deltaE_ciede2000(base_lab, final_lab)

    return float(delta_e_76[0, 0]), float(delta_e_2000[0, 0])

def parse_demographics(full_text: str, source_file: Path) -> dict[str, Any]:
    """
    Extract participant-level demographic data.
    """
    user_match = USER_RE.search(full_text)
    participant_uuid = user_match.group(1).strip() if user_match else None

    out: dict[str, Any] = {
        "participant_uuid": participant_uuid,
        "source_file": source_file.name,
        "levels_completed_from_filename": extract_levels_completed_from_filename(
            source_file
        ),
    }

    demo_match = DEMOGRAPHICS_RE.search(full_text)
    if not demo_match:
        return out

    demo = load_json_object(demo_match.group(1))
    if not demo:
        return out

    keys = demo.get("keys", [])
    values = demo.get("values", [])

    for key, value in zip(keys, values):
        clean_key = str(key).strip().replace(" ", "_")
        out[clean_key] = value

    if "age" in out:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")

    return out


def parse_finalcolors_payload(
    payload: str,
    base_color: str | None,
) -> list[dict[str, Any]]:
    """
    Parse a finalcolors payload.

    Expected structure in your data:
        8 hex colors
        8 direction/change tuples
        8 Lab-like tuples

    Example:
        finalcolors,
        63DC46 5BE552 ...
        (0.500; 0.000; 0.000) ...
        (78.753;-56.003;82.640) ...

    Returns one row per final color index.
    """
    hex_colors = re.findall(r"\b[A-Fa-f0-9]{6}\b", payload)
    base_lab_L, base_lab_a, base_lab_b = hex_to_lab_skimage(base_color)

    rows: list[dict[str, Any]] = []

    n = max(len(hex_colors), 8)

    for i in range(n):
        final_hex = hex_colors[i].upper() if i < len(hex_colors) else None

        final_r, final_g, final_b = hex_to_rgb255(final_hex)

        final_lab_L_from_hex, final_lab_a_from_hex, final_lab_b_from_hex = (
            hex_to_lab_skimage(final_hex)
        )

        deltaE76, deltaE2000 = deltaE_from_hex(base_color, final_hex)

        rows.append(
            {
            "final_index": i,
            "final_hex": final_hex,
            "final_R": final_r,
            "final_G": final_g,
            "final_B": final_b,

            # Lab values calculated from hex using scikit-image
            "base_lab_L_from_hex": base_lab_L,
            "base_lab_a_from_hex": base_lab_a,
            "base_lab_b_from_hex": base_lab_b,
            "final_lab_L_from_hex": final_lab_L_from_hex,
            "final_lab_a_from_hex": final_lab_a_from_hex,
            "final_lab_b_from_hex": final_lab_b_from_hex,

            # Color distances calculated using scikit-image
            "deltaE76": deltaE76,
            "deltaE2000": deltaE2000,
            }
        )

    return rows


def parse_participant_final_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Extract demographics and all finalcolors results from one participant file.

    The important part:
        level_number is assigned by sorting finalcolors entries by timestamp_ms.
        This is safer than trusting the order of Color ID blocks in the text file.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    demographics = parse_demographics(text, path)

    final_events: list[dict[str, Any]] = []

    for color_id, json_text in COLOR_BLOCK_RE.findall(text):
        block = load_json_object(json_text)

        if not block:
            continue

        session_timestamp = block.get("timestamp")
        logs = block.get("logs", []) or []

        current_base_color = color_id.upper()

        for log_index, log in enumerate(logs):
            timestamp_ms, event_type, payload = split_log_line(log)

            if event_type == "gamelevelbegun":
                if re.fullmatch(r"[A-Fa-f0-9]{6}", payload):
                    current_base_color = payload.upper()

            if event_type == "finalcolors":
                base_r, base_g, base_b = hex_to_rgb255(current_base_color)

                final_events.append(
                    {
                        "participant_uuid": demographics.get("participant_uuid"),
                        "userID": demographics.get("userID"),
                        "source_file": path.name,
                        "color_block_id": color_id.upper(),
                        "base_color": current_base_color,
                        "base_R": base_r,
                        "base_G": base_g,
                        "base_B": base_b,
                        "session_timestamp": session_timestamp,
                        "timestamp_ms": timestamp_ms,
                        "log_index": log_index,
                        "payload": payload,
                    }
                )


    final_events = sorted(
        final_events,
        key=lambda x: (
            x["session_timestamp"] or "",
            x["timestamp_ms"] if x["timestamp_ms"] is not None else float("inf"),
            x["log_index"],
        ),
    )

    long_rows: list[dict[str, Any]] = []

    attempt_counter_by_base_color: dict[str, int] = {}

    for event in final_events:
        base_color = event["base_color"]

        # Skip tutorial colors, for example 000000.
        if base_color in TUTORIAL_COLORS:
            continue

        # Skip unknown colors that are not part of the 8 real levels.
        if base_color not in LEVEL_BY_BASE_COLOR:
            continue

        level_number = LEVEL_BY_BASE_COLOR[base_color]

        # Count repeated completions of the same color for this participant.
        attempt_counter_by_base_color[base_color] = (
            attempt_counter_by_base_color.get(base_color, 0) + 1
        )

        attempt_number = attempt_counter_by_base_color[base_color]

        parsed_final_colors = parse_finalcolors_payload(
            event["payload"],
            event["base_color"],
        )

        for parsed in parsed_final_colors:
            long_rows.append(
                {
                    **demographics,
                    "level_number": level_number,
                    "attempt_number": attempt_number,
                    "color_block_id": event["color_block_id"],
                    "base_color": event["base_color"],
                    "base_R": event["base_R"],
                    "base_G": event["base_G"],
                    "base_B": event["base_B"],
                    "session_timestamp": event["session_timestamp"],
                    "timestamp_ms": event["timestamp_ms"],
                    "log_index": event["log_index"],
                    **parsed,
                }
            )

    return demographics, long_rows


def make_wide_level_row(level_df: pd.DataFrame) -> dict[str, Any]:
    """
    Convert one participant's long level data into one wide row.

    Input:
        8 rows, one for each final_index

    Output:
        one row with columns like:
            final_0_hex
            final_0_lab_L
            final_0_lab_a
            ...
            final_7_hex
            final_7_lab_L
            ...
    """
    if level_df.empty:
        return {}

    first = level_df.iloc[0]

    row: dict[str, Any] = {
        "participant_uuid": first.get("participant_uuid"),
        "userID": first.get("userID"),
        "age": first.get("age"),
        "biologicalSex": first.get("biologicalSex"),
        "eyeColor": first.get("eyeColor"),
        "colorBlindness": first.get("colorBlindness"),
        "Nationality": first.get("Nationality"),
        "Device_Model": first.get("Device_Model"),
        "Operating_System": first.get("Operating_System"),
        "source_file": first.get("source_file"),
        "levels_completed_from_filename": first.get("levels_completed_from_filename"),
        "level_number": first.get("level_number"),
        "attempt_number": first.get("attempt_number"),
        "color_block_id": first.get("color_block_id"),
        "base_color": first.get("base_color"),
        "base_R": first.get("base_R"),
        "base_G": first.get("base_G"),
        "base_B": first.get("base_B"),
        "session_timestamp": first.get("session_timestamp"),
        "timestamp_ms": first.get("timestamp_ms"),
    }

    for _, r in level_df.sort_values("final_index").iterrows():
        i = int(r["final_index"])

        row[f"final_{i}_hex"] = r.get("final_hex")
        row[f"final_{i}_R"] = r.get("final_R")
        row[f"final_{i}_G"] = r.get("final_G")
        row[f"final_{i}_B"] = r.get("final_B")

        row[f"final_{i}_base_lab_L_from_hex"] = r.get("base_lab_L_from_hex")
        row[f"final_{i}_base_lab_a_from_hex"] = r.get("base_lab_a_from_hex")
        row[f"final_{i}_base_lab_b_from_hex"] = r.get("base_lab_b_from_hex")

        row[f"final_{i}_lab_L_from_hex"] = r.get("final_lab_L_from_hex")
        row[f"final_{i}_lab_a_from_hex"] = r.get("final_lab_a_from_hex")
        row[f"final_{i}_lab_b_from_hex"] = r.get("final_lab_b_from_hex")

        row[f"final_{i}_deltaE76"] = r.get("deltaE76")
        row[f"final_{i}_deltaE2000"] = r.get("deltaE2000")

    deltaE76_values = pd.to_numeric(level_df["deltaE76"], errors="coerce").dropna()
    deltaE2000_values = pd.to_numeric(level_df["deltaE2000"], errors="coerce").dropna()

    if len(deltaE76_values) > 0:
        row["mean_deltaE76"] = deltaE76_values.mean()
        row["median_deltaE76"] = deltaE76_values.median()
        row["min_deltaE76"] = deltaE76_values.min()
        row["max_deltaE76"] = deltaE76_values.max()
        row["std_deltaE76"] = deltaE76_values.std()
    else:
        row["mean_deltaE76"] = None
        row["median_deltaE76"] = None
        row["min_deltaE76"] = None
        row["max_deltaE76"] = None
        row["std_deltaE76"] = None

    if len(deltaE2000_values) > 0:
        row["mean_deltaE2000"] = deltaE2000_values.mean()
        row["median_deltaE2000"] = deltaE2000_values.median()
        row["min_deltaE2000"] = deltaE2000_values.min()
        row["max_deltaE2000"] = deltaE2000_values.max()
        row["std_deltaE2000"] = deltaE2000_values.std()
    else:
        row["mean_deltaE2000"] = None
        row["median_deltaE2000"] = None
        row["min_deltaE2000"] = None
        row["max_deltaE2000"] = None
        row["std_deltaE2000"] = None

    return row


def create_combined_workbook() -> None:
    """
    Main function:
        - reads all participant .txt files
        - extracts finalcolors results
        - writes one combined Excel workbook
    """
    print(f"Project folder: {project_folder}")
    print(f"Input folder:   {input_folder}")
    print(f"Output file:    {output_file}")

    output_folder.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_folder.glob("*.txt"))

    if not input_files:
        raise SystemExit(f"No .txt files found in: {input_folder}")

    all_demographics: list[dict[str, Any]] = []
    all_long_rows: list[dict[str, Any]] = []

    failed_files: list[tuple[Path, Exception]] = []

    for input_file in input_files:
        try:
            demographics, long_rows = parse_participant_final_results(input_file)

            if long_rows:
                participant_levels_df = pd.DataFrame(long_rows)

                total_level_completions_found = (
                    participant_levels_df[
                        ["level_number", "base_color", "attempt_number"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )

                levels_found_list = sorted(
                    participant_levels_df["level_number"]
                    .dropna()
                    .astype(int)
                    .unique()
                )

                max_level_number_found = max(levels_found_list)

                expected_levels_before_highest = set(
                    range(1, max_level_number_found + 1)
                )

                actual_levels_found = set(levels_found_list)

                missing_levels_before_highest_list = sorted(
                    expected_levels_before_highest - actual_levels_found
                )

                levels_found = ", ".join(
                    str(level) for level in levels_found_list
                )

                missing_levels_before_highest = ", ".join(
                    str(level) for level in missing_levels_before_highest_list
                )

                has_missing_levels_before_highest = (
                    len(missing_levels_before_highest_list) > 0
                )

                completed_base_colors = ", ".join(
                    sorted(
                        participant_levels_df["base_color"].dropna().unique(),
                        key=lambda color: LEVEL_BY_BASE_COLOR.get(color, 999),
                    )
                )

            else:
                total_level_completions_found = 0
                max_level_number_found = 0
                completed_base_colors = ""
                levels_found = ""
                missing_levels_before_highest = ""
                has_missing_levels_before_highest = False

            all_demographics.append(
                {
                    **demographics,
                    "total_sublevels_done": len(long_rows),
                    "total_level_completions_found": total_level_completions_found,
                    "max_level_reached": max_level_number_found,
                    "levels_found": levels_found,
                    "missing_levels_before_highest": missing_levels_before_highest,
                    "has_missing_levels_before_highest": has_missing_levels_before_highest,
                    "completed_base_colors": completed_base_colors,
                }
            )

            all_long_rows.extend(long_rows)

        except Exception as exc:
            failed_files.append((input_file, exc))

    if not all_long_rows:
        raise SystemExit("No finalcolors results were found in any file.")

    participant_summary_df = pd.DataFrame(all_demographics)
    all_levels_long_df = pd.DataFrame(all_long_rows)

    # Sort for readability.
    sort_cols = [
        "level_number",
        "base_color",
        "participant_uuid",
        "attempt_number",
        "final_index",
    ]

    existing_sort_cols = [
        col for col in sort_cols if col in all_levels_long_df.columns
    ]

    all_levels_long_df = all_levels_long_df.sort_values(existing_sort_cols)

    # Create a short version:
    # one row per participant × completed level attempt.
    short_group_cols = [
        "participant_uuid",
        "userID",
        "age",
        "biologicalSex",
        "eyeColor",
        "colorBlindness",
        "Nationality",
        "Device_Model",
        "Operating_System",
        "source_file",
        "levels_completed_from_filename",
        "level_number",
        "attempt_number",
        "base_color",
        "base_R",
        "base_G",
        "base_B",
    ]

    existing_short_group_cols = [
        col for col in short_group_cols
        if col in all_levels_long_df.columns
    ]

    all_levels_short_df = (
        all_levels_long_df
        .groupby(existing_short_group_cols, dropna=False)
        .agg(
            n_final_colors=("final_index", "count"),

            mean_deltaE76=("deltaE76", "mean"),
            median_deltaE76=("deltaE76", "median"),
            min_deltaE76=("deltaE76", "min"),
            max_deltaE76=("deltaE76", "max"),
            std_deltaE76=("deltaE76", "std"),

            mean_deltaE2000=("deltaE2000", "mean"),
            median_deltaE2000=("deltaE2000", "median"),
            min_deltaE2000=("deltaE2000", "min"),
            max_deltaE2000=("deltaE2000", "max"),
            std_deltaE2000=("deltaE2000", "std"),
        )
        .reset_index()
    )
    # order sheet:
    # level first, then participant, then repeated attempts.
    short_sort_cols = [
        "level_number",
        "participant_uuid",
        "attempt_number",
    ]

    existing_short_sort_cols = [
        col for col in short_sort_cols
        if col in all_levels_short_df.columns
    ]

    all_levels_short_df = all_levels_short_df.sort_values(
        existing_short_sort_cols
    )

    max_level = int(all_levels_long_df["level_number"].max())

    # There are 8 real levels. The black 000000 color is treated as tutorial.
    number_of_level_sheets = max(8, max_level)

    print(f"Participants processed: {len(participant_summary_df)}")
    print(f"Final result rows found: {len(all_levels_long_df)}")
    print(f"Max completed level found: {max_level}")

    if output_file.exists():
        print(f"Overwriting existing file: {output_file}")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        participant_summary_df.to_excel(
            writer,
            sheet_name="participant_summary",
            index=False,
        )

        all_levels_short_df.to_excel(
            writer,
            sheet_name="all_levels_short",
            index=False,
        )

        all_levels_long_df.to_excel(
            writer,
            sheet_name="all_final_colors_long",
            index=False,
        )

        for level_number in range(1, number_of_level_sheets + 1):
            level_df = all_levels_long_df[
                all_levels_long_df["level_number"] == level_number
            ].copy()

            wide_rows: list[dict[str, Any]] = []

            if not level_df.empty:
                for _, participant_level_df in level_df.groupby(
                    ["participant_uuid", "source_file", "attempt_number"],
                    dropna=False,
                ):
                    wide_rows.append(make_wide_level_row(participant_level_df))

            level_wide_df = pd.DataFrame(wide_rows)

            sheet_name = safe_sheet_name(f"level_{level_number}")

            level_wide_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        # Basic Excel formatting.
        workbook = writer.book

        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]
            ws.freeze_panes = "A2"

            if ws.max_row > 1 and ws.max_column > 1:
                ws.auto_filter.ref = ws.dimensions

            # Set readable column widths.
            for col in ws.columns:
                col_letter = col[0].column_letter

                max_len = 0
                for cell in col[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))

                ws.column_dimensions[col_letter].width = min(
                    max(max_len + 2, 10),
                    35,
                )

    print(f"\nCreated combined Excel file:")
    print(f"  {output_file}")

    if failed_files:
        print("\nSome files failed:")

        for path, exc in failed_files:
            print(f"  ERROR: {path.name}: {exc}")


if __name__ == "__main__":
    create_combined_workbook()