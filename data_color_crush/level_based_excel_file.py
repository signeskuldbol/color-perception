"""
Color Crush data handler that creates 2 excel files:
    1. combined_final_results.xlsx
        Sheets:
            - participant_summary: one row per participant with demographics and level summary (highest level overview)
            - all_levels_short: one row per participant per level with summary info 
            - all_final_colors_long: all data rows for all participants and all levels' sublevels.
            - level_1 to level_8: one sheet per level with all kept attempts for that level

    2. discarded_results.xlsx
        Sheets:
        - empty_files: participant files with no log rows found
        - discarded_level_attempts: structurally incomplete attempts or attempts removed due to filtering.


What this script does:
- Reads all participant .txt files from the designated folder.
- Extracts participant demographics when available.
- Extracts raw gameplay log lines directly from the text files.
- Sorts all gameplay events by their numeric timestamp, not by Color ID block order (due to unsorted log entries).
- Ignores the tutorial color block 000000.
- Detects completed color-level attempts from gamelevelbegun to finalcolors.
- Keeps structurally valid attempts and only discards structurally incomplete attempts, such as:
    - no log rows found
    - a new gamelevelbegun before the previous level reached finalcolors
    - gamelevelend without finalcolors
- Converts raw timestamp differences from microseconds to milliseconds.
- Saves timing information as diagnostic data.
- Counts selected, deselected, and final selected colors for each sublevel.
- Calculates Delta E 76 and Delta E 2000 between the base color and each final color.
- Removes accidental zero-chosen attempts using the final filtering rules:
    - repeated zero-chosen attempts are removed if another attempt of the same level/color has choices
    - zero-chosen attempts under 30 seconds are removed
- Builds participant_summary from the final cleaned data, so it matches the kept result sheets.
- Writes the cleaned results to combined_final_results.xlsx.
- Writes empty files and discarded attempts to discarded_results.xlsx.

Expected project structure:
    color-perception/
    └── data_color_crush/
        ├── this script
        ├── "-USER DATA-"/
        │   ├── participant_file_1.txt
        │   ├── participant_file_2.txt
        │   └── ...
        └── excel_files/
            ├── combined_final_results.xlsx
            └── discarded_results.xlsx
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

discarded_output_file = output_folder / "discarded_results.xlsx"

EXPECTED_FINAL_COLORS_PER_LEVEL = 6

MAIN_COLOR_LEVELS = 8
# ============================================================
# FILTERING SETTINGS
# ============================================================

# attempts with 0 colors choosen where another attempt with the same base color has >0 colors choosen will be removed.
REMOVE_ZERO_REPEATED_ATTEMPTS = True
# remove attempts with 0 colors choosen if the attempt took less than 30 seconds.
REMOVE_FAST_ZERO_ATTEMPTS = True

# 30 seconds = 30,000 milliseconds.
FAST_ZERO_ATTEMPT_MAX_TIME_MS = 30_000

# An attempt with 0 chosen colors across all 6 sublevels is treated
# as a zero-chosen attempt.
ZERO_CHOSEN_COLOR_TOTAL = 0

# ============================================================
# REGEX PATTERNS
# ============================================================

USER_RE = re.compile(r"^USER:\s*(.+?)\s*$", re.MULTILINE)

DEMOGRAPHICS_RE = re.compile(
    r"Subcollection:\s*demographics.*?Data:\s*(\{.*?\})\s*(?=\n\s*Subcollection:|\Z)",
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

def build_combined_log_stream(full_text: str) -> list[dict[str, Any]]:
    """
    Build one combined log stream directly from the raw text.

    This does NOT depend on valid JSON blocks.
    It reads the file line by line and extracts quoted log lines.

    Important:
        Color ID blocks are only used as storage metadata.
        They are NOT used as level boundaries.

    The true gameplay order comes from the numeric timestamp at the
    beginning of each log line.
    """
    log_rows: list[dict[str, Any]] = []

    current_color_block_id = None
    current_block_index = -1
    current_block_session_timestamp = None

    file_order = 0

    color_id_re = re.compile(r"^\s*Color ID:\s*([A-Fa-f0-9]{6})\s*$")
    timestamp_re = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')

    # Matches lines like:
    # "132642495,emojirewarded,reshot-icon",
    # "133024729,finalcolors,EA39DF ..."
    log_line_re = re.compile(r'^\s*"([^"]*)"\s*,?\s*$')

    for raw_line_number, line in enumerate(full_text.splitlines(), start=1):
        color_match = color_id_re.match(line)

        if color_match:
            current_color_block_id = color_match.group(1).upper()
            current_block_index += 1
            current_block_session_timestamp = None
            continue

        timestamp_match = timestamp_re.search(line)

        if timestamp_match:
            current_block_session_timestamp = timestamp_match.group(1)
            continue

        log_match = log_line_re.match(line)

        if not log_match:
            continue

        log = log_match.group(1)

        timestamp_ms, event_type, payload = split_log_line(log)

        # Only keep real game-style log lines.
        if timestamp_ms is None or event_type is None:
            continue

        # Ignore tutorial / black color block completely.
        if current_color_block_id in TUTORIAL_COLORS:
            continue

        log_rows.append(
            {
                "file_order": file_order,
                "raw_line_number": raw_line_number,
                "timestamp_ms": timestamp_ms,
                "event_type": event_type,
                "payload": payload,
                "block_index": current_block_index,
                "color_block_id": current_color_block_id,
                "block_session_timestamp": current_block_session_timestamp,
                "block_log_index": None,
                "log": log,
            }
        )

        file_order += 1

    # This is the key fix:
    # gameplay order comes from the numeric timestamp, not from block order.
    log_rows = sorted(
        log_rows,
        key=lambda row: (
            row["timestamp_ms"],
            row["file_order"],
        ),
    )

    return log_rows

def parse_finalcolors_payload(
    payload: str,
    base_color: str | None,
) -> list[dict[str, Any]]:
    """
    Parse a finalcolors payload.

    Expected structure in data:
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

    n = EXPECTED_FINAL_COLORS_PER_LEVEL

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

def parse_participant_final_results(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
]:
    """
    Extract demographics and finalcolors results from one participant file.

    Returns:
        demographics
        kept_long_rows
        discarded_level_attempts
        is_empty_file

    It keeps structurally valid attempts and only discards attempts where
        the log structure is incomplete, for example:
            - no log rows found
            - new gamelevelbegun before previous finalcolors
            - gamelevelend without finalcolors
       
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    demographics = parse_demographics(text, path)

    level_attempt_events: list[dict[str, Any]] = []
    kept_long_rows: list[dict[str, Any]] = []
    discarded_level_attempts: list[dict[str, Any]] = []

    log_rows = build_combined_log_stream(text)

    if not log_rows:
        discarded_level_attempts.append(
            {
                **demographics,
                "discard_reason": "no_log_rows_found_in_raw_text",
            }
        )

    active_level = False

    current_base_color = None
    current_level_start_ms = None
    current_level_start_file_order = None
    current_level_started_in_color_block_id = None
    current_level_started_in_block_index = None
    current_level_started_session_timestamp = None

    current_sublevel_starts: list[int] = []
    current_sublevel_end_times: list[int | None] = []

    current_sublevel_selected_sets: list[set[str]] = []

    current_sublevel_selected_action_counts: list[int] = []
    current_sublevel_deselected_action_counts: list[int] = []
    current_sublevel_total_action_counts: list[int] = []

    console_output_count = 0
    console_error_count = 0

    for row in log_rows:
        timestamp_ms = row["timestamp_ms"]
        event_type = row["event_type"]
        payload = row["payload"]

        event_type_lower = (
            event_type.lower()
            if event_type is not None
            else ""
        )

        # ----------------------------------------------------
        # Start of a real color level attempt
        # ----------------------------------------------------
        if event_type_lower == "gamelevelbegun":
            # If a new level starts before the previous one had finalcolors,
            # the previous one is incomplete and should not leak forward.
            if active_level:
                discarded_level_attempts.append(
                    {
                        **demographics,
                        "discard_reason": "new_gamelevelbegun_before_previous_finalcolors",
                        "level_number": LEVEL_BY_BASE_COLOR.get(current_base_color),
                        "base_color": current_base_color,
                        "level_start_ms": current_level_start_ms,
                        "level_start_file_order": current_level_start_file_order,
                        "ended_before_finalcolors_file_order": row["file_order"],
                    }
                )

            if re.fullmatch(r"[A-Fa-f0-9]{6}", payload):
                current_base_color = payload.upper()
            else:
                current_base_color = None

            # Ignore tutorial or unknown colors.
            if (
                current_base_color is None
                or current_base_color in TUTORIAL_COLORS
                or current_base_color not in LEVEL_BY_BASE_COLOR
            ):
                active_level = False
                current_base_color = None
                continue

            active_level = True

            current_level_start_ms = timestamp_ms
            current_level_start_file_order = row["file_order"]
            current_level_started_in_color_block_id = row["color_block_id"]
            current_level_started_in_block_index = row["block_index"]
            current_level_started_session_timestamp = row["block_session_timestamp"]

            current_sublevel_starts = []
            current_sublevel_end_times = []

            current_sublevel_selected_sets = []

            current_sublevel_selected_action_counts = []
            current_sublevel_deselected_action_counts = []
            current_sublevel_total_action_counts = []

            console_output_count = 0
            console_error_count = 0

            continue

        # Ignore everything until a real gamelevelbegun has started.
        if not active_level:
            continue

        # ----------------------------------------------------
        # Console output / error messages
        # ----------------------------------------------------
        if event_type_lower == "consoleoutput":
            console_output_count += 1

            if "error" in str(payload).lower():
                console_error_count += 1

            continue

        # ----------------------------------------------------
        # A new sublevel starts at colorsgenerated,0
        # ----------------------------------------------------
        if event_type_lower == "colorsgenerated":
            payload_parts = str(payload).split()
            generated_index = None

            if payload_parts:
                try:
                    generated_index = int(payload_parts[0])
                except ValueError:
                    generated_index = None

            # Only colorsgenerated,0 starts a new sublevel.
            # colorsgenerated,1 through colorsgenerated,11 belong to
            # the same sublevel.
            if generated_index == 0 and timestamp_ms is not None:
                current_sublevel_starts.append(timestamp_ms)
                current_sublevel_end_times.append(None)

                current_sublevel_selected_sets.append(set())
                current_sublevel_selected_action_counts.append(0)
                current_sublevel_deselected_action_counts.append(0)
                current_sublevel_total_action_counts.append(0)

            continue

        # ----------------------------------------------------
        # Color selected inside current sublevel
        # ----------------------------------------------------
        if event_type_lower == "colorselected":
            if current_sublevel_selected_sets:
                current_sublevel_index = len(current_sublevel_selected_sets) - 1

                selected_id = str(payload).strip()

                if selected_id:
                    current_sublevel_selected_sets[current_sublevel_index].add(
                        selected_id
                    )

                current_sublevel_selected_action_counts[current_sublevel_index] += 1
                current_sublevel_total_action_counts[current_sublevel_index] += 1

            continue

        # ----------------------------------------------------
        # Color deselected inside current sublevel
        # ----------------------------------------------------
        if event_type_lower == "colordeselected":
            if current_sublevel_selected_sets:
                current_sublevel_index = len(current_sublevel_selected_sets) - 1

                selected_id = str(payload).strip()

                if selected_id:
                    current_sublevel_selected_sets[current_sublevel_index].discard(
                        selected_id
                    )

                current_sublevel_deselected_action_counts[current_sublevel_index] += 1
                current_sublevel_total_action_counts[current_sublevel_index] += 1

            continue

        # ----------------------------------------------------
        # End of one sublevel
        # ----------------------------------------------------
        if event_type_lower == "colorssubmitted":
            if current_sublevel_end_times and timestamp_ms is not None:
                current_sublevel_index = len(current_sublevel_end_times) - 1
                current_sublevel_end_times[current_sublevel_index] = timestamp_ms

            continue

        # ----------------------------------------------------
        # End of the whole color level attempt
        # ----------------------------------------------------
        if event_type_lower == "finalcolors":
            if current_sublevel_end_times and timestamp_ms is not None:
                last_index = len(current_sublevel_end_times) - 1

                if current_sublevel_end_times[last_index] is None:
                    current_sublevel_end_times[last_index] = timestamp_ms

            base_r, base_g, base_b = hex_to_rgb255(current_base_color)

            level_attempt_events.append(
                {
                    "participant_uuid": demographics.get("participant_uuid"),
                    "userID": demographics.get("userID"),
                    "source_file": path.name,

                    # Diagnostic info about where the attempt started/ended.
                    "level_started_in_color_block_id": current_level_started_in_color_block_id,
                    "level_ended_in_color_block_id": row["color_block_id"],
                    "level_started_in_block_index": current_level_started_in_block_index,
                    "level_ended_in_block_index": row["block_index"],
                    "crossed_color_block_boundary": (
                        current_level_started_in_block_index != row["block_index"]
                    ),

                    "color_block_id": current_level_started_in_color_block_id,
                    "base_color": current_base_color,
                    "base_R": base_r,
                    "base_G": base_g,
                    "base_B": base_b,

                    "session_timestamp": current_level_started_session_timestamp,
                    "finalcolors_block_session_timestamp": row["block_session_timestamp"],

                    "level_start_ms": current_level_start_ms,
                    "level_start_file_order": current_level_start_file_order,
                    "finalcolors_timestamp_ms": timestamp_ms,
                    "finalcolors_file_order": row["file_order"],

                    "sublevel_start_times": current_sublevel_starts.copy(),
                    "sublevel_end_times": current_sublevel_end_times.copy(),

                    "sublevel_final_selected_counts": [
                        len(selected_set)
                        for selected_set in current_sublevel_selected_sets
                    ],

                    "sublevel_selected_action_counts": current_sublevel_selected_action_counts.copy(),
                    "sublevel_deselected_action_counts": current_sublevel_deselected_action_counts.copy(),
                    "sublevel_total_action_counts": current_sublevel_total_action_counts.copy(),

                    "console_output_count": console_output_count,
                    "console_error_count": console_error_count,

                    "log_index": row["file_order"],
                    "payload": payload,
                }
            )

            # Very important:
            # the level is now finished, so nothing after this should leak
            # into the next level.
            active_level = False
            current_base_color = None
            current_level_start_ms = None
            current_level_start_file_order = None

            current_sublevel_starts = []
            current_sublevel_end_times = []
            current_sublevel_selected_sets = []
            current_sublevel_selected_action_counts = []
            current_sublevel_deselected_action_counts = []
            current_sublevel_total_action_counts = []

            console_output_count = 0
            console_error_count = 0

            continue

        # ----------------------------------------------------
        # gamelevelend without finalcolors = incomplete attempt
        # ----------------------------------------------------
        if event_type_lower == "gamelevelend":
            discarded_level_attempts.append(
                {
                    **demographics,
                    "discard_reason": "gamelevelend_without_finalcolors",
                    "level_number": LEVEL_BY_BASE_COLOR.get(current_base_color),
                    "base_color": current_base_color,
                    "level_start_ms": current_level_start_ms,
                    "level_start_file_order": current_level_start_file_order,
                    "gamelevelend_timestamp_ms": timestamp_ms,
                    "gamelevelend_file_order": row["file_order"],
                    "console_output_count": console_output_count,
                    "console_error_count": console_error_count,
                }
            )

            active_level = False
            current_base_color = None
            current_level_start_ms = None
            current_level_start_file_order = None

            current_sublevel_starts = []
            current_sublevel_end_times = []
            current_sublevel_selected_sets = []
            current_sublevel_selected_action_counts = []
            current_sublevel_deselected_action_counts = []
            current_sublevel_total_action_counts = []

            console_output_count = 0
            console_error_count = 0

            continue

    attempt_counter_by_base_color: dict[str, int] = {}

    for event in level_attempt_events:
        base_color = event["base_color"]

        if base_color in TUTORIAL_COLORS:
            continue

        if base_color not in LEVEL_BY_BASE_COLOR:
            continue

        level_number = LEVEL_BY_BASE_COLOR[base_color]

        attempt_counter_by_base_color[base_color] = (
            attempt_counter_by_base_color.get(base_color, 0) + 1
        )

        attempt_number = attempt_counter_by_base_color[base_color]

        parsed_final_colors = parse_finalcolors_payload(
            event["payload"],
            event["base_color"],
        )

        sublevel_start_times = event["sublevel_start_times"]
        sublevel_end_times = event["sublevel_end_times"]

        sublevel_final_selected_counts = event["sublevel_final_selected_counts"]
        sublevel_selected_action_counts = event["sublevel_selected_action_counts"]
        sublevel_deselected_action_counts = event["sublevel_deselected_action_counts"]
        sublevel_total_action_counts = event["sublevel_total_action_counts"]

        finalcolors_timestamp_ms = event["finalcolors_timestamp_ms"]

        level_start_ms = event["level_start_ms"]

        whole_level_duration_ms = None

        if level_start_ms is not None and finalcolors_timestamp_ms is not None:
            whole_level_duration_ms = (finalcolors_timestamp_ms - level_start_ms) / 1000 # from microseconds to milliseconds

        sublevel_durations: list[int | None] = []
        sublevel_missing_timing: list[bool] = []

        for i in range(EXPECTED_FINAL_COLORS_PER_LEVEL):
            duration = None

            if (
                i < len(sublevel_start_times)
                and i < len(sublevel_end_times)
                and sublevel_start_times[i] is not None
                and sublevel_end_times[i] is not None
            ):
                duration = (sublevel_end_times[i] - sublevel_start_times[i]) / 1000 # from microseconds to milliseconds

            sublevel_durations.append(duration)

            missing_timing = duration is None
            sublevel_missing_timing.append(missing_timing)

        total_final_selected_colors = sum(
            count
            for count in sublevel_final_selected_counts
            if count is not None
        )

        total_selected_actions = sum(
            count
            for count in sublevel_selected_action_counts
            if count is not None
        )

        total_deselected_actions = sum(
            count
            for count in sublevel_deselected_action_counts
            if count is not None
        )

        total_color_actions = sum(
            count
            for count in sublevel_total_action_counts
            if count is not None
        )

        # If the level attempt was structurally complete, keep all final-color rows.
        for parsed in parsed_final_colors:
            final_index = parsed["final_index"]

            sublevel_duration_ms = None
            sublevel_has_missing_timing = True
            final_selected_color_count = None
            selected_action_count = None
            deselected_action_count = None
            total_color_action_count = None

            chosen_color_count = None

            if final_index < EXPECTED_FINAL_COLORS_PER_LEVEL:
                sublevel_duration_ms = sublevel_durations[final_index]
                sublevel_has_missing_timing = sublevel_missing_timing[final_index]

                if final_index < len(sublevel_final_selected_counts):
                    final_selected_color_count = sublevel_final_selected_counts[final_index]
                    chosen_color_count = final_selected_color_count

                if final_index < len(sublevel_selected_action_counts):
                    selected_action_count = sublevel_selected_action_counts[final_index]

                if final_index < len(sublevel_deselected_action_counts):
                    deselected_action_count = sublevel_deselected_action_counts[final_index]

                if final_index < len(sublevel_total_action_counts):
                    total_color_action_count = sublevel_total_action_counts[final_index]


            kept_long_rows.append(
                {
                **demographics,
                "level_number": level_number,
                "attempt_number": attempt_number,
                "base_color": event["base_color"],
                "base_R": event["base_R"],
                "base_G": event["base_G"],
                "base_B": event["base_B"],

                "whole_level_duration_ms": whole_level_duration_ms,
                "sublevel_duration_ms": sublevel_duration_ms,
                "sublevel_has_missing_timing": sublevel_has_missing_timing,

                "total_final_selected_colors": total_final_selected_colors,
                "total_selected_actions": total_selected_actions,
                "total_deselected_actions": total_deselected_actions,
                "total_color_actions": total_color_actions,

                "sublevel_final_selected_color_count": final_selected_color_count,
                "sublevel_selected_action_count": selected_action_count,
                "sublevel_deselected_action_count": deselected_action_count,
                "sublevel_total_color_action_count": total_color_action_count,

                "sublevel_chosen_color_count": chosen_color_count,

                **parsed,
                }
            )

    is_empty_file = (
        len(kept_long_rows) == 0
        and len(discarded_level_attempts) == 0
    )

    return demographics, kept_long_rows, discarded_level_attempts, is_empty_file

def make_wide_level_row(level_df: pd.DataFrame) -> dict[str, Any]:
    """
    Convert one participant's long level data into one wide row.

    Input:
        6 rows, one for each final_index

    Output:
        one row with columns like:
            final_0_hex
            final_0_lab_L
            final_0_lab_a
            ...
            final_5_hex
            final_5_lab_L
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
        "base_color": first.get("base_color"),
        "base_R": first.get("base_R"),
        "base_G": first.get("base_G"),
        "base_B": first.get("base_B"),
        "whole_level_duration_ms": first.get("whole_level_duration_ms"),
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

        row[f"final_{i}_sublevel_final_selected_color_count"] = r.get(
            "sublevel_final_selected_color_count"
        )
        row[f"final_{i}_sublevel_selected_action_count"] = r.get(
            "sublevel_selected_action_count"
        )
        row[f"final_{i}_sublevel_deselected_action_count"] = r.get(
            "sublevel_deselected_action_count"
        )
        row[f"final_{i}_sublevel_total_color_action_count"] = r.get(
            "sublevel_total_color_action_count"
        )
        row[f"final_{i}_sublevel_duration_ms"] = r.get("sublevel_duration_ms")
        row[f"final_{i}_sublevel_has_missing_timing"] = r.get("sublevel_has_missing_timing")
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

def create_participant_summary_from_cleaned_data(
    all_levels_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create participant_summary from the final cleaned long data.

    This makes participant_summary match the data that is actually kept
    in all_final_colors_long and all_levels_short.
    """
    if all_levels_long_df.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []

    group_cols = [
        "participant_uuid",
        "source_file",
    ]

    for _, participant_df in all_levels_long_df.groupby(group_cols, dropna=False):
        first = participant_df.iloc[0]

        level_attempt_df = (
            participant_df[
                ["level_number", "base_color", "attempt_number"]
            ]
            .drop_duplicates()
        )

        total_level_completions_found = len(level_attempt_df)

        levels_found_list = sorted(
            participant_df["level_number"]
            .dropna()
            .astype(int)
            .unique()
        )

        if levels_found_list:
            max_level_number_found = max(levels_found_list)

            expected_levels_before_highest = set(
                range(1, max_level_number_found + 1)
            )

            actual_levels_found = set(levels_found_list)

            missing_levels_before_highest_list = sorted(
                expected_levels_before_highest - actual_levels_found
            )
        else:
            max_level_number_found = 0
            missing_levels_before_highest_list = []

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
                participant_df["base_color"].dropna().unique(),
                key=lambda color: LEVEL_BY_BASE_COLOR.get(color, 999),
            )
        )

        summary_rows.append(
            {
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
                "levels_completed_from_filename": first.get(
                    "levels_completed_from_filename"
                ),

                "total_sublevels_done": len(participant_df),
                "total_level_completions_found": total_level_completions_found,
                "max_level_reached": max_level_number_found,
                "levels_found": levels_found,
                "missing_levels_before_highest": missing_levels_before_highest,
                "has_missing_levels_before_highest": has_missing_levels_before_highest,
                "completed_base_colors": completed_base_colors,
            }
        )

    return pd.DataFrame(summary_rows)

def remove_zero_and_fast_attempts(
    all_levels_long_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove attempts that look like accidental or rushed non-attempts.

    Rule 1:
        Remove repeated attempts with 0 chosen colors if another attempt
        of the same participant/file/color has more than 0 chosen colors.

    Rule 2:
        Remove any attempt with 0 chosen colors if the whole level took
        less than FAST_ZERO_ATTEMPT_MAX_TIME_MS.
    """
    needed = [
        "participant_uuid",
        "source_file",
        "level_number",
        "base_color",
        "attempt_number",
        "total_final_selected_colors",
        "whole_level_duration_ms",
    ]

    missing = [
        col
        for col in needed
        if col not in all_levels_long_df.columns
    ]

    if missing:
        print(
            "Skipping zero/fast attempt filtering because these columns are missing:"
            f" {missing}"
        )
        return all_levels_long_df, pd.DataFrame()

    group_cols = [
        "participant_uuid",
        "source_file",
        "level_number",
        "base_color",
    ]

    attempt_cols = group_cols + ["attempt_number"]

    attempt_summary = (
        all_levels_long_df
        .groupby(attempt_cols, dropna=False)
        .agg(
            total_final_selected_colors=("total_final_selected_colors", "first"),
            total_color_actions=("total_color_actions", "first"),
            whole_level_duration_ms=("whole_level_duration_ms", "first"),
            mean_deltaE2000=("deltaE2000", "mean"),
            n_final_colors=("final_index", "count"),
        )
        .reset_index()
    )

    attempt_summary["n_attempts_same_color"] = (
        attempt_summary
        .groupby(group_cols, dropna=False)["attempt_number"]
        .transform("count")
    )

    attempt_summary["max_chosen_colors_same_color"] = (
        attempt_summary
        .groupby(group_cols, dropna=False)["total_final_selected_colors"]
        .transform("max")
    )

    attempt_summary["is_zero_chosen_attempt"] = (
        attempt_summary["total_final_selected_colors"]
        <= ZERO_CHOSEN_COLOR_TOTAL
    )

    attempt_summary["has_other_attempt_with_choices"] = (
        attempt_summary["n_attempts_same_color"] >= 2
    ) & (
        attempt_summary["max_chosen_colors_same_color"] > ZERO_CHOSEN_COLOR_TOTAL
    )

    attempt_summary["remove_repeated_zero_chosen_attempt"] = (
        REMOVE_ZERO_REPEATED_ATTEMPTS
        & attempt_summary["is_zero_chosen_attempt"]
        & attempt_summary["has_other_attempt_with_choices"]
    )

    attempt_summary["remove_fast_zero_chosen_attempt"] = (
        REMOVE_FAST_ZERO_ATTEMPTS
        & attempt_summary["is_zero_chosen_attempt"]
        & attempt_summary["whole_level_duration_ms"].notna()
        & (
            attempt_summary["whole_level_duration_ms"]
            < FAST_ZERO_ATTEMPT_MAX_TIME_MS
        )
    )

    attempt_summary["remove_attempt"] = (
        attempt_summary["remove_repeated_zero_chosen_attempt"]
        | attempt_summary["remove_fast_zero_chosen_attempt"]
    )

    attempts_to_remove = attempt_summary[
        attempt_summary["remove_attempt"]
    ].copy()

    if attempts_to_remove.empty:
        print("\nZero/fast attempt filter removed 0 attempts.")
        return all_levels_long_df, pd.DataFrame()

    remove_keys = attempts_to_remove[attempt_cols].copy()
    remove_keys["_remove_attempt"] = True

    marked_long_df = all_levels_long_df.merge(
        remove_keys,
        on=attempt_cols,
        how="left",
    )

    removed_long_df = marked_long_df[
        marked_long_df["_remove_attempt"] == True
    ].drop(columns=["_remove_attempt"])

    filtered_long_df = marked_long_df[
        marked_long_df["_remove_attempt"] != True
    ].drop(columns=["_remove_attempt"])

    discarded_rows: list[dict[str, Any]] = []

    for attempt_key, attempt_df in removed_long_df.groupby(
        attempt_cols,
        dropna=False,
    ):
        wide_row = make_wide_level_row(attempt_df)

        key_filter = pd.Series(True, index=attempts_to_remove.index)

        for col, value in zip(attempt_cols, attempt_key):
            key_filter = key_filter & (attempts_to_remove[col] == value)

        summary_row = attempts_to_remove[key_filter].iloc[0]

        if summary_row["remove_repeated_zero_chosen_attempt"]:
            discard_reason = (
                "repeated_zero_chosen_attempt_when_other_attempt_has_choices"
            )
        elif summary_row["remove_fast_zero_chosen_attempt"]:
            discard_reason = "fast_zero_chosen_attempt_under_30_seconds"
        else:
            discard_reason = "zero_or_fast_attempt_removed"

        wide_row["discard_reason"] = discard_reason
        wide_row["n_attempts_same_color"] = summary_row["n_attempts_same_color"]
        wide_row["total_final_selected_colors"] = summary_row[
            "total_final_selected_colors"
        ]
        wide_row["total_color_actions"] = summary_row["total_color_actions"]
        wide_row["whole_level_duration_ms"] = summary_row[
            "whole_level_duration_ms"
        ]
        wide_row["max_chosen_colors_same_color"] = summary_row[
            "max_chosen_colors_same_color"
        ]
        wide_row["remove_repeated_zero_chosen_attempt"] = summary_row[
            "remove_repeated_zero_chosen_attempt"
        ]
        wide_row["remove_fast_zero_chosen_attempt"] = summary_row[
            "remove_fast_zero_chosen_attempt"
        ]
        wide_row["fast_zero_attempt_max_time_ms"] = FAST_ZERO_ATTEMPT_MAX_TIME_MS

        discarded_rows.append(wide_row)

    discarded_attempts_df = pd.DataFrame(discarded_rows)

    print(
        "\nZero/fast attempt filter removed "
        f"{len(discarded_attempts_df)} attempts."
    )

    return filtered_long_df, discarded_attempts_df


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

    all_long_rows: list[dict[str, Any]] = []

    empty_files: list[dict[str, Any]] = []
    discarded_level_attempts_all: list[dict[str, Any]] = []

    failed_files: list[tuple[Path, Exception]] = []

    for input_file in input_files:
        try:
            demographics, kept_long_rows, discarded_level_attempts, is_empty_file = parse_participant_final_results(input_file)

            if is_empty_file:
                empty_files.append(
                    {
                        **demographics,
                        "discard_reason": "empty_file",
                    }
                )

            if discarded_level_attempts:
                discarded_level_attempts_all.extend(discarded_level_attempts)

            # Only add participants to the good Excel if they have kept rows.
            if not kept_long_rows:
                continue
                
            all_long_rows.extend(kept_long_rows)

        except Exception as exc:
            failed_files.append((input_file, exc))

    if not all_long_rows:
        raise SystemExit("No finalcolors results were found in any file.")

    all_levels_long_df = pd.DataFrame(all_long_rows)

    """Filtering rule:
    - any repeated attempts with 0 colors choosen where another attempt 
    with the same base color has >0 colors choosen will be removed.
    - any levels with 0 colors choosen and a duration of less than 30 seconds will be removed.
    """
    all_levels_long_df, repeated_discarded_attempts_df = (
        remove_zero_and_fast_attempts(all_levels_long_df)
    )

    participant_summary_df = create_participant_summary_from_cleaned_data(
        all_levels_long_df
    )

    if not repeated_discarded_attempts_df.empty:
        discarded_level_attempts_all.extend(
            repeated_discarded_attempts_df.to_dict("records")
        )

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
    # one row per participant x completed level attempt.
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

            whole_level_duration_ms=("whole_level_duration_ms", "first"),
            mean_sublevel_duration_ms=("sublevel_duration_ms", "mean"),
            min_sublevel_duration_ms=("sublevel_duration_ms", "min"),
            max_sublevel_duration_ms=("sublevel_duration_ms", "max"),

            total_final_selected_colors=("sublevel_final_selected_color_count", "sum"),
            mean_final_selected_colors=("sublevel_final_selected_color_count", "mean"),
            min_final_selected_colors=("sublevel_final_selected_color_count", "min"),
            max_final_selected_colors=("sublevel_final_selected_color_count", "max"),

            total_selected_actions=("sublevel_selected_action_count", "sum"),
            total_deselected_actions=("sublevel_deselected_action_count", "sum"),
            total_color_actions=("sublevel_total_color_action_count", "sum"),
            mean_color_actions=("sublevel_total_color_action_count", "mean"),

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
    number_of_level_sheets = max(MAIN_COLOR_LEVELS, max_level)

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

    empty_files_df = pd.DataFrame(empty_files)
    discarded_level_attempts_df = pd.DataFrame(discarded_level_attempts_all)

    print("\nDiscard summary:")
    print(f"  Empty files: {len(empty_files_df)}")
    print(f"  Total discarded attempts written: {len(discarded_level_attempts_df)}")

    if not discarded_level_attempts_df.empty and "discard_reason" in discarded_level_attempts_df.columns:
        print("\nDiscard reasons:")
        print(
            discarded_level_attempts_df["discard_reason"]
            .value_counts(dropna=False)
            .to_string()
        )

    with pd.ExcelWriter(discarded_output_file, engine="openpyxl") as writer:
        empty_files_df.to_excel(
            writer,
            sheet_name="empty_files",
            index=False,
        )

        discarded_level_attempts_df.to_excel(
            writer,
            sheet_name="discarded_level_attempts",
            index=False,
        )

        workbook = writer.book

        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]
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

    print("\nCreated discarded results file:")
    print(f"  {discarded_output_file}")

    if failed_files:
        print("\nSome files failed:")

        for path, exc in failed_files:
            print(f"  ERROR: {path.name}: {exc}")


if __name__ == "__main__":
    create_combined_workbook()