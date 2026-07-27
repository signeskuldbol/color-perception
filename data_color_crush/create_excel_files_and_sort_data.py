"""
Color Crush data handler that creates 2 excel files:
    1. combined_final_results.xlsx
        Sheets:
            - participant_summary: one row per participant with demographics and level summary
            - all_levels_short: one row per participant per level attempt with summary info
            - all_final_colors_long: one row per participant/level attempt/SUBLEVEL
            - level_1 to level_8: one sheet per level with all kept attempts for that level
            - payload_qc: per-attempt structural checks on the finalcolors payload

    2. discarded_results.xlsx
        Sheets:
            - empty_files: participant files with no log rows found
            - discarded_level_attempts: structurally incomplete or filtered-out attempts


=========================================================================
WHAT THE finalcolors PAYLOAD ACTUALLY CONTAINS  (verified against raw logs)
=========================================================================

A finalcolors payload looks like:

    finalcolors,<hex x8> <direction tuple x8> <logged Lab tuple x8>

Example (truncated):
    623FE8 5A50ED 5349E0 5A50ED 4D48E8 4E41E6 513EF8 4E41E6
    (0.500; 0.000; 0.000) ... x8
    (41,214;-0,355;-117,288) ... x8

There are EIGHT slots, one per UI compass position, but only SIX distinct
colors: slot 3 is always byte-identical to slot 1, and slot 7 to slot 5.
Each slot is the base color displaced along exactly ONE axis of the game's
own Lab-like space, by (direction magnitude x 11) units:

    slot 0  U+   ->  +a        slot 4  U-   ->  -a
    slot 1  L1+  ->  +L        slot 5  L1-  ->  -L
    slot 2  V+   ->  +b        slot 6  V-   ->  -b
    slot 3  L2+  ->  +L  (duplicate of slot 1)
    slot 7  L2-  ->  -L  (duplicate of slot 5)

So the UI's horizontal compass axis maps to the a axis, the vertical to
the b axis, and both diagonals to the L axis.

The six sublevels are played in the order of the six DISTINCT slots:

    sublevel 0 -> slot 0 (+a)      sublevel 3 -> slot 4 (-a)
    sublevel 1 -> slot 1 (+L)      sublevel 4 -> slot 5 (-L)
    sublevel 2 -> slot 2 (+b)      sublevel 5 -> slot 6 (-b)

(Confirmed by matching each sublevel's 12 generated candidate colors
against the payload slots: every match lands on the slot this ordering
predicts.)

IMPORTANT: the game's logged a/b values are NOT CIELAB a/b. Logged L*
matches a proper sRGB->CIELAB conversion to three decimals, but a and b
do not (e.g. base 5246E8 logs a=-5.855 b=-117.288 where CIELAB gives
a=+51.9 b=-80.0). Therefore:
    - Do NOT interpret logged a/b as CIELAB units.
    - Use the SLOT IDENTITY to know which axis was manipulated. It is
      exact and needs no reconstruction.
    - Use deltaE computed from the hex colors for perceptual distance,
      because the hex is what was actually displayed on screen.

Three outcome measures are therefore written for every sublevel:
    deltaE2000              perceptual distance (PRIMARY)
    deltaE76                perceptual distance, no chroma compression
    response_magnitude      the game's own logged response, in game units


What this script does:
- Reads all participant .txt files from the designated folder.
- Extracts participant demographics when available.
- Extracts raw gameplay log lines directly from the text files.
- Sorts all gameplay events by their numeric timestamp, not by Color ID
  block order (log entries are stored out of order, and a single level
  attempt can span two Color ID blocks).
- Drops the 000000 Color ID block entirely. It is the forced tutorial and
  contains no real data, so nothing from it is written to any sheet. Note
  the block does log a real base color (the tutorial is played in red),
  so the check is on the BLOCK, not on the color.
- Detects completed color-level attempts from gamelevelbegun to finalcolors.
- Discards structurally incomplete attempts (no log rows, a new
  gamelevelbegun before the previous finalcolors, gamelevelend without
  finalcolors, and attempts that did not log exactly six sublevels).
- Converts raw timestamp differences from microseconds to milliseconds.
  (Verified: one session spans ~630 s total with 24-128 s per level.)
- Counts selected, deselected, and final selected colors for each sublevel.
- Flags sublevels where the participant submitted NO selection. The game
  records these as magnitude exactly 1.000, i.e. the maximum possible
  error, so they are non-responses rather than perceptual judgements.
  They are KEPT (see KEEP_NONRESPONSE_SUBLEVELS) but flagged so any
  analysis can exclude them with one filter.
- Calculates Delta E 76 and Delta E 2000 between the base color and each
  of the six distinct final colors.
- Removes accidental zero-chosen attempts using the filtering rules below.
- Writes a payload_qc sheet so structural assumptions are verifiable on
  every new batch of participants rather than assumed.

Expected project structure:
    color-perception/
    └── data_color_crush/
        ├── this script
        ├── "-USER DATA-"/
        │   ├── participant_file_1.txt
        │   └── ...
        └── excel_files/
            ├── combined_final_results.xlsx
            └── discarded_results.xlsx
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76


# ============================================================
# PATH SETUP
# ============================================================

project_folder = Path(__file__).resolve().parent

# TODO: point this at the folder holding the participant .txt files.
input_folder = project_folder / "users_23_juli_2026"
output_folder = project_folder / "excel_files"

output_file = output_folder / "combined_final_results.xlsx"
discarded_output_file = output_folder / "discarded_results.xlsx"


# ============================================================
# PAYLOAD STRUCTURE  (see module docstring)
# ============================================================

PAYLOAD_SLOT_COUNT = 8

# Number of sublevels actually played per level attempt.
EXPECTED_FINAL_COLORS_PER_LEVEL = 6

MAIN_COLOR_LEVELS = 8

# The six distinct payload slots, in the order the sublevels are played.
# Slots 3 and 7 are excluded because they duplicate slots 1 and 5.
AXIS_SLOTS = [0, 1, 2, 4, 5, 6]

# Payload slots that duplicate an earlier slot: {duplicate: original}
DUPLICATE_SLOTS = {3: 1, 7: 5}

# slot -> (axis in the game's Lab-like space, sign)
SLOT_AXIS = {
    0: ("a", +1),
    1: ("L", +1),
    2: ("b", +1),
    3: ("L", +1),   # duplicate of slot 1
    4: ("a", -1),
    5: ("L", -1),
    6: ("b", -1),
    7: ("L", -1),   # duplicate of slot 5
}

# The game's UI compass label for each slot, kept for the spiderweb plots.
COMPASS_LABELS = {
    0: "U+",
    1: "L1+",
    2: "V+",
    3: "L2+",
    4: "U-",
    5: "L1-",
    6: "V-",
    7: "L2-",
}

# A direction magnitude of 1.0 corresponds to this many units along the
# axis in the game's own Lab-like space. Derived from the logged Lab
# tuples: displacement == magnitude * 11 for every slot in every attempt.
GAME_LAB_UNITS_PER_MAGNITUDE = 11.0

# The game logs magnitude exactly 1.000 when nothing was submitted for a
# sublevel. Anything at or above this is treated as sitting on the ceiling.
MAGNITUDE_CEILING = 0.999


# ============================================================
# FILTERING SETTINGS
# ============================================================

# Attempts with 0 colors chosen, where another attempt with the same base
# color has >0 chosen, are removed.
REMOVE_ZERO_REPEATED_ATTEMPTS = True

# Remove attempts with 0 colors chosen if the attempt took under 30 s.
REMOVE_FAST_ZERO_ATTEMPTS = True

# Remove ALL attempts with 0 colors chosen, regardless of timing.
# NOTE: while this is True the two rules above can never fire on their own,
# so the discard_reason will always read zero_chosen_attempt_all_removed.
# Individual rule flags are still written as separate columns.
REMOVE_ALL_ZERO_CHOSEN_ATTEMPTS = True

# Drop whole attempts that did not log exactly EXPECTED_FINAL_COLORS_PER_LEVEL
# sublevels. The payload still has its usual 8 slots in these cases, but the
# per-sublevel timings and selection counts are matched to the payload by
# POSITION, so a different number of sublevels means that matching is not
# trustworthy. Dropped attempts are recorded in discarded_results.xlsx.
DROP_ATTEMPTS_WITH_UNEXPECTED_SUBLEVEL_COUNT = True

# Keep individual sublevels that had no selection submitted. They are
# flagged with sublevel_is_nonresponse so the analysis can exclude them.
# Set to False to drop those single sublevels at source instead.
KEEP_NONRESPONSE_SUBLEVELS = True

# Treat the 000000 Color ID block as practice rather than a real level.
TREAT_000000_BLOCK_AS_TUTORIAL = True

# 30 seconds, expressed in the millisecond units of whole_level_duration_ms.
FAST_ZERO_ATTEMPT_MAX_TIME_MS = 30_000

ZERO_CHOSEN_COLOR_TOTAL = 0


# ============================================================
# REGEX PATTERNS
# ============================================================

USER_RE = re.compile(r"^USER:\s*(.+?)\s*$", re.MULTILINE)

DEMOGRAPHICS_RE = re.compile(
    r"Subcollection:\s*demographics.*?Data:\s*(\{.*?\})\s*(?=\n\s*Subcollection:|\Z)",
    re.DOTALL,
)

# One tolerant pattern for BOTH the direction tuples and the logged Lab
# tuples. In the current logs directions use period decimals and Lab uses
# comma decimals, but that is a locale artefact of the recording device,
# so neither separator is assumed. Tuples are assigned by POSITION:
# the first 8 are directions, the next 8 are logged Lab.
TUPLE_RE = re.compile(
    r"\(\s*(-?[\d]+[.,]?[\d]*)\s*;\s*(-?[\d]+[.,]?[\d]*)\s*;\s*(-?[\d]+[.,]?[\d]*)\s*\)"
)

HEX_RE = re.compile(r"\b[A-Fa-f0-9]{6}\b")


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

TUTORIAL_COLORS = {"000000"}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_decimal(value: str) -> float | None:
    """
    Parse a number that may use either a comma or a period as its decimal
    separator: '85,695' and '85.695' both become 85.695.

    Written this way because a single finalcolors payload mixes both
    conventions, and which one appears where depends on the participant's
    device locale rather than on the meaning of the field.
    """
    text = str(value).strip()

    if not text:
        return None

    # If both separators appear, assume the comma groups thousands.
    if "," in text and "." in text:
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def load_json_object(text: str) -> dict[str, Any] | None:
    """Load a JSON object safely."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def safe_sheet_name(name: str) -> str:
    """Excel sheet names cannot contain some characters and must be <= 31 chars."""
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return cleaned[:31]


def extract_levels_completed_from_filename(path: Path) -> int | None:
    """
    Extract the final number from filenames like
    20260420_094350_uuid_colors_processed_9.txt -> 9
    """
    match = re.search(r"_colors_processed_(\d+)$", path.stem)

    if match:
        return int(match.group(1))

    return None


def split_log_line(log: str) -> tuple[int | None, str | None, str]:
    """
    Split a log line like '41242,gamelevelbegun,DE3B62' into
    timestamp, event type and payload.
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
    """Convert a hex color to RGB in 0-1, shaped (1, 1, 3) for skimage."""
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
    """Convert a hex color to ordinary 0-255 RGB values."""
    if not hex_color:
        return None, None, None

    hex_color = hex_color.strip().replace("#", "")

    if not re.fullmatch(r"[A-Fa-f0-9]{6}", hex_color):
        return None, None, None

    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def hex_to_lab_skimage(
    hex_color: str | None,
) -> tuple[float | None, float | None, float | None]:
    """Convert a hex color to CIELAB using scikit-image (sRGB, D65)."""
    rgb = hex_to_rgb01(hex_color)

    if rgb is None:
        return None, None, None

    lab = rgb2lab(rgb)[0, 0]

    return float(lab[0]), float(lab[1]), float(lab[2])


def deltaE_from_hex(
    base_hex: str | None,
    final_hex: str | None,
) -> tuple[float | None, float | None]:
    """Calculate Delta E 76 and Delta E 2000 between two hex colors."""
    base_rgb = hex_to_rgb01(base_hex)
    final_rgb = hex_to_rgb01(final_hex)

    if base_rgb is None or final_rgb is None:
        return None, None

    base_lab = rgb2lab(base_rgb)
    final_lab = rgb2lab(final_rgb)

    return (
        float(deltaE_cie76(base_lab, final_lab)[0, 0]),
        float(deltaE_ciede2000(base_lab, final_lab)[0, 0]),
    )


def parse_demographics(full_text: str, source_file: Path) -> dict[str, Any]:
    """Extract participant-level demographic data."""
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

    for key, value in zip(demo.get("keys", []), demo.get("values", [])):
        clean_key = str(key).strip().replace(" ", "_")

        # Normalise the free-text demographic answers so that trivial
        # variants ('blue' vs 'Blue ') do not become separate groups when
        # more participants arrive.
        if isinstance(value, str):
            value = value.strip()

        out[clean_key] = value

    if "age" in out:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")

    return out


# ============================================================
# LOG STREAM
# ============================================================

def build_combined_log_stream(full_text: str) -> list[dict[str, Any]]:
    """
    Build one combined log stream directly from the raw text.

    Color ID blocks are storage metadata only; they are NOT level
    boundaries. A single level attempt can begin in one block and finish
    in another, so the true gameplay order comes from the numeric
    timestamp at the start of each log line.
    """
    log_rows: list[dict[str, Any]] = []

    current_color_block_id = None
    current_block_index = -1
    current_block_session_timestamp = None

    file_order = 0

    color_id_re = re.compile(r"^\s*Color ID:\s*([A-Fa-f0-9]{6})\s*$")
    timestamp_re = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')
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

        timestamp_ms, event_type, payload = split_log_line(log_match.group(1))

        if timestamp_ms is None or event_type is None:
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
                "in_tutorial_block": current_color_block_id in TUTORIAL_COLORS,
                "log": log_match.group(1),
            }
        )

        file_order += 1

    # Gameplay order comes from the numeric timestamp, not from block order.
    log_rows.sort(key=lambda row: (row["timestamp_ms"], row["file_order"]))

    return log_rows


# ============================================================
# PAYLOAD PARSING
# ============================================================

def parse_payload_slots(payload: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse all eight slots of a finalcolors payload.

    Returns:
        slots  -- one dict per payload slot (0-7), in payload order
        checks -- structural QC results for this payload
    """
    # Hex colors only ever appear before the first tuple, so restricting
    # the search avoids a 6-digit number inside a tuple being read as a
    # color.
    head = payload.split("(", 1)[0]
    hex_colors = [h.upper() for h in HEX_RE.findall(head)]

    tuples = [
        tuple(parse_decimal(v) for v in raw)
        for raw in TUPLE_RE.findall(payload)
    ]

    direction_tuples = tuples[:PAYLOAD_SLOT_COUNT]
    logged_lab_tuples = tuples[PAYLOAD_SLOT_COUNT : 2 * PAYLOAD_SLOT_COUNT]

    checks: dict[str, Any] = {
        "n_hex_found": len(hex_colors),
        "n_tuples_found": len(tuples),
        "n_direction_tuples": len(direction_tuples),
        "n_logged_lab_tuples": len(logged_lab_tuples),
        "hex_count_ok": len(hex_colors) == PAYLOAD_SLOT_COUNT,
        "tuple_count_ok": len(tuples) == 2 * PAYLOAD_SLOT_COUNT,
        "n_distinct_hex": len(set(hex_colors)),
    }

    # Verify the duplicate-slot structure the whole axis mapping rests on.
    duplicates_ok = True

    for duplicate_slot, original_slot in DUPLICATE_SLOTS.items():
        if max(duplicate_slot, original_slot) < len(hex_colors):
            match = hex_colors[duplicate_slot] == hex_colors[original_slot]
            checks[f"slot{duplicate_slot}_equals_slot{original_slot}"] = match
            duplicates_ok = duplicates_ok and match
        else:
            checks[f"slot{duplicate_slot}_equals_slot{original_slot}"] = None
            duplicates_ok = False

    checks["duplicate_slots_ok"] = duplicates_ok

    # The base color in the game's own Lab-like space. Each slot differs
    # from the base in exactly one coordinate, so the coordinate-wise
    # median across the eight slots recovers the base.
    base_game_lab: list[float | None] = [None, None, None]

    if logged_lab_tuples:
        for axis_index in range(3):
            values = [
                t[axis_index]
                for t in logged_lab_tuples
                if t[axis_index] is not None
            ]
            if values:
                base_game_lab[axis_index] = float(np.median(values))

    checks["base_game_lab_L"] = base_game_lab[0]
    checks["base_game_lab_a"] = base_game_lab[1]
    checks["base_game_lab_b"] = base_game_lab[2]

    axis_index_by_name = {"L": 0, "a": 1, "b": 2}

    slots: list[dict[str, Any]] = []
    worst_consistency_error = 0.0

    for slot in range(PAYLOAD_SLOT_COUNT):
        slot_hex = hex_colors[slot] if slot < len(hex_colors) else None

        direction_x = direction_y = direction_z = None
        magnitude = None

        if slot < len(direction_tuples):
            direction_x, direction_y, direction_z = direction_tuples[slot]

            if None not in (direction_x, direction_y, direction_z):
                magnitude = float(
                    np.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
                )

        logged_lab_L = logged_lab_a = logged_lab_b = None

        if slot < len(logged_lab_tuples):
            logged_lab_L, logged_lab_a, logged_lab_b = logged_lab_tuples[slot]

        axis, axis_sign = SLOT_AXIS[slot]
        axis_label = f"{'+' if axis_sign > 0 else '-'}{axis}"

        # Cross-check: the displacement the game logged along this slot's
        # axis should equal magnitude * GAME_LAB_UNITS_PER_MAGNITUDE.
        logged_axis_displacement = None
        consistency_error = None

        logged_triple = (logged_lab_L, logged_lab_a, logged_lab_b)
        axis_index = axis_index_by_name[axis]

        if (
            logged_triple[axis_index] is not None
            and base_game_lab[axis_index] is not None
        ):
            logged_axis_displacement = (
                logged_triple[axis_index] - base_game_lab[axis_index]
            )

            if magnitude is not None:
                expected = magnitude * GAME_LAB_UNITS_PER_MAGNITUDE * axis_sign
                consistency_error = abs(logged_axis_displacement - expected)
                worst_consistency_error = max(
                    worst_consistency_error, consistency_error
                )

        slots.append(
            {
                "payload_slot": slot,
                "compass_label": COMPASS_LABELS[slot],
                "axis": axis,
                "axis_sign": axis_sign,
                "axis_label": axis_label,
                "is_duplicate_slot": slot in DUPLICATE_SLOTS,
                "hex": slot_hex,
                "direction_x": direction_x,
                "direction_y": direction_y,
                "direction_z": direction_z,
                "direction_magnitude": magnitude,
                "response_magnitude_game_units": (
                    magnitude * GAME_LAB_UNITS_PER_MAGNITUDE
                    if magnitude is not None
                    else None
                ),
                "magnitude_at_ceiling": (
                    magnitude >= MAGNITUDE_CEILING
                    if magnitude is not None
                    else None
                ),
                "logged_game_lab_L": logged_lab_L,
                "logged_game_lab_a": logged_lab_a,
                "logged_game_lab_b": logged_lab_b,
                "logged_axis_displacement": logged_axis_displacement,
                "magnitude_consistency_error": consistency_error,
            }
        )

    checks["worst_magnitude_consistency_error"] = worst_consistency_error
    checks["magnitude_scale_ok"] = worst_consistency_error < 0.05

    return slots, checks


def build_sublevel_rows(
    payload: str,
    base_color: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build one row per SUBLEVEL (six rows) from a finalcolors payload.

    Sublevel i is the i-th distinct payload slot (AXIS_SLOTS), which is
    both the order the sublevels were played and a fixed axis. The two
    duplicate slots are dropped rather than being read as extra colors.
    """
    slots, checks = parse_payload_slots(payload)

    base_lab_L, base_lab_a, base_lab_b = hex_to_lab_skimage(base_color)

    rows: list[dict[str, Any]] = []

    for sublevel_index, slot_number in enumerate(AXIS_SLOTS):
        slot = slots[slot_number]
        final_hex = slot["hex"]

        final_r, final_g, final_b = hex_to_rgb255(final_hex)

        (
            final_lab_L_from_hex,
            final_lab_a_from_hex,
            final_lab_b_from_hex,
        ) = hex_to_lab_skimage(final_hex)

        deltaE76, deltaE2000 = deltaE_from_hex(base_color, final_hex)

        rows.append(
            {
                # Position in the level: 0-5, in the order actually played.
                "sublevel_index": sublevel_index,
                # Kept under the old name so existing code and sheets that
                # refer to final_index keep working.
                "final_index": sublevel_index,
                "payload_slot": slot_number,
                "compass_label": slot["compass_label"],
                "axis": slot["axis"],
                "axis_sign": slot["axis_sign"],
                "axis_label": slot["axis_label"],

                "final_hex": final_hex,
                "final_R": final_r,
                "final_G": final_g,
                "final_B": final_b,

                # CIELAB from the hex that was actually displayed.
                "base_lab_L_from_hex": base_lab_L,
                "base_lab_a_from_hex": base_lab_a,
                "base_lab_b_from_hex": base_lab_b,
                "final_lab_L_from_hex": final_lab_L_from_hex,
                "final_lab_a_from_hex": final_lab_a_from_hex,
                "final_lab_b_from_hex": final_lab_b_from_hex,

                # The game's own logged values. NOT CIELAB for a and b.
                "logged_game_lab_L": slot["logged_game_lab_L"],
                "logged_game_lab_a": slot["logged_game_lab_a"],
                "logged_game_lab_b": slot["logged_game_lab_b"],
                "logged_axis_displacement": slot["logged_axis_displacement"],

                # Outcome measures.
                "deltaE76": deltaE76,
                "deltaE2000": deltaE2000,
                "direction_magnitude": slot["direction_magnitude"],
                "response_magnitude": slot["response_magnitude_game_units"],
                "magnitude_at_ceiling": slot["magnitude_at_ceiling"],
                "magnitude_consistency_error": slot["magnitude_consistency_error"],
            }
        )

    return rows, checks


# ============================================================
# PARTICIPANT FILE PARSING
# ============================================================

def parse_participant_final_results(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
    dict[str, int],
]:
    """
    Extract demographics and finalcolors results from one participant file.

    Returns:
        demographics
        kept_long_rows
        kept_compass_rows
        payload_qc_rows
        discarded_level_attempts
        is_empty_file
        file_notes -- counts for the console summary, never written to a sheet
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    demographics = parse_demographics(text, path)

    level_attempt_events: list[dict[str, Any]] = []
    kept_long_rows: list[dict[str, Any]] = []
    kept_compass_rows: list[dict[str, Any]] = []
    payload_qc_rows: list[dict[str, Any]] = []
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
    active_in_tutorial_block = False

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

    # Counted for the console summary only; never written to a sheet.
    tutorial_attempts_skipped = [0]

    # Safety check: a finalcolors inside the tutorial block would mean a
    # COMPLETED attempt is being dropped, which would be real data loss
    # rather than tutorial noise. Never happens in the current logs.
    tutorial_finalcolors = sum(
        1
        for r in log_rows
        if r["in_tutorial_block"] and (r["event_type"] or "").lower() == "finalcolors"
    )

    def reset_attempt_state() -> None:
        nonlocal active_level, active_in_tutorial_block
        nonlocal current_base_color, current_level_start_ms
        nonlocal current_level_start_file_order
        nonlocal current_sublevel_starts, current_sublevel_end_times
        nonlocal current_sublevel_selected_sets
        nonlocal current_sublevel_selected_action_counts
        nonlocal current_sublevel_deselected_action_counts
        nonlocal current_sublevel_total_action_counts
        nonlocal console_output_count, console_error_count

        active_level = False
        active_in_tutorial_block = False
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

    for row in log_rows:
        timestamp_ms = row["timestamp_ms"]
        event_type_lower = (row["event_type"] or "").lower()
        payload = row["payload"]

        # ----------------------------------------------------
        # Start of a color level attempt
        # ----------------------------------------------------
        if event_type_lower == "gamelevelbegun":
            if active_level:
                discarded_level_attempts.append(
                    {
                        **demographics,
                        "discard_reason": "new_gamelevelbegun_before_previous_finalcolors",
                        "level_number": LEVEL_BY_BASE_COLOR.get(current_base_color),
                        "base_color": current_base_color,
                        "in_tutorial_block": active_in_tutorial_block,
                        "level_start_ms": current_level_start_ms,
                        "level_start_file_order": current_level_start_file_order,
                        "ended_before_finalcolors_file_order": row["file_order"],
                    }
                )

            reset_attempt_state()

            if re.fullmatch(r"[A-Fa-f0-9]{6}", payload):
                current_base_color = payload.upper()
            else:
                current_base_color = None

            if (
                current_base_color is None
                or current_base_color not in LEVEL_BY_BASE_COLOR
            ):
                current_base_color = None
                continue

            # The 000000 block is the forced tutorial. It contains no real
            # data, so it is dropped entirely and never written to any
            # output sheet. It does log a real base color, so the check has
            # to be on the block rather than on the color.
            if TREAT_000000_BLOCK_AS_TUTORIAL and row["in_tutorial_block"]:
                tutorial_attempts_skipped[0] += 1
                current_base_color = None
                continue

            active_level = True
            active_in_tutorial_block = row["in_tutorial_block"]

            current_level_start_ms = timestamp_ms
            current_level_start_file_order = row["file_order"]
            current_level_started_in_color_block_id = row["color_block_id"]
            current_level_started_in_block_index = row["block_index"]
            current_level_started_session_timestamp = row["block_session_timestamp"]

            continue

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

            if generated_index == 0 and timestamp_ms is not None:
                current_sublevel_starts.append(timestamp_ms)
                current_sublevel_end_times.append(None)
                current_sublevel_selected_sets.append(set())
                current_sublevel_selected_action_counts.append(0)
                current_sublevel_deselected_action_counts.append(0)
                current_sublevel_total_action_counts.append(0)

            continue

        # ----------------------------------------------------
        # Color selected / deselected inside the current sublevel
        # ----------------------------------------------------
        if event_type_lower in ("colorselected", "colordeselected"):
            if current_sublevel_selected_sets:
                index = len(current_sublevel_selected_sets) - 1
                selected_id = str(payload).strip()

                if event_type_lower == "colorselected":
                    if selected_id:
                        current_sublevel_selected_sets[index].add(selected_id)
                    current_sublevel_selected_action_counts[index] += 1
                else:
                    if selected_id:
                        current_sublevel_selected_sets[index].discard(selected_id)
                    current_sublevel_deselected_action_counts[index] += 1

                current_sublevel_total_action_counts[index] += 1

            continue

        # ----------------------------------------------------
        # End of one sublevel
        # ----------------------------------------------------
        if event_type_lower == "colorssubmitted":
            if current_sublevel_end_times and timestamp_ms is not None:
                index = len(current_sublevel_end_times) - 1

                if current_sublevel_end_times[index] is None:
                    current_sublevel_end_times[index] = timestamp_ms

            continue

        # ----------------------------------------------------
        # End of the whole color level attempt
        # ----------------------------------------------------
        if event_type_lower == "finalcolors":
            # The sixth sublevel has no colorssubmitted; it ends here.
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
                        len(s) for s in current_sublevel_selected_sets
                    ],
                    "sublevel_selected_action_counts": current_sublevel_selected_action_counts.copy(),
                    "sublevel_deselected_action_counts": current_sublevel_deselected_action_counts.copy(),
                    "sublevel_total_action_counts": current_sublevel_total_action_counts.copy(),

                    "n_sublevels_logged": len(current_sublevel_starts),

                    "console_output_count": console_output_count,
                    "console_error_count": console_error_count,

                    "log_index": row["file_order"],
                    "payload": payload,
                }
            )

            reset_attempt_state()
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

            reset_attempt_state()
            continue

    # --------------------------------------------------------
    # Turn completed attempts into sublevel rows
    # --------------------------------------------------------
    attempt_counter_by_base_color: dict[str, int] = {}

    for event in level_attempt_events:
        base_color = event["base_color"]

        if base_color not in LEVEL_BY_BASE_COLOR:
            continue

        level_number = LEVEL_BY_BASE_COLOR[base_color]

        attempt_counter_by_base_color[base_color] = (
            attempt_counter_by_base_color.get(base_color, 0) + 1
        )
        attempt_number = attempt_counter_by_base_color[base_color]

        # An attempt that did not log exactly six sublevels cannot have its
        # timings and selection counts matched to the payload slots by
        # position, so the whole attempt is dropped rather than kept with
        # misaligned per-sublevel data.
        if (
            DROP_ATTEMPTS_WITH_UNEXPECTED_SUBLEVEL_COUNT
            and event["n_sublevels_logged"] != EXPECTED_FINAL_COLORS_PER_LEVEL
        ):
            discarded_level_attempts.append(
                {
                    **demographics,
                    "discard_reason": "unexpected_sublevel_count",
                    "level_number": level_number,
                    "base_color": base_color,
                    "attempt_number": attempt_number,
                    "n_sublevels_logged": event["n_sublevels_logged"],
                    "n_sublevels_expected": EXPECTED_FINAL_COLORS_PER_LEVEL,
                    "level_start_ms": event["level_start_ms"],
                    "whole_level_duration_ms": (
                        (event["finalcolors_timestamp_ms"] - event["level_start_ms"])
                        / 1000
                        if event["level_start_ms"] is not None
                        and event["finalcolors_timestamp_ms"] is not None
                        else None
                    ),
                    "total_final_selected_colors": sum(
                        c
                        for c in event["sublevel_final_selected_counts"]
                        if c is not None
                    ),
                }
            )
            continue

        sublevel_rows, payload_checks = build_sublevel_rows(
            event["payload"],
            base_color,
        )

        payload_qc_rows.append(
            {
                "participant_uuid": event["participant_uuid"],
                "source_file": event["source_file"],
                "level_number": level_number,
                "attempt_number": attempt_number,
                "base_color": base_color,
                "n_sublevels_logged": event["n_sublevels_logged"],
                "n_sublevels_expected": EXPECTED_FINAL_COLORS_PER_LEVEL,
                "sublevel_count_ok": (
                    event["n_sublevels_logged"] == EXPECTED_FINAL_COLORS_PER_LEVEL
                ),
                **payload_checks,
            }
        )

        # Compass row: all eight slots, kept for the spiderweb plots.
        all_slots, _ = parse_payload_slots(event["payload"])

        compass_row: dict[str, Any] = {
            "participant_uuid": event["participant_uuid"],
            "source_file": event["source_file"],
            "level_number": level_number,
            "attempt_number": attempt_number,
            "base_color": base_color,
        }

        for slot in all_slots:
            prefix = f"dir_{slot['payload_slot']}_{slot['compass_label']}"
            compass_row[f"{prefix}_axis_label"] = slot["axis_label"]
            compass_row[f"{prefix}_hex"] = slot["hex"]
            compass_row[f"{prefix}_x"] = slot["direction_x"]
            compass_row[f"{prefix}_y"] = slot["direction_y"]
            compass_row[f"{prefix}_z"] = slot["direction_z"]
            compass_row[f"{prefix}_magnitude"] = slot["direction_magnitude"]
            compass_row[f"{prefix}_is_duplicate"] = slot["is_duplicate_slot"]

        kept_compass_rows.append(compass_row)

        # Attempt-level timing and action totals.
        sublevel_start_times = event["sublevel_start_times"]
        sublevel_end_times = event["sublevel_end_times"]
        sublevel_final_selected_counts = event["sublevel_final_selected_counts"]
        sublevel_selected_action_counts = event["sublevel_selected_action_counts"]
        sublevel_deselected_action_counts = event["sublevel_deselected_action_counts"]
        sublevel_total_action_counts = event["sublevel_total_action_counts"]

        level_start_ms = event["level_start_ms"]
        finalcolors_timestamp_ms = event["finalcolors_timestamp_ms"]

        whole_level_duration_ms = None

        if level_start_ms is not None and finalcolors_timestamp_ms is not None:
            # Raw timestamps are microseconds.
            whole_level_duration_ms = (
                finalcolors_timestamp_ms - level_start_ms
            ) / 1000

        sublevel_durations: list[float | None] = []
        sublevel_missing_timing: list[bool] = []

        for i in range(EXPECTED_FINAL_COLORS_PER_LEVEL):
            duration = None

            if (
                i < len(sublevel_start_times)
                and i < len(sublevel_end_times)
                and sublevel_start_times[i] is not None
                and sublevel_end_times[i] is not None
            ):
                duration = (sublevel_end_times[i] - sublevel_start_times[i]) / 1000

            sublevel_durations.append(duration)
            sublevel_missing_timing.append(duration is None)

        total_final_selected_colors = sum(
            c for c in sublevel_final_selected_counts if c is not None
        )
        total_selected_actions = sum(
            c for c in sublevel_selected_action_counts if c is not None
        )
        total_deselected_actions = sum(
            c for c in sublevel_deselected_action_counts if c is not None
        )
        total_color_actions = sum(
            c for c in sublevel_total_action_counts if c is not None
        )

        n_nonresponse_sublevels = sum(
            1
            for c in sublevel_final_selected_counts
            if c is not None and c == 0
        )

        for parsed in sublevel_rows:
            sublevel_index = parsed["sublevel_index"]

            sublevel_duration_ms = None
            sublevel_has_missing_timing = True
            final_selected_color_count = None
            selected_action_count = None
            deselected_action_count = None
            total_color_action_count = None

            if sublevel_index < EXPECTED_FINAL_COLORS_PER_LEVEL:
                sublevel_duration_ms = sublevel_durations[sublevel_index]
                sublevel_has_missing_timing = sublevel_missing_timing[sublevel_index]

                if sublevel_index < len(sublevel_final_selected_counts):
                    final_selected_color_count = sublevel_final_selected_counts[
                        sublevel_index
                    ]

                if sublevel_index < len(sublevel_selected_action_counts):
                    selected_action_count = sublevel_selected_action_counts[
                        sublevel_index
                    ]

                if sublevel_index < len(sublevel_deselected_action_counts):
                    deselected_action_count = sublevel_deselected_action_counts[
                        sublevel_index
                    ]

                if sublevel_index < len(sublevel_total_action_counts):
                    total_color_action_count = sublevel_total_action_counts[
                        sublevel_index
                    ]

            # A sublevel with nothing submitted is a non-response. The game
            # scores it as the maximum possible error, so it is not a
            # perceptual judgement.
            is_nonresponse = (
                final_selected_color_count is not None
                and final_selected_color_count == 0
            )

            if not KEEP_NONRESPONSE_SUBLEVELS and is_nonresponse:
                continue

            kept_long_rows.append(
                {
                    **demographics,
                    "level_number": level_number,
                    "attempt_number": attempt_number,
                    "base_color": base_color,
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
                    "n_nonresponse_sublevels": n_nonresponse_sublevels,

                    "sublevel_final_selected_color_count": final_selected_color_count,
                    "sublevel_selected_action_count": selected_action_count,
                    "sublevel_deselected_action_count": deselected_action_count,
                    "sublevel_total_color_action_count": total_color_action_count,
                    "sublevel_chosen_color_count": final_selected_color_count,
                    "sublevel_is_nonresponse": is_nonresponse,

                    **parsed,
                }
            )

    is_empty_file = (
        len(kept_long_rows) == 0 and len(discarded_level_attempts) == 0
    )

    file_notes = {
        "tutorial_attempts_skipped": tutorial_attempts_skipped[0],
        "tutorial_block_finalcolors": tutorial_finalcolors,
    }

    return (
        demographics,
        kept_long_rows,
        kept_compass_rows,
        payload_qc_rows,
        discarded_level_attempts,
        is_empty_file,
        file_notes,
    )


# ============================================================
# WIDE LEVEL ROWS
# ============================================================

def make_wide_level_row(level_df: pd.DataFrame) -> dict[str, Any]:
    """
    Convert one participant's long level data into one wide row, with one
    block of columns per sublevel. Columns are named by the axis the
    sublevel probed as well as its index, so a reader does not have to
    remember the ordering.
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
        "n_nonresponse_sublevels": first.get("n_nonresponse_sublevels"),
    }

    for _, r in level_df.sort_values("sublevel_index").iterrows():
        i = int(r["sublevel_index"])
        axis = r.get("axis_label", "?")
        prefix = f"sub{i}_{axis}"

        row[f"{prefix}_hex"] = r.get("final_hex")
        row[f"{prefix}_deltaE76"] = r.get("deltaE76")
        row[f"{prefix}_deltaE2000"] = r.get("deltaE2000")
        row[f"{prefix}_response_magnitude"] = r.get("response_magnitude")
        row[f"{prefix}_is_nonresponse"] = r.get("sublevel_is_nonresponse")
        row[f"{prefix}_duration_ms"] = r.get("sublevel_duration_ms")
        row[f"{prefix}_final_selected_color_count"] = r.get(
            "sublevel_final_selected_color_count"
        )
        row[f"{prefix}_total_color_action_count"] = r.get(
            "sublevel_total_color_action_count"
        )

    for measure in ("deltaE76", "deltaE2000", "response_magnitude"):
        values = pd.to_numeric(level_df[measure], errors="coerce").dropna()

        if len(values) > 0:
            row[f"mean_{measure}"] = values.mean()
            row[f"median_{measure}"] = values.median()
            row[f"min_{measure}"] = values.min()
            row[f"max_{measure}"] = values.max()
            row[f"std_{measure}"] = values.std()
        else:
            for stat in ("mean", "median", "min", "max", "std"):
                row[f"{stat}_{measure}"] = None

    return row


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

def create_participant_summary_from_cleaned_data(
    all_levels_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create participant_summary from the final cleaned long data."""
    if all_levels_long_df.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []

    for _, participant_df in all_levels_long_df.groupby(
        ["participant_uuid", "source_file"], dropna=False
    ):
        first = participant_df.iloc[0]

        level_attempt_df = participant_df[
            ["level_number", "base_color", "attempt_number"]
        ].drop_duplicates()

        levels_found_list = sorted(
            participant_df["level_number"].dropna().astype(int).unique()
        )

        if levels_found_list:
            max_level_number_found = max(levels_found_list)
            missing_levels_before_highest_list = sorted(
                set(range(1, max_level_number_found + 1)) - set(levels_found_list)
            )
        else:
            max_level_number_found = 0
            missing_levels_before_highest_list = []

        n_sublevels = len(participant_df)
        n_nonresponse = int(participant_df["sublevel_is_nonresponse"].sum())

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

                "total_sublevels_done": n_sublevels,
                "total_level_completions_found": len(level_attempt_df),
                "max_level_reached": max_level_number_found,
                "levels_found": ", ".join(str(l) for l in levels_found_list),
                "missing_levels_before_highest": ", ".join(
                    str(l) for l in missing_levels_before_highest_list
                ),
                "has_missing_levels_before_highest": bool(
                    missing_levels_before_highest_list
                ),
                "completed_base_colors": ", ".join(
                    sorted(
                        participant_df["base_color"].dropna().unique(),
                        key=lambda c: LEVEL_BY_BASE_COLOR.get(c, 999),
                    )
                ),

                "n_nonresponse_sublevels": n_nonresponse,
                "nonresponse_rate": (
                    n_nonresponse / n_sublevels if n_sublevels else np.nan
                ),

                "mean_deltaE2000": pd.to_numeric(
                    participant_df["deltaE2000"], errors="coerce"
                ).mean(),
            }
        )

    return pd.DataFrame(summary_rows)


# ============================================================
# ATTEMPT FILTERING
# ============================================================

def remove_zero_and_fast_attempts(
    all_levels_long_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove attempts that look like accidental or rushed non-attempts.

    Note this operates on whole ATTEMPTS. Individual non-response
    sublevels inside an otherwise real attempt are not removed here; they
    are flagged with sublevel_is_nonresponse instead.
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

    missing = [c for c in needed if c not in all_levels_long_df.columns]

    if missing:
        print(f"Skipping zero/fast attempt filtering, missing columns: {missing}")
        return all_levels_long_df, pd.DataFrame()

    group_cols = ["participant_uuid", "source_file", "level_number", "base_color"]
    attempt_cols = group_cols + ["attempt_number"]

    attempt_summary = (
        all_levels_long_df.groupby(attempt_cols, dropna=False)
        .agg(
            total_final_selected_colors=("total_final_selected_colors", "first"),
            total_color_actions=("total_color_actions", "first"),
            whole_level_duration_ms=("whole_level_duration_ms", "first"),
            mean_deltaE2000=("deltaE2000", "mean"),
            n_sublevels=("sublevel_index", "count"),
        )
        .reset_index()
    )

    attempt_summary["n_attempts_same_color"] = attempt_summary.groupby(
        group_cols, dropna=False
    )["attempt_number"].transform("count")

    attempt_summary["max_chosen_colors_same_color"] = attempt_summary.groupby(
        group_cols, dropna=False
    )["total_final_selected_colors"].transform("max")

    attempt_summary["is_zero_chosen_attempt"] = (
        attempt_summary["total_final_selected_colors"] <= ZERO_CHOSEN_COLOR_TOTAL
    )

    attempt_summary["has_other_attempt_with_choices"] = (
        attempt_summary["n_attempts_same_color"] >= 2
    ) & (attempt_summary["max_chosen_colors_same_color"] > ZERO_CHOSEN_COLOR_TOTAL)

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

    attempt_summary["remove_all_zero_chosen_attempt"] = (
        REMOVE_ALL_ZERO_CHOSEN_ATTEMPTS & attempt_summary["is_zero_chosen_attempt"]
    )

    attempt_summary["remove_attempt"] = (
        attempt_summary["remove_repeated_zero_chosen_attempt"]
        | attempt_summary["remove_fast_zero_chosen_attempt"]
        | attempt_summary["remove_all_zero_chosen_attempt"]
    )

    attempts_to_remove = attempt_summary[attempt_summary["remove_attempt"]].copy()

    if attempts_to_remove.empty:
        print("\nZero/fast attempt filter removed 0 attempts.")
        return all_levels_long_df, pd.DataFrame()

    remove_keys = attempts_to_remove[attempt_cols].copy()
    remove_keys["_remove_attempt"] = True

    marked = all_levels_long_df.merge(remove_keys, on=attempt_cols, how="left")

    removed_long_df = marked[marked["_remove_attempt"] == True].drop(
        columns=["_remove_attempt"]
    )
    filtered_long_df = marked[marked["_remove_attempt"] != True].drop(
        columns=["_remove_attempt"]
    )

    discarded_rows: list[dict[str, Any]] = []

    for attempt_key, attempt_df in removed_long_df.groupby(
        attempt_cols, dropna=False
    ):
        wide_row = make_wide_level_row(attempt_df)

        key_filter = pd.Series(True, index=attempts_to_remove.index)

        for col, value in zip(attempt_cols, attempt_key):
            key_filter = key_filter & (attempts_to_remove[col] == value)

        summary_row = attempts_to_remove[key_filter].iloc[0]

        if summary_row["remove_all_zero_chosen_attempt"]:
            discard_reason = "zero_chosen_attempt_all_removed"
        elif summary_row["remove_repeated_zero_chosen_attempt"]:
            discard_reason = (
                "repeated_zero_chosen_attempt_when_other_attempt_has_choices"
            )
        elif summary_row["remove_fast_zero_chosen_attempt"]:
            discard_reason = "fast_zero_chosen_attempt_under_30_seconds"
        else:
            discard_reason = "zero_or_fast_attempt_removed"

        wide_row.update(
            {
                "discard_reason": discard_reason,
                "n_attempts_same_color": summary_row["n_attempts_same_color"],
                "total_final_selected_colors": summary_row[
                    "total_final_selected_colors"
                ],
                "total_color_actions": summary_row["total_color_actions"],
                "whole_level_duration_ms": summary_row["whole_level_duration_ms"],
                "max_chosen_colors_same_color": summary_row[
                    "max_chosen_colors_same_color"
                ],
                "remove_repeated_zero_chosen_attempt": summary_row[
                    "remove_repeated_zero_chosen_attempt"
                ],
                "remove_fast_zero_chosen_attempt": summary_row[
                    "remove_fast_zero_chosen_attempt"
                ],
                "remove_all_zero_chosen_attempt": summary_row[
                    "remove_all_zero_chosen_attempt"
                ],
                "fast_zero_attempt_max_time_ms": FAST_ZERO_ATTEMPT_MAX_TIME_MS,
            }
        )

        discarded_rows.append(wide_row)

    discarded_attempts_df = pd.DataFrame(discarded_rows)

    print(
        f"\nZero/fast attempt filter removed {len(discarded_attempts_df)} attempts."
    )

    return filtered_long_df, discarded_attempts_df


# ============================================================
# EXCEL FORMATTING
# ============================================================

def format_workbook(workbook: Any, max_width: int = 35) -> None:
    """Freeze the header row, add autofilter and set readable widths."""
    for sheet_name in workbook.sheetnames:
        ws = workbook[sheet_name]
        ws.freeze_panes = "A2"

        if ws.max_row > 1 and ws.max_column > 1:
            ws.auto_filter.ref = ws.dimensions

        for col in ws.columns:
            col_letter = col[0].column_letter
            max_len = max(
                (len(str(c.value)) for c in col[:200] if c.value is not None),
                default=0,
            )
            ws.column_dimensions[col_letter].width = min(
                max(max_len + 2, 10), max_width
            )


# ============================================================
# MAIN
# ============================================================

def create_combined_workbook() -> None:
    """Read all participant .txt files and write the combined workbooks."""
    print(f"Project folder: {project_folder}")
    print(f"Input folder:   {input_folder}")
    print(f"Output file:    {output_file}")

    output_folder.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_folder.glob("*.txt"))

    if not input_files:
        raise SystemExit(f"No .txt files found in: {input_folder}")

    all_long_rows: list[dict[str, Any]] = []
    all_compass_rows: list[dict[str, Any]] = []
    all_payload_qc_rows: list[dict[str, Any]] = []
    empty_files: list[dict[str, Any]] = []
    discarded_level_attempts_all: list[dict[str, Any]] = []
    failed_files: list[tuple[Path, Exception]] = []
    tutorial_skipped_total = 0
    tutorial_completed_warnings: list[tuple[str, int]] = []

    for input_file in input_files:
        try:
            (
                demographics,
                kept_long_rows,
                kept_compass_rows,
                payload_qc_rows,
                discarded_level_attempts,
                is_empty_file,
                file_notes,
            ) = parse_participant_final_results(input_file)

            tutorial_skipped_total += file_notes["tutorial_attempts_skipped"]

            if file_notes["tutorial_block_finalcolors"]:
                tutorial_completed_warnings.append(
                    (input_file.name, file_notes["tutorial_block_finalcolors"])
                )

            if is_empty_file:
                empty_files.append({**demographics, "discard_reason": "empty_file"})

            if discarded_level_attempts:
                discarded_level_attempts_all.extend(discarded_level_attempts)

            if not kept_long_rows:
                continue

            all_long_rows.extend(kept_long_rows)
            all_compass_rows.extend(kept_compass_rows)
            all_payload_qc_rows.extend(payload_qc_rows)

        except Exception as exc:
            failed_files.append((input_file, exc))

    if not all_long_rows:
        raise SystemExit("No finalcolors results were found in any file.")

    all_levels_long_df = pd.DataFrame(all_long_rows)

    all_levels_long_df, repeated_discarded_attempts_df = (
        remove_zero_and_fast_attempts(all_levels_long_df)
    )

    merge_keys = [
        "participant_uuid",
        "source_file",
        "level_number",
        "attempt_number",
        "base_color",
    ]

    all_compass_df = pd.DataFrame(all_compass_rows)

    if not all_compass_df.empty:
        valid_keys = all_levels_long_df[merge_keys].drop_duplicates()
        all_compass_df = all_compass_df.merge(valid_keys, on=merge_keys, how="inner")

    payload_qc_df = pd.DataFrame(all_payload_qc_rows)

    if not payload_qc_df.empty:
        valid_keys = all_levels_long_df[merge_keys].drop_duplicates()
        payload_qc_df = payload_qc_df.merge(valid_keys, on=merge_keys, how="inner")

    participant_summary_df = create_participant_summary_from_cleaned_data(
        all_levels_long_df
    )

    if not repeated_discarded_attempts_df.empty:
        discarded_level_attempts_all.extend(
            repeated_discarded_attempts_df.to_dict("records")
        )

    sort_cols = [
        c
        for c in [
            "level_number",
            "base_color",
            "participant_uuid",
            "attempt_number",
            "sublevel_index",
        ]
        if c in all_levels_long_df.columns
    ]
    all_levels_long_df = all_levels_long_df.sort_values(sort_cols)

    # --------------------------------------------------------
    # Short sheet: one row per participant x level attempt
    # --------------------------------------------------------
    short_group_cols = [
        c
        for c in [
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
        if c in all_levels_long_df.columns
    ]

    all_levels_short_df = (
        all_levels_long_df.groupby(short_group_cols, dropna=False)
        .agg(
            n_sublevels=("sublevel_index", "count"),
            n_nonresponse_sublevels=("sublevel_is_nonresponse", "sum"),

            whole_level_duration_ms=("whole_level_duration_ms", "first"),
            mean_sublevel_duration_ms=("sublevel_duration_ms", "mean"),
            min_sublevel_duration_ms=("sublevel_duration_ms", "min"),
            max_sublevel_duration_ms=("sublevel_duration_ms", "max"),

            total_final_selected_colors=("sublevel_final_selected_color_count", "sum"),
            mean_final_selected_colors=("sublevel_final_selected_color_count", "mean"),
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

            mean_response_magnitude=("response_magnitude", "mean"),
            median_response_magnitude=("response_magnitude", "median"),
            min_response_magnitude=("response_magnitude", "min"),
            max_response_magnitude=("response_magnitude", "max"),
            std_response_magnitude=("response_magnitude", "std"),
        )
        .reset_index()
    )

    # Per-axis error, one column per axis, so the axis analysis needs no
    # reconstruction from Lab values.
    axis_pivot = all_levels_long_df.pivot_table(
        index=merge_keys,
        columns="axis_label",
        values="deltaE2000",
        aggfunc="mean",
    )
    axis_pivot.columns = [f"deltaE2000_axis_{c}" for c in axis_pivot.columns]
    all_levels_short_df = all_levels_short_df.merge(
        axis_pivot.reset_index(),
        on=[k for k in merge_keys if k in all_levels_short_df.columns],
        how="left",
    )

    magnitude_pivot = all_levels_long_df.pivot_table(
        index=merge_keys,
        columns="axis_label",
        values="response_magnitude",
        aggfunc="mean",
    )
    magnitude_pivot.columns = [f"magnitude_axis_{c}" for c in magnitude_pivot.columns]
    all_levels_short_df = all_levels_short_df.merge(
        magnitude_pivot.reset_index(),
        on=[k for k in merge_keys if k in all_levels_short_df.columns],
        how="left",
    )

    if not all_compass_df.empty:
        existing = [c for c in merge_keys if c in all_levels_short_df.columns]
        all_levels_short_df = all_levels_short_df.merge(
            all_compass_df, on=existing, how="left"
        )

    short_sort_cols = [
        c
        for c in ["level_number", "participant_uuid", "attempt_number"]
        if c in all_levels_short_df.columns
    ]
    all_levels_short_df = all_levels_short_df.sort_values(short_sort_cols)

    max_level = int(all_levels_long_df["level_number"].max())
    number_of_level_sheets = max(MAIN_COLOR_LEVELS, max_level)

    print(f"Participants processed: {len(participant_summary_df)}")
    print(f"Sublevel rows kept:     {len(all_levels_long_df)}")
    print(f"Level attempts kept:    {len(all_levels_short_df)}")
    print(f"Max completed level:    {max_level}")

    # --------------------------------------------------------
    # Structural QC report
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PAYLOAD STRUCTURE QC")
    print("=" * 70)

    if payload_qc_df.empty:
        print("  No payload QC rows.")
    else:
        for column, label in [
            ("hex_count_ok", "8 hex colors found"),
            ("tuple_count_ok", "16 tuples found"),
            ("duplicate_slots_ok", "slot3==slot1 and slot7==slot5"),
            ("magnitude_scale_ok", "displacement == magnitude x 11"),
            ("sublevel_count_ok", "6 sublevels logged"),
        ]:
            if column in payload_qc_df.columns:
                n_ok = int(payload_qc_df[column].sum())
                n_total = len(payload_qc_df)
                flag = "OK " if n_ok == n_total else "!! "
                print(f"  {flag}{label}: {n_ok}/{n_total} attempts")

        if "n_distinct_hex" in payload_qc_df.columns:
            print(
                "  distinct hex per payload: "
                f"{sorted(payload_qc_df['n_distinct_hex'].unique().tolist())}"
                " (expected [6])"
            )

    # Is magnitude at the ceiling the same thing as a non-response?
    if {"magnitude_at_ceiling", "sublevel_is_nonresponse"} <= set(
        all_levels_long_df.columns
    ):
        crosstab = pd.crosstab(
            all_levels_long_df["magnitude_at_ceiling"].fillna(False),
            all_levels_long_df["sublevel_is_nonresponse"],
        )
        print("\n  magnitude at ceiling (rows) vs non-response (cols):")
        print(crosstab.to_string().replace("\n", "\n    "))

        ceiling = all_levels_long_df["magnitude_at_ceiling"].fillna(False)
        nonresp = all_levels_long_df["sublevel_is_nonresponse"].fillna(False)

        if (ceiling == nonresp).all():
            print(
                "    -> ceiling and non-response coincide exactly, so the "
                "maximum value is never a real response."
            )
        else:
            print(
                "    -> they do NOT coincide: some ceiling values are real "
                "responses. Keeping non-responses is well justified."
            )

        n_nonresp = int(nonresp.sum())
        print(
            f"\n  non-response sublevels: {n_nonresp}/{len(all_levels_long_df)} "
            f"({100 * n_nonresp / len(all_levels_long_df):.1f}%)"
        )

    # --------------------------------------------------------
    # Write the workbook
    # --------------------------------------------------------
    if output_file.exists():
        print(f"\nOverwriting existing file: {output_file}")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        participant_summary_df.to_excel(
            writer, sheet_name="participant_summary", index=False
        )
        all_levels_short_df.to_excel(
            writer, sheet_name="all_levels_short", index=False
        )
        all_levels_long_df.to_excel(
            writer, sheet_name="all_final_colors_long", index=False
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

            if not level_wide_df.empty and not all_compass_df.empty:
                existing = [c for c in merge_keys if c in level_wide_df.columns]
                level_wide_df = level_wide_df.merge(
                    all_compass_df, on=existing, how="left"
                )

            level_wide_df.to_excel(
                writer,
                sheet_name=safe_sheet_name(f"level_{level_number}"),
                index=False,
            )

        payload_qc_df.to_excel(writer, sheet_name="payload_qc", index=False)

        format_workbook(writer.book)

    print(f"\nCreated combined Excel file:\n  {output_file}")

    # --------------------------------------------------------
    # Discarded workbook
    # --------------------------------------------------------
    empty_files_df = pd.DataFrame(empty_files)
    discarded_level_attempts_df = pd.DataFrame(discarded_level_attempts_all)

    print("\nDiscard summary:")
    print(
        f"  Tutorial-block attempts dropped (not saved anywhere): "
        f"{tutorial_skipped_total}"
    )

    if tutorial_completed_warnings:
        print(
            "  !! WARNING: a COMPLETED attempt was found inside the tutorial\n"
            "     block. That is real data being dropped, not tutorial noise.\n"
            "     Check these files:"
        )
        for name, count in tutorial_completed_warnings:
            print(f"       {name}: {count} finalcolors in the 000000 block")

    print(f"  Empty files: {len(empty_files_df)}")
    print(f"  Discarded attempts: {len(discarded_level_attempts_df)}")

    if (
        not discarded_level_attempts_df.empty
        and "discard_reason" in discarded_level_attempts_df.columns
    ):
        print("\nDiscard reasons:")
        print(
            discarded_level_attempts_df["discard_reason"]
            .value_counts(dropna=False)
            .to_string()
        )

    with pd.ExcelWriter(discarded_output_file, engine="openpyxl") as writer:
        empty_files_df.to_excel(writer, sheet_name="empty_files", index=False)
        discarded_level_attempts_df.to_excel(
            writer, sheet_name="discarded_level_attempts", index=False
        )
        format_workbook(writer.book, max_width=45)

    print(f"\nCreated discarded results file:\n  {discarded_output_file}")

    if failed_files:
        print("\nSome files failed:")
        for path, exc in failed_files:
            print(f"  ERROR: {path.name}: {exc}")


if __name__ == "__main__":
    create_combined_workbook()