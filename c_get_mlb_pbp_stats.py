import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from MLB_CODE.statcast_utils import get_spray_sector

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MIN_FILE_SIZE = 100  # bytes
HOME_TEAM_COORD_X = 125.0
HOME_TEAM_COORD_Y = 199.0


def compile_play_events(
    folder_path: Path, season_year: int
) -> pd.DataFrame:
    """
    Compile play-by-play hit event data from MLB Stats API JSON files.

    Reads all JSON game files from the specified folder, extracts hit events,
    and returns a DataFrame with comprehensive hit metrics and game information.

    Args:
        folder_path (Path): Directory containing JSON game files.
        season_year (Optional[int]): MLB season year. If None, will attempt to infer.

    Returns:
        pd.DataFrame: Compiled hit event data with columns for game info,
            batter/pitcher details, and hit metrics.

    Raises:
        ValueError: If folder_path does not exist, is not a directory,
            no JSON files found, or no hit events extracted.

    Example:
        >>> hit_df = compile_play_events(Path("MLB_CACHE/2024"), season_year=2024)
        >>> print(f"Compiled {len(hit_df)} hit events")
    """
    folder_path = Path(folder_path) / str(season_year)

    if not folder_path.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Starting compilation from {folder_path}")

    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {folder_path}")

    logger.info(f"Found {len(json_files)} JSON files to process")

    data: List[Dict] = []

    for filepath in json_files:
        try:
            _process_game_file(filepath, data)
        except Exception as e:
            logger.warning(f"Error processing {filepath.name}: {str(e)}")
            continue

    if not data:
        raise ValueError("No hit events extracted from any files")

    hit_df = pd.DataFrame(data)
    logger.info(f"✅ Compiled {len(hit_df)} hit events")

    return hit_df


def compile_and_export_play_events(
    folder_path: Path,
    season_year: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Path]:
    """
    Compile hit events and export to CSV.

    Args:
        folder_path (Path): Directory containing JSON game files.
        season_year (Optional[int]): MLB season year for output filename.
        output_dir (Optional[Path]): Directory for output file.
            Defaults to current directory.

    Returns:
        Tuple[pd.DataFrame, Path]: Compiled DataFrame and output file path.

    Raises:
        ValueError: If compilation fails.
        IOError: If unable to write output file.

    Example:
        >>> hit_df, output_path = compile_and_export_play_events(
        ...     Path("MLB_CACHE/2024"), season_year=2024
        ... )
        >>> print(f"Exported to {output_path}")
    """
    hit_df = compile_play_events(folder_path, season_year)

    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    year = season_year or 2024
    output_path = output_dir / f"MLB_HIT_DATA_{year}.csv"

    try:
        hit_df.to_csv(output_path, index=False)
        logger.info(f"✅ Exported {len(hit_df)} rows to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write output file: {str(e)}")
        raise

    return hit_df, output_path


def _process_game_file(filepath: Path, data: List[Dict]) -> None:
    """
    Process a single game JSON file and extract hit events.

    Args:
        filepath (Path): Path to game JSON file.
        data (List[Dict]): List to append extracted hit events to.

    Raises:
        json.JSONDecodeError: If JSON is malformed.
        KeyError: If expected JSON structure is missing.
    """
    # Skip files that are too small (likely empty or corrupted)
    if filepath.stat().st_size < MIN_FILE_SIZE:
        logger.debug(f"Skipping small file: {filepath.name}")
        return

    with open(filepath, "r") as f:
        game_data = json.load(f)

    all_plays = game_data.get("allplays", [])
    playsbyinning = game_data.get("playsbyinning", [])

    if not all_plays:
        logger.debug(f"No plays found in {filepath.name}")
        return

    home_id = _extract_home_team_id(playsbyinning)

    for play in all_plays:
        play_events = play.get("playevents", [])

        # Extract constants for this play
        constants = _extract_play_constants(play, home_id)

        # Extract hit data for each event in the play
        for event in play_events:
            if event.get("details", {}).get("isinplay"):
                hit_data = _extract_hit_data(event)

                if hit_data:
                    full_row = {**constants, **hit_data}
                    data.append(full_row)


def _extract_home_team_id(playsbyinning: List[Dict]) -> Optional[int]:
    """
    Extract home team ID from plays by inning.

    Args:
        playsbyinning (List[Dict]): Plays organized by inning.

    Returns:
        Optional[int]: Home team ID, or None if not found.
    """
    for inning in playsbyinning:
        home_team_hits = inning.get("hits", {}).get("home", [])
        for team in home_team_hits:
            home_id = team.get("team", {}).get("id")
            if home_id is not None:
                return home_id
    return None


def _extract_play_constants(play: Dict, home_id: Optional[int]) -> Dict:
    """
    Extract constant information for a play.

    Args:
        play (Dict): Play data from game JSON.
        home_id (Optional[int]): Home team ID.

    Returns:
        Dict: Dictionary with play metadata.
    """
    about = play.get("about", {})
    result = play.get("result", {})
    matchup = play.get("matchup", {})

    return {
        "start_time": about.get("starttime", ""),
        "end_time": about.get("endtime", ""),
        "home_team": home_id,
        "half_inning": about.get("halfinning", ""),
        "inning": about.get("inning", ""),
        "at_bat_index": about.get("atbatindex", ""),
        "outcome": result.get("eventtype", ""),
        "description": result.get("description", ""),
        "batter_id": matchup.get("batter", {}).get("id"),
        "batter_hand": matchup.get("batside", {}).get("code"),
        "batter_name": matchup.get("batter", {}).get("fullname"),
        "pitcher_id": matchup.get("pitcher", {}).get("id"),
        "pitcher_hand": matchup.get("pitchhand", {}).get("code"),
        "pitcher_name": matchup.get("pitcher", {}).get("fullname"),
    }


def _extract_hit_data(event: Dict) -> Optional[Dict]:
    """
    Extract hit metrics from an in-play event.

    Args:
        event (Dict): Play event data.

    Returns:
        Optional[Dict]: Hit data with spray angle and distance, or None if
            insufficient data.
    """
    hitdata = event.get("hitdata", {})
    coordinates = hitdata.get("coordinates", {})

    coordx = coordinates.get("coordx")
    coordy = coordinates.get("coordy")

    # Validate required coordinates
    if coordx is None or coordy is None:
        return None

    # Calculate spray angle and distance
    delta_x = float(coordx) - HOME_TEAM_COORD_X
    delta_y = float(coordy) - HOME_TEAM_COORD_Y
    spray_angle = np.degrees(np.arctan2(delta_x, delta_y))
    hit_distance = math.sqrt(delta_x**2 + delta_y**2)

    count = event.get("count", {})

    return {
        "count_outs": count.get("outs"),
        "count_balls": count.get("balls"),
        "count_strikes": count.get("strikes"),
        "location": hitdata.get("location"),
        "launch_speed": hitdata.get("launchspeed"),
        "launch_angle": hitdata.get("launchangle"),
        "hit_trajectory": hitdata.get("trajectory"),
        "hit_hardness": hitdata.get("hardness"),
        "total_distance": hitdata.get("totaldistance"),
        "hit_location_x": coordx,
        "hit_location_y": coordy,
        "hit_sector": get_spray_sector(spray_angle),
        "hit_distance": hit_distance,
        "spray_angle": spray_angle,
    }