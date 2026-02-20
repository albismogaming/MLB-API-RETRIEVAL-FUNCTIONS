"""
MLB Play-by-Play (PBP) Data Fetcher Module

This module provides utilities for fetching and caching MLB play-by-play data
from the MLB Stats API, organizing results by season and game.
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import mlbstatsapi
import pandas as pd
from tqdm import tqdm

# Configure module-level logger
logger = logging.getLogger(__name__)

# Suppress library warnings
warnings.filterwarnings("ignore")

# Constants
MIN_VALID_FILE_SIZE = 2048  # Bytes - minimum size for valid PBP data
API_RATE_LIMIT_DELAY = 0.1  # Seconds between API requests
REQUIRED_SCHEDULE_COLUMNS = {"game_pk", "date", "home_team", "away_team"}


def get_mlb_pbp_data(
    year: int,
    cache_dir: Path = Path("MLB_CACHE"),
    force_refresh: bool = False,
    rate_limit_delay: float = API_RATE_LIMIT_DELAY,
) -> Tuple[int, int]:
    """
    Fetch and cache MLB play-by-play data for a given year.

    Args:
        year (int): MLB season year (e.g., 2024).
        cache_dir (Path, optional): Base directory for caching PBP data.
            Year-specific data stored in cache_dir/year/. Defaults to Path("MLB_CACHE").
        force_refresh (bool, optional): If True, re-fetch all games even if cached.
            Defaults to False.
        rate_limit_delay (float, optional): Delay in seconds between API calls.
            Defaults to 0.1.

    Returns:
        Tuple[int, int]: (games_cached, games_skipped)

    Example:
        >>> cached, skipped = get_mlb_pbp_data(year=2024)
        >>> print(f"Cached: {cached}, Skipped: {skipped}")
        Cached: 2430, Skipped: 0
    """
    try:
        schedule_file = f"MLB_SCHEDULE_{year}.csv"
        year_cache_dir = Path(cache_dir) / str(year)
        
        logger.info(f"Loading {year} schedule from {schedule_file}")
        schedule = _load_and_validate_schedule(year)  # or pass schedule_file if you prefer
        
        year_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {year_cache_dir.resolve()}")

        # Initialize cache directory
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir.resolve()}")

        # Initialize API client
        mlb = mlbstatsapi.Mlb()

        # Process games
        logger.info(f"Processing {len(schedule)} games")
        cached_count, skipped_count = _process_games(
            schedule, mlb, cache_dir, force_refresh, rate_limit_delay
        )

        logger.info(
            f"✅ Completed: Cached {cached_count} games, Skipped {skipped_count} games"
        )
        return cached_count, skipped_count

    except FileNotFoundError as e:
        logger.error(f"Schedule file not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid schedule file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching PBP data: {str(e)}")
        raise


def _load_and_validate_schedule(year: int) -> pd.DataFrame:
    """
    Load schedule CSV and validate required columns.

    Args:
        year (int): MLB season year (e.g., 2024).

    Returns:
        pd.DataFrame: Validated schedule data.

    Raises:
        FileNotFoundError: If schedule file does not exist.
        ValueError: If required columns are missing.
    """
    schedule_file = f"MLB_SCHEDULE_{year}.csv"
    schedule_path = Path("MLB_SCHEDULES") / schedule_file

    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

    schedule = pd.read_csv(schedule_path)
    logger.debug(f"Loaded {year} schedule with {len(schedule)} rows")

    # Drop rows with missing critical IDs
    initial_count = len(schedule)
    schedule.dropna(subset=["game_pk", "home_id", "away_id"], inplace=True)
    dropped = initial_count - len(schedule)

    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing IDs")

    # Validate required columns
    missing_columns = REQUIRED_SCHEDULE_COLUMNS - set(schedule.columns)
    if missing_columns:
        raise ValueError(
            f"Schedule CSV missing required columns: {missing_columns}. "
            f"Available columns: {set(schedule.columns)}"
        )

    logger.info(f"Schedule validation passed: {len(schedule)} games for {year}")
    return schedule


def _process_games(
    schedule: pd.DataFrame,
    mlb: mlbstatsapi.Mlb,
    cache_dir: Path,
    force_refresh: bool,
    rate_limit_delay: float,
) -> Tuple[int, int]:
    """
    Process schedule and fetch/cache PBP data for each game.

    Args:
        schedule (pd.DataFrame): Schedule data from CSV.
        mlb (mlbstatsapi.Mlb): MLB API client instance.
        cache_dir (Path): Root cache directory.
        force_refresh (bool): If True, re-fetch all games.
        rate_limit_delay (float): Delay between API calls in seconds.

    Returns:
        Tuple[int, int]: (games_cached, games_skipped)
    """
    season_dirs: Dict[int, Path] = {}
    cached_count = 0
    skipped_count = 0

    with tqdm(
        schedule.itertuples(index=False),
        total=len(schedule),
        desc="Fetching PBP data",
    ) as pbar:
        for row in pbar:
            try:
                game_id = int(row.game_pk)
                game_year = pd.to_datetime(row.date).year
                home_team = row.home_team
                away_team = row.away_team

                # Get or create season directory
                if game_year not in season_dirs:
                    season_dir = cache_dir / f"{game_year}"
                    season_dir.mkdir(parents=True, exist_ok=True)
                    season_dirs[game_year] = season_dir
                else:
                    season_dir = season_dirs[game_year]

                # Check if should skip cached file
                game_file = season_dir / f"GAMEPK_{game_id}_{home_team}_VS_{away_team}.json"
                
                if not force_refresh and _is_valid_cache(game_file):
                    skipped_count += 1
                    pbar.set_postfix(
                        {"cached": cached_count, "skipped": skipped_count}
                    )
                    continue

                # Fetch from API
                game_data = _fetch_game_pbp(mlb, game_id)

                # Save to cache
                _save_game_pbp(game_data, game_file)
                cached_count += 1

                # Rate limiting
                time.sleep(rate_limit_delay)

                pbar.set_postfix({"cached": cached_count, "skipped": skipped_count})

            except Exception as e:
                logger.warning(f"Error processing game {row.game_pk}: {str(e)}")
                continue

    return cached_count, skipped_count


def _is_valid_cache(cache_file: Path) -> bool:
    """
    Check if cached file exists and has valid size.

    Args:
        cache_file (Path): Path to cached JSON file.

    Returns:
        bool: True if file exists and is valid size, False otherwise.
    """
    if not cache_file.exists():
        return False

    try:
        file_size = cache_file.stat().st_size
        return file_size > MIN_VALID_FILE_SIZE
    except Exception as e:
        logger.debug(f"Error checking cache file {cache_file}: {str(e)}")
        return False


def _fetch_game_pbp(mlb: mlbstatsapi.Mlb, game_id: int) -> Dict:
    """
    Fetch play-by-play data for a single game from API.

    Args:
        mlb (mlbstatsapi.Mlb): MLB API client instance.
        game_id (int): Game PK identifier.

    Returns:
        Dict: Play-by-play data as dictionary.

    Raises:
        Exception: If API request fails.
    """
    try:
        pbp_data = _to_dict(mlb.get_game_play_by_play(game_id))
        logger.debug(f"Fetched PBP data for game {game_id}")
        return pbp_data
    except Exception as e:
        logger.error(f"Failed to fetch PBP data for game {game_id}: {str(e)}")
        raise


def _save_game_pbp(game_data: Dict, output_file: Path) -> None:
    """
    Save play-by-play data to JSON cache file.

    Args:
        game_data (Dict): Game PBP data to save.
        output_file (Path): Path where to save JSON file.

    Raises:
        IOError: If unable to write file.
    """
    try:
        with open(output_file, "w") as f:
            json.dump(game_data, f, indent=2)
        logger.debug(f"Saved PBP data to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save PBP data to {output_file}: {str(e)}")
        raise


def _to_dict(obj: Any) -> Any:
    """
    Recursively convert class-based objects or lists into plain Python dictionaries.

    This utility function handles the mlbstatsapi library's custom objects by
    converting them to standard Python dicts for easier manipulation.

    Args:
        obj (Any): Object, list, dict, or primitive to convert.

    Returns:
        Any: Converted data structure with all objects as dicts.
    """
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        return obj


def load_pbp_cache(game_file: Path) -> Dict:
    """
    Load cached PBP data from JSON file.

    Utility function for loading cached game data after it's been fetched.

    Args:
        game_file (Path): Path to cached JSON file.

    Returns:
        Dict: Play-by-play data.

    Raises:
        FileNotFoundError: If cache file does not exist.
        json.JSONDecodeError: If JSON is malformed.
    """
    try:
        with open(game_file, "r") as f:
            data = json.load(f)
        logger.debug(f"Loaded PBP cache from {game_file}")
        return data
    except FileNotFoundError:
        logger.error(f"Cache file not found: {game_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in cache file {game_file}: {str(e)}")
        raise
