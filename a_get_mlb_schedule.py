"""
MLB Schedule Data Fetcher Module

This module provides utilities for fetching and processing MLB schedule data
from the MLB Stats API, saving results to CSV format.
"""

from pathlib import Path
from typing import Dict, List, Any
import logging
import pandas as pd
from tqdm import tqdm
import mlbstatsapi

# Configure module-level logger
logger = logging.getLogger(__name__)


def get_schedule_data(
    start_date: str = "",
    end_date: str = "",
    gametypes: str = "",
    filename: str = "MLB_SCHEDULE_",
    output_path: Path = Path("MLB_SCHEDULES"),
) -> Path:
    """
    Fetch MLB schedule data from the MLB Stats API and save to CSV.

    This function retrieves schedule data for a specified date range and optional
    game types, enriches it with team abbreviations, and saves the results to a
    CSV file. The function uses caching for efficient team data retrieval.

    Args:
        start_date (str, optional): Start date for schedule in 'YYYY-MM-DD' format.
            Defaults to "" (no filter).
        end_date (str, optional): End date for schedule in 'YYYY-MM-DD' format.
            Defaults to "" (no filter).
        gametypes (str, optional): Comma-separated game types (e.g., 'R', 'F', 'D').
            Defaults to "" (all game types).
        filename (str, optional): Base filename for output CSV.
            Defaults to "MLB_SCHEDULE_".
        output_path (Path, optional): Directory to save CSV file.
            Defaults to Path("MLB_SCHEDULES").

    Returns:
        Path: Absolute path to the saved CSV file.

    Raises:
        ValueError: If the schedule data is empty or malformed.
        IOError: If unable to write to the output file.
        KeyError: If required fields are missing from API response.

    Example:
        >>> output_file = get_schedule_data(
        ...     start_date="2024-04-01",
        ...     end_date="2024-10-31",
        ...     gametypes="R"
        ... )
        >>> print(output_file)
        /path/to/MLB_SCHEDULES/MLB_SCHEDULE_2024.csv
    """
    try:
        # Initialize API client
        mlb = mlbstatsapi.Mlb()
        logger.info(
            f"Fetching MLB schedule from {start_date or 'start'} "
            f"to {end_date or 'end'}"
        )

        # Fetch schedule data
        schedule = _fetch_schedule(mlb, start_date, end_date, gametypes)

        if not schedule.get("dates"):
            raise ValueError("No schedule data returned from API")

        # Prepare output directory and file path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        year = start_date[:4] if start_date else "ALL"
        output_file = output_path / f"{filename}{year}.csv"

        # Fetch all teams and build lookup dictionary
        logger.info("Fetching team data")
        all_teams = _fetch_team_abbreviations(mlb)

        # Process games and build rows
        logger.info("Processing schedule data")
        rows = _process_schedule_games(schedule, all_teams)

        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved schedule to {output_file.resolve()}")

        return output_file.resolve()

    except Exception as e:
        logger.error(f"Error fetching schedule data: {str(e)}")
        raise


def _fetch_schedule(
    mlb: mlbstatsapi.Mlb, start_date: str, end_date: str, gametypes: str
) -> Dict[str, Any]:
    """
    Fetch raw schedule data from MLB Stats API.

    Args:
        mlb (mlbstatsapi.Mlb): MLB API client instance.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        gametypes (str): Comma-separated game type codes.

    Returns:
        Dict[str, Any]: Schedule data with nested dates and games.

    Raises:
        Exception: If API request fails.
    """
    try:
        raw_schedule = mlb.get_schedule(
            start_date=start_date, end_date=end_date, gameTypes=gametypes
        )
        # Convert to dict if necessary (mlbstatsapi may return various types)
        schedule = _to_dict(raw_schedule)
        return schedule
    except Exception as e:
        logger.error(f"Failed to fetch schedule from API: {str(e)}")
        raise


def _fetch_team_abbreviations(mlb: mlbstatsapi.Mlb) -> Dict[int, str]:
    """
    Fetch all MLB teams and create ID-to-abbreviation mapping.

    This function makes a single API call to fetch all teams, avoiding the need
    for per-game team lookups.

    Args:
        mlb (mlbstatsapi.Mlb): MLB API client instance.

    Returns:
        Dict[int, str]: Mapping of team ID to team abbreviation.

    Raises:
        Exception: If API request fails or data is malformed.
    """
    try:
        teams_data = _to_dict(mlb.get_teams())
        all_teams = {team["id"]: team["abbreviation"] for team in teams_data}
        logger.debug(f"Loaded {len(all_teams)} teams")
        return all_teams
    except Exception as e:
        logger.error(f"Failed to fetch team data: {str(e)}")
        raise


def _process_schedule_games(
    schedule: Dict[str, Any], all_teams: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Process schedule dates and games into a flat list of row dictionaries.

    Args:
        schedule (Dict[str, Any]): Schedule data from API.
        all_teams (Dict[int, str]): Mapping of team ID to abbreviation.

    Returns:
        List[Dict[str, Any]]: List of game records ready for DataFrame.

    Raises:
        KeyError: If required fields are missing from game data.
    """
    rows = []
    total_games = sum(len(date["games"]) for date in schedule.get("dates", []))

    with tqdm(total=total_games, desc="Processing games") as pbar:
        for date in schedule.get("dates", []):
            for game in date.get("games", []):
                try:
                    row = _extract_game_row(game, all_teams)
                    rows.append(row)
                except KeyError as e:
                    logger.warning(f"Missing field in game data: {str(e)}")
                    continue
                finally:
                    pbar.update(1)

    logger.info(f"Processed {len(rows)} games")
    return rows


def _extract_game_row(game: Dict[str, Any], all_teams: Dict[int, str]) -> Dict[str, Any]:
    """
    Extract relevant fields from a single game record.

    Args:
        game (Dict[str, Any]): Game data from API.
        all_teams (Dict[int, str]): Mapping of team ID to abbreviation.

    Returns:
        Dict[str, Any]: Formatted game record for CSV output.

    Raises:
        KeyError: If required nested fields are missing.
    """
    home_team_id = game["teams"]["home"]["team"]["id"]
    away_team_id = game["teams"]["away"]["team"]["id"]

    home_record = game["teams"]["home"].get("leaguerecord", {})
    away_record = game["teams"]["away"].get("leaguerecord", {})

    return {
        "date": game.get("gamedate", ""),
        "game_pk": game.get("gamepk", ""),
        "home_id": home_team_id,
        "home_team": all_teams.get(home_team_id, "UNKNOWN"),
        "home_record": _format_record(home_record),
        "home_score": game["teams"]["home"].get("score", 0),
        "away_id": away_team_id,
        "away_team": all_teams.get(away_team_id, "UNKNOWN"),
        "away_record": _format_record(away_record),
        "away_score": game["teams"]["away"].get("score", 0),
        "venue_id": game.get("venue", {}).get("id", ""),
        "venue_name": game.get("venue", {}).get("name", ""),
        "status": game.get("status", {}).get("detailedState", ""),
    }


def _format_record(record: Dict[str, Any]) -> str:
    """
    Format a win-loss record into standardized string format.

    Args:
        record (Dict[str, Any]): Record dict with 'wins' and 'losses' keys.

    Returns:
        str: Formatted record string (e.g., "[45 - 35]").
    """
    wins = record.get("wins", 0)
    losses = record.get("losses", 0)
    return f"[{wins} - {losses}]"


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
