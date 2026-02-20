import logging
import warnings
from pathlib import Path
from typing import Optional
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
CUR_PATH = Path(__file__).resolve().parent.parent
STADIUM_DATA_PATH = Path(CUR_PATH, "STADIUM_DATA", "mlb_stadium_data.csv")
FIELD_BOUNDS = {"x_min": 0, "x_max": 250, "y_min": 0, "y_max": 250}

BASE_COORDS = {
    "1B": (150, 175),
    "2B": (125, 150),
    "3B": (100, 175),
    "PM": (125, 177),  # Pitcher's mound
}

PLAYER_COORDS = {
    "1B": (160, 165),
    "2B": (110, 145),
    "SS": (140, 145),
    "3B": (90, 165),
    "LF": (65, 95),
    "CF": (125, 75),
    "RF": (190, 100)
}

BASES_DF = pd.DataFrame.from_dict(BASE_COORDS, orient="index", columns=["x", "y"]).reset_index()
BASES_DF.rename(columns={"index": "base"}, inplace=True)

PLAYERS_DF = pd.DataFrame.from_dict(PLAYER_COORDS, orient="index", columns=["x", "y"]).reset_index()
PLAYERS_DF.rename(columns={"index": "base"}, inplace=True)


def load_stadium_data() -> pd.DataFrame:
    """
    Load stadium coordinate data from CSV file.

    Returns:
        pd.DataFrame: Stadium data with coordinates for all teams.

    Raises:
        FileNotFoundError: If stadium data file does not exist.
    """
    if not STADIUM_DATA_PATH.exists():
        raise FileNotFoundError(f"Stadium data not found at {STADIUM_DATA_PATH}")

    stadium_data = pd.read_csv(STADIUM_DATA_PATH)
    logger.debug(f"Loaded stadium data with {len(stadium_data)} segments")
    return stadium_data


def validate_hit_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare hit data for visualization.

    Args:
        data (pd.DataFrame): Raw hit event data.

    Returns:
        pd.DataFrame: Filtered data with valid coordinates.

    Raises:
        ValueError: If data is empty or missing required columns.
    """
    required_columns = ["hit_location_x", "hit_location_y"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Hit data missing required columns: {missing}")

    clean_data = data[
        data["hit_location_x"].notna() & data["hit_location_y"].notna()
    ].copy()

    if clean_data.empty:
        logger.warning("No valid hit coordinates found in data")

    logger.debug(f"Validated {len(clean_data)} hit records with valid coordinates")
    return clean_data


def _draw_stadium_outline(ax: plt.Axes, team: str, stadium_data: pd.DataFrame) -> None:
    """
    Draw stadium outline and walls on the axes.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
        team (str): Team name (lowercase for lookup).
        stadium_data (pd.DataFrame): Stadium coordinate data.

    Raises:
        ValueError: If no stadium data found for team.
    """
    coords = stadium_data[stadium_data["team"] == team.lower()]

    if coords.empty:
        raise ValueError(f"No stadium data found for team '{team}'")

    segments = coords["segment"].unique()

    for segment in segments:
        seg_data = coords[coords["segment"] == segment]

        if seg_data.empty:
            continue

        seg_color = seg_data["color"].iloc[0] if "color" in seg_data.columns else "grey"

        path = matplotlib.path.Path(seg_data[["x", "y"]].values)
        patch = patches.PathPatch(
            path, facecolor="None", edgecolor=seg_color, lw=1, zorder=3
        )
        ax.add_patch(patch)

    logger.debug(f"Drew stadium outline with {len(segments)} segments")


def _draw_bases(ax: plt.Axes) -> None:
    """
    Draw base markers on the field.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
    """
    sns.scatterplot(
        data=BASES_DF,
        x="x",
        y="y",
        ax=ax,
        color="black",
        s=25,
        marker="D",
        zorder=5,
        legend=False,
    )
    logger.debug("Drew base markers")


def _draw_players(ax: plt.Axes) -> None:
    """
    Draw player markers on the field.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
    """
    sns.scatterplot(
        data=PLAYERS_DF,
        x="x",
        y="y",
        ax=ax,
        color="red",
        s=25,
        marker="o",
        zorder=6,
        legend=False,
    )
    logger.debug("Drew base markers")


def _draw_scatter_plot(ax: plt.Axes, hit_data: pd.DataFrame, size: int) -> None:
    """
    Draw scatter plot of hit locations.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
        hit_data (pd.DataFrame): Hit event data with location coordinates.
        size (int): Marker size for scatter points.
    """
    if hit_data.empty:
        logger.warning("No data to plot for scatter")
        return

    sns.scatterplot(
        data=hit_data,
        x="hit_location_x",
        y="hit_location_y",
        hue="hit_sector" if "hit_sector" in hit_data.columns else None,
        palette="coolwarm" if "hit_sector" in hit_data.columns else None,
        ax=ax,
        s=size,
        color="navy" if "hit_sector" not in hit_data.columns else None,
        zorder=2,
    )
    logger.debug(f"Drew scatter plot with {len(hit_data)} points")


def _draw_kde_plot(ax: plt.Axes, hit_data: pd.DataFrame, cmap: str) -> None:
    """
    Draw KDE (kernel density estimation) contour plot.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
        hit_data (pd.DataFrame): Hit event data with location coordinates.
        cmap (str): Colormap name for KDE visualization.
    """
    if hit_data.empty:
        logger.warning("No data to plot for KDE")
        return

    sns.kdeplot(
        data=hit_data,
        x="hit_location_x",
        y="hit_location_y",
        cmap=cmap,
        ax=ax,
        zorder=1,
    )
    logger.debug("Drew KDE density plot")


def plot_stadium_spraychart(
    data: pd.DataFrame,
    team: str,
    show_scatter: bool = True,
    show_kde: bool = True,
    cmap: str = "coolwarm",
    size: int = 5,
    width: int = 10,
    height: int = 10,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Create an interactive stadium spraychart visualization.

    Plots hit locations on a baseball field, with optional KDE density overlay
    and filtering by hit outcome.

    Args:
        data (pd.DataFrame): Hit event data with columns:
            hit_location_x, hit_location_y, outcome, hit_sector (optional).
        team (str): MLB team name (case-insensitive, e.g., 'Yankees', 'boston').
        outcome (Optional[str]): Filter hits by outcome type.
            Valid values: 'single', 'double', 'triple', 'home_run', 'out', 'in_play'.
            If None, shows all hits. Defaults to None.
        show_scatter (bool, optional): Display scatter plot of hit locations.
            Defaults to True.
        show_kde (bool, optional): Display KDE density contours.
            Defaults to True.
        cmap (str, optional): Colormap for KDE visualization.
            Defaults to 'coolwarm'.
        size (int, optional): Marker size for scatter points. Defaults to 50.
        width (int, optional): Figure width in inches. Defaults to 10.
        height (int, optional): Figure height in inches. Defaults to 10.
        title (Optional[str], optional): Custom title for the chart.
            If None, auto-generates based on team and outcome. Defaults to None.

    Returns:
        plt.Axes: Matplotlib axes object for further customization.

    Raises:
        ValueError: If team not found, data missing required columns, or invalid outcome.
        FileNotFoundError: If stadium data file not found.

    Example:
        >>> hit_data = pd.read_csv("MLB_HIT_DATA_2024.csv")
        >>> ax = plot_stadium_spraychart(
        ...     hit_data,
        ...     team="Yankees",
        ...     outcome="home_run",
        ...     show_kde=True,
        ...     title="Yankees Home Runs - 2024"
        ... )
        >>> plt.show()
    """
    logger.info(
        f"Generating spraychart for {team}"
    )

    # Load and validate data
    stadium_data = load_stadium_data()
    clean_data = validate_hit_data(data)


    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Draw field
    _draw_stadium_outline(ax, team, stadium_data)
    _draw_bases(ax)
    _draw_players(ax)

    # Draw hit data
    if show_scatter:
        _draw_scatter_plot(ax, clean_data, size)

    if show_kde and not clean_data.empty:
        _draw_kde_plot(ax, clean_data, cmap)

    # Configure axes
    ax.set_xlim(FIELD_BOUNDS["x_min"], FIELD_BOUNDS["x_max"])
    ax.set_ylim(FIELD_BOUNDS["y_min"], FIELD_BOUNDS["y_max"])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Labels
    ax.set_xlabel("Distance from Left/Right Field Line (feet)", fontsize=10)
    ax.set_ylabel("Distance from Home Plate (feet)", fontsize=10)

    logger.info(f"Spraychart generated successfully for {team}")

    plt.show()
    return ax