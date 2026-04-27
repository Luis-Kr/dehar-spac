"""Site-level constants for DE-Har."""

from datetime import timezone, timedelta

SITE_LAT = 47.9344
SITE_LON = 7.6010
SITE_NAME = "DE-Har"
SITE_ELEVATION_M = 201

# Coordinate reference system and tower location in projected coords
SITE_CRS = "EPSG:32632"  # UTM zone 32N
SITE_UTM_X = 395509.71   # easting [m]  — pyproj(7.6010°E, 47.9344°N → EPSG:32632)
SITE_UTM_Y = 5309956.22  # northing [m]

SPECIES_CODES = {
    "C. betulus": "carpinus",
}

# Height thresholds for layer separation (meters)
OVERSTORY_MIN_HEIGHT = 12.0
UNDERSTORY_MAX_HEIGHT = 10.0

# Hydraulic stress threshold
SWP_CRITICAL_MPA = -1.0

# Timezones
# Eddy covariance convention: local standard time without DST
CET = timezone(timedelta(hours=1))
UTC = timezone.utc

# Missing value sentinel used in REddyProc / raw files
MISSING_VALUE_SENTINEL = -9999.0
