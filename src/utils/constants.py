"""Site-level constants for DE-Har."""

from datetime import timezone, timedelta

SITE_LAT = 47.9344
SITE_LON = 7.6010
SITE_NAME = "DE-Har"
SITE_ELEVATION_M = 201

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
