"""Custom exception hierarchy for the fwh_core package."""


class FWHException(Exception):
    """Base exception for fwh_core."""


class ConfigValidationError(FWHException):
    """Exception raised when a config is invalid."""


class DeviceResolutionError(FWHException):
    """Exception raised when a device resolution fails."""
