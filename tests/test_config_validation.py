import pytest

from itbound.config import ConfigError, validate_config


def test_validate_config_missing_required_keys():
    with pytest.raises(ConfigError):
        validate_config({})
