"""
Configuration management package for AI Filesystem Finder.

This package provides configuration parsing, validation, and management
functionality for the AI Filesystem Finder application.
"""

from .parser import (
    ConfigParser,
    ConfigParseResult,
    ConfigurationError,
    load_config,
    validate_config_file,
    create_config_template
)

__all__ = [
    'ConfigParser',
    'ConfigParseResult', 
    'ConfigurationError',
    'load_config',
    'validate_config_file',
    'create_config_template'
]