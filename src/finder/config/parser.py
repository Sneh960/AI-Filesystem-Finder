"""
YAML configuration parser for the AI Filesystem Finder.

This module provides functionality to load, parse, and validate YAML configuration files
for the AI Filesystem Finder. It handles configuration file discovery, parsing, validation,
and provides helpful error messages for configuration issues.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass

from ..models.config import FinderConfig, validate_config_dict


logger = logging.getLogger(__name__)


@dataclass
class ConfigParseResult:
    """
    Result of configuration parsing operation.
    
    Attributes:
        config: The parsed and validated configuration
        warnings: List of non-fatal warnings
        config_path: Path to the configuration file used
        is_default: Whether default configuration was used
    """
    config: FinderConfig
    warnings: List[str]
    config_path: Optional[Path]
    is_default: bool


class ConfigurationError(Exception):
    """Raised when configuration parsing or validation fails."""
    pass


class ConfigParser:
    """
    YAML configuration parser with validation and error handling.
    
    This class handles loading YAML configuration files, validating their contents,
    and converting them to FinderConfig objects. It supports configuration file
    discovery, default configuration generation, and comprehensive error reporting.
    """
    
    DEFAULT_CONFIG_NAMES = [
        '.finderagent.yaml',
        '.finderagent.yml',
        'finderagent.yaml',
        'finderagent.yml',
        '.finder.yaml',
        '.finder.yml'
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the configuration parser.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> ConfigParseResult:
        """
        Load and parse configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file. If None, searches for default files.
            
        Returns:
            ConfigParseResult containing parsed configuration and metadata
            
        Raises:
            ConfigurationError: If configuration is invalid or file cannot be read
        """
        try:
            if config_path:
                # Use specified configuration file
                config_path = Path(config_path)
                if not config_path.exists():
                    raise ConfigurationError(f"Configuration file not found: {config_path}")
                
                config_data = self._load_yaml_file(config_path)
                is_default = False
            else:
                # Search for default configuration files
                config_path, config_data = self._find_and_load_config()
                is_default = config_data is None
                
                if is_default:
                    config_data = self._get_default_config()
            
            # If config_data is empty or has no roots, merge with defaults
            if not config_data or not config_data.get('roots'):
                default_config = self._get_default_config()
                # Merge user config with defaults, user config takes precedence
                merged_config = default_config.copy()
                merged_config.update(config_data)
                config_data = merged_config
            
            # Validate the configuration data
            validated_data = self._validate_config_data(config_data)
            
            # Create FinderConfig object
            finder_config = FinderConfig.from_dict(validated_data)
            
            # Get validation warnings
            warnings = finder_config.validate_configuration()
            
            # Add parser-specific warnings
            parser_warnings = self._get_parser_warnings(finder_config, config_path, is_default)
            warnings.extend(parser_warnings)
            
            # Handle strict mode
            if self.strict_mode and warnings:
                raise ConfigurationError(f"Configuration warnings in strict mode: {'; '.join(warnings)}")
            
            self.logger.info(f"Configuration loaded successfully from {config_path or 'defaults'}")
            
            return ConfigParseResult(
                config=finder_config,
                warnings=warnings,
                config_path=config_path,
                is_default=is_default
            )
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            else:
                raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _find_and_load_config(self) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """
        Find and load configuration file from default locations.
        
        Returns:
            Tuple of (config_path, config_data) or (None, None) if not found
        """
        search_paths = [
            Path.cwd(),  # Current directory
            Path.home(),  # Home directory
            Path.home() / '.config' / 'finder-agent',  # XDG config directory
        ]
        
        for search_path in search_paths:
            for config_name in self.DEFAULT_CONFIG_NAMES:
                config_file = search_path / config_name
                if config_file.exists() and config_file.is_file():
                    try:
                        config_data = self._load_yaml_file(config_file)
                        self.logger.info(f"Found configuration file: {config_file}")
                        return config_file, config_data
                    except Exception as e:
                        self.logger.warning(f"Failed to load {config_file}: {e}")
                        continue
        
        self.logger.info("No configuration file found, using defaults")
        return None, None
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed YAML data as dictionary
            
        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle empty files
            if not content.strip():
                self.logger.warning(f"Configuration file is empty: {file_path}")
                return {}
            
            # Parse YAML
            data = yaml.safe_load(content)
            
            # Handle None result (empty YAML)
            if data is None:
                return {}
            
            # Ensure we have a dictionary
            if not isinstance(data, dict):
                raise ConfigurationError(f"Configuration file must contain a YAML object, got {type(data).__name__}")
            
            return data
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {file_path}: {e}") from e
        except (OSError, IOError) as e:
            raise ConfigurationError(f"Cannot read configuration file {file_path}: {e}") from e
    
    def _validate_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration data structure and values.
        
        Args:
            config_data: Raw configuration data from YAML
            
        Returns:
            Validated and normalized configuration data
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Use the existing validation function
            validated_data = validate_config_dict(config_data)
            return validated_data
            
        except ValueError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration when no config file is found.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'roots': [str(Path.cwd())],  # Default to current directory
            'ignore': [
                "**/.git/**",
                "**/.svn/**",
                "**/.hg/**",
                "**/node_modules/**",
                "**/venv/**",
                "**/.venv/**",
                "**/env/**",
                "**/__pycache__/**",
                "**/.pytest_cache/**",
                "**/target/**",
                "**/build/**",
                "**/dist/**",
                "**/.vscode/**",
                "**/.idea/**",
                "**/*.swp",
                "**/*.swo",
                "**/*~",
                "**/.DS_Store",
                "**/Thumbs.db",
                "**/*.tmp",
                "**/*.temp",
                "**/*.iso",
                "**/*.dmg",
                "**/*.exe",
                "**/*.msi"
            ],
            'vector_db': {
                'backend': 'lancedb',
                'path': '~/.finder-agent/index'
            },
            'embeddings': {
                'provider': 'anthropic',
                'model': 'claude-3-haiku',
                'batch_size': 10,
                'max_tokens': 8000
            },
            'limits': {
                'max_files': 200000,
                'max_bytes_per_file': 5000000,
                'max_results': 100,
                'timeout_seconds': 300,
                'max_concurrent': 4
            },
            'output': {
                'formats': ['markdown', 'json'],
                'directory': './runs',
                'timestamp_dirs': True,
                'include_metadata': True
            },
            'security': {
                'dry_run': True,  # Default to dry-run mode for safety
                'redact_patterns': ['api_key', 'password', 'secret', 'token', 'credential'],
                'max_snippet_length': 500,
                'allow_binary_files': False,
                'preview_max_files': 100,
                'preview_max_depth': 3
            }
        }
    
    def _get_parser_warnings(self, config: FinderConfig, config_path: Optional[Path], is_default: bool) -> List[str]:
        """
        Get parser-specific warnings.
        
        Args:
            config: The parsed configuration
            config_path: Path to configuration file (if any)
            is_default: Whether default configuration was used
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if is_default:
            warnings.append("No configuration file found, using default settings")
        
        # Check for environment variables that might be needed
        if config.embeddings.requires_api_key() and not config.embeddings.api_key:
            provider = config.embeddings.provider.value.upper()
            env_var = f"{provider}_API_KEY"
            if not os.getenv(env_var):
                warnings.append(f"API key not found in configuration or {env_var} environment variable")
        
        # Check for potentially problematic settings
        if len(config.roots) > 10:
            warnings.append(f"Large number of root directories ({len(config.roots)}) may impact performance")
        
        if config.limits.max_files > 1000000:
            warnings.append("Very high max_files limit may cause memory issues")
        
        if config.limits.max_bytes_per_file > 50000000:  # 50MB
            warnings.append("Very high max_bytes_per_file limit may cause memory issues")
        
        # Check for security concerns
        if not config.security.redact_patterns:
            warnings.append("No redaction patterns configured - sensitive data may be exposed in output")
        
        if config.security.allow_binary_files:
            warnings.append("Binary file processing enabled - may impact performance and security")
        
        return warnings
    
    def save_config(self, config: FinderConfig, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path where to save the configuration
            
        Raises:
            ConfigurationError: If file cannot be written
        """
        try:
            output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = config.to_dict()
            
            # Add header comment
            yaml_content = self._generate_yaml_with_comments(config_dict)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except (OSError, IOError) as e:
            raise ConfigurationError(f"Cannot write configuration file {output_path}: {e}") from e
    
    def _generate_yaml_with_comments(self, config_dict: Dict[str, Any]) -> str:
        """
        Generate YAML content with helpful comments.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            YAML content with comments
        """
        lines = [
            "# AI Filesystem Finder Configuration",
            "# This file configures the search behavior, security settings, and output preferences",
            "",
        ]
        
        # Add sections with comments
        sections = [
            ("roots", "Root directories to search (explicit allowlist for security)"),
            ("ignore", "Ignore patterns (gitignore-style syntax)"),
            ("vector_db", "Vector database configuration"),
            ("embeddings", "Embedding configuration"),
            ("limits", "System limits and performance"),
            ("output", "Output configuration"),
            ("security", "Security and privacy settings")
        ]
        
        for section_name, comment in sections:
            if section_name in config_dict:
                lines.append(f"# {comment}")
                section_yaml = yaml.dump({section_name: config_dict[section_name]}, 
                                       default_flow_style=False, 
                                       sort_keys=False)
                lines.append(section_yaml.rstrip())
                lines.append("")
        
        return "\n".join(lines)
    
    def validate_config_file(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate a configuration file without loading it into a FinderConfig.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_path}")
                return errors
            
            # Load and validate YAML
            config_data = self._load_yaml_file(config_path)
            
            # Validate structure
            self._validate_config_data(config_data)
            
            # Try to create FinderConfig
            FinderConfig.from_dict(config_data)
            
        except ConfigurationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected error validating configuration: {e}")
        
        return errors
    
    def generate_preview_report(self, config_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a preview report showing what would be searched.
        
        Args:
            config_path: Path to configuration file (optional)
            
        Returns:
            Preview report as markdown string
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        try:
            # Load configuration
            result = self.load_config(config_path)
            
            # Generate preview report
            preview_report = result.config.generate_preview_report()
            
            # Add configuration source info
            config_source = f"Configuration loaded from: {result.config_path or 'defaults'}"
            if result.warnings:
                config_source += f"\nWarnings: {len(result.warnings)} found"
            
            full_report = f"{preview_report}\n\n---\n\n*{config_source}*"
            
            return full_report
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            else:
                raise ConfigurationError(f"Failed to generate preview report: {e}") from e
    
    def get_config_template(self) -> str:
        """
        Get a template configuration file with all options and comments.
        
        Returns:
            YAML template as string
        """
        template_config = {
            'roots': [
                '.',  # Current directory
                '~/Documents',
                '~/Projects'
            ],
            'ignore': [
                '**/.git/**',
                '**/.svn/**',
                '**/.hg/**',
                '**/node_modules/**',
                '**/venv/**',
                '**/.venv/**',
                '**/env/**',
                '**/__pycache__/**',
                '**/.pytest_cache/**',
                '**/target/**',
                '**/build/**',
                '**/dist/**',
                '**/.vscode/**',
                '**/.idea/**',
                '**/*.swp',
                '**/*.swo',
                '**/*~',
                '**/.DS_Store',
                '**/Thumbs.db',
                '**/*.tmp',
                '**/*.temp',
                '**/*.iso',
                '**/*.dmg',
                '**/*.exe',
                '**/*.msi'
            ],
            'vector_db': {
                'backend': 'lancedb',
                'path': '~/.finder-agent/index'
            },
            'embeddings': {
                'provider': 'anthropic',
                'model': 'claude-3-haiku',
                'batch_size': 10,
                'max_tokens': 8000
            },
            'limits': {
                'max_files': 200000,
                'max_bytes_per_file': 5000000,
                'max_results': 100,
                'timeout_seconds': 300,
                'max_concurrent': 4
            },
            'output': {
                'formats': ['markdown', 'json'],
                'directory': './runs',
                'timestamp_dirs': True,
                'include_metadata': True
            },
            'security': {
                'dry_run': True,  # Enable dry-run mode by default for safety
                'redact_patterns': ['api_key', 'password', 'secret', 'token', 'credential'],
                'max_snippet_length': 500,
                'allow_binary_files': False,
                'preview_max_files': 100,
                'preview_max_depth': 3
            }
        }
        
        return self._generate_yaml_with_comments(template_config)


def load_config(config_path: Optional[Union[str, Path]] = None, strict_mode: bool = False) -> ConfigParseResult:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        strict_mode: Whether to treat warnings as errors
        
    Returns:
        ConfigParseResult containing parsed configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    parser = ConfigParser(strict_mode=strict_mode)
    return parser.load_config(config_path)


def validate_config_file(config_path: Union[str, Path]) -> List[str]:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation errors (empty if valid)
    """
    parser = ConfigParser()
    return parser.validate_config_file(config_path)


def create_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a template configuration file.
    
    Args:
        output_path: Where to save the template
        
    Raises:
        ConfigurationError: If template cannot be created
    """
    parser = ConfigParser()
    template_content = parser.get_config_template()
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
            
    except (OSError, IOError) as e:
        raise ConfigurationError(f"Cannot create template file {output_path}: {e}") from e


def generate_preview_report(config_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convenience function to generate a preview report.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Preview report as markdown string
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    parser = ConfigParser()
    return parser.generate_preview_report(config_path)