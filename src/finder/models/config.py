"""
Configuration data models for the AI Filesystem Finder.

This module defines the core data structures for managing application configuration,
including root directories, ignore patterns, vector database settings, and security options.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import re
import fnmatch
from pydantic import BaseModel, Field, field_validator, model_validator


class VectorBackend(Enum):
    """Supported vector database backends."""
    LANCEDB = "lancedb"
    CHROMA = "chroma"
    FAISS = "faiss"


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    JSON = "json"


class VectorDBConfig(BaseModel):
    """
    Configuration for vector database settings.
    
    Attributes:
        backend: Vector database backend to use
        path: Path to store the vector database
        dimension: Embedding dimension (auto-detected if None)
        index_params: Backend-specific index parameters
    """
    
    backend: VectorBackend = Field(VectorBackend.LANCEDB, description="Vector database backend")
    path: str = Field("~/.finder-agent/index", description="Path to store the vector database")
    dimension: Optional[int] = Field(None, gt=0, description="Embedding dimension")
    index_params: Dict[str, Any] = Field(default_factory=dict, description="Backend-specific index parameters")
    
    @field_validator('backend', mode='before')
    @classmethod
    def validate_backend(cls, v) -> VectorBackend:
        """Validate and convert backend to enum."""
        if isinstance(v, str):
            try:
                return VectorBackend(v)
            except ValueError:
                raise ValueError(f"Invalid vector backend: {v}")
        return v
    
    def model_post_init(self, __context) -> None:
        """Expand user path after initialization."""
        self.path = str(Path(self.path).expanduser())
    
    def get_full_path(self) -> Path:
        """Get the full resolved path for the vector database."""
        return Path(self.path).resolve()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.model_dump()
        data['backend'] = self.backend.value
        return data


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding generation.
    
    Attributes:
        provider: Embedding service provider
        model: Model name/identifier
        api_key: API key for the service (if required)
        batch_size: Number of texts to embed in one batch
        max_tokens: Maximum tokens per text chunk
    """
    
    provider: EmbeddingProvider = Field(EmbeddingProvider.ANTHROPIC, description="Embedding service provider")
    model: str = Field("claude-3-haiku", description="Model name/identifier")
    api_key: Optional[str] = Field(None, description="API key for the service")
    batch_size: int = Field(10, gt=0, description="Number of texts to embed in one batch")
    max_tokens: int = Field(8000, gt=0, description="Maximum tokens per text chunk")
    
    @field_validator('provider', mode='before')
    @classmethod
    def validate_provider(cls, v) -> EmbeddingProvider:
        """Validate and convert provider to enum."""
        if isinstance(v, str):
            try:
                return EmbeddingProvider(v)
            except ValueError:
                raise ValueError(f"Invalid embedding provider: {v}")
        return v
    
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key."""
        return self.provider in [EmbeddingProvider.ANTHROPIC, EmbeddingProvider.OPENAI]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.model_dump()
        data['provider'] = self.provider.value
        data['api_key'] = '***' if self.api_key else None  # Redact API key
        return data


class LimitsConfig(BaseModel):
    """
    Configuration for system limits and constraints.
    
    Attributes:
        max_files: Maximum number of files to process
        max_bytes_per_file: Maximum file size to process (bytes)
        max_results: Maximum number of search results to return
        timeout_seconds: Timeout for search operations
        max_concurrent: Maximum concurrent operations
    """
    
    max_files: int = Field(200000, gt=0, description="Maximum number of files to process")
    max_bytes_per_file: int = Field(5000000, gt=0, description="Maximum file size to process (bytes)")
    max_results: int = Field(100, gt=0, description="Maximum number of search results to return")
    timeout_seconds: int = Field(300, gt=0, description="Timeout for search operations")
    max_concurrent: int = Field(4, gt=0, description="Maximum concurrent operations")
    
    def get_max_size_human_readable(self) -> str:
        """Get max file size in human-readable format."""
        size = float(self.max_bytes_per_file)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.model_dump()
        data['max_size_human'] = self.get_max_size_human_readable()
        return data


class OutputConfig(BaseModel):
    """
    Configuration for output formatting and storage.
    
    Attributes:
        formats: List of output formats to generate
        directory: Base directory for output files
        timestamp_dirs: Whether to create timestamped subdirectories
        include_metadata: Whether to include detailed metadata in output
    """
    
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.JSON],
        description="List of output formats to generate"
    )
    directory: str = Field("runs", description="Base directory for output files")
    timestamp_dirs: bool = Field(True, description="Whether to create timestamped subdirectories")
    include_metadata: bool = Field(True, description="Whether to include detailed metadata in output")
    
    @field_validator('formats', mode='before')
    @classmethod
    def validate_formats(cls, v) -> List[OutputFormat]:
        """Validate and convert formats to enums."""
        if not isinstance(v, list):
            v = [v]
        
        normalized_formats = []
        for fmt in v:
            if isinstance(fmt, str):
                try:
                    normalized_formats.append(OutputFormat(fmt))
                except ValueError:
                    raise ValueError(f"Invalid output format: {fmt}")
            else:
                normalized_formats.append(fmt)
        
        return normalized_formats
    
    @field_validator('directory')
    @classmethod
    def validate_directory(cls, v: str) -> str:
        """Expand user path but keep relative paths relative."""
        if v.startswith('~'):
            return str(Path(v).expanduser())
        else:
            return str(Path(v))
    
    def get_output_path(self) -> Path:
        """Get the resolved output directory path."""
        return Path(self.directory).resolve()
    
    def supports_format(self, format_type: OutputFormat) -> bool:
        """Check if a specific output format is enabled."""
        return format_type in self.formats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.model_dump()
        data['formats'] = [fmt.value for fmt in self.formats]
        return data


class SecurityConfig(BaseModel):
    """
    Configuration for security and privacy settings.
    
    Attributes:
        dry_run: Whether to run in preview mode without executing
        redact_patterns: Patterns to redact from output for privacy
        max_snippet_length: Maximum length of content snippets
        allow_binary_files: Whether to process binary files
    """
    
    dry_run: bool = Field(False, description="Whether to run in preview mode without executing")
    redact_patterns: List[str] = Field(
        default_factory=lambda: ["api_key", "password", "secret"],
        description="Patterns to redact from output for privacy"
    )
    max_snippet_length: int = Field(500, gt=0, description="Maximum length of content snippets")
    allow_binary_files: bool = Field(False, description="Whether to process binary files")
    
    def model_post_init(self, __context) -> None:
        """Validate security configuration and compile patterns."""
        # Compile redaction patterns for efficiency
        self._compiled_patterns = []
        for pattern in self.redact_patterns:
            try:
                # Simple pattern matching for sensitive keywords
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                raise ValueError(f"Invalid redaction pattern '{pattern}': {e}")
    
    def should_redact(self, text: str) -> bool:
        """Check if text contains patterns that should be redacted."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def redact_text(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Redact sensitive patterns from text."""
        result = text
        for pattern in self._compiled_patterns:
            # Replace the entire line if it contains sensitive patterns
            if pattern.search(result):
                lines = result.split('\n')
                redacted_lines = []
                for line in lines:
                    if pattern.search(line):
                        redacted_lines.append(replacement)
                    else:
                        redacted_lines.append(line)
                result = '\n'.join(redacted_lines)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class FinderConfig(BaseModel):
    """
    Main configuration class for the AI Filesystem Finder.
    
    This class encapsulates all configuration settings including root directories,
    ignore patterns, vector database settings, and security options.
    
    Attributes:
        roots: List of root directories to search (allowlisted)
        ignore: List of ignore patterns (gitignore-style)
        vector_db: Vector database configuration
        embeddings: Embedding generation configuration
        limits: System limits and constraints
        output: Output formatting configuration
        security: Security and privacy settings
    """
    
    roots: List[str] = Field(default_factory=list, min_length=1, description="List of root directories to search")
    ignore: List[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/node_modules/**", 
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/venv/**",
            "**/.venv/**"
        ],
        description="List of ignore patterns (gitignore-style)"
    )
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig, description="Vector database configuration")
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Embedding generation configuration")
    limits: LimitsConfig = Field(default_factory=LimitsConfig, description="System limits and constraints")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output formatting configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security and privacy settings")
    
    def model_post_init(self, __context) -> None:
        """Validate and normalize the complete configuration."""
        self._validate_roots()
        self._normalize_ignore_patterns()
        self._compile_ignore_patterns()
    
    def _validate_roots(self) -> None:
        """Validate and normalize root directories."""
        if not self.roots:
            raise ValueError("At least one root directory must be specified")
        
        normalized_roots = []
        for root in self.roots:
            if not root or not root.strip():
                continue
            
            # Expand user path and resolve
            root_path = Path(root).expanduser().resolve()
            
            # Check if path exists (in non-dry-run mode)
            if not self.security.dry_run and not root_path.exists():
                raise ValueError(f"Root directory does not exist: {root_path}")
            
            # Check if it's actually a directory
            if not self.security.dry_run and root_path.exists() and not root_path.is_dir():
                raise ValueError(f"Root path is not a directory: {root_path}")
            
            normalized_roots.append(str(root_path))
        
        if not normalized_roots:
            raise ValueError("No valid root directories provided")
        
        self.roots = normalized_roots
    
    def _normalize_ignore_patterns(self) -> None:
        """Normalize ignore patterns to ensure consistency."""
        normalized_patterns = []
        for pattern in self.ignore:
            if not pattern or not pattern.strip():
                continue
            
            pattern = pattern.strip()
            
            # Ensure patterns are properly formatted
            if not pattern.startswith('**/') and not pattern.startswith('./') and not pattern.startswith('/'):
                # Add **/ prefix for relative patterns
                pattern = '**/' + pattern
            
            normalized_patterns.append(pattern)
        
        self.ignore = normalized_patterns
    
    def _compile_ignore_patterns(self) -> None:
        """Compile ignore patterns for efficient matching."""
        self._compiled_ignore_patterns = []
        for pattern in self.ignore:
            try:
                # Convert gitignore-style patterns to regex
                regex_pattern = self._gitignore_to_regex(pattern)
                self._compiled_ignore_patterns.append(re.compile(regex_pattern))
            except re.error as e:
                raise ValueError(f"Invalid ignore pattern '{pattern}': {e}")
    
    def _gitignore_to_regex(self, pattern: str) -> str:
        """Convert gitignore-style pattern to regex."""
        # Handle special gitignore patterns
        if pattern.startswith('**/'):
            # **/ means zero or more directories
            pattern = pattern[3:]
            regex = r'(^|.*/)' + fnmatch.translate(pattern).replace(r'\Z', '')
        elif pattern.endswith('/**'):
            # /** means everything inside directory
            pattern = pattern[:-3]
            regex = fnmatch.translate(pattern).replace(r'\Z', '') + r'(/.*)?'
        else:
            regex = fnmatch.translate(pattern)
        
        return regex
    
    def should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored based on ignore patterns."""
        # Normalize path for comparison
        normalized_path = str(Path(path).as_posix())
        
        for pattern in self._compiled_ignore_patterns:
            if pattern.match(normalized_path):
                return True
        
        return False
    
    def is_root_accessible(self, root: str) -> bool:
        """Check if a root directory is accessible."""
        try:
            root_path = Path(root)
            return root_path.exists() and root_path.is_dir()
        except (OSError, PermissionError):
            return False
    
    def get_accessible_roots(self) -> List[str]:
        """Get list of accessible root directories."""
        return [root for root in self.roots if self.is_root_accessible(root)]
    
    def get_inaccessible_roots(self) -> List[str]:
        """Get list of inaccessible root directories."""
        return [root for root in self.roots if not self.is_root_accessible(root)]
    
    def validate_configuration(self) -> List[str]:
        """Validate the complete configuration and return any warnings."""
        warnings = []
        
        # Check for inaccessible roots
        inaccessible = self.get_inaccessible_roots()
        if inaccessible:
            warnings.append(f"Inaccessible root directories: {', '.join(inaccessible)}")
        
        # Check if embedding provider requires API key
        if self.embeddings.requires_api_key() and not self.embeddings.api_key:
            warnings.append(f"Embedding provider {self.embeddings.provider.value} requires an API key")
        
        # Check vector DB path accessibility
        try:
            vector_path = self.vector_db.get_full_path()
            vector_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            warnings.append(f"Cannot create vector database directory: {self.vector_db.path}")
        
        # Check output directory accessibility
        try:
            output_path = self.output.get_output_path()
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            warnings.append(f"Cannot create output directory: {self.output.directory}")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        data = self.model_dump()
        data['vector_db'] = self.vector_db.to_dict()
        data['embeddings'] = self.embeddings.to_dict()
        data['limits'] = self.limits.to_dict()
        data['output'] = self.output.to_dict()
        data['security'] = self.security.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinderConfig':
        """Create configuration from dictionary representation."""
        return cls.model_validate(data)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        parts = [f"Roots: {len(self.roots)} directories"]
        parts.append(f"Ignore patterns: {len(self.ignore)}")
        parts.append(f"Vector DB: {self.vector_db.backend.value}")
        parts.append(f"Embeddings: {self.embeddings.provider.value}")
        parts.append(f"Dry run: {self.security.dry_run}")
        
        return " | ".join(parts)


class ConfigValidator(BaseModel):
    """
    Pydantic-based configuration validator for enhanced validation and type safety.
    
    This class provides additional validation on top of the dataclass-based configuration
    to ensure all settings are valid and consistent.
    """
    
    roots: List[str] = Field(..., min_length=1, description="List of root directories to search")
    ignore: List[str] = Field(default_factory=list, description="List of ignore patterns")
    vector_db_backend: str = Field(default="lancedb", description="Vector database backend")
    vector_db_path: str = Field(default="~/.finder-agent/index", description="Vector database path")
    embedding_provider: str = Field(default="anthropic", description="Embedding provider")
    embedding_model: str = Field(default="claude-3-haiku", description="Embedding model")
    max_files: int = Field(default=200000, gt=0, description="Maximum number of files to process")
    max_bytes_per_file: int = Field(default=5000000, gt=0, description="Maximum file size in bytes")
    max_results: int = Field(default=100, gt=0, description="Maximum search results")
    output_formats: List[str] = Field(default=["markdown", "json"], description="Output formats")
    output_directory: str = Field(default="runs", description="Output directory")
    dry_run: bool = Field(default=False, description="Whether to run in preview mode")
    
    @field_validator('roots')
    @classmethod
    def validate_roots(cls, v: List[str]) -> List[str]:
        """Validate root directories."""
        if not v:
            raise ValueError("At least one root directory must be specified")
        
        validated_roots = []
        for root in v:
            if not root or not root.strip():
                continue
            
            # Expand and normalize path
            root_path = Path(root).expanduser().resolve()
            validated_roots.append(str(root_path))
        
        if not validated_roots:
            raise ValueError("No valid root directories provided")
        
        return validated_roots
    
    @field_validator('ignore')
    @classmethod
    def validate_ignore_patterns(cls, v: List[str]) -> List[str]:
        """Validate ignore patterns."""
        validated_patterns = []
        for pattern in v:
            if not pattern or not pattern.strip():
                continue
            
            pattern = pattern.strip()
            
            # Test if pattern is valid regex-compatible
            try:
                # Convert gitignore-style to regex for validation
                if pattern.startswith('**/'):
                    test_pattern = pattern[3:]
                    fnmatch.translate(test_pattern)
                elif pattern.endswith('/**'):
                    test_pattern = pattern[:-3]
                    fnmatch.translate(test_pattern)
                else:
                    fnmatch.translate(pattern)
                
                validated_patterns.append(pattern)
            except Exception as e:
                raise ValueError(f"Invalid ignore pattern '{pattern}': {e}")
        
        return validated_patterns
    
    @field_validator('vector_db_backend')
    @classmethod
    def validate_vector_backend(cls, v: str) -> str:
        """Validate vector database backend."""
        valid_backends = ["lancedb", "chroma", "faiss"]
        if v.lower() not in valid_backends:
            raise ValueError(f"Invalid vector backend '{v}'. Must be one of: {valid_backends}")
        return v.lower()
    
    @field_validator('embedding_provider')
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        """Validate embedding provider."""
        valid_providers = ["anthropic", "openai", "local"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid embedding provider '{v}'. Must be one of: {valid_providers}")
        return v.lower()
    
    @field_validator('output_formats')
    @classmethod
    def validate_output_formats(cls, v: List[str]) -> List[str]:
        """Validate output formats."""
        valid_formats = ["markdown", "json"]
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Invalid output format '{fmt}'. Must be one of: {valid_formats}")
        return [fmt.lower() for fmt in v]
    
    @field_validator('vector_db_path', 'output_directory')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Validate file paths."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        
        # Expand user path
        expanded_path = str(Path(v).expanduser())
        return expanded_path
    
    @model_validator(mode='after')
    def validate_configuration_consistency(self):
        """Validate overall configuration consistency."""
        # Check if embedding provider requires API key (this would need to be checked at runtime)
        if self.embedding_provider in ["anthropic", "openai"]:
            # Note: API key validation would happen at runtime, not during config validation
            pass
        
        # Ensure output directory is writable (basic check)
        try:
            output_path = Path(self.output_directory).expanduser()
            if output_path.exists() and not output_path.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")
        except Exception as e:
            raise ValueError(f"Invalid output directory: {e}")
        
        return self
    
    @classmethod
    def validate_finder_config(cls, config: FinderConfig) -> List[str]:
        """
        Validate a FinderConfig instance using Pydantic validation.
        
        Args:
            config: The FinderConfig instance to validate
            
        Returns:
            List of validation warnings (empty if all valid)
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Convert FinderConfig to dict for Pydantic validation
            config_dict = {
                'roots': config.roots,
                'ignore': config.ignore,
                'vector_db_backend': config.vector_db.backend.value,
                'vector_db_path': config.vector_db.path,
                'embedding_provider': config.embeddings.provider.value,
                'embedding_model': config.embeddings.model,
                'max_files': config.limits.max_files,
                'max_bytes_per_file': config.limits.max_bytes_per_file,
                'max_results': config.limits.max_results,
                'output_formats': [fmt.value for fmt in config.output.formats],
                'output_directory': config.output.directory,
                'dry_run': config.security.dry_run,
            }
            
            # Validate using Pydantic
            cls(**config_dict)
            
            # Return any warnings from the original validation
            return config.validate_configuration()
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")


def validate_config_dict(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary using Pydantic.
    
    Args:
        config_data: Dictionary containing configuration data
        
    Returns:
        Validated and normalized configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Flatten nested configuration for validation
        flat_config = {}
        
        # Extract root-level settings
        flat_config['roots'] = config_data.get('roots', [])
        flat_config['ignore'] = config_data.get('ignore', [])
        flat_config['dry_run'] = config_data.get('security', {}).get('dry_run', False)
        
        # Extract vector DB settings
        vector_db = config_data.get('vector_db', {})
        flat_config['vector_db_backend'] = vector_db.get('backend', 'lancedb')
        flat_config['vector_db_path'] = vector_db.get('path', '~/.finder-agent/index')
        
        # Extract embedding settings
        embeddings = config_data.get('embeddings', {})
        flat_config['embedding_provider'] = embeddings.get('provider', 'anthropic')
        flat_config['embedding_model'] = embeddings.get('model', 'claude-3-haiku')
        
        # Extract limits
        limits = config_data.get('limits', {})
        flat_config['max_files'] = limits.get('max_files', 200000)
        flat_config['max_bytes_per_file'] = limits.get('max_bytes_per_file', 5000000)
        flat_config['max_results'] = limits.get('max_results', 100)
        
        # Extract output settings
        output = config_data.get('output', {})
        flat_config['output_formats'] = output.get('formats', ['markdown', 'json'])
        flat_config['output_directory'] = output.get('directory', 'runs')
        
        # Validate using Pydantic
        validator = ConfigValidator(**flat_config)
        
        # Return the original structure with validated values
        validated_config = config_data.copy()
        validated_config['roots'] = validator.roots
        validated_config['ignore'] = validator.ignore
        
        return validated_config
        
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")