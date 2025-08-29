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
        preview_max_files: Maximum number of files to show in preview mode
        preview_max_depth: Maximum directory depth to show in preview
    """
    
    dry_run: bool = Field(False, description="Whether to run in preview mode without executing")
    redact_patterns: List[str] = Field(
        default_factory=lambda: ["api_key", "password", "secret"],
        description="Patterns to redact from output for privacy"
    )
    max_snippet_length: int = Field(500, gt=0, description="Maximum length of content snippets")
    allow_binary_files: bool = Field(False, description="Whether to process binary files")
    preview_max_files: int = Field(100, gt=0, description="Maximum number of files to show in preview mode")
    preview_max_depth: int = Field(3, gt=0, description="Maximum directory depth to show in preview")
    
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
        """Validate and normalize root directories with allowlist security checks."""
        if not self.roots:
            raise ValueError("At least one root directory must be specified")
        
        normalized_roots = []
        for root in self.roots:
            if not root or not root.strip():
                continue
            
            # Check for dangerous patterns before path resolution
            self._validate_path_safety_raw(root)
            
            # Expand user path and resolve
            root_path = Path(root).expanduser().resolve()
            
            # Perform allowlist validation and security checks
            self._validate_allowlist_security(root_path)
            
            # Check if path exists (in non-dry-run mode)
            if not self.security.dry_run and not root_path.exists():
                raise ValueError(f"Root directory does not exist: {root_path}")
            
            # Check if it's actually a directory
            if not self.security.dry_run and root_path.exists() and not root_path.is_dir():
                raise ValueError(f"Root path is not a directory: {root_path}")
            
            normalized_roots.append(str(root_path))
        
        if not normalized_roots:
            raise ValueError("No valid root directories provided")
        
        # Validate for path traversal and overlapping roots
        self._validate_root_relationships(normalized_roots)
        
        self.roots = normalized_roots
    
    def _normalize_ignore_patterns(self) -> None:
        """Normalize ignore patterns to ensure consistency."""
        normalized_patterns = []
        for pattern in self.ignore:
            if not pattern or not pattern.strip():
                continue
            
            pattern = pattern.strip()
            
            # Skip empty patterns and comments
            if not pattern or pattern.startswith('#'):
                continue
            
            # Handle negation patterns specially
            if pattern.startswith('!'):
                # For negation patterns, normalize the part after !
                negated_pattern = pattern[1:]
                if (negated_pattern.startswith('**/') or 
                    negated_pattern.startswith('./') or 
                    negated_pattern.startswith('/') or
                    '/' in negated_pattern):
                    normalized_patterns.append(pattern)
                else:
                    # Add **/ prefix to the negated part
                    normalized_patterns.append('!' + '**/' + negated_pattern)
            # Don't modify patterns that are already properly formatted
            elif (pattern.startswith('**/') or 
                  pattern.startswith('./') or 
                  pattern.startswith('/')):
                normalized_patterns.append(pattern)
            else:
                # Add **/ prefix for relative patterns (including those with /)
                normalized_patterns.append('**/' + pattern)
        
        self.ignore = normalized_patterns
    
    def _compile_ignore_patterns(self) -> None:
        """Compile ignore patterns for efficient matching."""
        self._compiled_ignore_patterns = []
        for pattern in self.ignore:
            try:
                # Convert gitignore-style patterns to regex
                regex_pattern = self._gitignore_to_regex(pattern)
                is_negation = pattern.lstrip().startswith('!')
                
                # Store pattern info for processing
                pattern_info = {
                    'regex': re.compile(regex_pattern),
                    'is_negation': is_negation,
                    'original': pattern
                }
                self._compiled_ignore_patterns.append(pattern_info)
            except re.error as e:
                raise ValueError(f"Invalid ignore pattern '{pattern}': {e}")
    
    def _validate_allowlist_security(self, root_path: Path) -> None:
        """
        Validate root directory against allowlist security policies.
        
        This method implements explicit allowlist validation to ensure only
        explicitly configured directories can be searched, protecting sensitive
        system areas.
        
        Args:
            root_path: Resolved path to validate
            
        Raises:
            ValueError: If path violates allowlist security policies
        """
        # Define sensitive system directories that should be blocked
        sensitive_paths = self._get_sensitive_system_paths()
        
        # Check if root path is a sensitive system directory
        for sensitive_path in sensitive_paths:
            try:
                sensitive_resolved = Path(sensitive_path).expanduser().resolve()
                
                # Check if root is exactly a sensitive path
                if root_path == sensitive_resolved:
                    raise ValueError(
                        f"Root directory '{root_path}' is a sensitive system directory. "
                        f"For security, please specify a more specific subdirectory."
                    )
                
                # Check if root is a parent of sensitive paths (too broad)
                try:
                    sensitive_resolved.relative_to(root_path)
                    # If we get here, root_path contains the sensitive path
                    if not self.security.dry_run:
                        raise ValueError(
                            f"Root directory '{root_path}' contains sensitive system directory "
                            f"'{sensitive_resolved}'. Please use a more specific path or enable "
                            f"appropriate ignore patterns."
                        )
                except ValueError:
                    # relative_to failed, which means sensitive_path is not under root_path
                    # This is fine, continue checking
                    pass
                    
            except (OSError, RuntimeError):
                # Path resolution failed, skip this check
                continue
        
        # Validate path doesn't use dangerous patterns
        self._validate_path_safety(root_path)
    
    def _get_sensitive_system_paths(self) -> List[str]:
        """
        Get list of sensitive system paths that should be protected.
        
        Returns:
            List of sensitive system directory paths
        """
        import platform
        
        sensitive_paths = [
            # Universal sensitive paths
            "/",  # Root filesystem
            "/etc",  # System configuration
            "/var/log",  # System logs
            "/tmp",  # Temporary files (privacy concern)
            "/dev",  # Device files
            "/proc",  # Process information
            "/sys",  # System information
        ]
        
        system = platform.system().lower()
        
        if system == "linux":
            sensitive_paths.extend([
                "/boot",  # Boot files
                "/root",  # Root user home
                "/var/lib",  # System data
                "/usr/lib",  # System libraries
                "/lib",  # System libraries
                "/sbin",  # System binaries
                "/bin",  # System binaries
            ])
        elif system == "darwin":  # macOS
            sensitive_paths.extend([
                "/System",  # macOS system files
                "/Library/System",  # System library
                "/private",  # Private system files
                "/var/root",  # Root user home
                "/usr/lib",  # System libraries
                "/usr/libexec",  # System executables
                "/bin",  # System binaries
                "/sbin",  # System binaries
            ])
        elif system == "windows":
            sensitive_paths.extend([
                "C:\\Windows",  # Windows system
                "C:\\Program Files",  # Installed programs
                "C:\\Program Files (x86)",  # 32-bit programs
                "C:\\ProgramData",  # Application data
                "C:\\System Volume Information",  # System restore
                "C:\\$Recycle.Bin",  # Recycle bin
            ])
        
        return sensitive_paths
    
    def _validate_path_safety_raw(self, root_str: str) -> None:
        """
        Validate that the raw path string doesn't contain dangerous patterns.
        
        Args:
            root_str: Raw path string to validate
            
        Raises:
            ValueError: If path contains dangerous patterns
        """
        # Check for path traversal attempts
        dangerous_patterns = [
            "..",  # Parent directory traversal
            "//",  # Double slashes (can bypass filters)
            "\\\\",  # Windows UNC paths (if not intended)
        ]
        
        for pattern in dangerous_patterns:
            if pattern in root_str:
                raise ValueError(
                    f"Root directory '{root_str}' contains potentially dangerous "
                    f"pattern '{pattern}'. Please use a clean, absolute path."
                )
    
    def _validate_path_safety(self, root_path: Path) -> None:
        """
        Validate that the path doesn't contain dangerous patterns.
        
        Args:
            root_path: Path to validate
            
        Raises:
            ValueError: If path contains dangerous patterns
        """
        # Validate path can be resolved
        try:
            normalized = root_path.resolve()
            # Note: We don't check if normalized != root_path because resolve() 
            # is expected to normalize the path
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve root directory path '{root_path}': {e}")
    
    def _validate_root_relationships(self, normalized_roots: List[str]) -> None:
        """
        Validate relationships between root directories to prevent conflicts.
        
        Args:
            normalized_roots: List of normalized root directory paths
            
        Raises:
            ValueError: If roots have problematic relationships
        """
        if len(normalized_roots) <= 1:
            return
        
        # Check for duplicate roots
        unique_roots = set(normalized_roots)
        if len(unique_roots) != len(normalized_roots):
            duplicates = [root for root in normalized_roots if normalized_roots.count(root) > 1]
            raise ValueError(f"Duplicate root directories found: {set(duplicates)}")
        
        # Check for nested roots (one root contains another)
        for i, root1 in enumerate(normalized_roots):
            path1 = Path(root1)
            for j, root2 in enumerate(normalized_roots):
                if i == j:
                    continue
                
                path2 = Path(root2)
                
                # Check if root2 is nested under root1
                try:
                    path2.relative_to(path1)
                    # If we get here, path2 is under path1
                    raise ValueError(
                        f"Root directory '{root2}' is nested under another root '{root1}'. "
                        f"This can cause duplicate results and performance issues. "
                        f"Please remove the nested root or use ignore patterns."
                    )
                except ValueError as e:
                    # Check if this is a "not relative" error or our nested root error
                    if "is nested under another root" in str(e):
                        # Re-raise our validation error
                        raise
                    # Otherwise, relative_to failed, which means path2 is not under path1
                    # This is fine, continue checking
                    pass
    
    def _gitignore_to_regex(self, pattern: str) -> str:
        """
        Convert gitignore-style pattern to regex.
        
        Supports common gitignore patterns including:
        - Basic wildcards (* and ?)
        - Directory wildcards (**)
        - Directory-only patterns (trailing /)
        - Rooted patterns (leading /)
        - Character classes [abc] and [!abc]
        
        Args:
            pattern: Gitignore-style pattern
            
        Returns:
            Compiled regex pattern string
        """
        if not pattern or pattern.isspace():
            return r'(?!.*)'  # Never matches anything
        
        original_pattern = pattern
        
        # Handle negation patterns (starting with !)
        if pattern.startswith('!'):
            pattern = pattern[1:]  # Remove the ! for regex processing
            if not pattern:
                return r'(?!.*)'  # Invalid pattern
        
        # Handle comments and empty patterns
        if pattern.startswith('#') or not pattern.strip():
            return r'(?!.*)'  # Never matches anything
        
        # Check if pattern is directory-only (ends with /)
        is_directory_only = pattern.endswith('/')
        if is_directory_only:
            pattern = pattern[:-1]  # Remove trailing slash
        
        # Check if pattern is rooted (starts with /)
        is_rooted = pattern.startswith('/')
        if is_rooted:
            pattern = pattern[1:]  # Remove leading slash
        
        # Handle special cases
        if not pattern:
            # Pattern was just "/" - matches root directory
            return r'^/?$'
        
        # Convert gitignore pattern to regex using fnmatch as base
        # but handle ** specially
        if '**' in pattern:
            # Handle ** patterns
            parts = pattern.split('**')
            regex_parts = []
            
            for i, part in enumerate(parts):
                if i > 0:
                    # Add ** regex between parts
                    prev_part = parts[i-1]
                    
                    # Determine the type of ** pattern
                    if prev_part == '':
                        # Pattern starts with ** (like **/something)
                        if part.startswith('/'):
                            # **/ at start - zero or more directories
                            regex_parts.append(r'(?:[^/]+/)*')
                            part = part[1:]  # Remove leading /
                        else:
                            # ** at start without / - match anything
                            regex_parts.append(r'.*')
                    elif part == '':
                        # Pattern ends with ** (like something/**)
                        if prev_part.endswith('/'):
                            # /** at end - everything under directory
                            regex_parts.append(r'.*')
                        else:
                            # ** at end without / - match anything
                            regex_parts.append(r'.*')
                    else:
                        # ** in middle (like something/**/other)
                        if part.startswith('/') and prev_part.endswith('/'):
                            # /**/  - zero or more directories for most cases
                            # but one or more for patterns like build/**/temp
                            if len(prev_part.rstrip('/')) > 0 and len(part.lstrip('/')) > 0:
                                # Both sides have content - require at least one directory
                                regex_parts.append(r'(?:[^/]+/)+')
                            else:
                                # One side is empty - zero or more directories
                                regex_parts.append(r'(?:[^/]+/)*')
                            part = part[1:]  # Remove leading /
                        elif part.startswith('/'):
                            # **/ where previous doesn't end with /
                            regex_parts.append(r'(?:[^/]+/)+')
                            part = part[1:]  # Remove leading /
                        elif prev_part.endswith('/'):
                            # /** where next doesn't start with /
                            regex_parts.append(r'.*')
                        else:
                            # ** without surrounding slashes
                            regex_parts.append(r'.*')
                
                if part:
                    # Convert the non-** part using fnmatch
                    part_regex = fnmatch.translate(part)
                    # Remove the \Z anchor that fnmatch adds
                    part_regex = part_regex.replace(r'\Z', '')
                    regex_parts.append(part_regex)
            
            escaped_pattern = ''.join(regex_parts)
        else:
            # No ** patterns, use fnmatch directly
            escaped_pattern = fnmatch.translate(pattern)
            # Remove the \Z anchor that fnmatch adds
            escaped_pattern = escaped_pattern.replace(r'\Z', '')
            
            # For rooted patterns, ensure * doesn't match /
            if is_rooted and '*' in pattern:
                # Replace .* with [^/]* to prevent matching across directories
                escaped_pattern = escaped_pattern.replace('.*', '[^/]*')
        
        # Build the final regex
        if is_rooted:
            # Rooted pattern - must match from start of path
            if is_directory_only:
                # Directory pattern - match the directory and everything inside it
                regex = f'^{escaped_pattern}(/.*)?$'
            else:
                regex = f'^{escaped_pattern}$'
        else:
            # Non-rooted pattern - can match anywhere in path
            if '/' in original_pattern:
                # Pattern contains slash - match against full path
                if is_directory_only:
                    # Directory pattern - match the directory and everything inside it
                    regex = f'(^|/){escaped_pattern}(/.*)?$'
                else:
                    regex = f'(^|/){escaped_pattern}$'
            else:
                # Pattern has no slash - match against basename
                if is_directory_only:
                    # Directory pattern - match the directory and everything inside it
                    regex = f'(^|/){escaped_pattern}(/.*)?$'
                else:
                    regex = f'(^|/){escaped_pattern}(/|$)'
        
        return regex
    
    def should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.
        
        Processes patterns in order, with later patterns potentially overriding
        earlier ones (especially negation patterns starting with !).
        
        Args:
            path: File or directory path to check
            
        Returns:
            True if path should be ignored, False otherwise
        """
        # Normalize path for comparison - convert to POSIX style
        normalized_path = str(Path(path).as_posix())
        
        # Remove leading slash if present for consistent matching
        if normalized_path.startswith('/'):
            normalized_path = normalized_path[1:]
        
        # Track ignore state - start with not ignored
        should_ignore_path = False
        
        # Process patterns in order
        for i, pattern_info in enumerate(self._compiled_ignore_patterns):
            pattern_regex = pattern_info['regex']
            is_negation = pattern_info['is_negation']
            original_pattern = pattern_info['original']
            
            # Test if pattern matches
            if pattern_regex.search(normalized_path):
                if is_negation:
                    # Negation pattern - un-ignore the path
                    should_ignore_path = False
                else:
                    # Regular pattern - ignore the path
                    should_ignore_path = True
        
        return should_ignore_path
    
    def is_path_allowlisted(self, path: str) -> bool:
        """
        Check if a path is within the allowlisted root directories.
        
        This method enforces the allowlist security policy by ensuring that
        only paths within explicitly configured root directories can be accessed.
        
        Args:
            path: Path to check against allowlist
            
        Returns:
            True if path is within allowlisted roots, False otherwise
        """
        try:
            check_path = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            # Cannot resolve path, consider it not allowlisted
            return False
        
        # Check if path is under any of the allowlisted roots
        for root in self.roots:
            try:
                root_path = Path(root).resolve()
                
                # Check if check_path is under root_path
                try:
                    check_path.relative_to(root_path)
                    return True  # Path is under this root
                except ValueError:
                    # relative_to failed, path is not under this root
                    continue
                    
            except (OSError, RuntimeError):
                # Cannot resolve root path, skip it
                continue
        
        return False
    
    def validate_path_access(self, path: str) -> None:
        """
        Validate that a path can be accessed according to allowlist policies.
        
        This method provides runtime validation to ensure that file operations
        only occur within allowlisted directories.
        
        Args:
            path: Path to validate for access
            
        Raises:
            ValueError: If path is not allowlisted or violates security policies
        """
        if not self.is_path_allowlisted(path):
            raise ValueError(
                f"Path '{path}' is not within any allowlisted root directory. "
                f"Allowlisted roots: {self.roots}"
            )
        
        # Additional security check - ensure path doesn't violate ignore patterns
        if self.should_ignore(path):
            raise ValueError(
                f"Path '{path}' matches ignore patterns and should not be accessed"
            )
    
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
        
        # Allowlist security warnings
        allowlist_warnings = self._get_allowlist_warnings()
        warnings.extend(allowlist_warnings)
        
        return warnings
    
    def _get_allowlist_warnings(self) -> List[str]:
        """
        Get warnings related to allowlist security configuration.
        
        Returns:
            List of allowlist-related warning messages
        """
        warnings = []
        
        # Check for overly broad root directories
        for root in self.roots:
            try:
                root_path = Path(root).resolve()
                
                # Warn about very broad roots
                if str(root_path) in ["/", "C:\\", str(Path.home().parent)]:
                    warnings.append(
                        f"Root directory '{root}' is very broad and may impact performance "
                        f"and security. Consider using more specific directories."
                    )
                
                # Warn about system directories that might be sensitive
                sensitive_indicators = ["/usr", "/var", "/opt", "C:\\Program Files"]
                for indicator in sensitive_indicators:
                    if str(root_path).startswith(indicator):
                        warnings.append(
                            f"Root directory '{root}' appears to be in a system directory. "
                            f"Ensure this is intentional and appropriate ignore patterns are set."
                        )
                        break
                
            except (OSError, RuntimeError):
                # Cannot resolve path, skip warnings for this root
                continue
        
        # Check if user has too many roots (performance concern)
        if len(self.roots) > 20:
            warnings.append(
                f"Large number of root directories ({len(self.roots)}) may impact "
                f"performance. Consider consolidating or using fewer, broader roots "
                f"with appropriate ignore patterns."
            )
        
        # Check for missing ignore patterns that might be important for security
        important_ignores = [
            "**/.git/**",
            "**/.env",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**"
        ]
        
        missing_ignores = []
        for important_ignore in important_ignores:
            if not any(ignore == important_ignore for ignore in self.ignore):
                missing_ignores.append(important_ignore)
        
        if missing_ignores:
            warnings.append(
                f"Consider adding these common ignore patterns for security and performance: "
                f"{', '.join(missing_ignores[:3])}{'...' if len(missing_ignores) > 3 else ''}"
            )
        
        return warnings
    
    def get_preview_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what would be searched in dry-run mode.
        
        Returns:
            Dictionary containing preview information about the search scope
        """
        accessible_roots = self.get_accessible_roots()
        inaccessible_roots = self.get_inaccessible_roots()
        
        # Estimate file counts for accessible roots
        estimated_files = 0
        preview_files = []
        
        for root in accessible_roots[:5]:  # Limit to first 5 roots for preview
            try:
                root_path = Path(root)
                if root_path.exists() and root_path.is_dir():
                    file_count, sample_files = self._preview_directory_contents(
                        root_path, 
                        max_files=self.security.preview_max_files // len(accessible_roots),
                        max_depth=self.security.preview_max_depth
                    )
                    estimated_files += file_count
                    preview_files.extend(sample_files)
            except (OSError, PermissionError):
                continue
        
        return {
            'total_roots': len(self.roots),
            'accessible_roots': len(accessible_roots),
            'inaccessible_roots': len(inaccessible_roots),
            'estimated_files': estimated_files,
            'sample_files': preview_files[:self.security.preview_max_files],
            'ignore_patterns_count': len(self.ignore),
            'dry_run_enabled': self.security.dry_run,
            'limits': {
                'max_files': self.limits.max_files,
                'max_file_size': self.limits.get_max_size_human_readable(),
                'max_results': self.limits.max_results
            }
        }
    
    def _preview_directory_contents(self, root_path: Path, max_files: int = 50, max_depth: int = 3) -> tuple[int, List[Dict[str, Any]]]:
        """
        Preview directory contents without full traversal.
        
        Args:
            root_path: Root directory to preview
            max_files: Maximum number of sample files to collect
            max_depth: Maximum depth to traverse
            
        Returns:
            Tuple of (estimated_file_count, sample_files_list)
        """
        sample_files = []
        total_files = 0
        
        try:
            for current_depth in range(max_depth + 1):
                if len(sample_files) >= max_files:
                    break
                
                # Use iterdir for shallow traversal at each depth
                if current_depth == 0:
                    items_to_check = [root_path]
                else:
                    # Get directories from previous depth
                    dirs_at_depth = []
                    for item in root_path.rglob('*'):
                        if item.is_dir() and len(item.relative_to(root_path).parts) == current_depth:
                            dirs_at_depth.append(item)
                    items_to_check = dirs_at_depth
                
                for directory in items_to_check:
                    if len(sample_files) >= max_files:
                        break
                    
                    try:
                        for item in directory.iterdir():
                            if item.is_file():
                                # Check if file should be ignored
                                relative_path = str(item.relative_to(root_path))
                                if not self.should_ignore(relative_path):
                                    total_files += 1
                                    
                                    if len(sample_files) < max_files:
                                        try:
                                            stat = item.stat()
                                            sample_files.append({
                                                'path': str(item.relative_to(root_path)),
                                                'size': stat.st_size,
                                                'size_human': self._format_file_size(stat.st_size),
                                                'modified': stat.st_mtime,
                                                'extension': item.suffix.lower() if item.suffix else None
                                            })
                                        except (OSError, PermissionError):
                                            # Skip files we can't access
                                            continue
                    except (OSError, PermissionError):
                        # Skip directories we can't access
                        continue
        
        except (OSError, PermissionError):
            # If we can't access the root, return empty results
            pass
        
        return total_files, sample_files
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Human-readable size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def generate_preview_report(self) -> str:
        """
        Generate a human-readable preview report for dry-run mode.
        
        Returns:
            Formatted preview report as string
        """
        preview = self.get_preview_summary()
        
        report_lines = [
            "# AI Filesystem Finder - Search Preview",
            "",
            "## Configuration Summary",
            f"- **Dry Run Mode**: {'Enabled' if self.security.dry_run else 'Disabled'}",
            f"- **Total Root Directories**: {preview['total_roots']}",
            f"- **Accessible Roots**: {preview['accessible_roots']}",
            f"- **Inaccessible Roots**: {preview['inaccessible_roots']}",
            f"- **Ignore Patterns**: {preview['ignore_patterns_count']} configured",
            "",
            "## Search Scope Limits",
            f"- **Maximum Files**: {preview['limits']['max_files']:,}",
            f"- **Maximum File Size**: {preview['limits']['max_file_size']}",
            f"- **Maximum Results**: {preview['limits']['max_results']}",
            "",
            "## Estimated Search Scope",
            f"- **Estimated Files**: ~{preview['estimated_files']:,} files",
            ""
        ]
        
        # Add accessible roots
        if preview['accessible_roots'] > 0:
            report_lines.extend([
                "## Accessible Root Directories",
                ""
            ])
            for root in self.get_accessible_roots():
                report_lines.append(f"- `{root}`")
            report_lines.append("")
        
        # Add inaccessible roots if any
        if preview['inaccessible_roots'] > 0:
            report_lines.extend([
                "## ⚠️ Inaccessible Root Directories",
                ""
            ])
            for root in self.get_inaccessible_roots():
                report_lines.append(f"- `{root}` (not found or no permission)")
            report_lines.append("")
        
        # Add sample files
        if preview['sample_files']:
            report_lines.extend([
                "## Sample Files (Preview)",
                "",
                "| File Path | Size | Extension |",
                "|-----------|------|-----------|"
            ])
            
            for file_info in preview['sample_files'][:20]:  # Show max 20 files
                path = file_info['path']
                size = file_info['size_human']
                ext = file_info['extension'] or 'none'
                report_lines.append(f"| `{path}` | {size} | {ext} |")
            
            if len(preview['sample_files']) > 20:
                report_lines.append(f"| ... and {len(preview['sample_files']) - 20} more files | | |")
            
            report_lines.append("")
        
        # Add ignore patterns
        if self.ignore:
            report_lines.extend([
                "## Ignore Patterns",
                "",
                "The following patterns will be excluded from search:",
                ""
            ])
            for pattern in self.ignore[:10]:  # Show first 10 patterns
                report_lines.append(f"- `{pattern}`")
            
            if len(self.ignore) > 10:
                report_lines.append(f"- ... and {len(self.ignore) - 10} more patterns")
            
            report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if preview['estimated_files'] > self.limits.max_files:
            report_lines.append(f"⚠️ **Estimated files ({preview['estimated_files']:,}) exceeds limit ({self.limits.max_files:,})**")
            report_lines.append("   Consider adding more ignore patterns or reducing root directories.")
            report_lines.append("")
        
        if preview['inaccessible_roots'] > 0:
            report_lines.append("⚠️ **Some root directories are inaccessible**")
            report_lines.append("   Check paths and permissions for inaccessible roots.")
            report_lines.append("")
        
        if not self.security.dry_run:
            report_lines.append("💡 **Enable dry-run mode** to preview searches without execution:")
            report_lines.append("   Set `security.dry_run: true` in your configuration.")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "",
            "*This preview shows what would be searched based on your current configuration.*",
            "*Actual results may vary based on file system changes and search queries.*"
        ])
        
        return "\n".join(report_lines)
    
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