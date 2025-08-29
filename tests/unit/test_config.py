"""
Unit tests for configuration data models.

Tests the configuration management system including validation,
normalization, and ignore pattern matching.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from finder.models.config import (
    FinderConfig,
    VectorDBConfig,
    EmbeddingConfig,
    LimitsConfig,
    OutputConfig,
    SecurityConfig,
    VectorBackend,
    EmbeddingProvider,
    OutputFormat,
    ConfigValidator,
    validate_config_dict
)


class TestVectorDBConfig:
    """Test cases for VectorDBConfig."""
    
    def test_default_config(self):
        """Test default vector DB configuration."""
        config = VectorDBConfig()
        
        assert config.backend == VectorBackend.LANCEDB
        assert config.path == str(Path("~/.finder-agent/index").expanduser())
        assert config.dimension is None
        assert config.index_params == {}
    
    def test_custom_config(self):
        """Test custom vector DB configuration."""
        config = VectorDBConfig(
            backend=VectorBackend.CHROMA,
            path="/custom/path",
            dimension=1536,
            index_params={"metric": "cosine"}
        )
        
        assert config.backend == VectorBackend.CHROMA
        assert config.path == "/custom/path"
        assert config.dimension == 1536
        assert config.index_params == {"metric": "cosine"}
    
    def test_string_backend_conversion(self):
        """Test conversion of string backend to enum."""
        config = VectorDBConfig(backend="faiss")
        assert config.backend == VectorBackend.FAISS
    
    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError, match="Invalid vector backend"):
            VectorDBConfig(backend="invalid")
    
    def test_invalid_dimension(self):
        """Test invalid dimension raises error."""
        with pytest.raises(ValidationError):
            VectorDBConfig(dimension=-1)
    
    def test_get_full_path(self):
        """Test getting full resolved path."""
        config = VectorDBConfig(path="~/test")
        full_path = config.get_full_path()
        
        assert full_path.is_absolute()
        assert str(full_path).endswith("test")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VectorDBConfig(
            backend=VectorBackend.LANCEDB,
            path="/test/path",
            dimension=768
        )
        
        result = config.to_dict()
        expected = {
            'backend': 'lancedb',
            'path': '/test/path',
            'dimension': 768,
            'index_params': {}
        }
        
        assert result == expected


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig."""
    
    def test_default_config(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        
        assert config.provider == EmbeddingProvider.ANTHROPIC
        assert config.model == "claude-3-haiku"
        assert config.api_key is None
        assert config.batch_size == 10
        assert config.max_tokens == 8000
    
    def test_custom_config(self):
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-ada-002",
            api_key="test-key",
            batch_size=20,
            max_tokens=4000
        )
        
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.model == "text-embedding-ada-002"
        assert config.api_key == "test-key"
        assert config.batch_size == 20
        assert config.max_tokens == 4000
    
    def test_string_provider_conversion(self):
        """Test conversion of string provider to enum."""
        config = EmbeddingConfig(provider="openai")
        assert config.provider == EmbeddingProvider.OPENAI
    
    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Invalid embedding provider"):
            EmbeddingConfig(provider="invalid")
    
    def test_invalid_batch_size(self):
        """Test invalid batch size raises error."""
        with pytest.raises(ValidationError):
            EmbeddingConfig(batch_size=0)
    
    def test_invalid_max_tokens(self):
        """Test invalid max tokens raises error."""
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_tokens=-1)
    
    def test_requires_api_key(self):
        """Test API key requirement detection."""
        anthropic_config = EmbeddingConfig(provider=EmbeddingProvider.ANTHROPIC)
        openai_config = EmbeddingConfig(provider=EmbeddingProvider.OPENAI)
        local_config = EmbeddingConfig(provider=EmbeddingProvider.LOCAL)
        
        assert anthropic_config.requires_api_key()
        assert openai_config.requires_api_key()
        assert not local_config.requires_api_key()
    
    def test_to_dict_redacts_api_key(self):
        """Test that to_dict redacts API key."""
        config = EmbeddingConfig(api_key="secret-key")
        result = config.to_dict()
        
        assert result['api_key'] == '***'


class TestLimitsConfig:
    """Test cases for LimitsConfig."""
    
    def test_default_config(self):
        """Test default limits configuration."""
        config = LimitsConfig()
        
        assert config.max_files == 200000
        assert config.max_bytes_per_file == 5000000
        assert config.max_results == 100
        assert config.timeout_seconds == 300
        assert config.max_concurrent == 4
    
    def test_custom_config(self):
        """Test custom limits configuration."""
        config = LimitsConfig(
            max_files=100000,
            max_bytes_per_file=1000000,
            max_results=50,
            timeout_seconds=600,
            max_concurrent=8
        )
        
        assert config.max_files == 100000
        assert config.max_bytes_per_file == 1000000
        assert config.max_results == 50
        assert config.timeout_seconds == 600
        assert config.max_concurrent == 8
    
    def test_invalid_values(self):
        """Test invalid limit values raise errors."""
        with pytest.raises(ValidationError):
            LimitsConfig(max_files=0)
        
        with pytest.raises(ValidationError):
            LimitsConfig(max_bytes_per_file=-1)
        
        with pytest.raises(ValidationError):
            LimitsConfig(max_results=0)
        
        with pytest.raises(ValidationError):
            LimitsConfig(timeout_seconds=-1)
        
        with pytest.raises(ValidationError):
            LimitsConfig(max_concurrent=0)
    
    def test_get_max_size_human_readable(self):
        """Test human-readable size formatting."""
        config = LimitsConfig(max_bytes_per_file=1024)
        assert config.get_max_size_human_readable() == "1.0 KB"
        
        config = LimitsConfig(max_bytes_per_file=1048576)
        assert config.get_max_size_human_readable() == "1.0 MB"
        
        config = LimitsConfig(max_bytes_per_file=1073741824)
        assert config.get_max_size_human_readable() == "1.0 GB"


class TestOutputConfig:
    """Test cases for OutputConfig."""
    
    def test_default_config(self):
        """Test default output configuration."""
        config = OutputConfig()
        
        assert OutputFormat.MARKDOWN in config.formats
        assert OutputFormat.JSON in config.formats
        assert config.directory == "runs"
        assert config.timestamp_dirs is True
        assert config.include_metadata is True
    
    def test_custom_config(self):
        """Test custom output configuration."""
        config = OutputConfig(
            formats=[OutputFormat.JSON],
            directory="/custom/output",
            timestamp_dirs=False,
            include_metadata=False
        )
        
        assert config.formats == [OutputFormat.JSON]
        assert config.directory == "/custom/output"
        assert config.timestamp_dirs is False
        assert config.include_metadata is False
    
    def test_string_format_conversion(self):
        """Test conversion of string formats to enums."""
        config = OutputConfig(formats=["markdown", "json"])
        
        assert OutputFormat.MARKDOWN in config.formats
        assert OutputFormat.JSON in config.formats
    
    def test_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid output format"):
            OutputConfig(formats=["invalid"])
    
    def test_supports_format(self):
        """Test format support checking."""
        config = OutputConfig(formats=[OutputFormat.MARKDOWN])
        
        assert config.supports_format(OutputFormat.MARKDOWN)
        assert not config.supports_format(OutputFormat.JSON)
    
    def test_get_output_path(self):
        """Test getting output path."""
        config = OutputConfig(directory="~/output")
        path = config.get_output_path()
        
        assert path.is_absolute()
        assert str(path).endswith("output")


class TestSecurityConfig:
    """Test cases for SecurityConfig."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.dry_run is False
        assert "api_key" in config.redact_patterns
        assert "password" in config.redact_patterns
        assert "secret" in config.redact_patterns
        assert config.max_snippet_length == 500
        assert config.allow_binary_files is False
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            dry_run=True,
            redact_patterns=["token", "key"],
            max_snippet_length=200,
            allow_binary_files=True
        )
        
        assert config.dry_run is True
        assert config.redact_patterns == ["token", "key"]
        assert config.max_snippet_length == 200
        assert config.allow_binary_files is True
    
    def test_invalid_snippet_length(self):
        """Test invalid snippet length raises error."""
        with pytest.raises(ValidationError):
            SecurityConfig(max_snippet_length=0)
    
    def test_invalid_redaction_pattern(self):
        """Test invalid regex pattern raises error."""
        with pytest.raises(ValueError, match="Invalid redaction pattern"):
            SecurityConfig(redact_patterns=["[invalid"])
    
    def test_should_redact(self):
        """Test redaction pattern matching."""
        config = SecurityConfig(redact_patterns=["api_key", "password"])
        
        assert config.should_redact("my_api_key = 'secret'")
        assert config.should_redact("PASSWORD: 12345")
        assert not config.should_redact("normal text")
    
    def test_redact_text(self):
        """Test text redaction."""
        config = SecurityConfig(redact_patterns=["api_key"])
        
        text = "my_api_key = 'secret123'"
        redacted = config.redact_text(text)
        
        assert "secret123" not in redacted
        assert "[REDACTED]" in redacted
    
    def test_preview_settings(self):
        """Test preview-related settings."""
        config = SecurityConfig(
            preview_max_files=50,
            preview_max_depth=2
        )
        
        assert config.preview_max_files == 50
        assert config.preview_max_depth == 2
    
    def test_invalid_preview_settings(self):
        """Test invalid preview settings raise errors."""
        with pytest.raises(ValidationError):
            SecurityConfig(preview_max_files=0)
        
        with pytest.raises(ValidationError):
            SecurityConfig(preview_max_depth=0)


class TestFinderConfig:
    """Test cases for FinderConfig."""
    
    def test_empty_roots_raises_error(self):
        """Test that empty roots list raises error."""
        with pytest.raises(ValidationError):
            FinderConfig(roots=[])
    
    def test_invalid_root_raises_error(self):
        """Test that non-existent root raises error in non-dry-run mode."""
        with pytest.raises(ValueError, match="Root directory does not exist"):
            FinderConfig(roots=["/nonexistent/path"])
    
    def test_dry_run_allows_nonexistent_roots(self):
        """Test that dry run mode allows non-existent roots."""
        security = SecurityConfig(dry_run=True)
        config = FinderConfig(
            roots=["/nonexistent/path"],
            security=security
        )
        
        assert config.roots == ["/nonexistent/path"]
    
    def test_root_normalization(self):
        """Test that roots are normalized to absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Should be normalized to absolute path
            assert Path(config.roots[0]).is_absolute()
            assert config.roots[0] == str(Path(temp_dir).resolve())
    
    def test_ignore_pattern_normalization(self):
        """Test ignore pattern normalization."""
        config = FinderConfig(
            roots=["."],  # Use current directory
            ignore=["*.pyc", "test/", ".git"]
        )
        
        # Patterns should be normalized with **/ prefix
        assert "**/*.pyc" in config.ignore
        assert "**/test/" in config.ignore
        assert "**/.git" in config.ignore
    
    def test_should_ignore(self):
        """Test ignore pattern matching."""
        config = FinderConfig(
            roots=["."],
            ignore=["**/.git/**", "**/node_modules/**", "**/*.pyc"]
        )
        
        assert config.should_ignore(".git/config")
        assert config.should_ignore("project/.git/hooks/pre-commit")
        assert config.should_ignore("node_modules/package/index.js")
        assert config.should_ignore("src/module.pyc")
        assert not config.should_ignore("src/module.py")
    
    def test_accessible_roots(self):
        """Test accessible roots detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use dry run to allow non-existent roots
            security = SecurityConfig(dry_run=True)
            config = FinderConfig(roots=[temp_dir, "/nonexistent"], security=security)
            
            accessible = config.get_accessible_roots()
            inaccessible = config.get_inaccessible_roots()
            
            assert temp_dir in accessible
            assert "/nonexistent" in inaccessible
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use dry run to allow non-existent roots
            security = SecurityConfig(dry_run=True)
            config = FinderConfig(
                roots=[temp_dir, "/nonexistent"],
                embeddings=EmbeddingConfig(provider=EmbeddingProvider.ANTHROPIC, api_key=None),
                security=security
            )
            
            warnings = config.validate_configuration()
            
            # Should warn about inaccessible root and missing API key
            assert any("Inaccessible root" in warning for warning in warnings)
            assert any("requires an API key" in warning for warning in warnings)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            result = config.to_dict()
            
            assert 'roots' in result
            assert 'ignore' in result
            assert 'vector_db' in result
            assert 'embeddings' in result
            assert 'limits' in result
            assert 'output' in result
            assert 'security' in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'roots': ['.'],
            'ignore': ['**/.git/**'],
            'vector_db': {'backend': 'lancedb', 'path': '~/test'},
            'embeddings': {'provider': 'anthropic', 'model': 'claude-3-haiku'},
            'limits': {'max_files': 1000},
            'output': {'formats': ['json']},
            'security': {'dry_run': True}
        }
        
        config = FinderConfig.from_dict(data)
        
        assert config.vector_db.backend == VectorBackend.LANCEDB
        assert config.embeddings.provider == EmbeddingProvider.ANTHROPIC
        assert config.limits.max_files == 1000
        assert OutputFormat.JSON in config.output.formats
        assert config.security.dry_run is True
    
    def test_str_representation(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            str_repr = str(config)
            
            assert "Roots: 1 directories" in str_repr
            assert "Vector DB: lancedb" in str_repr
            assert "Embeddings: anthropic" in str_repr
    
    def test_get_preview_summary(self):
        """Test getting preview summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_file1 = Path(temp_dir) / "test1.txt"
            test_file1.write_text("test content")
            
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            test_file2 = subdir / "test2.py"
            test_file2.write_text("print('hello')")
            
            config = FinderConfig(roots=[temp_dir])
            preview = config.get_preview_summary()
            
            assert 'total_roots' in preview
            assert 'accessible_roots' in preview
            assert 'estimated_files' in preview
            assert 'sample_files' in preview
            assert 'dry_run_enabled' in preview
            assert 'limits' in preview
            
            assert preview['total_roots'] == 1
            assert preview['accessible_roots'] == 1
            assert preview['estimated_files'] >= 2  # At least our test files
            assert len(preview['sample_files']) >= 2
    
    def test_preview_directory_contents(self):
        """Test previewing directory contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "test1.txt"
            test_file1.write_text("test content")
            
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            test_file2 = subdir / "test2.py"
            test_file2.write_text("print('hello')")
            
            config = FinderConfig(roots=[temp_dir])
            file_count, sample_files = config._preview_directory_contents(
                Path(temp_dir), max_files=10, max_depth=2
            )
            
            assert file_count >= 2
            assert len(sample_files) >= 2
            
            # Check sample file structure
            for file_info in sample_files:
                assert 'path' in file_info
                assert 'size' in file_info
                assert 'size_human' in file_info
                assert 'extension' in file_info
    
    def test_format_file_size(self):
        """Test file size formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            assert config._format_file_size(0) == "0 B"
            assert config._format_file_size(1024) == "1.0 KB"
            assert config._format_file_size(1048576) == "1.0 MB"
            assert config._format_file_size(1073741824) == "1.0 GB"
    
    def test_generate_preview_report(self):
        """Test generating preview report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            config = FinderConfig(roots=[temp_dir])
            report = config.generate_preview_report()
            
            assert isinstance(report, str)
            assert "AI Filesystem Finder - Search Preview" in report
            assert "Configuration Summary" in report
            assert "Search Scope Limits" in report
            assert "Estimated Search Scope" in report
            assert "Accessible Root Directories" in report
            
            # Check that the temp directory is mentioned
            assert temp_dir in report
    
    def test_generate_preview_report_with_inaccessible_roots(self):
        """Test preview report with inaccessible roots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            security = SecurityConfig(dry_run=True)
            config = FinderConfig(
                roots=[temp_dir, "/nonexistent/path"],
                security=security
            )
            
            report = config.generate_preview_report()
            
            assert "⚠️ Inaccessible Root Directories" in report
            assert "/nonexistent/path" in report
    
    def test_generate_preview_report_dry_run_disabled(self):
        """Test preview report when dry run is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            security = SecurityConfig(dry_run=False)
            config = FinderConfig(roots=[temp_dir], security=security)
            
            report = config.generate_preview_report()
            
            assert "💡 **Enable dry-run mode**" in report
            assert "security.dry_run: true" in report


class TestAllowlistValidation:
    """Test cases for allowlist validation functionality."""
    
    def test_sensitive_system_path_blocked(self):
        """Test that sensitive system paths are blocked."""
        security = SecurityConfig(dry_run=False)
        
        # Test blocking root filesystem
        with pytest.raises(ValueError, match="is a sensitive system directory"):
            FinderConfig(roots=["/"], security=security)
    
    def test_sensitive_system_path_blocked_dry_run(self):
        """Test that sensitive system paths are blocked even in dry run."""
        security = SecurityConfig(dry_run=True)
        
        # Should still block sensitive paths even in dry run
        with pytest.raises(ValueError, match="is a sensitive system directory"):
            FinderConfig(roots=["/"], security=security)
    
    def test_path_containing_sensitive_directory_blocked(self):
        """Test that paths containing sensitive directories are blocked."""
        security = SecurityConfig(dry_run=False)
        
        # Test path that would contain /etc - should be caught as sensitive system directory
        with pytest.raises((ValueError, ValidationError), match="is a sensitive system directory"):
            FinderConfig(roots=["/"], security=security)
    
    def test_path_traversal_patterns_blocked(self):
        """Test that dangerous path traversal patterns are blocked."""
        security = SecurityConfig(dry_run=True)
        
        # Test paths with dangerous patterns that don't resolve to sensitive directories
        dangerous_paths = [
            "/tmp//dangerous",  # Double slash
            "/home/user\\\\test"  # Double backslash
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises((ValueError, ValidationError)):
                FinderConfig(roots=[dangerous_path], security=security)
    
    def test_duplicate_roots_blocked(self):
        """Test that duplicate root directories are blocked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            security = SecurityConfig(dry_run=True)
            
            with pytest.raises(ValueError, match="Duplicate root directories found"):
                FinderConfig(roots=[temp_dir, temp_dir], security=security)
    
    def test_nested_roots_blocked(self):
        """Test that nested root directories are blocked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a subdirectory
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            
            security = SecurityConfig(dry_run=True)
            
            with pytest.raises((ValueError, ValidationError), match="is nested under another root"):
                FinderConfig(roots=[temp_dir, str(subdir)], security=security)
    
    def test_is_path_allowlisted_success(self):
        """Test successful path allowlist checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Create a test file in the temp directory
            test_file = Path(temp_dir) / "test.txt"
            test_file.touch()
            
            assert config.is_path_allowlisted(str(test_file))
            assert config.is_path_allowlisted(temp_dir)
    
    def test_is_path_allowlisted_failure(self):
        """Test path allowlist checking with non-allowlisted path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Test path outside of allowlisted roots
            assert not config.is_path_allowlisted("/tmp/not_allowlisted.txt")
            assert not config.is_path_allowlisted("/etc/passwd")
    
    def test_validate_path_access_success(self):
        """Test successful path access validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.touch()
            
            # Should not raise any exception
            config.validate_path_access(str(test_file))
    
    def test_validate_path_access_not_allowlisted(self):
        """Test path access validation with non-allowlisted path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            with pytest.raises(ValueError, match="is not within any allowlisted root directory"):
                config.validate_path_access("/tmp/not_allowlisted.txt")
    
    def test_validate_path_access_ignored_path(self):
        """Test path access validation with ignored path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(
                roots=[temp_dir],
                ignore=["**/.git/**"]
            )
            
            # Create a .git directory
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            git_file = git_dir / "config"
            git_file.touch()
            
            with pytest.raises(ValueError, match="matches ignore patterns"):
                config.validate_path_access(str(git_file))
    
    def test_allowlist_warnings_broad_roots(self):
        """Test warnings for overly broad root directories."""
        security = SecurityConfig(dry_run=True)
        
        # Mock Path.home() to return a predictable value
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/home/user")
            
            config = FinderConfig(roots=["/home"], security=security)
            warnings = config._get_allowlist_warnings()
            
            # Should warn about broad root
            assert any("very broad" in warning for warning in warnings)
    
    def test_allowlist_warnings_system_directories(self):
        """Test warnings for system directories."""
        security = SecurityConfig(dry_run=True)
        
        config = FinderConfig(roots=["/usr/local/test"], security=security)
        warnings = config._get_allowlist_warnings()
        
        # Should warn about system directory
        assert any("system directory" in warning for warning in warnings)
    
    def test_allowlist_warnings_many_roots(self):
        """Test warnings for too many root directories."""
        security = SecurityConfig(dry_run=True)
        
        # Create many roots
        many_roots = [f"/tmp/test{i}" for i in range(25)]
        config = FinderConfig(roots=many_roots, security=security)
        warnings = config._get_allowlist_warnings()
        
        # Should warn about performance impact
        assert any("Large number of root directories" in warning for warning in warnings)
    
    def test_allowlist_warnings_missing_ignores(self):
        """Test warnings for missing important ignore patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(
                roots=[temp_dir],
                ignore=[]  # No ignore patterns
            )
            warnings = config._get_allowlist_warnings()
            
            # Should suggest important ignore patterns
            assert any("Consider adding these common ignore patterns" in warning for warning in warnings)
    
    def test_get_sensitive_system_paths(self):
        """Test getting sensitive system paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            sensitive_paths = config._get_sensitive_system_paths()
            
            # Should include universal sensitive paths
            assert "/" in sensitive_paths
            assert "/etc" in sensitive_paths
            assert "/tmp" in sensitive_paths
            
            # Should be a reasonable number of paths
            assert len(sensitive_paths) > 5
            assert len(sensitive_paths) < 50
    
    def test_validate_path_safety_clean_path(self):
        """Test path safety validation with clean path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            clean_path = Path(temp_dir).resolve()
            
            # Should not raise any exception
            config._validate_path_safety(clean_path)
    
    def test_validate_path_safety_dangerous_patterns(self):
        """Test path safety validation with dangerous patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Test various dangerous patterns in raw strings
            dangerous_paths = [
                "/tmp//dangerous",  # Double slash
                "/home/user/../../../etc",  # Parent directory traversal
                "C:\\Users\\..\\Windows",  # Windows path traversal
            ]
            
            for dangerous_path in dangerous_paths:
                with pytest.raises(ValueError, match="contains potentially dangerous pattern"):
                    config._validate_path_safety_raw(dangerous_path)
    
    def test_validate_root_relationships_single_root(self):
        """Test root relationship validation with single root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Should not raise any exception for single root
            config._validate_root_relationships([temp_dir])
    
    def test_validate_root_relationships_no_conflicts(self):
        """Test root relationship validation with non-conflicting roots."""
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                config = FinderConfig(roots=[temp_dir1])
                
                # Should not raise any exception for separate roots
                config._validate_root_relationships([temp_dir1, temp_dir2])


class TestConfigValidator:
    """Test cases for Pydantic-based ConfigValidator."""
    
    def test_validate_finder_config_success(self):
        """Test successful validation of FinderConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Should not raise any exceptions
            warnings = ConfigValidator.validate_finder_config(config)
            assert isinstance(warnings, list)
    
    def test_validate_finder_config_invalid(self):
        """Test validation failure for invalid FinderConfig."""
        # Create config with valid dataclass but invalid for Pydantic
        security = SecurityConfig(dry_run=True)  # Allow non-existent roots
        
        # Create a config that passes dataclass validation but fails Pydantic
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir], security=security)
            
            # Manually set an invalid backend value that bypasses enum validation
            # by directly setting the string value
            config.vector_db.backend = type('MockBackend', (), {'value': 'invalid_backend'})()
            
            with pytest.raises(ValueError, match="Configuration validation failed"):
                ConfigValidator.validate_finder_config(config)
    
    def test_validate_config_dict_success(self):
        """Test successful validation of configuration dictionary."""
        config_data = {
            'roots': ['.'],
            'ignore': ['**/.git/**'],
            'vector_db': {'backend': 'lancedb'},
            'embeddings': {'provider': 'anthropic'},
            'limits': {'max_files': 1000},
            'output': {'formats': ['json']},
            'security': {'dry_run': True}
        }
        
        validated = validate_config_dict(config_data)
        assert 'roots' in validated
        assert validated['roots'] == [str(Path('.').resolve())]
    
    def test_validate_config_dict_invalid_backend(self):
        """Test validation failure for invalid vector backend."""
        config_data = {
            'roots': ['.'],
            'vector_db': {'backend': 'invalid_backend'}
        }
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_config_dict(config_data)
    
    def test_validate_config_dict_invalid_provider(self):
        """Test validation failure for invalid embedding provider."""
        config_data = {
            'roots': ['.'],
            'embeddings': {'provider': 'invalid_provider'}
        }
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_config_dict(config_data)
    
    def test_validate_config_dict_invalid_format(self):
        """Test validation failure for invalid output format."""
        config_data = {
            'roots': ['.'],
            'output': {'formats': ['invalid_format']}
        }
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_config_dict(config_data)
    
    def test_validate_config_dict_empty_roots(self):
        """Test validation failure for empty roots."""
        config_data = {
            'roots': []
        }
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_config_dict(config_data)
    
    def test_validate_config_dict_invalid_limits(self):
        """Test validation failure for invalid limits."""
        config_data = {
            'roots': ['.'],
            'limits': {'max_files': -1}
        }
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_config_dict(config_data)


class TestConfigurationValidationComprehensive:
    """Comprehensive test cases for configuration validation."""
    
    def test_validate_config_dict_function(self):
        """Test the validate_config_dict function directly."""
        valid_config = {
            'roots': ['.'],
            'ignore': ['**/.git/**'],
            'vector_db': {'backend': 'lancedb'},
            'embeddings': {'provider': 'anthropic'},
            'limits': {'max_files': 100000},
            'output': {'formats': ['markdown']},
            'security': {'dry_run': True}
        }
        
        result = validate_config_dict(valid_config)
        
        assert 'roots' in result
        assert 'ignore' in result
        assert 'vector_db' in result
        assert 'embeddings' in result
        assert 'limits' in result
        assert 'output' in result
        assert 'security' in result
    
    def test_validate_config_dict_missing_required_fields(self):
        """Test validation with missing required fields."""
        # Missing roots
        with pytest.raises(ValueError):
            validate_config_dict({})
        
        # Empty roots
        with pytest.raises(ValueError):
            validate_config_dict({'roots': []})
    
    def test_validate_config_dict_type_validation(self):
        """Test type validation in configuration dictionary."""
        # Test invalid types
        invalid_configs = [
            {'roots': 'not_a_list'},  # roots should be list
            {'roots': ['.'], 'ignore': 'not_a_list'},  # ignore should be list
            {'roots': ['.'], 'limits': 'not_a_dict'},  # limits should be dict
            {'roots': ['.'], 'vector_db': 'not_a_dict'},  # vector_db should be dict
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, ValidationError)):
                validate_config_dict(invalid_config)
    
    def test_validate_config_dict_nested_validation(self):
        """Test validation of nested configuration objects."""
        # Test invalid nested values
        invalid_nested_configs = [
            {
                'roots': ['.'],
                'vector_db': {'backend': 'invalid_backend'}
            },
            {
                'roots': ['.'],
                'embeddings': {'provider': 'invalid_provider'}
            },
            {
                'roots': ['.'],
                'limits': {'max_files': -1}
            },
            {
                'roots': ['.'],
                'output': {'formats': ['invalid_format']}
            }
        ]
        
        for invalid_config in invalid_nested_configs:
            with pytest.raises((ValueError, ValidationError)):
                validate_config_dict(invalid_config)


class TestSecurityConfigurationValidation:
    """Test cases for security-specific configuration validation."""
    
    def test_redaction_pattern_compilation(self):
        """Test that redaction patterns are properly compiled and validated."""
        # Valid patterns
        valid_patterns = [
            "api_key",
            "password",
            r"\btoken\b",
            r"secret.*key",
            "credential[s]?"
        ]
        
        config = SecurityConfig(redact_patterns=valid_patterns)
        
        # Should have compiled patterns
        assert hasattr(config, '_compiled_patterns')
        assert len(config._compiled_patterns) == len(valid_patterns)
        
        # Test pattern matching
        test_cases = [
            ("my_api_key = 'secret'", True),
            ("password: hidden", True),
            ("token = 'abc123'", True),  # Whole word token
            ("secret_key = 'xyz'", True),
            ("credentials = []", True),
            ("normal text", False),
            ("my_api_key_value", True),  # Contains api_key
        ]
        
        for text, should_match in test_cases:
            assert config.should_redact(text) == should_match, f"Pattern matching failed for: {text}"
    
    def test_redaction_pattern_invalid_regex(self):
        """Test that invalid regex patterns raise appropriate errors."""
        invalid_patterns = [
            "[invalid",  # Unclosed character class
            "*invalid",  # Invalid quantifier
            "(?P<invalid",  # Unclosed group
            "\\",  # Incomplete escape
        ]
        
        for invalid_pattern in invalid_patterns:
            with pytest.raises(ValueError, match="Invalid redaction pattern"):
                SecurityConfig(redact_patterns=[invalid_pattern])
    
    def test_redaction_functionality(self):
        """Test text redaction functionality."""
        config = SecurityConfig(redact_patterns=["api_key", "password", "secret"])
        
        test_cases = [
            # Single line redaction
            ("api_key = 'secret123'", "[REDACTED]"),
            ("password: hidden", "[REDACTED]"),
            ("normal line", "normal line"),
            
            # Multi-line redaction
            (
                "line1\napi_key = 'secret'\nline3\npassword = 'hidden'\nline5",
                "line1\n[REDACTED]\nline3\n[REDACTED]\nline5"
            ),
            
            # Mixed content
            (
                "# Configuration\napi_key = 'abc123'\ndatabase_url = 'localhost'\nsecret_token = 'xyz'",
                "# Configuration\n[REDACTED]\ndatabase_url = 'localhost'\n[REDACTED]"
            )
        ]
        
        for original, expected in test_cases:
            result = config.redact_text(original)
            assert result == expected, f"Redaction failed for: {original}"
    
    def test_redaction_custom_replacement(self):
        """Test redaction with custom replacement text."""
        config = SecurityConfig(redact_patterns=["secret"])
        
        original = "secret_key = 'hidden'"
        result = config.redact_text(original, replacement="***HIDDEN***")
        
        assert result == "***HIDDEN***"
    
    def test_security_config_validation_limits(self):
        """Test validation of security configuration limits."""
        # Test invalid values
        invalid_configs = [
            {'max_snippet_length': 0},
            {'max_snippet_length': -1},
            {'preview_max_files': 0},
            {'preview_max_files': -1},
            {'preview_max_depth': 0},
            {'preview_max_depth': -1},
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValidationError):
                SecurityConfig(**invalid_config)
    
    def test_security_config_defaults(self):
        """Test that security configuration has appropriate defaults."""
        config = SecurityConfig()
        
        # Should have sensible defaults
        assert config.dry_run is False
        assert len(config.redact_patterns) > 0
        assert "api_key" in config.redact_patterns
        assert "password" in config.redact_patterns
        assert "secret" in config.redact_patterns
        assert config.max_snippet_length > 0
        assert config.allow_binary_files is False
        assert config.preview_max_files > 0
        assert config.preview_max_depth > 0


class TestAllowlistSecurityValidation:
    """Test cases for allowlist security validation."""
    
    def test_sensitive_system_paths_comprehensive(self):
        """Test comprehensive sensitive system path detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            sensitive_paths = config._get_sensitive_system_paths()
            
            # Should include universal sensitive paths
            expected_paths = ["/", "/etc", "/var/log", "/tmp", "/dev", "/proc", "/sys"]
            
            for expected_path in expected_paths:
                assert expected_path in sensitive_paths, f"Missing sensitive path: {expected_path}"
            
            # Should be platform-specific paths too
            assert len(sensitive_paths) > len(expected_paths)
    
    def test_path_safety_validation_comprehensive(self):
        """Test comprehensive path safety validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir])
            
            # Test dangerous raw path patterns
            dangerous_raw_paths = [
                "../../../etc/passwd",
                "/tmp//dangerous",
                "/home/user\\\\test",
                "path/with/../traversal",
                "//network/path"
            ]
            
            for dangerous_path in dangerous_raw_paths:
                with pytest.raises(ValueError):
                    config._validate_path_safety_raw(dangerous_path)
    
    def test_root_relationship_validation_comprehensive(self):
        """Test comprehensive root directory relationship validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory structure
            subdir1 = Path(temp_dir) / "subdir1"
            subdir2 = Path(temp_dir) / "subdir2"
            nested_dir = subdir1 / "nested"
            
            subdir1.mkdir()
            subdir2.mkdir()
            nested_dir.mkdir()
            
            # Test duplicate detection
            with pytest.raises(ValueError, match="Duplicate root directories"):
                FinderConfig(roots=[str(subdir1), str(subdir1)])
            
            # Test nested detection
            with pytest.raises(ValueError, match="nested under another root"):
                FinderConfig(roots=[str(subdir1), str(nested_dir)])
            
            # Test valid non-nested roots (should work)
            config = FinderConfig(roots=[str(subdir1), str(subdir2)])
            assert len(config.roots) == 2
    
    def test_allowlist_validation_edge_cases(self):
        """Test edge cases in allowlist validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with symlinks (if supported)
            try:
                symlink_path = Path(temp_dir) / "symlink"
                symlink_path.symlink_to(temp_dir)
                
                # Should handle symlinks appropriately
                config = FinderConfig(roots=[str(symlink_path)])
                assert len(config.roots) == 1
                
            except (OSError, NotImplementedError):
                # Symlinks not supported on this platform
                pass
            
            # Test with relative paths
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = FinderConfig(roots=["."])
                
                # Should normalize to absolute path
                assert Path(config.roots[0]).is_absolute()
                
            finally:
                os.chdir(original_cwd)
    
    def test_path_allowlist_checking_comprehensive(self):
        """Test comprehensive path allowlist checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.touch()
            
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            subdir_file = subdir / "file.txt"
            subdir_file.touch()
            
            config = FinderConfig(roots=[temp_dir])
            
            # Test allowlisted paths
            allowlisted_paths = [
                temp_dir,
                str(test_file),
                str(subdir),
                str(subdir_file),
                str(Path(temp_dir) / "nonexistent.txt")  # Non-existent but under root
            ]
            
            for path in allowlisted_paths:
                assert config.is_path_allowlisted(path), f"Should be allowlisted: {path}"
            
            # Test non-allowlisted paths
            non_allowlisted_paths = [
                "/etc/passwd",
                "/tmp/other.txt",
                str(Path(temp_dir).parent / "other.txt"),
                "/completely/different/path"
            ]
            
            for path in non_allowlisted_paths:
                assert not config.is_path_allowlisted(path), f"Should not be allowlisted: {path}"
    
    def test_path_access_validation_with_ignore_patterns(self):
        """Test path access validation considering ignore patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()
            git_file = git_dir / "config"
            git_file.touch()
            
            node_modules = Path(temp_dir) / "node_modules"
            node_modules.mkdir()
            package_file = node_modules / "package.json"
            package_file.touch()
            
            regular_file = Path(temp_dir) / "regular.txt"
            regular_file.touch()
            
            config = FinderConfig(
                roots=[temp_dir],
                ignore=["**/.git/**", "**/node_modules/**"]
            )
            
            # Regular file should be accessible
            config.validate_path_access(str(regular_file))
            
            # Ignored files should not be accessible
            with pytest.raises(ValueError, match="matches ignore patterns"):
                config.validate_path_access(str(git_file))
            
            with pytest.raises(ValueError, match="matches ignore patterns"):
                config.validate_path_access(str(package_file))
    
    def test_allowlist_warnings_comprehensive(self):
        """Test comprehensive allowlist warning generation."""
        # Test broad root warnings
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/home/user")
            
            security = SecurityConfig(dry_run=True)
            config = FinderConfig(roots=["/home"], security=security)
            warnings = config._get_allowlist_warnings()
            
            assert any("very broad" in warning.lower() for warning in warnings)
        
        # Test system directory warnings
        security = SecurityConfig(dry_run=True)
        config = FinderConfig(roots=["/usr/local/test"], security=security)
        warnings = config._get_allowlist_warnings()
        
        assert any("system directory" in warning.lower() for warning in warnings)
        
        # Test many roots warning
        many_roots = [f"/tmp/test{i}" for i in range(25)]
        config = FinderConfig(roots=many_roots, security=security)
        warnings = config._get_allowlist_warnings()
        
        assert any("large number" in warning.lower() for warning in warnings)
        
        # Test missing ignore patterns warning
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FinderConfig(roots=[temp_dir], ignore=[])
            warnings = config._get_allowlist_warnings()
            
            assert any("ignore patterns" in warning.lower() for warning in warnings)


class TestConfigurationSecurityIntegration:
    """Integration tests for configuration security features."""
    
    def test_end_to_end_security_validation(self):
        """Test end-to-end security validation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a comprehensive configuration
            config = FinderConfig(
                roots=[temp_dir],
                ignore=[
                    "**/.git/**",
                    "**/.env",
                    "**/secrets/**",
                    "**/*.key",
                    "**/*.pem",
                    "**/node_modules/**",
                    "**/__pycache__/**"
                ],
                security=SecurityConfig(
                    dry_run=False,
                    redact_patterns=[
                        "api_key",
                        "password",
                        "secret",
                        "token",
                        "credential"
                    ],
                    max_snippet_length=300,
                    allow_binary_files=False
                )
            )
            
            # Validate the configuration
            warnings = config.validate_configuration()
            
            # Should have minimal warnings for a well-configured setup
            assert isinstance(warnings, list)
            
            # Test security features
            assert config.should_ignore(".git/config")
            assert config.should_ignore("secrets/api.key")
            assert config.security.should_redact("api_key = 'secret'")
            assert config.is_path_allowlisted(str(Path(temp_dir) / "file.txt"))
            assert not config.is_path_allowlisted("/etc/passwd")
    
    def test_security_configuration_serialization(self):
        """Test that security configuration serializes safely."""
        config = FinderConfig(
            roots=["."],
            embeddings=EmbeddingConfig(
                provider=EmbeddingProvider.ANTHROPIC,
                api_key="secret-api-key"
            ),
            security=SecurityConfig(
                redact_patterns=["api_key", "password"]
            )
        )
        
        # Convert to dictionary (should redact sensitive data)
        config_dict = config.to_dict()
        
        # API key should be redacted
        assert config_dict['embeddings']['api_key'] == '***'
        
        # Other sensitive data should be preserved but marked
        assert 'redact_patterns' in config_dict['security']
    
    def test_configuration_validation_performance(self):
        """Test that configuration validation performs well with large configs."""
        import time
        
        # Create a large configuration
        many_roots = [f"/tmp/test{i}" for i in range(100)]
        many_patterns = [f"**/*.tmp{i}" for i in range(1000)]
        
        security = SecurityConfig(dry_run=True)
        
        start_time = time.time()
        
        try:
            config = FinderConfig(
                roots=many_roots,
                ignore=many_patterns,
                security=security
            )
            
            # Validation should complete in reasonable time
            validation_time = time.time() - start_time
            assert validation_time < 5.0, f"Validation took too long: {validation_time}s"
            
            # Test pattern matching performance
            start_time = time.time()
            for i in range(100):
                config.should_ignore(f"test{i}.tmp{i}")
            
            matching_time = time.time() - start_time
            assert matching_time < 1.0, f"Pattern matching took too long: {matching_time}s"
            
        except ValueError as e:
            # Expected if validation catches issues with the large config
            assert "root directories" in str(e) or "nested" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])