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


if __name__ == "__main__":
    pytest.main([__file__])