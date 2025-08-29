"""
Unit tests for the filesystem walker module.

Tests directory traversal, pattern matching, metadata extraction,
and filtering functionality of the FSWalker class.
"""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest

from src.finder.tools.fs_walker import FSWalker, create_filename_patterns, create_extension_patterns
from src.finder.models.config import FinderConfig
from src.finder.models.search_results import MatchType


class TestFSWalker:
    """Test cases for the FSWalker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_root = Path(self.temp_dir)
        
        # Create test directory structure
        self._create_test_structure()
        
        # Create test configuration
        self.config = FinderConfig(
            roots=[str(self.test_root)],
            ignore=["**/.git/**", "**/node_modules/**", "**/__pycache__/**"],
            limits={'max_files': 1000, 'max_bytes_per_file': 1000000}
        )
        
        self.walker = FSWalker(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_structure(self):
        """Create a test directory structure with various file types."""
        # Create directories
        (self.test_root / "src").mkdir()
        (self.test_root / "docs").mkdir()
        (self.test_root / "tests").mkdir()
        (self.test_root / ".git").mkdir()
        (self.test_root / "node_modules").mkdir()
        
        # Create test files
        test_files = [
            "src/main.py",
            "src/utils.py", 
            "src/config.json",
            "docs/readme.md",
            "docs/api.txt",
            "tests/test_main.py",
            ".git/config",
            "node_modules/package.json",
            "requirements.txt",
            "setup.py"
        ]
        
        for file_path in test_files:
            full_path = self.test_root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Content of {file_path}")
    
    def test_walk_paths_basic(self):
        """Test basic directory walking functionality."""
        patterns = [r"\.py$"]  # Match Python files
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        # Should find Python files but not ignored ones
        python_files = [m for m in matches if m.path.endswith('.py')]
        assert len(python_files) >= 3  # main.py, utils.py, test_main.py, setup.py
        
        # Check that ignored files are not included
        git_files = [m for m in matches if '.git' in m.path]
        assert len(git_files) == 0
    
    def test_ignore_patterns(self):
        """Test that ignore patterns are properly respected."""
        patterns = [r".*"]  # Match all files
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        # Should not include files in ignored directories
        paths = [m.path for m in matches]
        
        assert not any('.git' in path for path in paths)
        assert not any('node_modules' in path for path in paths)
    
    def test_pattern_matching(self):
        """Test regex pattern matching for filenames."""
        # Test extension pattern
        patterns = [r"\.md$"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        assert len(matches) >= 1
        assert all(m.path.endswith('.md') for m in matches)
        
        # Test filename pattern
        patterns = [r"main"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        assert len(matches) >= 1
        assert any('main' in Path(m.path).name for m in matches)
    
    def test_metadata_extraction(self):
        """Test file metadata extraction."""
        patterns = [r"requirements\.txt$"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        assert len(matches) == 1
        match = matches[0]
        
        assert match.metadata is not None
        assert match.metadata.size > 0
        assert match.metadata.modified_time is not None
        assert match.metadata.extension == '.txt'
        assert match.match_type == MatchType.LITERAL
    
    def test_apply_filters_size(self):
        """Test size-based filtering."""
        # Create matches with different sizes
        patterns = [r".*"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        # Filter by minimum size
        filters = {'size_min': 10}
        filtered = self.walker.apply_filters(matches, filters)
        
        assert all(m.metadata.size >= 10 for m in filtered)
        
        # Filter by maximum size
        filters = {'size_max': 100}
        filtered = self.walker.apply_filters(matches, filters)
        
        assert all(m.metadata.size <= 100 for m in filtered)
    
    def test_apply_filters_extension(self):
        """Test extension-based filtering."""
        patterns = [r".*"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        # Filter by extension
        filters = {'extensions': ['.py', '.txt']}
        filtered = self.walker.apply_filters(matches, filters)
        
        assert all(m.metadata.extension in ['.py', '.txt'] for m in filtered)
    
    def test_apply_filters_date(self):
        """Test date-based filtering."""
        patterns = [r".*"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        # Filter by modification date (recent files)
        recent_date = datetime.now() - timedelta(days=1)
        filters = {'modified_after': recent_date}
        filtered = self.walker.apply_filters(matches, filters)
        
        # All test files should be recent
        assert len(filtered) > 0
        assert all(m.metadata.modified_time >= recent_date for m in filtered)
    
    def test_stats_tracking(self):
        """Test that statistics are properly tracked."""
        self.walker.reset_stats()
        
        patterns = [r"\.py$"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        stats = self.walker.get_stats()
        
        assert stats['files_scanned'] > 0
        assert stats['files_matched'] > 0
        assert stats['directories_traversed'] > 0
        assert stats['files_ignored'] >= 0
    
    def test_binary_file_detection(self):
        """Test binary file detection."""
        # Create a binary file
        binary_file = self.test_root / "test.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        # Create a text file
        text_file = self.test_root / "test.txt"
        text_file.write_text("This is a text file")
        
        patterns = [r"test\.(bin|txt)$"]
        matches = list(self.walker.walk_paths([str(self.test_root)], patterns))
        
        binary_match = next((m for m in matches if m.path.endswith('.bin')), None)
        text_match = next((m for m in matches if m.path.endswith('.txt')), None)
        
        assert binary_match is not None
        assert text_match is not None
        assert binary_match.metadata.is_binary is True
        assert text_match.metadata.is_binary is False


class TestPatternCreation:
    """Test cases for pattern creation utility functions."""
    
    def test_create_filename_patterns(self):
        """Test filename pattern creation from query text."""
        patterns = create_filename_patterns("config file")
        
        assert len(patterns) > 0
        # Should create patterns for individual words and combined
        assert any('config' in pattern for pattern in patterns)
        assert any('file' in pattern for pattern in patterns)
    
    def test_create_extension_patterns(self):
        """Test extension pattern creation."""
        patterns = create_extension_patterns(['py', '.txt', 'json'])
        
        assert len(patterns) == 3
        # All patterns should end with $ to match end of filename
        assert all(pattern.endswith('$') for pattern in patterns)
        # Should handle extensions with and without leading dot
        assert any(r'\.py$' in pattern for pattern in patterns)
        assert any(r'\.txt$' in pattern for pattern in patterns)
        assert any(r'\.json$' in pattern for pattern in patterns)
    
    def test_empty_query_patterns(self):
        """Test pattern creation with empty query."""
        patterns = create_filename_patterns("")
        assert len(patterns) == 0
        
        patterns = create_filename_patterns("   ")
        assert len(patterns) == 0
    
    def test_special_characters_in_query(self):
        """Test pattern creation with special regex characters."""
        patterns = create_filename_patterns("file.config [test]")
        
        # Should escape special regex characters
        assert len(patterns) > 0
        # Patterns should be valid regex (no exceptions when compiling)
        import re
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern generated: {pattern}")


class TestDateParsing:
    """Test cases for date filter parsing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.config = FinderConfig(roots=[self.temp_dir])
        self.walker = FSWalker(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parse_relative_dates(self):
        """Test parsing of relative date strings."""
        # Test days
        result = self.walker._parse_date_filter("7 days")
        assert result is not None
        assert isinstance(result, datetime)
        
        # Test weeks
        result = self.walker._parse_date_filter("2 weeks")
        assert result is not None
        
        # Test months
        result = self.walker._parse_date_filter("1 month")
        assert result is not None
    
    def test_parse_absolute_dates(self):
        """Test parsing of absolute date strings."""
        # Test ISO format
        result = self.walker._parse_date_filter("2023-12-01")
        assert result is not None
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 1
        
        # Test with time
        result = self.walker._parse_date_filter("2023-12-01 15:30:00")
        assert result is not None
        assert result.hour == 15
        assert result.minute == 30
    
    def test_parse_datetime_objects(self):
        """Test parsing of datetime objects."""
        now = datetime.now()
        result = self.walker._parse_date_filter(now)
        assert result == now
    
    def test_parse_timestamps(self):
        """Test parsing of Unix timestamps."""
        timestamp = 1640995200  # 2022-01-01 00:00:00 UTC
        result = self.walker._parse_date_filter(timestamp)
        assert result is not None
        assert isinstance(result, datetime)
    
    def test_parse_invalid_dates(self):
        """Test parsing of invalid date strings."""
        result = self.walker._parse_date_filter("invalid date")
        assert result is None
        
        result = self.walker._parse_date_filter("2023-13-45")  # Invalid date
        assert result is None