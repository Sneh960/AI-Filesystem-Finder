"""
Unit tests for search results data models.

Tests the FileMatch, SearchResults, TextSnippet, and FileMetadata classes
to ensure proper validation, functionality, and data integrity.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from finder.models.search_results import (
    FileMatch, SearchResults, TextSnippet, FileMetadata, MatchType
)
from finder.models.search_query import SearchQuery


class TestFileMetadata:
    """Test cases for FileMetadata class."""
    
    def test_basic_creation(self):
        """Test basic metadata creation with required fields."""
        now = datetime.now()
        metadata = FileMetadata(
            size=1024,
            modified_time=now
        )
        
        assert metadata.size == 1024
        assert metadata.modified_time == now
        assert metadata.created_time is None
        assert metadata.owner is None
        assert metadata.extension is None
        assert metadata.is_binary is False
    
    def test_full_creation(self):
        """Test metadata creation with all fields."""
        now = datetime.now()
        created = now - timedelta(days=1)
        
        metadata = FileMetadata(
            size=2048,
            modified_time=now,
            created_time=created,
            owner="testuser",
            extension="py",
            mime_type="text/x-python",
            permissions="644",
            is_binary=False
        )
        
        assert metadata.size == 2048
        assert metadata.extension == ".py"  # Should be normalized
        assert metadata.owner == "testuser"
        assert metadata.mime_type == "text/x-python"
        assert metadata.permissions == "644"
    
    def test_extension_normalization(self):
        """Test that file extensions are normalized with leading dot."""
        metadata = FileMetadata(size=100, modified_time=datetime.now(), extension="txt")
        assert metadata.extension == ".txt"
        
        metadata2 = FileMetadata(size=100, modified_time=datetime.now(), extension=".py")
        assert metadata2.extension == ".py"
        
        metadata3 = FileMetadata(size=100, modified_time=datetime.now(), extension="JS")
        assert metadata3.extension == ".js"
    
    def test_negative_size_validation(self):
        """Test that negative file sizes are rejected."""
        with pytest.raises(ValidationError):
            FileMetadata(size=-1, modified_time=datetime.now())
    
    def test_human_readable_size(self):
        """Test human-readable size formatting."""
        metadata = FileMetadata(size=1024, modified_time=datetime.now())
        assert "1.0 KB" in metadata.get_size_human_readable()
        
        metadata2 = FileMetadata(size=1048576, modified_time=datetime.now())  # 1MB
        assert "1.0 MB" in metadata2.get_size_human_readable()
    
    def test_is_recent(self):
        """Test recent file detection."""
        now = datetime.now()
        recent = now - timedelta(days=3)
        old = now - timedelta(days=10)
        
        recent_metadata = FileMetadata(size=100, modified_time=recent)
        old_metadata = FileMetadata(size=100, modified_time=old)
        
        assert recent_metadata.is_recent(days=7) is True
        assert old_metadata.is_recent(days=7) is False
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        now = datetime.now()
        metadata = FileMetadata(
            size=1024,
            modified_time=now,
            extension="py",
            owner="testuser"
        )
        
        result = metadata.to_dict()
        
        assert result['size'] == 1024
        assert result['extension'] == ".py"
        assert result['owner'] == "testuser"
        assert 'size_human' in result
        assert 'modified_time' in result


class TestTextSnippet:
    """Test cases for TextSnippet class."""
    
    def test_basic_creation(self):
        """Test basic snippet creation."""
        snippet = TextSnippet(
            content="def hello():\n    print('world')",
            start_line=10,
            end_line=11
        )
        
        assert snippet.content == "def hello():\n    print('world')"
        assert snippet.start_line == 10
        assert snippet.end_line == 11
        assert snippet.get_line_count() == 2
    
    def test_with_match_highlighting(self):
        """Test snippet with match position highlighting."""
        content = "This is a test line with important content"
        snippet = TextSnippet(
            content=content,
            start_line=5,
            end_line=5,
            match_line=5,
            match_start=25,  # "important" starts at position 25
            match_end=34     # "important" ends at position 34
        )
        
        highlighted = snippet.get_highlighted_content()
        assert "**important**" in highlighted
        assert highlighted == "This is a test line with **important** content"
    
    def test_line_validation(self):
        """Test line number validation."""
        with pytest.raises(ValidationError):
            TextSnippet(content="test", start_line=0, end_line=1)
        
        with pytest.raises(ValueError, match="End line must be >= start line"):
            TextSnippet(content="test", start_line=5, end_line=3)
    
    def test_match_position_validation(self):
        """Test match position validation."""
        content = "short"
        
        with pytest.raises(ValueError, match="Invalid match position"):
            TextSnippet(
                content=content,
                start_line=1,
                end_line=1,
                match_start=5,
                match_end=3
            )
        
        with pytest.raises(ValueError, match="Match end position exceeds content length"):
            TextSnippet(
                content=content,
                start_line=1,
                end_line=1,
                match_start=0,
                match_end=10
            )
    
    def test_match_line_validation(self):
        """Test match line validation."""
        with pytest.raises(ValueError, match="Match line must be within snippet range"):
            TextSnippet(
                content="test",
                start_line=5,
                end_line=7,
                match_line=10
            )
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        snippet = TextSnippet(
            content="test content",
            start_line=1,
            end_line=1,
            match_start=5,
            match_end=12
        )
        
        result = snippet.to_dict()
        
        assert result['content'] == "test content"
        assert result['start_line'] == 1
        assert result['end_line'] == 1
        assert result['line_count'] == 1
        assert 'highlighted_content' in result


class TestFileMatch:
    """Test cases for FileMatch class."""
    
    def test_basic_creation(self):
        """Test basic file match creation."""
        match = FileMatch(
            path="/home/user/test.py",
            score=0.85,
            match_type=MatchType.SEMANTIC
        )
        
        assert match.score == 0.85
        assert match.match_type == MatchType.SEMANTIC
        assert match.get_filename() == "test.py"
        assert "/home/user" in match.get_directory()
        assert match.get_extension() == ".py"
    
    def test_string_match_type_conversion(self):
        """Test automatic conversion of string match types to enum."""
        match = FileMatch(
            path="/test.txt",
            score=0.5,
            match_type="literal"  # String instead of enum
        )
        
        assert match.match_type == MatchType.LITERAL
    
    def test_invalid_match_type(self):
        """Test rejection of invalid match types."""
        with pytest.raises(ValueError, match="Invalid match type"):
            FileMatch(
                path="/test.txt",
                score=0.5,
                match_type="invalid_type"
            )
    
    def test_score_validation(self):
        """Test score validation."""
        with pytest.raises(ValidationError):
            FileMatch(path="/test.txt", score=1.5, match_type=MatchType.LITERAL)
        
        with pytest.raises(ValidationError):
            FileMatch(path="/test.txt", score=-0.1, match_type=MatchType.LITERAL)
    
    def test_empty_path_validation(self):
        """Test empty path validation."""
        with pytest.raises(ValidationError):
            FileMatch(path="", score=0.5, match_type=MatchType.LITERAL)
        
        with pytest.raises(ValidationError):
            FileMatch(path="   ", score=0.5, match_type=MatchType.LITERAL)
    
    def test_snippet_management(self):
        """Test snippet addition and management."""
        match = FileMatch(
            path="/test.py",
            score=0.7,
            match_type=MatchType.CODE
        )
        
        assert match.has_snippets() is False
        assert match.get_snippet_count() == 0
        assert match.get_primary_snippet() is None
        
        snippet = TextSnippet(content="def test():", start_line=1, end_line=1)
        match.add_snippet(snippet)
        
        assert match.has_snippets() is True
        assert match.get_snippet_count() == 1
        assert match.get_primary_snippet() == snippet
    
    def test_with_metadata(self):
        """Test file match with metadata."""
        metadata = FileMetadata(size=1024, modified_time=datetime.now())
        match = FileMatch(
            path="/test.py",
            score=0.8,
            match_type=MatchType.LITERAL,
            metadata=metadata
        )
        
        assert match.metadata == metadata
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        match = FileMatch(
            path="/home/user/test.py",
            score=0.75,
            match_type=MatchType.SEMANTIC,
            rank=1
        )
        
        result = match.to_dict()
        
        assert result['path'] == match.path
        assert result['filename'] == "test.py"
        assert result['score'] == 0.75
        assert result['match_type'] == "semantic"
        assert result['rank'] == 1
        assert result['snippet_count'] == 0
    
    def test_string_representation(self):
        """Test string representation."""
        match = FileMatch(
            path="/test.py",
            score=0.85,
            match_type=MatchType.CODE
        )
        
        str_repr = str(match)
        assert "test.py" in str_repr
        assert "0.850" in str_repr
        assert "code" in str_repr


class TestSearchResults:
    """Test cases for SearchResults class."""
    
    def create_sample_query(self):
        """Create a sample search query for testing."""
        return SearchQuery(
            text="test query",
            roots=["/home/user"]
        )
    
    def create_sample_match(self, path="/test.py", score=0.8):
        """Create a sample file match for testing."""
        return FileMatch(
            path=path,
            score=score,
            match_type=MatchType.SEMANTIC
        )
    
    def test_basic_creation(self):
        """Test basic search results creation."""
        query = self.create_sample_query()
        results = SearchResults(query=query)
        
        assert results.query == query
        assert results.get_match_count() == 0
        assert results.total_scanned == 0
        assert results.execution_time == 0.0
        assert results.has_errors() is False
    
    def test_with_matches(self):
        """Test search results with file matches."""
        query = self.create_sample_query()
        match1 = self.create_sample_match("/test1.py", 0.9)
        match2 = self.create_sample_match("/test2.py", 0.7)
        
        results = SearchResults(
            query=query,
            matches=[match1, match2],
            total_scanned=100,
            execution_time=1.5
        )
        
        assert results.get_match_count() == 2
        assert results.total_scanned == 100
        assert results.execution_time == 1.5
        
        # Check that ranks were assigned
        assert match1.rank == 1
        assert match2.rank == 2
    
    def test_negative_validation(self):
        """Test validation of negative values."""
        query = self.create_sample_query()
        
        with pytest.raises(ValidationError):
            SearchResults(query=query, total_scanned=-1)
        
        with pytest.raises(ValidationError):
            SearchResults(query=query, execution_time=-1.0)
    
    def test_top_matches(self):
        """Test getting top matches by score."""
        query = self.create_sample_query()
        matches = [
            self.create_sample_match("/low.py", 0.3),
            self.create_sample_match("/high.py", 0.9),
            self.create_sample_match("/medium.py", 0.6)
        ]
        
        results = SearchResults(query=query, matches=matches)
        top_matches = results.get_top_matches(2)
        
        assert len(top_matches) == 2
        assert top_matches[0].score == 0.9
        assert top_matches[1].score == 0.6
    
    def test_matches_by_type(self):
        """Test filtering matches by type."""
        query = self.create_sample_query()
        literal_match = FileMatch(path="/lit.py", score=0.8, match_type=MatchType.LITERAL)
        semantic_match = FileMatch(path="/sem.py", score=0.7, match_type=MatchType.SEMANTIC)
        code_match = FileMatch(path="/code.py", score=0.6, match_type=MatchType.CODE)
        
        results = SearchResults(
            query=query,
            matches=[literal_match, semantic_match, code_match]
        )
        
        semantic_matches = results.get_matches_by_type(MatchType.SEMANTIC)
        assert len(semantic_matches) == 1
        assert semantic_matches[0] == semantic_match
    
    def test_average_score(self):
        """Test average score calculation."""
        query = self.create_sample_query()
        matches = [
            self.create_sample_match("/test1.py", 0.8),
            self.create_sample_match("/test2.py", 0.6),
            self.create_sample_match("/test3.py", 1.0)
        ]
        
        results = SearchResults(query=query, matches=matches)
        avg_score = results.get_average_score()
        
        assert abs(avg_score - 0.8) < 0.001  # (0.8 + 0.6 + 1.0) / 3
    
    def test_score_distribution(self):
        """Test score distribution calculation."""
        query = self.create_sample_query()
        matches = [
            self.create_sample_match("/excellent.py", 0.9),  # excellent
            self.create_sample_match("/good.py", 0.7),       # good
            self.create_sample_match("/fair.py", 0.5),       # fair
            self.create_sample_match("/poor.py", 0.2)        # poor
        ]
        
        results = SearchResults(query=query, matches=matches)
        distribution = results.get_score_distribution()
        
        assert distribution['excellent'] == 1
        assert distribution['good'] == 1
        assert distribution['fair'] == 1
        assert distribution['poor'] == 1
    
    def test_error_management(self):
        """Test error addition and detection."""
        query = self.create_sample_query()
        results = SearchResults(query=query)
        
        assert results.has_errors() is False
        
        results.add_error("Test error message")
        assert results.has_errors() is True
        assert len(results.errors) == 1
        assert "Test error message" in results.errors
    
    def test_match_management(self):
        """Test adding matches and rank reassignment."""
        query = self.create_sample_query()
        results = SearchResults(query=query)
        
        match1 = self.create_sample_match("/test1.py", 0.8)
        match2 = self.create_sample_match("/test2.py", 0.6)
        
        results.add_match(match1)
        assert results.get_match_count() == 1
        assert match1.rank == 1
        
        results.add_match(match2)
        assert results.get_match_count() == 2
        assert match2.rank == 2
    
    def test_sorting(self):
        """Test different sorting methods."""
        query = self.create_sample_query()
        match1 = self.create_sample_match("/zebra.py", 0.6)
        match2 = self.create_sample_match("/alpha.py", 0.9)
        
        results = SearchResults(query=query, matches=[match1, match2])
        
        # Test sort by score (default descending)
        results.sort_by_score()
        assert results.matches[0].score == 0.9
        assert results.matches[0].rank == 1
        
        # Test sort by path
        results.sort_by_path()
        assert "alpha.py" in results.matches[0].path
        assert results.matches[0].rank == 1
    
    def test_filtering_and_limiting(self):
        """Test result filtering and limiting."""
        query = self.create_sample_query()
        matches = [
            self.create_sample_match("/high.py", 0.9),
            self.create_sample_match("/medium.py", 0.6),
            self.create_sample_match("/low.py", 0.3)
        ]
        
        results = SearchResults(query=query, matches=matches)
        
        # Test score filtering
        results.filter_by_score(0.5)
        assert results.get_match_count() == 2  # Only high and medium remain
        
        # Test result limiting
        results.limit_results(1)
        assert results.get_match_count() == 1
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        query = self.create_sample_query()
        match = self.create_sample_match("/test.py", 0.8)
        
        results = SearchResults(
            query=query,
            matches=[match],
            total_scanned=50,
            execution_time=2.0
        )
        
        result_dict = results.to_dict()
        
        assert 'query' in result_dict
        assert 'matches' in result_dict
        assert result_dict['match_count'] == 1
        assert result_dict['total_scanned'] == 50
        assert result_dict['execution_time'] == 2.0
        assert 'timestamp' in result_dict
        assert 'average_score' in result_dict
        assert 'score_distribution' in result_dict
    
    def test_string_representation(self):
        """Test string representation."""
        query = self.create_sample_query()
        match = self.create_sample_match("/test.py", 0.8)
        
        results = SearchResults(
            query=query,
            matches=[match],
            total_scanned=100,
            execution_time=1.5
        )
        
        str_repr = str(results)
        assert "Found 1 matches" in str_repr
        assert "Scanned 100 files" in str_repr
        assert "1.50s" in str_repr


if __name__ == "__main__":
    pytest.main([__file__])