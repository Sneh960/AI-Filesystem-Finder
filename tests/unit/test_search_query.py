"""
Unit tests for the SearchQuery data model.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
from finder.models.search_query import SearchQuery


class TestSearchQuery:
    """Test cases for SearchQuery dataclass."""
    
    def test_basic_query_creation(self):
        """Test creating a basic search query."""
        query = SearchQuery(
            text="find my documents",
            roots=["/home/user/documents"]
        )
        
        assert query.text == "find my documents"
        assert len(query.roots) == 1
        assert query.max_results == 50
        assert not query.has_seed_file()
        assert not query.has_filters()
    
    def test_query_with_filters(self):
        """Test creating a query with metadata filters."""
        query = SearchQuery(
            text="python files",
            roots=["/home/user/projects"],
            filters={
                'extensions': ['.py', '.pyx'],
                'size_max': 1000000,
                'modified_after': '2024-01-01'
            }
        )
        
        assert query.has_filters()
        assert query.get_extension_filters() == ['.py', '.pyx']
        size_min, size_max = query.get_size_range()
        assert size_min is None
        assert size_max == 1000000
    
    def test_query_with_seed_file(self):
        """Test creating a query with a seed file."""
        query = SearchQuery(
            text="similar documents",
            roots=["/home/user/docs"],
            seed_file="/home/user/example.txt"
        )
        
        assert query.has_seed_file()
        assert query.seed_file == "/home/user/example.txt"
    
    def test_extension_normalization(self):
        """Test that file extensions are properly normalized."""
        query = SearchQuery(
            text="code files",
            roots=["/home/user"],
            filters={'extension': 'py'}  # Without dot
        )
        
        extensions = query.get_extension_filters()
        assert '.py' in extensions
    
    def test_multiple_extensions(self):
        """Test handling multiple file extensions."""
        query = SearchQuery(
            text="web files",
            roots=["/home/user"],
            filters={'extensions': ['html', '.css', 'js']}
        )
        
        extensions = query.get_extension_filters()
        assert '.html' in extensions
        assert '.css' in extensions
        assert '.js' in extensions
    
    def test_size_filter_normalization(self):
        """Test size filter normalization."""
        query = SearchQuery(
            text="large files",
            roots=["/home/user"],
            filters={'size': {'min': 1000, 'max': 5000}}
        )
        
        size_min, size_max = query.get_size_range()
        assert size_min == 1000
        assert size_max == 5000
    
    def test_root_path_normalization(self):
        """Test that root paths are normalized."""
        query = SearchQuery(
            text="test",
            roots=["~/documents", "/home/user/../user/projects"]
        )
        
        # Paths should be expanded and resolved
        assert all(Path(root).is_absolute() for root in query.roots)
    
    def test_empty_query_validation(self):
        """Test that empty queries raise validation errors."""
        with pytest.raises(ValidationError):
            SearchQuery(text="", roots=["/home/user"])
        
        with pytest.raises(ValueError, match="Search query text cannot be empty"):
            SearchQuery(text="   ", roots=["/home/user"])
    
    def test_empty_roots_validation(self):
        """Test that empty roots raise validation errors."""
        with pytest.raises(ValidationError):
            SearchQuery(text="test", roots=[])
        
        with pytest.raises(ValueError, match="No valid root directories provided"):
            SearchQuery(text="test", roots=["", "   "])
    
    def test_invalid_filter_keys(self):
        """Test that invalid filter keys raise validation errors."""
        with pytest.raises(ValueError, match="Unknown filter key"):
            SearchQuery(
                text="test",
                roots=["/home/user"],
                filters={'invalid_key': 'value'}
            )
    
    def test_to_dict_conversion(self):
        """Test converting query to dictionary."""
        query = SearchQuery(
            text="test query",
            roots=["/home/user"],
            filters={'extensions': ['.txt']},
            seed_file="/path/to/seed.txt",
            max_results=100
        )
        
        data = query.to_dict()
        assert data['text'] == "test query"
        assert data['roots'] == ["/home/user"]
        assert data['filters'] == {'extensions': ['.txt']}
        assert data['seed_file'] == "/path/to/seed.txt"
        assert data['max_results'] == 100
    
    def test_from_dict_creation(self):
        """Test creating query from dictionary."""
        data = {
            'text': 'test query',
            'roots': ['/home/user'],
            'filters': {'extensions': ['.py']},
            'seed_file': '/path/to/seed.py',
            'max_results': 25
        }
        
        query = SearchQuery.from_dict(data)
        assert query.text == 'test query'
        assert query.roots == ['/home/user']
        assert query.filters == {'extensions': ['.py']}
        assert query.seed_file == '/path/to/seed.py'
        assert query.max_results == 25
    
    def test_string_representation(self):
        """Test string representation of query."""
        query = SearchQuery(
            text="find documents",
            roots=["/home/user/docs", "/home/user/projects"],
            filters={'extensions': ['.txt', '.md']},
            seed_file="/path/to/example.txt"
        )
        
        str_repr = str(query)
        assert "find documents" in str_repr
        assert "2 directories" in str_repr
        assert "Filters:" in str_repr
        assert "Seed file:" in str_repr
    
    def test_date_filter_normalization(self):
        """Test date filter normalization with different formats."""
        query = SearchQuery(
            text="recent files",
            roots=["/home/user"],
            filters={
                'modified': {'after': '2024-01-01', 'before': '2024-12-31'},
                'created': '2024-06-01'  # Single value
            }
        )
        
        assert 'modified_after' in query.filters
        assert 'modified_before' in query.filters
        assert 'created_after' in query.filters
        assert query.filters['modified_after'] == '2024-01-01'
        assert query.filters['modified_before'] == '2024-12-31'
        assert query.filters['created_after'] == '2024-06-01'
    
    def test_size_filter_single_value(self):
        """Test size filter with single value (treated as max)."""
        query = SearchQuery(
            text="small files",
            roots=["/home/user"],
            filters={'size': 1000000}
        )
        
        size_min, size_max = query.get_size_range()
        assert size_min is None
        assert size_max == 1000000