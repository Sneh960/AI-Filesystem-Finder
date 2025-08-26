"""
Search query data models for the AI Filesystem Finder.

This module defines the core data structures for representing search queries,
including natural language text, search roots, filters, and configuration options.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path


@dataclass
class SearchQuery:
    """
    Represents a search query with all parameters and constraints.
    
    This dataclass encapsulates all information needed to execute a search,
    including the natural language query, target directories, metadata filters,
    and optional seed files for similarity search.
    
    Attributes:
        text: Natural language search query from the user
        roots: List of root directories to search within
        filters: Dictionary of metadata filters (extension, size, date, owner)
        seed_file: Optional path to a file for similarity-based search
        max_results: Maximum number of results to return (default: 50)
    """
    
    text: str
    roots: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    seed_file: Optional[str] = None
    max_results: int = 50
    
    def __post_init__(self):
        """Validate and normalize the search query after initialization."""
        self._validate_query()
        self._normalize_roots()
        self._validate_filters()
    
    def _validate_query(self) -> None:
        """Validate the search query text."""
        if not self.text or not self.text.strip():
            raise ValueError("Search query text cannot be empty")
        
        # Normalize whitespace
        self.text = self.text.strip()
    
    def _normalize_roots(self) -> None:
        """Normalize and validate root directory paths."""
        if not self.roots:
            raise ValueError("At least one root directory must be specified")
        
        normalized_roots = []
        for root in self.roots:
            if not root or not root.strip():
                continue
            
            # Convert to Path for normalization
            root_path = Path(root).expanduser().resolve()
            normalized_roots.append(str(root_path))
        
        if not normalized_roots:
            raise ValueError("No valid root directories provided")
        
        self.roots = normalized_roots
    
    def _validate_filters(self) -> None:
        """Validate and normalize filter parameters."""
        if not self.filters:
            return
        
        # Validate known filter types
        valid_filter_keys = {
            'extension', 'extensions', 'ext',
            'size_min', 'size_max', 'size',
            'modified_after', 'modified_before', 'modified',
            'created_after', 'created_before', 'created',
            'owner', 'type', 'mime_type'
        }
        
        for key in self.filters:
            if key not in valid_filter_keys:
                raise ValueError(f"Unknown filter key: {key}")
        
        # Normalize extension filters
        self._normalize_extension_filters()
        self._normalize_size_filters()
        self._normalize_date_filters()
    
    def _normalize_extension_filters(self) -> None:
        """Normalize file extension filters."""
        ext_keys = ['extension', 'extensions', 'ext']
        extensions = []
        
        for key in ext_keys:
            if key in self.filters:
                ext_value = self.filters.pop(key)
                if isinstance(ext_value, str):
                    extensions.append(ext_value)
                elif isinstance(ext_value, list):
                    extensions.extend(ext_value)
        
        if extensions:
            # Normalize extensions (ensure they start with .)
            normalized_exts = []
            for ext in extensions:
                if not ext.startswith('.'):
                    ext = '.' + ext
                normalized_exts.append(ext.lower())
            
            self.filters['extensions'] = normalized_exts
    
    def _normalize_size_filters(self) -> None:
        """Normalize file size filters."""
        if 'size' in self.filters:
            size_value = self.filters.pop('size')
            if isinstance(size_value, dict):
                if 'min' in size_value:
                    self.filters['size_min'] = size_value['min']
                if 'max' in size_value:
                    self.filters['size_max'] = size_value['max']
            elif isinstance(size_value, (int, str)):
                # Assume it's a maximum size
                self.filters['size_max'] = size_value
    
    def _normalize_date_filters(self) -> None:
        """Normalize date filters, converting relative dates to absolute."""
        date_keys = ['modified', 'created']
        
        for base_key in date_keys:
            if base_key in self.filters:
                date_value = self.filters.pop(base_key)
                if isinstance(date_value, dict):
                    if 'after' in date_value:
                        self.filters[f'{base_key}_after'] = date_value['after']
                    if 'before' in date_value:
                        self.filters[f'{base_key}_before'] = date_value['before']
                else:
                    # Assume it's an "after" filter
                    self.filters[f'{base_key}_after'] = date_value
    
    def has_seed_file(self) -> bool:
        """Check if this query uses a seed file for similarity search."""
        return self.seed_file is not None and self.seed_file.strip() != ""
    
    def has_filters(self) -> bool:
        """Check if this query has any metadata filters applied."""
        return bool(self.filters)
    
    def get_extension_filters(self) -> List[str]:
        """Get normalized file extension filters."""
        return self.filters.get('extensions', [])
    
    def get_size_range(self) -> tuple[Optional[int], Optional[int]]:
        """Get size filter range as (min_size, max_size)."""
        return (
            self.filters.get('size_min'),
            self.filters.get('size_max')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the search query to a dictionary representation."""
        return {
            'text': self.text,
            'roots': self.roots,
            'filters': self.filters,
            'seed_file': self.seed_file,
            'max_results': self.max_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchQuery':
        """Create a SearchQuery instance from a dictionary."""
        return cls(
            text=data['text'],
            roots=data['roots'],
            filters=data.get('filters', {}),
            seed_file=data.get('seed_file'),
            max_results=data.get('max_results', 50)
        )
    
    def __str__(self) -> str:
        """String representation of the search query."""
        parts = [f"Query: '{self.text}'"]
        parts.append(f"Roots: {len(self.roots)} directories")
        
        if self.has_filters():
            parts.append(f"Filters: {len(self.filters)} applied")
        
        if self.has_seed_file():
            parts.append(f"Seed file: {Path(self.seed_file).name}")
        
        parts.append(f"Max results: {self.max_results}")
        
        return " | ".join(parts)