"""
Search query data models for the AI Filesystem Finder.

This module defines the core data structures for representing search queries,
including natural language text, search roots, filters, and configuration options.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class SearchQuery(BaseModel):
    """
    Represents a search query with all parameters and constraints.
    
    This Pydantic model encapsulates all information needed to execute a search,
    including the natural language query, target directories, metadata filters,
    and optional seed files for similarity search.
    
    Attributes:
        text: Natural language search query from the user
        roots: List of root directories to search within
        filters: Dictionary of metadata filters (extension, size, date, owner)
        seed_file: Optional path to a file for similarity-based search
        max_results: Maximum number of results to return (default: 50)
    """
    
    text: str = Field(..., min_length=1, description="Natural language search query")
    roots: List[str] = Field(..., min_length=1, description="List of root directories to search")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    seed_file: Optional[str] = Field(None, description="Optional seed file for similarity search")
    max_results: int = Field(50, gt=0, le=1000, description="Maximum number of results")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and normalize the search query text."""
        if not v or not v.strip():
            raise ValueError("Search query text cannot be empty")
        return v.strip()
    
    @field_validator('roots')
    @classmethod
    def validate_roots(cls, v: List[str]) -> List[str]:
        """Validate and normalize root directory paths."""
        if not v:
            raise ValueError("At least one root directory must be specified")
        
        normalized_roots = []
        for root in v:
            if not root or not root.strip():
                continue
            
            # Convert to Path for normalization
            root_path = Path(root).expanduser().resolve()
            normalized_roots.append(str(root_path))
        
        if not normalized_roots:
            raise ValueError("No valid root directories provided")
        
        return normalized_roots
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize filter parameters."""
        if not v:
            return {}
        
        # Validate known filter types
        valid_filter_keys = {
            'extension', 'extensions', 'ext',
            'size_min', 'size_max', 'size',
            'modified_after', 'modified_before', 'modified',
            'created_after', 'created_before', 'created',
            'owner', 'type', 'mime_type'
        }
        
        for key in v:
            if key not in valid_filter_keys:
                raise ValueError(f"Unknown filter key: {key}")
        
        return cls._normalize_filters(v)
    
    @classmethod
    def _normalize_filters(cls, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filter parameters."""
        normalized = filters.copy()
        
        # Normalize extension filters
        cls._normalize_extension_filters(normalized)
        cls._normalize_size_filters(normalized)
        cls._normalize_date_filters(normalized)
        
        return normalized
    
    @classmethod
    def _normalize_extension_filters(cls, filters: Dict[str, Any]) -> None:
        """Normalize file extension filters."""
        ext_keys = ['extension', 'extensions', 'ext']
        extensions = []
        
        for key in ext_keys:
            if key in filters:
                ext_value = filters.pop(key)
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
            
            filters['extensions'] = normalized_exts
    
    @classmethod
    def _normalize_size_filters(cls, filters: Dict[str, Any]) -> None:
        """Normalize file size filters."""
        if 'size' in filters:
            size_value = filters.pop('size')
            if isinstance(size_value, dict):
                if 'min' in size_value:
                    filters['size_min'] = size_value['min']
                if 'max' in size_value:
                    filters['size_max'] = size_value['max']
            elif isinstance(size_value, (int, str)):
                # Assume it's a maximum size
                filters['size_max'] = size_value
    
    @classmethod
    def _normalize_date_filters(cls, filters: Dict[str, Any]) -> None:
        """Normalize date filters, converting relative dates to absolute."""
        date_keys = ['modified', 'created']
        
        for base_key in date_keys:
            if base_key in filters:
                date_value = filters.pop(base_key)
                if isinstance(date_value, dict):
                    if 'after' in date_value:
                        filters[f'{base_key}_after'] = date_value['after']
                    if 'before' in date_value:
                        filters[f'{base_key}_before'] = date_value['before']
                else:
                    # Assume it's an "after" filter
                    filters[f'{base_key}_after'] = date_value
    
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
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchQuery':
        """Create a SearchQuery instance from a dictionary."""
        return cls.model_validate(data)
    
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