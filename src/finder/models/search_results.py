"""
Search results data models for the AI Filesystem Finder.

This module defines the core data structures for representing search results,
including file matches, metadata, text snippets, and complete search result sets.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class MatchType(Enum):
    """Enumeration of different search match types."""
    LITERAL = "literal"
    SEMANTIC = "semantic" 
    CODE = "code"


class FileMetadata(BaseModel):
    """
    Metadata information about a matched file.
    
    Contains filesystem metadata that can be used for filtering,
    ranking, and display purposes.
    
    Attributes:
        size: File size in bytes
        modified_time: Last modification timestamp
        created_time: File creation timestamp (if available)
        owner: File owner/user (if available)
        extension: File extension (e.g., '.py', '.txt')
        mime_type: MIME type of the file (if detected)
        permissions: File permissions as octal string (e.g., '644')
        is_binary: Whether the file appears to be binary
    """
    
    size: int = Field(..., ge=0, description="File size in bytes")
    modified_time: datetime = Field(..., description="Last modification timestamp")
    created_time: Optional[datetime] = Field(None, description="File creation timestamp")
    owner: Optional[str] = Field(None, description="File owner/user")
    extension: Optional[str] = Field(None, description="File extension")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    permissions: Optional[str] = Field(None, description="File permissions as octal string")
    is_binary: bool = Field(False, description="Whether the file appears to be binary")
    
    @field_validator('extension')
    @classmethod
    def validate_extension(cls, v: Optional[str]) -> Optional[str]:
        """Normalize extension to include leading dot."""
        if v is None:
            return v
        if not v.startswith('.'):
            return '.' + v.lower()
        return v.lower()
    
    def get_size_human_readable(self) -> str:
        """Get file size in human-readable format."""
        size = float(self.size)  # Work with a copy to avoid modifying original
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def is_recent(self, days: int = 7) -> bool:
        """Check if file was modified within the specified number of days."""
        if not self.modified_time:
            return False
        
        now = datetime.now()
        delta = now - self.modified_time
        return delta.days <= days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        data = self.model_dump()
        data['size_human'] = self.get_size_human_readable()
        if data['modified_time']:
            data['modified_time'] = self.modified_time.isoformat()
        if data['created_time']:
            data['created_time'] = self.created_time.isoformat()
        return data


class TextSnippet(BaseModel):
    """
    A text snippet showing context around a search match.
    
    Represents a portion of file content that contains or surrounds
    a search match, with line numbers and highlighting information.
    
    Attributes:
        content: The actual text content of the snippet
        start_line: Line number where the snippet starts (1-based)
        end_line: Line number where the snippet ends (1-based)
        match_line: Line number of the actual match within the snippet
        match_start: Character position where match starts in the content
        match_end: Character position where match ends in the content
        context_before: Number of lines included before the match
        context_after: Number of lines included after the match
    """
    
    content: str = Field(..., description="The actual text content of the snippet")
    start_line: int = Field(..., ge=1, description="Line number where the snippet starts")
    end_line: int = Field(..., ge=1, description="Line number where the snippet ends")
    match_line: Optional[int] = Field(None, ge=1, description="Line number of the actual match")
    match_start: Optional[int] = Field(None, ge=0, description="Character position where match starts")
    match_end: Optional[int] = Field(None, ge=0, description="Character position where match ends")
    context_before: int = Field(2, ge=0, description="Number of lines included before the match")
    context_after: int = Field(2, ge=0, description="Number of lines included after the match")
    
    @model_validator(mode='after')
    def validate_snippet(self):
        """Validate snippet data consistency."""
        if self.end_line < self.start_line:
            raise ValueError("End line must be >= start line")
        
        if self.match_line is not None:
            if self.match_line < self.start_line or self.match_line > self.end_line:
                raise ValueError("Match line must be within snippet range")
        
        if self.match_start is not None and self.match_end is not None:
            if self.match_end < self.match_start:
                raise ValueError("Invalid match position")
            if self.match_end > len(self.content):
                raise ValueError("Match end position exceeds content length")
        
        return self
    
    def get_highlighted_content(self, highlight_start: str = "**", highlight_end: str = "**") -> str:
        """Get content with match highlighted using specified markers."""
        if self.match_start is None or self.match_end is None:
            return self.content
        
        before = self.content[:self.match_start]
        match = self.content[self.match_start:self.match_end]
        after = self.content[self.match_end:]
        
        return f"{before}{highlight_start}{match}{highlight_end}{after}"
    
    def get_line_count(self) -> int:
        """Get the number of lines in this snippet."""
        return self.end_line - self.start_line + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snippet to dictionary representation."""
        data = self.model_dump()
        data['highlighted_content'] = self.get_highlighted_content()
        data['line_count'] = self.get_line_count()
        return data


class FileMatch(BaseModel):
    """
    Represents a single file that matches the search criteria.
    
    Contains the file path, relevance score, match type, contextual snippets,
    and metadata about the matched file.
    
    Attributes:
        path: Absolute path to the matched file
        score: Relevance score (0.0 to 1.0, higher is more relevant)
        match_type: Type of search that found this match
        snippets: List of text snippets showing match context
        metadata: File metadata (size, dates, permissions, etc.)
        rank: Optional rank in the result set (1-based)
    """
    
    path: str = Field(..., min_length=1, description="Absolute path to the matched file")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_type: MatchType = Field(..., description="Type of search that found this match")
    snippets: List[TextSnippet] = Field(default_factory=list, description="Text snippets showing match context")
    metadata: Optional[FileMetadata] = Field(None, description="File metadata")
    rank: Optional[int] = Field(None, ge=1, description="Rank in the result set")
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate and normalize file path."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        # Normalize path
        return str(Path(v).resolve())
    
    @field_validator('match_type', mode='before')
    @classmethod
    def validate_match_type(cls, v) -> MatchType:
        """Ensure match_type is MatchType enum."""
        if isinstance(v, str):
            try:
                return MatchType(v)
            except ValueError:
                raise ValueError(f"Invalid match type: {v}")
        return v
    
    def get_filename(self) -> str:
        """Get just the filename without directory path."""
        return Path(self.path).name
    
    def get_directory(self) -> str:
        """Get the directory containing this file."""
        return str(Path(self.path).parent)
    
    def get_extension(self) -> Optional[str]:
        """Get the file extension."""
        return Path(self.path).suffix.lower() if Path(self.path).suffix else None
    
    def has_snippets(self) -> bool:
        """Check if this match has any text snippets."""
        return len(self.snippets) > 0
    
    def get_snippet_count(self) -> int:
        """Get the number of snippets for this match."""
        return len(self.snippets)
    
    def get_primary_snippet(self) -> Optional[TextSnippet]:
        """Get the first/primary snippet if available."""
        return self.snippets[0] if self.snippets else None
    
    def add_snippet(self, snippet: TextSnippet) -> None:
        """Add a text snippet to this match."""
        self.snippets.append(snippet)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file match to dictionary representation."""
        data = self.model_dump()
        data['filename'] = self.get_filename()
        data['directory'] = self.get_directory()
        data['extension'] = self.get_extension()
        data['match_type'] = self.match_type.value
        data['snippets'] = [snippet.to_dict() for snippet in self.snippets]
        data['snippet_count'] = self.get_snippet_count()
        if self.metadata:
            data['metadata'] = self.metadata.to_dict()
        return data
    
    def __str__(self) -> str:
        """String representation of the file match."""
        parts = [f"{self.get_filename()} (score: {self.score:.3f})"]
        parts.append(f"Type: {self.match_type.value}")
        
        if self.has_snippets():
            parts.append(f"Snippets: {self.get_snippet_count()}")
        
        if self.metadata:
            parts.append(f"Size: {self.metadata.get_size_human_readable()}")
        
        return " | ".join(parts)


class SearchResults(BaseModel):
    """
    Complete results from a search operation.
    
    Contains all matched files, execution metadata, and the original query
    that produced these results.
    
    Attributes:
        query: The original search query that produced these results
        matches: List of file matches, typically sorted by relevance
        total_scanned: Total number of files examined during search
        execution_time: Time taken to execute the search in seconds
        search_plan: The search plan that was executed (if available)
        timestamp: When the search was executed
        errors: Any errors encountered during search
    """
    
    query: 'SearchQuery' = Field(..., description="The original search query")
    matches: List[FileMatch] = Field(default_factory=list, description="List of file matches")
    total_scanned: int = Field(0, ge=0, description="Total number of files examined")
    execution_time: float = Field(0.0, ge=0.0, description="Time taken to execute the search")
    search_plan: Optional[Any] = Field(None, description="The search plan that was executed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the search was executed")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during search")
    
    def model_post_init(self, __context) -> None:
        """Process search results after initialization."""
        # Assign ranks to matches if not already set
        self._assign_ranks()
    
    def _assign_ranks(self) -> None:
        """Assign rank numbers to matches based on their order."""
        for i, match in enumerate(self.matches, 1):
            match.rank = i
    
    def get_match_count(self) -> int:
        """Get the total number of matches."""
        return len(self.matches)
    
    def get_top_matches(self, n: int = 10) -> List[FileMatch]:
        """Get the top N matches by score."""
        return sorted(self.matches, key=lambda m: m.score, reverse=True)[:n]
    
    def get_matches_by_type(self, match_type: MatchType) -> List[FileMatch]:
        """Get all matches of a specific type."""
        return [match for match in self.matches if match.match_type == match_type]
    
    def get_average_score(self) -> float:
        """Get the average relevance score across all matches."""
        if not self.matches:
            return 0.0
        return sum(match.score for match in self.matches) / len(self.matches)
    
    def get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of scores in ranges."""
        if not self.matches:
            return {}
        
        ranges = {
            'excellent': 0,  # 0.8-1.0
            'good': 0,       # 0.6-0.8
            'fair': 0,       # 0.4-0.6
            'poor': 0        # 0.0-0.4
        }
        
        for match in self.matches:
            if match.score >= 0.8:
                ranges['excellent'] += 1
            elif match.score >= 0.6:
                ranges['good'] += 1
            elif match.score >= 0.4:
                ranges['fair'] += 1
            else:
                ranges['poor'] += 1
        
        return ranges
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during search."""
        return len(self.errors) > 0
    
    def add_error(self, error: str) -> None:
        """Add an error message to the results."""
        self.errors.append(error)
    
    def add_match(self, match: FileMatch) -> None:
        """Add a file match to the results."""
        self.matches.append(match)
        # Re-assign ranks after adding
        self._assign_ranks()
    
    def sort_by_score(self, reverse: bool = True) -> None:
        """Sort matches by relevance score."""
        self.matches.sort(key=lambda m: m.score, reverse=reverse)
        self._assign_ranks()
    
    def sort_by_path(self) -> None:
        """Sort matches alphabetically by file path."""
        self.matches.sort(key=lambda m: m.path)
        self._assign_ranks()
    
    def filter_by_score(self, min_score: float) -> None:
        """Remove matches below the specified score threshold."""
        self.matches = [match for match in self.matches if match.score >= min_score]
        self._assign_ranks()
    
    def limit_results(self, max_results: int) -> None:
        """Limit the number of results to the specified maximum."""
        if max_results > 0:
            self.matches = self.matches[:max_results]
            self._assign_ranks()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search results to dictionary representation."""
        data = self.model_dump()
        data['query'] = self.query.to_dict()
        data['matches'] = [match.to_dict() for match in self.matches]
        data['match_count'] = self.get_match_count()
        data['timestamp'] = self.timestamp.isoformat()
        data['average_score'] = self.get_average_score()
        data['score_distribution'] = self.get_score_distribution()
        data['has_errors'] = self.has_errors()
        return data
    
    def __str__(self) -> str:
        """String representation of search results."""
        parts = [f"Found {self.get_match_count()} matches"]
        parts.append(f"Scanned {self.total_scanned} files")
        parts.append(f"Took {self.execution_time:.2f}s")
        
        if self.has_errors():
            parts.append(f"Errors: {len(self.errors)}")
        
        return " | ".join(parts)


# Rebuild models to resolve forward references
from .search_query import SearchQuery
SearchResults.model_rebuild()