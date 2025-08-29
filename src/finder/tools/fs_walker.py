"""
Filesystem walker for the AI Filesystem Finder.

This module provides functionality to traverse directories, match files based on patterns,
extract metadata, and apply filters. It respects ignore patterns and provides efficient
directory traversal with configurable constraints.
"""

import os
import re
import stat
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import mimetypes
try:
    import pwd
    import grp
except ImportError:
    # Windows doesn't have pwd/grp modules
    pwd = None
    grp = None

from ..models.search_results import FileMatch, FileMetadata, MatchType
from ..models.config import FinderConfig


logger = logging.getLogger(__name__)


class FSWalker:
    """
    Filesystem walker that traverses directories and matches files.
    
    This class provides efficient directory traversal with support for:
    - Ignore pattern matching (gitignore-style)
    - Regex pattern matching for filenames and paths
    - Metadata extraction and filtering
    - Parallel processing for large directory trees
    """
    
    def __init__(self, config: FinderConfig):
        """
        Initialize the filesystem walker.
        
        Args:
            config: Configuration object containing roots, ignore patterns, and limits
        """
        self.config = config
        self._compiled_ignore_patterns = getattr(config, '_compiled_ignore_patterns', [])
        self._stats = {
            'files_scanned': 0,
            'files_matched': 0,
            'directories_traversed': 0,
            'files_ignored': 0,
            'errors': 0
        }
    
    def walk_paths(self, roots: List[str], patterns: List[str]) -> Iterator[FileMatch]:
        """
        Walk through root directories and yield matching files.
        
        Args:
            roots: List of root directory paths to search
            patterns: List of regex patterns to match against filenames/paths
            
        Yields:
            FileMatch objects for files that match the patterns
        """
        compiled_patterns = self._compile_patterns(patterns)
        
        for root in roots:
            try:
                root_path = Path(root).resolve()
                if not root_path.exists():
                    logger.warning(f"Root directory does not exist: {root_path}")
                    continue
                
                if not root_path.is_dir():
                    logger.warning(f"Root path is not a directory: {root_path}")
                    continue
                
                logger.info(f"Walking directory tree: {root_path}")
                yield from self._walk_directory(root_path, compiled_patterns)
                
            except Exception as e:
                logger.error(f"Error walking root {root}: {e}")
                self._stats['errors'] += 1
    
    def _walk_directory(self, root_path: Path, compiled_patterns: List[re.Pattern]) -> Iterator[FileMatch]:
        """
        Recursively walk a single directory tree.
        
        Args:
            root_path: Root directory to walk
            compiled_patterns: Compiled regex patterns for matching
            
        Yields:
            FileMatch objects for matching files
        """
        try:
            for current_dir, subdirs, files in os.walk(root_path):
                current_path = Path(current_dir)
                self._stats['directories_traversed'] += 1
                
                # Filter subdirectories based on ignore patterns
                subdirs[:] = [d for d in subdirs if not self._should_ignore(current_path / d)]
                
                # Process files in current directory
                for filename in files:
                    file_path = current_path / filename
                    
                    # Check ignore patterns
                    if self._should_ignore(file_path):
                        self._stats['files_ignored'] += 1
                        continue
                    
                    self._stats['files_scanned'] += 1
                    
                    # Check if file matches any pattern
                    if self._matches_patterns(file_path, compiled_patterns):
                        try:
                            file_match = self._create_file_match(file_path)
                            if file_match:
                                self._stats['files_matched'] += 1
                                yield file_match
                        except Exception as e:
                            logger.warning(f"Error processing file {file_path}: {e}")
                            self._stats['errors'] += 1
                    
                    # Check limits
                    if self._stats['files_scanned'] >= self.config.limits.max_files:
                        logger.warning(f"Reached maximum file limit: {self.config.limits.max_files}")
                        return
                        
        except Exception as e:
            logger.error(f"Error walking directory {root_path}: {e}")
            self._stats['errors'] += 1
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """
        Compile regex patterns for efficient matching.
        
        Args:
            patterns: List of regex pattern strings
            
        Returns:
            List of compiled regex patterns
        """
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        return compiled
    
    def _should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file or directory should be ignored based on ignore patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the path should be ignored
        """
        if not self._compiled_ignore_patterns:
            return False
        
        # Convert path to string relative to search roots for pattern matching
        path_str = str(file_path)
        
        # Track if path is ignored (default: not ignored)
        is_ignored = False
        
        for pattern_info in self._compiled_ignore_patterns:
            regex = pattern_info['regex']
            is_negation = pattern_info['is_negation']
            
            if regex.search(path_str):
                if is_negation:
                    # Negation pattern matches - don't ignore
                    is_ignored = False
                else:
                    # Regular ignore pattern matches - ignore
                    is_ignored = True
        
        return is_ignored 
   
    def _matches_patterns(self, file_path: Path, compiled_patterns: List[re.Pattern]) -> bool:
        """
        Check if a file matches any of the compiled patterns.
        
        Args:
            file_path: Path to check
            compiled_patterns: List of compiled regex patterns
            
        Returns:
            True if the file matches any pattern
        """
        if not compiled_patterns:
            return True  # No patterns means match all files
        
        # Check against filename
        filename = file_path.name
        full_path = str(file_path)
        
        for pattern in compiled_patterns:
            if pattern.search(filename) or pattern.search(full_path):
                return True
        
        return False
    
    def _create_file_match(self, file_path: Path) -> Optional[FileMatch]:
        """
        Create a FileMatch object for a matched file.
        
        Args:
            file_path: Path to the matched file
            
        Returns:
            FileMatch object or None if file cannot be processed
        """
        try:
            metadata = self._extract_metadata(file_path)
            if not metadata:
                return None
            
            # Check file size limits
            if metadata.size > self.config.limits.max_bytes_per_file:
                logger.debug(f"Skipping large file: {file_path} ({metadata.size} bytes)")
                return None
            
            return FileMatch(
                path=str(file_path.resolve()),
                score=1.0,  # Default score for literal matches
                match_type=MatchType.LITERAL,
                snippets=[],  # No snippets for filesystem walker
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Error creating file match for {file_path}: {e}")
            return None
    
    def _extract_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """
        Extract metadata from a file.
        
        Args:
            file_path: Path to extract metadata from
            
        Returns:
            FileMetadata object or None if extraction fails
        """
        try:
            stat_result = file_path.stat()
            
            # Get basic file information
            size = stat_result.st_size
            modified_time = datetime.fromtimestamp(stat_result.st_mtime)
            created_time = None
            
            # Try to get creation time (platform dependent)
            if hasattr(stat_result, 'st_birthtime'):
                # macOS
                created_time = datetime.fromtimestamp(stat_result.st_birthtime)
            elif hasattr(stat_result, 'st_ctime'):
                # Linux/Windows (ctime is not creation time on Linux, but close enough)
                created_time = datetime.fromtimestamp(stat_result.st_ctime)
            
            # Get file owner (Unix-like systems)
            owner = None
            try:
                if pwd and hasattr(stat_result, 'st_uid'):
                    owner = pwd.getpwuid(stat_result.st_uid).pw_name
                elif hasattr(stat_result, 'st_uid'):
                    owner = str(stat_result.st_uid)
            except (KeyError, AttributeError):
                # User not found or Windows
                owner = str(stat_result.st_uid) if hasattr(stat_result, 'st_uid') else None
            
            # Get file extension
            extension = file_path.suffix.lower() if file_path.suffix else None
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Get permissions
            permissions = oct(stat.S_IMODE(stat_result.st_mode))
            
            # Check if file is binary
            is_binary = self._is_binary_file(file_path)
            
            return FileMetadata(
                size=size,
                modified_time=modified_time,
                created_time=created_time,
                owner=owner,
                extension=extension,
                mime_type=mime_type,
                permissions=permissions,
                is_binary=is_binary
            )
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
            return None  
  
    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Check if a file appears to be binary.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file appears to be binary
        """
        try:
            # Read a small chunk to check for binary content
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
            
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True
            
            # Check for high ratio of non-printable characters
            if chunk:
                printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
                ratio = printable_chars / len(chunk)
                return ratio < 0.7  # If less than 70% printable, consider binary
            
            return False
            
        except Exception:
            # If we can't read the file, assume it might be binary
            return True
    
    def apply_filters(self, matches: List[FileMatch], filters: Dict[str, Any]) -> List[FileMatch]:
        """
        Apply metadata filters to a list of file matches.
        
        Args:
            matches: List of FileMatch objects to filter
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered list of FileMatch objects
        """
        if not filters:
            return matches
        
        filtered_matches = []
        
        for match in matches:
            if self._passes_filters(match, filters):
                filtered_matches.append(match)
        
        return filtered_matches
    
    def _passes_filters(self, match: FileMatch, filters: Dict[str, Any]) -> bool:
        """
        Check if a file match passes all specified filters.
        
        Args:
            match: FileMatch to check
            filters: Dictionary of filter criteria
            
        Returns:
            True if the match passes all filters
        """
        if not match.metadata:
            return False
        
        metadata = match.metadata
        
        # Extension filters
        if 'extensions' in filters:
            extensions = filters['extensions']
            if metadata.extension not in extensions:
                return False
        
        # Size filters
        if 'size_min' in filters:
            if metadata.size < filters['size_min']:
                return False
        
        if 'size_max' in filters:
            if metadata.size > filters['size_max']:
                return False
        
        # Date filters
        if 'modified_after' in filters:
            after_date = self._parse_date_filter(filters['modified_after'])
            if after_date and metadata.modified_time < after_date:
                return False
        
        if 'modified_before' in filters:
            before_date = self._parse_date_filter(filters['modified_before'])
            if before_date and metadata.modified_time > before_date:
                return False
        
        if 'created_after' in filters:
            after_date = self._parse_date_filter(filters['created_after'])
            if after_date and metadata.created_time and metadata.created_time < after_date:
                return False
        
        if 'created_before' in filters:
            before_date = self._parse_date_filter(filters['created_before'])
            if before_date and metadata.created_time and metadata.created_time > before_date:
                return False
        
        # Owner filter
        if 'owner' in filters:
            if metadata.owner != filters['owner']:
                return False
        
        # MIME type filter
        if 'mime_type' in filters:
            if metadata.mime_type != filters['mime_type']:
                return False
        
        return True
    
    def _parse_date_filter(self, date_value: Union[str, datetime, int]) -> Optional[datetime]:
        """
        Parse a date filter value into a datetime object.
        
        Args:
            date_value: Date value to parse (string, datetime, or timestamp)
            
        Returns:
            Parsed datetime object or None if parsing fails
        """
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, (int, float)):
            try:
                return datetime.fromtimestamp(date_value)
            except (ValueError, OSError):
                return None
        
        if isinstance(date_value, str):
            # Handle relative date strings
            date_value = date_value.lower().strip()
            
            # Parse relative dates like "7 days", "1 week", "2 months"
            relative_patterns = [
                (r'(\d+)\s*days?', lambda n: datetime.now() - timedelta(days=int(n))),
                (r'(\d+)\s*weeks?', lambda n: datetime.now() - timedelta(weeks=int(n))),
                (r'(\d+)\s*months?', lambda n: datetime.now() - timedelta(days=int(n) * 30)),
                (r'(\d+)\s*years?', lambda n: datetime.now() - timedelta(days=int(n) * 365)),
            ]
            
            for pattern, calculator in relative_patterns:
                match = re.match(pattern, date_value)
                if match:
                    try:
                        return calculator(match.group(1))
                    except (ValueError, OverflowError):
                        continue
            
            # Try to parse absolute date strings
            date_formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the filesystem walking operation.
        
        Returns:
            Dictionary containing operation statistics
        """
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset the statistics counters."""
        self._stats = {
            'files_scanned': 0,
            'files_matched': 0,
            'directories_traversed': 0,
            'files_ignored': 0,
            'errors': 0
        }


def create_filename_patterns(query_text: str) -> List[str]:
    """
    Create regex patterns for filename matching based on query text.
    
    Args:
        query_text: Natural language query text
        
    Returns:
        List of regex patterns for filename matching
    """
    patterns = []
    
    # Split query into words and create patterns
    words = re.findall(r'\w+', query_text.lower())
    
    for word in words:
        # Exact word match in filename
        patterns.append(rf'\b{re.escape(word)}\b')
        
        # Word as part of filename (case insensitive)
        patterns.append(rf'{re.escape(word)}')
    
    # Create pattern for all words (AND logic)
    if len(words) > 1:
        word_patterns = [rf'(?=.*{re.escape(word)})' for word in words]
        patterns.append(''.join(word_patterns) + r'.*')
    
    return patterns


def create_extension_patterns(extensions: List[str]) -> List[str]:
    """
    Create regex patterns for file extension matching.
    
    Args:
        extensions: List of file extensions (with or without leading dot)
        
    Returns:
        List of regex patterns for extension matching
    """
    patterns = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        
        # Match files with this extension
        patterns.append(rf'{re.escape(ext)}$')
    
    return patterns