"""
Unit tests for gitignore-style ignore pattern matching.

Tests the comprehensive gitignore pattern implementation including
wildcards, negation, directory patterns, and edge cases.
"""

import pytest
import tempfile
from pathlib import Path

from finder.models.config import FinderConfig, SecurityConfig


class TestIgnorePatterns:
    """Test cases for gitignore-style ignore patterns."""
    
    def test_basic_wildcards(self):
        """Test basic wildcard patterns (* and ?)."""
        config = FinderConfig(
            roots=["."],
            ignore=["*.pyc", "test?.txt", "temp*"]
        )
        
        # Test * wildcard
        assert config.should_ignore("module.pyc")
        assert config.should_ignore("src/module.pyc")
        assert not config.should_ignore("module.py")
        
        # Test ? wildcard
        assert config.should_ignore("test1.txt")
        assert config.should_ignore("testA.txt")
        assert not config.should_ignore("test12.txt")
        assert not config.should_ignore("test.txt")
        
        # Test * at start
        assert config.should_ignore("temp123")
        assert config.should_ignore("temporary")
        assert not config.should_ignore("mytemp")
    
    def test_directory_wildcards(self):
        """Test ** directory wildcard patterns."""
        config = FinderConfig(
            roots=["."],
            ignore=["**/node_modules/**", "**/*.log", "build/**/temp"]
        )
        
        # Test **/ prefix
        assert config.should_ignore("node_modules/package/index.js")
        assert config.should_ignore("project/node_modules/lib/file.js")
        assert config.should_ignore("deep/path/node_modules/anything")
        
        # Test **/*.ext pattern
        assert config.should_ignore("app.log")
        assert config.should_ignore("logs/error.log")
        assert config.should_ignore("deep/path/debug.log")
        
        # Test middle ** pattern
        assert config.should_ignore("build/debug/temp")
        assert config.should_ignore("build/release/obj/temp")
        assert not config.should_ignore("build/temp")  # No intermediate directory
    
    def test_directory_only_patterns(self):
        """Test patterns that match directories only (trailing /)."""
        config = FinderConfig(
            roots=["."],
            ignore=["build/", "*.tmp/", "**/cache/"]
        )
        
        # These should match directories
        assert config.should_ignore("build/")
        assert config.should_ignore("build")  # Directory without trailing slash
        assert config.should_ignore("project.tmp/")
        assert config.should_ignore("project.tmp")
        assert config.should_ignore("deep/path/cache/")
        assert config.should_ignore("cache/")
    
    def test_rooted_patterns(self):
        """Test patterns that are rooted (start with /)."""
        config = FinderConfig(
            roots=["."],
            ignore=["/build", "/src/*.py", "/*.log"]
        )
        
        # Rooted patterns should only match at root
        assert config.should_ignore("build")
        assert config.should_ignore("src/main.py")
        assert config.should_ignore("error.log")
        
        # Should not match in subdirectories
        assert not config.should_ignore("project/build")
        assert not config.should_ignore("lib/src/main.py")
        assert not config.should_ignore("logs/error.log")
    
    def test_negation_patterns(self):
        """Test negation patterns (starting with !)."""
        config = FinderConfig(
            roots=["."],
            ignore=[
                "*.log",
                "!important.log",
                "temp/*",
                "!temp/keep.txt",
                "**/build/**",
                "!**/build/assets/**"
            ]
        )
        
        # Basic negation
        assert config.should_ignore("debug.log")
        assert not config.should_ignore("important.log")  # Negated
        
        # Directory negation
        assert config.should_ignore("temp/delete.txt")
        assert not config.should_ignore("temp/keep.txt")  # Negated
        
        # Complex negation with **
        assert config.should_ignore("project/build/obj/file.o")
        assert not config.should_ignore("project/build/assets/style.css")  # Negated
    
    def test_character_classes(self):
        """Test character class patterns [abc] and [!abc]."""
        config = FinderConfig(
            roots=["."],
            ignore=["test[0-9].txt", "file[!abc].py", "doc[a-z].md"]
        )
        
        # Positive character classes
        assert config.should_ignore("test1.txt")
        assert config.should_ignore("test9.txt")
        assert not config.should_ignore("testA.txt")
        assert not config.should_ignore("test10.txt")
        
        # Negative character classes
        assert config.should_ignore("filed.py")
        assert config.should_ignore("file1.py")
        assert not config.should_ignore("filea.py")
        assert not config.should_ignore("fileb.py")
        assert not config.should_ignore("filec.py")
        
        # Range character classes
        assert config.should_ignore("doca.md")
        assert config.should_ignore("docz.md")
        assert not config.should_ignore("doc1.md")
        assert not config.should_ignore("docA.md")
    
    def test_escaped_characters(self):
        """Test escaped special characters."""
        config = FinderConfig(
            roots=["."],
            ignore=[r"\*.txt", r"file\?.py", r"\!important", r"\#comment"]
        )
        
        # Escaped wildcards should be treated as literals
        assert config.should_ignore("*.txt")
        assert not config.should_ignore("test.txt")
        
        assert config.should_ignore("file?.py")
        assert not config.should_ignore("file1.py")
        
        # Escaped special characters
        assert config.should_ignore("!important")
        assert config.should_ignore("#comment")
    
    def test_complex_patterns(self):
        """Test complex real-world patterns."""
        config = FinderConfig(
            roots=["."],
            ignore=[
                # Python
                "**/__pycache__/**",
                "*.py[cod]",
                "*.so",
                ".Python",
                "build/",
                "develop-eggs/",
                "dist/",
                "downloads/",
                "eggs/",
                ".eggs/",
                "lib/",
                "lib64/",
                "parts/",
                "sdist/",
                "var/",
                "wheels/",
                "*.egg-info/",
                ".installed.cfg",
                "*.egg",
                
                # Node.js
                "node_modules/",
                "npm-debug.log*",
                "yarn-debug.log*",
                "yarn-error.log*",
                ".npm",
                ".yarn-integrity",
                
                # IDEs
                ".vscode/",
                ".idea/",
                "*.swp",
                "*.swo",
                "*~",
                
                # OS
                ".DS_Store",
                "Thumbs.db",
                
                # Git
                ".git/",
                ".gitignore",
                
                # But keep some important files
                "!.gitkeep",
                "!important.pyc"
            ]
        )
        
        # Python patterns
        assert config.should_ignore("__pycache__/module.cpython-39.pyc")
        assert config.should_ignore("src/__pycache__/test.pyc")
        assert config.should_ignore("module.pyc")
        assert config.should_ignore("module.pyo")
        assert config.should_ignore("module.pyd")
        assert config.should_ignore("library.so")
        assert config.should_ignore("build/")
        assert config.should_ignore("dist/package.tar.gz")
        
        # Node.js patterns
        assert config.should_ignore("node_modules/package/index.js")
        assert config.should_ignore("npm-debug.log")
        assert config.should_ignore("yarn-error.log.1")
        
        # IDE patterns
        assert config.should_ignore(".vscode/settings.json")
        assert config.should_ignore(".idea/workspace.xml")
        assert config.should_ignore("file.swp")
        assert config.should_ignore("backup~")
        
        # OS patterns
        assert config.should_ignore(".DS_Store")
        assert config.should_ignore("Thumbs.db")
        
        # Git patterns
        assert config.should_ignore(".git/config")
        assert config.should_ignore(".gitignore")
        
        # Negated patterns
        assert not config.should_ignore(".gitkeep")
        assert not config.should_ignore("important.pyc")
        
        # Should not ignore regular files
        assert not config.should_ignore("main.py")
        assert not config.should_ignore("src/module.py")
        assert not config.should_ignore("README.md")
    
    def test_pattern_order_matters(self):
        """Test that pattern order affects the final result."""
        # First config: ignore then negate
        config1 = FinderConfig(
            roots=["."],
            ignore=["*.log", "!important.log", "*.log"]  # Re-ignore after negation
        )
        
        # Second config: negate then ignore
        config2 = FinderConfig(
            roots=["."],
            ignore=["!important.log", "*.log"]  # Ignore after negation
        )
        
        # In config1, important.log should be ignored (last *.log wins)
        assert config1.should_ignore("important.log")
        
        # In config2, important.log should be ignored (*.log overrides negation)
        assert config2.should_ignore("important.log")
    
    def test_empty_and_comment_patterns(self):
        """Test handling of empty patterns and comments."""
        config = FinderConfig(
            roots=["."],
            ignore=[
                "",  # Empty pattern
                "   ",  # Whitespace only
                "# This is a comment",
                "*.pyc",  # Valid pattern
                "# Another comment",
                "*.log"  # Another valid pattern
            ]
        )
        
        # Only valid patterns should work
        assert config.should_ignore("test.pyc")
        assert config.should_ignore("debug.log")
        
        # Comments and empty patterns should not match anything
        assert not config.should_ignore("# This is a comment")
        assert not config.should_ignore("")
    
    def test_edge_cases(self):
        """Test edge cases and malformed patterns."""
        config = FinderConfig(
            roots=["."],
            ignore=[
                "*",  # Matches everything in current directory
                "**",  # Matches everything everywhere
                "/",  # Root directory only
                "a/",  # Directory named 'a'
                "*/b",  # File 'b' in any immediate subdirectory
                "**/c/**",  # Everything under any directory named 'c'
            ]
        )
        
        # * should match files in current directory
        assert config.should_ignore("file.txt")
        assert config.should_ignore("anything")
        
        # ** should match everything
        assert config.should_ignore("deep/path/file.txt")
        
        # Directory patterns
        assert config.should_ignore("a/")
        assert config.should_ignore("a")
        
        # Subdirectory patterns
        assert config.should_ignore("dir/b")
        assert not config.should_ignore("deep/dir/b")  # Not immediate subdirectory
        
        # Everything under 'c'
        assert config.should_ignore("c/file.txt")
        assert config.should_ignore("path/c/deep/file.txt")
    
    def test_path_normalization(self):
        """Test that paths are properly normalized for matching."""
        config = FinderConfig(
            roots=["."],
            ignore=["temp/*.txt", "build/debug/"]
        )
        
        # Different path formats should all work
        assert config.should_ignore("temp/file.txt")
        assert config.should_ignore("temp\\file.txt")  # Windows-style path
        assert config.should_ignore("./temp/file.txt")  # Relative path
        
        # Directory matching
        assert config.should_ignore("build/debug/")
        assert config.should_ignore("build/debug")
        assert config.should_ignore("build\\debug\\")  # Windows-style
    
    def test_case_sensitivity(self):
        """Test case sensitivity in pattern matching."""
        config = FinderConfig(
            roots=["."],
            ignore=["*.TXT", "Build/", "NODE_MODULES/"]
        )
        
        # Patterns should be case-sensitive by default
        assert config.should_ignore("file.TXT")
        assert not config.should_ignore("file.txt")  # Different case
        
        assert config.should_ignore("Build/")
        assert not config.should_ignore("build/")  # Different case
        
        assert config.should_ignore("NODE_MODULES/")
        assert not config.should_ignore("node_modules/")  # Different case


class TestIgnorePatternValidation:
    """Test validation of ignore patterns."""
    
    def test_invalid_regex_patterns(self):
        """Test that invalid regex patterns raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid ignore pattern"):
            FinderConfig(
                roots=["."],
                ignore=["[invalid"]  # Unclosed character class
            )
    
    def test_valid_complex_patterns(self):
        """Test that complex but valid patterns are accepted."""
        # This should not raise any errors
        config = FinderConfig(
            roots=["."],
            ignore=[
                "**/*.{js,ts,jsx,tsx}",  # Note: This won't work as expected in our implementation
                "**/node_modules/**",
                "!**/node_modules/important/**",
                "build/[0-9][0-9][0-9]/",
                "temp[!0-9]*.log"
            ]
        )
        
        # Basic functionality should work
        assert config.should_ignore("src/node_modules/package/index.js")
        assert not config.should_ignore("src/node_modules/important/file.js")


class TestIgnorePatternPerformance:
    """Test performance characteristics of ignore patterns."""
    
    def test_many_patterns_performance(self):
        """Test performance with many ignore patterns."""
        # Create a config with many patterns
        patterns = []
        for i in range(1000):
            patterns.append(f"temp{i}/**")
            patterns.append(f"*.tmp{i}")
            patterns.append(f"!important{i}.tmp{i}")
        
        config = FinderConfig(roots=["."], ignore=patterns)
        
        # Test that pattern matching still works efficiently
        assert config.should_ignore("temp500/file.txt")
        assert config.should_ignore("test.tmp500")
        assert not config.should_ignore("important500.tmp500")
        assert not config.should_ignore("regular_file.txt")
    
    def test_deep_path_performance(self):
        """Test performance with very deep paths."""
        config = FinderConfig(
            roots=["."],
            ignore=["**/node_modules/**", "**/*.log", "!**/important/**"]
        )
        
        # Create a very deep path
        deep_path = "/".join([f"level{i}" for i in range(100)]) + "/file.log"
        
        # Should still work efficiently
        assert config.should_ignore(deep_path)
        
        # Test with node_modules in deep path
        deep_node_path = "/".join([f"level{i}" for i in range(50)]) + "/node_modules/package/index.js"
        assert config.should_ignore(deep_node_path)


if __name__ == "__main__":
    pytest.main([__file__])