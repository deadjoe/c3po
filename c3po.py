"""
C3PO - INFO File Conversion Tool

Features:
- Support for large files with automatic chunking and parallel processing
- Preserve document structure and formatting
- Content validation to ensure conversion quality
- Support for parallel processing of large files

Usage: python c3po.py input_file.info output_file.txt [--no-verify] [--workers=N] [--debug] [--log=LEVEL] [--enable-semantic-markers]

Arguments:
  --no-verify: Skip content validation
  --workers=N: Set number of worker threads
  --debug: Enable debug mode
  --log=LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  --enable-semantic-markers: Enable semantic marking feature
"""

import os
import sys
import re
import time
import random
import hashlib
import io
import logging
import concurrent.futures
import traceback
from collections import defaultdict
import math

# Check whether tqdm library is available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Version information
__version__ = "1.0.0"
__date__ = "2025-03-27"
__author__ = "C3PO Team"

class ErrorHandler:
    """Error handling class, provides unified error handling mechanism"""
    
    def __init__(self, log_manager):
        """Initialize error handler
        
        Args:
            log_manager: log manager instance
        """
        self.log = log_manager
    
    def handle_error(self, error, operation_name, fatal=False, return_value=None):
        """Handle error
        
        Args:
            error: exception object
            operation_name: operation name
            fatal: whether it is a fatal error
            return_value: value to return if not a fatal error
            
        Returns:
            if not a fatal error, returns return_value; otherwise throws an exception
        """
        error_msg = f"{operation_name} error: {str(error)}"
        
        if fatal:
            error_msg += f"\n{traceback.format_exc()}"
            self.log.error(error_msg)
            raise error
        else:
            self.log.warning(error_msg)
            return return_value
    
    def try_operation(self, operation, operation_name, args=None, kwargs=None, fatal=False, return_value=None):
        """Try to execute an operation, handle possible exceptions
        
        Args:
            operation: function to execute
            operation_name: operation name
            args: positional arguments
            kwargs: keyword arguments
            fatal: whether it is a fatal error
            return_value: value to return if not a fatal error
            
        Returns:
            operation result or return_value
        """
        args = args or ()
        kwargs = kwargs or {}
        
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            return self.handle_error(e, operation_name, fatal, return_value)

class ConfigManager:
    """Configuration management class, responsible for managing all configuration parameters and regex patterns"""
    
    def __init__(self, config_file=None):
        """Initialize configuration manager
        
        Args:
            config_file: configuration file path (optional)
        """
        # Default configuration
        self.default_config = {
            'chunk_size': 1024 * 1024,  # 1MB
            'large_file_threshold': 10 * 1024 * 1024,  # 10MB
            'max_workers': None,  # default uses CPU core count
            'verify': True,  # content validation enabled by default
            'debug': False,  # Debug mode disabled by default
            'log_level': logging.INFO,  # default log level
            'enable_semantic_markers': False,  # semantic marking disabled by default
            'encoding': 'utf-8',  # default encoding
            'fallback_encoding': 'latin1',  # fallback encoding
            'max_content_size': 10 * 1024 * 1024,  # maximum content size
            'max_matches_per_type': 1000,  # maximum matches per type
            'validation_sample_size': 100 * 1024,  # validation sample size
            'min_chunk_size': 512 * 1024,  # minimum chunk size
            'max_chunk_size_memory_ratio': 0.1,  # maximum ratio of chunk size to available memory
            'color_enabled': True,  # whether to enable colored logs
            'progress_enabled': True,  # whether to enable progress display
        }
        
        # user configuration
        self.user_config = {}
        
        # load configuration file
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # regular expression patterns
        self.regex_patterns = {
            # file header information
            'headers': re.compile(r'^\s*\*{3}.*?\*{3}\s*$', re.MULTILINE),

            # separator
            'separator': re.compile(r'^\s*[-=_]{3,}\s*$', re.MULTILINE),

            # menu area
            'menu': re.compile(r'^\s*\[.*?menu.*?\]\s*$.*?(?=\n\n)', re.MULTILINE | re.DOTALL),

            # menu items - preserve item names
            'menu_items': re.compile(r'^\s*\[([^]]+)\].*?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),

            # control characters
            'control_chars': re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'),

            # empty line normalization
            'empty_lines': re.compile(r'\n{3,}'),
        }
        
        # SQL keywords - used to identify code blocks
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
            'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN',
            'INSERT INTO', 'UPDATE', 'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE',
            'DROP TABLE', 'TRUNCATE TABLE', 'BEGIN TRANSACTION', 'COMMIT', 'ROLLBACK'
        ]
        
        # semantic markers - used to enhance document structure
        self.semantic_markers = {
            'code_block': {
                'pattern': re.compile(r'```.*?```', re.DOTALL),
                'tag': 'CODE'
            },
            'sql_block': {
                'pattern': re.compile(r'(?i)(' + '|'.join(self.sql_keywords) + r')\s+\w+'),
                'tag': 'SQL'
            },
            'table': {
                'pattern': re.compile(r'^\s*\|.*\|\s*$(?:\s*\|[-:]+\|[-:]+\|\s*$)?', re.MULTILINE),
                'tag': 'TABLE'
            },
            'list': {
                'pattern': re.compile(r'(?:^\s*[-*+]\s+.*$(?:\n\s*[-*+]\s+.*$)+)|(?:^\s*\d+\.\s+.*$(?:\n\s*\d+\.\s+.*$)+)', re.MULTILINE),
                'tag': 'LIST'
            },
            'heading': {
                'pattern': re.compile(r'^\s*#{1,6}\s+.*$', re.MULTILINE),
                'tag': 'HEADING'
            }
        }
    
    def _load_config_file(self, config_file):
        """Load configuration from file
        
        Args:
            config_file: configuration file path
        """
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                self.user_config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load configuration file: {str(e)}")
    
    def get_config(self, key, default=None):
        """Get configuration value
        
        Args:
            key: configuration key name
            default: default value if key does not exist
            
        Returns:
            configuration value
        """
        # prioritize user configuration, then default configuration, finally the provided default value
        return self.user_config.get(key, self.default_config.get(key, default))
    
    def set_config(self, key, value):
        """Set configuration value
        
        Args:
            key: configuration key name
            value: configuration value
        """
        self.user_config[key] = value
    
    def save_config(self, config_file):
        """Save configuration to file
        
        Args:
            config_file: configuration file path
            
        Returns:
            bool: whether save was successful
        """
        try:
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, indent=4)
            return True
        except Exception as e:
            print(f"Warning: Failed to save configuration file: {str(e)}")
            return False
    
    def get_pattern(self, key):
        """Get regular expression pattern
        
        Args:
            key: pattern key name
            
        Returns:
            regular expression patterns
        """
        return self.regex_patterns.get(key)
    
    def get_all_patterns(self):
        """Get all regular expression patterns
        
        Returns:
            dictionary of all regular expression patterns
        """
        return self.regex_patterns
    
    def get_semantic_marker(self, key):
        """Get semantic marker
        
        Args:
            key: marker key name
            
        Returns:
            semantic marker
        """
        return self.semantic_markers.get(key)
    
    def get_all_semantic_markers(self):
        """Get all semantic markers
        
        Returns:
            Dictionary of all semantic markers
        """
        return self.semantic_markers
    
    def get_semantic_markers(self):
        """Get semantic marker definitions (compatible interface)
        
        Returns:
            dict: Dictionary of semantic marker definitions
        """
        return self.get_all_semantic_markers()

class LogManager:
    """Log management class, encapsulates log system settings and usage"""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
    }
    
    def __init__(self, level=logging.INFO):
        """Initialize log manager
        
        Args:
            level: log level
        """
        self.logger = self._setup_logging(level)
        self.progress_start_time = None
        self.progress_last_update = None
        self.progress_total = None
        self.progress_current = None
        self.progress_enabled = True
        self.color_enabled = True
        
        # Check if running in terminal environment
        self.is_terminal = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _setup_logging(self, level):
        """Configure logging system
        
        Args:
            level: log level
            
        Returns:
            log object
        """
        # Create log format
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.StreamHandler()  # Output to console
            ]
        )
        
        # Return log object
        return logging.getLogger('c3po')
    
    def _colorize(self, text, color):
        """Add color to text
        
        Args:
            text: text to be colored
            color: color code
            
        Returns:
            str: colored text
        """
        if not self.color_enabled or not self.is_terminal:
            return text
            
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['RESET']}"
    
    def set_level(self, level):
        """Set log level
        
        Args:
            level: log level
        """
        self.logger.setLevel(level)
    
    def enable_color(self, enabled=True):
        """Enable or disable colored logs
        
        Args:
            enabled: whether to enable
        """
        self.color_enabled = enabled
    
    def debug(self, message):
        """Log debug information
        
        Args:
            message: log message
        """
        self.logger.debug(self._colorize(message, 'CYAN'))
    
    def info(self, message):
        """Log general information
        
        Args:
            message: log message
        """
        self.logger.info(self._colorize(message, 'WHITE'))
    
    def warning(self, message):
        """Log warning information
        
        Args:
            message: log message
        """
        self.logger.warning(self._colorize(message, 'YELLOW'))
    
    def error(self, message):
        """Log error information
        
        Args:
            message: log message
        """
        self.logger.error(self._colorize(message, 'RED'))
    
    def critical(self, message):
        """Log critical error information
        
        Args:
            message: log message
        """
        self.logger.critical(self._colorize(f"{self.COLORS['BOLD']}{message}", 'RED'))
    
    def success(self, message):
        """Log success information
        
        Args:
            message: log message
        """
        self.logger.info(self._colorize(message, 'GREEN'))
    
    def print_progress(self, message, step=None, total_steps=None):
        """Print progress information
        
        Args:
            message: progress message
            step: current step
            total_steps: total steps
        """
        if step is not None and total_steps is not None:
            log_msg = f"[{step}/{total_steps}] {message}"
        else:
            log_msg = message
        
        # Use green for progress information, consistent with success information
        self.logger.info(self._colorize(log_msg, 'GREEN'))
    
    def start_progress(self, total, message="Processing progress"):
        """Start progress tracking
        
        Args:
            total: total count
            message: progress message prefix
        """
        if not self.progress_enabled:
            return
            
        self.progress_start_time = time.time()
        self.progress_last_update = time.time()
        self.progress_total = total
        self.progress_current = 0
        self.progress_message = message
        
        self.info(f"{message}: 0% (0/{total})")
    
    def update_progress(self, current=None, increment=None):
        """Update progress
        
        Args:
            current: current count
            increment: increment
        """
        if not self.progress_enabled or self.progress_total is None:
            return
            
        # Update current progress
        if current is not None:
            self.progress_current = current
        elif increment is not None:
            self.progress_current += increment
        
        # Calculate progress percentage
        percent = self.progress_current / self.progress_total * 100
        
        # Control update frequency to avoid excessive output
        current_time = time.time()
        if current_time - self.progress_last_update < 0.5 and percent < 100:
            return
            
        self.progress_last_update = current_time
        
        # Calculate elapsed time and estimated remaining time
        elapsed = current_time - self.progress_start_time
        if self.progress_current > 0:
            remaining = elapsed * (self.progress_total - self.progress_current) / self.progress_current
            eta = self._format_time(remaining)
            elapsed_str = self._format_time(elapsed)
            
            # Build progress message
            msg = f"{self.progress_message}: {percent:.1f}% ({self.progress_current}/{self.progress_total}) - Time elapsed: {elapsed_str}, Estimated remaining: {eta}"
        else:
            msg = f"{self.progress_message}: {percent:.1f}% ({self.progress_current}/{self.progress_total})"
        
        self.info(msg)
    
    def end_progress(self, message=None):
        """End progress tracking
        
        Args:
            message: end message
        """
        if not self.progress_enabled or self.progress_total is None:
            return
            
        elapsed = time.time() - self.progress_start_time
        elapsed_str = self._format_time(elapsed)
        
        if message:
            self.info(f"{message} - Total time: {elapsed_str}")
        else:
            self.info(f"{self.progress_message} completed - Total time: {elapsed_str}")
        
        # Reset progress status
        self.progress_start_time = None
        self.progress_last_update = None
        self.progress_total = None
        self.progress_current = None
    
    def _format_time(self, seconds):
        """Format time
        
        Args:
            seconds: seconds
            
        Returns:
            str: formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.1f}h {minutes:.0f}min"
    
    def print_separator(self, char='-', length=60):
        """Print separator
        
        Args:
            char: separator character
            length: separator length
        """
        self.info(char * length)
    
    def enable_progress(self, enabled=True):
        """Enable or disable progress display
        
        Args:
            enabled: whether to enable
        """
        self.progress_enabled = enabled

class FileHandler:
    """File handling class, handles file read and write operations"""
    
    def __init__(self, log_manager):
        """Initialize file handler
        
        Args:
            log_manager: log manager instance
        """
        self.log = log_manager
    
    def get_file_info(self, file_path, large_file_threshold):
        """Get file information
        
        Args:
            file_path: file path
            large_file_threshold: large file threshold
        
        Returns:
            dict: dictionary containing file size, modification time and other information, returns None if file does not exist
        """
        if not os.path.exists(file_path):
            self.log.error(f"File not found: {file_path}")
            return None
        
        stats = os.stat(file_path)
        info = {
            'size': stats.st_size,
            'modified': stats.st_mtime,
            'is_large': stats.st_size > large_file_threshold
        }
        
        self.log.debug(f"File info: {file_path}, size: {self.format_size(info['size'])}, last modified: {time.ctime(info['modified'])}")
        return info
    
    def safe_open_file(self, file_path, mode='r', encoding='utf-8'):
        """Safely open file, handle encoding errors
        
        Args:
            file_path: file path
            mode: open mode
            encoding: encoding method
        
        Returns:
            file: file object
        
        Raises:
            IOError: if file cannot be opened or read
        """
        try:
            self.log.debug(f"Attempting to open file with {encoding} encoding: {file_path}")
            return open(file_path, mode=mode, encoding=encoding, errors='replace')
        except UnicodeDecodeError:
            self.log.warning(f"Failed to open file with {encoding} encoding, trying Latin-1 encoding...")
            return open(file_path, mode=mode, encoding='latin1', errors='replace')
        except Exception as e:
            self.log.error(f"Failed to open file: {file_path}, error: {str(e)}")
            raise
    
    def create_string_buffer(self):
        """Create string buffer
        
        Returns:
            StringIO: string buffer object
        """
        return io.StringIO()
    
    def format_size(self, size_bytes):
        """Format file size
        
        Args:
            size_bytes: file size (bytes)
        
        Returns:
            str: formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.2f} MB"
        else:
            return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"
    
    def format_time(self, seconds):
        """Format time
        
        Args:
            seconds: seconds
        
        Returns:
            str: formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f} s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} min"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} h "

class TextProcessor:
    """Text processing class, handles text chunks and format markers"""
    
    def __init__(self, config_manager, log_manager):
        """Initialize text processor
        
        Args:
            config_manager: configuration manager instance
            log_manager: log manager instance
        """
        self.config = config_manager
        self.log = log_manager
    
    def process_chunk(self, chunk, patterns, is_first_chunk=False, is_last_chunk=False):
        """Process a single text chunk, used for parallel processing
        
        Args:
            chunk: text chunk to process
            patterns: list of regular expression patterns to apply
            is_first_chunk: whether it is the first chunk
            is_last_chunk: whether it is the last chunk
        
        Returns:
            tuple: (processed content, dictionary of removed content hashes, replacement count)
        """
        content = chunk
        removed_content_hashes = defaultdict(int)
        replaced_count = 0
        
        # Only process complete paragraphs, avoid cutting content at chunk boundaries
        if not is_first_chunk and not is_last_chunk:
            # Find the start of the first complete paragraph
            first_para_start = content.find('\n\n')
            if first_para_start > 0:
                content = content[first_para_start + 2:]
                self.log.debug(f"Trimmed to first complete paragraph in chunk, removed {first_para_start + 2} characters")
            
            # Find the end of the last complete paragraph
            last_para_end = content.rfind('\n\n')
            if last_para_end > 0:
                content = content[:last_para_end]
                self.log.debug(f"Trimmed to last complete paragraph in chunk, kept {last_para_end} characters")
        
        # Apply format processing
        for pattern_key in patterns:
            pattern = self.config.get_pattern(pattern_key)
            
            # Use iterators instead of loading all matches at once
            for match in pattern.finditer(content):
                match_text = match.group(0)
                match_hash = hashlib.md5(str(match_text).encode()).hexdigest()
                removed_content_hashes[match_hash] += 1
                replaced_count += 1
            
            # Perform replacement
            if pattern_key == 'menu_items':
                # Special handling for menu items, preserve item names
                content = pattern.sub(r'\1', content)
            else:
                content = pattern.sub('', content)
        
        self.log.debug(f"Chunk processing completed, removed {replaced_count} items")
        return content, removed_content_hashes, replaced_count
    
    def process_format_markers(self, content, pattern_desc, tqdm_available=False):
        """Process format markers
        
        Args:
            content: text content to process
            pattern_desc: list of pattern descriptions to apply, in the format [(description, pattern key name), ...]
            tqdm_available: whether tqdm library is available
        
        Returns:
            tuple: (processed content, dictionary of removed content hashes, replacement count)
        """
        # Use dictionary to record summary information of removed content, rather than storing complete content
        removed_content_hashes = defaultdict(int)
        replaced_count = 0
        
        # Process different types of content in order
        pattern_iter = pattern_desc
        if tqdm_available:
            from tqdm import tqdm as tqdm_lib
            pattern_iter = tqdm_lib(pattern_iter, desc="    Removing format markers")
        
        for desc, pattern_key in pattern_iter:
            if not tqdm_available:
                self.log.info(f"    Processing {desc}...")
            
            pattern = self.config.get_pattern(pattern_key)
            matches = pattern.findall(content)
            
            # Record hash values of removed content
            for match in matches:
                match_hash = hashlib.md5(str(match).encode()).hexdigest()
                removed_content_hashes[match_hash] += 1
                replaced_count += 1
            
            # Perform replacement
            if pattern_key == 'menu_items':
                # Special handling for menu items, preserve item names
                content = pattern.sub(r'\1', content)
            else:
                content = pattern.sub('', content)
        
        self.log.info(f"    Total removed format markers: {replaced_count}")
        return content, removed_content_hashes, replaced_count
    
    def add_metadata_markers(self, content, max_content_size=10 * 1024 * 1024, max_matches_per_type=1000, debug=False, enable_semantic_markers=False):
        """Add semantic metadata markers to content (compatible interface)
        
        This method is kept as a compatible interface, actual processing is delegated to SemanticProcessor
        
        Args:
            content: text content to process
            max_content_size: maximum content size, content exceeding this size will be processed in chunks
            max_matches_per_type: maximum matches per type
            debug: whether to output debug information
            enable_semantic_markers: whether to enable semantic marking feature (disabled by default)
        
        Returns:
            str: content with metadata markers added
        """
        # If semantic marker feature is disabled, return original content directly
        if not enable_semantic_markers:
            self.log.debug("Semantic marking feature disabled, returning original content")
            return content
        
        # Temporarily disable all processing, return original content directly
        if debug:
            self.log.warning("Warning: All semantic marking processing temporarily disabled, returning original content")
            return content
            
        try:
            # Use new semantic processor
            from semantic_processor import SemanticProcessor
            processor = SemanticProcessor(self.config, self.log)
            
            # Determine whether to use chunked processing based on content size
            if len(content) > max_content_size:
                return processor.process_chunked(content, chunk_size=max_content_size)
            else:
                return processor.process(content)
        except Exception as e:
            self.log.error(f"Semantic processing error: {str(e)}")
            # Return original content on error
            return content
    
    def get_tqdm_instance(self, iterable=None, total=None, desc=None):
        """Get tqdm progress bar instance, if available
        
        Args:
            iterable: iterable object
            total: total count
            desc: description
        
        Returns:
            tqdm or iterable: tqdm instance or original iterable object
        """
        global TQDM_AVAILABLE
        if TQDM_AVAILABLE:
            from tqdm import tqdm
            return tqdm(iterable, total=total, desc=desc)
        else:
            if desc:
                self.log.info(f"{desc}...")
            return iterable

class ContentValidator:
    """Content validation class, verifies content consistency before and after processing"""
    
    def __init__(self, config_manager, log_manager):
        """Initialize content validator
        
        Args:
            config_manager: configuration manager instance
            log_manager: log manager instance
        """
        self.config = config_manager
        self.log = log_manager
        
        # Precompile regular expressions
        self.re_paragraphs = re.compile(r'\n\s*\n')
        self.re_whitespace = re.compile(r'\s+')
        self.re_keywords = re.compile(r'\b([A-Z]{2,})\b')
        self.re_words = re.compile(r'\b\w+\b')
    
    def validate(self, original, processed, removed_hashes=None, file_size=None, is_large_file=False):
        """Unified content validation entry point
        
        Args:
            original: original content (can be complete content or sample dictionary)
            processed: processed content
            removed_hashes: hash value dictionary of removed content (optional)
            file_size: original file size (bytes)
            is_large_file: whether it is a large file
            
        Returns:
            tuple: (validation result boolean, detailed report)
        """
        self.log.info("Starting content validation...")
        
        # Choose appropriate validation method based on file size and content type
        if is_large_file:
            # Large files use sampling validation
            if isinstance(original, dict):
                # Already a sample dictionary
                return self._validate_with_sampling(original, processed, removed_hashes, file_size)
            else:
                # Need to create sample
                samples = self._create_samples(original)
                return self._validate_with_sampling(samples, processed, removed_hashes, file_size or len(original))
        else:
            # Small files use simple validation
            if isinstance(original, dict):
                # Use complete content in dictionary
                if 'full' in original:
                    return self._validate_content_sample(original['full'], processed)
                else:
                    # Merge samples
                    combined = ''.join(original.values())
                    return self._validate_content_sample(combined, processed)
            else:
                # Use complete content directly
                return self._validate_content_sample(original, processed)
    
    def _create_samples(self, content):
        """Create sample from content
        
        Args:
            content: complete content
            
        Returns:
            dict: sample dictionary
        """
        sample_size = self.config.get_config('validation_sample_size', 100 * 1024)
        samples = {}
        
        # Extract beginning, middle and end parts
        if len(content) <= sample_size * 3:
            # Content is small, use complete content
            samples['full'] = content
        else:
            # Content is large, extract samples
            samples['start'] = content[:sample_size]
            middle_start = (len(content) - sample_size) // 2
            samples['middle'] = content[middle_start:middle_start + sample_size]
            samples['end'] = content[-sample_size:]
            
            # Random sampling
            for i in range(3):
                random_start = random.randint(0, max(0, len(content) - sample_size))
                samples[f'random_{i}'] = content[random_start:random_start + sample_size]
        
        return samples
    
    # Change original public method to private method
    def _simple_validate_content(self, original, processed, removed_hashes=None):
        """Simplified content validation function (internal use)
        
        Args:
            original: original content
            processed: processed content
            removed_hashes: hash value dictionary of removed content (optional)
        
        Returns:
            tuple: (validation result boolean, detailed report)
        """
        # For large files, only use samples for validation
        if len(original) > 1024 * 1024 or len(processed) > 1024 * 1024:
            self.log.info("Large file detected, using sampling validation method")
            # Take 100KB from the beginning and end of the file for validation
            sample_size = self.config.get_config('validation_sample_size', 100 * 1024)
            orig_start = original[:sample_size]
            orig_end = original[-sample_size:] if len(original) > sample_size else ""
            proc_start = processed[:sample_size]
            proc_end = processed[-sample_size:] if len(processed) > sample_size else ""
            
            # Merge samples
            original_sample = orig_start + orig_end
            processed_sample = proc_start + proc_end
            
            self.log.debug(f"Original content sample size: {len(original_sample)}, processed content sample size: {len(processed_sample)}")
            
            # Use samples for validation
            return self._validate_content_sample(original_sample, processed_sample)
        else:
            self.log.info("File is small, performing complete validation")
            # Small files are validated directly
            return self._validate_content_sample(original, processed)
    
    def _validate_content_sample(self, original, processed):
        """Validate content samples (internal use)
        
        Args:
            original: original content sample
            processed: processed content sample
        
        Returns:
            tuple: (validation result boolean, detailed report)
        """
        results = {}
        
        # 1. Paragraph count validation
        orig_paragraphs = len(self.re_paragraphs.split(original))
        proc_paragraphs = len(self.re_paragraphs.split(processed))
        para_ratio = proc_paragraphs / max(1, orig_paragraphs)
        results['paragraph_ratio'] = para_ratio
        self.log.debug(f"Paragraph count: original={orig_paragraphs}, processed={proc_paragraphs}, ratio={para_ratio:.2f}")
        
        # 2. Content length validation
        orig_length = len(self.re_whitespace.sub('', original))
        proc_length = len(self.re_whitespace.sub('', processed))
        length_ratio = proc_length / max(1, orig_length)
        results['length_ratio'] = length_ratio
        self.log.debug(f"Content length: original={orig_length}, processed={proc_length}, ratio={length_ratio:.2f}")
        
        # 3. Keyword retention validation
        orig_keywords = set(self.re_keywords.findall(original))
        proc_keywords = set(self.re_keywords.findall(processed))
        
        # Calculate keyword retention rate
        if orig_keywords:
            keyword_retention = len(proc_keywords.intersection(orig_keywords)) / len(orig_keywords)
            self.log.debug(f"Keywords: original={len(orig_keywords)}, retained={len(proc_keywords.intersection(orig_keywords))}, retention rate={keyword_retention:.2%}")
        else:
            keyword_retention = 1.0
            self.log.debug("No keywords found, default retention rate is 100%")
        results['keyword_retention'] = keyword_retention
        
        # 4. Content sampling similarity check
        similarity_scores = []
        # Randomly select several content blocks for similarity comparison
        sample_size = min(3, max(1, orig_paragraphs // 50))
        
        if orig_paragraphs > 5:
            # Select sample paragraphs from original text
            paragraphs = self.re_paragraphs.split(original)
            self.log.debug(f"Selected {sample_size} paragraphs for similarity check")
            
            # Use stratified sampling to ensure samples are selected from different parts of the document
            if len(paragraphs) > sample_size * 3:
                # Stratified sampling
                section_size = len(paragraphs) // sample_size
                samples = [paragraphs[i * section_size] for i in range(sample_size)]
                self.log.debug(f"Using stratified sampling, selecting one paragraph every {section_size} paragraphs")
            else:
                # Random sampling
                samples = random.sample(paragraphs, min(sample_size, len(paragraphs)))
                self.log.debug("Using random sampling")
            
            # Calculate similarity
            for i, sample in enumerate(samples):
                if len(sample) < 20:  # Ignore samples that are too short
                    self.log.debug(f"Sample {i + 1} is too short, skipping")
                    continue
                
                # Calculate best similarity
                best_score = self.calculate_best_similarity(sample, processed)
                self.log.debug(f"Sample {i + 1} best similarity: {best_score:.2%}")
                if best_score > 0:
                    similarity_scores.append(best_score)
        
        # Calculate average sampling similarity
        avg_similarity = sum(similarity_scores) / max(1, len(similarity_scores)) if similarity_scores else 0
        results['sampled_similarity'] = avg_similarity
        self.log.debug(f"Average sampling similarity: {avg_similarity:.2%}")
        
        # Comprehensive scoring and analysis
        # Set weights and thresholds for each dimension
        weights = {
            'paragraph_ratio': 0.3,  # threshold: 0.5-1.5
            'length_ratio': 0.4,  # threshold: 0.7-1.2
            'keyword_retention': 0.2,  # threshold: 0.25 (lower requirement)
            'sampled_similarity': 0.1,  # threshold: 0.1 (lower requirement)
        }
        
        # Calculate weighted score
        score = 0
        for metric, value in results.items():
            # For ratio-type metrics, if it exceeds 1.5, use 1.5 to calculate the score
            if metric in ['paragraph_ratio', 'length_ratio'] and value > 1.5:
                score += weights.get(metric, 0) * 1.5
            else:
                score += weights.get(metric, 0) * value
        
        self.log.debug(f"Content integrity weighted score: {min(score, 1.0) * 100:.2f}%")
        
        # Generate detailed report
        report = [
            f"Content integrity score: {min(score, 1.0) * 100:.2f}%",
            f"- Paragraph retention ratio: {min(results['paragraph_ratio'], 1.5):.2f} (ideal: 0.5-1.5)",
            f"- Content length ratio: {min(results['length_ratio'], 1.5):.2f} (ideal: 0.7-1.2)",
            f"- Keyword retention rate: {results['keyword_retention']:.2%} (ideal: >25%)",
            f"- Content sampling similarity: {results['sampled_similarity']:.2%} (ideal: >10%)"
        ]
        
        # Final determination - using more relaxed standards, suitable for simplified conversion
        if (results['length_ratio'] < 0.6 or
                results['keyword_retention'] < 0.25):  # significantly reduced keyword retention requirement
            self.log.warning("Content validation failed")
            return False, '\n'.join(report)
        else:
            self.log.info("Content validation passed")
            return True, '\n'.join(report)
    
    def calculate_best_similarity(self, sample, processed_text):
        """Calculate the best similarity between sample and processed text
        
        Args:
            sample: Sample text
            processed_text: Complete processed text
        
        Returns:
            float: Best similarity score
        """
        # For large texts, only use the first and last 100KB for similarity calculation
        if len(processed_text) > 200 * 1024:
            self.log.debug("Processed text is large, only using the first and last 100KB for similarity calculation")
            sample_size = 100 * 1024
            proc_start = processed_text[:sample_size]
            proc_end = processed_text[-sample_size:]
            processed_text = proc_start + proc_end
        
        proc_paragraphs = self.re_paragraphs.split(processed_text)
        best_score = 0
        
        # For large documents, use random sampling for comparison
        if len(proc_paragraphs) > 50:
            self.log.debug(f"Processed text has {len(proc_paragraphs)} paragraphs > 50, using random sampling of 50 paragraphs for comparison")
            compare_set = random.sample(proc_paragraphs, 50)
        else:
            self.log.debug(f"Processed text has {len(proc_paragraphs)} paragraphs <= 50, using all paragraphs for comparison")
            compare_set = proc_paragraphs
        
        for pp in compare_set:
            # Use more efficient similarity calculation
            s1 = set(sample.lower().split())
            s2 = set(pp.lower().split())
            if not s1 or not s2:
                continue
            
            # Use Jaccard similarity, faster calculation
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            score = intersection / max(1, union)
            
            best_score = max(best_score, score)
        
        return best_score
    
    def _validate_with_sampling(self, original_samples, processed, removed_hashes=None, file_size=None):
        """Validate large file content using sampling and statistical methods (internal use)
        
        Args:
            original_samples: sample dictionary of original content, keys are position identifiers
            processed: complete processed content
            removed_hashes: hash value dictionary of removed content (optional)
            file_size: original file size (bytes)
            
        Returns:
            tuple: (validation result boolean, detailed report)
        """
        self.log.info("Starting sampling statistical verification...")
        
        # Initialize results collector
        metrics = {
            'paragraph_ratios': [],
            'length_ratios': [],
            'keyword_retentions': [],
            'similarity_scores': [],
            'entropy_changes': [],
            'statistical_significance': []
        }
        
        # Basic statistical information of processed content
        proc_paragraphs = len(self.re_paragraphs.split(processed))
        proc_length = len(self.re_whitespace.sub('', processed))
        proc_keywords = set(self.re_keywords.findall(processed))
        
        # Calculate entropy of processed text
        proc_entropy = self._calculate_entropy(processed)
        
        # Validate each sample
        for position, sample in original_samples.items():
            self.log.debug(f"Validating sample: {position}")
            
            # 1. Paragraph count ratio
            orig_paragraphs = len(self.re_paragraphs.split(sample))
            para_ratio = proc_paragraphs / max(1, orig_paragraphs * (file_size / max(1, len(sample))))
            metrics['paragraph_ratios'].append(min(para_ratio, 1.5))
            
            # 2. Content length ratio
            orig_length = len(self.re_whitespace.sub('', sample))
            length_ratio = proc_length / max(1, orig_length * (file_size / max(1, len(sample))))
            metrics['length_ratios'].append(min(length_ratio, 1.5))
            
            # 3. Keyword retention rate
            orig_keywords_sample = set(self.re_keywords.findall(sample))
            if orig_keywords_sample:
                keyword_retention = len(proc_keywords.intersection(orig_keywords_sample)) / len(orig_keywords_sample)
                metrics['keyword_retentions'].append(keyword_retention)
            
            # 4. Content similarity
            best_score = self.calculate_best_similarity(sample, processed)
            metrics['similarity_scores'].append(best_score)
            
            # 5. Entropy change - measuring information content change
            orig_entropy = self._calculate_entropy(sample)
            entropy_change = proc_entropy / max(0.001, orig_entropy)  # prevent division by zero
            metrics['entropy_changes'].append(min(entropy_change, 1.5))
            
            # 6. Statistical significance - comparing word frequency distributions
            significance = self._calculate_statistical_significance(sample, processed)
            metrics['statistical_significance'].append(significance)
        
        # Calculate average values for each metric
        avg_metrics = {k: sum(v) / max(1, len(v)) for k, v in metrics.items() if v}
        
        # Set weights
        weights = {
            'paragraph_ratios': 0.15,
            'length_ratios': 0.25,
            'keyword_retentions': 0.20,
            'similarity_scores': 0.15,
            'entropy_changes': 0.15,
            'statistical_significance': 0.10
        }
        
        # Calculate weighted score
        score = 0
        for metric, value in avg_metrics.items():
            score += weights.get(metric, 0) * value
        
        # Generate detailed report
        report = [
            f"Content integrity score: {min(score, 1.0) * 100:.2f}%",
            f"- Paragraph retention ratio: {avg_metrics.get('paragraph_ratios', 0):.2f} (ideal: 0.5-1.5)",
            f"- Content length ratio: {avg_metrics.get('length_ratios', 0):.2f} (ideal: 0.7-1.2)",
            f"- Keyword retention rate: {avg_metrics.get('keyword_retentions', 0):.2%} (ideal: >25%)",
            f"- Content sampling similarity: {avg_metrics.get('similarity_scores', 0):.2%} (ideal: >10%)",
            f"- Information entropy ratio: {avg_metrics.get('entropy_changes', 0):.2f} (ideal: 0.8-1.2)",
            f"- Statistical significance: {avg_metrics.get('statistical_significance', 0):.2f} (ideal: >0.6)"
        ]
        
        # Final determination - using more comprehensive evaluation criteria
        if (avg_metrics.get('length_ratios', 0) < 0.6 or
                avg_metrics.get('keyword_retentions', 0) < 0.25 or
                avg_metrics.get('statistical_significance', 0) < 0.5):
            self.log.warning("Content validation failed")
            return False, '\n'.join(report)
        else:
            self.log.info("Content validation passed")
            return True, '\n'.join(report)
    
    def _calculate_entropy(self, text):
        """Calculate the information entropy of a text
        
        Args:
            text: text to calculate entropy for
            
        Returns:
            float: information entropy value
        """
        # Remove whitespace characters to focus on actual content
        text = self.re_whitespace.sub('', text)
        if not text:
            return 0
            
        # Calculate character frequency
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
            
        # Calculate entropy
        length = len(text)
        entropy = 0
        for count in freq.values():
            probability = count / length
            entropy -= probability * (math.log(probability) / math.log(2))
            
        return entropy
    
    def _calculate_statistical_significance(self, sample, full_text):
        """Calculate the statistical significance of a sample compared to the full text
        
        Args:
            sample: sample text
            full_text: full text
            
        Returns:
            float: statistical significance score (0-1)
        """
        # Simplify to word frequency distribution comparison
        # Extract words (simple tokenization)
        sample_words = self.re_words.findall(sample.lower())
        full_words = self.re_words.findall(full_text.lower())
        
        if not sample_words or not full_words:
            return 1.0  # If no words, default to perfect match
            
        # Calculate word frequency in sample
        sample_freq = {}
        for word in sample_words:
            if len(word) > 1:  # Ignore single-character words
                sample_freq[word] = sample_freq.get(word, 0) + 1
                
        # Calculate word frequency in full text
        full_freq = {}
        for word in full_words:
            if len(word) > 1:  # Ignore single-character words
                full_freq[word] = full_freq.get(word, 0) + 1
                
        # Calculate frequency vector cosine similarity
        common_words = set(sample_freq.keys()).intersection(set(full_freq.keys()))
        if not common_words:
            return 0.0
            
        # Calculate dot product of frequency vectors
        dot_product = sum(sample_freq[word] * full_freq[word] for word in common_words)
        sample_magnitude = math.sqrt(sum(freq**2 for freq in sample_freq.values()))
        full_magnitude = math.sqrt(sum(freq**2 for freq in full_freq.values()))
        
        if sample_magnitude == 0 or full_magnitude == 0:
            return 0.0
            
        return dot_product / (sample_magnitude * full_magnitude)

class Converter:
    """Converter class, coordinates the work of various components to implement a complete conversion process"""
    
    def __init__(self, config_manager, log_manager, file_handler, text_processor, content_validator, error_handler):
        """Initialize the converter
        
        Args:
            config_manager: configuration manager instance
            log_manager: log manager instance
            file_handler: file handler instance
            text_processor: text processor instance
            content_validator: content validator instance
            error_handler: error handler instance
        """
        self.config = config_manager
        self.log = log_manager
        self.file = file_handler
        self.processor = text_processor
        self.validator = content_validator
        self.error_handler = error_handler
        
        # Initialize semantic processor (delayed import to avoid circular dependency)
        self.semantic_processor = None
    
    def convert(self, input_file, output_file, verify=True, chunk_size=1024 * 1024, max_workers=None, debug=False, enable_semantic_markers=False):
        """Convert info file to plain text, supports large file processing, maintains simple conversion
        
        Args:
            input_file: input file path
            output_file: output file path
            verify: whether to verify content
            chunk_size: chunk size
            max_workers: maximum number of worker threads
            debug: whether to output debug information
            enable_semantic_markers: whether to enable semantic marking feature (disabled by default)
        
        Returns:
            tuple: (success flag, message)
        """
        try:
            # Check if input file exists
            if not os.path.exists(input_file):
                return self.error_handler.handle_error(Exception(f"File not found: {input_file}"), "File check", fatal=True)
            
            # Get total file size for progress display
            file_info = self.file.get_file_info(input_file, self.config.get_config('large_file_threshold'))
            if file_info is None:
                return self.error_handler.handle_error(Exception(f"Unable to get file information for {input_file}"), "File information retrieval", fatal=True)
            
            file_size = file_info['size']
            
            # Dynamically adjust chunk_size
            if file_size > self.config.get_config('large_file_threshold'):
                # Adjust chunk_size based on file size
                if file_size > 1024 * 1024 * 1024:  # larger than 1GB
                    chunk_size = 4 * 1024 * 1024  # 4MB
                elif file_size > 100 * 1024 * 1024:  # larger than 100MB
                    chunk_size = 2 * 1024 * 1024  # 2MB
                
                # Adjust chunk_size based on available memory
                try:
                    import psutil
                    available_memory = psutil.virtual_memory().available
                    # Ensure chunk_size does not exceed 10% of available memory
                    max_chunk_size = available_memory // 10
                    if chunk_size > max_chunk_size:
                        chunk_size = max(max_chunk_size, 512 * 1024)  # minimum not less than 512KB
                        self.log.info(f"Adjusted chunk_size to {chunk_size / 1024 / 1024:.2f} MB based on available memory")
                except ImportError:
                    self.log.debug("psutil library is not available, unable to adjust chunk_size based on available memory")
                
                # Dynamically adjust the number of worker threads
                if max_workers is None:
                    import multiprocessing
                    max_workers = max(1, multiprocessing.cpu_count() - 1)  # reserve one core for the system
                
                self.log.info(f"[Notice] File is large ({file_size / 1024 / 1024:.2f} MB), using chunked parallel processing")
                self.log.info(f"Chunk size: {chunk_size / 1024 / 1024:.2f} MB, number of worker threads: {max_workers}")
                use_chunking = True
            else:
                use_chunking = False
            
            self.log.print_progress(f"Reading file: {input_file}", 1, 6)
            
            # Streamed reading and processing to reduce memory usage
            if use_chunking:
                # Use streamed processing to avoid loading the entire file at once
                original_content_samples = {}  # use dictionary to store sampled content, keys are position identifiers
                processed_chunks = []
                removed_content_hashes = defaultdict(int)
                total_replaced_count = 0
                
                # First stage: reading and basic processing
                with self.file.safe_open_file(input_file) as f:
                    # Determine the total number of chunks
                    total_chunks = (file_size + chunk_size - 1) // chunk_size
                    self.log.info(f"File will be divided into {total_chunks} chunks for processing")
                    
                    # Parallel processing configuration
                    pattern_keys = [
                        "headers", "separator", "menu", "menu_items", "control_chars",
                    ]
                    
                    # Create thread pool
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        chunks = []
                        chunk_positions = []
                        
                        # Read all chunks
                        position = 0
                        self.log.start_progress(total_chunks, "Reading file chunks")
                        chunk_count = 0
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            chunk_positions.append(position)
                            position += len(chunk)
                            chunk_count += 1
                            self.log.update_progress(chunk_count)
                        self.log.end_progress()
                        
                        self.log.info(f"Read {len(chunks)} chunks, total size: {position} bytes")
                        
                        # Submit parallel processing tasks
                        self.log.print_progress("Starting format marker processing", 2, 6)
                        self.log.start_progress(len(chunks), "Processing file chunks")
                        for i, chunk in enumerate(chunks):
                            is_first = (i == 0)
                            is_last = (i == len(chunks) - 1)
                            future = executor.submit(
                                self.processor.process_chunk,
                                chunk,
                                pattern_keys,
                                is_first,
                                is_last
                            )
                            futures.append(future)
                        
                        # Use index mapping to ensure results are in the correct order
                        future_to_index = {future: i for i, future in enumerate(futures)}
                        sorted_results = [None] * len(chunks)
                        
                        # Process results
                        completed = 0
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                # Get the correct index
                                index = future_to_index[future]
                                sorted_results[index] = future.result()
                                completed += 1
                                self.log.update_progress(completed)
                            except Exception as e:
                                return self.error_handler.handle_error(e, "Chunk processing", fatal=True)
                        self.log.end_progress()
                        
                        # Merge results
                        self.log.start_progress(len(sorted_results), "Merging processing results")
                        for i, (processed_chunk, chunk_hashes, replaced_count) in enumerate(sorted_results):
                            processed_chunks.append(processed_chunk)
                            for hash_val, count in chunk_hashes.items():
                                removed_content_hashes[hash_val] += count
                            total_replaced_count += replaced_count
                            
                            # Save more samples for verification, including the beginning, middle, and end parts
                            if i == 0:
                                original_content_samples['start'] = chunks[i]
                            elif i == len(sorted_results) // 2:
                                original_content_samples['middle'] = chunks[i]
                            elif i == len(sorted_results) - 1:
                                original_content_samples['end'] = chunks[i]
                            # Additional random sampling
                            elif i % (max(1, len(chunks) // 10)) == 0:  # approximately sample 10% of chunks
                                original_content_samples[f'sample_{i}'] = chunks[i]
                            
                            self.log.update_progress(i + 1)
                        self.log.end_progress()
                
                # Merge processed chunks
                content = ''.join(processed_chunks)
                self.log.info(f"    Total removed {total_replaced_count} items")
            
            else:
                # Small files are processed directly
                try:
                    with self.file.safe_open_file(input_file) as f:
                        original_content = f.read()
                except IOError as e:
                    return self.error_handler.handle_error(e, "File reading", fatal=True)
                
                # Create a copy of the content for processing
                content = original_content
                original_content_samples = {'full': original_content}  # small files save complete content
                
                self.log.print_progress("Starting format marker processing", 2, 6)
                
                # Content processing - first stage: remove format markers
                pattern_desc = [
                    ("file header information", 'headers'),
                    ("separator", 'separator'),
                    ("menu area", 'menu'),
                    ("menu items", 'menu_items'),
                    ("control characters", 'control_chars'),
                ]
                
                try:
                    # Process format markers
                    content, removed_content_hashes, replaced_count = self.processor.process_format_markers(
                        content, pattern_desc, self.processor.get_tqdm_instance() is not None
                    )
                except Exception as e:
                    return self.error_handler.handle_error(e, "Format marker processing", fatal=True)
                
                self.log.info(f"    Total removed {replaced_count} format items")
            
            # Normalize empty lines - a simple formatting that does not significantly modify the content
            self.log.print_progress("Normalizing document format", 3, 6)
            try:
                content = self.config.get_pattern('empty_lines').sub('\n\n', content)
            except Exception as e:
                return self.error_handler.handle_error(e, "Empty line normalization", fatal=True)
            
            # Apply semantic processing (if enabled)
            if enable_semantic_markers:
                self.log.print_progress("Applying semantic markers", 4, 6)
                try:
                    # Delayed import of semantic processor to avoid circular dependency
                    if self.semantic_processor is None:
                        from semantic_processor import SemanticProcessor
                        self.semantic_processor = SemanticProcessor(self.config, self.log, self.error_handler)
                    
                    # Determine whether to use chunked processing based on content size
                    max_content_size = self.config.get_config('max_content_size')
                    if len(content) > max_content_size:
                        content = self.semantic_processor.process_chunked(content, chunk_size=max_content_size)
                    else:
                        content = self.semantic_processor.process(content)
                except Exception as e:
                    self.log.warning(f"Semantic processing error: {str(e)}, using original content")
                    # Semantic processing failure should not interrupt the entire conversion process
            
            self.log.print_progress(f"Saving to: {output_file}", 5, 6)
            # Save processed text
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except IOError as e:
                return self.error_handler.handle_error(e, "File saving", fatal=True)
            
            # Content verification
            validation_result = None
            if verify:
                self.log.print_progress("Performing content integrity verification", 6, 6)
                
                try:
                    # Improved verification method using multiple statistical indicators
                    if use_chunking:
                        # Large files use sampling verification
                        validation_result = self.validator.validate(
                            original_content_samples,
                            content,
                            removed_content_hashes,
                            file_size,
                        )
                    else:
                        # Small files use complete verification
                        validation_result = self.validator.validate(
                            original_content_samples['full'],
                            content,
                            removed_content_hashes,
                            file_size,
                        )
                    
                    if not validation_result[0]:
                        self.log.warning(f"Warning: Content verification failed - {validation_result[1]}")
                        return False, f"Warning: Content verification failed - {validation_result[1]}"
                except Exception as e:
                    return self.error_handler.handle_error(e, "Content verification", fatal=False, return_value=(False, "Content verification failed, but file processing is complete"))
            else:
                self.log.print_progress("Skipping content verification", 6, 6)
            
            return True, f"Conversion successful!\n  - Content integrity score: {validation_result[1].split('Content integrity score: ')[1].replace('- ', '  - ') if validation_result else ''}"
            
        except Exception as e:
            return self.error_handler.handle_error(e, "Conversion process", fatal=True)

class Application:
    """Application class, handles command-line arguments, initializes and runs the converter"""
    
    def __init__(self):
        """Initialize application"""
        self.config = ConfigManager()
        self.log = LogManager()
        self.file = FileHandler(self.log)
        self.processor = TextProcessor(self.config, self.log)
        self.validator = ContentValidator(self.config, self.log)
        self.error_handler = ErrorHandler(self.log)
        self.converter = Converter(self.config, self.log, self.file, self.processor, self.validator, self.error_handler)
    
    def parse_args(self):
        """Parse command-line arguments
        
        Returns:
            dict: parsed argument dictionary
        """
        if len(sys.argv) < 3:
            print("Usage: python c3po.py input_file.info output_file.txt [--no-verify] [--workers=N] [--debug] [--log=LEVEL] [--enable-semantic-markers]")
            sys.exit(1)
        
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        # Parse optional arguments
        verify = self.config.get_config('verify')
        max_workers = self.config.get_config('max_workers')
        debug = self.config.get_config('debug')
        log_level = self.config.get_config('log_level')
        enable_semantic_markers = self.config.get_config('enable_semantic_markers')
        
        for arg in sys.argv[3:]:
            if arg.startswith('--workers='):
                try:
                    max_workers = int(arg.split('=')[1])
                    if max_workers <= 0:
                        self.log.warning("Number of worker threads must be greater than 0, using default value")
                        max_workers = self.config.get_config('max_workers')
                except ValueError:
                    self.log.warning("Warning: Invalid worker count, using default value")
            elif arg == '--debug':
                debug = True
                log_level = logging.DEBUG
            elif arg == '--no-verify':
                verify = False
            elif arg == '--enable-semantic-markers':
                enable_semantic_markers = True
            elif arg.startswith('--log='):
                level_name = arg.split('=')[1].upper()
                if hasattr(logging, level_name):
                    log_level = getattr(logging, level_name)
                else:
                    self.log.warning(f"Warning: Invalid log level '{level_name}', using default value")
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'verify': verify,
            'max_workers': max_workers,
            'debug': debug,
            'log_level': log_level,
            'enable_semantic_markers': enable_semantic_markers
        }
    
    def run(self):
        """Run application"""
        # Parse arguments
        args = self.parse_args()
        
        # Set log level
        self.log.set_level(args['log_level'])
        
        # Display processing information
        self.log.print_separator()
        self.log.info(f"C3PO - INFO File Conversion Tool v{__version__}")
        self.log.info(f"Input file: {args['input_file']}")
        self.log.info(f"Output file: {args['output_file']}")
        self.log.info(f"Number of worker threads: {args['max_workers'] if args['max_workers'] else 'automatic'}")
        self.log.info(f"Debug mode: {'enabled' if args['debug'] else 'disabled'}")
        self.log.info(f"Content verification: {'enabled' if args['verify'] else 'disabled'}")
        self.log.info(f"Semantic marking: {'enabled' if args['enable_semantic_markers'] else 'disabled'}")
        self.log.info(f"Log level: {logging.getLevelName(args['log_level'])}")
        self.log.print_separator()
        
        # Perform conversion
        start_time = time.time()
        success, message = self.converter.convert(
            args['input_file'],
            args['output_file'],
            verify=args['verify'],
            max_workers=args['max_workers'],
            debug=args['debug'],
            enable_semantic_markers=args['enable_semantic_markers']
        )
        
        # Display results
        elapsed = time.time() - start_time
        self.log.print_separator()
        if success:
            self.log.success(f"✓ {message}")
            self.log.success(f"Processing time: {self.file.format_time(elapsed)}")
        else:
            self.log.error(f"× {message}")
        self.log.print_separator()
        
        # Return status code
        sys.exit(0 if success else 1)

# Create converter instance
config = ConfigManager()
log = LogManager()
file_handler = FileHandler(log)
processor = TextProcessor(config, log)
validator = ContentValidator(config, log)
error_handler = ErrorHandler(log)
converter = Converter(config, log, file_handler, processor, validator, error_handler)

# Replace original function call
def convert_info_to_text_with_progress(input_file, output_file, verify=True, chunk_size=1024 * 1024, max_workers=None, debug=False, enable_semantic_markers=False):
    return converter.convert(input_file, output_file, verify, chunk_size, max_workers, debug, enable_semantic_markers)

def main():
    """Main program entry"""
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
