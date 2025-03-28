#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Processing Module - Responsible for identifying and marking semantic structures in text

This module can identify structures such as tables, code blocks, lists, etc. in text and add appropriate markings.
Designed as an independent module that can be easily enabled or disabled.
"""

import re
import logging
import json
from collections import defaultdict

class SemanticProcessor:
    """Semantic Processor - Responsible for identifying and marking semantic structures in text
    
    This module can identify structures such as tables, code blocks, lists, etc. in text and add appropriate markings.
    Designed as an independent module that can be easily enabled or disabled.
    """
    
    def __init__(self, config, log, error_handler=None):
        """Initialize semantic processor
        
        Args:
            config: Configuration manager
            log: Log manager
            error_handler: Error handler, uses internal error handling if None
        """
        self.config = config
        self.log = log
        self.error_handler = error_handler
        
        # Get enabled status from config, but default to True since calling this processor is usually intentional
        self.enabled = True
        
        # Get semantic marker definitions
        self.markers = config.get_semantic_markers()
        
        # Initialize statistics
        self.stats = {marker_type: 0 for marker_type in self.markers.keys()}
        self.stats['total'] = 0
        
        # Initialize detailed statistics
        self.detailed_stats = {marker_type: [] for marker_type in self.markers.keys()}
    
    def is_enabled(self):
        """Check if semantic processing is enabled
        
        Returns:
            bool: Whether semantic processing is enabled
        """
        return self.enabled
    
    def enable(self):
        """Enable semantic processing"""
        self.enabled = True
        self.log.info("Semantic processing enabled")
    
    def disable(self):
        """Disable semantic processing"""
        self.enabled = False
        self.log.info("Semantic processing disabled")
    
    def process(self, content, format_type='html'):
        """Process content, add semantic markings
        
        Args:
            content: Text content to process
            format_type: Marking format type, options: 'html', 'markdown', 'json'
            
        Returns:
            str: Content with semantic markings added
        """
        # Check if enabled
        if not self.enabled:
            self.log.debug("Semantic processing disabled, returning original content")
            return content
        
        try:
            # Reset statistics
            for key in self.stats:
                self.stats[key] = 0
            
            # Reset detailed statistics
            for key in self.detailed_stats:
                self.detailed_stats[key] = []
            
            # Process according to different format types
            if format_type == 'html':
                result = self._process_html_format(content)
            elif format_type == 'markdown':
                result = self._process_markdown_format(content)
            elif format_type == 'json':
                result = self._process_json_format(content)
            else:
                self.log.warning(f"Unsupported markup format: {format_type}, returning original content")
                return content
            
            # Record processing statistics
            self.log.info(f"Semantic processing completed, added {self.stats['total']} markers")
            
            # Record detailed statistics
            for marker_type, count in self.stats.items():
                if marker_type != 'total' and count > 0:
                    self.log.info(f"- {marker_type}: {count} markers")
                    # Record first 5 examples (if available)
                    samples = self.detailed_stats[marker_type][:5]
                    for i, sample in enumerate(samples, 1):
                        truncated_text = sample['text'][:50] + ('...' if len(sample['text']) > 50 else '')
                        self.log.debug(f"  Example {i}: {truncated_text}")
            
            return result
        except Exception as e:
            error_msg = f"Semantic processing error: {str(e)}"
            if self.error_handler:
                self.error_handler.handle_error(e, "Semantic processing", fatal=False)
            else:
                self.log.error(error_msg)
            # Return original content on error
            return content
    
    def process_chunked(self, content, chunk_size=1024*1024):
        """Process semantic markings for large files in chunks
        
        Args:
            content: Text content to process
            chunk_size: Chunk size
            
        Returns:
            str: Content with semantic markings added
        """
        # Check if enabled
        if not self.enabled:
            return content
        
        try:
            # If content is small, process directly
            if len(content) <= chunk_size:
                return self.process(content)
            
            # Process in chunks
            self.log.info(f"Content is large ({len(content)/1024/1024:.2f} MB), using chunked processing")
            
            # Calculate number of chunks
            total_chunks = (len(content) + chunk_size - 1) // chunk_size
            self.log.debug(f"Divided into {total_chunks} chunks for processing")
            
            # Save original statistics
            original_stats = self.stats.copy()
            original_detailed_stats = {k: v.copy() for k, v in self.detailed_stats.items()}
            
            # Reset statistics
            for key in self.stats:
                self.stats[key] = 0
            for key in self.detailed_stats:
                self.detailed_stats[key] = []
            
            # Process each chunk
            processed_chunks = []
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                
                # Save current statistics
                current_stats = self.stats.copy()
                current_detailed_stats = {k: v.copy() for k, v in self.detailed_stats.items()}
                
                # Process current chunk
                processed_chunk = self.process(chunk)
                processed_chunks.append(processed_chunk)
                
                # Record each chunk's processing results
                self.log.info(f"Semantic processing completed, added {self.stats['total'] - current_stats['total']} markers")
                
                # Adjust position offsets in detailed statistics
                for marker_type in self.detailed_stats:
                    for item in self.detailed_stats[marker_type][len(current_detailed_stats[marker_type]):]:
                        item['start'] += i
                        item['end'] += i
            
            # Merge processed chunks
            result = ''.join(processed_chunks)
            
            # Generate overall report
            self.log.info(f"All chunks processed, total {self.stats['total']} markers added")
            
            return result
        except Exception as e:
            error_msg = f"Chunked semantic processing error: {str(e)}"
            if self.error_handler:
                self.error_handler.handle_error(e, "Chunked semantic processing", fatal=False)
            else:
                self.log.error(error_msg)
            # Return original content on error
            return content
    
    def _process_html_format(self, content):
        """Process content using HTML format
        
        Args:
            content: Text content to process
            
        Returns:
            str: Content with HTML markings added
        """
        marked_content = content
        
        # Process each type of marking
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag']
            
            # Use iterator to find matches
            matches = list(pattern.finditer(marked_content))
            if matches:
                self.log.debug(f"Found {len(matches)} {marker_type} markers")
                
                # Replace from back to front to avoid position offset issues
                for match in reversed(matches):
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # Add HTML markings
                    marked_text = f"<{tag}>{match_text}</{tag}>"
                    marked_content = marked_content[:start] + marked_text + marked_content[end:]
                    
                    # Update statistics
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # Record detailed statistics
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        return marked_content
    
    def _process_markdown_format(self, content):
        """Process content using Markdown format
        
        Args:
            content: Text content to process
            
        Returns:
            str: Content with Markdown markings added
        """
        marked_content = content
        
        # Process each type of marking
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag'].lower()
            
            # Use iterator to find matches
            matches = list(pattern.finditer(marked_content))
            if matches:
                self.log.debug(f"Found {len(matches)} {marker_type} markers")
                
                # Replace from back to front to avoid position offset issues
                for match in reversed(matches):
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # Add Markdown markings
                    if tag == 'code':
                        marked_text = f"```\n{match_text}\n```"
                    elif tag == 'heading':
                        marked_text = f"{match_text}\n{'=' * len(match_text.strip())}"
                    elif tag == 'table':
                        # Tables are already in Markdown format, keep unchanged
                        marked_text = match_text
                    else:
                        # Add comments for other types
                        marked_text = f"<!-- {tag}_start -->\n{match_text}\n<!-- {tag}_end -->"
                    
                    marked_content = marked_content[:start] + marked_text + marked_content[end:]
                    
                    # Update statistics
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # Record detailed statistics
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        return marked_content
    
    def _process_json_format(self, content):
        """Extract structured information from content, return in JSON format
        
        Args:
            content: Text content to process
            
        Returns:
            str: Structured content in JSON format
        """
        structure = {
            "content": content,
            "semantic_elements": []
        }
        
        # Process each type of marking
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag']
            
            # Use iterator to find matches
            matches = list(pattern.finditer(content))
            if matches:
                self.log.debug(f"Found {len(matches)} {marker_type} markers")
                
                for match in matches:
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # Add to structured information
                    element = {
                        "type": tag,
                        "text": match_text,
                        "start": start,
                        "end": end
                    }
                    structure["semantic_elements"].append(element)
                    
                    # Update statistics
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # Record detailed statistics
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        # Sort by position
        structure["semantic_elements"].sort(key=lambda x: x["start"])
        
        return json.dumps(structure, ensure_ascii=False, indent=2)
    
    def get_stats(self):
        """Get processing statistics
        
        Returns:
            dict: Processing statistics
        """
        return self.stats
    
    def generate_report(self):
        """Generate detailed statistical report for semantic processing
        
        Returns:
            str: Formatted statistical report
        """
        if self.stats['total'] == 0:
            return "No semantic processing or no semantic structures found"
        
        report = []
        report.append("Semantic Processing Statistics Report")
        report.append("=" * 30)
        report.append(f"Total markers added: {self.stats['total']}")
        report.append("")
        
        # Sort by count, from most to least
        sorted_stats = sorted(
            [(k, v) for k, v in self.stats.items() if k != 'total' and v > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        for marker_type, count in sorted_stats:
            report.append(f"{marker_type}: {count} markers")
        
        return "\n".join(report)
