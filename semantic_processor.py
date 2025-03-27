#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义处理模块 - 负责识别和标记文本中的语义结构

该模块可以识别文本中的表格、代码块、列表等结构，并添加适当的标记。
设计为独立模块，可以方便地启用或禁用。
"""

import re
import logging
import json
from collections import defaultdict

class SemanticProcessor:
    """语义处理器 - 负责识别和标记文本中的语义结构
    
    该模块可以识别文本中的表格、代码块、列表等结构，并添加适当的标记。
    设计为独立模块，可以方便地启用或禁用。
    """
    
    def __init__(self, config, log, error_handler=None):
        """初始化语义处理器
        
        Args:
            config: 配置管理器
            log: 日志管理器
            error_handler: 错误处理器，如果为None则使用内部错误处理
        """
        self.config = config
        self.log = log
        self.error_handler = error_handler
        
        # 获取配置中的启用状态，但默认设为True，因为调用此处理器时通常是有意启用的
        self.enabled = True
        
        # 获取语义标记定义
        self.markers = config.get_semantic_markers()
        
        # 初始化统计信息
        self.stats = {marker_type: 0 for marker_type in self.markers.keys()}
        self.stats['total'] = 0
        
        # 初始化详细统计信息
        self.detailed_stats = {marker_type: [] for marker_type in self.markers.keys()}
    
    def is_enabled(self):
        """检查语义处理是否启用
        
        Returns:
            bool: 是否启用语义处理
        """
        return self.enabled
    
    def enable(self):
        """启用语义处理"""
        self.enabled = True
        self.log.info("语义处理已启用")
    
    def disable(self):
        """禁用语义处理"""
        self.enabled = False
        self.log.info("语义处理已禁用")
    
    def process(self, content, format_type='html'):
        """处理内容，添加语义标记
        
        Args:
            content: 要处理的文本内容
            format_type: 标记格式类型，可选值：'html', 'markdown', 'json'
        
        Returns:
            str: 添加了语义标记的内容
        """
        # 检查是否启用
        if not self.enabled:
            self.log.debug("语义处理已禁用，返回原始内容")
            return content
        
        try:
            # 重置统计信息
            for key in self.stats:
                self.stats[key] = 0
            
            # 重置详细统计信息
            for key in self.detailed_stats:
                self.detailed_stats[key] = []
            
            # 根据不同格式类型处理
            if format_type == 'html':
                result = self._process_html_format(content)
            elif format_type == 'markdown':
                result = self._process_markdown_format(content)
            elif format_type == 'json':
                result = self._process_json_format(content)
            else:
                self.log.warning(f"不支持的标记格式: {format_type}，返回原始内容")
                return content
            
            # 记录处理统计
            self.log.info(f"语义处理完成，共添加 {self.stats['total']} 个标记")
            
            # 记录详细统计信息
            for marker_type, count in self.stats.items():
                if marker_type != 'total' and count > 0:
                    self.log.info(f"- {marker_type}: {count} 个")
                    # 记录前5个示例（如果有）
                    samples = self.detailed_stats[marker_type][:5]
                    for i, sample in enumerate(samples, 1):
                        truncated_text = sample['text'][:50] + ('...' if len(sample['text']) > 50 else '')
                        self.log.debug(f"  示例{i}: {truncated_text}")
            
            return result
        except Exception as e:
            error_msg = f"语义处理出错: {str(e)}"
            if self.error_handler:
                self.error_handler.handle_error(e, "语义处理", fatal=False)
            else:
                self.log.error(error_msg)
            # 出错时返回原始内容
            return content
    
    def process_chunked(self, content, chunk_size=1024*1024):
        """分块处理大文件的语义标记
        
        Args:
            content: 要处理的文本内容
            chunk_size: 分块大小
        
        Returns:
            str: 添加了语义标记的内容
        """
        # 检查是否启用
        if not self.enabled:
            return content
        
        # 如果内容较小，直接处理
        if len(content) <= chunk_size:
            return self.process(content)
        
        try:
            # 分块处理
            self.log.info(f"内容较大 ({len(content)/1024/1024:.2f} MB)，使用分块处理")
            
            # 计算块数
            total_chunks = (len(content) + chunk_size - 1) // chunk_size
            self.log.debug(f"将分为 {total_chunks} 个块进行处理")
            
            # 保存原始统计信息
            original_stats = self.stats.copy()
            original_detailed_stats = {k: v.copy() for k, v in self.detailed_stats.items()}
            
            # 重置统计信息
            for key in self.stats:
                self.stats[key] = 0
            for key in self.detailed_stats:
                self.detailed_stats[key] = []
            
            # 处理每个块
            processed_chunks = []
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                
                # 保存当前统计信息
                current_stats = self.stats.copy()
                current_detailed_stats = {k: v.copy() for k, v in self.detailed_stats.items()}
                
                # 处理当前块
                processed_chunk = self.process(chunk)
                processed_chunks.append(processed_chunk)
                
                # 记录每个块的处理结果
                self.log.info(f"语义处理完成，共添加 {self.stats['total'] - current_stats['total']} 个标记")
                
                # 调整详细统计信息中的位置偏移
                for marker_type in self.detailed_stats:
                    for item in self.detailed_stats[marker_type][len(current_detailed_stats[marker_type]):]:
                        item['start'] += i
                        item['end'] += i
            
            # 合并处理后的块
            result = ''.join(processed_chunks)
            
            # 生成总体报告
            self.log.info(f"所有块处理完成，总计添加 {self.stats['total']} 个标记")
            
            return result
        except Exception as e:
            error_msg = f"分块语义处理出错: {str(e)}"
            if self.error_handler:
                self.error_handler.handle_error(e, "分块语义处理", fatal=False)
            else:
                self.log.error(error_msg)
            # 出错时返回原始内容
            return content
    
    def _process_html_format(self, content):
        """使用HTML格式处理内容
        
        Args:
            content: 要处理的文本内容
        
        Returns:
            str: 添加了HTML标记的内容
        """
        marked_content = content
        
        # 处理每种类型的标记
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag']
            
            # 使用迭代器查找匹配项
            matches = list(pattern.finditer(marked_content))
            if matches:
                self.log.debug(f"找到 {len(matches)} 个 {marker_type} 标记")
                
                # 从后向前替换，避免位置偏移问题
                for match in reversed(matches):
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # 添加HTML标记
                    marked_text = f"<{tag}>{match_text}</{tag}>"
                    marked_content = marked_content[:start] + marked_text + marked_content[end:]
                    
                    # 更新统计信息
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # 记录详细统计信息
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        return marked_content
    
    def _process_markdown_format(self, content):
        """使用Markdown格式处理内容
        
        Args:
            content: 要处理的文本内容
        
        Returns:
            str: 添加了Markdown标记的内容
        """
        marked_content = content
        
        # 处理每种类型的标记
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag'].lower()
            
            # 使用迭代器查找匹配项
            matches = list(pattern.finditer(marked_content))
            if matches:
                self.log.debug(f"找到 {len(matches)} 个 {marker_type} 标记")
                
                # 从后向前替换，避免位置偏移问题
                for match in reversed(matches):
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # 添加Markdown标记
                    if tag == 'code':
                        marked_text = f"```\n{match_text}\n```"
                    elif tag == 'heading':
                        marked_text = f"{match_text}\n{'=' * len(match_text.strip())}"
                    elif tag == 'table':
                        # 表格已经是Markdown格式，保持不变
                        marked_text = match_text
                    else:
                        # 其他类型添加注释
                        marked_text = f"<!-- {tag}_start -->\n{match_text}\n<!-- {tag}_end -->"
                    
                    marked_content = marked_content[:start] + marked_text + marked_content[end:]
                    
                    # 更新统计信息
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # 记录详细统计信息
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        return marked_content
    
    def _process_json_format(self, content):
        """提取内容的结构化信息，以JSON格式返回
        
        Args:
            content: 要处理的文本内容
        
        Returns:
            str: JSON格式的结构化内容
        """
        structure = {
            "content": content,
            "semantic_elements": []
        }
        
        # 处理每种类型的标记
        for marker_type, marker_info in self.markers.items():
            pattern = marker_info['pattern']
            tag = marker_info['tag']
            
            # 使用迭代器查找匹配项
            matches = list(pattern.finditer(content))
            if matches:
                self.log.debug(f"找到 {len(matches)} 个 {marker_type} 标记")
                
                for match in matches:
                    start, end = match.span()
                    match_text = match.group(0)
                    
                    # 添加到结构化信息
                    element = {
                        "type": tag,
                        "text": match_text,
                        "start": start,
                        "end": end
                    }
                    structure["semantic_elements"].append(element)
                    
                    # 更新统计信息
                    self.stats[marker_type] += 1
                    self.stats['total'] += 1
                    
                    # 记录详细统计信息
                    self.detailed_stats[marker_type].append({
                        'text': match_text,
                        'start': start,
                        'end': end
                    })
        
        # 按位置排序
        structure["semantic_elements"].sort(key=lambda x: x["start"])
        
        return json.dumps(structure, ensure_ascii=False, indent=2)
    
    def get_stats(self):
        """获取处理统计信息
        
        Returns:
            dict: 处理统计信息
        """
        return self.stats
    
    def generate_report(self):
        """生成语义处理的详细统计报告
        
        Returns:
            str: 格式化的统计报告
        """
        if self.stats['total'] == 0:
            return "未进行语义处理或未找到任何语义结构"
        
        report = []
        report.append("语义处理统计报告")
        report.append("=" * 20)
        report.append(f"总计添加标记: {self.stats['total']} 个")
        report.append("")
        
        # 按数量排序，从多到少
        sorted_stats = sorted(
            [(k, v) for k, v in self.stats.items() if k != 'total' and v > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        for marker_type, count in sorted_stats:
            report.append(f"{marker_type}: {count} 个")
        
        return "\n".join(report)
