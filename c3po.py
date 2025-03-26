"""
C3PO - INFO文件转换工具
将INFO格式文件转换为纯文本格式，保持干净的从源文件到输出文件的转换流程。

功能：
- 移除文件头信息、分隔线、菜单区域等格式标记
- 保留文档核心内容
- 提供内容验证功能，确保转换质量
- 支持大文件并行处理

用法: python c3po.py 输入文件.info 输出文件.txt [--no-verify] [--workers=N] [--debug] [--log=LEVEL]

参数:
  --no-verify: 跳过内容验证
  --workers=N: 设置工作线程数
  --debug: 启用调试模式
  --log=LEVEL: 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
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

# 检查tqdm库是否可用
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 版本信息
__version__ = "1.0.0"
__date__ = "2025-03-27"
__author__ = "C3PO Team"

class ErrorHandler:
    """错误处理类，提供统一的错误处理机制"""
    
    def __init__(self, log_manager):
        """初始化错误处理器
        
        Args:
            log_manager: 日志管理器实例
        """
        self.log = log_manager
    
    def handle_error(self, error, operation_name, fatal=False, return_value=None):
        """处理错误
        
        Args:
            error: 异常对象
            operation_name: 操作名称
            fatal: 是否为致命错误
            return_value: 如果不是致命错误，返回的值
            
        Returns:
            如果不是致命错误，返回return_value；否则抛出异常
        """
        error_msg = f"{operation_name}错误: {str(error)}"
        
        if fatal:
            error_msg += f"\n{traceback.format_exc()}"
            self.log.error(error_msg)
            raise error
        else:
            self.log.warning(error_msg)
            return return_value
    
    def try_operation(self, operation, operation_name, args=None, kwargs=None, fatal=False, return_value=None):
        """尝试执行操作，处理可能的异常
        
        Args:
            operation: 要执行的函数
            operation_name: 操作名称
            args: 位置参数
            kwargs: 关键字参数
            fatal: 是否为致命错误
            return_value: 如果不是致命错误，返回的值
            
        Returns:
            操作结果或return_value
        """
        args = args or ()
        kwargs = kwargs or {}
        
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            return self.handle_error(e, operation_name, fatal, return_value)

class ConfigManager:
    """配置管理类，负责管理所有配置参数和正则表达式模式"""
    
    def __init__(self, config_file=None):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径（可选）
        """
        # 默认配置
        self.default_config = {
            'chunk_size': 1024 * 1024,  # 1MB
            'large_file_threshold': 10 * 1024 * 1024,  # 10MB
            'max_workers': None,  # 默认使用CPU核心数
            'verify': True,  # 默认启用内容验证
            'debug': False,  # 默认关闭调试模式
            'log_level': logging.INFO,  # 默认日志级别
            'enable_semantic_markers': False,  # 默认禁用语义标记
            'encoding': 'utf-8',  # 默认编码
            'fallback_encoding': 'latin1',  # 备用编码
            'max_content_size': 10 * 1024 * 1024,  # 最大内容大小
            'max_matches_per_type': 1000,  # 每种类型最大匹配数量
            'validation_sample_size': 100 * 1024,  # 验证样本大小
            'min_chunk_size': 512 * 1024,  # 最小块大小
            'max_chunk_size_memory_ratio': 0.1,  # 块大小占可用内存的最大比例
            'color_enabled': True,  # 是否启用彩色日志
            'progress_enabled': True,  # 是否启用进度显示
        }
        
        # 用户配置
        self.user_config = {}
        
        # 加载配置文件
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # 正则表达式模式
        self.regex_patterns = {
            # 文件头信息
            'headers': re.compile(r'^\s*\*{3}.*?\*{3}\s*$', re.MULTILINE),

            # 分隔线
            'separator': re.compile(r'^\s*[-=_]{3,}\s*$', re.MULTILINE),

            # 菜单区域
            'menu': re.compile(r'^\s*\[.*?菜单.*?\]\s*$.*?(?=\n\n)', re.MULTILINE | re.DOTALL),

            # 菜单条目 - 保留条目名称
            'menu_items': re.compile(r'^\s*\[([^]]+)\].*?(?=\n\n|\Z)', re.MULTILINE | re.DOTALL),

            # 控制字符
            'control_chars': re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'),

            # 空行规范化
            'empty_lines': re.compile(r'\n{3,}'),
        }
        
        # SQL关键词 - 用于识别代码块
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
            'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN',
            'INSERT INTO', 'UPDATE', 'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE',
            'DROP TABLE', 'TRUNCATE TABLE', 'BEGIN TRANSACTION', 'COMMIT', 'ROLLBACK'
        ]
        
        # 语义标记 - 用于增强文档结构
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
        """从配置文件加载配置
        
        Args:
            config_file: 配置文件路径
        """
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                self.user_config = json.load(f)
        except Exception as e:
            print(f"警告: 加载配置文件失败: {str(e)}")
    
    def get_config(self, key, default=None):
        """获取配置值
        
        Args:
            key: 配置键名
            default: 默认值，如果键不存在
            
        Returns:
            配置值
        """
        # 优先使用用户配置，然后是默认配置，最后是提供的默认值
        return self.user_config.get(key, self.default_config.get(key, default))
    
    def set_config(self, key, value):
        """设置配置值
        
        Args:
            key: 配置键名
            value: 配置值
        """
        self.user_config[key] = value
    
    def save_config(self, config_file):
        """保存配置到文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, indent=4)
            return True
        except Exception as e:
            print(f"警告: 保存配置文件失败: {str(e)}")
            return False
    
    def get_pattern(self, key):
        """获取正则表达式模式
        
        Args:
            key: 模式键名
            
        Returns:
            正则表达式模式
        """
        return self.regex_patterns.get(key)
    
    def get_all_patterns(self):
        """获取所有正则表达式模式
        
        Returns:
            所有正则表达式模式的字典
        """
        return self.regex_patterns
    
    def get_semantic_marker(self, key):
        """获取语义标记
        
        Args:
            key: 标记键名
            
        Returns:
            语义标记
        """
        return self.semantic_markers.get(key)
    
    def get_all_semantic_markers(self):
        """获取所有语义标记
        
        Returns:
            所有语义标记的字典
        """
        return self.semantic_markers

class LogManager:
    """日志管理类，封装日志系统的设置和使用"""
    
    # ANSI颜色代码
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
        """初始化日志管理器
        
        Args:
            level: 日志级别
        """
        self.logger = self._setup_logging(level)
        self.progress_start_time = None
        self.progress_last_update = None
        self.progress_total = None
        self.progress_current = None
        self.progress_enabled = True
        self.color_enabled = True
        
        # 检测是否在终端环境中运行
        self.is_terminal = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _setup_logging(self, level):
        """配置日志系统
        
        Args:
            level: 日志级别
            
        Returns:
            日志对象
        """
        # 创建日志格式
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 配置日志
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.StreamHandler()  # 输出到控制台
            ]
        )
        
        # 返回日志对象
        return logging.getLogger('c3po')
    
    def _colorize(self, text, color):
        """为文本添加颜色
        
        Args:
            text: 要着色的文本
            color: 颜色代码
            
        Returns:
            str: 着色后的文本
        """
        if not self.color_enabled or not self.is_terminal:
            return text
            
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['RESET']}"
    
    def set_level(self, level):
        """设置日志级别
        
        Args:
            level: 日志级别
        """
        self.logger.setLevel(level)
    
    def enable_color(self, enabled=True):
        """启用或禁用彩色日志
        
        Args:
            enabled: 是否启用
        """
        self.color_enabled = enabled
    
    def debug(self, message):
        """记录调试信息
        
        Args:
            message: 日志消息
        """
        self.logger.debug(self._colorize(message, 'CYAN'))
    
    def info(self, message):
        """记录一般信息
        
        Args:
            message: 日志消息
        """
        self.logger.info(self._colorize(message, 'WHITE'))
    
    def warning(self, message):
        """记录警告信息
        
        Args:
            message: 日志消息
        """
        self.logger.warning(self._colorize(message, 'YELLOW'))
    
    def error(self, message):
        """记录错误信息
        
        Args:
            message: 日志消息
        """
        self.logger.error(self._colorize(message, 'RED'))
    
    def critical(self, message):
        """记录严重错误信息
        
        Args:
            message: 日志消息
        """
        self.logger.critical(self._colorize(f"{self.COLORS['BOLD']}{message}", 'RED'))
    
    def success(self, message):
        """记录成功信息
        
        Args:
            message: 日志消息
        """
        self.logger.info(self._colorize(message, 'GREEN'))
    
    def print_progress(self, message, step=None, total_steps=None):
        """打印进度信息
        
        Args:
            message: 进度消息
            step: 当前步骤
            total_steps: 总步骤数
        """
        if step is not None and total_steps is not None:
            log_msg = f"[{step}/{total_steps}] {message}"
        else:
            log_msg = message
        
        self.info(log_msg)
    
    def start_progress(self, total, message="处理进度"):
        """开始进度跟踪
        
        Args:
            total: 总数量
            message: 进度消息前缀
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
        """更新进度
        
        Args:
            current: 当前数量
            increment: 增量
        """
        if not self.progress_enabled or self.progress_total is None:
            return
            
        # 更新当前进度
        if current is not None:
            self.progress_current = current
        elif increment is not None:
            self.progress_current += increment
        
        # 计算进度百分比
        percent = self.progress_current / self.progress_total * 100
        
        # 控制更新频率，避免过多输出
        current_time = time.time()
        if current_time - self.progress_last_update < 0.5 and percent < 100:
            return
            
        self.progress_last_update = current_time
        
        # 计算经过的时间和预估剩余时间
        elapsed = current_time - self.progress_start_time
        if self.progress_current > 0:
            remaining = elapsed * (self.progress_total - self.progress_current) / self.progress_current
            eta = self._format_time(remaining)
            elapsed_str = self._format_time(elapsed)
            
            # 构建进度消息
            msg = f"{self.progress_message}: {percent:.1f}% ({self.progress_current}/{self.progress_total}) - 已用时间: {elapsed_str}, 预计剩余: {eta}"
        else:
            msg = f"{self.progress_message}: {percent:.1f}% ({self.progress_current}/{self.progress_total})"
        
        self.info(msg)
    
    def end_progress(self, message=None):
        """结束进度跟踪
        
        Args:
            message: 结束消息
        """
        if not self.progress_enabled or self.progress_total is None:
            return
            
        elapsed = time.time() - self.progress_start_time
        elapsed_str = self._format_time(elapsed)
        
        if message:
            self.info(f"{message} - 总用时: {elapsed_str}")
        else:
            self.info(f"{self.progress_message}完成 - 总用时: {elapsed_str}")
        
        # 重置进度状态
        self.progress_start_time = None
        self.progress_last_update = None
        self.progress_total = None
        self.progress_current = None
    
    def _format_time(self, seconds):
        """格式化时间
        
        Args:
            seconds: 秒数
            
        Returns:
            str: 格式化后的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.1f}小时{minutes:.0f}分钟"
    
    def print_separator(self, char='-', length=60):
        """打印分隔线
        
        Args:
            char: 分隔字符
            length: 分隔线长度
        """
        self.info(char * length)
    
    def enable_progress(self, enabled=True):
        """启用或禁用进度显示
        
        Args:
            enabled: 是否启用
        """
        self.progress_enabled = enabled

class FileHandler:
    """文件处理类，处理文件读写操作"""
    
    def __init__(self, log_manager):
        """初始化文件处理器
        
        Args:
            log_manager: 日志管理器实例
        """
        self.log = log_manager
    
    def get_file_info(self, file_path, large_file_threshold):
        """获取文件信息
        
        Args:
            file_path: 文件路径
            large_file_threshold: 大文件阈值
        
        Returns:
            dict: 包含文件大小、修改时间等信息的字典，如果文件不存在则返回None
        """
        if not os.path.exists(file_path):
            self.log.error(f"文件不存在: {file_path}")
            return None
        
        stats = os.stat(file_path)
        info = {
            'size': stats.st_size,
            'modified': stats.st_mtime,
            'is_large': stats.st_size > large_file_threshold
        }
        
        self.log.debug(f"文件信息: {file_path}, 大小: {self.format_size(info['size'])}, 最后修改: {time.ctime(info['modified'])}")
        return info
    
    def safe_open_file(self, file_path, mode='r', encoding='utf-8'):
        """安全打开文件，处理编码错误
        
        Args:
            file_path: 文件路径
            mode: 打开模式
            encoding: 编码方式
        
        Returns:
            file: 文件对象
        
        Raises:
            IOError: 如果文件无法打开或读取
        """
        try:
            self.log.debug(f"尝试使用 {encoding} 编码打开文件: {file_path}")
            return open(file_path, mode=mode, encoding=encoding, errors='replace')
        except UnicodeDecodeError:
            self.log.warning(f"使用 {encoding} 编码打开文件失败，尝试使用 Latin-1 编码...")
            return open(file_path, mode=mode, encoding='latin1', errors='replace')
        except Exception as e:
            self.log.error(f"打开文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def create_string_buffer(self):
        """创建字符串缓冲区
        
        Returns:
            StringIO: 字符串缓冲区对象
        """
        return io.StringIO()
    
    def format_size(self, size_bytes):
        """格式化文件大小
        
        Args:
            size_bytes: 文件大小（字节）
        
        Returns:
            str: 格式化后的大小字符串
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
        """格式化时间
        
        Args:
            seconds: 秒数
        
        Returns:
            str: 格式化后的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f} 秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} 分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} 小时"

class TextProcessor:
    """文本处理类，处理文本块和格式标记"""
    
    def __init__(self, config_manager, log_manager):
        """初始化文本处理器
        
        Args:
            config_manager: 配置管理器实例
            log_manager: 日志管理器实例
        """
        self.config = config_manager
        self.log = log_manager
    
    def process_chunk(self, chunk, patterns, is_first_chunk=False, is_last_chunk=False):
        """处理单个文本块，用于并行处理
        
        Args:
            chunk: 要处理的文本块
            patterns: 要应用的正则表达式模式列表
            is_first_chunk: 是否为第一个块
            is_last_chunk: 是否为最后一个块
        
        Returns:
            tuple: (处理后的内容, 移除内容的哈希值字典, 替换计数)
        """
        content = chunk
        removed_content_hashes = defaultdict(int)
        replaced_count = 0
        
        # 只处理完整的段落，避免在块边界切断内容
        if not is_first_chunk and not is_last_chunk:
            # 找到第一个完整段落的开始
            first_para_start = content.find('\n\n')
            if first_para_start > 0:
                content = content[first_para_start + 2:]
                self.log.debug(f"截取块的第一个完整段落，移除了 {first_para_start + 2} 个字符")
            
            # 找到最后一个完整段落的结束
            last_para_end = content.rfind('\n\n')
            if last_para_end > 0:
                content = content[:last_para_end]
                self.log.debug(f"截取块的最后一个完整段落，保留了 {last_para_end} 个字符")
        
        # 应用格式处理
        for pattern_key in patterns:
            pattern = self.config.get_pattern(pattern_key)
            
            # 使用迭代器而不是一次性加载所有匹配
            for match in pattern.finditer(content):
                match_text = match.group(0)
                match_hash = hashlib.md5(str(match_text).encode()).hexdigest()
                removed_content_hashes[match_hash] += 1
                replaced_count += 1
            
            # 执行替换
            if pattern_key == 'menu_items':
                # 特殊处理菜单项，保留条目名
                content = pattern.sub(r'\1', content)
            else:
                content = pattern.sub('', content)
        
        self.log.debug(f"块处理完成，移除了 {replaced_count} 项内容")
        return content, removed_content_hashes, replaced_count
    
    def process_format_markers(self, content, pattern_desc, tqdm_available=False):
        """处理格式标记
        
        Args:
            content: 要处理的文本内容
            pattern_desc: 要应用的模式描述列表，格式为 [(描述, 模式键名), ...]
            tqdm_available: tqdm库是否可用
        
        Returns:
            tuple: (处理后的内容, 移除内容的哈希值字典, 替换计数)
        """
        # 使用字典记录移除的内容摘要信息，而非存储完整内容
        removed_content_hashes = defaultdict(int)
        replaced_count = 0
        
        # 按顺序处理不同类型的内容
        pattern_iter = pattern_desc
        if tqdm_available:
            from tqdm import tqdm as tqdm_lib
            pattern_iter = tqdm_lib(pattern_iter, desc="    移除格式标记")
        
        for desc, pattern_key in pattern_iter:
            if not tqdm_available:
                self.log.info(f"    处理 {desc}...")
            
            pattern = self.config.get_pattern(pattern_key)
            matches = pattern.findall(content)
            
            # 记录移除内容的哈希值
            for match in matches:
                match_hash = hashlib.md5(str(match).encode()).hexdigest()
                removed_content_hashes[match_hash] += 1
                replaced_count += 1
            
            # 执行替换
            if pattern_key == 'menu_items':
                # 特殊处理菜单项，保留条目名
                content = pattern.sub(r'\1', content)
            else:
                content = pattern.sub('', content)
        
        self.log.info(f"    总共移除了 {replaced_count} 项格式内容")
        return content, removed_content_hashes, replaced_count
    
    def add_metadata_markers(self, content, max_content_size=10 * 1024 * 1024, max_matches_per_type=1000, debug=False, enable_semantic_markers=False):
        """为内容添加语义元数据标记
        
        Args:
            content: 要处理的文本内容
            max_content_size: 最大内容大小，超过此大小将分块处理
            max_matches_per_type: 每种类型最大匹配数量
            debug: 是否输出调试信息
            enable_semantic_markers: 是否启用语义标记功能（默认禁用）
        
        Returns:
            str: 添加了元数据标记的内容
        """
        # 如果禁用语义标记功能，直接返回原始内容
        if not enable_semantic_markers:
            self.log.debug("语义标记功能已禁用，返回原始内容")
            return content
        
        # 临时禁用所有处理，直接返回原始内容
        if debug:
            self.log.warning("警告：所有语义标记处理已临时禁用，直接返回原始内容")
        
        # 这里保留此函数接口，但实际不执行任何处理，直接返回原始内容
        # 未来可以根据需要实现语义标记功能
        return content
    
    def get_tqdm_instance(self, iterable=None, total=None, desc=None):
        """获取tqdm进度条实例，如果可用
        
        Args:
            iterable: 可迭代对象
            total: 总数
            desc: 描述
        
        Returns:
            tqdm或iterable: tqdm实例或原始可迭代对象
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
    """内容验证类，验证处理前后的内容一致性"""
    
    def __init__(self, config_manager, log_manager):
        """初始化内容验证器
        
        Args:
            config_manager: 配置管理器实例
            log_manager: 日志管理器实例
        """
        self.config = config_manager
        self.log = log_manager
        
        # 预编译正则表达式
        self.re_paragraphs = re.compile(r'\n\s*\n')
        self.re_whitespace = re.compile(r'\s+')
        self.re_keywords = re.compile(r'\b([A-Z]{2,})\b')
        self.re_words = re.compile(r'\b\w+\b')
    
    def validate(self, original, processed, removed_hashes=None, file_size=None, is_large_file=False):
        """统一的内容验证入口
        
        Args:
            original: 原始内容（可以是完整内容或样本字典）
            processed: 处理后的内容
            removed_hashes: 移除内容的哈希值字典（可选）
            file_size: 原始文件大小（字节）
            is_large_file: 是否为大文件
            
        Returns:
            tuple: (验证结果布尔值, 详细报告)
        """
        self.log.info("开始内容验证...")
        
        # 根据文件大小和内容类型选择合适的验证方法
        if is_large_file:
            # 大文件使用采样验证
            if isinstance(original, dict):
                # 已经是样本字典
                return self._validate_with_sampling(original, processed, removed_hashes, file_size)
            else:
                # 需要创建样本
                samples = self._create_samples(original)
                return self._validate_with_sampling(samples, processed, removed_hashes, file_size or len(original))
        else:
            # 小文件使用简单验证
            if isinstance(original, dict):
                # 使用字典中的完整内容
                if 'full' in original:
                    return self._validate_content_sample(original['full'], processed)
                else:
                    # 合并样本
                    combined = ''.join(original.values())
                    return self._validate_content_sample(combined, processed)
            else:
                # 直接使用完整内容
                return self._validate_content_sample(original, processed)
    
    def _create_samples(self, content):
        """从内容创建样本
        
        Args:
            content: 完整内容
            
        Returns:
            dict: 样本字典
        """
        sample_size = self.config.get_config('validation_sample_size', 100 * 1024)
        samples = {}
        
        # 提取开头、中间和结尾部分
        if len(content) <= sample_size * 3:
            # 内容较小，使用完整内容
            samples['full'] = content
        else:
            # 内容较大，提取样本
            samples['start'] = content[:sample_size]
            middle_start = (len(content) - sample_size) // 2
            samples['middle'] = content[middle_start:middle_start + sample_size]
            samples['end'] = content[-sample_size:]
            
            # 随机采样
            for i in range(3):
                random_start = random.randint(0, max(0, len(content) - sample_size))
                samples[f'random_{i}'] = content[random_start:random_start + sample_size]
        
        return samples
    
    # 将原来的public方法改为private方法
    def _simple_validate_content(self, original, processed, removed_hashes=None):
        """简化的内容验证函数（内部使用）
        
        Args:
            original: 原始内容
            processed: 处理后的内容
            removed_hashes: 移除内容的哈希值字典（可选）
        
        Returns:
            tuple: (验证结果布尔值, 详细报告)
        """
        # 对于大文件，只使用样本进行验证
        if len(original) > 1024 * 1024 or len(processed) > 1024 * 1024:
            self.log.info("检测到大文件，使用采样验证方法")
            # 从文件开头和结尾各取100KB进行验证
            sample_size = self.config.get_config('validation_sample_size', 100 * 1024)
            orig_start = original[:sample_size]
            orig_end = original[-sample_size:] if len(original) > sample_size else ""
            proc_start = processed[:sample_size]
            proc_end = processed[-sample_size:] if len(processed) > sample_size else ""
            
            # 合并样本
            original_sample = orig_start + orig_end
            processed_sample = proc_start + proc_end
            
            self.log.debug(f"原始内容样本大小: {len(original_sample)}, 处理后内容样本大小: {len(processed_sample)}")
            
            # 使用样本进行验证
            return self._validate_content_sample(original_sample, processed_sample)
        else:
            self.log.info("文件较小，进行完整验证")
            # 小文件直接验证
            return self._validate_content_sample(original, processed)
    
    def _validate_content_sample(self, original, processed):
        """验证内容样本（内部使用）
        
        Args:
            original: 原始内容样本
            processed: 处理后的内容样本
        
        Returns:
            tuple: (验证结果布尔值, 详细报告)
        """
        results = {}
        
        # 1. 段落数量验证
        orig_paragraphs = len(self.re_paragraphs.split(original))
        proc_paragraphs = len(self.re_paragraphs.split(processed))
        para_ratio = proc_paragraphs / max(1, orig_paragraphs)
        results['paragraph_ratio'] = para_ratio
        self.log.debug(f"段落数量: 原始={orig_paragraphs}, 处理后={proc_paragraphs}, 比例={para_ratio:.2f}")
        
        # 2. 内容长度验证
        orig_length = len(self.re_whitespace.sub('', original))
        proc_length = len(self.re_whitespace.sub('', processed))
        length_ratio = proc_length / max(1, orig_length)
        results['length_ratio'] = length_ratio
        self.log.debug(f"内容长度: 原始={orig_length}, 处理后={proc_length}, 比例={length_ratio:.2f}")
        
        # 3. 关键词保留验证
        orig_keywords = set(self.re_keywords.findall(original))
        proc_keywords = set(self.re_keywords.findall(processed))
        
        # 计算关键词保留率
        if orig_keywords:
            keyword_retention = len(proc_keywords.intersection(orig_keywords)) / len(orig_keywords)
            self.log.debug(f"关键词: 原始={len(orig_keywords)}, 保留={len(proc_keywords.intersection(orig_keywords))}, 保留率={keyword_retention:.2%}")
        else:
            keyword_retention = 1.0
            self.log.debug("未找到关键词，默认保留率为100%")
        results['keyword_retention'] = keyword_retention
        
        # 4. 内容采样相似度检查
        similarity_scores = []
        # 随机选择几个内容块进行相似度比较
        sample_size = min(3, max(1, orig_paragraphs // 50))
        
        if orig_paragraphs > 5:
            # 从原文中选择样本段落
            paragraphs = self.re_paragraphs.split(original)
            self.log.debug(f"选择 {sample_size} 个段落进行相似度检查")
            
            # 使用分层采样，确保从文档不同部分选择样本
            if len(paragraphs) > sample_size * 3:
                # 分层采样
                section_size = len(paragraphs) // sample_size
                samples = [paragraphs[i * section_size] for i in range(sample_size)]
                self.log.debug(f"使用分层采样，每 {section_size} 个段落选择一个")
            else:
                # 随机采样
                samples = random.sample(paragraphs, min(sample_size, len(paragraphs)))
                self.log.debug("使用随机采样")
            
            # 计算相似度
            for i, sample in enumerate(samples):
                if len(sample) < 20:  # 忽略太短的样本
                    self.log.debug(f"样本 {i + 1} 太短，跳过")
                    continue
                
                # 计算最佳相似度
                best_score = self.calculate_best_similarity(sample, processed)
                self.log.debug(f"样本 {i + 1} 最佳相似度: {best_score:.2%}")
                if best_score > 0:
                    similarity_scores.append(best_score)
        
        # 计算平均采样相似度
        avg_similarity = sum(similarity_scores) / max(1, len(similarity_scores)) if similarity_scores else 0
        results['sampled_similarity'] = avg_similarity
        self.log.debug(f"平均采样相似度: {avg_similarity:.2%}")
        
        # 综合评分与分析
        # 为每个维度设置权重和阈值
        weights = {
            'paragraph_ratio': 0.3,  # 阈值: 0.5-1.5
            'length_ratio': 0.4,  # 阈值: 0.7-1.2
            'keyword_retention': 0.2,  # 阈值: 0.25 (降低要求)
            'sampled_similarity': 0.1,  # 阈值: 0.1 (降低要求)
        }
        
        # 计算加权得分
        score = 0
        for metric, value in results.items():
            # 对于比例类指标，如果超过1.5，则使用1.5计算得分
            if metric in ['paragraph_ratio', 'length_ratio'] and value > 1.5:
                score += weights.get(metric, 0) * 1.5
            else:
                score += weights.get(metric, 0) * value
        
        self.log.debug(f"内容完整性加权得分: {min(score, 1.0) * 100:.2f}%")
        
        # 生成详细报告
        report = [
            f"内容完整性评分: {min(score, 1.0) * 100:.2f}%",
            f"- 段落保留比例: {min(results['paragraph_ratio'], 1.5):.2f} (理想: 0.5-1.5)",
            f"- 内容长度比例: {min(results['length_ratio'], 1.5):.2f} (理想: 0.7-1.2)",
            f"- 关键词保留率: {results['keyword_retention']:.2%} (理想: >25%)",
            f"- 内容采样相似度: {results['sampled_similarity']:.2%} (理想: >10%)"
        ]
        
        # 最终判定 - 使用更宽松的标准，适用于简化版转换
        if (results['length_ratio'] < 0.6 or
                results['keyword_retention'] < 0.25):  # 大幅降低关键词保留率要求
            self.log.warning("内容验证失败")
            return False, '\n'.join(report)
        else:
            self.log.info("内容验证通过")
            return True, '\n'.join(report)
    
    def calculate_best_similarity(self, sample, processed_text):
        """计算样本与处理后文本的最佳相似度
        
        Args:
            sample: 样本文本
            processed_text: 处理后的完整文本
        
        Returns:
            float: 最佳相似度分数
        """
        # 对于大文本，只使用前后各100KB进行相似度计算
        if len(processed_text) > 200 * 1024:
            self.log.debug("处理文本较大，只使用前后各100KB进行相似度计算")
            sample_size = 100 * 1024
            proc_start = processed_text[:sample_size]
            proc_end = processed_text[-sample_size:]
            processed_text = proc_start + proc_end
        
        proc_paragraphs = self.re_paragraphs.split(processed_text)
        best_score = 0
        
        # 对于大文档，随机采样进行比较
        if len(proc_paragraphs) > 50:
            self.log.debug(f"处理文本段落数 {len(proc_paragraphs)} > 50，随机采样50个段落进行比较")
            compare_set = random.sample(proc_paragraphs, 50)
        else:
            self.log.debug(f"处理文本段落数 {len(proc_paragraphs)} <= 50，使用全部段落进行比较")
            compare_set = proc_paragraphs
        
        for pp in compare_set:
            # 使用更高效的相似度计算
            s1 = set(sample.lower().split())
            s2 = set(pp.lower().split())
            if not s1 or not s2:
                continue
            
            # 使用Jaccard相似度，计算更快
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            score = intersection / max(1, union)
            
            best_score = max(best_score, score)
        
        return best_score
    
    def _validate_with_sampling(self, original_samples, processed, removed_hashes=None, file_size=None):
        """使用采样和统计学方法验证大文件内容（内部使用）
        
        Args:
            original_samples: 原始内容的样本字典，键为位置标识
            processed: 处理后的完整内容
            removed_hashes: 移除内容的哈希值字典（可选）
            file_size: 原始文件大小（字节）
            
        Returns:
            tuple: (验证结果布尔值, 详细报告)
        """
        self.log.info("开始采样统计验证...")
        
        # 初始化结果收集器
        metrics = {
            'paragraph_ratios': [],
            'length_ratios': [],
            'keyword_retentions': [],
            'similarity_scores': [],
            'entropy_changes': [],
            'statistical_significance': []
        }
        
        # 处理后内容的基本统计信息
        proc_paragraphs = len(self.re_paragraphs.split(processed))
        proc_length = len(self.re_whitespace.sub('', processed))
        proc_keywords = set(self.re_keywords.findall(processed))
        
        # 计算处理后文本的熵
        proc_entropy = self._calculate_entropy(processed)
        
        # 对每个样本进行验证
        for position, sample in original_samples.items():
            self.log.debug(f"验证样本: {position}")
            
            # 1. 段落数量比例
            orig_paragraphs = len(self.re_paragraphs.split(sample))
            para_ratio = proc_paragraphs / max(1, orig_paragraphs * (file_size / max(1, len(sample))))
            metrics['paragraph_ratios'].append(min(para_ratio, 1.5))
            
            # 2. 内容长度比例
            orig_length = len(self.re_whitespace.sub('', sample))
            length_ratio = proc_length / max(1, orig_length * (file_size / max(1, len(sample))))
            metrics['length_ratios'].append(min(length_ratio, 1.5))
            
            # 3. 关键词保留率
            orig_keywords_sample = set(self.re_keywords.findall(sample))
            if orig_keywords_sample:
                keyword_retention = len(proc_keywords.intersection(orig_keywords_sample)) / len(orig_keywords_sample)
                metrics['keyword_retentions'].append(keyword_retention)
            
            # 4. 内容相似度
            best_score = self.calculate_best_similarity(sample, processed)
            metrics['similarity_scores'].append(best_score)
            
            # 5. 熵变化 - 测量信息量变化
            orig_entropy = self._calculate_entropy(sample)
            entropy_change = proc_entropy / max(0.001, orig_entropy)  # 防止除零
            metrics['entropy_changes'].append(min(entropy_change, 1.5))
            
            # 6. 统计显著性 - 使用词频分布比较
            significance = self._calculate_statistical_significance(sample, processed)
            metrics['statistical_significance'].append(significance)
        
        # 计算各指标的平均值
        avg_metrics = {k: sum(v) / max(1, len(v)) for k, v in metrics.items() if v}
        
        # 设置权重
        weights = {
            'paragraph_ratios': 0.15,
            'length_ratios': 0.25,
            'keyword_retentions': 0.20,
            'similarity_scores': 0.15,
            'entropy_changes': 0.15,
            'statistical_significance': 0.10
        }
        
        # 计算加权得分
        score = 0
        for metric, value in avg_metrics.items():
            score += weights.get(metric, 0) * value
        
        # 生成详细报告
        report = [
            f"内容完整性评分: {min(score, 1.0) * 100:.2f}%",
            f"- 段落保留比例: {avg_metrics.get('paragraph_ratios', 0):.2f} (理想: 0.5-1.5)",
            f"- 内容长度比例: {avg_metrics.get('length_ratios', 0):.2f} (理想: 0.7-1.2)",
            f"- 关键词保留率: {avg_metrics.get('keyword_retentions', 0):.2%} (理想: >25%)",
            f"- 内容采样相似度: {avg_metrics.get('similarity_scores', 0):.2%} (理想: >10%)",
            f"- 信息熵比例: {avg_metrics.get('entropy_changes', 0):.2f} (理想: 0.8-1.2)",
            f"- 统计显著性: {avg_metrics.get('statistical_significance', 0):.2f} (理想: >0.6)"
        ]
        
        # 最终判定 - 使用更全面的评判标准
        if (avg_metrics.get('length_ratios', 0) < 0.6 or
                avg_metrics.get('keyword_retentions', 0) < 0.25 or
                avg_metrics.get('statistical_significance', 0) < 0.5):
            self.log.warning("内容验证失败")
            return False, '\n'.join(report)
        else:
            self.log.info("内容验证通过")
            return True, '\n'.join(report)
    
    def _calculate_entropy(self, text):
        """计算文本的信息熵
        
        Args:
            text: 要计算熵的文本
            
        Returns:
            float: 信息熵值
        """
        # 移除空白字符以专注于实际内容
        text = self.re_whitespace.sub('', text)
        if not text:
            return 0
            
        # 计算字符频率
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
            
        # 计算熵
        length = len(text)
        entropy = 0
        for count in freq.values():
            probability = count / length
            entropy -= probability * (math.log(probability) / math.log(2))
            
        return entropy
    
    def _calculate_statistical_significance(self, sample, full_text):
        """计算样本与完整文本之间的统计显著性
        
        Args:
            sample: 样本文本
            full_text: 完整文本
            
        Returns:
            float: 统计显著性得分 (0-1)
        """
        # 简化为词频分布比较
        # 提取词汇（简单分词）
        sample_words = self.re_words.findall(sample.lower())
        full_words = self.re_words.findall(full_text.lower())
        
        if not sample_words or not full_words:
            return 1.0  # 如果没有词汇，默认为完全匹配
            
        # 计算样本中的词频
        sample_freq = {}
        for word in sample_words:
            if len(word) > 1:  # 忽略单字符词
                sample_freq[word] = sample_freq.get(word, 0) + 1
                
        # 计算完整文本中的词频
        full_freq = {}
        for word in full_words:
            if len(word) > 1:  # 忽略单字符词
                full_freq[word] = full_freq.get(word, 0) + 1
                
        # 计算频率分布的相似度
        common_words = set(sample_freq.keys()).intersection(set(full_freq.keys()))
        if not common_words:
            return 0.0
            
        # 计算频率向量的余弦相似度
        dot_product = sum(sample_freq[word] * full_freq[word] for word in common_words)
        sample_magnitude = math.sqrt(sum(freq**2 for freq in sample_freq.values()))
        full_magnitude = math.sqrt(sum(freq**2 for freq in full_freq.values()))
        
        if sample_magnitude == 0 or full_magnitude == 0:
            return 0.0
            
        return dot_product / (sample_magnitude * full_magnitude)

class Converter:
    """转换器类，协调各组件工作，实现完整的转换流程"""
    
    def __init__(self, config_manager, log_manager, file_handler, text_processor, content_validator, error_handler):
        """初始化转换器
        
        Args:
            config_manager: 配置管理器实例
            log_manager: 日志管理器实例
            file_handler: 文件处理器实例
            text_processor: 文本处理器实例
            content_validator: 内容验证器实例
            error_handler: 错误处理器实例
        """
        self.config = config_manager
        self.log = log_manager
        self.file = file_handler
        self.processor = text_processor
        self.validator = content_validator
        self.error_handler = error_handler
    
    def convert(self, input_file, output_file, verify=True, chunk_size=1024 * 1024, max_workers=None, debug=False, enable_semantic_markers=False):
        """转换info文件到纯文本，支持大文件处理，保持简单转换
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            verify: 是否验证内容
            chunk_size: 分块大小
            max_workers: 最大工作线程数
            debug: 是否输出调试信息
            enable_semantic_markers: 是否启用语义标记功能（默认禁用）
        
        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_file):
                return self.error_handler.handle_error(Exception(f"文件不存在: {input_file}"), "文件检查", fatal=True)
            
            # 获取文件总大小用于进度显示
            file_info = self.file.get_file_info(input_file, self.config.get_config('large_file_threshold'))
            if file_info is None:
                return self.error_handler.handle_error(Exception(f"无法获取文件 {input_file} 信息"), "文件信息获取", fatal=True)
            
            file_size = file_info['size']
            
            # 动态调整chunk_size
            if file_size > self.config.get_config('large_file_threshold'):
                # 根据文件大小动态调整chunk_size
                if file_size > 1024 * 1024 * 1024:  # 大于1GB
                    chunk_size = 4 * 1024 * 1024  # 4MB
                elif file_size > 100 * 1024 * 1024:  # 大于100MB
                    chunk_size = 2 * 1024 * 1024  # 2MB
                
                # 根据可用内存进一步调整
                try:
                    import psutil
                    available_memory = psutil.virtual_memory().available
                    # 确保chunk_size不超过可用内存的10%
                    max_chunk_size = available_memory // 10
                    if chunk_size > max_chunk_size:
                        chunk_size = max(max_chunk_size, 512 * 1024)  # 最小不低于512KB
                        self.log.info(f"根据可用内存调整chunk_size为 {chunk_size / 1024 / 1024:.2f} MB")
                except ImportError:
                    self.log.debug("psutil库不可用，无法根据可用内存调整chunk_size")
                
                # 动态调整工作线程数
                if max_workers is None:
                    import multiprocessing
                    max_workers = max(1, multiprocessing.cpu_count() - 1)  # 预留一个核心给系统
                
                self.log.info(f"[注意] 文件较大 ({file_size / 1024 / 1024:.2f} MB)，将使用分块并行处理")
                self.log.info(f"块大小: {chunk_size / 1024 / 1024:.2f} MB, 工作线程数: {max_workers}")
                use_chunking = True
            else:
                use_chunking = False
            
            self.log.print_progress(f"正在读取文件: {input_file}", 1, 5)
            
            # 流式读取和处理，减少内存使用
            if use_chunking:
                # 使用流式处理，避免一次性加载整个文件
                original_content_samples = {}  # 使用字典存储采样内容，键为位置标识
                processed_chunks = []
                removed_content_hashes = defaultdict(int)
                total_replaced_count = 0
                
                # 第一阶段：读取和基本处理
                with self.file.safe_open_file(input_file) as f:
                    # 确定总块数
                    total_chunks = (file_size + chunk_size - 1) // chunk_size
                    self.log.info(f"文件将分为 {total_chunks} 个块进行处理")
                    
                    # 并行处理配置
                    pattern_keys = [
                        "headers", "separator", "menu", "menu_items", "control_chars",
                    ]
                    
                    # 创建线程池
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        chunks = []
                        chunk_positions = []
                        
                        # 读取所有块
                        position = 0
                        self.log.start_progress(total_chunks, "读取文件块")
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
                        
                        self.log.info(f"读取了 {len(chunks)} 个块，总大小: {position} 字节")
                        
                        # 提交并行处理任务
                        self.log.start_progress(len(chunks), "处理文件块")
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
                        
                        # 使用索引映射来确保结果按正确顺序排列
                        future_to_index = {future: i for i, future in enumerate(futures)}
                        sorted_results = [None] * len(chunks)
                        
                        # 处理结果
                        completed = 0
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                # 获取正确的索引
                                index = future_to_index[future]
                                sorted_results[index] = future.result()
                                completed += 1
                                self.log.update_progress(completed)
                            except Exception as e:
                                return self.error_handler.handle_error(e, "块处理", fatal=True)
                        self.log.end_progress()
                        
                        # 合并结果
                        self.log.start_progress(len(sorted_results), "合并处理结果")
                        for i, (processed_chunk, chunk_hashes, replaced_count) in enumerate(sorted_results):
                            processed_chunks.append(processed_chunk)
                            for hash_val, count in chunk_hashes.items():
                                removed_content_hashes[hash_val] += count
                            total_replaced_count += replaced_count
                            
                            # 为验证保存更多样本，包括开头、中间和结尾部分
                            if i == 0:
                                original_content_samples['start'] = chunks[i]
                            elif i == len(sorted_results) // 2:
                                original_content_samples['middle'] = chunks[i]
                            elif i == len(sorted_results) - 1:
                                original_content_samples['end'] = chunks[i]
                            # 额外随机采样
                            elif i % (max(1, len(chunks) // 10)) == 0:  # 大约采样10%的块
                                original_content_samples[f'sample_{i}'] = chunks[i]
                            
                            self.log.update_progress(i + 1)
                        self.log.end_progress()
                
                # 合并处理后的块
                content = ''.join(processed_chunks)
                self.log.info(f"    总共移除了 {total_replaced_count} 项格式内容")
            
            else:
                # 小文件直接处理
                try:
                    with self.file.safe_open_file(input_file) as f:
                        original_content = f.read()
                except IOError as e:
                    return self.error_handler.handle_error(e, "文件读取", fatal=True)
                
                # 创建内容副本用于处理
                content = original_content
                original_content_samples = {'full': original_content}  # 小文件保存完整内容
                
                self.log.print_progress("开始优化处理格式标记", 2, 5)
                
                # 内容处理 - 第一阶段：移除格式标记
                pattern_desc = [
                    ("文件头信息", 'headers'),
                    ("分隔线", 'separator'),
                    ("菜单区域", 'menu'),
                    ("菜单条目", 'menu_items'),
                    ("控制字符", 'control_chars'),
                ]
                
                try:
                    # 处理格式标记
                    content, removed_content_hashes, replaced_count = self.processor.process_format_markers(
                        content, pattern_desc, self.processor.get_tqdm_instance() is not None
                    )
                except Exception as e:
                    return self.error_handler.handle_error(e, "格式标记处理", fatal=True)
                
                self.log.info(f"    总共移除了 {replaced_count} 项格式内容")
            
            # 规范化空行 - 这是一个简单的格式化，不会大幅修改内容
            self.log.print_progress("规范化文档格式", 3, 5)
            try:
                content = self.config.get_pattern('empty_lines').sub('\n\n', content)
            except Exception as e:
                return self.error_handler.handle_error(e, "空行规范化", fatal=True)
            
            self.log.print_progress(f"正在保存到: {output_file}", 4, 5)
            # 保存处理后的文本
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except IOError as e:
                return self.error_handler.handle_error(e, "文件保存", fatal=True)
            
            # 内容验证
            validation_result = None
            if verify:
                self.log.print_progress("执行内容完整性验证", 5, 5)
                
                try:
                    # 改进的验证方法，使用统计学多指标验证
                    if use_chunking:
                        # 大文件使用采样验证
                        validation_result = self.validator.validate(
                            original_content_samples,
                            content,
                            removed_content_hashes,
                            file_size,
                            is_large_file=True
                        )
                    else:
                        # 小文件使用完整验证
                        validation_result = self.validator.validate(
                            original_content_samples['full'],
                            content,
                            removed_content_hashes,
                            file_size,
                            is_large_file=False
                        )
                    
                    if not validation_result[0]:
                        self.log.warning(f"警告：内容验证失败 - {validation_result[1]}")
                        return False, f"警告：内容验证失败 - {validation_result[1]}"
                    self.log.info(f"    {validation_result[1]}")
                except Exception as e:
                    return self.error_handler.handle_error(e, "内容验证", fatal=False, return_value=(False, "内容验证过程出错，但文件处理已完成"))
            else:
                self.log.print_progress("跳过内容验证", 5, 5)
            
            return True, f"转换成功! {validation_result[1] if validation_result else ''}"
            
        except Exception as e:
            return self.error_handler.handle_error(e, "转换过程", fatal=True)

class Application:
    """应用类，处理命令行参数，初始化和运行转换器"""
    
    def __init__(self):
        """初始化应用"""
        self.config = ConfigManager()
        self.log = LogManager()
        self.file = FileHandler(self.log)
        self.processor = TextProcessor(self.config, self.log)
        self.validator = ContentValidator(self.config, self.log)
        self.error_handler = ErrorHandler(self.log)
        self.converter = Converter(self.config, self.log, self.file, self.processor, self.validator, self.error_handler)
    
    def parse_args(self):
        """解析命令行参数
        
        Returns:
            dict: 解析后的参数字典
        """
        if len(sys.argv) < 3:
            print("用法: python c3po.py 输入文件.info 输出文件.txt [--no-verify] [--workers=N] [--debug] [--log=LEVEL]")
            sys.exit(1)
        
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        # 解析可选参数
        verify = self.config.get_config('verify')
        max_workers = self.config.get_config('max_workers')
        debug = self.config.get_config('debug')
        log_level = self.config.get_config('log_level')
        
        for arg in sys.argv[3:]:
            if arg.startswith('--workers='):
                try:
                    max_workers = int(arg.split('=')[1])
                    if max_workers <= 0:
                        self.log.warning("工作线程数必须大于0，将使用默认值")
                        max_workers = self.config.get_config('max_workers')
                except ValueError:
                    self.log.warning("警告: 无效的工作线程数，将使用默认值")
            elif arg == '--debug':
                debug = True
                log_level = logging.DEBUG
            elif arg == '--no-verify':
                verify = False
            elif arg.startswith('--log='):
                level_name = arg.split('=')[1].upper()
                if hasattr(logging, level_name):
                    log_level = getattr(logging, level_name)
                else:
                    self.log.warning(f"警告: 无效的日志级别 '{level_name}'，将使用默认值")
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'verify': verify,
            'max_workers': max_workers,
            'debug': debug,
            'log_level': log_level
        }
    
    def run(self):
        """运行应用"""
        # 解析参数
        args = self.parse_args()
        
        # 设置日志级别
        self.log.set_level(args['log_level'])
        
        # 显示处理信息
        self.log.print_separator()
        self.log.info(f"C3PO - INFO文件转换工具 v{__version__}")
        self.log.info(f"输入文件: {args['input_file']}")
        self.log.info(f"输出文件: {args['output_file']}")
        self.log.info(f"工作线程: {args['max_workers'] if args['max_workers'] else '自动'}")
        self.log.info(f"调试模式: {'开启' if args['debug'] else '关闭'}")
        self.log.info(f"内容验证: {'开启' if args['verify'] else '关闭'}")
        self.log.info(f"日志级别: {logging.getLevelName(args['log_level'])}")
        self.log.print_separator()
        
        # 执行转换
        start_time = time.time()
        success, message = self.converter.convert(
            args['input_file'],
            args['output_file'],
            verify=args['verify'],
            max_workers=args['max_workers'],
            debug=args['debug'],
            enable_semantic_markers=False  # 禁用语义标记
        )
        
        # 显示结果
        elapsed = time.time() - start_time
        self.log.print_separator()
        if success:
            self.log.success(f"✓ {message}")
            self.log.success(f"处理耗时: {self.file.format_time(elapsed)}")
        else:
            self.log.error(f"✗ {message}")
        self.log.print_separator()
        
        # 返回状态码
        sys.exit(0 if success else 1)

# 创建转换器实例
config = ConfigManager()
log = LogManager()
file_handler = FileHandler(log)
processor = TextProcessor(config, log)
validator = ContentValidator(config, log)
error_handler = ErrorHandler(log)
converter = Converter(config, log, file_handler, processor, validator, error_handler)

# 替换原有的函数调用
def convert_info_to_text_with_progress(input_file, output_file, verify=True, chunk_size=1024 * 1024, max_workers=None, debug=False, enable_semantic_markers=False):
    return converter.convert(input_file, output_file, verify, chunk_size, max_workers, debug, enable_semantic_markers)

def main():
    """主程序入口"""
    app = Application()
    app.run()

if __name__ == "__main__":
    main()
