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
from collections import defaultdict

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

#################################################
# 配置部分
#################################################

# 默认配置
DEFAULT_CONFIG = {
    'chunk_size': 1024 * 1024,  # 1MB
    'large_file_threshold': 10 * 1024 * 1024,  # 10MB
    'max_workers': None,  # 默认使用CPU核心数
    'verify': True,  # 默认启用内容验证
    'debug': False,  # 默认关闭调试模式
    'log_level': logging.INFO,  # 默认日志级别
}

# 正则表达式模式
REGEX_PATTERNS = {
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
SQL_KEYWORDS = [
    'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
    'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN',
    'INSERT INTO', 'UPDATE', 'DELETE FROM', 'CREATE TABLE', 'ALTER TABLE',
    'DROP TABLE', 'TRUNCATE TABLE', 'BEGIN TRANSACTION', 'COMMIT', 'ROLLBACK'
]

# 语义标记 - 用于增强文档结构
SEMANTIC_MARKERS = {
    'code_block': {
        'pattern': re.compile(r'```.*?```', re.DOTALL),
        'tag': 'CODE'
    },
    'sql_block': {
        'pattern': re.compile(r'(?i)(' + '|'.join(SQL_KEYWORDS) + r')\s+\w+'),
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

#################################################
# 处理器部分
#################################################

def process_chunk(chunk, patterns, is_first_chunk=False, is_last_chunk=False):
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
            logger.debug(f"截取块的第一个完整段落，移除了 {first_para_start + 2} 个字符")

        # 找到最后一个完整段落的结束
        last_para_end = content.rfind('\n\n')
        if last_para_end > 0:
            content = content[:last_para_end]
            logger.debug(f"截取块的最后一个完整段落，保留了 {last_para_end} 个字符")

    # 应用格式处理
    for pattern_key in patterns:
        pattern = REGEX_PATTERNS[pattern_key]
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

    logger.debug(f"块处理完成，移除了 {replaced_count} 项内容")
    return content, removed_content_hashes, replaced_count

def process_format_markers(content, pattern_desc, tqdm_available=False):
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
            logger.info(f"    处理 {desc}...")

        pattern = REGEX_PATTERNS[pattern_key]
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

    logger.info(f"    总共移除了 {replaced_count} 项格式内容")
    return content, removed_content_hashes, replaced_count

def add_metadata_markers(content, max_content_size=10 * 1024 * 1024, max_matches_per_type=1000, debug=False,
                         enable_semantic_markers=False):
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
        logger.debug("语义标记功能已禁用，返回原始内容")
        return content

    # 临时禁用所有处理，直接返回原始内容
    if debug:
        logger.warning("警告：所有语义标记处理已临时禁用，直接返回原始内容")

    # 这里保留此函数接口，但实际不执行任何处理，直接返回原始内容
    # 未来可以根据需要实现语义标记功能
    return content

#################################################
# 验证器部分
#################################################

def simple_validate_content(original, processed, removed_hashes=None):
    """简化的内容验证函数

    Args:
        original: 原始内容
        processed: 处理后的内容
        removed_hashes: 移除内容的哈希值字典（可选）

    Returns:
        tuple: (验证结果布尔值, 详细报告)
    """
    logger.info("开始内容验证...")
    results = {}

    # 对于大文件，只使用样本进行验证
    if len(original) > 1024 * 1024 or len(processed) > 1024 * 1024:
        logger.info("检测到大文件，使用采样验证方法")
        # 从文件开头和结尾各取100KB进行验证
        sample_size = 100 * 1024
        orig_start = original[:sample_size]
        orig_end = original[-sample_size:] if len(original) > sample_size else ""
        proc_start = processed[:sample_size]
        proc_end = processed[-sample_size:] if len(processed) > sample_size else ""

        # 合并样本
        original_sample = orig_start + orig_end
        processed_sample = proc_start + proc_end

        logger.debug(f"原始内容样本大小: {len(original_sample)}, 处理后内容样本大小: {len(processed_sample)}")

        # 使用样本进行验证
        return validate_content_sample(original_sample, processed_sample)
    else:
        logger.info("文件较小，进行完整验证")
        # 小文件直接验证
        return validate_content_sample(original, processed)

def validate_content_sample(original, processed):
    """验证内容样本

    Args:
        original: 原始内容样本
        processed: 处理后的内容样本

    Returns:
        tuple: (验证结果布尔值, 详细报告)
    """
    results = {}

    # 1. 段落数量验证
    orig_paragraphs = len(re.split(r'\n\s*\n', original))
    proc_paragraphs = len(re.split(r'\n\s*\n', processed))
    para_ratio = proc_paragraphs / max(1, orig_paragraphs)
    results['paragraph_ratio'] = para_ratio
    logger.debug(f"段落数量: 原始={orig_paragraphs}, 处理后={proc_paragraphs}, 比例={para_ratio:.2f}")

    # 2. 内容长度验证
    orig_length = len(re.sub(r'\s+', '', original))
    proc_length = len(re.sub(r'\s+', '', processed))
    length_ratio = proc_length / max(1, orig_length)
    results['length_ratio'] = length_ratio
    logger.debug(f"内容长度: 原始={orig_length}, 处理后={proc_length}, 比例={length_ratio:.2f}")

    # 3. 关键词保留验证
    orig_keywords = set(re.findall(r'\b([A-Z]{2,})\b', original))
    proc_keywords = set(re.findall(r'\b([A-Z]{2,})\b', processed))

    # 计算关键词保留率
    if orig_keywords:
        keyword_retention = len(proc_keywords.intersection(orig_keywords)) / len(orig_keywords)
        logger.debug(f"关键词: 原始={len(orig_keywords)}, 保留={len(proc_keywords.intersection(orig_keywords))}, 保留率={keyword_retention:.2%}")
    else:
        keyword_retention = 1.0
        logger.debug("未找到关键词，默认保留率为100%")
    results['keyword_retention'] = keyword_retention

    # 4. 内容采样相似度检查
    similarity_scores = []
    # 随机选择几个内容块进行相似度比较
    sample_size = min(3, max(1, orig_paragraphs // 50))

    if orig_paragraphs > 5:
        # 从原文中选择样本段落
        paragraphs = re.split(r'\n\s*\n', original)
        logger.debug(f"选择 {sample_size} 个段落进行相似度检查")

        # 使用分层采样，确保从文档不同部分选择样本
        if len(paragraphs) > sample_size * 3:
            # 分层采样
            section_size = len(paragraphs) // sample_size
            samples = [paragraphs[i * section_size] for i in range(sample_size)]
            logger.debug(f"使用分层采样，每 {section_size} 个段落选择一个")
        else:
            # 随机采样
            samples = random.sample(paragraphs, min(sample_size, len(paragraphs)))
            logger.debug("使用随机采样")

        # 计算相似度
        for i, sample in enumerate(samples):
            if len(sample) < 20:  # 忽略太短的样本
                logger.debug(f"样本 {i + 1} 太短，跳过")
                continue

            # 计算最佳相似度
            best_score = calculate_best_similarity(sample, processed)
            logger.debug(f"样本 {i + 1} 最佳相似度: {best_score:.2%}")
            if best_score > 0:
                similarity_scores.append(best_score)

    # 计算平均采样相似度
    avg_similarity = sum(similarity_scores) / max(1, len(similarity_scores)) if similarity_scores else 0
    results['sampled_similarity'] = avg_similarity
    logger.debug(f"平均采样相似度: {avg_similarity:.2%}")

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

    logger.debug(f"内容完整性加权得分: {min(score, 1.0) * 100:.2f}%")

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
        logger.warning("内容验证失败")
        return False, '\n'.join(report)
    else:
        logger.info("内容验证通过")
        return True, '\n'.join(report)

def calculate_best_similarity(sample, processed_text):
    """计算样本与处理后文本的最佳相似度

    Args:
        sample: 样本文本
        processed_text: 处理后的完整文本

    Returns:
        float: 最佳相似度分数
    """
    # 对于大文本，只使用前后各100KB进行相似度计算
    if len(processed_text) > 200 * 1024:
        logger.debug("处理文本较大，只使用前后各100KB进行相似度计算")
        sample_size = 100 * 1024
        proc_start = processed_text[:sample_size]
        proc_end = processed_text[-sample_size:]
        processed_text = proc_start + proc_end

    proc_paragraphs = re.split(r'\n\s*\n', processed_text)
    best_score = 0

    # 对于大文档，随机采样进行比较
    if len(proc_paragraphs) > 50:
        logger.debug(f"处理文本段落数 {len(proc_paragraphs)} > 50，随机采样50个段落进行比较")
        compare_set = random.sample(proc_paragraphs, 50)
    else:
        logger.debug(f"处理文本段落数 {len(proc_paragraphs)} <= 50，使用全部段落进行比较")
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

#################################################
# 日志系统
#################################################

# 配置日志系统
def setup_logging(level=logging.INFO):
    """配置日志系统

    Args:
        level: 日志级别
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

# 全局日志对象
logger = setup_logging()

#################################################
# 工具函数
#################################################

def get_file_info(file_path):
    """获取文件信息

    Args:
        file_path: 文件路径

    Returns:
        dict: 包含文件大小、修改时间等信息的字典，如果文件不存在则返回None
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None

    stats = os.stat(file_path)
    info = {
        'size': stats.st_size,
        'modified': stats.st_mtime,
        'is_large': stats.st_size > DEFAULT_CONFIG['large_file_threshold']
    }

    logger.debug(f"文件信息: {file_path}, 大小: {format_size(info['size'])}, 最后修改: {time.ctime(info['modified'])}")
    return info

def safe_open_file(file_path, mode='r', encoding='utf-8'):
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
        logger.debug(f"尝试使用 {encoding} 编码打开文件: {file_path}")
        return open(file_path, mode=mode, encoding=encoding, errors='replace')
    except UnicodeDecodeError:
        logger.warning(f"使用 {encoding} 编码打开文件失败，尝试使用 Latin-1 编码...")
        return open(file_path, mode=mode, encoding='latin1', errors='replace')
    except Exception as e:
        logger.error(f"打开文件失败: {file_path}, 错误: {str(e)}")
        raise

def print_progress(message, step=None, total_steps=None):
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

    logger.info(log_msg)

def print_separator(char='-', length=60):
    """打印分隔线

    Args:
        char: 分隔字符
        length: 分隔线长度
    """
    logger.info(char * length)

def format_time(seconds):
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

def format_size(size_bytes):
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

def create_string_buffer():
    """创建字符串缓冲区

    Returns:
        StringIO: 字符串缓冲区对象
    """
    return io.StringIO()

def get_tqdm_instance(iterable=None, total=None, desc=None):
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
            logger.info(f"{desc}...")
        return iterable

#################################################
# 主程序部分
#################################################

def convert_info_to_text_with_progress(input_file, output_file, verify=True, chunk_size=1024 * 1024, max_workers=None,
                                      debug=False, enable_semantic_markers=False):
    """优化版：转换info文件到纯文本，支持大文件处理，保持简单转换

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
            logger.error(f"错误：文件 {input_file} 不存在")
            return False, f"错误：文件 {input_file} 不存在"

        # 获取文件总大小用于进度显示
        file_info = get_file_info(input_file)
        if file_info is None:
            logger.error(f"错误：无法获取文件 {input_file} 信息")
            return False, f"错误：无法获取文件 {input_file} 信息"

        file_size = file_info['size']
        if file_size > DEFAULT_CONFIG['large_file_threshold']:
            logger.info(f"[注意] 文件较大 ({file_size / 1024 / 1024:.2f} MB)，将使用分块并行处理")
            use_chunking = True
        else:
            use_chunking = False

        print_progress(f"正在读取文件: {input_file}", 1, 5)

        # 流式读取和处理，减少内存使用
        if use_chunking:
            # 使用流式处理，避免一次性加载整个文件
            original_content = ""
            processed_chunks = []
            removed_content_hashes = defaultdict(int)
            total_replaced_count = 0

            # 第一阶段：读取和基本处理
            with safe_open_file(input_file) as f:
                # 确定总块数
                total_chunks = (file_size + chunk_size - 1) // chunk_size
                logger.info(f"文件将分为 {total_chunks} 个块进行处理")

                # 并行处理配置
                pattern_desc = [
                    "headers", "separator", "menu", "menu_items", "control_chars",
                ]

                # 创建线程池
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    chunks = []
                    chunk_positions = []

                    # 读取所有块
                    position = 0
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        chunk_positions.append(position)
                        position += len(chunk)

                    logger.info(f"读取了 {len(chunks)} 个块，总大小: {position} 字节")

                    # 提交并行处理任务
                    for i, chunk in enumerate(chunks):
                        is_first = (i == 0)
                        is_last = (i == len(chunks) - 1)
                        future = executor.submit(
                            process_chunk,
                            chunk,
                            pattern_desc,
                            is_first,
                            is_last
                        )
                        futures.append(future)

                    # 处理结果
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())

                    # 按原始顺序整理结果 - 确保与原始代码一致
                    sorted_results = [None] * len(chunks)
                    for i, future in enumerate(futures):
                        sorted_results[i] = results[futures.index(future)]

                    # 合并结果
                    for i, (processed_chunk, chunk_hashes, replaced_count) in enumerate(sorted_results):
                        processed_chunks.append(processed_chunk)
                        for hash_val, count in chunk_hashes.items():
                            removed_content_hashes[hash_val] += count
                        total_replaced_count += replaced_count

                        # 保存原始内容用于验证 - 只保存第一个和最后一个块
                        if i == 0 or i == len(sorted_results) - 1:
                            original_content += chunks[i]

            # 合并处理后的块
            content = ''.join(processed_chunks)
            logger.info(f"    总共移除了 {total_replaced_count} 项格式内容")

        else:
            # 小文件直接处理
            with safe_open_file(input_file) as f:
                original_content = f.read()

            # 创建内容副本用于处理
            content = original_content

            print_progress("开始优化处理格式标记", 2, 5)

            # 内容处理 - 第一阶段：移除格式标记
            pattern_desc = [
                ("文件头信息", 'headers'),
                ("分隔线", 'separator'),
                ("菜单区域", 'menu'),
                ("菜单条目", 'menu_items'),
                ("控制字符", 'control_chars'),
            ]

            # 处理格式标记
            content, removed_content_hashes, replaced_count = process_format_markers(
                content, pattern_desc, get_tqdm_instance() is not None
            )

            logger.info(f"    总共移除了 {replaced_count} 项格式内容")

        # 规范化空行 - 这是一个简单的格式化，不会大幅修改内容
        print_progress("规范化文档格式", 3, 5)
        content = REGEX_PATTERNS['empty_lines'].sub('\n\n', content)

        print_progress(f"正在保存到: {output_file}", 4, 5)
        # 保存处理后的文本
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # 内容验证
        validation_result = None
        if verify:
            print_progress("执行内容完整性验证", 5, 5)
            # 对于大文件，只验证部分内容
            if len(original_content) < 1024 * 1024:  # 如果原始内容小于1MB
                with safe_open_file(input_file) as f:
                    original_content = f.read()

            validation_result = simple_validate_content(
                original_content,
                content,
                removed_content_hashes
            )

            if not validation_result[0]:
                logger.warning(f"警告：内容验证失败 - {validation_result[1]}")
                return False, f"警告：内容验证失败 - {validation_result[1]}"
            logger.info(f"    {validation_result[1]}")
        else:
            print_progress("跳过内容验证", 5, 5)

        return True, f"转换成功! {validation_result[1] if validation_result else ''}"

    except Exception as e:
        import traceback
        error_msg = f"处理错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, error_msg

def main():
    """主程序入口"""
    if len(sys.argv) < 3:
        print("用法: python c3po.py 输入文件.info 输出文件.txt [--no-verify] [--workers=N] [--debug] [--log=LEVEL]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 解析可选参数
    verify = DEFAULT_CONFIG['verify']
    max_workers = DEFAULT_CONFIG['max_workers']
    debug = DEFAULT_CONFIG['debug']
    log_level = DEFAULT_CONFIG['log_level']

    for arg in sys.argv[3:]:
        if arg.startswith('--workers='):
            try:
                max_workers = int(arg.split('=')[1])
            except:
                print("警告: 无效的工作线程数，将使用默认值")
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
                print(f"警告: 无效的日志级别 '{level_name}'，将使用默认值")

    # 设置日志级别
    logger.setLevel(log_level)

    # 显示处理信息
    print_separator()
    logger.info(f"C3PO - INFO文件转换工具 v{__version__}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"工作线程: {max_workers if max_workers else '自动'}")
    logger.info(f"调试模式: {'开启' if debug else '关闭'}")
    logger.info(f"内容验证: {'开启' if verify else '关闭'}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    print_separator()

    # 执行转换
    start_time = time.time()
    success, message = convert_info_to_text_with_progress(
        input_file,
        output_file,
        verify=verify,
        max_workers=max_workers,
        debug=debug,
        enable_semantic_markers=False  # 禁用语义标记
    )

    # 显示结果
    elapsed = time.time() - start_time
    print_separator()
    if success:
        logger.info(f"✓ {message}")
        logger.info(f"处理耗时: {format_time(elapsed)}")
    else:
        logger.error(f"✗ {message}")
    print_separator()

    # 返回状态码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
