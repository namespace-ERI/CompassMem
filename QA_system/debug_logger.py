#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM交互日志管理模块
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DebugLogger:
    """LLM交互日志管理器 - 每个item的所有QA和交互记录在一个文件中"""
    
    def __init__(self, debug_mode: bool = False, base_dir: str = "/share/project/zyt/hyy/Memory/QA_system/llm_debug"):
        self.debug_mode = debug_mode
        self.llm_log_dir = None
        self.interaction_counter = 0  # 用于生成唯一的交互ID
        self.item_data = {}  # 存储每个item的所有数据 {item_id: {'qa_list': [], 'current_qa_interactions': []}}
        
        # 始终初始化日志记录（不仅限于debug模式）
        self._init_logging(base_dir)
    
    def _init_logging(self, base_dir: str):
        """创建LLM交互日志目录"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.llm_log_dir = Path(base_dir) / timestamp
            self.llm_log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📝 LLM交互日志目录: {self.llm_log_dir}")
        except Exception as e:
            logger.warning(f"初始化LLM交互日志目录失败: {e}")
    
    def log_llm_interaction(self, phase: str, formatted_prompt: str, raw_response: str, 
                           extra_info: Optional[Dict[str, Any]] = None):
        """记录LLM交互到内存中，等待与QA结果一起写入
        
        Args:
            phase: Interaction phase (e.g., 'check_sufficiency', 'generate_answer', 'refinement_query')
            formatted_prompt: The formatted prompt sent to LLM
            raw_response: The raw response from LLM
            extra_info: Additional information to log (e.g., path_id, round_num, node_id, item_id)
        """
        if self.llm_log_dir is None:
            return
        
        try:
            # 生成唯一的交互ID
            self.interaction_counter += 1
            interaction_id = f"{self.interaction_counter:04d}"
            
            # 获取item_id
            item_id = extra_info.get('item_id', 'unknown') if extra_info else 'unknown'
            
            # 初始化item数据结构
            if item_id not in self.item_data:
                self.item_data[item_id] = {
                    'item_id': item_id,
                    'qa_list': [],
                    'current_qa_interactions': []
                }
            
            # 构建交互记录
            interaction_record = {
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "prompt": formatted_prompt,
                "response": raw_response
            }
            
            # 添加元数据
            if extra_info:
                interaction_record["metadata"] = extra_info
            
            # 添加到当前QA的交互列表
            self.item_data[item_id]['current_qa_interactions'].append(interaction_record)
            
            if self.debug_mode:
                logger.debug(f"📝 记录LLM交互 #{interaction_id} (item: {item_id}, phase: {phase})")
                
        except Exception as e:
            logger.warning(f"记录LLM交互失败: {e}")
    
    def log_qa_result(self, item_id: str, question: str, result: Dict[str, Any]):
        """记录单个QA结果和对应的LLM交互到内存中"""
        if self.llm_log_dir is None:
            return
        
        try:
            # 初始化item数据结构
            if item_id not in self.item_data:
                self.item_data[item_id] = {
                    'item_id': item_id,
                    'qa_list': [],
                    'current_qa_interactions': []
                }
            
            # 构建QA记录（包含答案和交互）
            qa_record = {
                'question': question,
                'answer': result.get('answer', ''),
                'elapsed_time_seconds': result.get('elapsed_time_seconds', 0),
                'exploration_type': result.get('exploration_type', 'single_path'),
                'num_paths': result.get('num_paths', 0),
                'num_sufficient_paths': result.get('num_sufficient_paths', 0),
                'used_refinement': result.get('used_refinement', False),
                'refined_query': result.get('refined_query', ''),
                'visited_nodes': result.get('visited_nodes', []),
                'path_details': result.get('path_details', []),
                'refinement_details': result.get('refinement_details', {}),
                'llm_interactions': self.item_data[item_id]['current_qa_interactions'].copy()
            }
            
            # 添加到QA列表
            self.item_data[item_id]['qa_list'].append(qa_record)
            
            # 清空当前QA的交互列表，准备下一个QA
            self.item_data[item_id]['current_qa_interactions'] = []
            
            if self.debug_mode:
                logger.debug(f"📝 记录QA结果 (item: {item_id}, question: {question[:50]}...)")
                
        except Exception as e:
            logger.warning(f"记录QA结果失败: {e}")
    
    def finalize_item(self, item_id: str):
        """完成一个item的处理，将所有数据写入文件"""
        if self.llm_log_dir is None or item_id not in self.item_data:
            return
        
        try:
            # 生成文件名
            filename = f"{item_id}_debug.json"
            filepath = self.llm_log_dir / filename
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.item_data[item_id], f, ensure_ascii=False, indent=2)
            
            logger.info(f"📝 写入item调试文件: {filepath}")
            
            # 清理已写入的数据
            del self.item_data[item_id]
            
        except Exception as e:
            logger.warning(f"写入item调试文件失败: {e}")
