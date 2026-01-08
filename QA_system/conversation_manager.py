#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话数据管理模块 - 支持 locomo 和 narrativeQA 两种格式
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话数据管理器 - 自动检测并支持多种数据格式"""
    
    def __init__(self, qa_data_path: str):
        self.qa_data_path = qa_data_path
        self.data_format = None  # 'locomo' or 'narrativeqa'
        self.raw_data = None  # 保存原始数据用于 item_id 匹配
        self.conversation_data = self._load_conversation_data()
    
    def _detect_data_format(self, data: List[Dict[str, Any]]) -> str:
        """自动检测数据格式"""
        if not data:
            return 'unknown'
        
        first_item = data[0]
        
        # 检查是否有 document_id 字段（narrativeQA 特征）
        if 'document_id' in first_item:
            return 'narrativeqa'
        
        # 检查是否有传统的 locomo 格式特征
        if 'conversation' in first_item:
            conv = first_item['conversation']
            # 检查是否有 session_X 键（跳过 session_X_date_time）
            for key in conv.keys():
                if key.startswith('session_') and not key.endswith('_date_time'):
                    # 检查 session 中的对话是否有 speaker 字段
                    if isinstance(conv[key], list) and len(conv[key]) > 0:
                        if 'speaker' in conv[key][0]:
                            return 'locomo'
                        else:
                            # 没有 speaker 字段，可能是 narrativeQA 格式
                            return 'narrativeqa'
        
        return 'unknown'
    
    def _load_conversation_data(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """加载对话数据，建立item_index -> utterance_id -> 对话内容的映射"""
        logger.info(f"加载对话数据: {self.qa_data_path}")
        
        with open(self.qa_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.raw_data = data  # 保存原始数据
        
        # 自动检测数据格式
        self.data_format = self._detect_data_format(data)
        logger.info(f"检测到数据格式: {self.data_format}")
        
        if self.data_format == 'locomo':
            return self._load_locomo_data(data)
        elif self.data_format == 'narrativeqa':
            return self._load_narrativeqa_data(data)
        else:
            logger.warning(f"未知的数据格式，使用默认加载方式")
            return self._load_locomo_data(data)
    
    def _load_locomo_data(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """加载 locomo 格式的数据"""
        conversation_data = {}
        
        for item_index, item in enumerate(data):
            if 'conversation' in item:
                conv = item['conversation']
                conversation_map = {}
                
                for key, value in conv.items():
                    if key.startswith('session_') and not key.endswith('_date_time'):
                        session_name = key
                        if isinstance(value, list):
                            for utterance in value:
                                if 'dia_id' in utterance and 'text' in utterance:
                                    conversation_map[utterance['dia_id']] = {
                                        'speaker': utterance.get('speaker', ''),
                                        'text': utterance['text'],
                                        'session': session_name,
                                        'date_time': conv.get(f'{session_name}_date_time', ''),
                                        'format': 'locomo'
                                    }
                
                conversation_data[item_index] = conversation_map
        
        logger.info(f"加载了 {len(conversation_data)} 个 locomo item 的对话数据")
        return conversation_data
    
    def _load_narrativeqa_data(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """加载 narrativeQA 格式的数据"""
        conversation_data = {}
        
        for item_index, item in enumerate(data):
            if 'conversation' in item:
                conv = item['conversation']
                conversation_map = {}
                
                for key, value in conv.items():
                    if key.startswith('session_'):
                        session_name = key
                        if isinstance(value, list):
                            for utterance in value:
                                if 'dia_id' in utterance and 'text' in utterance:
                                    # narrativeQA 格式：纯文本，无 speaker/timestamp
                                    conversation_map[utterance['dia_id']] = {
                                        'text': utterance['text'],
                                        'session': session_name,
                                        'format': 'narrativeqa'
                                    }
                
                conversation_data[item_index] = conversation_map
        
        logger.info(f"加载了 {len(conversation_data)} 个 narrativeQA item 的对话数据")
        return conversation_data
    
    def get_item_index(self, item_id: str) -> Optional[int]:
        """根据item_id获取对应的索引 - 支持多种格式"""
        # Locomo 格式：locomo_item1, locomo_item2, ...
        if item_id.startswith('locomo_item'):
            try:
                return int(item_id.replace('locomo_item', '')) - 1
            except ValueError:
                pass
        
        # NarrativeQA 格式：通过 document_id 匹配
        if self.raw_data:
            for idx, item in enumerate(self.raw_data):
                # 检查 document_id 字段
                if item.get('document_id') == item_id:
                    return idx
                # 检查 item_id 字段（如果有的话）
                if item.get('item_id') == item_id:
                    return idx
        
        logger.warning(f"无法找到 item_id '{item_id}' 对应的索引")
        return None
    
    def extract_utterance_texts(self, utterance_refs: List[str], item_index: Optional[int] = None) -> List[str]:
        """提取utterance文本内容 - 根据数据格式自动调整输出格式"""
        if item_index is None:
            return []
            
        conv_map = self.conversation_data.get(item_index, {})
        actual_texts = []
        
        for ref in utterance_refs:
            if ref in conv_map:
                utterance_info = conv_map[ref]
                data_format = utterance_info.get('format', self.data_format)
                
                if data_format == 'locomo':
                    # Locomo 格式：timestamp: speaker: text
                    timestamp = utterance_info.get('date_time', '')
                    speaker = utterance_info.get('speaker', '')
                    text = utterance_info['text']
                    
                    if timestamp:
                        formatted_text = f"{timestamp}: {speaker}: {text}"
                    else:
                        formatted_text = f"{speaker}: {text}"
                    actual_texts.append(formatted_text)
                    
                elif data_format == 'narrativeqa':
                    # NarrativeQA 格式：纯文本
                    text = utterance_info['text']
                    actual_texts.append(text)
                    
                else:
                    # 未知格式：纯文本
                    actual_texts.append(utterance_info['text'])
            else:
                actual_texts.append(f"[Missing utterance: {ref}]")
                
        return actual_texts
