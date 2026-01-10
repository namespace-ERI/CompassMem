#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue data management module - supports both locomo and narrativeQA formats
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """Dialogue data manager - automatically detects and supports multiple data formats"""
    
    def __init__(self, qa_data_path: str):
        self.qa_data_path = qa_data_path
        self.data_format = None  # 'locomo' or 'narrativeqa'
        self.raw_data = None  # Save raw data for item_id matching
        self.conversation_data = self._load_conversation_data()
    
    def _detect_data_format(self, data: List[Dict[str, Any]]) -> str:
        """Automatically detect data format"""
        if not data:
            return 'unknown'
        
        first_item = data[0]
        
        # Check if document_id field exists (narrativeQA feature)
        if 'document_id' in first_item:
            return 'narrativeqa'
        
        # Check for traditional locomo format features
        if 'conversation' in first_item:
            conv = first_item['conversation']
            # Check if session_X keys exist (skip session_X_date_time)
            for key in conv.keys():
                if key.startswith('session_') and not key.endswith('_date_time'):
                    # Check if conversation in session has speaker field
                    if isinstance(conv[key], list) and len(conv[key]) > 0:
                        if 'speaker' in conv[key][0]:
                            return 'locomo'
                        else:
                            # No speaker field, might be narrativeQA format
                            return 'narrativeqa'
        
        return 'unknown'
    
    def _load_conversation_data(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Load dialogue data, establish item_index -> utterance_id -> dialogue content mapping"""
        logger.info(f"Loading dialogue data: {self.qa_data_path}")
        
        with open(self.qa_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.raw_data = data  # Save raw data
        
        # Automatically detect data format
        self.data_format = self._detect_data_format(data)
        logger.info(f"Detected data format: {self.data_format}")
        
        if self.data_format == 'locomo':
            return self._load_locomo_data(data)
        elif self.data_format == 'narrativeqa':
            return self._load_narrativeqa_data(data)
        else:
            logger.warning(f"Unknown data format, using default loading method")
            return self._load_locomo_data(data)
    
    def _load_locomo_data(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Load locomo format data"""
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
        
        logger.info(f"Loaded dialogue data for {len(conversation_data)} locomo items")
        return conversation_data
    
    def _load_narrativeqa_data(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Load narrativeQA format data"""
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
                                    # narrativeQA format: plain text, no speaker/timestamp
                                    conversation_map[utterance['dia_id']] = {
                                        'text': utterance['text'],
                                        'session': session_name,
                                        'format': 'narrativeqa'
                                    }
                
                conversation_data[item_index] = conversation_map
        
        logger.info(f"Loaded dialogue data for {len(conversation_data)} narrativeQA items")
        return conversation_data
    
    def get_item_index(self, item_id: str) -> Optional[int]:
        """Get corresponding index based on item_id - supports multiple formats"""
        # Locomo format: locomo_item1, locomo_item2, ...
        if item_id.startswith('locomo_item'):
            try:
                return int(item_id.replace('locomo_item', '')) - 1
            except ValueError:
                pass
        
        # NarrativeQA format: match through document_id
        if self.raw_data:
            for idx, item in enumerate(self.raw_data):
                # Check document_id field
                if item.get('document_id') == item_id:
                    return idx
                # Check item_id field (if exists)
                if item.get('item_id') == item_id:
                    return idx
        
        logger.warning(f"Cannot find index for item_id '{item_id}'")
        return None
    
    def extract_utterance_texts(self, utterance_refs: List[str], item_index: Optional[int] = None) -> List[str]:
        """Extract utterance text content - automatically adjusts output format based on data format"""
        if item_index is None:
            return []
            
        conv_map = self.conversation_data.get(item_index, {})
        actual_texts = []
        
        for ref in utterance_refs:
            if ref in conv_map:
                utterance_info = conv_map[ref]
                data_format = utterance_info.get('format', self.data_format)
                
                if data_format == 'locomo':
                    # Locomo format: timestamp: speaker: text
                    timestamp = utterance_info.get('date_time', '')
                    speaker = utterance_info.get('speaker', '')
                    text = utterance_info['text']
                    
                    if timestamp:
                        formatted_text = f"{timestamp}: {speaker}: {text}"
                    else:
                        formatted_text = f"{speaker}: {text}"
                    actual_texts.append(formatted_text)
                    
                elif data_format == 'narrativeqa':
                    # NarrativeQA format: plain text
                    text = utterance_info['text']
                    actual_texts.append(text)
                    
                else:
                    # Unknown format: plain text
                    actual_texts.append(utterance_info['text'])
            else:
                actual_texts.append(f"[Missing utterance: {ref}]")
                
        return actual_texts
