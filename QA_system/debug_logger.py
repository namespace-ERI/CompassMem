#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM interaction log management module
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DebugLogger:
    """LLM interaction log manager - all QA and interaction records for each item in one file"""
    
    def __init__(self, debug_mode: bool = False, base_dir: str = "./llm_debug"):
        self.debug_mode = debug_mode
        self.llm_log_dir = None
        self.interaction_counter = 0  # For generating unique interaction IDs
        self.item_data = {}  # Store all data for each item {item_id: {'qa_list': [], 'current_qa_interactions': []}}
        
        # Always initialize logging (not limited to debug mode)
        self._init_logging(base_dir)
    
    def _init_logging(self, base_dir: str):
        """Create LLM interaction log directory"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.llm_log_dir = Path(base_dir) / timestamp
            self.llm_log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📝 LLM interaction log directory: {self.llm_log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM interaction log directory: {e}")
    
    def log_llm_interaction(self, phase: str, formatted_prompt: str, raw_response: str, 
                           extra_info: Optional[Dict[str, Any]] = None):
        """Record LLM interaction to memory, waiting to write with QA results
        
        Args:
            phase: Interaction phase (e.g., 'check_sufficiency', 'generate_answer', 'refinement_query')
            formatted_prompt: The formatted prompt sent to LLM
            raw_response: The raw response from LLM
            extra_info: Additional information to log (e.g., path_id, round_num, node_id, item_id)
        """
        if self.llm_log_dir is None:
            return
        
        try:
            # Generate unique interaction ID
            self.interaction_counter += 1
            interaction_id = f"{self.interaction_counter:04d}"
            
            # Get item_id
            item_id = extra_info.get('item_id', 'unknown') if extra_info else 'unknown'
            
            # Initialize item data structure
            if item_id not in self.item_data:
                self.item_data[item_id] = {
                    'item_id': item_id,
                    'qa_list': [],
                    'current_qa_interactions': []
                }
            
            # Build interaction record
            interaction_record = {
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "prompt": formatted_prompt,
                "response": raw_response
            }
            
            # Add metadata
            if extra_info:
                interaction_record["metadata"] = extra_info
            
            # Add to current QA's interaction list
            self.item_data[item_id]['current_qa_interactions'].append(interaction_record)
            
            if self.debug_mode:
                logger.debug(f"📝 Recorded LLM interaction #{interaction_id} (item: {item_id}, phase: {phase})")
                
        except Exception as e:
            logger.warning(f"Failed to record LLM interaction: {e}")
    
    def log_qa_result(self, item_id: str, question: str, result: Dict[str, Any]):
        """Record single QA result and corresponding LLM interactions to memory"""
        if self.llm_log_dir is None:
            return
        
        try:
            # Initialize item data structure
            if item_id not in self.item_data:
                self.item_data[item_id] = {
                    'item_id': item_id,
                    'qa_list': [],
                    'current_qa_interactions': []
                }
            
            # Build QA record (including answer and interactions)
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
            
            # Add to QA list
            self.item_data[item_id]['qa_list'].append(qa_record)
            
            # Clear current QA's interaction list, prepare for next QA
            self.item_data[item_id]['current_qa_interactions'] = []
            
            if self.debug_mode:
                logger.debug(f"📝 Recorded QA result (item: {item_id}, question: {question[:50]}...)")
                
        except Exception as e:
            logger.warning(f"Failed to record QA result: {e}")
    
    def finalize_item(self, item_id: str):
        """Complete processing of an item, write all data to file"""
        if self.llm_log_dir is None or item_id not in self.item_data:
            return
        
        try:
            # Generate filename
            filename = f"{item_id}_debug.json"
            filepath = self.llm_log_dir / filename
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.item_data[item_id], f, ensure_ascii=False, indent=2)
            
            logger.info(f"📝 Wrote item debug file: {filepath}")
            
            # Clean up written data
            del self.item_data[item_id]
            
        except Exception as e:
            logger.warning(f"Failed to write item debug file: {e}")
