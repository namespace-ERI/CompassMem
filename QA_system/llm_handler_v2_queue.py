#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM interaction handling module - V2 Queue version
Supports three-action mechanism: skip, expand, answer
Supports selecting multiple next nodes (not just one)
"""

import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from openai import OpenAI
from prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMHandlerV2Queue:
    """LLM interaction handler - V2 Queue version (supports multi-node selection) - OpenAI API version"""
    
    def __init__(self, model_name: str, api_base: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        """Initialize LLM handler
        
        Args:
            model_name: Model name (configured in vLLM service)
            api_base: Base URL of API service
            api_key: API key (can use "EMPTY" for local deployment)
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        
        # Token statistics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0  # Call count
        
        # Initialize model
        self._init_models()
        
    def _init_models(self):
        """Initialize OpenAI client"""
        logger.info(f"Initializing OpenAI client, connecting to: {self.api_base}")
        logger.info(f"Using model: {self.model_name}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        logger.info(f"OpenAI client initialized")
        
        # Set sampling parameters
        self.generation_params = {
            'temperature': 0.6,
            'max_tokens': 8192,
            'top_p': 0.95
        }
    
    def _generate(self, messages: List[Dict[str, str]]) -> Tuple[str, None]:
        """Generate response using OpenAI API and track tokens
        
        Args:
            messages: Message list, format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            
        Returns:
            Tuple of (raw_response, None)
            Note: Second return value is None to maintain compatibility with original interface (originally returned output_ids)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.generation_params
            )
            
            raw_response = response.choices[0].message.content
            
            # Track tokens (if API returned usage information)
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # If API didn't return usage information, use estimated values
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                logger.warning("API did not return token usage information")
            
            # Accumulate statistics
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.call_count += 1
            
            logger.debug(f"This call - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total: {total_tokens}")
            
            return raw_response, None
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _extract_non_thinking_content(self, raw_response: str, output_ids: list = None) -> str:
        """Extract non-thinking content from model output
        
        Args:
            raw_response: Raw response from model
            output_ids: Keep this parameter for interface compatibility, but not used
        """
        # Use regex to remove thinking tag content
        text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = text.strip()
        
        return text
    
    def generate_subgoals(self, question: str) -> Tuple[List[str], str, str]:
        """Use LLM to break down question into subgoals
        
        Args:
            question: Original question
            
        Returns:
            Tuple of (subgoals_list, raw_response, formatted_prompt)
            subgoals_list: ['subgoal1', 'subgoal2', ...]
        """
        prompt = PromptTemplates.get_planner_prompt(question)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('planner')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Parse subgoals
        subgoals = self._parse_subgoals(response)
        
        logger.info(f"Generated {len(subgoals)} sub-goals:")
        for i, sg in enumerate(subgoals):
            logger.info(f"  {i+1}. {sg}")
        
        formatted_prompt = str(messages)
        return subgoals, raw_response, formatted_prompt
    
    def _parse_subgoals(self, response: str) -> List[str]:
        """Parse subgoals list
        
        Returns:
            List of subgoal strings
        """
        subgoals = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match "Sub-goal X: ..." format
            match = re.match(r'Sub-goal\s+\d+\s*:\s*(.+)', line, re.IGNORECASE)
            if match:
                subgoal = match.group(1).strip()
                subgoals.append(subgoal)
        
        # If not found, return empty list
        if not subgoals:
            logger.warning("Failed to parse sub-goals from response")
        
        return subgoals
    
    def check_action(self, question: str, kept_nodes_info: str, current_info: str, 
                     neighbor_info: str, round_num: int, 
                     subgoals: List[str] = None, subgoal_status: dict = None) -> Tuple[str, List[str], List[int], str, str]:
        """Let LLM decide what action to take (skip/expand/answer) and select next nodes (possibly multiple)
        
        Args:
            question: Question
            kept_nodes_info: Information of kept nodes
            current_info: Current node information
            neighbor_info: Neighbor node information
            round_num: Current round number
            subgoals: Subgoal list (optional, V3.5 new)
            subgoal_status: Subgoal status dictionary (optional, V3.5 new)
            
        Returns:
            Tuple of (action, next_node_ids, satisfied_subgoals, raw_response, formatted_prompt)
            action: 'skip', 'expand', or 'answer'
            next_node_ids: If skip or expand, specified next node ID list (can be multiple)
            satisfied_subgoals: List of satisfied subgoal indices
        """
        
        # Choose different prompt based on whether subgoals exist
        if subgoals and subgoal_status is not None:
            prompt = PromptTemplates.get_action_decision_prompt_with_subgoals(
                question, subgoals, subgoal_status, kept_nodes_info, current_info, neighbor_info
            )
        else:
            # Backward compatibility: if no subgoals, use original prompt
            prompt = PromptTemplates.get_action_decision_prompt(
                question, kept_nodes_info, current_info, neighbor_info
            )
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('action_decision')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Parse response
        action = self._parse_action(response)
        next_nodes = self._parse_next_nodes(response)  # Changed to plural, returns list
        
        # Parse satisfied subgoals (if subgoal version is used)
        satisfied_subgoals = []
        if subgoals:
            satisfied_subgoals = self._parse_satisfied_subgoals(response)
        
        logger.info(f"Round {round_num}: LLM decided ACTION={action}, NEXT_NODES={next_nodes}, SATISFIED_SUBGOALS={satisfied_subgoals}")
        
        # For interface compatibility, return string representation of messages as formatted_prompt
        formatted_prompt = str(messages)
        return action, next_nodes, satisfied_subgoals, raw_response, formatted_prompt
    
    def _parse_satisfied_subgoals(self, response: str) -> List[int]:
        """Parse list of satisfied subgoal indices
        
        Returns:
            List of satisfied subgoal indices (0-based)
        """
        # Find SATISFIED_SUBGOALS: [1, 2, 3] format
        match = re.search(r'SATISFIED_SUBGOALS\s*:\s*\[([^\]]*)\]', response, re.IGNORECASE)
        
        if match:
            numbers_str = match.group(1)
            # Extract all numbers
            numbers = re.findall(r'\d+', numbers_str)
            # Convert to 0-based indices
            satisfied = [int(n) - 1 for n in numbers if int(n) > 0]
            return satisfied
        
        return []
    
    def select_top_k_nodes(self, question: str, subgoals: List[str], 
                          candidate_nodes: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """Select nodes to actually explore from top-k candidate nodes
        
        Args:
            question: Question
            subgoals: Subgoal list
            candidate_nodes: Candidate node list [{'node_id': 'N1', 'summary': '...', 'similarity': 0.8}, ...]
            
        Returns:
            Tuple of (selected_node_ids, raw_response, formatted_prompt)
        """
        prompt = PromptTemplates.get_top_k_node_selection_prompt(
            question, subgoals, candidate_nodes
        )
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('top_k_selection')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Parse selected nodes
        selected_nodes = self._parse_selected_nodes_from_top_k(response, 
                                                               [n['node_id'] for n in candidate_nodes])
        
        logger.info(f"LLM selected {len(selected_nodes)} nodes from top-k candidates: {selected_nodes}")
        
        formatted_prompt = str(messages)
        return selected_nodes, raw_response, formatted_prompt
    
    def _parse_selected_nodes_from_top_k(self, response: str, valid_node_ids: List[str]) -> List[str]:
        """Parse list of selected node IDs from top-k
        
        Returns:
            List of selected node IDs
        """
        # Find node list after "Selected Nodes:"
        match = re.search(r'Selected\s+Nodes\s*:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        
        if match:
            nodes_str = match.group(1)
            # Extract all node IDs
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            # Filter valid node IDs
            valid_selected = [nid.upper() for nid in node_ids if nid.upper() in valid_node_ids]
            return valid_selected
        
        # If not found, try to find node IDs in entire response
        node_ids = re.findall(r'\bN\d+\b', response)
        seen = set()
        unique_nodes = []
        for nid in node_ids:
            nid_upper = nid.upper()
            if nid_upper in valid_node_ids and nid_upper not in seen:
                seen.add(nid_upper)
                unique_nodes.append(nid_upper)
        
        return unique_nodes
    
    def _parse_action(self, response: str) -> str:
        """Parse action"""
        # Find ACTION: xxx
        action_match = re.search(r'ACTION\s*:\s*(SKIP|EXPAND|ANSWER)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()
            return action.lower()  # Return lowercase
        
        # If not found, try to find these words in response
        response_upper = response.upper()
        if 'ACTION: ANSWER' in response_upper or 'ACTION:ANSWER' in response_upper:
            return 'answer'
        elif 'ACTION: EXPAND' in response_upper or 'ACTION:EXPAND' in response_upper:
            return 'expand'
        elif 'ACTION: SKIP' in response_upper or 'ACTION:SKIP' in response_upper:
            return 'skip'
        
        # Default to expand (conservative strategy)
        logger.warning(f"Unable to parse action, defaulting to expand")
        return 'expand'
    
    def _parse_next_nodes(self, response: str) -> List[str]:
        """Parse next batch of node IDs (possibly multiple, up to 3)
        
        Returns:
            List of node IDs (e.g., ['N15', 'N22', 'N30']), returns at most 3
        """
        # Find NEXT_NODES: or NEXT_NODE:
        next_nodes_match = re.search(r'NEXT_NODES?\s*:\s*([^\n]+)', response, re.IGNORECASE)
        
        if next_nodes_match:
            nodes_str = next_nodes_match.group(1).strip()
            
            # Check if it's NONE
            if 'NONE' in nodes_str.upper():
                return []
            
            # Extract all node IDs (format like N15, N22, N30 or N15,N22,N30)
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            if node_ids:
                # Convert to unified format (uppercase) and limit to at most 3
                node_ids = [nid.upper() for nid in node_ids][:3]
                if len(node_ids) > 3:
                    logger.warning(f"LLM selected {len(node_ids)} nodes, limiting to first 3")
                logger.info(f"Parsed {len(node_ids)} nodes: {node_ids}")
                return node_ids
        
        # If NEXT_NODES: not found, try to find node IDs in entire response
        # But only if it doesn't explicitly say NONE
        if not re.search(r'NEXT_NODES?\s*:\s*NONE', response, re.IGNORECASE):
            node_ids = re.findall(r'\bN\d+\b', response)
            if node_ids:
                # Deduplicate and maintain order, limit to at most 3
                seen = set()
                unique_nodes = []
                for nid in node_ids:
                    nid_upper = nid.upper()
                    if nid_upper not in seen and len(unique_nodes) < 3:
                        seen.add(nid_upper)
                        unique_nodes.append(nid_upper)
                
                if unique_nodes:
                    logger.info(f"Extracted {len(unique_nodes)} nodes from response: {unique_nodes}")
                    return unique_nodes
        
        return []
    
    def generate_answer(self, question: str, context: str, category: int = None) -> Tuple[str, str, str]:
        """Generate answer based on context
        
        Args:
            question: Question
            context: Context information (original text of kept nodes)
            category: Question category (optional), category 3 uses special prompt
            
        Returns:
            Tuple of (answer, raw_response, formatted_prompt)
        """
        
        # Choose prompt template based on category
        if category == 3:
            prompt = PromptTemplates.get_answer_generation_prompt_category3(question, context)
        else:
            prompt = PromptTemplates.get_answer_generation_prompt(question, context)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('answer_generation')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # For interface compatibility, return string representation of messages as formatted_prompt
        formatted_prompt = str(messages)
        return response.strip(), raw_response, formatted_prompt
    
    def generate_refinement_query(self, original_question: str, context_so_far: str,
                                  subgoals: List[str] = None, subgoal_status: dict = None) -> Tuple[str, str, str]:
        """Generate a refined query to find missing information
        
        Args:
            original_question: The original question being asked
            context_so_far: Information collected so far
            subgoals: Subgoal list (optional, V3.5 new)
            subgoal_status: Subgoal status dictionary (optional, V3.5 new)
            
        Returns:
            Tuple of (refined_query, raw_response, formatted_prompt)
        """
        # Choose different prompt based on whether subgoals exist
        if subgoals and subgoal_status is not None:
            prompt = PromptTemplates.get_refinement_query_prompt_with_subgoals(
                original_question, subgoals, subgoal_status, context_so_far
            )
        else:
            # Backward compatibility: use original prompt
            prompt = PromptTemplates.get_refinement_query_prompt(original_question, context_so_far)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('refinement')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Extract the new query from the response
        new_query = original_question  # Fallback to original
        lines = response.split('\n')
        for line in lines:
            if 'New Query:' in line or 'new query:' in line.lower():
                new_query = line.split(':', 1)[1].strip()
                break
        
        # For interface compatibility, return string representation of messages as formatted_prompt
        formatted_prompt = str(messages)
        return new_query, raw_response, formatted_prompt
    
    def select_nodes_from_cluster(self, question: str, cluster_id: str, 
                                  member_summaries: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """Select multiple relevant member nodes from cluster (based on summary)
        
        Args:
            question: Question
            cluster_id: Cluster ID
            member_summaries: List of member node summary information
                              [{'node_id': 'N1', 'summary': '...', 'people': [...], 'time': '...'}, ...]
        
        Returns:
            Tuple of (selected_node_ids, raw_response, formatted_prompt)
            selected_node_ids: List of node IDs (may contain multiple nodes)
        """
        if not member_summaries:
            return [], "", ""
        
        # Use unified prompt template
        prompt = PromptTemplates.get_cluster_node_selection_prompt(question, member_summaries)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('cluster_selection')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Parse selected nodes (possibly multiple)
        selected_node_ids = self._parse_selected_nodes(response, [m['node_id'] for m in member_summaries])
        
        # For interface compatibility, return string representation of messages as formatted_prompt
        formatted_prompt = str(messages)
        return selected_node_ids, raw_response, formatted_prompt
    
    def _parse_selected_nodes(self, response: str, valid_node_ids: List[str]) -> List[str]:
        """Parse node IDs selected by LLM (possibly multiple, up to 3)
        
        Returns:
            List of valid node IDs, returns at most 3
        """
        # Method 1: Find ID(s) after "Selected Nodes:" or "Selected Node:"
        selected_pattern = r'Selected\s+Nodes?\s*:\s*([^\n]+)'
        match = re.search(selected_pattern, response, re.IGNORECASE)
        
        if match:
            nodes_str = match.group(1).strip()
            # Extract all node IDs
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            if node_ids:
                # Filter valid node IDs, limit to at most 3
                valid_selected = [nid.upper() for nid in node_ids if nid.upper() in valid_node_ids][:3]
                if valid_selected:
                    if len([nid for nid in node_ids if nid.upper() in valid_node_ids]) > 3:
                        logger.warning(f"LLM selected more than 3 nodes from cluster, limiting to first 3")
                    logger.info(f"Parsed {len(valid_selected)} nodes from cluster: {valid_selected}")
                    return valid_selected
        
        # Method 2: Find all valid node IDs appearing in response
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, response)
        valid_matches = []
        seen = set()
        for match in matches:
            match_upper = match.upper()
            if match_upper in valid_node_ids and match_upper not in seen and len(valid_matches) < 3:
                seen.add(match_upper)
                valid_matches.append(match_upper)
        
        if valid_matches:
            logger.info(f"Extracted {len(valid_matches)} valid nodes from response: {valid_matches}")
            return valid_matches
        
        return []
    
    def _parse_selected_node(self, response: str, valid_node_ids: List[str]) -> Optional[str]:
        """Parse single node ID selected by LLM (kept for backward compatibility)"""
        # Method 1: Find ID after "Selected Node:"
        selected_pattern = r'Selected\s+Node\s*:\s*(N\d+)'
        match = re.search(selected_pattern, response, re.IGNORECASE)
        if match:
            node_id = match.group(1)
            if node_id in valid_node_ids:
                return node_id
        
        # Method 2: Find first valid node ID appearing in response
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, response)
        for match in matches:
            if match in valid_node_ids:
                return match
        
        return None
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get token usage statistics
        
        Returns:
            Dict containing:
                - total_prompt_tokens: Total prompt tokens
                - total_completion_tokens: Total completion tokens
                - total_tokens: Total tokens
                - call_count: Total call count
                - avg_prompt_tokens: Average prompt tokens
                - avg_completion_tokens: Average completion tokens
        """
        stats = {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'call_count': self.call_count,
        }
        
        if self.call_count > 0:
            stats['avg_prompt_tokens'] = self.total_prompt_tokens / self.call_count
            stats['avg_completion_tokens'] = self.total_completion_tokens / self.call_count
        else:
            stats['avg_prompt_tokens'] = 0
            stats['avg_completion_tokens'] = 0
        
        return stats
    
    def reset_token_stats(self):
        """Reset token statistics"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        logger.info("Token statistics reset")
    
    def log_token_stats(self):
        """Log current token usage statistics to log"""
        stats = self.get_token_stats()
        logger.info("=" * 60)
        logger.info("Token Usage Statistics:")
        logger.info(f"  Total Calls: {stats['call_count']}")
        logger.info(f"  Total Prompt Tokens: {stats['total_prompt_tokens']:,}")
        logger.info(f"  Total Completion Tokens: {stats['total_completion_tokens']:,}")
        logger.info(f"  Total Tokens: {stats['total_tokens']:,}")
        if stats['call_count'] > 0:
            logger.info(f"  Average Prompt Tokens: {stats['avg_prompt_tokens']:.1f}")
            logger.info(f"  Average Completion Tokens: {stats['avg_completion_tokens']:.1f}")
        logger.info("=" * 60)

