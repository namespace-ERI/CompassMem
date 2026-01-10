#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster node exploration module
Responsible for selecting low-level member nodes from high-level cluster nodes for exploration
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ClusterExplorer:
    """Cluster node explorer"""
    
    def __init__(self, 
                 graph_loader,
                 hierarchical_embedding_manager,
                 llm_handler,
                 context_formatter):
        self.graph_loader = graph_loader
        self.embedding_manager = hierarchical_embedding_manager
        self.llm_handler = llm_handler
        self.context_formatter = context_formatter
    
    def select_start_node_from_clusters(self,
                                       question: str,
                                       top_k_clusters: List[Dict[str, Any]],
                                       graph: Dict[str, Any],
                                       item_id: str) -> Optional[Dict[str, Any]]:
        """Select the best starting node from top-k cluster nodes
        
        Process:
        1. For each cluster node, get summaries of its member nodes
        2. Present to LLM, let LLM choose which node to explore
        3. Return the selected node
        
        Args:
            question: User question
            top_k_clusters: List of top-k cluster nodes
            graph: Graph data
            item_id: Item ID
            
        Returns:
            Selected starting node information
        """
        if not top_k_clusters:
            logger.warning("No cluster nodes available for selection")
            return None
        
        # Build cluster information prompts
        cluster_info_list = []
        
        for idx, cluster in enumerate(top_k_clusters):
            cluster_id = cluster['node_id']
            member_ids = cluster['member_nodes']
            n_members = cluster['n_members']
            similarity = cluster['similarity']
            
            # Get detailed information of member nodes
            member_summaries = self._get_member_summaries(member_ids, graph)
            
            cluster_info = {
                'cluster_idx': idx,
                'cluster_id': cluster_id,
                'n_members': n_members,
                'similarity': similarity,
                'member_summaries': member_summaries
            }
            
            cluster_info_list.append(cluster_info)
        
        # Let LLM choose which node to explore
        selected_node_id, raw_response, formatted_prompt = self._ask_llm_to_select_node(
            question,
            cluster_info_list,
            graph,
            item_id
        )
        
        if not selected_node_id:
            # Fallback: If LLM doesn't make a clear choice, select the most similar node from the first cluster
            logger.warning("LLM did not clearly select a node, using fallback strategy")
            first_cluster = top_k_clusters[0]
            selected_node = self.embedding_manager.find_best_member_node(
                question, 
                first_cluster['member_nodes'],
                graph
            )
        else:
            # Get complete information of the selected node
            selected_node = self._get_node_info(selected_node_id, graph)
        
        if selected_node:
            logger.info(f"✅ Selected starting node from cluster: {selected_node['node_id']}")
        else:
            logger.error("❌ Failed to select starting node from cluster")
        
        return selected_node
    
    def _get_member_summaries(self, 
                             member_ids: List[str], 
                             graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get summary information of member nodes
        
        Args:
            member_ids: List of member node IDs
            graph: Graph data
            
        Returns:
            List of member node summaries
        """
        summaries = []
        
        for node in graph.get('nodes', []):
            if node['id'] in member_ids:
                summary = {
                    'node_id': node['id'],
                    'summary': node.get('summaries', [''])[0] if node.get('summaries') else '',
                    'people': node.get('people', []),
                    'time': node.get('time_explicit', [''])[0] if node.get('time_explicit') else ''
                }
                summaries.append(summary)
        
        return summaries
    
    def _get_node_info(self, 
                      node_id: str, 
                      graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get complete node information
        
        Args:
            node_id: Node ID
            graph: Graph data
            
        Returns:
            Node information dictionary
        """
        for node in graph.get('nodes', []):
            if node['id'] == node_id:
                return {
                    'node_id': node['id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None)
                }
        return None
    
    def _ask_llm_to_select_node(self,
                               question: str,
                               cluster_info_list: List[Dict[str, Any]],
                               graph: Dict[str, Any],
                               item_id: str) -> tuple:
        """Let LLM select the most relevant node from cluster member nodes
        
        Args:
            question: User question
            cluster_info_list: List of cluster information
            graph: Graph data
            item_id: Item ID
            
        Returns:
            (selected_node_id, raw_response, formatted_prompt)
        """
        # Build prompt
        prompt = self._build_selection_prompt(question, cluster_info_list)
        
        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information retrieval assistant who needs to select the most relevant memory node based on the user's question."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        raw_response = self.llm_handler.generate(messages, temperature=0.0, max_tokens=500)
        
        # Parse LLM's selection
        selected_node_id = self._parse_node_selection(raw_response, cluster_info_list)
        
        return selected_node_id, raw_response, prompt
    
    def _build_selection_prompt(self,
                               question: str,
                               cluster_info_list: List[Dict[str, Any]]) -> str:
        """Build prompt for node selection
        
        Args:
            question: User question
            cluster_info_list: List of cluster information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are helping to find the most relevant starting point to answer a question.

We have identified {len(cluster_info_list)} relevant clusters of memory nodes. Each cluster contains multiple related nodes.
Your task is to select the MOST RELEVANT node as the starting point for exploration.

QUESTION: {question}

AVAILABLE NODES:

"""
        
        for cluster_info in cluster_info_list:
            cluster_idx = cluster_info['cluster_idx']
            cluster_id = cluster_info['cluster_id']
            n_members = cluster_info['n_members']
            similarity = cluster_info['similarity']
            member_summaries = cluster_info['member_summaries']
            
            prompt += f"Cluster {cluster_idx + 1} (ID: {cluster_id}, Similarity: {similarity:.3f}, {n_members} nodes):\n"
            
            for member in member_summaries[:10]:  # Show at most 10 members
                node_id = member['node_id']
                summary = member['summary']
                people = ', '.join(member['people']) if member['people'] else 'Unknown'
                time = member['time'] if member['time'] else 'Unknown'
                
                prompt += f"  - {node_id}: {summary}\n"
                if people != 'Unknown' or time != 'Unknown':
                    prompt += f"    (People: {people}, Time: {time})\n"
            
            if n_members > 10:
                prompt += f"  ... and {n_members - 10} more nodes\n"
            
            prompt += "\n"
        
        prompt += """INSTRUCTIONS:
1. Analyze which node is most directly related to answering the question
2. Select the node that likely contains the most relevant information
3. Provide your selection in the format below

RESPONSE FORMAT:
Selected Node: [NODE_ID]
Reason: [Brief explanation]

Example:
Selected Node: N5
Reason: This node discusses the specific event mentioned in the question.
"""
        
        return prompt
    
    def _parse_node_selection(self,
                             raw_response: str,
                             cluster_info_list: List[Dict[str, Any]]) -> Optional[str]:
        """Parse LLM's node selection result
        
        Args:
            raw_response: LLM's raw response
            cluster_info_list: List of cluster information
            
        Returns:
            Selected node ID, returns None if parsing fails
        """
        # Collect all possible node IDs
        all_node_ids = []
        for cluster_info in cluster_info_list:
            for member in cluster_info['member_summaries']:
                all_node_ids.append(member['node_id'])
        
        # Method 1: Find ID after "Selected Node:"
        import re
        selected_pattern = r'Selected\s+Node\s*:\s*(N\d+)'
        match = re.search(selected_pattern, raw_response, re.IGNORECASE)
        if match:
            node_id = match.group(1)
            if node_id in all_node_ids:
                logger.info(f"Parsed node from 'Selected Node:': {node_id}")
                return node_id
        
        # Method 2: Find node ID at the beginning of response (usually LLM outputs directly)
        first_line = raw_response.split('\n')[0]
        for node_id in all_node_ids:
            if node_id in first_line:
                logger.info(f"Parsed node from first line of response: {node_id}")
                return node_id
        
        # Method 3: Find the first valid node ID appearing in the response
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, raw_response)
        for match in matches:
            if match in all_node_ids:
                logger.info(f"Found node through regex matching: {match}")
                return match
        
        logger.warning(f"Failed to parse node ID from LLM response: {raw_response[:200]}")
        return None
