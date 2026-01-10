#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context information formatting module - V2 version
Supports three-action mechanism, formats retained node information
"""

from typing import Dict, List, Any, Optional


class ContextFormatterV2:
    """Context information formatter - V2 version"""
    
    def __init__(self, conversation_manager, graph_loader, use_relation: bool = True):
        self.conversation_manager = conversation_manager
        self.graph_loader = graph_loader
        self.use_relation = use_relation
    
    def format_current_node_info(self, node: Dict[str, Any], item_id: str) -> str:
        """Format current node information - returns text with timestamps"""
        if not node:
            return "No current node information."
            
        parts = []
        
        # Extract actual dialogue content (with timestamps)
        if node.get('utterance_refs'):
            item_index = self.conversation_manager.get_item_index(item_id) if item_id else None
            actual_texts = self.conversation_manager.extract_utterance_texts(node['utterance_refs'], item_index)
            if actual_texts:
                parts.extend(actual_texts)
        # If no utterance_refs, use texts field
        elif node.get('texts'):
            node_texts = node['texts']
            if isinstance(node_texts, list):
                parts.extend(node_texts)
            else:
                parts.append(str(node_texts))
                    
        # Add people information
        if node.get('people'):
            parts.append(f"People: {', '.join(node['people'])}")
            
        # Add time information
        if node.get('time_explicit'):
            parts.append(f"Time: {'; '.join(node['time_explicit'])}")
            
        return "\n".join(parts)
    
    def format_kept_nodes_info(self, kept_node_ids: List[str], item_id: str, include_metadata: bool = False) -> str:
        """Format retained node information - returns all retained nodes' text with timestamps
        
        Args:
            kept_node_ids: List of retained node IDs
            item_id: Item ID
            include_metadata: Whether to include People and Time metadata (for answer generation)
            
        Returns:
            Formatted retained node information
        """
        if not kept_node_ids:
            return ""
        
        item_index = self.conversation_manager.get_item_index(item_id) if item_id else None
        all_parts = []
        
        for node_id in kept_node_ids:
            node = self.graph_loader.get_node_by_id(node_id, item_id)
            if not node or node.get('node_type') == 'cluster':
                continue
            
            node_parts = []
            
            # Extract actual dialogue text (with timestamps)
            if node.get('utterance_refs'):
                actual_texts = self.conversation_manager.extract_utterance_texts(
                    node['utterance_refs'], 
                    item_index
                )
                node_parts.extend(actual_texts)
            elif node.get('texts'):
                node_texts = node['texts']
                if isinstance(node_texts, list):
                    node_parts.extend(node_texts)
                else:
                    node_parts.append(str(node_texts))
            
            # If including metadata, add People and Time information
            if include_metadata:
                if node.get('people'):
                    node_parts.append(f"People: {', '.join(node['people'])}")
                if node.get('time_explicit'):
                    node_parts.append(f"Time: {'; '.join(node['time_explicit'])}")
            
            if node_parts:
                all_parts.append("\n".join(node_parts))
        
        if not all_parts:
            return ""
        
        return "\n\n".join(all_parts)
    
    def format_neighbor_info(self, neighbors: List[Dict[str, Any]], current_node_id: str = None) -> str:
        """Format neighbor node information - returns summary and relation
        
        Decide whether to include relation information based on self.use_relation
        """
        if not neighbors:
            return "No related nodes found."
        
        # Adjust prompt based on whether relation is used
        if self.use_relation:
            parts = [
                "The following are brief neighbor summaries and relation hints. These are not full texts; use them only as navigation signals to decide which node to explore next."
            ]
        else:
            parts = [
                "The following are brief neighbor summaries. These are not full texts; use them only as navigation signals to decide which node to explore next."
            ]
        
        for neighbor in neighbors:
            neighbor_id = neighbor.get('node_id', 'UNKNOWN')
            
            # Only use summary, don't provide full text
            summary_text = ''
            if neighbor.get('summaries'):
                summary_text = '; '.join(neighbor['summaries'])

            # If relation is enabled, build relation information
            if self.use_relation:
                relation_sentences = []
                for rel in neighbor.get('relations', []):
                    rel_type = rel.get('type', 'related')
                    source = rel.get('source')
                    target = rel.get('target')
                    
                    # Direction text, if current_node_id exists, describe from its perspective
                    if current_node_id and (source == current_node_id or target == current_node_id):
                        other = target if source == current_node_id else source
                        direction_text = f"current node -[{rel_type}]-> {other}" if source == current_node_id else f"{other} -[{rel_type}]-> current node"
                    else:
                        # Fallback to undirected connection description
                        if source and target:
                            direction_text = f"{source} -[{rel_type}]- {target}"
                        else:
                            direction_text = rel_type
                    relation_sentences.append(f"Relation: {direction_text}.")

                # Assemble description of single neighbor
                if relation_sentences:
                    relation_text = ' '.join(relation_sentences)
                else:
                    relation_text = "Relation: (no explicit relation details)."

                if summary_text:
                    parts.append(f"Node {neighbor_id}: {summary_text}. {relation_text}")
                else:
                    parts.append(f"Node {neighbor_id}: {relation_text}")
            else:
                # Don't use relation, only show summary
                if summary_text:
                    parts.append(f"Node {neighbor_id}: {summary_text}.")
                else:
                    parts.append(f"Node {neighbor_id}: (no summary available)")
                        
        return "\n".join(parts)
    
    def build_final_context_from_kept_nodes(self, kept_node_ids: List[str], item_id: str) -> str:
        """Build final context from retained nodes (for answer generation)
        
        Args:
            kept_node_ids: List of retained node IDs
            item_id: Item ID
            
        Returns:
            Formatted context string (including text, People and Time information)
        """
        # When generating answer, need to include complete metadata (People and Time)
        return self.format_kept_nodes_info(kept_node_ids, item_id, include_metadata=True)
    
    def build_final_context_from_visited_nodes(self, visited_node_ids: List[str], item_id: str) -> str:
        """Build final context from all visited nodes (used when no kept nodes)
        
        Args:
            visited_node_ids: List of all visited node IDs
            item_id: Item ID
            
        Returns:
            Formatted context string
        """
        item_index = self.conversation_manager.get_item_index(item_id) if item_id else None
        all_texts = []
        
        for node_id in visited_node_ids:
            node = self.graph_loader.get_node_by_id(node_id, item_id)
            if not node or node.get('node_type') == 'cluster':
                continue
            
            # Extract actual dialogue text
            if node.get('utterance_refs'):
                actual_texts = self.conversation_manager.extract_utterance_texts(
                    node['utterance_refs'], 
                    item_index
                )
                all_texts.extend(actual_texts)
            elif node.get('texts'):
                node_texts = node['texts']
                if isinstance(node_texts, list):
                    all_texts.extend(node_texts)
                else:
                    all_texts.append(str(node_texts))
        
        # Deduplicate but maintain order
        unique_texts = []
        seen = set()
        for text in all_texts:
            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        return '\n\n'.join(unique_texts)
