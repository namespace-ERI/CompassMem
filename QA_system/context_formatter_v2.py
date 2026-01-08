#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文信息格式化模块 - V2版本
支持三动作机制，格式化已保留节点信息
"""

from typing import Dict, List, Any, Optional


class ContextFormatterV2:
    """上下文信息格式化器 - V2版本"""
    
    def __init__(self, conversation_manager, graph_loader, use_relation: bool = True):
        self.conversation_manager = conversation_manager
        self.graph_loader = graph_loader
        self.use_relation = use_relation
    
    def format_current_node_info(self, node: Dict[str, Any], item_id: str) -> str:
        """格式化当前节点信息 - 返回text原文（带时间戳）"""
        if not node:
            return "No current node information."
            
        parts = []
        
        # 提取实际对话内容（带时间戳）
        if node.get('utterance_refs'):
            item_index = self.conversation_manager.get_item_index(item_id) if item_id else None
            actual_texts = self.conversation_manager.extract_utterance_texts(node['utterance_refs'], item_index)
            if actual_texts:
                parts.extend(actual_texts)
        # 如果没有utterance_refs，使用texts字段
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
        """格式化已保留节点的信息 - 返回所有保留节点的text原文（带时间戳）
        
        Args:
            kept_node_ids: 已保留的节点ID列表
            item_id: 项目ID
            include_metadata: 是否包含People和Time元数据（用于生成答案时）
            
        Returns:
            格式化的已保留节点信息
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
            
            # 提取实际对话文本（带时间戳）
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
            
            # 如果包含元数据，添加People和Time信息
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
        """格式化邻居节点信息 - 返回summary和relation
        
        根据 self.use_relation 决定是否包含 relation 信息
        """
        if not neighbors:
            return "No related nodes found."
        
        # 根据是否使用relation调整提示语
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
            
            # 只使用summary，不提供text原文
            summary_text = ''
            if neighbor.get('summaries'):
                summary_text = '; '.join(neighbor['summaries'])

            # 如果启用relation，构建relation信息
            if self.use_relation:
                relation_sentences = []
                for rel in neighbor.get('relations', []):
                    rel_type = rel.get('type', 'related')
                    source = rel.get('source')
                    target = rel.get('target')
                    
                    # 方向文本，如果有current_node_id则以其为起点描述
                    if current_node_id and (source == current_node_id or target == current_node_id):
                        other = target if source == current_node_id else source
                        direction_text = f"current node -[{rel_type}]-> {other}" if source == current_node_id else f"{other} -[{rel_type}]-> current node"
                    else:
                        # 回退为无方向的连接描述
                        if source and target:
                            direction_text = f"{source} -[{rel_type}]- {target}"
                        else:
                            direction_text = rel_type
                    relation_sentences.append(f"Relation: {direction_text}.")

                # 组装单个邻居的描述
                if relation_sentences:
                    relation_text = ' '.join(relation_sentences)
                else:
                    relation_text = "Relation: (no explicit relation details)."

                if summary_text:
                    parts.append(f"Node {neighbor_id}: {summary_text}. {relation_text}")
                else:
                    parts.append(f"Node {neighbor_id}: {relation_text}")
            else:
                # 不使用relation，只显示summary
                if summary_text:
                    parts.append(f"Node {neighbor_id}: {summary_text}.")
                else:
                    parts.append(f"Node {neighbor_id}: (no summary available)")
                        
        return "\n".join(parts)
    
    def build_final_context_from_kept_nodes(self, kept_node_ids: List[str], item_id: str) -> str:
        """从已保留节点构建最终上下文（用于生成答案）
        
        Args:
            kept_node_ids: 已保留的节点ID列表
            item_id: 项目ID
            
        Returns:
            格式化的上下文字符串（包含text原文、People和Time信息）
        """
        # 生成答案时需要包含完整的元数据信息（People和Time）
        return self.format_kept_nodes_info(kept_node_ids, item_id, include_metadata=True)
    
    def build_final_context_from_visited_nodes(self, visited_node_ids: List[str], item_id: str) -> str:
        """从所有访问过的节点构建最终上下文（当没有保留节点时使用）
        
        Args:
            visited_node_ids: 所有访问过的节点ID列表
            item_id: 项目ID
            
        Returns:
            格式化的上下文字符串
        """
        item_index = self.conversation_manager.get_item_index(item_id) if item_id else None
        all_texts = []
        
        for node_id in visited_node_ids:
            node = self.graph_loader.get_node_by_id(node_id, item_id)
            if not node or node.get('node_type') == 'cluster':
                continue
            
            # 提取实际对话文本
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
        
        # 去重但保持顺序
        unique_texts = []
        seen = set()
        for text in all_texts:
            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        return '\n\n'.join(unique_texts)

