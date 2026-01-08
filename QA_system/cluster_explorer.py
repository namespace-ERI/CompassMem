#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚类节点探索模块
负责从高层聚类节点选择低层成员节点进行探索
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ClusterExplorer:
    """聚类节点探索器"""
    
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
        """从top-k聚类节点中选择最佳的起始节点
        
        流程：
        1. 对每个聚类节点，获取其成员节点的summary
        2. 展示给LLM，让LLM选择要探索哪个节点
        3. 返回选中的节点
        
        Args:
            question: 用户问题
            top_k_clusters: top-k聚类节点列表
            graph: 图数据
            item_id: 项目ID
            
        Returns:
            选中的起始节点信息
        """
        if not top_k_clusters:
            logger.warning("没有聚类节点可供选择")
            return None
        
        # 构建聚类信息提示
        cluster_info_list = []
        
        for idx, cluster in enumerate(top_k_clusters):
            cluster_id = cluster['node_id']
            member_ids = cluster['member_nodes']
            n_members = cluster['n_members']
            similarity = cluster['similarity']
            
            # 获取成员节点的详细信息
            member_summaries = self._get_member_summaries(member_ids, graph)
            
            cluster_info = {
                'cluster_idx': idx,
                'cluster_id': cluster_id,
                'n_members': n_members,
                'similarity': similarity,
                'member_summaries': member_summaries
            }
            
            cluster_info_list.append(cluster_info)
        
        # 让LLM选择要探索的节点
        selected_node_id, raw_response, formatted_prompt = self._ask_llm_to_select_node(
            question,
            cluster_info_list,
            graph,
            item_id
        )
        
        if not selected_node_id:
            # 兜底：如果LLM没有明确选择，选择第一个聚类中相似度最高的节点
            logger.warning("LLM未明确选择节点，使用兜底策略")
            first_cluster = top_k_clusters[0]
            selected_node = self.embedding_manager.find_best_member_node(
                question, 
                first_cluster['member_nodes'],
                graph
            )
        else:
            # 获取选中节点的完整信息
            selected_node = self._get_node_info(selected_node_id, graph)
        
        if selected_node:
            logger.info(f"✅ 从聚类中选择起始节点: {selected_node['node_id']}")
        else:
            logger.error("❌ 无法从聚类中选择起始节点")
        
        return selected_node
    
    def _get_member_summaries(self, 
                             member_ids: List[str], 
                             graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取成员节点的summary信息
        
        Args:
            member_ids: 成员节点ID列表
            graph: 图数据
            
        Returns:
            成员节点summary列表
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
        """获取节点的完整信息
        
        Args:
            node_id: 节点ID
            graph: 图数据
            
        Returns:
            节点信息字典
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
        """让LLM从聚类的成员节点中选择最相关的节点
        
        Args:
            question: 用户问题
            cluster_info_list: 聚类信息列表
            graph: 图数据
            item_id: 项目ID
            
        Returns:
            (selected_node_id, raw_response, formatted_prompt)
        """
        # 构建prompt
        prompt = self._build_selection_prompt(question, cluster_info_list)
        
        # 调用LLM
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的信息检索助手，需要根据用户问题选择最相关的记忆节点。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        raw_response = self.llm_handler.generate(messages, temperature=0.0, max_tokens=500)
        
        # 解析LLM的选择
        selected_node_id = self._parse_node_selection(raw_response, cluster_info_list)
        
        return selected_node_id, raw_response, prompt
    
    def _build_selection_prompt(self,
                               question: str,
                               cluster_info_list: List[Dict[str, Any]]) -> str:
        """构建节点选择的prompt
        
        Args:
            question: 用户问题
            cluster_info_list: 聚类信息列表
            
        Returns:
            格式化的prompt字符串
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
            
            for member in member_summaries[:10]:  # 最多显示10个成员
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
        """解析LLM的节点选择结果
        
        Args:
            raw_response: LLM的原始回复
            cluster_info_list: 聚类信息列表
            
        Returns:
            选中的节点ID，如果解析失败返回None
        """
        # 收集所有可能的节点ID
        all_node_ids = []
        for cluster_info in cluster_info_list:
            for member in cluster_info['member_summaries']:
                all_node_ids.append(member['node_id'])
        
        # 方法1：查找"Selected Node:"后面的ID
        import re
        selected_pattern = r'Selected\s+Node\s*:\s*(N\d+)'
        match = re.search(selected_pattern, raw_response, re.IGNORECASE)
        if match:
            node_id = match.group(1)
            if node_id in all_node_ids:
                logger.info(f"从'Selected Node:'解析到节点: {node_id}")
                return node_id
        
        # 方法2：查找响应开头的节点ID（通常LLM会直接输出）
        first_line = raw_response.split('\n')[0]
        for node_id in all_node_ids:
            if node_id in first_line:
                logger.info(f"从响应首行解析到节点: {node_id}")
                return node_id
        
        # 方法3：查找响应中出现的第一个有效节点ID
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, raw_response)
        for match in matches:
            if match in all_node_ids:
                logger.info(f"通过正则匹配找到节点: {match}")
                return match
        
        logger.warning(f"无法从LLM响应中解析节点ID: {raw_response[:200]}")
        return None

