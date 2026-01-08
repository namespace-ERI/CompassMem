#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化图数据加载和管理模块
支持加载包含聚类节点的图数据
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class HierarchicalGraphLoader:
    """层次化图数据加载器"""
    
    def __init__(self, graphs_dir: str):
        self.graphs_dir = Path(graphs_dir)
        self.graphs = {}
        
    def load_graph(self, item_id: str) -> Dict[str, Any]:
        """按需加载图数据（包含聚类信息）"""
        if item_id not in self.graphs:
            graph_path = self.graphs_dir / f"{item_id}.json"
            if graph_path.exists():
                logger.info(f"加载层次化图: {graph_path}")
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    
                # 验证是否包含聚类信息
                if 'cluster_nodes' not in graph_data:
                    logger.warning(f"图 {item_id} 不包含聚类节点！")
                    graph_data['cluster_nodes'] = []
                    graph_data['cluster_edges'] = []
                else:
                    n_clusters = len(graph_data['cluster_nodes'])
                    n_cluster_edges = len(graph_data.get('cluster_edges', []))
                    logger.info(f"  包含 {n_clusters} 个聚类节点, {n_cluster_edges} 条聚类边")
                    
                self.graphs[item_id] = graph_data
            else:
                logger.warning(f"图文件不存在: {graph_path}")
                self.graphs[item_id] = {
                    'nodes': [], 
                    'edges': [],
                    'cluster_nodes': [],
                    'cluster_edges': []
                }
        return self.graphs[item_id]
    
    def get_node_by_id(self, node_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """根据节点ID获取节点信息（支持普通节点和聚类节点）"""
        if item_id not in self.graphs:
            return None
            
        graph = self.graphs[item_id]
        
        # 首先在普通节点中查找
        for node in graph.get('nodes', []):
            if node['id'] == node_id:
                return {
                    'node_id': node['id'],
                    'node_type': 'normal',
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None)
                }
        
        # 在聚类节点中查找
        for cluster_node in graph.get('cluster_nodes', []):
            if cluster_node['id'] == node_id:
                return {
                    'node_id': cluster_node['id'],
                    'node_type': 'cluster',
                    'cluster_id': cluster_node.get('cluster_id'),
                    'member_nodes': cluster_node.get('member_nodes', []),
                    'n_members': cluster_node.get('n_members', 0),
                    'summaries': cluster_node.get('summaries', []),
                    'embedding': cluster_node.get('embedding', None)
                }
        
        return None
    
    def get_neighbor_nodes(self, node_id: str, item_id: str) -> List[Dict[str, Any]]:
        """获取指定节点的1跳邻居节点
        
        注意：这里只返回普通节点的邻居，不包括聚类关系
        """
        if item_id not in self.graphs:
            return []
            
        graph = self.graphs[item_id]
        neighbor_ids = set()
        
        # 只查找原始边（不包括聚类边）
        for edge in graph.get('edges', []):
            if edge['source'] == node_id:
                neighbor_ids.add(edge['target'])
            elif edge['target'] == node_id:
                neighbor_ids.add(edge['source'])
        
        # 获取邻居节点信息
        neighbors = []
        for neighbor_id in neighbor_ids:
            neighbor_node = self.get_node_by_id(neighbor_id, item_id)
            if neighbor_node and neighbor_node.get('node_type') == 'normal':
                relations = self.get_node_relations(node_id, neighbor_id, item_id)
                neighbor_node['relations'] = relations
                neighbors.append(neighbor_node)
                
        return neighbors
    
    def get_node_relations(self, source_id: str, target_id: str, item_id: str) -> List[Dict[str, Any]]:
        """获取两个节点之间的关系（只查找原始边）"""
        if item_id not in self.graphs:
            return []
            
        relations = []
        graph = self.graphs[item_id]
        
        # 只查找原始边
        for edge in graph.get('edges', []):
            if (edge['source'] == source_id and edge['target'] == target_id) or \
               (edge['source'] == target_id and edge['target'] == source_id):
                relations.append({
                    'type': edge['type'],
                    'evidence': edge.get('evidence', []),
                    'source': edge.get('source'),
                    'target': edge.get('target')
                })
                
        return relations
    
    def get_cluster_members(self, cluster_id: str, item_id: str) -> List[Dict[str, Any]]:
        """获取聚类的所有成员节点
        
        Args:
            cluster_id: 聚类节点ID
            item_id: 项目ID
            
        Returns:
            成员节点列表
        """
        if item_id not in self.graphs:
            return []
        
        graph = self.graphs[item_id]
        
        # 找到聚类节点
        cluster_node = None
        for c_node in graph.get('cluster_nodes', []):
            if c_node['id'] == cluster_id:
                cluster_node = c_node
                break
        
        if not cluster_node:
            return []
        
        # 获取所有成员节点的完整信息
        member_ids = cluster_node.get('member_nodes', [])
        members = []
        
        for node in graph.get('nodes', []):
            if node['id'] in member_ids:
                members.append({
                    'node_id': node['id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None)
                })
        
        return members

