#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical graph data loading and management module
Supports loading graph data containing cluster nodes
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class HierarchicalGraphLoader:
    """Hierarchical graph data loader"""
    
    def __init__(self, graphs_dir: str):
        self.graphs_dir = Path(graphs_dir)
        self.graphs = {}
        
    def load_graph(self, item_id: str) -> Dict[str, Any]:
        """Load graph data on demand (containing cluster information)"""
        if item_id not in self.graphs:
            graph_path = self.graphs_dir / f"{item_id}.json"
            if graph_path.exists():
                logger.info(f"Loading hierarchical graph: {graph_path}")
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    
                # Verify if cluster information is included
                if 'cluster_nodes' not in graph_data:
                    logger.warning(f"Graph {item_id} does not contain cluster nodes!")
                    graph_data['cluster_nodes'] = []
                    graph_data['cluster_edges'] = []
                else:
                    n_clusters = len(graph_data['cluster_nodes'])
                    n_cluster_edges = len(graph_data.get('cluster_edges', []))
                    logger.info(f"  Contains {n_clusters} cluster nodes, {n_cluster_edges} cluster edges")
                    
                self.graphs[item_id] = graph_data
            else:
                logger.warning(f"Graph file does not exist: {graph_path}")
                self.graphs[item_id] = {
                    'nodes': [], 
                    'edges': [],
                    'cluster_nodes': [],
                    'cluster_edges': []
                }
        return self.graphs[item_id]
    
    def get_node_by_id(self, node_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get node information by node ID (supports both regular and cluster nodes)"""
        if item_id not in self.graphs:
            return None
            
        graph = self.graphs[item_id]
        
        # First search in regular nodes
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
        
        # Search in cluster nodes
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
        """Get 1-hop neighbor nodes of specified node
        
        Note: Only returns neighbors of regular nodes, does not include cluster relationships
        """
        if item_id not in self.graphs:
            return []
            
        graph = self.graphs[item_id]
        neighbor_ids = set()
        
        # Only search original edges (not cluster edges)
        for edge in graph.get('edges', []):
            if edge['source'] == node_id:
                neighbor_ids.add(edge['target'])
            elif edge['target'] == node_id:
                neighbor_ids.add(edge['source'])
        
        # Get neighbor node information
        neighbors = []
        for neighbor_id in neighbor_ids:
            neighbor_node = self.get_node_by_id(neighbor_id, item_id)
            if neighbor_node and neighbor_node.get('node_type') == 'normal':
                relations = self.get_node_relations(node_id, neighbor_id, item_id)
                neighbor_node['relations'] = relations
                neighbors.append(neighbor_node)
                
        return neighbors
    
    def get_node_relations(self, source_id: str, target_id: str, item_id: str) -> List[Dict[str, Any]]:
        """Get relationships between two nodes (only search original edges)"""
        if item_id not in self.graphs:
            return []
            
        relations = []
        graph = self.graphs[item_id]
        
        # Only search original edges
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
        """Get all member nodes of a cluster
        
        Args:
            cluster_id: Cluster node ID
            item_id: Item ID
            
        Returns:
            List of member nodes
        """
        if item_id not in self.graphs:
            return []
        
        graph = self.graphs[item_id]
        
        # Find cluster node
        cluster_node = None
        for c_node in graph.get('cluster_nodes', []):
            if c_node['id'] == cluster_id:
                cluster_node = c_node
                break
        
        if not cluster_node:
            return []
        
        # Get complete information of all member nodes
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

