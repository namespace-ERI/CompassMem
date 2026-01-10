#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical graph embedding computation and similarity management module V3
Supports cluster node matching and member node selection
New: Direct retrieval of all node summaries (for improved initial localization)
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class HierarchicalEmbeddingManagerV3:
    """Hierarchical embedding computation manager V3 - supports direct node retrieval"""
    
    def __init__(self, model_name: str, gpu_id: int = 0, similarity_threshold: float = 0.7):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.similarity_threshold = similarity_threshold
        
        # Initialize model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Hierarchical Embedding model V3 loaded to device: {self.device}")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """Average pooling"""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        masked = last_hidden_states * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    def compute_embedding(self, text: str, is_query: bool = True) -> np.ndarray:
        """Compute embedding for a single text
        
        Args:
            text: Text to encode
            is_query: Whether it's a query text (kept for compatibility, bge-m3 doesn't need prefix)
        """
        # bge-m3 model doesn't need prefix
        text = text.strip()
            
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().astype(np.float32)[0]
    
    def compute_similarity(self, query_embedding: np.ndarray, node_embedding: np.ndarray) -> float:
        """Compute cosine similarity"""
        dot_product = np.dot(query_embedding, node_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_node = np.linalg.norm(node_embedding)
        
        if norm_query == 0 or norm_node == 0:
            return 0.0
            
        return float(dot_product / (norm_query * norm_node))
    
    def find_top_k_nodes_by_summary(self,
                                    question: str,
                                    graph: Dict[str, Any],
                                    k: int = 5) -> List[Dict[str, Any]]:
        """[New] Directly retrieve from all node summaries to find top-k most relevant nodes
        
        This is V3's core improvement: directly find nodes instead of finding clusters first
        
        Args:
            question: Question text
            graph: Graph data (containing all nodes)
            k: Number of nodes to return
            
        Returns:
            Top-k node list, each containing: node_id, similarity, summary, cluster_id, etc.
        """
        # Get question embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # Get all non-cluster nodes (exclude cluster nodes)
        all_nodes = []
        for node in graph.get('nodes', []):
            # Skip cluster nodes
            if node['id'].startswith('cluster_'):
                continue
            # Must have summary
            summaries = node.get('summaries', [])
            if not summaries or not summaries[0]:
                continue
            
            all_nodes.append(node)
        
        if not all_nodes:
            logger.warning("No valid nodes (with summary) in graph!")
            return []
        
        logger.info(f"Starting retrieval from {len(all_nodes)} node summaries...")
        
        # Calculate similarity with all node summaries
        similarities = []
        
        for node in all_nodes:
            node_id = node['id']
            # Use summary as retrieval text
            summary = node.get('summaries', [''])[0]
            
            # If node already has embedding, use it directly; otherwise compute summary embedding
            if 'embedding' in node and node['embedding']:
                node_embedding = np.array(node['embedding'], dtype=np.float32)
            else:
                node_embedding = self.compute_embedding(summary, is_query=False)
            
            similarity = self.compute_similarity(question_embedding, node_embedding)
            
            similarities.append({
                'node_id': node_id,
                'similarity': similarity,
                'summary': summary,
                'texts': node.get('texts', []),
                'summaries': node.get('summaries', []),
                'people': node.get('people', []),
                'time_explicit': node.get('time_explicit', []),
                'utterance_refs': node.get('utterance_refs', []),
                'embedding': node.get('embedding', None),
                'session_ids': node.get('session_ids', []),
                # Record which cluster this node belongs to (for later selection from cluster)
                'cluster_id': self._find_cluster_for_node(node_id, graph)
            })
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k
        top_k = similarities[:k]
        
        logger.info(f"Found {len(all_nodes)} nodes, returning top-{k}")
        
        return top_k
    
    def _find_cluster_for_node(self, node_id: str, graph: Dict[str, Any]) -> Optional[str]:
        """Find the cluster ID that a node belongs to
        
        Args:
            node_id: Node ID
            graph: Graph data
            
        Returns:
            Cluster node ID, or None if not found
        """
        cluster_nodes = graph.get('cluster_nodes', [])
        for cluster in cluster_nodes:
            member_nodes = cluster.get('member_nodes', [])
            if node_id in member_nodes:
                return cluster['id']
        return None
    
    def find_top_k_cluster_nodes(self, 
                                  question: str, 
                                  graph: Dict[str, Any], 
                                  k: int = 5) -> List[Dict[str, Any]]:
        """Find top-k cluster nodes most similar to the question (kept for compatibility)
        
        Args:
            question: Question text
            graph: Graph data (containing cluster_nodes)
            k: Number of cluster nodes to return
            
        Returns:
            Top-k cluster node list, each containing: node_id, similarity, embedding, member_nodes, etc.
        """
        # Get question embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # Get cluster nodes
        cluster_nodes = graph.get('cluster_nodes', [])
        
        if not cluster_nodes:
            logger.warning("No cluster nodes in graph!")
            return []
        
        # Calculate similarity with all cluster nodes
        similarities = []
        
        for cluster_node in cluster_nodes:
            cluster_id = cluster_node['id']
            cluster_embedding = cluster_node.get('embedding', None)
            
            if cluster_embedding is None:
                logger.warning(f"Cluster node {cluster_id} missing embedding")
                continue
            
            # Directly use stored cluster embedding (already averaged)
            cluster_embedding_np = np.array(cluster_embedding, dtype=np.float32)
            similarity = self.compute_similarity(question_embedding, cluster_embedding_np)
            
            similarities.append({
                'node_id': cluster_id,
                'cluster_id': cluster_node['cluster_id'],
                'similarity': similarity,
                'embedding': cluster_embedding,
                'member_nodes': cluster_node.get('member_nodes', []),
                'n_members': cluster_node.get('n_members', 0),
                'session_ids': cluster_node.get('session_ids', []),
                'people': cluster_node.get('people', [])
            })
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k
        top_k = similarities[:k]
        
        logger.info(f"Found {len(cluster_nodes)} cluster nodes, returning top-{k}:")
        for i, node in enumerate(top_k):
            logger.info(f"  {i+1}. {node['node_id']} (similarity: {node['similarity']:.3f}, "
                       f"members: {node['n_members']})")
        
        return top_k
    
    def find_top_k_members_in_cluster(self,
                                     question: str,
                                     cluster: Dict[str, Any],
                                     graph: Dict[str, Any],
                                     k: int = 3) -> List[Dict[str, Any]]:
        """[New] Find top-k most relevant nodes among member nodes in specified cluster
        
        Args:
            question: Question text
            cluster: Cluster information (containing member_nodes)
            graph: Graph data
            k: Number of nodes to return
            
        Returns:
            Top-k member node list
        """
        member_node_ids = cluster.get('member_nodes', [])
        if not member_node_ids:
            logger.warning(f"Cluster {cluster.get('id', 'unknown')} has no member nodes")
            return []
        
        question_embedding = self.compute_embedding(question, is_query=True)
        
        member_similarities = []
        
        for node in graph.get('nodes', []):
            if node['id'] not in member_node_ids:
                continue
            
            # Use node's embedding (if available)
            if 'embedding' in node and node['embedding']:
                node_embedding = np.array(node['embedding'], dtype=np.float32)
            # Otherwise compute embedding from text
            elif 'texts' in node and node['texts']:
                node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
                node_embedding = self.compute_embedding(node_text, is_query=False)
            else:
                continue
            
            similarity = self.compute_similarity(question_embedding, node_embedding)
            
            member_similarities.append({
                'node_id': node['id'],
                'similarity': similarity,
                'texts': node.get('texts', []),
                'summaries': node.get('summaries', []),
                'people': node.get('people', []),
                'time_explicit': node.get('time_explicit', []),
                'utterance_refs': node.get('utterance_refs', []),
                'embedding': node.get('embedding', None)
            })
        
        # Sort by similarity in descending order
        member_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k
        top_k = member_similarities[:k]
        
        if top_k:
            logger.info(f"Found top-{k} from {len(member_node_ids)} members in cluster {cluster.get('id', 'unknown')}:")
            for i, node in enumerate(top_k):
                logger.info(f"    {i+1}. {node['node_id']} (similarity: {node['similarity']:.3f})")
        
        return top_k
    
    def find_best_member_node(self, 
                             question: str, 
                             member_node_ids: List[str],
                             graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the node most similar to the question among cluster member nodes (kept for compatibility)
        
        Args:
            question: Question text
            member_node_ids: List of member node IDs
            graph: Graph data
            
        Returns:
            Best member node information
        """
        question_embedding = self.compute_embedding(question, is_query=True)
        
        best_similarity = -1
        best_node = None
        
        for node in graph.get('nodes', []):
            if node['id'] not in member_node_ids:
                continue
                
            # Use node's embedding (if available)
            if 'embedding' in node and node['embedding']:
                node_embedding = np.array(node['embedding'], dtype=np.float32)
            # Otherwise compute embedding from text
            elif 'texts' in node and node['texts']:
                node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
                node_embedding = self.compute_embedding(node_text, is_query=False)
            else:
                continue
            
            similarity = self.compute_similarity(question_embedding, node_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_node = {
                    'node_id': node['id'],
                    'similarity': similarity,
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None)
                }
        
        if best_node:
            logger.info(f"Found best node among members: {best_node['node_id']} "
                       f"(similarity: {best_node['similarity']:.3f})")
        
        return best_node
    
    def compute_node_similarity_to_query(self,
                                         question: str,
                                         node: Dict[str, Any]) -> float:
        """Compute similarity between a single node and the question
        
        Args:
            question: Question text
            node: Node information
            
        Returns:
            Similarity score
        """
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # Use node's embedding (if available)
        if 'embedding' in node and node['embedding']:
            node_embedding = np.array(node['embedding'], dtype=np.float32)
        # Otherwise compute embedding from text
        elif 'texts' in node and node['texts']:
            node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
            node_embedding = self.compute_embedding(node_text, is_query=False)
        else:
            return 0.0
        
        return self.compute_similarity(question_embedding, node_embedding)
    
    def compute_node_max_similarity_to_subgoals(self,
                                               node: Dict[str, Any],
                                               unsatisfied_subgoals: List[str]) -> float:
        """Compute maximum similarity between node and unsatisfied subgoals (for queue sorting)
        
        Args:
            node: Node information (containing summary or texts)
            unsatisfied_subgoals: List of unsatisfied subgoal texts
            
        Returns:
            Maximum similarity score with any subgoal
        """
        if not unsatisfied_subgoals:
            return 0.0
        
        # Get node's text representation (prefer summary)
        node_text = ''
        if 'summaries' in node and node['summaries']:
            node_text = node['summaries'][0] if isinstance(node['summaries'], list) else str(node['summaries'])
        elif 'texts' in node and node['texts']:
            node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
        
        if not node_text:
            return 0.0
        
        # Compute node embedding (prefer existing one)
        if 'embedding' in node and node['embedding']:
            node_embedding = np.array(node['embedding'], dtype=np.float32)
        else:
            node_embedding = self.compute_embedding(node_text, is_query=False)
        
        # Compute similarity with each unsatisfied subgoal, take maximum
        max_similarity = 0.0
        for subgoal in unsatisfied_subgoals:
            subgoal_embedding = self.compute_embedding(subgoal, is_query=True)
            similarity = self.compute_similarity(subgoal_embedding, node_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def sort_nodes_by_subgoal_relevance(self,
                                       nodes: List[Dict[str, Any]],
                                       unsatisfied_subgoals: List[str]) -> List[Dict[str, Any]]:
        """Sort node list by similarity to unsatisfied subgoals
        
        Args:
            nodes: Node list
            unsatisfied_subgoals: List of unsatisfied subgoal texts
            
        Returns:
            Node list sorted by similarity in descending order
        """
        if not nodes or not unsatisfied_subgoals:
            return nodes
        
        # Compute maximum similarity for each node
        nodes_with_scores = []
        for node in nodes:
            score = self.compute_node_max_similarity_to_subgoals(node, unsatisfied_subgoals)
            nodes_with_scores.append((node, score))
        
        # Sort by score in descending order
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted node list
        sorted_nodes = [node for node, score in nodes_with_scores]
        
        logger.debug(f"Sorted {len(nodes)} nodes by subgoal relevance. Top scores: "
                    f"{[score for _, score in nodes_with_scores[:3]]}")
        
        return sorted_nodes

