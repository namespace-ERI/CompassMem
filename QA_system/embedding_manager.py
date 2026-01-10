#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding calculation and similarity management module
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Embedding calculation manager"""
    
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
        logger.info(f"Embedding model loaded to device: {self.device}")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """Average pooling"""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        masked = last_hidden_states * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    def compute_embedding(self, text: str, is_query: bool = True) -> np.ndarray:
        """Calculate embedding for single text
        
        Args:
            text: Text to encode
            is_query: Whether it is query text (kept for compatibility, bge-m3 doesn't need prefix)
        """
        # bge-m3 model does not need prefix
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
        """Calculate cosine similarity"""
        dot_product = np.dot(query_embedding, node_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_node = np.linalg.norm(node_embedding)
        
        if norm_query == 0 or norm_node == 0:
            return 0.0
            
        return dot_product / (norm_query * norm_node)
    
    def find_best_node(self, question: str, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single node most similar to question as entry point"""
        # Get question embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # Calculate similarity with all nodes
        best_similarity = -1
        best_node = None
        
        for node in graph.get('nodes', []):
            if 'texts' in node and node['texts']:
                # Use first text as main content for embedding
                node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
                node_embedding = self.compute_embedding(node_text, is_query=False)
                
                similarity = self.compute_similarity(question_embedding, node_embedding)
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_node = {
                        'node_id': node['id'],
                        'similarity': similarity,
                        'texts': node['texts'],
                        'summaries': node.get('summaries', []),
                        'people': node.get('people', []),
                        'time_explicit': node.get('time_explicit', []),
                        'utterance_refs': node.get('utterance_refs', [])
                    }
        
        return best_node
    
    def find_top_k_nodes(self, question: str, graph: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Find top-k nodes most similar to question as entry points, prefer nodes from different sessions
        
        Args:
            question: Query question
            graph: Graph data
            k: Number of top nodes to return
            
        Returns:
            List of top-k nodes with similarity scores, preferring nodes from different sessions
        """
        # Get question embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # Calculate similarity with all nodes
        node_similarities = []
        
        for node in graph.get('nodes', []):
            if 'texts' in node and node['texts']:
                # Use first text as main content for embedding
                node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
                node_embedding = self.compute_embedding(node_text, is_query=False)
                
                similarity = self.compute_similarity(question_embedding, node_embedding)
                
                if similarity >= self.similarity_threshold:
                    node_similarities.append({
                        'node_id': node['id'],
                        'similarity': similarity,
                        'texts': node['texts'],
                        'summaries': node.get('summaries', []),
                        'people': node.get('people', []),
                        'time_explicit': node.get('time_explicit', []),
                        'utterance_refs': node.get('utterance_refs', []),
                        'session_ids': node.get('session_ids', [])  # Keep session information
                    })
        
        # Sort by similarity in descending order
        node_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Prefer selecting nodes from different sessions
        selected_nodes = []
        used_sessions = set()
        
        # First round: only select first session for each node (main source)
        for node in node_similarities:
            if len(selected_nodes) >= k:
                break
            
            # Get primary session of this node (first session_id)
            node_sessions = node.get('session_ids', [])
            if not node_sessions:
                # If no session information, can still select
                selected_nodes.append(node)
                continue
            
            primary_session = node_sessions[0]
            
            # If this session hasn't been used, select this node
            if primary_session not in used_sessions:
                selected_nodes.append(node)
                used_sessions.add(primary_session)
        
        # Second round: if haven't reached k nodes, consider selecting high similarity nodes from used sessions
        if len(selected_nodes) < k:
            for node in node_similarities:
                if len(selected_nodes) >= k:
                    break
                
                # Skip already selected nodes
                if node in selected_nodes:
                    continue
                
                selected_nodes.append(node)
        
        logger.info(f"Selected {len(selected_nodes)} nodes from {len(used_sessions)} different sessions")
        
        return selected_nodes
