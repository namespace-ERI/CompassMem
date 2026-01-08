#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding计算和相似度管理模块
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Embedding计算管理器"""
    
    def __init__(self, model_name: str, gpu_id: int = 0, similarity_threshold: float = 0.7):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.similarity_threshold = similarity_threshold
        
        # 初始化模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # 设置设备
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Embedding模型已加载到设备: {self.device}")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """平均池化"""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        masked = last_hidden_states * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    def compute_embedding(self, text: str, is_query: bool = True) -> np.ndarray:
        """计算单个文本的embedding
        
        Args:
            text: 要编码的文本
            is_query: 是否为查询文本（仅为兼容性保留，bge-m3不需要前缀）
        """
        # bge-m3模型不需要添加前缀
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
        """计算余弦相似度"""
        dot_product = np.dot(query_embedding, node_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_node = np.linalg.norm(node_embedding)
        
        if norm_query == 0 or norm_node == 0:
            return 0.0
            
        return dot_product / (norm_query * norm_node)
    
    def find_best_node(self, question: str, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """找到与问题最相似的单个节点作为入口点"""
        # 获取问题embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # 计算与所有节点的相似度
        best_similarity = -1
        best_node = None
        
        for node in graph.get('nodes', []):
            if 'texts' in node and node['texts']:
                # 使用第一个text作为主要内容进行embedding
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
        """找到与问题最相似的 top-k 个节点作为入口点，优先选择来自不同session的节点
        
        Args:
            question: 查询问题
            graph: 图数据
            k: 返回的top节点数量
            
        Returns:
            List of top-k nodes with similarity scores, preferring nodes from different sessions
        """
        # 获取问题embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # 计算与所有节点的相似度
        node_similarities = []
        
        for node in graph.get('nodes', []):
            if 'texts' in node and node['texts']:
                # 使用第一个text作为主要内容进行embedding
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
                        'session_ids': node.get('session_ids', [])  # 保留session信息
                    })
        
        # 按相似度降序排序
        node_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 优先选择来自不同session的节点
        selected_nodes = []
        used_sessions = set()
        
        # 第一轮：每个节点只选择它的第一个session（主要来源）
        for node in node_similarities:
            if len(selected_nodes) >= k:
                break
            
            # 获取该节点的主要session（第一个session_id）
            node_sessions = node.get('session_ids', [])
            if not node_sessions:
                # 如果没有session信息，仍然可以选择
                selected_nodes.append(node)
                continue
            
            primary_session = node_sessions[0]
            
            # 如果该session尚未被使用，选择此节点
            if primary_session not in used_sessions:
                selected_nodes.append(node)
                used_sessions.add(primary_session)
        
        # 第二轮：如果还没达到k个，考虑从已使用session中选择高相似度节点
        if len(selected_nodes) < k:
            for node in node_similarities:
                if len(selected_nodes) >= k:
                    break
                
                # 跳过已经选择的节点
                if node in selected_nodes:
                    continue
                
                selected_nodes.append(node)
        
        logger.info(f"Selected {len(selected_nodes)} nodes from {len(used_sessions)} different sessions")
        
        return selected_nodes