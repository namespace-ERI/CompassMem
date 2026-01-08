#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化图的Embedding计算和相似度管理模块 V3
支持聚类节点匹配和成员节点选择
新增：直接检索所有节点summary的功能（用于改进首次定位）
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class HierarchicalEmbeddingManagerV3:
    """层次化Embedding计算管理器 V3 - 支持直接节点检索"""
    
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
        logger.info(f"层次化Embedding模型V3已加载到设备: {self.device}")
    
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
            
        return float(dot_product / (norm_query * norm_node))
    
    def find_top_k_nodes_by_summary(self,
                                    question: str,
                                    graph: Dict[str, Any],
                                    k: int = 5) -> List[Dict[str, Any]]:
        """【新增】直接在所有节点的summary中检索，找到top-k个最相关的节点
        
        这是V3的核心改进：不再先找聚类，而是直接找节点
        
        Args:
            question: 问题文本
            graph: 图数据（包含所有nodes）
            k: 返回的节点数量
            
        Returns:
            Top-k节点列表，每个包含: node_id, similarity, summary, cluster_id等信息
        """
        # 获取问题embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # 获取所有非聚类节点（排除cluster节点）
        all_nodes = []
        for node in graph.get('nodes', []):
            # 跳过聚类节点
            if node['id'].startswith('cluster_'):
                continue
            # 必须有summary
            summaries = node.get('summaries', [])
            if not summaries or not summaries[0]:
                continue
            
            all_nodes.append(node)
        
        if not all_nodes:
            logger.warning("图中没有有效的节点（含summary）！")
            return []
        
        logger.info(f"开始在 {len(all_nodes)} 个节点的summary中检索...")
        
        # 计算与所有节点summary的相似度
        similarities = []
        
        for node in all_nodes:
            node_id = node['id']
            # 使用summary作为检索文本
            summary = node.get('summaries', [''])[0]
            
            # 如果节点已经有embedding，直接使用；否则计算summary的embedding
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
                # 记录这个节点属于哪个聚类（用于后续从聚类中选择）
                'cluster_id': self._find_cluster_for_node(node_id, graph)
            })
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回top-k
        top_k = similarities[:k]
        
        logger.info(f"找到 {len(all_nodes)} 个节点，返回top-{k}")
        # for i, node in enumerate(top_k):
        #     cluster_info = f", cluster: {node['cluster_id']}" if node['cluster_id'] else ", cluster: None"
        #     logger.info(f"  {i+1}. {node['node_id']} (similarity: {node['similarity']:.3f}{cluster_info})")
        #     logger.info(f"      Summary: {node['summary'][:100]}...")
        
        return top_k
    
    def _find_cluster_for_node(self, node_id: str, graph: Dict[str, Any]) -> Optional[str]:
        """查找节点所属的聚类ID
        
        Args:
            node_id: 节点ID
            graph: 图数据
            
        Returns:
            聚类节点的ID，如果找不到返回None
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
        """找到与问题最相似的top-k个聚类节点（保留用于兼容性）
        
        Args:
            question: 问题文本
            graph: 图数据（包含cluster_nodes）
            k: 返回的聚类节点数量
            
        Returns:
            Top-k聚类节点列表，每个包含: node_id, similarity, embedding, member_nodes等信息
        """
        # 获取问题embedding
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # 获取聚类节点
        cluster_nodes = graph.get('cluster_nodes', [])
        
        if not cluster_nodes:
            logger.warning("图中没有聚类节点！")
            return []
        
        # 计算与所有聚类节点的相似度
        similarities = []
        
        for cluster_node in cluster_nodes:
            cluster_id = cluster_node['id']
            cluster_embedding = cluster_node.get('embedding', None)
            
            if cluster_embedding is None:
                logger.warning(f"聚类节点 {cluster_id} 缺少embedding")
                continue
            
            # 直接使用存储的聚类embedding（已经是平均值）
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
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回top-k
        top_k = similarities[:k]
        
        logger.info(f"找到 {len(cluster_nodes)} 个聚类节点，返回top-{k}:")
        for i, node in enumerate(top_k):
            logger.info(f"  {i+1}. {node['node_id']} (similarity: {node['similarity']:.3f}, "
                       f"members: {node['n_members']})")
        
        return top_k
    
    def find_top_k_members_in_cluster(self,
                                     question: str,
                                     cluster: Dict[str, Any],
                                     graph: Dict[str, Any],
                                     k: int = 3) -> List[Dict[str, Any]]:
        """【新增】在指定聚类的成员节点中找到top-k个最相关的节点
        
        Args:
            question: 问题文本
            cluster: 聚类信息（包含member_nodes）
            graph: 图数据
            k: 返回的节点数量
            
        Returns:
            Top-k成员节点列表
        """
        member_node_ids = cluster.get('member_nodes', [])
        if not member_node_ids:
            logger.warning(f"聚类 {cluster.get('id', 'unknown')} 没有成员节点")
            return []
        
        question_embedding = self.compute_embedding(question, is_query=True)
        
        member_similarities = []
        
        for node in graph.get('nodes', []):
            if node['id'] not in member_node_ids:
                continue
            
            # 使用节点的embedding（如果有）
            if 'embedding' in node and node['embedding']:
                node_embedding = np.array(node['embedding'], dtype=np.float32)
            # 否则使用文本计算embedding
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
        
        # 按相似度降序排序
        member_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回top-k
        top_k = member_similarities[:k]
        
        if top_k:
            logger.info(f"从聚类 {cluster.get('id', 'unknown')} 的 {len(member_node_ids)} 个成员中找到top-{k}:")
            for i, node in enumerate(top_k):
                logger.info(f"    {i+1}. {node['node_id']} (similarity: {node['similarity']:.3f})")
        
        return top_k
    
    def find_best_member_node(self, 
                             question: str, 
                             member_node_ids: List[str],
                             graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """在聚类的成员节点中找到与问题最相似的节点（保留用于兼容性）
        
        Args:
            question: 问题文本
            member_node_ids: 成员节点ID列表
            graph: 图数据
            
        Returns:
            最佳成员节点信息
        """
        question_embedding = self.compute_embedding(question, is_query=True)
        
        best_similarity = -1
        best_node = None
        
        for node in graph.get('nodes', []):
            if node['id'] not in member_node_ids:
                continue
                
            # 使用节点的embedding（如果有）
            if 'embedding' in node and node['embedding']:
                node_embedding = np.array(node['embedding'], dtype=np.float32)
            # 否则使用文本计算embedding
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
            logger.info(f"在成员节点中找到最佳节点: {best_node['node_id']} "
                       f"(similarity: {best_node['similarity']:.3f})")
        
        return best_node
    
    def compute_node_similarity_to_query(self,
                                         question: str,
                                         node: Dict[str, Any]) -> float:
        """计算单个节点与问题的相似度
        
        Args:
            question: 问题文本
            node: 节点信息
            
        Returns:
            相似度分数
        """
        question_embedding = self.compute_embedding(question, is_query=True)
        
        # 使用节点的embedding（如果有）
        if 'embedding' in node and node['embedding']:
            node_embedding = np.array(node['embedding'], dtype=np.float32)
        # 否则使用文本计算embedding
        elif 'texts' in node and node['texts']:
            node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
            node_embedding = self.compute_embedding(node_text, is_query=False)
        else:
            return 0.0
        
        return self.compute_similarity(question_embedding, node_embedding)
    
    def compute_node_max_similarity_to_subgoals(self,
                                               node: Dict[str, Any],
                                               unsatisfied_subgoals: List[str]) -> float:
        """计算节点与未满足subgoals的最大相似度（用于队列排序）
        
        Args:
            node: 节点信息（包含summary或texts）
            unsatisfied_subgoals: 未满足的subgoal文本列表
            
        Returns:
            与任意subgoal的最大相似度分数
        """
        if not unsatisfied_subgoals:
            return 0.0
        
        # 获取节点的文本表示（优先使用summary）
        node_text = ''
        if 'summaries' in node and node['summaries']:
            node_text = node['summaries'][0] if isinstance(node['summaries'], list) else str(node['summaries'])
        elif 'texts' in node and node['texts']:
            node_text = node['texts'][0] if isinstance(node['texts'], list) else str(node['texts'])
        
        if not node_text:
            return 0.0
        
        # 计算节点的embedding（优先使用已有的）
        if 'embedding' in node and node['embedding']:
            node_embedding = np.array(node['embedding'], dtype=np.float32)
        else:
            node_embedding = self.compute_embedding(node_text, is_query=False)
        
        # 计算与每个未满足subgoal的相似度，取最大值
        max_similarity = 0.0
        for subgoal in unsatisfied_subgoals:
            subgoal_embedding = self.compute_embedding(subgoal, is_query=True)
            similarity = self.compute_similarity(subgoal_embedding, node_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def sort_nodes_by_subgoal_relevance(self,
                                       nodes: List[Dict[str, Any]],
                                       unsatisfied_subgoals: List[str]) -> List[Dict[str, Any]]:
        """根据节点与未满足subgoals的相似度对节点列表排序
        
        Args:
            nodes: 节点列表
            unsatisfied_subgoals: 未满足的subgoal文本列表
            
        Returns:
            按相似度降序排序的节点列表
        """
        if not nodes or not unsatisfied_subgoals:
            return nodes
        
        # 计算每个节点的最大相似度
        nodes_with_scores = []
        for node in nodes:
            score = self.compute_node_max_similarity_to_subgoals(node, unsatisfied_subgoals)
            nodes_with_scores.append((node, score))
        
        # 按分数降序排序
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的节点列表
        sorted_nodes = [node for node, score in nodes_with_scores]
        
        logger.debug(f"Sorted {len(nodes)} nodes by subgoal relevance. Top scores: "
                    f"{[score for _, score in nodes_with_scores[:3]]}")
        
        return sorted_nodes

