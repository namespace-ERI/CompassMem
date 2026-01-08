#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于层次化图的多轮检索问答系统 - V3 改进版
支持两阶段检索策略 + 三动作机制 + 多节点队列管理

主要改进（V3相较于V2）：
1. **改进的两阶段检索策略**：
   - 阶段1：检索大量候选节点（如top-50或top-100）
   - 阶段2：取前top5节点直接加入探索队列
   - 阶段3：继续遍历检索结果，按顺序收集聚类ID（去重），直到得到top3个聚类
   - 阶段4：从这些聚类中使用LLM选择成员节点
   - 阶段5：所有节点加入队列，开始多路径探索
2. 不再依赖聚类中心embedding的精确度
3. 提高召回率，降低首次定位的误判

使用场景：
- 当聚类中心embedding不够精确时
- 需要更高的召回率时
- 希望首次定位更准确时
"""

import json
import logging
import argparse
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入层次化模块
from hierarchical_graph_loader import HierarchicalGraphLoader
from hierarchical_embedding_manager_v3 import HierarchicalEmbeddingManagerV3
from cluster_explorer import ClusterExplorer

# 导入原有模块
from conversation_manager import ConversationManager
from debug_logger import DebugLogger

# 导入V2新模块（队列版本+ 并发）
from llm_handler_v2_queue import LLMHandlerV2Queue
from context_formatter_v2 import ContextFormatterV2
from multi_path_explorer_v2_queue import (
    MultiPathExplorerV2Queue, 
    PathExplorerV2Queue,
    GlobalNodeQueue,
    SharedExplorationState
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HierarchicalGraphQASystemV3:
    """层次化图问答系统 - V3改进版（两阶段检索 + 三动作机制 + 多节点队列）"""
    
    def __init__(self, 
                 graphs_dir: str = "/share/project/zyt/hyy/Memory/build_graph/graphs_llm_clustered",
                 qa_data_path: str = "/share/project/zyt/hyy/Memory/data/locomo/locomo10.json",
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 embedding_model: str = "/share/project/zyt/hyy/Model/bge-m3",
                 top_k_nodes: int = 5,  # V3: 首次直接检索节点的数量
                 top_k_per_cluster: int = 3,  # V3: 从每个聚类中选择的节点数量
                 n_paths: int = 3,  # V3: 固定的探索路径数量
                 similarity_threshold: float = 0.7,
                 max_rounds: int = 3,
                 debug_mode: bool = False,
                 debug_items: int = 1,
                 debug_qa_per_item: int = 2,
                 embedding_gpu_id: int = 0,
                 api_base: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY",
                 enable_refinement: bool = True,
                 use_relation: bool = True,
                 enable_early_stopping: bool = False,  # V3.5新增：早停开关，默认关闭
                 use_subgoal_planning: bool = True,  # V3.5新增：是否使用subgoal规划
                 enable_concurrent: bool = False):  # V3.6新增：是否启用并发探索
        """初始化层次化问答系统V3版本 - OpenAI API版本（支持并发）"""
        self.max_rounds = max_rounds
        self.top_k_nodes = top_k_nodes  # V3新增
        self.top_k_per_cluster = top_k_per_cluster  # V3新增
        self.n_paths = n_paths  # V3新增：固定路径数量
        self.debug_mode = debug_mode
        self.debug_items = debug_items
        self.debug_qa_per_item = debug_qa_per_item
        self.qa_data_path = qa_data_path
        self.enable_refinement = enable_refinement
        self.use_relation = use_relation
        self.api_base = api_base
        self.api_key = api_key
        self.enable_early_stopping = enable_early_stopping  # V3.5新增
        self.use_subgoal_planning = use_subgoal_planning  # V3.5新增
        self.enable_concurrent = enable_concurrent  # V3.6新增
        
        # 统计信息
        self.stats = {
            'total_questions': 0,
            'skipped_questions': 0,
            'all_skip_paths': 0,
            'total_paths': 0,
            'total_multi_node_selections': 0,
            'avg_nodes_per_selection': 0.0,
            'total_direct_nodes': 0,  # V3新增：直接检索的节点数
            'total_clusters_explored': 0,  # V3新增：探索的聚类数
            'total_cluster_nodes': 0,  # V3新增：从聚类中选择的节点数
            'avg_direct_node_similarity': 0.0,  # V3新增：直接检索节点的平均相似度
            # Token统计
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'llm_call_count': 0,
        }
        
        # 记录all-skip案例的详细信息
        self.all_skip_cases = []
        
        # 初始化各个管理器
        logger.info("初始化层次化系统V3组件...")
        
        # 层次化图数据加载器
        self.graph_loader = HierarchicalGraphLoader(graphs_dir)
        
        # 对话数据管理器
        self.conversation_manager = ConversationManager(qa_data_path)
        
        # 层次化Embedding管理器V3
        self.embedding_manager = HierarchicalEmbeddingManagerV3(
            embedding_model, 
            embedding_gpu_id, 
            similarity_threshold
        )
        
        # LLM处理器V2 Queue版本 - OpenAI API版本
        self.llm_handler = LLMHandlerV2Queue(
            model_name, 
            api_base, 
            api_key
        )
        
        # 上下文格式化器V2
        self.context_formatter = ContextFormatterV2(
            self.conversation_manager,
            self.graph_loader,
            use_relation
        )
        
        # 聚类探索器（用于辅助选择）
        self.cluster_explorer = ClusterExplorer(
            self.graph_loader,
            self.embedding_manager,
            self.llm_handler,  
            self.context_formatter
        )
        
        # 多路探索器V2 Queue版本
        self.multi_path_explorer = MultiPathExplorerV2Queue(
            self.graph_loader,
            self.embedding_manager,
            self.llm_handler,
            self.context_formatter,
            max_rounds
        )
        
        # 调试日志器
        self.debug_logger = DebugLogger(debug_mode)
        
        # 加载QA数据
        self.qa_data = self._load_qa_data()
        
        # 显示GPU内存使用情况（仅Embedding模型）
        self._log_gpu_memory_usage(embedding_gpu_id)
        
        logger.info(f"=== V3 改进版参数 ===")
        logger.info(f"首次直接检索节点数 (top_k_nodes): {top_k_nodes}")
        logger.info(f"每个聚类选择节点数 (top_k_per_cluster): {top_k_per_cluster}")
        logger.info(f"最大探索轮数: {max_rounds}")
        logger.info(f"LLM API Base: {api_base}")
        logger.info(f"LLM Model: {model_name}")
        logger.info(f"====================")
    
    def _load_qa_data(self) -> List[Dict[str, Any]]:
        """加载QA数据"""
        logger.info(f"加载QA数据: {self.qa_data_path}")
        with open(self.qa_data_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        logger.info(f"加载了 {len(qa_data)} 个QA项")
        return qa_data
    
    def _log_gpu_memory_usage(self, embedding_gpu_id: int):
        """记录GPU内存使用情况（仅Embedding模型）"""
        if torch.cuda.is_available():
            logger.info("=== GPU内存使用情况 ===")
            logger.info(f"Embedding模型使用GPU: {embedding_gpu_id}")
            logger.info(f"LLM模型: 通过API调用（后端部署）")
            
            # 只显示embedding GPU的信息
            if embedding_gpu_id < torch.cuda.device_count():
                i = embedding_gpu_id
                gpu_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_free = memory_total - memory_reserved
                
                logger.info(f"GPU {i} ({gpu_name}) [Embedding]:")
                logger.info(f"  总内存: {memory_total:.2f} GB")
                logger.info(f"  已分配: {memory_allocated:.2f} GB ({memory_allocated/memory_total*100:.1f}%)")
                logger.info(f"  已保留: {memory_reserved:.2f} GB ({memory_reserved/memory_total*100:.1f}%)")
                logger.info(f"  可用: {memory_free:.2f} GB ({memory_free/memory_total*100:.1f}%)")
    
    def answer_question(self, question: str, item_id: str, category: int = None) -> Dict[str, Any]:
        """使用V3两阶段检索策略回答问题"""
        start_time = time.time()
        
        logger.info(f"Processing question: {question[:80]}... (graph: {item_id})")
        logger.info(f"Mode: Two-Stage Retrieval (Direct nodes: {self.top_k_nodes}, "
                   f"Per-cluster: {self.top_k_per_cluster}), "
                   f"Refinement: {'Enabled' if self.enable_refinement else 'Disabled'}, "
                   f"Use Relation: {'Yes' if self.use_relation else 'No'}, "
                   f"Subgoal Planning: {'Enabled' if self.use_subgoal_planning else 'Disabled'}, "
                   f"Early Stopping: {'Enabled' if self.enable_early_stopping else 'Disabled'}, "
                   f"Concurrent: {'Enabled' if self.enable_concurrent else 'Disabled'}")
        
        # V3.5新增：使用planner生成subgoals
        subgoals = []
        planner_raw_response = ""
        if self.use_subgoal_planning:
            logger.info("=== Planner: Decomposing question into sub-goals ===")
            try:
                subgoals, planner_raw_response, _ = self.llm_handler.generate_subgoals(question)
                if not subgoals:
                    logger.warning("Planner failed to generate sub-goals, proceeding without subgoal tracking")
            except Exception as e:
                logger.error(f"Planner error: {e}, proceeding without subgoal tracking")
        
        # 加载对应的图
        graph = self.graph_loader.load_graph(item_id)
        
        # 检查图是否包含聚类信息
        if not graph.get('cluster_nodes'):
            logger.error(f"Graph {item_id} does not contain cluster nodes!")
            return {
                'question': question,
                'item_id': item_id,
                'answer': "",
                'exploration_type': 'hierarchical_v3_two_stage_subgoal',
                'error': 'No cluster nodes in graph'
            }
        
        # 两阶段检索策略（传入subgoals和category）
        result = self._answer_with_two_stage_retrieval(
            question, graph, item_id, subgoals, category
        )
        
        # 添加subgoal相关信息到结果
        result['subgoals'] = subgoals
        result['planner_response'] = planner_raw_response
        
        # 记录用时
        elapsed_time = time.time() - start_time
        result['elapsed_time_seconds'] = round(elapsed_time, 2)
        logger.info(f"✅ Question answered in {elapsed_time:.2f} seconds")
        
        return result
    
    def _select_members_from_cluster(self,
                                     question: str,
                                     cluster: Dict[str, Any],
                                     graph: Dict[str, Any],
                                     item_id: str) -> List[Dict[str, Any]]:
        """从单个聚类中使用 LLM 选择多个成员节点（基于 summary）
        
        这是从V2复制的方法，用于阶段4
        使用LLM基于summary选择，失败时fallback到embedding
        
        Returns:
            List of selected nodes (可能是多个)
        """
        member_ids = cluster.get('member_nodes', [])
        if not member_ids:
            return []
        
        # 获取成员节点的 summary 信息
        member_summaries = []
        for node in graph.get('nodes', []):
            if node['id'] in member_ids:
                summary_info = {
                    'node_id': node['id'],
                    'summary': node.get('summaries', [''])[0] if node.get('summaries') else '',
                    'people': node.get('people', []),
                    'time': node.get('time_explicit', [''])[0] if node.get('time_explicit') else ''
                }
                member_summaries.append(summary_info)
        
        if not member_summaries:
            return []
        
        # 使用LLM选择节点（可能多个）
        selected_node_ids, raw_response, formatted_prompt = self.llm_handler.select_nodes_from_cluster(
            question,
            cluster['id'],
            member_summaries
        )
        
        if selected_node_ids:
            logger.info(f"Agent selects {len(selected_node_ids)} 个节点: {selected_node_ids} from {cluster['id']}")
            selected_nodes = []
            for node_id in selected_node_ids:
                node = self.graph_loader.get_node_by_id(node_id, item_id)
                if node:
                    selected_nodes.append(node)
            return selected_nodes
        
        # 如果 LLM 没有明确选择，则不选择
        logger.warning(f"LLM does not select nodes, return none")
        # fallback_node = self.embedding_manager.find_best_member_node(
        #     question,
        #     member_ids,
        #     graph
        # )
        return []
    
    def _answer_with_two_stage_retrieval(self,
                                        question: str,
                                        graph: Dict[str, Any],
                                        item_id: str,
                                        subgoals: List[str] = None,
                                        category: int = None) -> Dict[str, Any]:
        """V3核心方法：两阶段检索策略（V3.5增强版）
        
        阶段1：直接检索所有节点的summary，获取大量候选节点（如top-50）
        阶段2：使用LLM从top-k候选节点中筛选真正相关的节点
        阶段3：继续往后遍历，按顺序去重收集聚类ID，直到得到top_k_per_cluster（默认3）个聚类
        阶段4：从这些聚类中使用LLM选择成员节点
        阶段5：所有节点加入队列，开始多路径探索（支持subgoal跟踪和优先级排序）
        """
        
        # 初始化全局subgoal状态（如果使用subgoal）
        global_subgoal_status = {}
        if subgoals:
            global_subgoal_status = {i: False for i in range(len(subgoals))}
        
        # ========== 阶段1：直接检索节点summary（检索更多候选）==========
        # 为了找到足够的聚类，需要检索更多的节点（比如50个）
        retrieval_pool_size = max(50, self.top_k_nodes * 10)  # 检索池大小
        logger.info(f"=== Stage 1: Retrieve nodes by summary ===")
        
        # 直接检索更多节点作为候选池
        all_retrieved_nodes = self.embedding_manager.find_top_k_nodes_by_summary(
            question,
            graph,
            k=retrieval_pool_size
        )
        
        if not all_retrieved_nodes:
            logger.warning("Stage 1: No relevant nodes found")
            return {
                'question': question,
                'item_id': item_id,
                'answer': "",
                'exploration_type': 'hierarchical_v3_two_stage_subgoal',
                'error': 'No relevant nodes found in stage 1'
            }
        
        logger.info(f"Stage 1 completed: Retrieved {len(all_retrieved_nodes)} candidate nodes")
        
        # ========== 阶段2：使用LLM从top-k候选节点中筛选（V3.5新增）==========
        logger.info(f"=== Stage 2: LLM selects from top {self.top_k_nodes} candidate nodes ===")
        
        # 准备候选节点（取前top_k_nodes个）
        direct_top_candidates = all_retrieved_nodes[:self.top_k_nodes]
        
        # 构建候选节点信息（给LLM看）
        candidate_nodes_for_llm = []
        for node in direct_top_candidates:
            candidate_nodes_for_llm.append({
                'node_id': node['node_id'],
                'summary': node.get('summaries', [''])[0] if node.get('summaries') else '',
                'similarity': node['similarity']
            })
        
        # V3.5新增：让LLM选择哪些节点真正相关
        start_nodes = []
        if subgoals:
            # 如果有subgoals，使用subgoal版本的筛选
            selected_node_ids, llm_selection_response, _ = self.llm_handler.select_top_k_nodes(
                question, subgoals, candidate_nodes_for_llm
            )
        else:
            # 向后兼容：如果没有subgoals，选择所有候选节点
            selected_node_ids = [n['node_id'] for n in direct_top_candidates]
            llm_selection_response = "No subgoal planning, using all candidates"
        
        # 根据LLM的选择构建起始节点列表
        for node in direct_top_candidates:
            if node['node_id'] in selected_node_ids:
                start_nodes.append({
                    'node_id': node['node_id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None),
                    'source': 'direct_retrieval_llm_selected',  # 标记来源
                    'similarity': node['similarity']
                })
        
        # 统计直接检索+LLM筛选的结果
        llm_selected = len(start_nodes) > 0
        if start_nodes:
            logger.info(f"Stage 2 completed: LLM selected {len(start_nodes)}/{len(direct_top_candidates)} nodes")
            logger.info(f"Selected nodes: {[n['node_id'] for n in start_nodes]}")
        else:
            logger.warning("Stage 2: LLM did not select any nodes, using fallback (top 2 candidates)")
            # Fallback：至少选择前2个
            for node in direct_top_candidates[:2]:
                start_nodes.append({
                    'node_id': node['node_id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None),
                    'source': 'direct_retrieval_fallback',
                    'similarity': node['similarity']
                })
        
        # 统一计算 avg_direct_similarity（确保在所有情况下都有值）
        if start_nodes:
            direct_similarities = [node['similarity'] for node in start_nodes]
            avg_direct_similarity = sum(direct_similarities) / len(direct_similarities)
            self.stats['total_direct_nodes'] += len(start_nodes)
            self.stats['avg_direct_node_similarity'] = avg_direct_similarity
            if llm_selected:
                logger.info(f"Average similarity of selected nodes: {avg_direct_similarity:.3f}")
            else:
                logger.info(f"Fallback nodes selected: {len(start_nodes)}, avg similarity: {avg_direct_similarity:.3f}")
        else:
            # 如果仍然没有节点（理论上不应该发生），设置默认值
            avg_direct_similarity = 0.0
            logger.warning("No direct nodes selected, setting avg_direct_similarity to 0.0")
        
        # ========== 阶段3：从所有检索结果中按顺序收集聚类ID（去重，直到得到top_k_per_cluster个）==========
        logger.info(f"=== Stage 3: Collect cluster nodes ===")
        
        # 按照节点检索顺序，遍历所有检索结果，收集聚类ID（保持顺序，去重）
        cluster_ids_ordered = []
        seen_clusters = set()
        last_checked_idx = 0
        
        # 如果 top_k_per_cluster 为 0，跳过聚类收集
        if self.top_k_per_cluster > 0:
            for i, node in enumerate(all_retrieved_nodes):
                last_checked_idx = i
                cluster_id = node.get('cluster_id')
                if cluster_id and cluster_id not in seen_clusters:
                    cluster_ids_ordered.append(cluster_id)
                    seen_clusters.add(cluster_id)
                    logger.debug(f"  Node #{i+1} ({node['node_id']}) -> New cluster: {cluster_id}")
                    
                    # 如果已经收集到足够的聚类，停止
                    if len(cluster_ids_ordered) >= self.top_k_per_cluster:
                        logger.info(f"  Collected {self.top_k_per_cluster} different clusters, stop traversal (at node #{i+1})")
                        break
        else:
            logger.info(f"  top_k_per_cluster=0, skip cluster collection")
        
        logger.info(f"Stage 3 completed: Collected {len(cluster_ids_ordered)} clusters from {last_checked_idx+1} retrieved nodes (in order of first appearance): {cluster_ids_ordered}")
        
        # ========== 阶段4：选择这些聚类进行成员节点选择 ==========
        logger.info(f"=== Stage 4: Select members from {len(cluster_ids_ordered)} clusters ===")
        
        # 如果没有聚类信息，直接使用阶段2的节点
        if not cluster_ids_ordered:
            logger.warning("Stage 4: No relevant clusters found, only use direct nodes from stage 2")
        else:
            # 注意：cluster_ids_ordered 在阶段3中已经收集好了（遍历所有检索节点直到得到足够的聚类）
            # 这里直接使用，不需要再截取
            
            # 获取这些聚类节点的完整信息
            top_k_clusters = []
            for cluster_node in graph.get('cluster_nodes', []):
                if cluster_node['id'] in cluster_ids_ordered:
                    top_k_clusters.append(cluster_node)
            
            # 按照cluster_ids_ordered的顺序排序top_k_clusters
            cluster_id_to_node = {c['id']: c for c in top_k_clusters}
            top_k_clusters = [cluster_id_to_node[cid] for cid in cluster_ids_ordered if cid in cluster_id_to_node]
            
            logger.info(f"Select members from {len(top_k_clusters)} clusters")
            for i, cluster in enumerate(top_k_clusters):
                logger.info(f"  Cluster {i+1}: {cluster['id']} , number of members: {cluster.get('n_members', 0)}")
            
            self.stats['total_clusters_explored'] += len(top_k_clusters)
            
            # 从这些聚类中选择成员节点（使用V2的LLM选择逻辑）
            
            cluster_nodes_added = 0
            for i, cluster in enumerate(top_k_clusters):
                # 使用V2的逻辑：从聚类中选择成员节点（可能多个）
                selected_members = self._select_members_from_cluster(
                    question,
                    cluster,
                    graph,
                    item_id
                )
                
                if selected_members:
                    # 添加到起始节点列表（避免重复）
                    existing_node_ids = {node['node_id'] for node in start_nodes}
                    for member in selected_members:
                        if member['node_id'] not in existing_node_ids:
                            member['source'] = 'cluster_retrieval'  # 标记来源
                            start_nodes.append(member)
                            cluster_nodes_added += 1
                            existing_node_ids.add(member['node_id'])
                    
                    node_ids = [n['node_id'] for n in selected_members if n['node_id'] not in existing_node_ids or n in selected_members]
                    logger.info(f"  Cluster {i+1} ({cluster['id']}): selected {len(selected_members)} nodes")
            
            self.stats['total_cluster_nodes'] += cluster_nodes_added
            logger.info(f"Stage 4 completed: selected {cluster_nodes_added} nodes from {len(top_k_clusters)} clusters")
        
        # ========== 总结两阶段检索结果 ==========
        direct_count = sum(1 for n in start_nodes if n.get('source', '').startswith('direct_retrieval'))
        cluster_count = sum(1 for n in start_nodes if n.get('source') == 'cluster_retrieval')
        
        logger.info(f"========================")
        logger.info(f"Direct retrieval: {direct_count} nodes")
        logger.info(f"Cluster retrieval: {cluster_count} nodes")
        logger.info(f"Total: {len(start_nodes)} start nodes")
        logger.info(f"========================")
        
        if not start_nodes:
            return {
                'question': question,
                'item_id': item_id,
                'answer': "",
                'exploration_type': 'hierarchical_v3_two_stage',
                'error': 'No start nodes after retrieval'
            }
        
        # ========== 阶段5：多路径探索（全局优先级队列 + subgoal跟踪） ==========
        logger.info(f"=== Stage 5: {self.n_paths} paths with global priority queue (subgoal tracking) ===")
        if subgoals:
            logger.info(f"Subgoals to satisfy: {len(subgoals)}")
            for i, sg in enumerate(subgoals):
                logger.info(f"  {i+1}. {sg}")
        
        # V3.5重构：创建全局优先级队列（所有paths共享）
        from multi_path_explorer_v2_queue import GlobalNodeQueue
        global_node_queue = GlobalNodeQueue(embedding_manager=self.embedding_manager)
        
        # 将所有start_nodes加入全局队列
        unsatisfied_subgoals = [subgoals[i] for i in range(len(subgoals))] if subgoals else []
        global_node_queue.enqueue_nodes(
            [node['node_id'] for node in start_nodes],
            self.graph_loader,
            item_id,
            set(),  # 初始时没有visited节点
            unsatisfied_subgoals=unsatisfied_subgoals
        )
        
        logger.info(f"全局队列初始化完成，共 {global_node_queue.size()} 个节点")
        
        # 创建固定数量的路径，每条路径从全局队列中取一个节点初始化
        paths = []
        for i in range(min(self.n_paths, len(start_nodes))):
            start_node = global_node_queue.get_next_node()
            if start_node:
                # V3.5：传入subgoals
                path = PathExplorerV2Queue(i, start_node, item_id, subgoals=subgoals)
                paths.append(path)
                logger.info(f"Path {i}: initialize node {start_node['node_id']}")
        
        if not paths:
            logger.warning("无法创建探索路径")
            return {
                'question': question,
                'item_id': item_id,
                'answer': "",
                'exploration_type': 'hierarchical_v3.5_global_queue',
                'error': 'No paths created'
            }
        
        # V3.6：选择并发或顺序探索
        if self.enable_concurrent:
            # ========== 并发探索模式 ==========
            logger.info(f"=== 使用并发探索（{self.n_paths}个线程） ===")
            
            # 创建线程安全的共享状态
            shared_state = SharedExplorationState(subgoals=subgoals)
            
            # 使用ThreadPoolExecutor进行并发探索
            with ThreadPoolExecutor(max_workers=self.n_paths, thread_name_prefix="PathExplorer") as executor:
                # 提交所有path的探索任务
                future_to_path = {}
                for path in paths:
                    future = executor.submit(
                        self.multi_path_explorer.explore_path_concurrent,
                        path=path,
                        question=question,
                        max_rounds=self.max_rounds,
                        shared_state=shared_state,
                        global_node_queue=global_node_queue,
                        enable_early_stopping=self.enable_early_stopping
                    )
                    future_to_path[future] = path
                
                # 等待所有任务完成（支持早停）
                completed_count = 0
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result_path = future.result()
                        completed_count += 1
                        logger.info(f"Path {result_path.path_id} 完成 ({completed_count}/{len(paths)})")
                        
                        # 检查是否应该早停
                        if self.enable_early_stopping and shared_state.should_early_stop():
                            logger.info("⚡ 早停标志已设置，取消剩余任务")
                            # 取消其他未完成的任务
                            for f in future_to_path:
                                if not f.done():
                                    f.cancel()
                            break
                    except Exception as e:
                        logger.error(f"Path {path.path_id} 执行失败: {e}", exc_info=True)
            
            # 从共享状态获取结果
            global_visited = shared_state.visited
            global_kept_nodes = shared_state.get_kept_nodes()
            if subgoals:
                global_subgoal_status = shared_state.get_subgoal_status()
            
            logger.info(f"并发探索完成: {len(global_visited)} 个节点已访问, "
                       f"{len(global_kept_nodes)} 个节点已保留")
            if subgoals and global_subgoal_status:
                satisfied_count = sum(1 for v in global_subgoal_status.values() if v)
                logger.info(f"Subgoal完成情况: {satisfied_count}/{len(global_subgoal_status)} satisfied")
        else:
            # ========== 顺序探索模式（原有逻辑） ==========
            global_visited = set()
            global_kept_nodes = []
            
            # 探索每条路径（使用全局队列 + 早停机制 + subgoal跟踪）
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"=== Round {round_num}/{self.max_rounds} ===")
                logger.info(f"全局优先级队列剩余: {global_node_queue.size()} 个节点")
                
                # 显示当前subgoal完成情况
                if subgoals and global_subgoal_status:
                    satisfied_count = sum(1 for v in global_subgoal_status.values() if v)
                    logger.info(f"Subgoal progress: {satisfied_count}/{len(global_subgoal_status)} satisfied")
                
                # V3.5改进：早停检查（如果启用且使用subgoal）
                # 早停条件：所有subgoal都已满足
                if self.enable_early_stopping and subgoals and global_subgoal_status:
                    if all(global_subgoal_status.values()):
                        satisfied_count = sum(1 for v in global_subgoal_status.values() if v)
                        logger.info(f"⚡ 早停开启：所有 {satisfied_count}/{len(global_subgoal_status)} 个subgoal已满足，停止探索")
                        break
                
                # V3.5：检查是否所有subgoal都已满足（如果启用）
                if subgoals and global_subgoal_status and all(global_subgoal_status.values()):
                    logger.info(f"⚡ 所有subgoal已满足，可以尝试生成答案")
                    # 注意：不立即停止，让路径自然结束或通过answer action停止
                
                for path in paths:
                    if path.has_answer:
                        continue
                    
                    # 检查current_node是否为None（路径已停止探索）
                    if path.current_node is None:
                        # 从全局队列取新节点
                        if not global_node_queue.is_empty():
                            path.current_node = global_node_queue.get_next_node()
                            logger.info(f"Path {path.path_id}: 取新节点 {path.current_node['node_id']} 继续")
                        else:
                            continue
                    
                    # V3.5：传入全局队列
                    self.multi_path_explorer._explore_path_step(
                        path=path,
                        question=question,
                        round_num=round_num,
                        global_visited=global_visited,
                        global_kept_nodes=global_kept_nodes,
                        global_subgoal_status=global_subgoal_status if subgoals else None,
                        global_node_queue=global_node_queue
                    )
        
        # V3.5全局队列：处理队列中剩余节点（简化逻辑）
        logger.info("=== 阶段5.1: 处理全局队列中剩余节点 ===")
        
        answer_paths_count = sum(1 for p in paths if p.has_answer)
        max_extra_rounds = self.max_rounds * 2 if answer_paths_count == 0 else self.max_rounds
        
        extra_round = 0
        while not global_node_queue.is_empty() and extra_round < max_extra_rounds:
            # V3.5：早停检查（基于subgoal完成情况）
            if self.enable_early_stopping and subgoals and global_subgoal_status:
                if all(global_subgoal_status.values()):
                    logger.info(f"⚡ 早停开启：所有subgoal已满足，停止队列处理")
                    break
            
            logger.info(f"全局队列还有 {global_node_queue.size()} 个节点待探索")
            
            # 找一个没有答案的path来探索队列中的节点
            active_path = None
            for path in paths:
                if not path.has_answer:
                    active_path = path
                    break
            
            if not active_path:
                logger.info("所有paths都已找到答案，停止队列处理")
                break
            
            # 从全局队列取节点
            next_node = global_node_queue.get_next_node()
            if next_node:
                active_path.current_node = next_node
                logger.info(f"Path {active_path.path_id}: 从全局队列取节点 {next_node['node_id']}")
                
                # 根据是否并发模式选择不同的探索方式
                if self.enable_concurrent:
                    # 并发模式：使用shared_state（注意这里是单线程，但要保持状态一致）
                    self.multi_path_explorer._explore_path_step(
                        path=active_path,
                        question=question,
                        round_num=self.max_rounds + 1 + extra_round,
                        shared_state=shared_state,
                        global_node_queue=global_node_queue
                    )
                else:
                    # 顺序模式：使用原有参数
                    self.multi_path_explorer._explore_path_step(
                        path=active_path,
                        question=question,
                        round_num=self.max_rounds + 1 + extra_round,
                        global_visited=global_visited,
                        global_kept_nodes=global_kept_nodes,
                        global_subgoal_status=global_subgoal_status if subgoals else None,
                        global_node_queue=global_node_queue
                    )
                extra_round += 1
            else:
                break
        
        logger.info(f"阶段5.1完成: 额外处理了 {extra_round} 个节点")
        
        # 更新统计信息
        total_multi_selections = sum(path.multi_node_selection_count for path in paths)
        total_nodes_selected = sum(path.total_nodes_selected for path in paths)
        self.stats['total_multi_node_selections'] += total_multi_selections
        if total_multi_selections > 0:
            avg_nodes = total_nodes_selected / total_multi_selections
            self.stats['avg_nodes_per_selection'] = avg_nodes
        
        # 分析路径情况
        answer_paths = [p for p in paths if p.has_answer]
        all_no_answer = len(answer_paths) == 0
        
        # Refinement 阶段（V3.6改进：基于subgoal完成度和队列状态触发）
        used_refinement = False
        refined_query = ""
        refinement_result = {}
        
        # 新触发逻辑：
        # 1. 全局队列已空（节点探索完）
        # 2. 启用了 refinement
        # 3. 如果使用 subgoal，则检查是否有未满足的 subgoal
        # 4. 如果不使用 subgoal，则回退到原逻辑（所有路径无答案）
        should_refine = False
        refine_reason = ""
        
        if self.enable_refinement and paths:
            if subgoals and global_subgoal_status is not None:
                # 使用 subgoal 模式：检查队列是否为空且有未满足的 subgoal
                if global_node_queue.is_empty() and not all(global_subgoal_status.values()):
                    should_refine = True
                    satisfied_count = sum(1 for v in global_subgoal_status.values() if v)
                    refine_reason = f"队列已空且仍有 {len(subgoals) - satisfied_count}/{len(subgoals)} 个subgoal未满足"
            else:
                # 不使用 subgoal 模式：回退到原逻辑（所有路径无答案）
                if all_no_answer:
                    should_refine = True
                    refine_reason = "所有路径未找到答案"
        
        if should_refine:
            logger.info(f"=== 阶段5.5: Query Refinement & Re-exploration (触发原因: {refine_reason}) ===")
            
            # V3.6：使用subgoal信息生成refinement query
            refinement_result = self._refinement_and_reexplore_v3(
                question, graph, item_id, paths, global_kept_nodes, global_visited,
                subgoals=subgoals, global_subgoal_status=global_subgoal_status
            )
            
            if refinement_result.get('used_refinement'):
                used_refinement = True
                refined_query = refinement_result.get('refined_query', '')
                paths = refinement_result.get('paths', paths)
                answer_paths = [p for p in paths if p.has_answer]
                
                # 合并refinement探索的kept nodes到主流程的global_kept_nodes
                refinement_kept_nodes = refinement_result.get('refinement_kept_nodes', [])
                for node_id in refinement_kept_nodes:
                    if node_id not in global_kept_nodes:
                        global_kept_nodes.append(node_id)
                logger.info(f"Refinement探索保留了 {len(refinement_kept_nodes)} 个节点，已合并到主流程")
                
                # 更新全局subgoal状态（从refinement paths中获取）
                if subgoals and refinement_result.get('updated_subgoal_status'):
                    global_subgoal_status.update(refinement_result.get('updated_subgoal_status'))
        
        # 生成最终答案（继承V2逻辑，传入start_nodes作为fallback）
        final_answer = self._generate_final_answer_v2(
            question, answer_paths, paths, global_kept_nodes, item_id, category, start_nodes
        )
        
        # 统计所有path情况（注意：属性名是all_skipped，不是is_all_skip）
        all_skip_paths = [p for p in paths if p.all_skipped]
        self.stats['all_skip_paths'] += len(all_skip_paths)
        self.stats['total_paths'] += len(paths)
        
        return {
            'question': question,
            'item_id': item_id,
            'answer': final_answer,
            'exploration_type': 'hierarchical_v3.5_two_stage_subgoal',
            'n_paths': len(paths),
            'n_answer_paths': len(answer_paths),
            'n_all_skip_paths': len(all_skip_paths),
            'used_refinement': used_refinement,
            'refined_query': refined_query if used_refinement else "",
            'retrieval_stats': {
                'direct_nodes': direct_count,
                'cluster_nodes': cluster_count,
                'total_start_nodes': len(start_nodes),
                'avg_direct_similarity': avg_direct_similarity
            },
            'subgoal_stats': {
                'total_subgoals': len(subgoals) if subgoals else 0,
                'satisfied_subgoals': sum(1 for v in global_subgoal_status.values() if v) if global_subgoal_status else 0
            } if subgoals else {}
        }
    
    def _refinement_and_reexplore_v3(self,
                                    question: str,
                                    graph: Dict[str, Any],
                                    item_id: str,
                                    initial_paths: List[PathExplorerV2Queue],
                                    global_kept_nodes: List[str],
                                    global_visited: Set[str],
                                    subgoals: List[str] = None,
                                    global_subgoal_status: dict = None) -> Dict[str, Any]:
        """V3版本的Refinement：使用两阶段检索策略重新探索（V3.5增强支持subgoals）"""
        
        # 收集初始探索的上下文（使用global_kept_nodes，而不是path.kept_nodes）
        if not global_kept_nodes:
            logger.warning("无法收集初始上下文（无kept nodes），跳过Refinement")
            return {'used_refinement': False}
        
        # 使用context_formatter获取完整的kept nodes信息（包含时间戳等）
        initial_context_str = self.context_formatter.format_kept_nodes_info(
            global_kept_nodes, item_id, include_metadata=False  # refinement只需要文本，不需要元数据
        )
        
        if not initial_context_str:
            logger.warning("无法收集初始上下文，跳过Refinement")
            return {'used_refinement': False}
        
        # V3.5：使用LLM生成refined query（支持subgoal）
        if subgoals and global_subgoal_status is not None:
            refined_query, _, _ = self.llm_handler.generate_refinement_query(
                question, initial_context_str,
                subgoals=subgoals, subgoal_status=global_subgoal_status
            )
        else:
            # 向后兼容：不使用subgoal
            refined_query, _, _ = self.llm_handler.generate_refinement_query(
                question, initial_context_str
            )
        
        if not refined_query or refined_query == question:
            logger.warning("Refinement未生成新查询，保持原查询")
            return {'used_refinement': False}
        
        logger.info(f"Refined query: {refined_query}")
        
        # 使用refined query重新执行两阶段检索
        logger.info("=== 使用refined query重新执行两阶段检索 ===")
        
        # 阶段1：直接检索
        direct_top_nodes = self.embedding_manager.find_top_k_nodes_by_summary(
            refined_query,
            graph,
            k=self.top_k_nodes
        )
        
        if not direct_top_nodes:
            logger.warning("Refinement阶段1：未找到相关节点")
            return {'used_refinement': False}
        
        # 阶段2：使用 LLM 从候选节点中筛选（和主流程保持一致）
        logger.info(f"=== Refinement Stage 2: LLM selects from top {len(direct_top_nodes)} candidate nodes ===")
        
        # 构建候选节点信息（给LLM看）
        candidate_nodes_for_llm = []
        for node in direct_top_nodes:
            candidate_nodes_for_llm.append({
                'node_id': node['node_id'],
                'summary': node.get('summaries', [''])[0] if node.get('summaries') else '',
                'similarity': node.get('similarity', 0.0)
            })
        
        # 使用 LLM 筛选节点
        start_nodes = []
        if subgoals:
            # 如果有 subgoals，使用 subgoal 版本的筛选
            selected_node_ids, llm_selection_response, _ = self.llm_handler.select_top_k_nodes(
                refined_query, subgoals, candidate_nodes_for_llm
            )
        else:
            # 向后兼容：如果没有 subgoals，选择所有候选节点
            selected_node_ids = [n['node_id'] for n in direct_top_nodes]
            llm_selection_response = "No subgoal planning, using all candidates"
        
        # 根据 LLM 的选择构建起始节点列表
        for node in direct_top_nodes:
            if node['node_id'] in selected_node_ids:
                start_nodes.append({
                    'node_id': node['node_id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None),
                    'source': 'direct_retrieval_refined_llm_selected'
                })
        
        if start_nodes:
            logger.info(f"Refinement Stage 2: LLM selected {len(start_nodes)}/{len(direct_top_nodes)} nodes")
            logger.info(f"Selected nodes: {[n['node_id'] for n in start_nodes]}")
        else:
            # Fallback：如果 LLM 一个都没选，至少保留前 2 个
            logger.warning("Refinement: LLM did not select any nodes, using fallback (top 2 candidates)")
            for node in direct_top_nodes[:2]:
                start_nodes.append({
                    'node_id': node['node_id'],
                    'texts': node.get('texts', []),
                    'summaries': node.get('summaries', []),
                    'people': node.get('people', []),
                    'time_explicit': node.get('time_explicit', []),
                    'utterance_refs': node.get('utterance_refs', []),
                    'embedding': node.get('embedding', None),
                    'source': 'direct_retrieval_refined_fallback'
                })
        
        # 收集聚类ID，受 top_k_per_cluster 参数控制
        cluster_ids_ordered = []
        seen_clusters = set()
        
        if self.top_k_per_cluster > 0:
            for node in direct_top_nodes:
                cluster_id = node.get('cluster_id')
                if cluster_id and cluster_id not in seen_clusters:
                    cluster_ids_ordered.append(cluster_id)
                    seen_clusters.add(cluster_id)
                    
                    # 如果已经收集到足够的聚类，停止
                    if len(cluster_ids_ordered) >= self.top_k_per_cluster:
                        break
            
            logger.info(f"Refinement: 从 {len(cluster_ids_ordered)} 个聚类中使用LLM选择成员节点")
        else:
            logger.info(f"Refinement: top_k_per_cluster=0, skip cluster collection")
        
        if cluster_ids_ordered:
            relevant_clusters = []
            for cluster_node in graph.get('cluster_nodes', []):
                if cluster_node['id'] in cluster_ids_ordered:
                    relevant_clusters.append(cluster_node)
            
            # 保持顺序
            cluster_id_to_node = {c['id']: c for c in relevant_clusters}
            relevant_clusters = [cluster_id_to_node[cid] for cid in cluster_ids_ordered if cid in cluster_id_to_node]
            
            for i, cluster in enumerate(relevant_clusters):
                # 使用 LLM 智能选择，和主流程保持一致
                selected_members = self._select_members_from_cluster(
                    refined_query,
                    cluster,
                    graph,
                    item_id
                )
                
                if selected_members:
                    existing_node_ids = {node['node_id'] for node in start_nodes}
                    for member in selected_members:
                        if member['node_id'] not in existing_node_ids:
                            member['source'] = 'cluster_retrieval_refined'
                            start_nodes.append(member)
                            existing_node_ids.add(member['node_id'])
                    
                    logger.info(f"  Refinement Cluster {i+1} ({cluster['id']}): LLM selected {len(selected_members)} nodes")
        
        logger.info(f"Refinement后找到 {len(start_nodes)} 个起始节点")
        
        if not start_nodes:
            return {'used_refinement': False}
        
        # V3.5：重新探索（使用全局优先级队列 + 固定路径数量 + subgoal跟踪）
        # 创建refinement专用的全局节点队列
        refinement_global_queue = GlobalNodeQueue(self.embedding_manager)
        
        # V3.5：初始化refinement的subgoal状态（继承全局状态）
        refinement_subgoal_status = {}
        if subgoals and global_subgoal_status is not None:
            refinement_subgoal_status = global_subgoal_status.copy()
        
        # 将所有start_nodes添加到全局队列（自动按优先级排序）
        # 创建空的refinement_visited用于enqueue_nodes
        refinement_visited = set()
        unsatisfied_subgoals = []
        if subgoals and refinement_subgoal_status:
            unsatisfied_subgoals = [subgoals[i] for i, satisfied in refinement_subgoal_status.items() if not satisfied]
        
        refinement_global_queue.enqueue_nodes(
            [node['node_id'] for node in start_nodes],
            self.graph_loader,
            item_id,
            refinement_visited,
            unsatisfied_subgoals=unsatisfied_subgoals
        )
        
        # 创建固定数量的路径，每个路径从全局队列取一个节点作为起点
        new_paths = []
        for i in range(min(self.n_paths, len(start_nodes))):
            start_node = refinement_global_queue.get_next_node()
            if start_node:
                path = PathExplorerV2Queue(i, start_node, item_id, subgoals=subgoals)
                new_paths.append(path)
                logger.info(f"Refinement Path {i}: 初始化节点 {start_node['node_id']}")
            else:
                break
        
        # Refinement探索使用新的kept_nodes（visited已在上面初始化）
        refinement_kept_nodes = []
        
        # 主探索循环：固定max_rounds轮
        for round_num in range(1, self.max_rounds + 1):
            # V3.5改进：早停检查（基于subgoal完成情况）
            if self.enable_early_stopping and subgoals and refinement_subgoal_status:
                if all(refinement_subgoal_status.values()):
                    logger.info("⚡ Refinement早停：所有subgoal已满足")
                    break
            
            for path in new_paths:
                if path.has_answer:
                    continue
                
                # 如果当前路径没有节点，从全局队列取一个
                if path.current_node is None:
                    next_node = refinement_global_queue.get_next_node()
                    if next_node:
                        path.current_node = next_node
                        logger.info(f"Refinement Path {path.path_id}: 从全局队列取节点 {next_node['node_id']}")
                    else:
                        continue  # 队列已空，跳过该路径
                
                self.multi_path_explorer._explore_path_step(
                    path=path,
                    question=refined_query,
                    round_num=round_num,
                    global_visited=refinement_visited,
                    global_kept_nodes=refinement_kept_nodes,
                    global_node_queue=refinement_global_queue,  # 传入全局队列
                    global_subgoal_status=refinement_subgoal_status if subgoals else None
                )
        
        # 处理refinement全局队列中剩余节点
        logger.info(f"=== Refinement阶段：处理全局队列剩余节点 ===")
        # V3.5改进：早停检查（基于subgoal完成情况）
        if self.enable_early_stopping and subgoals and refinement_subgoal_status:
            if all(refinement_subgoal_status.values()):
                logger.info("⚡ Refinement早停开启：所有subgoal已满足，跳过队列处理")
            else:
                # 继续处理队列
                while not refinement_global_queue.is_empty():
                    # 每次循环都检查subgoal是否全部满足
                    if self.enable_early_stopping and all(refinement_subgoal_status.values()):
                        logger.info("⚡ Refinement：所有subgoal已满足，停止队列处理")
                        break
                    
                    next_node = refinement_global_queue.get_next_node()
                    if not next_node:
                        break
                    
                    # 找一个空闲的路径
                    path_to_use = None
                    for p in new_paths:
                        if p.current_node is None and not p.has_answer:
                            path_to_use = p
                            break
                    
                    if path_to_use:
                        path_to_use.current_node = next_node
                        logger.info(f"Refinement Path {path_to_use.path_id}: 从队列取节点 {next_node['node_id']}")
                        
                        for new_round in range(1, self.max_rounds + 1):
                            if path_to_use.has_answer:
                                break
                            
                            self.multi_path_explorer._explore_path_step(
                                path=path_to_use,
                                question=refined_query,
                                round_num=new_round,
                                global_visited=refinement_visited,
                                global_kept_nodes=refinement_kept_nodes,
                                global_node_queue=refinement_global_queue,
                                global_subgoal_status=refinement_subgoal_status if subgoals else None
                            )
                            
                            # 每轮探索后检查subgoal完成情况
                            if self.enable_early_stopping and subgoals and all(refinement_subgoal_status.values()):
                                logger.info("⚡ Refinement：探索过程中所有subgoal已满足，停止")
                                break
                        
                        path_to_use.current_node = None  # 标记为空闲
                    else:
                        logger.warning("Refinement: 没有空闲路径处理队列节点")
                        break
        else:
            # 早停关闭或没有subgoal，按原逻辑处理队列
            while not refinement_global_queue.is_empty():
                next_node = refinement_global_queue.get_next_node()
                if not next_node:
                    break
                
                # 找一个空闲的路径
                path_to_use = None
                for p in new_paths:
                    if p.current_node is None and not p.has_answer:
                        path_to_use = p
                        break
                
                if path_to_use:
                    path_to_use.current_node = next_node
                    logger.info(f"Refinement Path {path_to_use.path_id}: 从队列取节点 {next_node['node_id']}")
                    
                    for new_round in range(1, self.max_rounds + 1):
                        if path_to_use.has_answer:
                            break
                        
                        self.multi_path_explorer._explore_path_step(
                            path=path_to_use,
                            question=refined_query,
                            round_num=new_round,
                            global_visited=refinement_visited,
                            global_kept_nodes=refinement_kept_nodes,
                            global_node_queue=refinement_global_queue,
                            global_subgoal_status=refinement_subgoal_status if subgoals else None
                        )
                    
                    path_to_use.current_node = None  # 标记为空闲
                else:
                    logger.warning("Refinement: 没有空闲路径处理队列节点")
                    break
        
        
        answer_paths = [p for p in new_paths if p.has_answer]
        logger.info(f"Refinement complete: {len(answer_paths)}/{len(new_paths)} paths with answer")
        if subgoals and refinement_subgoal_status:
            satisfied_count = sum(1 for v in refinement_subgoal_status.values() if v)
            logger.info(f"Refinement subgoal progress: {satisfied_count}/{len(refinement_subgoal_status)} satisfied")
        
        return {
            'used_refinement': True,
            'refined_query': refined_query,
            'paths': new_paths,
            'answer_paths': answer_paths,
            'refinement_kept_nodes': refinement_kept_nodes,
            'updated_subgoal_status': refinement_subgoal_status if subgoals else {}
        }
    
    def _generate_final_answer_v2(self,
                                 question: str,
                                 answer_paths: List[PathExplorerV2Queue],
                                 all_paths: List[PathExplorerV2Queue],
                                 global_kept_nodes: List[str],  # 实际是node_id字符串列表
                                 item_id: str,
                                 category: int = None,
                                 start_nodes: List[Dict[str, Any]] = None) -> str:
        """生成最终答案（继承V2逻辑）
        
        Args:
            question: 问题
            answer_paths: 返回ANSWER的路径列表
            all_paths: 所有路径列表
            global_kept_nodes: 全局保留节点ID列表
            item_id: 项目ID
            category: 问题类别
            start_nodes: 最开始直接检索的top-k节点（作为fallback）
        """
        
        if not answer_paths:
            if not all_paths:
                return "(No information found)"
            
            # 没有answer路径，使用所有kept节点
            logger.info(f"无answer路径，使用 {len(global_kept_nodes)} 个全局kept节点生成答案")
            
            if not global_kept_nodes:
                # 如果没有kept节点，使用最开始直接检索的start_nodes作为fallback
                if start_nodes:
                    logger.warning(f"没有保留节点，使用最开始直接检索的 {len(start_nodes)} 个top-k节点作为fallback")
                    fallback_node_ids = [node['node_id'] for node in start_nodes]
                    
                    # 使用context_formatter构建完整的上下文
                    final_context = self.context_formatter.build_final_context_from_kept_nodes(
                        fallback_node_ids, item_id
                    )
                    
                    if not final_context:
                        return "(No sufficient information found)"
                    
                    # 生成答案
                    final_answer, _, _ = self.llm_handler.generate_answer(question, final_context, category)
                    return final_answer
                else:
                    return "(No sufficient information found)"
            
            # 使用context_formatter构建完整的上下文（包含时间戳、People、Time等完整信息）
            final_context = self.context_formatter.build_final_context_from_kept_nodes(
                global_kept_nodes, item_id
            )
            
            if not final_context:
                return "(Unable to build context from kept nodes)"
            
            # 生成答案（只取返回元组的第一个元素：干净的答案）
            final_answer, _, _ = self.llm_handler.generate_answer(question, final_context, category)
            return final_answer
        
        # 有answer路径，使用所有kept nodes生成答案（不只是answer node）
        logger.info(f"使用 {len(answer_paths)} 条answer路径，共 {len(global_kept_nodes)} 个保留节点生成答案")
        
        # 使用所有global_kept_nodes，而不是只用answer_nodes
        # 因为之前EXPAND的节点也包含重要信息
        if not global_kept_nodes:
            # 如果有answer路径但没有kept节点，使用start_nodes作为fallback
            if start_nodes:
                logger.warning(f"有answer路径但没有保留节点，使用最开始直接检索的 {len(start_nodes)} 个top-k节点作为fallback")
                fallback_node_ids = [node['node_id'] for node in start_nodes]
                
                # 使用context_formatter构建完整的上下文
                final_context = self.context_formatter.build_final_context_from_kept_nodes(
                    fallback_node_ids, item_id
                )
                
                if not final_context:
                    return "(Answer paths found but no kept nodes available)"
                
                # 生成答案
                final_answer, _, _ = self.llm_handler.generate_answer(question, final_context, category)
                return final_answer
            else:
                return "(Answer paths found but no kept nodes available)"
        
        # 使用context_formatter构建完整的上下文（包含时间戳、People、Time等完整信息）
        final_context = self.context_formatter.build_final_context_from_kept_nodes(
            global_kept_nodes, item_id
        )
        
        if not final_context:
            return "(Unable to build context from kept nodes)"
        
        # 生成答案（只取返回元组的第一个元素：干净的答案）
        final_answer, _, _ = self.llm_handler.generate_answer(question, final_context, category)
        return final_answer
    
    def run(self, output_dir: str = None):
        """运行QA系统"""
        total_start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("层次化图问答系统 V3 - 两阶段检索策略")
        logger.info("=" * 80)
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = "_debug" if self.debug_mode else ""
            output_dir = f"/share/project/zyt/hyy/Memory/QA_system/output/qa_results_hierarchical_v3_two_stage{mode_suffix}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        all_results = []
        total_questions = 0
        total_qa_time = 0.0
        
        # 调试模式：只处理指定数量的items
        items_to_process = self.qa_data
        if self.debug_mode:
            items_to_process = self.qa_data[:self.debug_items]
            logger.info(f"🔧 调试模式：只处理前 {self.debug_items} 个items")
        
        for idx, qa_item in enumerate(items_to_process):
            item_start_time = time.time()
            
            # 兼容多种数据格式：
            # 1. item_id 字段（如果有）
            # 2. document_id 字段（narrativeQA）
            # 3. 根据索引生成 locomo_itemN（locomo）
            item_id = qa_item.get('item_id')
            if not item_id:
                item_id = qa_item.get('document_id')  # narrativeQA format
            if not item_id:
                item_id = f"locomo_item{idx + 1}"  # fallback for locomo
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Item: {item_id}")
            logger.info(f"{'='*80}\n")
            
            item_results = {
                'item_id': item_id,
                'qa_results': [],
                'session_summary': qa_item.get('session_summary', {})
            }
            
            # 兼容两种数据格式：questions 或 qa 字段
            questions = qa_item.get('questions', []) or qa_item.get('qa', [])
            
            # Debug模式限制
            if self.debug_mode:
                questions = questions[:self.debug_qa_per_item]
                logger.info(f"Debug Mode: item {item_id} 只处理前 {self.debug_qa_per_item} 个QA")
            
            for i, qa in enumerate(questions):
                question = qa.get('question', '')
                category = qa.get('category', -1)
                
                if not question:
                    continue
                
                # 跳过category=5的问题
                if category == 5:
                    logger.info(f"Skip QA {i+1}/{len(questions)} (category=5): {question[:50]}...")
                    self.stats['skipped_questions'] += 1
                    continue
                
                self.stats['total_questions'] += 1
                
                logger.info(f"\nDeal with QA {i+1}/{len(questions)} (category={category}): {question[:50]}...")
                
                try:
                    # 回答问题
                    result = self.answer_question(question, item_id, category)
                    
                    # 获取并更新token统计
                    token_stats = self.llm_handler.get_token_stats()
                    result['prompt_tokens'] = token_stats['total_prompt_tokens'] - self.stats['total_prompt_tokens']
                    result['completion_tokens'] = token_stats['total_completion_tokens'] - self.stats['total_completion_tokens']
                    result['total_tokens'] = result['prompt_tokens'] + result['completion_tokens']
                    
                    # 更新全局统计
                    self.stats['total_prompt_tokens'] = token_stats['total_prompt_tokens']
                    self.stats['total_completion_tokens'] = token_stats['total_completion_tokens']
                    self.stats['total_tokens'] = token_stats['total_tokens']
                    self.stats['llm_call_count'] = token_stats['call_count']
                    
                    # 添加ground truth和category
                    result['ground_truth'] = qa.get('answer', '')
                    result['evidence'] = qa.get('evidence', [])
                    result['category'] = category
                    
                    logger.info(f"A: {str(result['answer'])[:200]}...")
                    logger.info(f"Ground Truth: {str(result['ground_truth'])[:200]}...")
                    logger.info(f"Tokens: {result['total_tokens']} (Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']})")
                    
                    item_results['qa_results'].append(result)
                    total_questions += 1
                    total_qa_time += result.get('elapsed_time_seconds', 0)
                    
                except Exception as e:
                    logger.error(f"Failed to process question: {question}, error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    item_results['qa_results'].append({
                        'question': question,
                        'item_id': item_id,
                        'answer': '',
                        'error': str(e),
                        'ground_truth': qa.get('answer', ''),
                        'evidence': qa.get('evidence', []),
                        'category': category
                    })
            
            # 记录item处理时间
            item_elapsed_time = time.time() - item_start_time
            item_results['item_elapsed_time_seconds'] = round(item_elapsed_time, 2)
            
            # 保存单个item的结果
            item_output_path = output_path / f"{item_id}_qa_results.json"
            with open(item_output_path, 'w', encoding='utf-8') as f:
                json.dump(item_results, f, ensure_ascii=False, indent=2)
            
            all_results.append(item_results)
            logger.info(f"✅ 完成 {item_id}: {len(item_results['qa_results'])} 个QA (用时: {item_elapsed_time:.2f}秒)")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算总用时
        total_elapsed_time = time.time() - total_start_time
        
        # 保存所有结果和统计信息
        self._save_all_results(output_path, all_results, total_questions, total_qa_time, total_elapsed_time)
        
        # 打印统计
        self._print_stats()
        
        logger.info(f"🎉 所有QA处理完成，结果保存在: {output_path}")
        
        return all_results
    
    def _save_all_results(self, output_path: Path, all_results: List[Dict[str, Any]], 
                          total_questions: int, total_qa_time: float, total_elapsed_time: float):
        """保存所有结果和统计信息"""
        # 生成统计摘要
        summary = {
            'total_items': len(all_results),
            'total_questions': total_questions,
            'skipped_questions': self.stats['skipped_questions'],
            'processed_questions': total_questions,
            'total_elapsed_time_seconds': round(total_elapsed_time, 2),
            'total_qa_time_seconds': round(total_qa_time, 2),
            'average_time_per_question': round(total_qa_time / total_questions, 2) if total_questions > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'v3_two_stage_stats': {
                'total_paths': self.stats['total_paths'],
                'all_skip_paths': self.stats['all_skip_paths'],
                'all_skip_path_ratio': round(self.stats['all_skip_paths'] / self.stats['total_paths'], 3) if self.stats['total_paths'] > 0 else 0,
                'multi_node_selections': self.stats['total_multi_node_selections'],
                'avg_nodes_per_selection': round(self.stats['avg_nodes_per_selection'], 2),
                'retrieval_stats': {
                    'total_direct_nodes': self.stats['total_direct_nodes'],
                    'total_clusters_explored': self.stats['total_clusters_explored'],
                    'total_cluster_nodes': self.stats['total_cluster_nodes'],
                    'avg_direct_node_similarity': round(self.stats['avg_direct_node_similarity'], 3)
                },
                'token_stats': {
                    'total_prompt_tokens': self.stats['total_prompt_tokens'],
                    'total_completion_tokens': self.stats['total_completion_tokens'],
                    'total_tokens': self.stats['total_tokens'],
                    'llm_call_count': self.stats['llm_call_count'],
                    'avg_tokens_per_question': round(self.stats['total_tokens'] / total_questions, 1) if total_questions > 0 else 0
                }
            }
        }
        
        # 保存所有结果
        all_output_path = output_path / "all_qa_results.json"
        with open(all_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 所有结果已保存到: {all_output_path}")
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("\n" + "=" * 80)
        logger.info("V3系统统计信息")
        logger.info("=" * 80)
        logger.info(f"总问题数: {self.stats['total_questions']}")
        logger.info(f"跳过的问题数 (category=5): {self.stats['skipped_questions']}")
        logger.info(f"总路径数: {self.stats['total_paths']}")
        logger.info(f"全部skip的路径数: {self.stats['all_skip_paths']}")
        logger.info(f"多节点选择次数: {self.stats['total_multi_node_selections']}")
        logger.info(f"平均每次选择节点数: {self.stats['avg_nodes_per_selection']:.2f}")
        logger.info("\n=== V3两阶段检索统计 ===")
        logger.info(f"直接检索节点总数: {self.stats['total_direct_nodes']}")
        logger.info(f"探索的聚类总数: {self.stats['total_clusters_explored']}")
        logger.info(f"从聚类中选择的节点总数: {self.stats['total_cluster_nodes']}")
        logger.info(f"平均直接检索相似度: {self.stats['avg_direct_node_similarity']:.3f}")
        
        logger.info("\n=== Token使用统计 ===")
        logger.info(f"LLM总调用次数: {self.stats['llm_call_count']}")
        logger.info(f"总Prompt Tokens: {self.stats['total_prompt_tokens']:,}")
        logger.info(f"总Completion Tokens: {self.stats['total_completion_tokens']:,}")
        logger.info(f"总Tokens: {self.stats['total_tokens']:,}")
        if self.stats['total_questions'] > 0:
            avg_tokens_per_question = self.stats['total_tokens'] / self.stats['total_questions']
            logger.info(f"平均每问题Token消耗: {avg_tokens_per_question:.1f}")
        if self.stats['llm_call_count'] > 0:
            avg_prompt = self.stats['total_prompt_tokens'] / self.stats['llm_call_count']
            avg_completion = self.stats['total_completion_tokens'] / self.stats['llm_call_count']
            logger.info(f"平均每次调用 - Prompt: {avg_prompt:.1f}, Completion: {avg_completion:.1f}")
        
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='层次化图问答系统 V3 - 两阶段检索策略')
    
    # 基础参数
    parser.add_argument('--graphs_dir',
                       default='/share/project/zyt/hyy/Memory/build_graph/graphs_llm_clustered',
                       help='图数据目录')
    parser.add_argument('--qa_data_path',
                       default='/share/project/zyt/hyy/Memory/data/locomo/locomo10.json',
                       help='QA数据文件路径')
    
    # 模型参数
    parser.add_argument('--model_name',
                       default='/share/project/zyt/hyy/Model/Qwen3-8B',
                       help='LLM模型名称或路径')
    parser.add_argument('--embedding_model',
                       default='/share/project/zyt/hyy/Model/bge-m3',
                       help='Embedding模型名称或路径')
    
    # V3新参数
    parser.add_argument('--top_k_nodes', type=int, default=5,
                       help='首次直接检索的节点数量（默认5）')
    parser.add_argument('--top_k_per_cluster', type=int, default=3,
                       help='从每个聚类中选择的节点数量（默认3）')
    parser.add_argument('--n_paths', type=int, default=3,
                       help='固定的探索路径数量（默认3）')
    
    # 探索参数
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                       help='相似度阈值')
    parser.add_argument('--max_rounds', type=int, default=2,
                       help='最大探索轮数')
    
    # GPU参数（仅Embedding）
    parser.add_argument('--embedding_gpu_id', type=int, default=7,
                       help='Embedding模型使用的GPU ID')
    
    # API参数（用于LLM）
    parser.add_argument('--api_base', default='http://localhost:8000/v1',
                       help='LLM API的基础URL')
    parser.add_argument('--api_key', default='EMPTY',
                       help='API密钥（本地部署时可以使用"EMPTY"）')
    
    # 功能开关
    parser.add_argument('--disable_refinement', action='store_true',
                       help='禁用query refinement')
    parser.add_argument('--no_relation', action='store_true',
                       help='不使用关系信息')
    parser.add_argument('--enable_early_stopping', action='store_true',
                       help='启用早停机制（默认关闭）')
    parser.add_argument('--disable_subgoal_planning', action='store_true',
                       help='禁用subgoal规划机制（默认启用）')
    parser.add_argument('--enable_concurrent', action='store_true',
                       help='启用并发探索（默认关闭，启用后使用多线程并发探索）')
    
    # Debug参数
    parser.add_argument('--debug_mode', action='store_true',
                       help='调试模式')
    parser.add_argument('--debug_items', type=int, default=1,
                       help='调试模式下处理的item数量')
    parser.add_argument('--debug_qa_per_item', type=int, default=2,
                       help='调试模式下每个item处理的QA数量')
    
    args = parser.parse_args()
    
    # Handle refinement and relation flags (和V2保持一致)
    enable_refinement = not args.disable_refinement
    use_relation = not args.no_relation
    enable_early_stopping = args.enable_early_stopping
    use_subgoal_planning = not args.disable_subgoal_planning
    enable_concurrent = args.enable_concurrent
    
    # 初始化系统
    qa_system = HierarchicalGraphQASystemV3(
        graphs_dir=args.graphs_dir,
        qa_data_path=args.qa_data_path,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        top_k_nodes=args.top_k_nodes,
        top_k_per_cluster=args.top_k_per_cluster,
        n_paths=args.n_paths,
        similarity_threshold=args.similarity_threshold,
        max_rounds=args.max_rounds,
        debug_mode=args.debug_mode,
        debug_items=args.debug_items,
        debug_qa_per_item=args.debug_qa_per_item,
        embedding_gpu_id=args.embedding_gpu_id,
        api_base=args.api_base,
        api_key=args.api_key,
        enable_refinement=enable_refinement,
        use_relation=use_relation,
        enable_early_stopping=enable_early_stopping,
        use_subgoal_planning=use_subgoal_planning,
        enable_concurrent=enable_concurrent
    )
    
    # 运行系统
    qa_system.run()


if __name__ == '__main__':
    main()

