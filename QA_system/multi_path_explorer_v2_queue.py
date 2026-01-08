#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-path exploration module - V2 Queue版本
支持三动作机制（skip/expand/answer）+ 多节点队列管理 + 并发探索
"""

import logging
from typing import List, Dict, Any, Set, Optional
from queue import PriorityQueue
import heapq
import threading

logger = logging.getLogger(__name__)


class GlobalNodeQueue:
    """全局节点优先级队列管理器（所有paths共享，线程安全）"""
    
    def __init__(self, embedding_manager=None):
        self.queue = PriorityQueue()  # PriorityQueue本身是线程安全的
        self.priority_cache = {}  # 缓存节点相似度 {node_id: (priority, node)}
        self.cache_lock = threading.Lock()  # 保护priority_cache的锁
        self.counter = 0  # 用于保证相同优先级时按插入顺序
        self.counter_lock = threading.Lock()  # 保护counter的锁
        self.embedding_manager = embedding_manager
        self.max_queue_size = 0  # 记录队列的最大长度
        self.stats_lock = threading.Lock()  # 保护统计信息的锁
        
    def enqueue_nodes(self, node_ids: List[str], graph_loader, item_id: str,
                     global_visited: Set[str], unsatisfied_subgoals: List[str] = None):
        """将多个节点加入全局优先级队列（线程安全）
        
        优化要点：
        1. 每个节点只计算一次相似度（使用缓存）
        2. 使用PriorityQueue自动维护全局排序
        3. 插入时只计算新节点，O(k log n)
        4. 线程安全：使用锁保护共享资源
        """
        # 收集未访问的节点（需要加锁检查global_visited）
        nodes_to_add = []
        for node_id in node_ids:
            if node_id not in global_visited:  # global_visited在外部加锁
                node = graph_loader.get_node_by_id(node_id, item_id)
                if node:
                    nodes_to_add.append(node)
        
        if not nodes_to_add:
            return 0
        
        added_count = 0
        
        # 只对新节点计算相似度并插入
        for node in nodes_to_add:
            node_id = node.get('node_id')
            
            # 检查缓存，避免重复计算（加锁）
            with self.cache_lock:
                if node_id in self.priority_cache:
                    priority, _ = self.priority_cache[node_id]
                    logger.debug(f"Global Queue: Using cached priority for {node_id}: {-priority:.3f}")
                else:
                    # 只为新节点计算相似度
                    if self.embedding_manager and unsatisfied_subgoals and len(unsatisfied_subgoals) > 0:
                        # 计算与未满足subgoal的最大相似度
                        similarity = self.embedding_manager.compute_node_max_similarity_to_subgoals(
                            node, unsatisfied_subgoals
                        )
                        # 使用负值作为优先级（PriorityQueue是最小堆，我们要最大相似度）
                        priority = -similarity
                    else:
                        # 没有subgoal时，使用FIFO顺序（优先级相同）
                        priority = 0.0
                    
                    # 缓存优先级
                    self.priority_cache[node_id] = (priority, node)
                    logger.debug(f"Global Queue: Computed priority for {node_id}: {-priority:.3f}")
            
            # 获取counter（加锁）
            with self.counter_lock:
                counter = self.counter
                self.counter += 1
            
            # 插入优先级队列：(priority, counter, node)
            # PriorityQueue.put()本身是线程安全的
            self.queue.put((priority, counter, node))
            added_count += 1
        
        # 更新统计信息（加锁）
        if added_count > 0:
            current_size = self.queue.qsize()
            with self.stats_lock:
                if current_size > self.max_queue_size:
                    self.max_queue_size = current_size
            
            logger.info(f"Global Queue: 添加了 {added_count}/{len(node_ids)} 个节点, "
                       f"当前队列长度: {current_size}")
        
        return added_count
    
    def get_next_node(self):
        """从队列中取出最高优先级的节点（线程安全）"""
        if self.queue.empty():
            return None
        # PriorityQueue.get()本身是线程安全的
        priority, counter, node = self.queue.get()
        logger.debug(f"Global Queue: 取出节点 {node['node_id']} (priority: {-priority:.3f})")
        return node
    
    def is_empty(self):
        """检查队列是否为空（线程安全）"""
        return self.queue.empty()
    
    def size(self):
        """获取队列大小（线程安全）"""
        return self.queue.qsize()


class SharedExplorationState:
    """管理并发探索中的共享状态（线程安全）"""
    
    def __init__(self, subgoals: List[str] = None):
        self.visited = set()  # 全局已访问节点
        self.visited_lock = threading.Lock()
        
        self.kept_nodes = []  # 全局保留节点
        self.kept_nodes_lock = threading.Lock()
        
        self.subgoal_status = {}  # subgoal完成状态
        if subgoals:
            self.subgoal_status = {i: False for i in range(len(subgoals))}
        self.subgoal_lock = threading.Lock()
        
        self.early_stop_flag = False  # 早停标志
        self.early_stop_lock = threading.Lock()
    
    def add_visited(self, node_id: str) -> bool:
        """添加已访问节点（线程安全）
        
        Returns:
            True if node was not visited before, False if already visited
        """
        with self.visited_lock:
            if node_id in self.visited:
                return False
            self.visited.add(node_id)
            return True
    
    def is_visited(self, node_id: str) -> bool:
        """检查节点是否已访问（线程安全）"""
        with self.visited_lock:
            return node_id in self.visited
    
    def add_kept_node(self, node_id: str):
        """添加保留节点（线程安全）"""
        with self.kept_nodes_lock:
            if node_id not in self.kept_nodes:
                self.kept_nodes.append(node_id)
    
    def get_kept_nodes(self) -> List[str]:
        """获取保留节点列表的副本（线程安全）"""
        with self.kept_nodes_lock:
            return self.kept_nodes.copy()
    
    def update_subgoal_status(self, satisfied_indices: List[int]):
        """更新subgoal状态（线程安全）"""
        with self.subgoal_lock:
            for idx in satisfied_indices:
                if idx in self.subgoal_status:
                    self.subgoal_status[idx] = True
    
    def get_subgoal_status(self) -> dict:
        """获取subgoal状态的副本（线程安全）"""
        with self.subgoal_lock:
            return self.subgoal_status.copy()
    
    def all_subgoals_satisfied(self) -> bool:
        """检查是否所有subgoal都已满足（线程安全）"""
        with self.subgoal_lock:
            if not self.subgoal_status:
                return False
            return all(self.subgoal_status.values())
    
    def set_early_stop(self):
        """设置早停标志（线程安全）"""
        with self.early_stop_lock:
            self.early_stop_flag = True
    
    def should_early_stop(self) -> bool:
        """检查是否应该早停（线程安全）"""
        with self.early_stop_lock:
            return self.early_stop_flag
    
    def get_visited_set(self) -> Set[str]:
        """获取visited set的副本（用于传递给GlobalNodeQueue）"""
        with self.visited_lock:
            return self.visited.copy()


class PathExplorerV2Queue:
    """Handles exploration of a single path - V3.5全局队列版本（所有paths共享一个优先级队列）"""
    
    def __init__(self, path_id: int, start_node: Dict[str, Any], item_id: str, 
                 subgoals: List[str] = None):
        self.path_id = path_id
        self.item_id = item_id
        self.start_node = start_node
        self.current_node = start_node
        self.local_visited = []  # Nodes visited in this path (ordered)
        self.round_details = []  # Details of each round in this path
        self.has_answer = False  # 是否已经决定可以回答
        self.final_decision = None
        self.all_skipped = True  # 是否所有节点都被skip了
        
        # V3.5重构：移除独立队列，使用全局队列
        # self.node_queue 已移除，改用全局队列
        self.multi_node_selection_count = 0  # 多节点选择的次数（选择>1个节点）
        self.total_nodes_selected = 0  # 总共选择的节点数
        
        # V3.5新增：subgoal跟踪
        self.subgoals = subgoals if subgoals else []
        # subgoal状态字典 {index: True/False}，初始都是False
        self.subgoal_status = {i: False for i in range(len(self.subgoals))} if subgoals else {}
        
    def add_round(self, round_info: Dict[str, Any]):
        """Add information about a round of exploration"""
        self.round_details.append(round_info)
        self.local_visited.append(round_info.get('current_node'))
        
        # 检查是否有节点被保留（expand或answer）
        if round_info.get('action') in ['expand', 'answer']:
            self.all_skipped = False
    
    def update_subgoal_status(self, satisfied_subgoal_indices: List[int]):
        """更新subgoal状态
        
        Args:
            satisfied_subgoal_indices: 满足的subgoal索引列表（0-based）
        """
        for idx in satisfied_subgoal_indices:
            if idx in self.subgoal_status:
                self.subgoal_status[idx] = True
        
        # 记录到日志
        satisfied_count = sum(1 for v in self.subgoal_status.values() if v)
        total_count = len(self.subgoal_status)
        if total_count > 0:
            logger.info(f"Path {self.path_id}: Subgoal progress: {satisfied_count}/{total_count} satisfied")
    
    def get_unsatisfied_subgoals(self) -> List[str]:
        """获取未满足的subgoal文本列表"""
        unsatisfied = []
        for i, satisfied in self.subgoal_status.items():
            if not satisfied and i < len(self.subgoals):
                unsatisfied.append(self.subgoals[i])
        return unsatisfied
    
    def all_subgoals_satisfied(self) -> bool:
        """检查是否所有subgoal都已满足"""
        if not self.subgoal_status:
            return False
        return all(self.subgoal_status.values())


class MultiPathExplorerV2Queue:
    """Multi-path exploration manager - V2 Queue版本（支持多节点队列）"""
    
    def __init__(self, 
                 graph_loader,
                 embedding_manager,
                 llm_handler,
                 context_formatter,
                 max_rounds: int = 3):
        self.graph_loader = graph_loader
        self.embedding_manager = embedding_manager
        self.llm_handler = llm_handler
        self.context_formatter = context_formatter
        self.max_rounds = max_rounds
    
    def _explore_path_step(self,
                          path: PathExplorerV2Queue,
                          question: str,
                          round_num: int,
                          shared_state: SharedExplorationState = None,
                          global_node_queue: GlobalNodeQueue = None,
                          # 向后兼容参数
                          global_visited: Set[str] = None,
                          global_kept_nodes: List[str] = None,
                          global_subgoal_status: dict = None) -> None:
        """探索路径的一步（支持全局队列 + subgoal跟踪 + 并发探索）
        
        Args:
            path: The path being explored
            question: The question being answered
            round_num: Current round number
            shared_state: 共享探索状态（线程安全，推荐使用）
            global_node_queue: 全局节点队列（必需）
            
            # 向后兼容参数（如果提供shared_state，这些参数被忽略）
            global_visited: Globally visited nodes (deprecated, use shared_state)
            global_kept_nodes: Globally kept nodes (deprecated, use shared_state)
            global_subgoal_status: 全局subgoal状态字典 (deprecated, use shared_state)
        """
        # 向后兼容：如果没有shared_state，使用旧参数
        if shared_state is None:
            # 使用旧的非线程安全接口
            _global_visited = global_visited if global_visited is not None else set()
            _global_kept_nodes = global_kept_nodes if global_kept_nodes is not None else []
            _global_subgoal_status = global_subgoal_status if global_subgoal_status is not None else {}
        else:
            # 使用新的线程安全接口
            _global_visited = None  # 通过shared_state访问
            _global_kept_nodes = None  # 通过shared_state访问
            _global_subgoal_status = None  # 通过shared_state访问
        
        # 检查current_node是否为None（路径已停止探索）
        if path.current_node is None:
            logger.info(f"Path {path.path_id}: current_node is None, path已停止探索，跳过")
            return
        
        current_node_id = path.current_node['node_id']
        
        # Skip if already visited globally（线程安全）
        if shared_state is not None:
            # 使用线程安全的add_visited，它会原子性地检查并添加
            if not shared_state.add_visited(current_node_id):
                logger.info(f"Path {path.path_id}: Node {current_node_id} already visited globally, skipping")
                path.has_answer = False
                return
        else:
            # 向后兼容：非线程安全版本
            if current_node_id in _global_visited:
                logger.info(f"Path {path.path_id}: Node {current_node_id} already visited globally, skipping")
                path.has_answer = False
                return
            _global_visited.add(current_node_id)
        
        # 获取已保留节点的信息（线程安全）
        if shared_state is not None:
            kept_nodes_list = shared_state.get_kept_nodes()
        else:
            kept_nodes_list = _global_kept_nodes
        
        kept_nodes_info = self.context_formatter.format_kept_nodes_info(kept_nodes_list, path.item_id)
        
        # Get current node information
        current_info = self.context_formatter.format_current_node_info(
            path.current_node, 
            path.item_id
        )
        
        # Get neighbor nodes (filtered by global visited)（线程安全）
        neighbors = self.graph_loader.get_neighbor_nodes(current_node_id, path.item_id)
        if shared_state is not None:
            neighbors = [n for n in neighbors if not shared_state.is_visited(n.get('node_id'))]
        else:
            neighbors = [n for n in neighbors if n.get('node_id') not in _global_visited]
        neighbor_info = self.context_formatter.format_neighbor_info(neighbors, current_node_id)
        
        logger.info(f"Path {path.path_id}, Round {round_num}: Node {current_node_id}, "
                   f"{len(neighbors)} unvisited neighbors, {len(kept_nodes_list)} kept nodes")
        
        # Ask LLM to decide action (支持subgoal版本)
        if shared_state is not None:
            current_subgoal_status = shared_state.get_subgoal_status()
        else:
            current_subgoal_status = _global_subgoal_status
        
        if path.subgoals and current_subgoal_status:
            action, next_node_ids, satisfied_subgoals, raw_decision, formatted_prompt = self.llm_handler.check_action(
                question, kept_nodes_info, current_info, neighbor_info, round_num,
                subgoals=path.subgoals, subgoal_status=current_subgoal_status
            )
            
            # 更新全局subgoal状态（线程安全）
            if shared_state is not None:
                shared_state.update_subgoal_status(satisfied_subgoals)
            else:
                for idx in satisfied_subgoals:
                    if idx in _global_subgoal_status:
                        _global_subgoal_status[idx] = True
            
            # 更新path本地的subgoal状态
            path.update_subgoal_status(satisfied_subgoals)
        else:
            # 向后兼容：不使用subgoal
            action, next_node_ids, satisfied_subgoals, raw_decision, formatted_prompt = self.llm_handler.check_action(
                question, kept_nodes_info, current_info, neighbor_info, round_num
            )
        
        # Record round information
        round_info = {
            'round': round_num,
            'path_id': path.path_id,
            'current_node': current_node_id,
            'current_info': current_info,
            'current_response': raw_decision,
            'formatted_prompt': formatted_prompt,
            'neighbors': [n['node_id'] for n in neighbors],
            'neighbor_info': neighbor_info,
            'neighbor_count': len(neighbors),
            'item_id': path.item_id,
            'action': action,
            'next_nodes': next_node_ids,
            'satisfied_subgoals': satisfied_subgoals if path.subgoals else []
        }
        path.add_round(round_info)
        
        # 根据action更新状态（线程安全）
        if action == 'answer':
            # 保留当前节点并停止探索
            if shared_state is not None:
                shared_state.add_kept_node(current_node_id)
            else:
                if current_node_id not in _global_kept_nodes:
                    _global_kept_nodes.append(current_node_id)
            path.has_answer = True
            path.final_decision = raw_decision
            logger.info(f"Path {path.path_id}: ACTION=ANSWER, kept node {current_node_id}, stopping exploration")
            return
        
        elif action == 'expand':
            # 保留当前节点并继续探索
            if shared_state is not None:
                shared_state.add_kept_node(current_node_id)
            else:
                if current_node_id not in _global_kept_nodes:
                    _global_kept_nodes.append(current_node_id)
            logger.info(f"Path {path.path_id}: ACTION=EXPAND, kept node {current_node_id}, "
                       f"continue to {len(next_node_ids)} nodes: {next_node_ids}")
        
        elif action == 'skip':
            # 丢弃当前节点，探索新节点
            logger.info(f"Path {path.path_id}: ACTION=SKIP, discarded node {current_node_id}, "
                       f"move to {len(next_node_ids)} nodes: {next_node_ids}")
        
        # 处理多个next_node_ids
        if next_node_ids and len(next_node_ids) > 0:
            # 优化：如果只有1个节点，直接设置为current_node继续探索
            if len(next_node_ids) == 1:
                next_node = self.graph_loader.get_node_by_id(next_node_ids[0], path.item_id)
                # 检查节点是否已访问（线程安全）
                is_visited = shared_state.is_visited(next_node_ids[0]) if shared_state else (next_node_ids[0] in _global_visited)
                
                if next_node and not is_visited:
                    path.current_node = next_node
                    logger.info(f"Path {path.path_id}: 只有1个下一节点 {next_node_ids[0]}，直接继续探索")
                else:
                    # 节点无效或已访问，从全局队列取
                    logger.warning(f"Path {path.path_id}: 选择的节点 {next_node_ids[0]} 无效或已访问")
                    if global_node_queue and not global_node_queue.is_empty():
                        path.current_node = global_node_queue.get_next_node()
                        logger.info(f"Path {path.path_id}: 从全局队列取节点 {path.current_node['node_id']}")
                    elif neighbors:
                        fallback_node = neighbors[0]
                        path.current_node = fallback_node
                        logger.info(f"Path {path.path_id}: Fallback to neighbor {fallback_node['node_id']}")
                    else:
                        path.current_node = None
                        logger.info(f"Path {path.path_id}: No more nodes to explore")
            
            # 如果有2个或更多节点，加入全局队列
            elif len(next_node_ids) >= 2:
                # 获取未满足的subgoal列表（用于队列排序）
                unsatisfied_subgoals = path.get_unsatisfied_subgoals() if path.subgoals else []
                
                # 将所有next nodes加入全局队列（线程安全）
                if global_node_queue:
                    # 获取visited set用于GlobalNodeQueue（线程安全）
                    if shared_state:
                        visited_for_queue = shared_state.get_visited_set()
                    else:
                        visited_for_queue = _global_visited
                    
                    added = global_node_queue.enqueue_nodes(
                        next_node_ids, 
                        self.graph_loader, 
                        path.item_id,
                        visited_for_queue,
                        unsatisfied_subgoals=unsatisfied_subgoals
                    )
                    
                    if added > 0:
                        path.total_nodes_selected += added
                        if added > 1:
                            path.multi_node_selection_count += 1
                        
                        # 从全局队列取出最高优先级的节点继续探索
                        next_node = global_node_queue.get_next_node()
                        if next_node:
                            path.current_node = next_node
                            logger.info(f"Path {path.path_id}: 从全局队列取出节点 {next_node['node_id']} 作为下一个探索目标")
                        else:
                            path.current_node = None
                    else:
                        # 如果所有节点都已访问，从全局队列取或fallback
                        logger.warning(f"Path {path.path_id}: 所有选择的节点都已访问")
                        if not global_node_queue.is_empty():
                            path.current_node = global_node_queue.get_next_node()
                            logger.info(f"Path {path.path_id}: 从全局队列取节点 {path.current_node['node_id']}")
                        elif neighbors:
                            fallback_node = neighbors[0]
                            path.current_node = fallback_node
                            logger.info(f"Path {path.path_id}: Fallback to neighbor {fallback_node['node_id']}")
                        else:
                            logger.info(f"Path {path.path_id}: No more nodes to explore")
                            path.current_node = None
                else:
                    logger.error("global_node_queue is None!")
                    path.current_node = None
        else:
            # LLM认为所有邻居都不相关
            logger.info(f"Path {path.path_id}: LLM认为所有邻居节点都不相关 (NEXT_NODES=NONE)")
            # 尝试从全局队列取节点
            if global_node_queue and not global_node_queue.is_empty():
                path.current_node = global_node_queue.get_next_node()
                logger.info(f"Path {path.path_id}: 从全局队列取节点 {path.current_node['node_id']} 继续探索")
            else:
                path.current_node = None
                logger.info(f"Path {path.path_id}: 全局队列为空，停止该路径探索")
    
    def explore_path_concurrent(self,
                               path: PathExplorerV2Queue,
                               question: str,
                               max_rounds: int,
                               shared_state: SharedExplorationState,
                               global_node_queue: GlobalNodeQueue,
                               enable_early_stopping: bool = False) -> PathExplorerV2Queue:
        """并发探索单个路径的worker函数（线程安全）
        
        Args:
            path: 要探索的路径
            question: 问题
            max_rounds: 最大探索轮数
            shared_state: 共享探索状态（线程安全）
            global_node_queue: 全局节点队列
            enable_early_stopping: 是否启用早停
            
        Returns:
            探索完成的路径对象
        """
        logger.info(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: 开始并发探索")
        
        try:
            for round_num in range(1, max_rounds + 1):
                # 检查早停条件（线程安全）
                if enable_early_stopping and shared_state.should_early_stop():
                    logger.info(f"Path {path.path_id}: 早停标志已设置，停止探索")
                    break
                
                # 检查subgoal是否全部满足（线程安全）
                if enable_early_stopping and path.subgoals and shared_state.all_subgoals_satisfied():
                    logger.info(f"Path {path.path_id}: 所有subgoal已满足，停止探索")
                    # 设置早停标志，通知其他线程
                    shared_state.set_early_stop()
                    break
                
                # 如果路径已找到答案，停止探索
                if path.has_answer:
                    logger.info(f"Path {path.path_id}: 已找到答案，停止探索")
                    break
                
                # 如果当前节点为None，从全局队列取新节点
                if path.current_node is None:
                    if not global_node_queue.is_empty():
                        path.current_node = global_node_queue.get_next_node()
                        if path.current_node:
                            logger.info(f"Path {path.path_id}: 从全局队列取新节点 {path.current_node['node_id']}")
                        else:
                            break
                    else:
                        logger.info(f"Path {path.path_id}: 全局队列为空且无当前节点，停止探索")
                        break
                
                # 探索一步（线程安全）
                self._explore_path_step(
                    path=path,
                    question=question,
                    round_num=round_num,
                    shared_state=shared_state,
                    global_node_queue=global_node_queue
                )
                
                # 检查是否达到答案后设置早停
                if enable_early_stopping and path.has_answer and path.subgoals:
                    if shared_state.all_subgoals_satisfied():
                        shared_state.set_early_stop()
            
            logger.info(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: 完成探索 "
                       f"(has_answer={path.has_answer}, rounds={len(path.round_details)})")
        
        except Exception as e:
            logger.error(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: 探索出错: {e}", exc_info=True)
        
        return path
    
    def refine_and_re_explore(self,
                             question: str,
                             initial_kept_nodes: List[str],
                             graph: Dict[str, Any],
                             item_id: str,
                             global_visited: Set[str],
                             global_kept_nodes: List[str],
                             top_k_clusters: int = 3) -> Dict[str, Any]:
        """生成refined query并重新探索（层次化版本，支持队列）
        
        Args:
            question: Original question
            initial_kept_nodes: Initially kept node IDs
            graph: Knowledge graph
            item_id: Item identifier
            global_visited: Already visited nodes
            global_kept_nodes: Globally kept nodes list
            top_k_clusters: Number of clusters for refinement exploration
            
        Returns:
            Dictionary with refinement results
        """
        logger.info("=== Starting Query Refinement (V2-Queue) ===")
        
        # 构建已保留信息的上下文
        if initial_kept_nodes:
            initial_context = self.context_formatter.build_final_context_from_kept_nodes(
                initial_kept_nodes, item_id
            )
        else:
            # 如果没有保留节点，使用所有访问过的节点
            initial_context = self.context_formatter.build_final_context_from_visited_nodes(
                list(global_visited), item_id
            )
        
        # Generate refined query
        refined_query, raw_refinement, formatted_refinement_prompt = self.llm_handler.generate_refinement_query(
            question, initial_context
        )
        
        logger.info(f"Refined query: {refined_query}")
        
        # 用refined query找新的聚类节点
        all_top_clusters = self.embedding_manager.find_top_k_cluster_nodes(
            refined_query,
            graph,
            k=top_k_clusters * 2
        )
        
        # 过滤：只保留有未访问成员的聚类
        available_clusters = []
        for cluster in all_top_clusters:
            member_nodes = cluster.get('member_nodes', [])
            unvisited_members = [m for m in member_nodes if m not in global_visited]
            if unvisited_members:
                available_clusters.append(cluster)
                if len(available_clusters) >= top_k_clusters:
                    break
        
        if not available_clusters:
            logger.warning("Refinement: No new clusters found")
            return {
                'refined_query': refined_query,
                'raw_refinement': raw_refinement,
                'formatted_refinement_prompt': formatted_refinement_prompt,
                'paths': [],
                'answer_paths': [],
                'success': False,
                'refined_clusters': []
            }
        
        logger.info(f"Refinement: Found {len(available_clusters)} available clusters")
        
        # 为每个聚类选择未访问成员节点（使用LLM基于summary，可能多个）
        start_nodes = []
        for i, cluster in enumerate(available_clusters):
            member_nodes = cluster.get('member_nodes', [])
            unvisited_member_ids = [m for m in member_nodes if m not in global_visited]
            
            if not unvisited_member_ids:
                continue
            
            # 获取未访问成员节点的summary信息
            member_summaries = []
            for node in graph.get('nodes', []):
                if node['id'] in unvisited_member_ids:
                    summary_info = {
                        'node_id': node['id'],
                        'summary': node.get('summaries', [''])[0] if node.get('summaries') else '',
                        'people': node.get('people', []),
                        'time': node.get('time_explicit', [''])[0] if node.get('time_explicit') else ''
                    }
                    member_summaries.append(summary_info)
            
            if not member_summaries:
                continue
            
            # 使用LLM选择节点（用refined_query，可能多个）
            selected_node_ids, _, _ = self.llm_handler.select_nodes_from_cluster(
                refined_query,
                cluster['node_id'],
                member_summaries
            )
            
            if selected_node_ids:
                for node_id in selected_node_ids:
                    selected_member = self.graph_loader.get_node_by_id(node_id, item_id)
                    if selected_member:
                        start_nodes.append(selected_member)
                logger.info(f"Refinement cluster {i+1} ({cluster['node_id']}): "
                          f"LLM selected {len(selected_node_ids)} nodes: {selected_node_ids}")
            else:
                # 兜底：使用embedding相似度选择最佳节点
                logger.warning(f"Refinement cluster {i+1}: LLM failed, using embedding fallback")
                best_member = self.embedding_manager.find_best_member_node(
                    refined_query,
                    unvisited_member_ids,
                    graph
                )
                if best_member:
                    start_nodes.append(best_member)
                    logger.info(f"Refinement cluster {i+1} ({cluster['node_id']}): "
                              f"Embedding selected node {best_member['node_id']}")
        
        if not start_nodes:
            logger.warning("Refinement: No start nodes selected")
            return {
                'refined_query': refined_query,
                'raw_refinement': raw_refinement,
                'formatted_refinement_prompt': formatted_refinement_prompt,
                'paths': [],
                'answer_paths': [],
                'success': False,
                'refined_clusters': [c['node_id'] for c in available_clusters]
            }
        
        # 初始化新路径（使用队列版本）
        new_paths = [PathExplorerV2Queue(i, node, item_id) for i, node in enumerate(start_nodes)]
        
        # 探索新路径（使用较少轮数）+ 队列处理
        refinement_rounds = max(1, self.max_rounds - 1)
        
        for round_num in range(1, refinement_rounds + 1):
            logger.info(f"=== Refinement Round {round_num}/{refinement_rounds} ===")
            
            for path in new_paths:
                if path.has_answer:
                    continue
                
                self._explore_path_step(
                    path=path,
                    question=question,  # 仍使用原始问题
                    round_num=round_num,
                    global_visited=global_visited,
                    global_kept_nodes=global_kept_nodes
                )
        
        # 处理refinement路径的队列
        logger.info("=== Refinement: 处理队列中剩余节点 ===")
        for path in new_paths:
            if path.has_answer:
                continue
            
            queue_round = refinement_rounds + 1
            while not path.node_queue.empty() and not path.has_answer:
                logger.info(f"Refinement Path {path.path_id}: 队列中还有 {path.node_queue.qsize()} 个节点待探索")
                
                # 从优先级队列中取出下一个节点
                priority, counter, next_node = path.node_queue.get()
                path.current_node = next_node
                
                # 继续探索
                self._explore_path_step(
                    path=path,
                    question=question,
                    round_num=queue_round,
                    global_visited=global_visited,
                    global_kept_nodes=global_kept_nodes
                )
                queue_round += 1
                
                # 为了避免无限循环，设置最大队列处理轮数
                if queue_round > refinement_rounds * 3:
                    logger.warning(f"Refinement Path {path.path_id}: 队列处理轮数超过限制，停止")
                    break
        
        answer_paths = [p for p in new_paths if p.has_answer]
        
        logger.info(f"Refinement complete: {len(answer_paths)}/{len(new_paths)} paths with answer")
        
        return {
            'refined_query': refined_query,
            'raw_refinement': raw_refinement,
            'formatted_refinement_prompt': formatted_refinement_prompt,
            'paths': new_paths,
            'answer_paths': answer_paths,
            'success': len(answer_paths) > 0,
            'refined_clusters': [c['node_id'] for c in available_clusters]
        }

