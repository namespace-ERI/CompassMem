#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-path exploration module - V2 Queue version
Supports three-action mechanism (skip/expand/answer) + multi-node queue management + concurrent exploration
"""

import logging
from typing import List, Dict, Any, Set, Optional
from queue import PriorityQueue
import heapq
import threading

logger = logging.getLogger(__name__)


class GlobalNodeQueue:
    """Global node priority queue manager (shared by all paths, thread-safe)"""
    
    def __init__(self, embedding_manager=None):
        self.queue = PriorityQueue()  # PriorityQueue is thread-safe
        self.priority_cache = {}  # Cache node similarity {node_id: (priority, node)}
        self.cache_lock = threading.Lock()  # Lock to protect priority_cache
        self.counter = 0  # Used to ensure FIFO order for same priority
        self.counter_lock = threading.Lock()  # Lock to protect counter
        self.embedding_manager = embedding_manager
        self.max_queue_size = 0  # Record maximum queue length
        self.stats_lock = threading.Lock()  # Lock to protect statistics
        
    def enqueue_nodes(self, node_ids: List[str], graph_loader, item_id: str,
                     global_visited: Set[str], unsatisfied_subgoals: List[str] = None):
        """Add multiple nodes to global priority queue (thread-safe)
        
        Optimization points:
        1. Compute similarity for each node only once (using cache)
        2. Use PriorityQueue to automatically maintain global sorting
        3. Only compute for new nodes on insertion, O(k log n)
        4. Thread-safe: use locks to protect shared resources
        """
        # Collect unvisited nodes (need to check global_visited with lock)
        nodes_to_add = []
        for node_id in node_ids:
            if node_id not in global_visited:  # global_visited is locked externally
                node = graph_loader.get_node_by_id(node_id, item_id)
                if node:
                    nodes_to_add.append(node)
        
        if not nodes_to_add:
            return 0
        
        added_count = 0
        
        # Only compute similarity for new nodes and insert
        for node in nodes_to_add:
            node_id = node.get('node_id')
            
            # Check cache to avoid duplicate computation (with lock)
            with self.cache_lock:
                if node_id in self.priority_cache:
                    priority, _ = self.priority_cache[node_id]
                    logger.debug(f"Global Queue: Using cached priority for {node_id}: {-priority:.3f}")
                else:
                    # Only compute similarity for new nodes
                    if self.embedding_manager and unsatisfied_subgoals and len(unsatisfied_subgoals) > 0:
                        # Compute maximum similarity with unsatisfied subgoals
                        similarity = self.embedding_manager.compute_node_max_similarity_to_subgoals(
                            node, unsatisfied_subgoals
                        )
                        # Use negative value as priority (PriorityQueue is min-heap, we want max similarity)
                        priority = -similarity
                    else:
                        # When no subgoals, use FIFO order (same priority)
                        priority = 0.0
                    
                    # Cache priority
                    self.priority_cache[node_id] = (priority, node)
                    logger.debug(f"Global Queue: Computed priority for {node_id}: {-priority:.3f}")
            
            # Get counter (with lock)
            with self.counter_lock:
                counter = self.counter
                self.counter += 1
            
            # Insert into priority queue: (priority, counter, node)
            # PriorityQueue.put() is thread-safe
            self.queue.put((priority, counter, node))
            added_count += 1
        
        # Update statistics (with lock)
        if added_count > 0:
            current_size = self.queue.qsize()
            with self.stats_lock:
                if current_size > self.max_queue_size:
                    self.max_queue_size = current_size
            
            logger.info(f"Global Queue: Added {added_count}/{len(node_ids)} nodes, "
                       f"current queue length: {current_size}")
        
        return added_count
    
    def get_next_node(self):
        """Get highest priority node from queue (thread-safe)"""
        if self.queue.empty():
            return None
        # PriorityQueue.get() is thread-safe
        priority, counter, node = self.queue.get()
        logger.debug(f"Global Queue: Retrieved node {node['node_id']} (priority: {-priority:.3f})")
        return node
    
    def is_empty(self):
        """Check if queue is empty (thread-safe)"""
        return self.queue.empty()
    
    def size(self):
        """Get queue size (thread-safe)"""
        return self.queue.qsize()


class SharedExplorationState:
    """Manage shared state in concurrent exploration (thread-safe)"""
    
    def __init__(self, subgoals: List[str] = None):
        self.visited = set()  # Global visited nodes
        self.visited_lock = threading.Lock()
        
        self.kept_nodes = []  # Global kept nodes
        self.kept_nodes_lock = threading.Lock()
        
        self.subgoal_status = {}  # Subgoal completion status
        if subgoals:
            self.subgoal_status = {i: False for i in range(len(subgoals))}
        self.subgoal_lock = threading.Lock()
        
        self.early_stop_flag = False  # Early stop flag
        self.early_stop_lock = threading.Lock()
    
    def add_visited(self, node_id: str) -> bool:
        """Add visited node (thread-safe)
        
        Returns:
            True if node was not visited before, False if already visited
        """
        with self.visited_lock:
            if node_id in self.visited:
                return False
            self.visited.add(node_id)
            return True
    
    def is_visited(self, node_id: str) -> bool:
        """Check if node has been visited (thread-safe)"""
        with self.visited_lock:
            return node_id in self.visited
    
    def add_kept_node(self, node_id: str):
        """Add kept node (thread-safe)"""
        with self.kept_nodes_lock:
            if node_id not in self.kept_nodes:
                self.kept_nodes.append(node_id)
    
    def get_kept_nodes(self) -> List[str]:
        """Get copy of kept node list (thread-safe)"""
        with self.kept_nodes_lock:
            return self.kept_nodes.copy()
    
    def update_subgoal_status(self, satisfied_indices: List[int]):
        """Update subgoal status (thread-safe)"""
        with self.subgoal_lock:
            for idx in satisfied_indices:
                if idx in self.subgoal_status:
                    self.subgoal_status[idx] = True
    
    def get_subgoal_status(self) -> dict:
        """Get copy of subgoal status (thread-safe)"""
        with self.subgoal_lock:
            return self.subgoal_status.copy()
    
    def all_subgoals_satisfied(self) -> bool:
        """Check if all subgoals are satisfied (thread-safe)"""
        with self.subgoal_lock:
            if not self.subgoal_status:
                return False
            return all(self.subgoal_status.values())
    
    def set_early_stop(self):
        """Set early stop flag (thread-safe)"""
        with self.early_stop_lock:
            self.early_stop_flag = True
    
    def should_early_stop(self) -> bool:
        """Check if early stop should be triggered (thread-safe)"""
        with self.early_stop_lock:
            return self.early_stop_flag
    
    def get_visited_set(self) -> Set[str]:
        """Get copy of visited set (for passing to GlobalNodeQueue)"""
        with self.visited_lock:
            return self.visited.copy()


class PathExplorerV2Queue:
    """Handles exploration of a single path - V3.5 global queue version (all paths share one priority queue)"""
    
    def __init__(self, path_id: int, start_node: Dict[str, Any], item_id: str, 
                 subgoals: List[str] = None):
        self.path_id = path_id
        self.item_id = item_id
        self.start_node = start_node
        self.current_node = start_node
        self.local_visited = []  # Nodes visited in this path (ordered)
        self.round_details = []  # Details of each round in this path
        self.has_answer = False  # Whether answer decision has been made
        self.final_decision = None
        self.all_skipped = True  # Whether all nodes were skipped
        
        # V3.5 refactoring: remove independent queue, use global queue
        # self.node_queue removed, use global queue instead
        self.multi_node_selection_count = 0  # Number of multi-node selections (selecting >1 nodes)
        self.total_nodes_selected = 0  # Total number of nodes selected
        
        # V3.5 new: subgoal tracking
        self.subgoals = subgoals if subgoals else []
        # Subgoal status dictionary {index: True/False}, all False initially
        self.subgoal_status = {i: False for i in range(len(self.subgoals))} if subgoals else {}
        
    def add_round(self, round_info: Dict[str, Any]):
        """Add information about a round of exploration"""
        self.round_details.append(round_info)
        self.local_visited.append(round_info.get('current_node'))
        
        # Check if any node was kept (expand or answer)
        if round_info.get('action') in ['expand', 'answer']:
            self.all_skipped = False
    
    def update_subgoal_status(self, satisfied_subgoal_indices: List[int]):
        """Update subgoal status
        
        Args:
            satisfied_subgoal_indices: List of satisfied subgoal indices (0-based)
        """
        for idx in satisfied_subgoal_indices:
            if idx in self.subgoal_status:
                self.subgoal_status[idx] = True
        
        # Log to file
        satisfied_count = sum(1 for v in self.subgoal_status.values() if v)
        total_count = len(self.subgoal_status)
        if total_count > 0:
            logger.info(f"Path {self.path_id}: Subgoal progress: {satisfied_count}/{total_count} satisfied")
    
    def get_unsatisfied_subgoals(self) -> List[str]:
        """Get list of unsatisfied subgoal texts"""
        unsatisfied = []
        for i, satisfied in self.subgoal_status.items():
            if not satisfied and i < len(self.subgoals):
                unsatisfied.append(self.subgoals[i])
        return unsatisfied
    
    def all_subgoals_satisfied(self) -> bool:
        """Check if all subgoals are satisfied"""
        if not self.subgoal_status:
            return False
        return all(self.subgoal_status.values())


class MultiPathExplorerV2Queue:
    """Multi-path exploration manager - V2 Queue version (supports multi-node queue)"""
    
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
                          # Backward compatibility parameters
                          global_visited: Set[str] = None,
                          global_kept_nodes: List[str] = None,
                          global_subgoal_status: dict = None) -> None:
        """Explore one step of a path (supports global queue + subgoal tracking + concurrent exploration)
        
        Args:
            path: The path being explored
            question: The question being answered
            round_num: Current round number
            shared_state: Shared exploration state (thread-safe, recommended)
            global_node_queue: Global node queue (required)
            
            # Backward compatibility parameters (ignored if shared_state is provided)
            global_visited: Globally visited nodes (deprecated, use shared_state)
            global_kept_nodes: Globally kept nodes (deprecated, use shared_state)
            global_subgoal_status: Global subgoal status dictionary (deprecated, use shared_state)
        """
        # Backward compatibility: if no shared_state, use old parameters
        if shared_state is None:
            # Use old non-thread-safe interface
            _global_visited = global_visited if global_visited is not None else set()
            _global_kept_nodes = global_kept_nodes if global_kept_nodes is not None else []
            _global_subgoal_status = global_subgoal_status if global_subgoal_status is not None else {}
        else:
            # Use new thread-safe interface
            _global_visited = None  # Access via shared_state
            _global_kept_nodes = None  # Access via shared_state
            _global_subgoal_status = None  # Access via shared_state
        
        # Check if current_node is None (path has stopped exploration)
        if path.current_node is None:
            logger.info(f"Path {path.path_id}: current_node is None, path stopped exploration, skipping")
            return
        
        current_node_id = path.current_node['node_id']
        
        # Skip if already visited globally (thread-safe)
        if shared_state is not None:
            # Use thread-safe add_visited, which atomically checks and adds
            if not shared_state.add_visited(current_node_id):
                logger.info(f"Path {path.path_id}: Node {current_node_id} already visited globally, skipping")
                path.has_answer = False
                return
        else:
            # Backward compatibility: non-thread-safe version
            if current_node_id in _global_visited:
                logger.info(f"Path {path.path_id}: Node {current_node_id} already visited globally, skipping")
                path.has_answer = False
                return
            _global_visited.add(current_node_id)
        
        # Get information of kept nodes (thread-safe)
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
        
        # Get neighbor nodes (filtered by global visited) (thread-safe)
        neighbors = self.graph_loader.get_neighbor_nodes(current_node_id, path.item_id)
        if shared_state is not None:
            neighbors = [n for n in neighbors if not shared_state.is_visited(n.get('node_id'))]
        else:
            neighbors = [n for n in neighbors if n.get('node_id') not in _global_visited]
        neighbor_info = self.context_formatter.format_neighbor_info(neighbors, current_node_id)
        
        logger.info(f"Path {path.path_id}, Round {round_num}: Node {current_node_id}, "
                   f"{len(neighbors)} unvisited neighbors, {len(kept_nodes_list)} kept nodes")
        
        # Ask LLM to decide action (support subgoal version)
        if shared_state is not None:
            current_subgoal_status = shared_state.get_subgoal_status()
        else:
            current_subgoal_status = _global_subgoal_status
        
        if path.subgoals and current_subgoal_status:
            action, next_node_ids, satisfied_subgoals, raw_decision, formatted_prompt = self.llm_handler.check_action(
                question, kept_nodes_info, current_info, neighbor_info, round_num,
                subgoals=path.subgoals, subgoal_status=current_subgoal_status
            )
            
            # Update global subgoal status (thread-safe)
            if shared_state is not None:
                shared_state.update_subgoal_status(satisfied_subgoals)
            else:
                for idx in satisfied_subgoals:
                    if idx in _global_subgoal_status:
                        _global_subgoal_status[idx] = True
            
            # Update path-local subgoal status
            path.update_subgoal_status(satisfied_subgoals)
        else:
            # Backward compatibility: don't use subgoals
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
        
        # Update state based on action (thread-safe)
        if action == 'answer':
            # Keep current node and stop exploration
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
            # Keep current node and continue exploration
            if shared_state is not None:
                shared_state.add_kept_node(current_node_id)
            else:
                if current_node_id not in _global_kept_nodes:
                    _global_kept_nodes.append(current_node_id)
            logger.info(f"Path {path.path_id}: ACTION=EXPAND, kept node {current_node_id}, "
                       f"continue to {len(next_node_ids)} nodes: {next_node_ids}")
        
        elif action == 'skip':
            # Discard current node, explore new nodes
            logger.info(f"Path {path.path_id}: ACTION=SKIP, discarded node {current_node_id}, "
                       f"move to {len(next_node_ids)} nodes: {next_node_ids}")
        
        # Handle multiple next_node_ids
        if next_node_ids and len(next_node_ids) > 0:
            # Optimization: if only 1 node, directly set as current_node and continue
            if len(next_node_ids) == 1:
                next_node = self.graph_loader.get_node_by_id(next_node_ids[0], path.item_id)
                # Check if node is already visited (thread-safe)
                is_visited = shared_state.is_visited(next_node_ids[0]) if shared_state else (next_node_ids[0] in _global_visited)
                
                if next_node and not is_visited:
                    path.current_node = next_node
                    logger.info(f"Path {path.path_id}: Only 1 next node {next_node_ids[0]}, directly continue exploration")
                else:
                    # Node invalid or already visited, get from global queue
                    logger.warning(f"Path {path.path_id}: Selected node {next_node_ids[0]} is invalid or already visited")
                    if global_node_queue and not global_node_queue.is_empty():
                        path.current_node = global_node_queue.get_next_node()
                        logger.info(f"Path {path.path_id}: Get node {path.current_node['node_id']} from global queue")
                    elif neighbors:
                        fallback_node = neighbors[0]
                        path.current_node = fallback_node
                        logger.info(f"Path {path.path_id}: Fallback to neighbor {fallback_node['node_id']}")
                    else:
                        path.current_node = None
                        logger.info(f"Path {path.path_id}: No more nodes to explore")
            
            # If 2 or more nodes, add to global queue
            elif len(next_node_ids) >= 2:
                # Get unsatisfied subgoal list (for queue sorting)
                unsatisfied_subgoals = path.get_unsatisfied_subgoals() if path.subgoals else []
                
                # Add all next nodes to global queue (thread-safe)
                if global_node_queue:
                    # Get visited set for GlobalNodeQueue (thread-safe)
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
                        
                        # Get highest priority node from global queue to continue exploration
                        next_node = global_node_queue.get_next_node()
                        if next_node:
                            path.current_node = next_node
                            logger.info(f"Path {path.path_id}: Retrieved node {next_node['node_id']} from global queue as next exploration target")
                        else:
                            path.current_node = None
                    else:
                        # If all nodes already visited, get from global queue or fallback
                        logger.warning(f"Path {path.path_id}: All selected nodes already visited")
                        if not global_node_queue.is_empty():
                            path.current_node = global_node_queue.get_next_node()
                            logger.info(f"Path {path.path_id}: Get node {path.current_node['node_id']} from global queue")
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
            # LLM thinks all neighbors are irrelevant
            logger.info(f"Path {path.path_id}: LLM thinks all neighbor nodes are irrelevant (NEXT_NODES=NONE)")
            # Try to get node from global queue
            if global_node_queue and not global_node_queue.is_empty():
                path.current_node = global_node_queue.get_next_node()
                logger.info(f"Path {path.path_id}: Get node {path.current_node['node_id']} from global queue to continue exploration")
            else:
                path.current_node = None
                logger.info(f"Path {path.path_id}: Global queue is empty, stop this path exploration")
    
    def explore_path_concurrent(self,
                               path: PathExplorerV2Queue,
                               question: str,
                               max_rounds: int,
                               shared_state: SharedExplorationState,
                               global_node_queue: GlobalNodeQueue,
                               enable_early_stopping: bool = False) -> PathExplorerV2Queue:
        """Worker function for concurrent exploration of a single path (thread-safe)
        
        Args:
            path: Path to explore
            question: Question
            max_rounds: Maximum exploration rounds
            shared_state: Shared exploration state (thread-safe)
            global_node_queue: Global node queue
            enable_early_stopping: Whether to enable early stopping
            
        Returns:
            Completed path object
        """
        logger.info(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: Starting concurrent exploration")
        
        try:
            for round_num in range(1, max_rounds + 1):
                # Check early stop condition (thread-safe)
                if enable_early_stopping and shared_state.should_early_stop():
                    logger.info(f"Path {path.path_id}: Early stop flag set, stopping exploration")
                    break
                
                # Check if all subgoals are satisfied (thread-safe)
                if enable_early_stopping and path.subgoals and shared_state.all_subgoals_satisfied():
                    logger.info(f"Path {path.path_id}: All subgoals satisfied, stopping exploration")
                    # Set early stop flag to notify other threads
                    shared_state.set_early_stop()
                    break
                
                # If path has found answer, stop exploration
                if path.has_answer:
                    logger.info(f"Path {path.path_id}: Answer found, stopping exploration")
                    break
                
                # If current node is None, get new node from global queue
                if path.current_node is None:
                    if not global_node_queue.is_empty():
                        path.current_node = global_node_queue.get_next_node()
                        if path.current_node:
                            logger.info(f"Path {path.path_id}: Get new node {path.current_node['node_id']} from global queue")
                        else:
                            break
                    else:
                        logger.info(f"Path {path.path_id}: Global queue empty and no current node, stopping exploration")
                        break
                
                # Explore one step (thread-safe)
                self._explore_path_step(
                    path=path,
                    question=question,
                    round_num=round_num,
                    shared_state=shared_state,
                    global_node_queue=global_node_queue
                )
                
                # Check if answer reached and set early stop
                if enable_early_stopping and path.has_answer and path.subgoals:
                    if shared_state.all_subgoals_satisfied():
                        shared_state.set_early_stop()
            
            logger.info(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: Exploration completed "
                       f"(has_answer={path.has_answer}, rounds={len(path.round_details)})")
        
        except Exception as e:
            logger.error(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: Exploration error: {e}", exc_info=True)
        
        return path
    
        try:
            for round_num in range(1, max_rounds + 1):
                # Check early stop condition (thread-safe)
                if enable_early_stopping and shared_state.should_early_stop():
                    logger.info(f"Path {path.path_id}: Early stop flag set, stopping exploration")
                    break
                
                # Check if all subgoals are satisfied (thread-safe)
                if enable_early_stopping and path.subgoals and shared_state.all_subgoals_satisfied():
                    logger.info(f"Path {path.path_id}: All subgoals satisfied, stopping exploration")
                    # Set early stop flag to notify other threads
                    shared_state.set_early_stop()
                    break
                
                # If path has found answer, stop exploration
                if path.has_answer:
                    logger.info(f"Path {path.path_id}: Answer found, stopping exploration")
                    break
                
                # If current node is None, get new node from global queue
                if path.current_node is None:
                    if not global_node_queue.is_empty():
                        path.current_node = global_node_queue.get_next_node()
                        if path.current_node:
                            logger.info(f"Path {path.path_id}: Get new node {path.current_node['node_id']} from global queue")
                        else:
                            break
                    else:
                        logger.info(f"Path {path.path_id}: Global queue empty and no current node, stopping exploration")
                        break
                
                # Explore one step (thread-safe)
                self._explore_path_step(
                    path=path,
                    question=question,
                    round_num=round_num,
                    shared_state=shared_state,
                    global_node_queue=global_node_queue
                )
                
                # Check if answer reached and set early stop
                if enable_early_stopping and path.has_answer and path.subgoals:
                    if shared_state.all_subgoals_satisfied():
                        shared_state.set_early_stop()
            
            logger.info(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: Exploration completed "
                       f"(has_answer={path.has_answer}, rounds={len(path.round_details)})")
        
        except Exception as e:
            logger.error(f"[Thread-{threading.current_thread().name}] Path {path.path_id}: Exploration error: {e}", exc_info=True)
        
        return path
        
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

