#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM交互处理模块 - V2 Queue版本
支持三动作机制：skip, expand, answer
支持选择多个下一节点（而不只是一个）
"""

import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from openai import OpenAI
from prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMHandlerV2Queue:
    """LLM交互处理器 - V2 Queue版本（支持多节点选择）- OpenAI API版本"""
    
    def __init__(self, model_name: str, api_base: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        """初始化LLM处理器
        
        Args:
            model_name: 模型名称（在vllm服务中配置的模型名）
            api_base: API服务的基础URL
            api_key: API密钥（本地部署时可以使用"EMPTY"）
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        
        # Token统计
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0  # 调用次数
        
        # 初始化模型
        self._init_models()
        
    def _init_models(self):
        """初始化OpenAI客户端"""
        logger.info(f"初始化OpenAI客户端，连接到: {self.api_base}")
        logger.info(f"使用模型: {self.model_name}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        logger.info(f"OpenAI客户端已初始化")
        
        # 设置采样参数
        self.generation_params = {
            'temperature': 0.6,
            'max_tokens': 8192,
            'top_p': 0.95
        }
    
    def _generate(self, messages: List[Dict[str, str]]) -> Tuple[str, None]:
        """使用OpenAI API生成响应并统计token
        
        Args:
            messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            
        Returns:
            Tuple of (raw_response, None)
            注意：第二个返回值为None以保持与原接口兼容（原来返回output_ids）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.generation_params
            )
            
            raw_response = response.choices[0].message.content
            
            # 统计tokens（如果API返回了usage信息）
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # 如果API没有返回usage信息，使用估算值
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                logger.warning("API未返回token使用信息")
            
            # 累加统计
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.call_count += 1
            
            logger.debug(f"本次调用 - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total: {total_tokens}")
            
            return raw_response, None
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            raise
    
    def _extract_non_thinking_content(self, raw_response: str, output_ids: list = None) -> str:
        """Extract non-thinking content from model output
        
        Args:
            raw_response: 模型的原始响应
            output_ids: 保留此参数以保持接口兼容性，但不再使用
        """
        # 使用正则表达式去除thinking标签内容
        text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = text.strip()
        
        return text
    
    def generate_subgoals(self, question: str) -> Tuple[List[str], str, str]:
        """使用LLM将问题拆分为subgoals
        
        Args:
            question: 原始问题
            
        Returns:
            Tuple of (subgoals_list, raw_response, formatted_prompt)
            subgoals_list: ['subgoal1', 'subgoal2', ...]
        """
        prompt = PromptTemplates.get_planner_prompt(question)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('planner')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # 解析subgoals
        subgoals = self._parse_subgoals(response)
        
        logger.info(f"Generated {len(subgoals)} sub-goals:")
        for i, sg in enumerate(subgoals):
            logger.info(f"  {i+1}. {sg}")
        
        formatted_prompt = str(messages)
        return subgoals, raw_response, formatted_prompt
    
    def _parse_subgoals(self, response: str) -> List[str]:
        """解析subgoals列表
        
        Returns:
            List of subgoal strings
        """
        subgoals = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配 "Sub-goal X: ..." 格式
            match = re.match(r'Sub-goal\s+\d+\s*:\s*(.+)', line, re.IGNORECASE)
            if match:
                subgoal = match.group(1).strip()
                subgoals.append(subgoal)
        
        # 如果没找到，返回空列表
        if not subgoals:
            logger.warning("Failed to parse sub-goals from response")
        
        return subgoals
    
    def check_action(self, question: str, kept_nodes_info: str, current_info: str, 
                     neighbor_info: str, round_num: int, 
                     subgoals: List[str] = None, subgoal_status: dict = None) -> Tuple[str, List[str], List[int], str, str]:
        """让LLM判断应该采取什么动作（skip/expand/answer）并选择下一节点（可能多个）
        
        Args:
            question: 问题
            kept_nodes_info: 已保留节点的信息
            current_info: 当前节点信息
            neighbor_info: 邻居节点信息
            round_num: 当前轮数
            subgoals: subgoal列表（可选，V3.5新增）
            subgoal_status: subgoal状态字典（可选，V3.5新增）
            
        Returns:
            Tuple of (action, next_node_ids, satisfied_subgoals, raw_response, formatted_prompt)
            action: 'skip', 'expand', 或 'answer'
            next_node_ids: 如果是skip或expand，指定的下一批节点ID列表（可以是多个）
            satisfied_subgoals: 满足的subgoal索引列表
        """
        
        # 根据是否有subgoals选择不同的prompt
        if subgoals and subgoal_status is not None:
            prompt = PromptTemplates.get_action_decision_prompt_with_subgoals(
                question, subgoals, subgoal_status, kept_nodes_info, current_info, neighbor_info
            )
        else:
            # 向后兼容：如果没有subgoals，使用原来的prompt
            prompt = PromptTemplates.get_action_decision_prompt(
                question, kept_nodes_info, current_info, neighbor_info
            )
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('action_decision')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # 解析响应
        action = self._parse_action(response)
        next_nodes = self._parse_next_nodes(response)  # 改成复数，返回列表
        
        # 解析满足的subgoals（如果使用了subgoal版本）
        satisfied_subgoals = []
        if subgoals:
            satisfied_subgoals = self._parse_satisfied_subgoals(response)
        
        logger.info(f"Round {round_num}: LLM决定 ACTION={action}, NEXT_NODES={next_nodes}, SATISFIED_SUBGOALS={satisfied_subgoals}")
        
        # 为保持接口兼容，返回messages的字符串表示作为formatted_prompt
        formatted_prompt = str(messages)
        return action, next_nodes, satisfied_subgoals, raw_response, formatted_prompt
    
    def _parse_satisfied_subgoals(self, response: str) -> List[int]:
        """解析满足的subgoal索引列表
        
        Returns:
            List of satisfied subgoal indices (0-based)
        """
        # 查找 SATISFIED_SUBGOALS: [1, 2, 3] 格式
        match = re.search(r'SATISFIED_SUBGOALS\s*:\s*\[([^\]]*)\]', response, re.IGNORECASE)
        
        if match:
            numbers_str = match.group(1)
            # 提取所有数字
            numbers = re.findall(r'\d+', numbers_str)
            # 转换为0-based索引
            satisfied = [int(n) - 1 for n in numbers if int(n) > 0]
            return satisfied
        
        return []
    
    def select_top_k_nodes(self, question: str, subgoals: List[str], 
                          candidate_nodes: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """从top-k候选节点中选择真正要探索的节点
        
        Args:
            question: 问题
            subgoals: subgoal列表
            candidate_nodes: 候选节点列表 [{'node_id': 'N1', 'summary': '...', 'similarity': 0.8}, ...]
            
        Returns:
            Tuple of (selected_node_ids, raw_response, formatted_prompt)
        """
        prompt = PromptTemplates.get_top_k_node_selection_prompt(
            question, subgoals, candidate_nodes
        )
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('top_k_selection')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # 解析选择的节点
        selected_nodes = self._parse_selected_nodes_from_top_k(response, 
                                                               [n['node_id'] for n in candidate_nodes])
        
        logger.info(f"LLM selected {len(selected_nodes)} nodes from top-k candidates: {selected_nodes}")
        
        formatted_prompt = str(messages)
        return selected_nodes, raw_response, formatted_prompt
    
    def _parse_selected_nodes_from_top_k(self, response: str, valid_node_ids: List[str]) -> List[str]:
        """解析从top-k中选择的节点ID列表
        
        Returns:
            List of selected node IDs
        """
        # 查找 "Selected Nodes:" 后面的节点列表
        match = re.search(r'Selected\s+Nodes\s*:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        
        if match:
            nodes_str = match.group(1)
            # 提取所有节点ID
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            # 过滤有效的节点ID
            valid_selected = [nid.upper() for nid in node_ids if nid.upper() in valid_node_ids]
            return valid_selected
        
        # 如果没找到，尝试在整个响应中查找节点ID
        node_ids = re.findall(r'\bN\d+\b', response)
        seen = set()
        unique_nodes = []
        for nid in node_ids:
            nid_upper = nid.upper()
            if nid_upper in valid_node_ids and nid_upper not in seen:
                seen.add(nid_upper)
                unique_nodes.append(nid_upper)
        
        return unique_nodes
    
    def _parse_action(self, response: str) -> str:
        """解析动作"""
        # 查找 ACTION: xxx
        action_match = re.search(r'ACTION\s*:\s*(SKIP|EXPAND|ANSWER)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()
            return action.lower()  # 返回小写
        
        # 如果没找到，尝试在响应中查找这些词
        response_upper = response.upper()
        if 'ACTION: ANSWER' in response_upper or 'ACTION:ANSWER' in response_upper:
            return 'answer'
        elif 'ACTION: EXPAND' in response_upper or 'ACTION:EXPAND' in response_upper:
            return 'expand'
        elif 'ACTION: SKIP' in response_upper or 'ACTION:SKIP' in response_upper:
            return 'skip'
        
        # 默认返回 expand（保守策略）
        logger.warning(f"无法解析动作，默认使用 expand")
        return 'expand'
    
    def _parse_next_nodes(self, response: str) -> List[str]:
        """解析下一批节点ID（可能是多个，最多3个）
        
        Returns:
            List of node IDs (e.g., ['N15', 'N22', 'N30'])，最多返回3个
        """
        # 查找 NEXT_NODES: 或 NEXT_NODE:
        next_nodes_match = re.search(r'NEXT_NODES?\s*:\s*([^\n]+)', response, re.IGNORECASE)
        
        if next_nodes_match:
            nodes_str = next_nodes_match.group(1).strip()
            
            # 检查是否是 NONE
            if 'NONE' in nodes_str.upper():
                return []
            
            # 提取所有节点ID（格式如 N15, N22, N30 或 N15,N22,N30）
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            if node_ids:
                # 转换为统一格式（首字母大写）并限制为最多3个
                node_ids = [nid.upper() for nid in node_ids][:3]
                if len(node_ids) > 3:
                    logger.warning(f"LLM选择了 {len(node_ids)} 个节点，限制为前3个")
                logger.info(f"解析到 {len(node_ids)} 个节点: {node_ids}")
                return node_ids
        
        # 如果没找到 NEXT_NODES:，尝试在整个响应中查找节点ID
        # 但只在没有明确说NONE的情况下
        if not re.search(r'NEXT_NODES?\s*:\s*NONE', response, re.IGNORECASE):
            node_ids = re.findall(r'\bN\d+\b', response)
            if node_ids:
                # 去重并保持顺序，限制为最多3个
                seen = set()
                unique_nodes = []
                for nid in node_ids:
                    nid_upper = nid.upper()
                    if nid_upper not in seen and len(unique_nodes) < 3:
                        seen.add(nid_upper)
                        unique_nodes.append(nid_upper)
                
                if unique_nodes:
                    logger.info(f"从响应中提取到 {len(unique_nodes)} 个节点: {unique_nodes}")
                    return unique_nodes
        
        return []
    
    def generate_answer(self, question: str, context: str, category: int = None) -> Tuple[str, str, str]:
        """基于上下文生成答案
        
        Args:
            question: 问题
            context: 上下文信息（已保留节点的原文）
            category: 问题类别（可选），category3会使用特殊的prompt
            
        Returns:
            Tuple of (answer, raw_response, formatted_prompt)
        """
        
        # 根据category选择prompt模板
        if category == 3:
            prompt = PromptTemplates.get_answer_generation_prompt_category3(question, context)
        else:
            prompt = PromptTemplates.get_answer_generation_prompt(question, context)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('answer_generation')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # 为保持接口兼容，返回messages的字符串表示作为formatted_prompt
        formatted_prompt = str(messages)
        return response.strip(), raw_response, formatted_prompt
    
    def generate_refinement_query(self, original_question: str, context_so_far: str,
                                  subgoals: List[str] = None, subgoal_status: dict = None) -> Tuple[str, str, str]:
        """Generate a refined query to find missing information
        
        Args:
            original_question: The original question being asked
            context_so_far: Information collected so far
            subgoals: subgoal列表（可选，V3.5新增）
            subgoal_status: subgoal状态字典（可选，V3.5新增）
            
        Returns:
            Tuple of (refined_query, raw_response, formatted_prompt)
        """
        # 根据是否有subgoals选择不同的prompt
        if subgoals and subgoal_status is not None:
            prompt = PromptTemplates.get_refinement_query_prompt_with_subgoals(
                original_question, subgoals, subgoal_status, context_so_far
            )
        else:
            # 向后兼容：使用原来的prompt
            prompt = PromptTemplates.get_refinement_query_prompt(original_question, context_so_far)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('refinement')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # Extract the new query from the response
        new_query = original_question  # Fallback to original
        lines = response.split('\n')
        for line in lines:
            if 'New Query:' in line or 'new query:' in line.lower():
                new_query = line.split(':', 1)[1].strip()
                break
        
        # 为保持接口兼容，返回messages的字符串表示作为formatted_prompt
        formatted_prompt = str(messages)
        return new_query, raw_response, formatted_prompt
    
    def select_nodes_from_cluster(self, question: str, cluster_id: str, 
                                  member_summaries: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """从聚类中选择多个相关成员节点（基于summary）
        
        Args:
            question: 问题
            cluster_id: 聚类ID
            member_summaries: 成员节点的summary信息列表
                              [{'node_id': 'N1', 'summary': '...', 'people': [...], 'time': '...'}, ...]
        
        Returns:
            Tuple of (selected_node_ids, raw_response, formatted_prompt)
            selected_node_ids: List of node IDs (可能包含多个节点)
        """
        if not member_summaries:
            return [], "", ""
        
        # 使用统一的prompt模板
        prompt = PromptTemplates.get_cluster_node_selection_prompt(question, member_summaries)
        
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_message('cluster_selection')},
            {"role": "user", "content": prompt}
        ]
        
        raw_response, output_ids = self._generate(messages)
        response = self._extract_non_thinking_content(raw_response, output_ids)
        
        # 解析选择的节点（可能多个）
        selected_node_ids = self._parse_selected_nodes(response, [m['node_id'] for m in member_summaries])
        
        # 为保持接口兼容，返回messages的字符串表示作为formatted_prompt
        formatted_prompt = str(messages)
        return selected_node_ids, raw_response, formatted_prompt
    
    def _parse_selected_nodes(self, response: str, valid_node_ids: List[str]) -> List[str]:
        """解析LLM选择的节点ID（可能多个，最多3个）
        
        Returns:
            List of valid node IDs，最多返回3个
        """
        # 方法1: 查找 "Selected Nodes:" 或 "Selected Node:" 后面的 ID(s)
        selected_pattern = r'Selected\s+Nodes?\s*:\s*([^\n]+)'
        match = re.search(selected_pattern, response, re.IGNORECASE)
        
        if match:
            nodes_str = match.group(1).strip()
            # 提取所有节点ID
            node_ids = re.findall(r'N\d+', nodes_str, re.IGNORECASE)
            if node_ids:
                # 过滤出有效的节点ID，限制为最多3个
                valid_selected = [nid.upper() for nid in node_ids if nid.upper() in valid_node_ids][:3]
                if valid_selected:
                    if len([nid for nid in node_ids if nid.upper() in valid_node_ids]) > 3:
                        logger.warning(f"LLM从聚类中选择了超过3个节点，限制为前3个")
                    logger.info(f"从聚类中解析到 {len(valid_selected)} 个节点: {valid_selected}")
                    return valid_selected
        
        # 方法2: 查找响应中出现的所有有效节点 ID
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, response)
        valid_matches = []
        seen = set()
        for match in matches:
            match_upper = match.upper()
            if match_upper in valid_node_ids and match_upper not in seen and len(valid_matches) < 3:
                seen.add(match_upper)
                valid_matches.append(match_upper)
        
        if valid_matches:
            logger.info(f"从响应中提取到 {len(valid_matches)} 个有效节点: {valid_matches}")
            return valid_matches
        
        return []
    
    def _parse_selected_node(self, response: str, valid_node_ids: List[str]) -> Optional[str]:
        """解析LLM选择的单个节点ID（保留用于向后兼容）"""
        # 方法1: 查找 "Selected Node:" 后面的 ID
        selected_pattern = r'Selected\s+Node\s*:\s*(N\d+)'
        match = re.search(selected_pattern, response, re.IGNORECASE)
        if match:
            node_id = match.group(1)
            if node_id in valid_node_ids:
                return node_id
        
        # 方法2: 查找响应中出现的第一个有效节点 ID
        pattern = r'\bN\d+\b'
        matches = re.findall(pattern, response)
        for match in matches:
            if match in valid_node_ids:
                return match
        
        return None
    
    def get_token_stats(self) -> Dict[str, int]:
        """获取token使用统计信息
        
        Returns:
            Dict包含:
                - total_prompt_tokens: 总prompt tokens数
                - total_completion_tokens: 总completion tokens数
                - total_tokens: 总tokens数
                - call_count: 总调用次数
                - avg_prompt_tokens: 平均prompt tokens数
                - avg_completion_tokens: 平均completion tokens数
        """
        stats = {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'call_count': self.call_count,
        }
        
        if self.call_count > 0:
            stats['avg_prompt_tokens'] = self.total_prompt_tokens / self.call_count
            stats['avg_completion_tokens'] = self.total_completion_tokens / self.call_count
        else:
            stats['avg_prompt_tokens'] = 0
            stats['avg_completion_tokens'] = 0
        
        return stats
    
    def reset_token_stats(self):
        """重置token统计信息"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        logger.info("Token统计信息已重置")
    
    def log_token_stats(self):
        """记录当前token使用统计到日志"""
        stats = self.get_token_stats()
        logger.info("=" * 60)
        logger.info("Token使用统计:")
        logger.info(f"  总调用次数: {stats['call_count']}")
        logger.info(f"  总Prompt Tokens: {stats['total_prompt_tokens']:,}")
        logger.info(f"  总Completion Tokens: {stats['total_completion_tokens']:,}")
        logger.info(f"  总Tokens: {stats['total_tokens']:,}")
        if stats['call_count'] > 0:
            logger.info(f"  平均Prompt Tokens: {stats['avg_prompt_tokens']:.1f}")
            logger.info(f"  平均Completion Tokens: {stats['avg_completion_tokens']:.1f}")
        logger.info("=" * 60)

