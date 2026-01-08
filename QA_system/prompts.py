#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Templates for Hierarchical QA System
统一管理系统中所有的prompt模板
"""


class PromptTemplates:
    """Prompt模板集合"""
    
    @staticmethod
    def get_action_decision_prompt(question: str, kept_nodes_info: str, 
                                   current_info: str, neighbor_info: str) -> str:
        """获取动作决策的prompt（skip/expand/answer）
        
        Args:
            question: 问题
            kept_nodes_info: 已保留节点的信息
            current_info: 当前节点信息
            neighbor_info: 邻居节点信息
            
        Returns:
            完整的prompt字符串
        """
        return f"""You are an expert information evaluator. Your task is to decide which action to take for the current node based on how relevant and sufficient it is for answering the given question.
        You have THREE possible actions:

1. **SKIP**: The current node is NOT helpful for answering the question.
   - Use SKIP when the current node contains completely irrelevant information that doesn't help answer the question at all
   - The current node will be DISCARDED, not used in final answer
   - You should specify which neighbor node(s) to explore next, OR specify NONE if ALL neighbors are irrelevant
   - **Multi-node selection rules**:
     * Maximum 3 nodes per selection
     * ONLY select nodes that appear HIGHLY relevant based on their summary and relation
     * If NO neighbors appear relevant to the question, specify NONE - this is valid and encouraged when appropriate
     * Prefer selecting 1 highly relevant node over 2-3 moderately relevant ones
     * Select multiple (2-3) ONLY when they clearly address different aspects of the question
   - When choosing next nodes, consider BOTH the neighbor's SUMMARY (what topic it covers) and RELATION (how it connects to current node) to identify which might contain relevant information
   - Example: If the question asks about travel dates, skip nodes about daily meals or unrelated hobbies

2. **EXPAND**: The current node IS helpful but NOT sufficient alone to fully answer the question.
   - Use EXPAND when the current node contains useful information, but you need MORE context or details to provide a complete answer
   - The current node will be KEPT and used in the final answer
   - You should specify which neighbor node(s) to explore next, OR specify NONE if ALL neighbors are irrelevant
   - **Multi-node selection rules**:
     * Maximum 3 nodes per selection
     * ONLY select nodes that appear HIGHLY relevant based on their summary and relation
     * If NO neighbors appear relevant to the question, specify NONE - the current node will still be kept
     * Be selective - quality over quantity
     * Select multiple (2-3) ONLY when they clearly provide complementary information about the SAME topic
   - When choosing next nodes, use BOTH the SUMMARY (to see if the topic is relevant) and RELATION (to understand the connection type, e.g., temporal_next for chronological events, spatial_related for location context, causal for cause-effect)
   - Example: If the question asks "When and where did Alice travel?", current node says "Alice went somewhere in May", keep it and explore nodes that might contain destination details

3. **ANSWER**: ONLY use this when you are CERTAIN you can provide a COMPLETE and ACCURATE answer to the question.
   - Use ANSWER ONLY when the previously kept information + current node together provide ALL necessary details to fully answer the question
   - The current node will be KEPT and exploration will STOP
   - No need to specify next nodes
   - Be conservative: If there's ANY doubt or missing detail, use EXPAND instead
   - Example: Question "When did Alice visit India?", and you have nodes stating "Alice visited India" and "The trip was in May 2022" → Now you can ANSWER

CRITICAL GUIDELINES:
- **ANSWER is the HIGHEST bar**: Only use it when you're confident the answer is complete. When in doubt, choose EXPAND
- **Navigate using SUMMARY + RELATION**: Neighbor summaries tell you WHAT the node is about, relations tell you HOW it connects (temporal, causal, spatial, etc.). Use both to make informed choices
- **Be precise**: SKIP truly irrelevant content, EXPAND when you need more, ANSWER only when certain
- **Summaries are previews, not full content**: The actual detailed content will be revealed when you explore the node

RESPONSE FORMAT (follow strictly):

ACTION: [SKIP/EXPAND/ANSWER]
NEXT_NODES: [NODE_ID1, NODE_ID2, ...]
REASONING: [Brief explanation considering: (1) relevance of current node, (2) completeness of accumulated information, (3) why the chosen next nodes (based on their summaries and relations) are HIGHLY relevant OR why all neighbors are irrelevant if NEXT_NODES is NONE]

Example 1 - EXPAND with single node (most common case):
ACTION: EXPAND
NEXT_NODES: N15
REASONING: Current node mentions Alice traveled but lacks dates. Keeping it as it's relevant. N15's summary specifically discusses "May 2022 India trip timeline" with temporal_next relation, making it the most directly relevant for finding the missing date information.

Example 2 - EXPAND with two nodes (multi-aspect question):
ACTION: EXPAND
NEXT_NODES: N15, N22
REASONING: Question asks "when and where" Alice traveled. Current node confirms the trip. N15's summary discusses "trip dates" with temporal_next relation. N22's summary specifically mentions "India destinations visited" with spatial_related relation. Both are HIGHLY relevant and address different required aspects.

Example 3 - SKIP with single node:
ACTION: SKIP
NEXT_NODES: N8
REASONING: Current node talks about food preferences, completely irrelevant to the question about travel dates. Skipping it. N8's summary specifically mentions "vacation planning and dates" with temporal_next relation, making it the most relevant next option.

Example 3.5 - SKIP with no relevant neighbors:
ACTION: SKIP
NEXT_NODES: NONE
REASONING: Current node discusses daily meal routines, completely irrelevant to the question about travel dates. All available neighbor nodes also discuss unrelated topics like hobbies and shopping preferences - none contain information about travel or dates. Skipping this entire branch.

Example 4 - EXPAND with three nodes (MAX, rarely used):
ACTION: EXPAND
NEXT_NODES: N30, N35, N40
REASONING: Question asks about complete India trip itinerary. Current node confirms the trip. N30 explicitly mentions "departure May 4", N35 discusses "visited Mumbai May 5-10", N40 covers "Delhi May 11-15". All three contain specific required information with temporal_next relations, forming a clear sequence.

Example 5 - EXPAND with no relevant neighbors:
ACTION: EXPAND
NEXT_NODES: NONE
REASONING: Current node confirms Alice traveled to India in May 2022, which is highly relevant and should be kept. However, all neighbor nodes discuss unrelated topics (work schedule, family events) with no connection to the India trip. Keeping current node but no further exploration from this branch is needed.

Example 6 - ANSWER (sufficient aggregation):
ACTION: ANSWER
NEXT_NODES: NONE
REASONING: Combined with kept nodes, I now have comprehensive information: (1) Alice traveled to India [Node N5], (2) Departure date May 4, 2022 [Node N10], (3) Return date May 18, 2022 [Node N10], (4) Visited Mumbai and Delhi [Node N15]. All aspects of "When and where did Alice visit India?" are thoroughly covered.


I will provide you with the following for this question:
- The question itself
- Previously kept (saved) information
- The current node's information
- Information about neighbor nodes available for exploration
All relevant details for this decision are given below:
QUESTION: {question}

PREVIOUSLY KEPT INFORMATION:
{kept_nodes_info if kept_nodes_info else "(No information kept yet)"}

CURRENT NODE INFORMATION:
{current_info}

NEIGHBOR NODES (available for exploration):
{neighbor_info}
Now, make your decision:
"""

    @staticmethod
    def get_answer_generation_prompt(question: str, context: str) -> str:
        """获取生成答案的prompt
        
        Args:
            question: 问题
            context: 上下文信息（已保留节点的原文）
            
        Returns:
            完整的prompt字符串
        """
        return f"""Your task is to answer the QUESTION based on the provided CONTEXT.

Requirements:
- Be concise and direct. Provide ONLY the answer in the form of **a short phrase**, not a sentence. No explanations or additional commentary.
- If the context contains direct statements that answer the question, use the original wording from the context
- If the context doesn't have direct statements, you may summarize and infer the answer from the relevant information
- If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from "4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
- Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March, 2023" based on the memory timestamp.
- If you are uncertain or lack sufficient information, do not state that the information is insufficient. Instead, provide a reasonable and well-justified answer based on general knowledge.
- Keep your answer brief and to the point


CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    @staticmethod
    def get_answer_generation_prompt_category3(question: str, context: str) -> str:
        """获取生成答案的prompt (专门针对category3问题)
        
        Args:
            question: 问题
            context: 上下文信息（已保留节点的原文）
            
        Returns:
            完整的prompt字符串
        """
        return f"""Your task is to answer the QUESTION based on the provided CONTEXT.

Requirements:
- Be concise and direct. Provide ONLY the answer in the form of **a short phrase**, not a sentence. No explanations or additional commentary.
- You MUST analyze and reason from the content to derive the answer.
- Synthesize information across different parts of the context and apply logical reasoning to arrive at your answer.
- If you are uncertain or lack sufficient information, do not state that the information is insufficient. Instead, provide a reasonable and well-justified answer based on general knowledge and your inference from the context.
- Use SHORT PHRASES rather than complete sentences to keep your answer brief and to the point.


CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    @staticmethod
    def get_refinement_query_prompt(original_question: str, context_so_far: str) -> str:
        """获取生成refined query的prompt
        
        Args:
            original_question: 原始问题
            context_so_far: 目前收集到的信息
            
        Returns:
            完整的prompt字符串
        """
        return f"""You are an assistant whose role is to evaluate existing information, detect gaps with respect to the original question, and generate an appropriate next query.

TASK:
Analyze what key information is STILL MISSING to answer the original question completely.
Then, generate a NEW search query that would help fill the gaps.

Your new query should:
1. Focus on the specific gaps in the current information
2. Be clear and specific
3. Use different keywords or phrases than the original question
4. Target information that would complement what we already have
5. Do not repeat or simply restate the original question in the new query.

ORIGINAL QUESTION:
{original_question}

INFORMATION COLLECTED SO FAR:
{context_so_far}

RESPONSE FORMAT:
Missing Information: [Describe what information is still needed]
New Query: [Your refined search query - this should be a single clear question or search phrase]

Generate your response:"""

    @staticmethod
    def get_cluster_node_selection_prompt(question: str, member_summaries: list) -> str:
        """获取从聚类中选择节点的prompt
        
        Args:
            question: 问题
            member_summaries: 成员节点的summary信息列表
                             [{'node_id': 'N1', 'summary': '...', 'people': [...], 'time': '...'}, ...]
            
        Returns:
            完整的prompt字符串
        """
        # 构建节点列表部分
        nodes_text = ""
        for member in member_summaries:
            node_id = member['node_id']
            summary = member.get('summary', '')
            people = ', '.join(member.get('people', [])) if member.get('people') else 'Unknown'
            time = member.get('time', '') if member.get('time') else 'Unknown'
            
            nodes_text += f"- {node_id}: {summary}\n"
            if people != 'Unknown' or time != 'Unknown':
                nodes_text += f"  (People: {people}, Time: {time})\n"
        
        return f"""You are selecting the most relevant memory node(s) to answer a question.

QUESTION: {question}

AVAILABLE NODES:

{nodes_text}

INSTRUCTIONS:
Select the node(s) that are HIGHLY relevant to answering the question.
- **Be selective**: Only choose nodes that are HIGHLY relevant to the question
- **Maximum 3 nodes**: Select at most 3 nodes per cluster
- If ONLY ONE node is clearly the most relevant, select just that one
- Select multiple nodes (2-3) ONLY when they are ALL highly relevant AND provide complementary information:
  * Information is distributed across multiple memories about the SAME event/topic
  * The question has multiple specific aspects that DIFFERENT nodes clearly address
  * Multiple nodes provide different pieces of the SAME answer
- Consider the summary content, people involved, and time information
- **Do NOT select nodes that are only tangentially related or might be vaguely relevant**

RESPONSE FORMAT:
Selected Nodes: [NODE_ID1, NODE_ID2, ...] 
Reason: [Brief explanation of why these specific nodes are HIGHLY relevant]

Example 1 - Single node (most common):
Selected Nodes: N5
Reason: This node directly contains information about Alice's India trip, which is exactly what the question asks about.

Example 2 - Multiple nodes (only when highly complementary):
Selected Nodes: N5, N12
Reason: N5 covers when Alice went to India (May 2022), and N12 describes where she visited (Mumbai, Delhi). Both are essential pieces directly answering the "when and where" question.
"""

    @staticmethod
    def get_planner_prompt(question: str) -> str:
        """获取问题规划的prompt，将问题拆分为subgoals
        
        Args:
            question: 原始问题
            
        Returns:
            完整的prompt字符串
        """
        return f"""You are a strategic planning assistant. Your task is to analyze a question and break it down into 2-5 specific sub-goals that need to be satisfied to fully answer the question.

QUESTION: {question}

INSTRUCTIONS:
1. Analyze what information components are needed to fully answer this question
2. Break down the question into 2-5 specific, concrete sub-goals
3. Each sub-goal should represent a distinct piece of information needed
4. Sub-goals should be:
   - Specific and clear (not vague)
   - Independently verifiable (can determine if it's satisfied)
   - Collectively sufficient (together they fully answer the question)
   - Atomic (each sub-goal addresses ONE aspect)

RESPONSE FORMAT (follow strictly):
Sub-goal 1: [First specific information need]
Sub-goal 2: [Second specific information need]
Sub-goal 3: [Third specific information need]
...

EXAMPLE:

Question: "When and where did Alice visit during her Asia trip in 2022?"
Sub-goal 1: Identify that Alice took an Asia trip in 2022
Sub-goal 2: Find the specific dates or time period of the trip
Sub-goal 3: Identify which countries or cities she visited
Sub-goal 4: Determine the order or sequence of locations if multiple places

Question: "Who attended the birthday party?"
Sub-goal 1: Identify which birthday party is being asked about
Sub-goal 2: Find the complete list of attendees

Now analyze the question and generate sub-goals:"""

    @staticmethod
    def get_action_decision_prompt_with_subgoals(question: str, subgoals: list, 
                                                  subgoal_status: dict, kept_nodes_info: str, 
                                                  current_info: str, neighbor_info: str) -> str:
        """获取动作决策的prompt（支持subgoal跟踪）
        
        Args:
            question: 问题
            subgoals: subgoal列表 ['subgoal1', 'subgoal2', ...]
            subgoal_status: subgoal状态字典 {0: True/False, 1: True/False, ...}
            kept_nodes_info: 已保留节点的信息
            current_info: 当前节点信息
            neighbor_info: 邻居节点信息
            
        Returns:
            完整的prompt字符串
        """
        # 构建subgoal状态显示
        subgoals_text = "SUB-GOALS (Information needs to fully answer the question):\n"
        for i, subgoal in enumerate(subgoals):
            status = "✓ SATISFIED" if subgoal_status.get(i, False) else "✗ NOT YET SATISFIED"
            subgoals_text += f"{i+1}. {subgoal} [{status}]\n"
        
        return f"""You are an expert information evaluator. Your task is to decide which action to take for the current node based on how relevant and sufficient it is for answering the given question.

You have THREE possible actions:

1. **SKIP**: The current node is NOT helpful for answering the question or satisfying any sub-goals.
   - Use SKIP when the current node contains completely irrelevant information
   - The current node will be DISCARDED, not used in final answer
   - You should specify which neighbor node(s) to explore next, OR specify NONE if ALL neighbors are irrelevant
   - **Multi-node selection rules**: Maximum 3 nodes, only select HIGHLY relevant ones

2. **EXPAND**: The current node IS helpful and helps satisfy some sub-goals, but NOT all sub-goals are satisfied yet.
   - Use EXPAND when the current node contains useful information for one or more sub-goals
   - The current node will be KEPT and used in the final answer
   - You should specify which neighbor node(s) to explore next to satisfy remaining sub-goals, OR specify NONE if no neighbors are relevant
   - **CRITICAL**: You MUST indicate which sub-goals are now satisfied by this node + previously kept information
   - Only mark a sub-goal as satisfied if you have DIRECT evidence in current or kept nodes

3. **ANSWER**: Use ONLY when ALL sub-goals are SATISFIED (or nearly all).
   - Use ANSWER when the previously kept information + current node together satisfy ALL sub-goals
   - The current node will be KEPT and exploration will STOP
   - **CRITICAL**: You MUST list ALL satisfied sub-goals to confirm completeness
   - Be conservative: If ANY sub-goal remains unsatisfied, use EXPAND instead

CRITICAL GUIDELINES:
- **Check sub-goals systematically**: For each action, explicitly evaluate which sub-goals are satisfied
- **ANSWER only when complete**: Use ANSWER only when ALL (or all critical) sub-goals are satisfied
- **Navigate strategically**: Choose next nodes that are likely to help satisfy remaining unsatisfied sub-goals
- **Be explicit about progress**: Always indicate which sub-goals your current decision addresses

RESPONSE FORMAT (follow strictly):

ACTION: [SKIP/EXPAND/ANSWER]
NEXT_NODES: [NODE_ID1, NODE_ID2, ...] (or NONE)
SATISFIED_SUBGOALS: [1, 3, 4] (REQUIRED for EXPAND/ANSWER: list sub-goal numbers NOW satisfied; [] for SKIP)
REASONING: [Brief explanation: (1) what information current node provides, (2) which sub-goals it helps satisfy, (3) which sub-goals remain unsatisfied, (4) why chosen next nodes target remaining sub-goals OR why no neighbors are relevant]

**IMPORTANT**: 
- For SKIP: SATISFIED_SUBGOALS must be []
- For EXPAND/ANSWER: You MUST provide SATISFIED_SUBGOALS list, even if empty
- Only include sub-goal numbers that NOW have DIRECT evidence in current + kept nodes
- Do NOT mark a sub-goal as satisfied based on speculation or future nodes

Example 1 - EXPAND (partial progress):
ACTION: EXPAND
NEXT_NODES: N15, N22
SATISFIED_SUBGOALS: [1, 2]
REASONING: Current node confirms Alice's Asia trip in 2022 (satisfies sub-goal 1) and mentions "departed May 4" (satisfies sub-goal 2 for dates). However, sub-goal 3 (locations visited) and sub-goal 4 (sequence) remain unsatisfied. N15 mentions "Mumbai visit" and N22 discusses "travel itinerary" - both highly relevant for the missing location information.

Example 2 - ANSWER (all satisfied):
ACTION: ANSWER
NEXT_NODES: NONE
SATISFIED_SUBGOALS: [1, 2, 3, 4]
REASONING: With current node + kept information, all sub-goals are now satisfied: (1) Asia trip in 2022 confirmed, (2) dates May 4-18 identified, (3) visited India, Thailand, Singapore, (4) sequence was India→Thailand→Singapore. Complete answer available.

Example 3 - SKIP (not helpful):
ACTION: SKIP
NEXT_NODES: N30
SATISFIED_SUBGOALS: []
REASONING: Current node discusses Alice's work schedule, completely irrelevant to any sub-goals about her Asia trip. N30 mentions "vacation destinations" which could be relevant for sub-goal 3 (locations).

QUESTION: {question}

{subgoals_text}

PREVIOUSLY KEPT INFORMATION:
{kept_nodes_info if kept_nodes_info else "(No information kept yet)"}

CURRENT NODE INFORMATION:
{current_info}

NEIGHBOR NODES (available for exploration):
{neighbor_info}

Now, make your decision:"""

    @staticmethod
    def get_top_k_node_selection_prompt(question: str, subgoals: list, 
                                       candidate_nodes: list) -> str:
        """获取从top-k候选节点中选择的prompt
        
        Args:
            question: 问题
            subgoals: subgoal列表
            candidate_nodes: 候选节点列表 [{'node_id': 'N1', 'summary': '...', 'similarity': 0.8}, ...]
            
        Returns:
            完整的prompt字符串
        """
        # 构建subgoal文本
        subgoals_text = "Sub-goals needed to answer this question:\n"
        for i, subgoal in enumerate(subgoals):
            subgoals_text += f"{i+1}. {subgoal}\n"
        
        # 构建候选节点列表
        nodes_text = ""
        for node in candidate_nodes:
            node_id = node['node_id']
            summary = node.get('summary', '')
            similarity = node.get('similarity', 0.0)
            nodes_text += f"- {node_id} (similarity: {similarity:.3f}): {summary}\n"
        
        return f"""You are selecting the most promising memory nodes to explore for answering a question.

QUESTION: {question}

{subgoals_text}

CANDIDATE NODES (retrieved by semantic similarity):
{nodes_text}

INSTRUCTIONS:
Select the nodes that are HIGHLY LIKELY to contain information relevant to one or more sub-goals.
- **Be selective**: Only choose nodes whose summaries clearly indicate relevance to specific sub-goals
- **Maximum 5 nodes**: Select at most 5 nodes to explore
- **Diversity**: Try to select nodes that address different sub-goals if possible
- **Quality over quantity**: It's better to select 2 highly relevant nodes than 5 marginally relevant ones
- If a node's summary is vague or doesn't clearly relate to any sub-goal, DON'T select it
- Consider both the summary content and the similarity score

RESPONSE FORMAT:
Selected Nodes: [NODE_ID1, NODE_ID2, ...]
Reasoning: [Brief explanation of why each selected node is likely relevant to specific sub-goals]

Example:
Selected Nodes: N5, N12, N18
Reasoning: N5's summary mentions "Asia travel May 2022" which directly addresses sub-goals 1 and 2 (trip confirmation and dates). N12 discusses "visited Mumbai and Bangkok" relevant for sub-goal 3 (locations). N18 mentions "travel itinerary sequence" which addresses sub-goal 4 (order of locations). These three nodes collectively cover most sub-goals.

Now make your selection:"""

    @staticmethod
    def get_refinement_query_prompt_with_subgoals(original_question: str, 
                                                   subgoals: list, 
                                                   subgoal_status: dict,
                                                   context_so_far: str) -> str:
        """获取生成refined query的prompt（支持subgoal状态）
        
        Args:
            original_question: 原始问题
            subgoals: subgoal列表
            subgoal_status: subgoal状态字典
            context_so_far: 目前收集到的信息
            
        Returns:
            完整的prompt字符串
        """
        # 构建subgoal状态显示
        satisfied_goals = []
        unsatisfied_goals = []
        for i, subgoal in enumerate(subgoals):
            if subgoal_status.get(i, False):
                satisfied_goals.append(f"{i+1}. {subgoal}")
            else:
                unsatisfied_goals.append(f"{i+1}. {subgoal}")
        
        satisfied_text = "\n".join(satisfied_goals) if satisfied_goals else "(None yet)"
        unsatisfied_text = "\n".join(unsatisfied_goals) if unsatisfied_goals else "(All satisfied)"
        
        return f"""You are an assistant whose role is to generate a refined search query to find missing information.

ORIGINAL QUESTION:
{original_question}

SUB-GOALS STATUS:
Satisfied sub-goals:
{satisfied_text}

Unsatisfied sub-goals:
{unsatisfied_text}

INFORMATION COLLECTED SO FAR:
{context_so_far}

TASK:
Generate a NEW search query that specifically targets the UNSATISFIED sub-goals.

Your new query should:
1. Focus on the specific unsatisfied sub-goals
2. Be clear and specific
3. Use different keywords or phrases than the original question
4. Target information that would help satisfy the remaining sub-goals
5. NOT repeat the original question

RESPONSE FORMAT:
New Query: [Your refined search query - single clear question or search phrase targeting unsatisfied sub-goals]
Target Sub-goals: [List which sub-goal numbers this query aims to satisfy]

Generate your response:"""

    @staticmethod
    def get_system_message(task_type: str) -> str:
        """获取系统消息
        
        Args:
            task_type: 任务类型 ('action_decision', 'answer_generation', 
                      'refinement', 'cluster_selection', 'planner', 'top_k_selection')
            
        Returns:
            系统消息字符串
        """
        system_messages = {
            'action_decision': "You are a helpful assistant that evaluates information relevance and makes exploration decisions.",
            'answer_generation': "You are a helpful assistant that answers questions based on provided context information.",
            'refinement': "You are a helpful assistant that helps refine search queries to find missing information.",
            'cluster_selection': "You are a helpful assistant that selects the most relevant memory node(s) based on summaries.",
            'planner': "You are a strategic planning assistant that breaks down questions into sub-goals.",
            'top_k_selection': "You are a helpful assistant that selects the most promising nodes to explore based on their summaries."
        }
        return system_messages.get(task_type, "You are a helpful assistant.")

