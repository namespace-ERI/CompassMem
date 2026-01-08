#!/bin/bash
# 运行层次化问答系统 V3 - 两阶段检索策略版本

# 设置工作目录
cd /share/project/zyt/hyy/Memory/QA_system

# 配置参数
GRAPHS_DIR="/share/project/zyt/hyy/Memory/build_graph/graphs_llm_14B_clustered"
QA_DATA_PATH="/share/project/zyt/hyy/Memory/data/locomo/locomo10.json"
MODEL_NAME="Qwen2.5-14B"  # vllm服务中的模型名称
EMBEDDING_MODEL="/share/project/zyt/hyy/Model/bge-m3"

# API配置（用于LLM）
API_BASE="http://localhost:8003/v1"  # vllm API地址
API_KEY="sk-DFS67w1gKg33DrKbOnGQOSjaEGw6aLi0gcvJcSRV8TIx0Yq2"  # API密钥，本地部署可使用EMPTY

# V3新参数：两阶段检索配置
TOP_K_NODES=5  # 首次直接检索的节点数量
TOP_K_PER_CLUSTER=3  # 聚类数量
N_PATHS=3  # 固定的探索路径数量

# V3.5新参数：subgoal和早停
ENABLE_EARLY_STOPPING=""  # 默认关闭早停（空字符串表示不加flag）
DISABLE_SUBGOAL_PLANNING=""  # 默认启用subgoal planning（空字符串表示不加flag）

# V3.6新参数：并发探索
# ENABLE_CONCURRENT=""  # 默认关闭并发（空字符串表示不加flag）
# 如果要启用并发，取消下面这行的注释：
ENABLE_CONCURRENT="--enable_concurrent"

# 其他参数
SIMILARITY_THRESHOLD=0.8
MAX_ROUNDS=3
EMBEDDING_GPU_ID=2

# 运行模式选择
MODE=${1:-"full"}  # full, debug, single, fast

echo "========================================"
echo "层次化QA系统 V3.6 - 两阶段检索策略 + Subgoal Planning + 并发探索"
echo "改进点：先直接检索节点再从聚类补充 + LLM筛选top-k + Subgoal跟踪 + 线程安全并发"
echo "使用OpenAI API调用方式"
echo "========================================"
echo "运行模式: $MODE"
echo "图数据目录: $GRAPHS_DIR"
echo "QA数据路径: $QA_DATA_PATH"
echo "LLM模型: $MODEL_NAME"
echo "LLM API地址: $API_BASE"
echo "Embedding模型: $EMBEDDING_MODEL"
echo "首次直接检索节点数 (top_k_nodes): $TOP_K_NODES"
echo "每聚类选择节点数 (top_k_per_cluster): $TOP_K_PER_CLUSTER"
echo "固定探索路径数量 (n_paths): $N_PATHS"
echo "最大轮数: $MAX_ROUNDS"
echo "Embedding GPU: $EMBEDDING_GPU_ID"
echo "Subgoal Planning: $([ -z \"$DISABLE_SUBGOAL_PLANNING\" ] && echo \"Enabled\" || echo \"Disabled\")"
echo "Early Stopping: $([ -z \"$ENABLE_EARLY_STOPPING\" ] && echo \"Disabled\" || echo \"Enabled\")"
echo "Concurrent Exploration: $([ -z \"$ENABLE_CONCURRENT\" ] && echo \"Disabled\" || echo \"Enabled\")"
echo "========================================"

case $MODE in
    "debug")
        echo "🔧 调试模式：只处理1个item，每个item 2个QA"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --debug_mode \
            --debug_items 1 \
            --debug_qa_per_item 2 \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "single")
        echo "🔍 单项目模式：处理1个item的所有QA"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --debug_mode \
            --debug_items 1 \
            --debug_qa_per_item 999 \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "full")
        echo "🚀 完整模式：处理所有item和QA"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "no_refinement")
        echo "🚫 无Refinement模式：处理所有item和QA，但禁用query refinement"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --disable_refinement \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "no_relation")
        echo "🔗 无Relation模式：处理所有item和QA，但不使用关系信息"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --no_relation \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "fast")
        echo "⚡ 快速模式：更激进的参数配置"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes 3 \
            --top_k_per_cluster 2 \
            --similarity_threshold 0.85 \
            --max_rounds 2 \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --disable_refinement \
            --enable_early_stopping \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "high_recall")
        echo "📊 高召回模式：增加检索节点数量"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes 8 \
            --top_k_per_cluster 5 \
            --similarity_threshold 0.75 \
            --max_rounds 3 \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING \
            $ENABLE_CONCURRENT
        ;;
    
    "concurrent")
        echo "🧵 并发模式：启用多线程并发探索（推荐3-5个paths）"
        python hierarchical_main_qa_system_v3.py \
            --graphs_dir "$GRAPHS_DIR" \
            --qa_data_path "$QA_DATA_PATH" \
            --model_name "$MODEL_NAME" \
            --embedding_model "$EMBEDDING_MODEL" \
            --top_k_nodes $TOP_K_NODES \
            --top_k_per_cluster $TOP_K_PER_CLUSTER \
            --n_paths $N_PATHS \
            --similarity_threshold $SIMILARITY_THRESHOLD \
            --max_rounds $MAX_ROUNDS \
            --embedding_gpu_id $EMBEDDING_GPU_ID \
            --api_base "$API_BASE" \
            --api_key "$API_KEY" \
            --enable_concurrent \
            $ENABLE_EARLY_STOPPING \
            $DISABLE_SUBGOAL_PLANNING
        ;;
    
    *)
        echo "❌ 未知模式: $MODE"
        echo "可用模式: debug, single, full, no_refinement, no_relation, fast, high_recall, concurrent"
        echo ""
        echo "模式说明："
        echo "  debug         - 调试模式 (1 item, 2 QA)"
        echo "  single        - 单项目模式 (1 item, 所有 QA)"
        echo "  full          - 完整模式 (所有数据)"
        echo "  no_refinement - 禁用 Refinement"
        echo "  no_relation   - 不使用关系信息"
        echo "  fast          - 快速模式 (更少节点) ⚡"
        echo "  high_recall   - 高召回模式 (更多节点) 📊"
        echo "  concurrent    - 并发模式 (多线程探索) 🧵"
        exit 1
        ;;
esac

echo "========================================"
echo "✅ 运行完成！"
echo "========================================"

