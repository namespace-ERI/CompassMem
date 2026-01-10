#!/bin/bash
# Run Hierarchical QA System V3 - Two-Stage Retrieval Strategy Version

# Set working directory  
# cd /path/to/QA_system

# Configuration parameters
GRAPHS_DIR="./graphs_llm_clustered"
QA_DATA_PATH="./data/locomo10.json"
MODEL_NAME="Qwen2.5-14B"  # Model name in vLLM service
EMBEDDING_MODEL="./models/bge-m3"

# API configuration (for LLM)
API_BASE="http://localhost:8003/v1"  # vLLM API address
API_KEY="EMPTY"  # API key, use EMPTY for local deployment

# V3 new parameters: Two-stage retrieval configuration
TOP_K_NODES=5  # Number of nodes for first direct retrieval
TOP_K_PER_CLUSTER=3  # Number of clusters
N_PATHS=3  # Fixed number of exploration paths

# V3.5 new parameters: subgoal and early stopping
ENABLE_EARLY_STOPPING=""  # Default: early stopping disabled (empty string means no flag)
DISABLE_SUBGOAL_PLANNING=""  # Default: subgoal planning enabled (empty string means no flag)

# V3.6 new parameters: Concurrent exploration
# ENABLE_CONCURRENT=""  # Default: concurrent disabled (empty string means no flag)
# To enable concurrent, uncomment the line below:
ENABLE_CONCURRENT="--enable_concurrent"

# Other parameters
SIMILARITY_THRESHOLD=0.8
MAX_ROUNDS=3
EMBEDDING_GPU_ID=2

# Run mode selection
MODE=${1:-"full"}  # full, debug, single, fast

echo "========================================"
echo "Hierarchical QA System V3.6 - Two-Stage Retrieval + Subgoal Planning + Concurrent Exploration"
echo "Improvements: Direct node retrieval first then cluster supplement + LLM top-k selection + Subgoal tracking + Thread-safe concurrency"
echo "Using OpenAI API calling method"
echo "========================================"
echo "Run mode: $MODE"
echo "Graphs directory: $GRAPHS_DIR"
echo "QA data path: $QA_DATA_PATH"
echo "LLM model: $MODEL_NAME"
echo "LLM API address: $API_BASE"
echo "Embedding model: $EMBEDDING_MODEL"
echo "First direct retrieval nodes (top_k_nodes): $TOP_K_NODES"
echo "Nodes per cluster (top_k_per_cluster): $TOP_K_PER_CLUSTER"
echo "Fixed exploration paths (n_paths): $N_PATHS"
echo "Max rounds: $MAX_ROUNDS"
echo "Embedding GPU: $EMBEDDING_GPU_ID"
echo "Subgoal Planning: $([ -z \"$DISABLE_SUBGOAL_PLANNING\" ] && echo \"Enabled\" || echo \"Disabled\")"
echo "Early Stopping: $([ -z \"$ENABLE_EARLY_STOPPING\" ] && echo \"Disabled\" || echo \"Enabled\")"
echo "Concurrent Exploration: $([ -z \"$ENABLE_CONCURRENT\" ] && echo \"Disabled\" || echo \"Enabled\")"
echo "========================================"

case $MODE in
    "debug")
        echo "🔧 Debug mode: Process only 1 item, 2 QA per item"
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
        echo "🔍 Single item mode: Process all QA for 1 item"
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
        echo "🚀 Full mode: Process all items and QA"
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
        echo "🚫 No Refinement mode: Process all items and QA, but disable query refinement"
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
        echo "🔗 No Relation mode: Process all items and QA, but don't use relation information"
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
        echo "⚡ Fast mode: More aggressive parameter configuration"
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
        echo "📊 High recall mode: Increase retrieval node count"
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
        echo "🧵 Concurrent mode: Enable multi-threaded concurrent exploration (recommend 3-5 paths)"
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
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: debug, single, full, no_refinement, no_relation, fast, high_recall, concurrent"
        echo ""
        echo "Mode descriptions:"
        echo "  debug         - Debug mode (1 item, 2 QA)"
        echo "  single        - Single item mode (1 item, all QA)"
        echo "  full          - Full mode (all data)"
        echo "  no_refinement - Disable Refinement"
        echo "  no_relation   - Don't use relation information"
        echo "  fast          - Fast mode (fewer nodes) ⚡"
        echo "  high_recall   - High recall mode (more nodes) 📊"
        echo "  concurrent    - Concurrent mode (multi-threaded exploration) 🧵"
        exit 1
        ;;
esac

echo "========================================"
echo "✅ Run completed!"
echo "========================================"

