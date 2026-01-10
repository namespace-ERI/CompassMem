#!/bin/bash
# Start vLLM Service - Qwen3-8B

# Model path
MODEL_PATH="./models/Qwen3-8B"

# Automatically create .env file (if not exists)
if [ ! -f ".env" ]; then
    echo "Creating .env configuration file..."
    cat > .env << 'EOF'
VLLM_BASE_URL=http://localhost:8006/v1
VLLM_API_KEY=EMPTY
MODEL=Qwen3-8B
LLM_TEMPERATURE=0.6
LLM_TOP_P=0.95
LLM_MAX_TOKENS=32768
EOF
    echo "✓ .env file created"
fi

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Parse GPU parameters
# Supported: ./start_vllm.sh --gpus 0,1,2 or ./start_vllm.sh 0,1,2 or CUDA_VISIBLE_DEVICES=0,1,2 ./start_vllm.sh
GPU_IDS=""
if [ "$1" = "--gpus" ] && [ -n "$2" ]; then
    GPU_IDS="$2"
elif [ -n "$1" ] && [[ "$1" =~ ^[0-9,]+$ ]]; then
    GPU_IDS="$1"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_IDS="$CUDA_VISIBLE_DEVICES"
fi

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$GPU_IDS" ]; then
        # Use specified GPUs
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo "Using specified GPUs: $GPU_IDS (total: $GPU_COUNT)"
    else
        # Use all GPUs
        GPU_COUNT=$TOTAL_GPU_COUNT
        if [ $GPU_COUNT -gt 1 ]; then
            echo "Detected $GPU_COUNT GPUs, enabling multi-GPU mode"
        else
            echo "Detected $GPU_COUNT GPU, using single GPU mode"
        fi
    fi
else
    GPU_COUNT=1
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        echo "Using specified GPUs: $GPU_IDS (total: $GPU_COUNT)"
    else
        echo "No GPU detected, using single GPU mode"
    fi
fi

# Start vLLM
echo "Starting vLLM service..."
echo "Model: $MODEL_PATH"
echo "GPU count: $GPU_COUNT"
if [ $GPU_COUNT -gt 1 ]; then
    echo "Multi-GPU mode: Tensor Parallel Size = $GPU_COUNT"
fi
if [ -n "$GPU_IDS" ]; then
    echo "Using GPU IDs: $GPU_IDS"
fi
echo ""

vllm serve "$MODEL_PATH" \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --served-model-name "Qwen3-8B" \
    --port 8006 \
    --tensor-parallel-size $GPU_COUNT \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --enable-prefix-caching

