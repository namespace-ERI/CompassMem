#!/bin/bash
# 启动 vLLM 服务 - Qwen3-8B

# 模型路径
MODEL_PATH="/share/project/zyt/hyy/Model/Qwen3-8B"

# 自动创建 .env 文件（如果不存在）
if [ ! -f ".env" ]; then
    echo "创建 .env 配置文件..."
    cat > .env << 'EOF'
VLLM_BASE_URL=http://localhost:8006/v1
VLLM_API_KEY=EMPTY
MODEL=Qwen3-8B
LLM_TEMPERATURE=0.6
LLM_TOP_P=0.95
LLM_MAX_TOKENS=32768
EOF
    echo "✓ .env 文件已创建"
fi

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 解析GPU参数
# 支持: ./start_vllm.sh --gpus 0,1,2 或 ./start_vllm.sh 0,1,2 或 CUDA_VISIBLE_DEVICES=0,1,2 ./start_vllm.sh
GPU_IDS=""
if [ "$1" = "--gpus" ] && [ -n "$2" ]; then
    GPU_IDS="$2"
elif [ -n "$1" ] && [[ "$1" =~ ^[0-9,]+$ ]]; then
    GPU_IDS="$1"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_IDS="$CUDA_VISIBLE_DEVICES"
fi

# 检测GPU数量
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$GPU_IDS" ]; then
        # 使用指定的GPU
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo "指定使用GPU: $GPU_IDS (共 $GPU_COUNT 个)"
    else
        # 使用所有GPU
        GPU_COUNT=$TOTAL_GPU_COUNT
        if [ $GPU_COUNT -gt 1 ]; then
            echo "检测到 $GPU_COUNT 个GPU，启用多卡运行"
        else
            echo "检测到 $GPU_COUNT 个GPU，使用单GPU模式"
        fi
    fi
else
    GPU_COUNT=1
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        echo "指定使用GPU: $GPU_IDS (共 $GPU_COUNT 个)"
    else
        echo "未检测到GPU，使用单GPU模式"
    fi
fi

# 启动 vLLM
echo "启动 vLLM 服务..."
echo "模型: $MODEL_PATH"
echo "GPU数量: $GPU_COUNT"
if [ $GPU_COUNT -gt 1 ]; then
    echo "多卡模式: Tensor Parallel Size = $GPU_COUNT"
fi
if [ -n "$GPU_IDS" ]; then
    echo "使用的GPU ID: $GPU_IDS"
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

