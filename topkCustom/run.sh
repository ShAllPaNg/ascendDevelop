#!/bin/bash

# 设置环境变量
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${ASCEND_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${ASCEND_HOME}/python/site-packages:${PYTHONPATH}

# CPU模式设置（用于无NPU环境测试）
export ASCEND_CPU_EXECUTE=1

echo "=========================================="
echo "Running TopK Custom Operator Test"
echo "Mode: CPU Simulation"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_EXEC="${SCRIPT_DIR}/build/tests/topk_test"

if [ ! -f "${TEST_EXEC}" ]; then
    echo "Error: Test executable not found: ${TEST_EXEC}"
    echo "Please run ./build.sh first."
    exit 1
fi

${TEST_EXEC}

echo "=========================================="
echo "Test execution completed"
echo "=========================================="
