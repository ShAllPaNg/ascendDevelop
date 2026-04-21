#!/bin/bash

# 设置环境变量
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${ASCEND_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${ASCEND_HOME}/python/site-packages:${PYTHONPATH}

# 创建构建目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p ${BUILD_DIR}

# 进入构建目录
cd ${BUILD_DIR}

# 执行CMake配置和编译
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++

make -j$(nproc)

echo "Build complete. Executable: ${BUILD_DIR}/tests/topk_test"
