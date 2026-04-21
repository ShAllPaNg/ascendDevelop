cmake_minimum_required(VERSION 3.14)
project(topk_custom CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN路径
set(ASCEND_CANN_DIR $ENV{ASCEND_AICPU_PATH})
if(NOT ASCEND_CANN_DIR)
    set(ASCEND_CANN_DIR /home/yzc/Ascend/cann-8.5.0)
endif()

# 包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_op
    ${ASCEND_CANN_DIR}/x86_64-linux/include
    ${ASCEND_CANN_DIR}/include
)

# 链接路径
link_directories(
    ${ASCEND_CANN_DIR}/x86_64-linux/lib64
)

# 源文件（CPU模式只需要CPU参考实现）
set(SRC_FILES
    cpu_op/topk_custom_cpu.cpp
)

# 可执行文件（纯CPU测试，不依赖AscendC）
add_executable(topk_test tests/topk_test.cpp ${SRC_FILES})

# AscendC 算子（NPU模式，使用 AscendC 编译器单独编译）
# add_library(topk_custom_obj OBJECT src/topk_custom.cpp)

# 链接库（纯CPU测试不需要NPU库）
target_link_libraries(topk_test
    pthread
    m
    dl
)

# 编译选项
target_compile_options(topk_test PRIVATE
    -g
    -O2
    -fPIC
)
