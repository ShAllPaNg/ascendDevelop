#ifndef TOPK_CUSTOM_CPU_H
#define TOPK_CUSTOM_CPU_H

#include <cstdint>
#include <vector>

/**
 * @brief TopK CPU参考实现
 *
 * 用于验证AscendC算子正确性
 */

// half类型定义（用于CPU模拟）
struct half {
    uint16_t data;

    half() = default;
    half(float f);
    operator float() const;
};

/**
 * @brief TopK CPU实现（快速选择算法）
 *
 * @param input 输入数据数组
 * @param output 输出索引数组（TopK对应的索引）
 * @param N 输入数据个数
 * @param K 需要获取的TopK个数
 */
void topk_cpu_half(const half* input, int32_t* output, int32_t N, int32_t K);
void topk_cpu_float(const float* input, int32_t* output, int32_t N, int32_t K);

/**
 * @brief 通用TopK实现（模板）
 */
template<typename T>
void topk_cpu_impl(const T* input, int32_t* output, int32_t N, int32_t K);

/**
 * @brief 验证TopK结果正确性
 *
 * @param input 原始输入数据
 * @param output 输出索引
 * @param N 输入数据个数
 * @param K TopK个数
 * @return true 如果结果正确
 */
bool verify_topk_half(const half* input, const int32_t* output, int32_t N, int32_t K);
bool verify_topk_float(const float* input, const int32_t* output, int32_t N, int32_t K);

#endif // TOPK_CUSTOM_CPU_H
