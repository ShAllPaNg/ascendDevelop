#include "topk_custom_cpu.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>

// half实现
half::half(float f) {
    // 简化的float到half转换
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));

    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    uint16_t half_exponent, half_mantissa;

    if (exponent == 255) {
        // Inf or NaN
        half_exponent = 31;
        half_mantissa = mantissa ? 0x200 : 0;
    } else if (exponent >= 127 - 15 + 31) {
        // Overflow - infinity
        half_exponent = 31;
        half_mantissa = 0;
    } else if (exponent <= 127 - 15) {
        // Underflow - zero
        half_exponent = 0;
        half_mantissa = 0;
    } else {
        half_exponent = exponent - 127 + 15;
        half_mantissa = mantissa >> 13;
    }

    data = (sign << 15) | (half_exponent << 10) | half_mantissa;
}

half::operator float() const {
    uint16_t half_bits = data;
    uint32_t sign = (half_bits >> 15) & 0x1;
    uint32_t exponent = (half_bits >> 10) & 0x1F;
    uint32_t mantissa = half_bits & 0x3FF;

    uint32_t float_exponent, float_mantissa;

    if (exponent == 31) {
        float_exponent = 255;
        float_mantissa = mantissa ? mantissa << 13 : 0;
    } else if (exponent == 0) {
        float_exponent = mantissa ? 127 - 15 - 1 : 0;
        float_mantissa = mantissa << 13;
        // Normalize subnormal
        while (!(float_mantissa & 0x400000) && float_exponent > 1) {
            float_mantissa <<= 1;
            float_exponent--;
        }
    } else {
        float_exponent = exponent + 127 - 15;
        float_mantissa = mantissa << 13;
    }

    uint32_t bits = (sign << 31) | (float_exponent << 23) | float_mantissa;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// 比较函数（用于索引排序，降序）
template<typename T>
static bool compare_desc(const T* input, int32_t i, int32_t j) {
    return static_cast<float>(input[i]) > static_cast<float>(input[j]);
}

// 快速选择实现
template<typename T>
static void quickselect_impl(const T* input, int32_t* indices, int32_t left, int32_t right, int32_t k) {
    while (left < right) {
        // 选择最右侧元素作为pivot
        int32_t pivot_idx = right;
        float pivot_val = static_cast<float>(input[indices[pivot_idx]]);

        // 分区
        int32_t i = left - 1;
        for (int32_t j = left; j < right; j++) {
            if (static_cast<float>(input[indices[j]]) >= pivot_val) {
                i++;
                std::swap(indices[i], indices[j]);
            }
        }
        std::swap(indices[i + 1], indices[right]);
        pivot_idx = i + 1;

        // 判断下一步处理哪个分区
        if (k <= pivot_idx) {
            right = pivot_idx - 1;
        } else {
            left = pivot_idx + 1;
        }
    }
}

template<typename T>
void topk_cpu_impl(const T* input, int32_t* output, int32_t N, int32_t K) {
    if (N <= 0 || K <= 0 || K > N) {
        return;
    }

    // 初始化索引数组
    std::vector<int32_t> indices(N);
    for (int32_t i = 0; i < N; i++) {
        indices[i] = i;
    }

    // 快速选择找到TopK
    quickselect_impl(input, indices.data(), 0, N - 1, K);

    // 复制前K个索引到输出
    // 注意：quickselect后前K个元素是TopK，但可能不是完全排序的
    for (int32_t i = 0; i < K; i++) {
        output[i] = indices[i];
    }
}

void topk_cpu_half(const half* input, int32_t* output, int32_t N, int32_t K) {
    topk_cpu_impl(input, output, N, K);
}

void topk_cpu_float(const float* input, int32_t* output, int32_t N, int32_t K) {
    topk_cpu_impl(input, output, N, K);
}

// 验证函数
template<typename T>
bool verify_topk_impl(const T* input, const int32_t* output, int32_t N, int32_t K) {
    if (K <= 0 || K > N) {
        return false;
    }

    // 检查输出索引是否有效
    for (int32_t i = 0; i < K; i++) {
        if (output[i] < 0 || output[i] >= N) {
            return false;
        }
    }

    // 检查是否有重复索引
    for (int32_t i = 0; i < K; i++) {
        for (int32_t j = i + 1; j < K; j++) {
            if (output[i] == output[j]) {
                return false;
            }
        }
    }

    // 检查输出的K个索引对应的值是否都是最大的
    // 找到输出中对应的最小值
    float min_in_output = static_cast<float>(input[output[0]]);
    for (int32_t i = 1; i < K; i++) {
        float val = static_cast<float>(input[output[i]]);
        if (val < min_in_output) {
            min_in_output = val;
        }
    }

    // 检查未选中的元素是否都小于等于min_in_output
    for (int32_t i = 0; i < N; i++) {
        // 检查i是否在output中
        bool in_output = false;
        for (int32_t j = 0; j < K; j++) {
            if (output[j] == i) {
                in_output = true;
                break;
            }
        }

        if (!in_output) {
            if (static_cast<float>(input[i]) > min_in_output + 1e-6f) {
                return false;
            }
        }
    }

    return true;
}

bool verify_topk_half(const half* input, const int32_t* output, int32_t N, int32_t K) {
    return verify_topk_impl(input, output, N, K);
}

bool verify_topk_float(const float* input, const int32_t* output, int32_t N, int32_t K) {
    return verify_topk_impl(input, output, N, K);
}
