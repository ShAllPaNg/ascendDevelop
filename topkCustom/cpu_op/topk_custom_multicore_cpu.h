#ifndef TOPK_CUSTOM_MULTICORE_CPU_H
#define TOPK_CUSTOM_MULTICORE_CPU_H

#include "topk_custom_cpu.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>

/**
 * @brief TopK 多核并行实现（CPU参考版本）
 *
 * 策略：
 * 1. 每个核找出本地最大值
 * 2. 主核计算全局第 K 大的阈值
 * 3. 每个核选出所有大于阈值的元素
 * 4. 主核从候选中选出最终 TopK
 */
template<typename T>
void topk_multicore_cpu_impl(
    const T* input,
    int32_t* output,
    int32_t N,
    int32_t K,
    int32_t numCores
) {
    // 特殊情况处理
    if (K >= N) {
        for (int32_t i = 0; i < N; ++i) {
            output[i] = i;
        }
        return;
    }

    // 计算每个核处理的数据块大小
    int32_t blockSize = (N + numCores - 1) / numCores;

    // 存储每个核的本地最大值
    std::vector<T> localMax(numCores);

    std::vector<std::thread> threads;

    // 第一阶段：每个核找出本地最大值
    for (int32_t coreId = 0; coreId < numCores; ++coreId) {
        threads.emplace_back([&, coreId]() {
            int32_t start = coreId * blockSize;
            int32_t end = std::min(start + blockSize, N);

            if (start >= N) {
                localMax[coreId] = input[0];  // 默认值，不会被使用
                return;
            }

            T maxVal = input[start];
            int32_t maxIdx = start;
            for (int32_t i = start + 1; i < end; ++i) {
                if (static_cast<float>(input[i]) > static_cast<float>(maxVal)) {
                    maxVal = input[i];
                    maxIdx = i;
                }
            }
            localMax[coreId] = maxVal;
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    threads.clear();

    // 收集所有核的最大值索引
    std::vector<std::pair<float, int32_t>> maxValues;
    for (int32_t i = 0; i < numCores; ++i) {
        int32_t start = i * blockSize;
        int32_t end = std::min(start + blockSize, N);
        if (start < N) {
            float val = static_cast<float>(localMax[i]);
            maxValues.push_back({val, i});
        }
    }

    // 按最大值降序排序
    std::sort(maxValues.begin(), maxValues.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // 计算全局第 K 大的阈值
    // 前几个核的 TopK 一定能进入全局 TopK
    // 后面的核需要筛选
    int32_t thresholdIdx = -1;
    float threshold = 0;
    int32_t count = 0;

    for (const auto& [val, coreId] : maxValues) {
        int32_t start = coreId * blockSize;
        int32_t end = std::min(start + blockSize, N);
        int32_t localN = end - start;
        int32_t localK = std::min(K, localN);

        if (count + localK <= K) {
            count += localK;
            thresholdIdx = coreId;
        } else {
            break;
        }
    }

    // 计算阈值：从排在前面的核中取第 K 个最大的值
    std::vector<int32_t> tempCandidates;
    for (int32_t i = 0; i <= thresholdIdx && i < static_cast<int32_t>(maxValues.size()); ++i) {
        int32_t coreId = maxValues[i].second;
        int32_t start = coreId * blockSize;
        int32_t end = std::min(start + blockSize, N);

        for (int32_t j = start; j < end; ++j) {
            tempCandidates.push_back(j);
        }
    }

    // 对前几个核的所有元素进行快速选择，找第 K 个
    if (static_cast<int32_t>(tempCandidates.size()) > K) {
        std::vector<int32_t> indices = tempCandidates;
        int32_t left = 0;
        int32_t right = indices.size() - 1;
        int32_t k = K;

        while (left < right) {
            int32_t pivotIdx = indices[(left + right) / 2];
            T pivotVal = input[pivotIdx];

            int32_t i = left - 1;
            for (int32_t j = left; j < right; ++j) {
                int32_t idx = indices[j];
                if (static_cast<float>(input[idx]) >= static_cast<float>(pivotVal)) {
                    i++;
                    std::swap(indices[i], indices[j]);
                }
            }
            std::swap(indices[i + 1], indices[right]);
            int32_t pivotPos = i + 1;

            if (k <= pivotPos) {
                right = pivotPos - 1;
            } else {
                left = pivotPos + 1;
            }
        }

        threshold = static_cast<float>(input[indices[K - 1]]);
    }

    // 第二阶段：每个核选出所有大于阈值的元素
    struct CoreResult {
        std::vector<int32_t> indices;
    };
    std::vector<CoreResult> coreResults(numCores);

    for (int32_t coreId = 0; coreId < numCores; ++coreId) {
        threads.emplace_back([&, coreId]() {
            int32_t start = coreId * blockSize;
            int32_t end = std::min(start + blockSize, N);

            if (start >= N) return;

            // 收集所有大于阈值的元素
            for (int32_t i = start; i < end; ++i) {
                if (static_cast<float>(input[i]) > threshold - 1e-6f) {
                    coreResults[coreId].indices.push_back(i);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 合并所有候选
    std::vector<int32_t> allCandidates;
    for (int32_t i = 0; i < numCores; ++i) {
        for (int32_t idx : coreResults[i].indices) {
            allCandidates.push_back(idx);
        }
    }

    // 从候选中选出最终 TopK
    int32_t totalCandidates = allCandidates.size();
    if (totalCandidates <= K) {
        for (int32_t i = 0; i < totalCandidates; ++i) {
            output[i] = allCandidates[i];
        }
        return;
    }

    // 快速选择
    int32_t left = 0;
    int32_t right = totalCandidates - 1;
    int32_t k = K;

    while (left < right) {
        int32_t pivotIdx = allCandidates[(left + right) / 2];
        T pivotVal = input[pivotIdx];

        int32_t i = left - 1;
        for (int32_t j = left; j < right; ++j) {
            int32_t idx = allCandidates[j];
            if (static_cast<float>(input[idx]) >= static_cast<float>(pivotVal)) {
                i++;
                std::swap(allCandidates[i], allCandidates[j]);
            }
        }
        std::swap(allCandidates[i + 1], allCandidates[right]);
        int32_t pivotPos = i + 1;

        if (k <= pivotPos) {
            right = pivotPos - 1;
        } else {
            left = pivotPos + 1;
        }
    }

    for (int32_t i = 0; i < K; ++i) {
        output[i] = allCandidates[i];
    }
}

// 接口函数
inline void topk_multicore_cpu_float(const float* input, int32_t* output, int32_t N, int32_t K, int32_t numCores = 4) {
    topk_multicore_cpu_impl(input, output, N, K, numCores);
}

inline void topk_multicore_cpu_half(const half* input, int32_t* output, int32_t N, int32_t K, int32_t numCores = 4) {
    topk_multicore_cpu_impl(input, output, N, K, numCores);
}

#endif // TOPK_CUSTOM_MULTICORE_CPU_H
