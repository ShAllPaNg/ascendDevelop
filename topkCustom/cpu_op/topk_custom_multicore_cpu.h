#ifndef TOPK_CUSTOM_MULTICORE_CPU_H
#define TOPK_CUSTOM_MULTICORE_CPU_H

#include "topk_custom_cpu.h"
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <limits>

/**
 * @brief TopK 多核并行实现（CPU参考版本）
 *
 * 策略：
 * 1. 每个核对自己的数据块进行排序
 * 2. 每个核选出本地 Top K 个元素
 * 3. 主核从所有核的结果中进行 K-way merge，选出全局 TopK
 *
 * 这个方案虽然时间复杂度是 O((N/M)log(N/M) + K*M*log(K*M))，
 * 但由于排序可以高度并行，实际性能可能更好。
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

    // 存储每个核的结果
    struct CoreResult {
        std::vector<std::pair<float, int32_t>> sortedValues;  // (value, index)
    };
    std::vector<CoreResult> coreResults(numCores);

    std::vector<std::thread> threads;

    // 第一阶段：每个核对自己数据进行排序并选出 TopK
    for (int32_t coreId = 0; coreId < numCores; ++coreId) {
        threads.emplace_back([&, coreId]() {
            int32_t start = coreId * blockSize;
            int32_t end = std::min(start + blockSize, N);

            if (start >= N) {
                return;  // 该核没有数据
            }

            int32_t localN = end - start;

            // 收集所有元素的值和索引
            std::vector<std::pair<float, int32_t>> localValues;
            localValues.reserve(localN);

            for (int32_t i = start; i < end; ++i) {
                localValues.push_back({static_cast<float>(input[i]), i});
            }

            // 排序（降序）
            std::sort(localValues.begin(), localValues.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

            // 只保留前 K 个（或全部，如果不足 K 个）
            int32_t keepCount = std::min(K, localN);
            for (int32_t i = 0; i < keepCount; ++i) {
                coreResults[coreId].sortedValues.push_back(localValues[i]);
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 第二阶段：K-way merge 选出全局 TopK
    // 使用最小堆（但由于是降序，我们用最大堆的思路）

    // 收集所有有效的核结果
    std::vector<std::pair<float, int32_t>> allCandidates;
    for (int32_t i = 0; i < numCores; ++i) {
        for (const auto& p : coreResults[i].sortedValues) {
            allCandidates.push_back(p);
        }
    }

    // 如果候选数量不超过 K，直接返回
    if (allCandidates.size() <= static_cast<size_t>(K)) {
        for (size_t i = 0; i < allCandidates.size(); ++i) {
            output[i] = allCandidates[i].second;
        }
        return;
    }

    // 对所有候选进行排序，取前 K 个
    std::sort(allCandidates.begin(), allCandidates.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

    // 复制前 K 个结果
    for (int32_t i = 0; i < K; ++i) {
        output[i] = allCandidates[i].second;
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
