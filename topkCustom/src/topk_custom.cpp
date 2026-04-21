#include "topk_custom.h"

namespace acl {

// ============================================================
// TopKCustom 类实现（单核版本）
// ============================================================

template<typename T>
__aicore__ inline void TopKCustom<T>::initIndices(
    LocalTensor<int32_t>& indices,
    int32_t N
) {
    for (int32_t i = 0; i < N; ++i) {
        indices.SetValue(i, i);
    }
}

template<typename T>
__aicore__ inline void TopKCustom<T>::swap(
    LocalTensor<int32_t>& indices,
    int32_t i,
    int32_t j
) {
    int32_t temp = indices.GetValue(i);
    indices.SetValue(i, indices.GetValue(j));
    indices.SetValue(j, temp);
}

template<typename T>
__aicore__ inline int32_t TopKCustom<T>::partition(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& indices,
    int32_t left,
    int32_t right
) {
    // 使用最右侧元素作为pivot
    int32_t pivot_idx = indices.GetValue(right);
    T pivot_val = inputTensor.GetValue(pivot_idx);

    int32_t i = left - 1;

    for (int32_t j = left; j < right; ++j) {
        int32_t idx_j = indices.GetValue(j);
        T val_j = inputTensor.GetValue(idx_j);

        // 降序：大的放左边
        if (val_j.GetValue() >= pivot_val.GetValue()) {
            i++;
            swap(indices, i, j);
        }
    }

    swap(indices, i + 1, right);
    return i + 1;
}

template<typename T>
__aicore__ inline void TopKCustom<T>::quickselect(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& indices,
    int32_t left,
    int32_t right,
    int32_t k
) {
    while (left < right) {
        int32_t pivot_pos = partition(inputTensor, indices, left, right);

        if (k <= pivot_pos) {
            right = pivot_pos - 1;
        } else {
            left = pivot_pos + 1;
        }
    }
}

template<typename T>
__aicore__ inline void TopKCustom<T>::process(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    int32_t N,
    int32_t K
) {
    // 获取tensor buffer
    auto& ub = inputTensor.GetBuffer();

    // 分配索引工作数组
    LocalTensor<int32_t> indices;
    indices = ub.Alloc<int32_t>(N);

    // 初始化索引数组 [0, 1, 2, ..., N-1]
    initIndices(indices, N);

    // 执行快速选择算法，找到TopK
    quickselect(inputTensor, indices, 0, N - 1, K);

    // 复制前K个索引到输出
    for (int32_t i = 0; i < K; ++i) {
        outputTensor.SetValue(i, indices.GetValue(i));
    }

    // 释放工作数组
    ub.Free(indices);
}

// ============================================================
// Kernel 入口函数 - 单核版本
// ============================================================

extern "C" __global__ __aicore__ void topk_custom_half(
    LocalTensor<half>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<half>& tempTensor,
    int32_t N,
    int32_t K
) {
    TopKCustom<half> op;
    op.process(inputTensor, outputTensor, N, K);
}

extern "C" __global__ __aicore__ void topk_custom_float(
    LocalTensor<float>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<float>& tempTensor,
    int32_t N,
    int32_t K
) {
    TopKCustom<float> op;
    op.process(inputTensor, outputTensor, N, K);
}

// ============================================================
// TopKCustomMultiCore 类实现（多核并行版本）
// ============================================================

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::swap(
    LocalTensor<int32_t>& indices,
    LocalTensor<T>& values,
    int32_t i,
    int32_t j
) {
    int32_t tempIdx = indices.GetValue(i);
    indices.SetValue(i, indices.GetValue(j));
    indices.SetValue(j, tempIdx);

    T tempVal = values.GetValue(i);
    values.SetValue(i, values.GetValue(j));
    values.SetValue(j, tempVal);
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::localSort(
    LocalTensor<int32_t>& indices,
    LocalTensor<T>& values,
    int32_t n
) {
    // 使用冒泡排序（简单实现，适合小规模数据）
    for (int32_t i = 0; i < n - 1; ++i) {
        for (int32_t j = 0; j < n - i - 1; ++j) {
            T val1 = values.GetValue(j);
            T val2 = values.GetValue(j + 1);
            if (val1.GetValue() < val2.GetValue()) {
                swap(indices, values, j, j + 1);
            }
        }
    }
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::syncCores(
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    // AscendC 中的 barrier 同步
    // 注意：实际使用时需要使用 AscendC 提供的同步 API
    // 这里是简化实现

    // 原子递增计数器
    // AtomicAdd(sharedTensor, 1, 1);

    // 等待所有核到达
    // while (sharedTensor.GetValue(1) < tilingData.coreNum) {
    //     // 忙等待
    // }

    // 最后到达的核重置计数器
    // if (tilingData.coreId == tilingData.coreNum - 1) {
    //     sharedTensor.SetValue(1, 0);
    //     sharedTensor.SetValue(2, sharedTensor.GetValue(2) + 1);
    // }
}

template<typename T>
__aicore__ inline int32_t TopKCustomMultiCore<T>::localSortAndSelect(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    // 计算当前核的数据范围
    int32_t start = tilingData.coreId * tilingData.blockSize;
    int32_t end = start + tilingData.blockSize;
    if (end > tilingData.totalLength) {
        end = tilingData.totalLength;
    }

    if (start >= tilingData.totalLength) {
        return 0;  // 该核没有数据
    }

    int32_t localN = end - start;
    int32_t localK = tilingData.topK;
    if (localK > localN) {
        localK = localN;
    }

    auto& ub = inputTensor.GetBuffer();

    // 分配本地工作空间
    LocalTensor<int32_t> localIndices = ub.Alloc<int32_t>(localN);
    LocalTensor<T> localValues = ub.Alloc<T>(localN);

    // 收集数据和索引
    for (int32_t i = 0; i < localN; ++i) {
        localIndices.SetValue(i, start + i);
        localValues.SetValue(i, inputTensor.GetValue(start + i));
    }

    // 排序（降序）
    localSort(localIndices, localValues, localN);

    // 将前 localK 个结果写入共享内存
    int32_t offset = tilingData.coreId * tilingData.topK * 2;  // 每个元素占2个位置（索引+值）

    for (int32_t i = 0; i < localK; ++i) {
        sharedTensor.SetValue(offset + i * 2, localIndices.GetValue(i));  // 索引
        sharedTensor.SetValue(offset + i * 2 + 1, (int32_t)localValues.GetValue(i).GetValue());  // 值
    }

    ub.Free(localIndices);
    ub.Free(localValues);

    return localK;
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::process(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    // 第一阶段：每个核对自己数据进行排序并选出 TopK
    localSortAndSelect(inputTensor, outputTensor, sharedTensor, tilingData);

    // 同步，确保所有核完成
    syncCores(sharedTensor, tilingData);

    // 第二阶段：主核从所有核的结果中选出全局 TopK
    if (tilingData.coreId == 0) {
        auto& ub = inputTensor.GetBuffer();

        int32_t totalCandidates = tilingData.coreNum * tilingData.topK;

        // 临时存储所有候选
        LocalTensor<int32_t> allIndices = ub.Alloc<int32_t>(totalCandidates);
        LocalTensor<T> allValues = ub.Alloc<T>(totalCandidates);

        int32_t count = 0;
        for (int32_t core = 0; core < tilingData.coreNum; ++core) {
            int32_t offset = core * tilingData.topK * 2;
            for (int32_t i = 0; i < tilingData.topK; ++i) {
                int32_t idx = sharedTensor.GetValue(offset + i * 2);
                int32_t val = sharedTensor.GetValue(offset + i * 2 + 1);

                allIndices.SetValue(count, idx);
                allValues.SetValue(count, T((float)val));
                count++;
            }
        }

        // 对所有候选进行排序（降序）
        localSort(allIndices, allValues, totalCandidates);

        // 复制前 K 个结果到输出
        for (int32_t i = 0; i < tilingData.topK; ++i) {
            outputTensor.SetValue(i, allIndices.GetValue(i));
        }

        ub.Free(allIndices);
        ub.Free(allValues);
    }
}

// ============================================================
// Kernel 入口函数 - 多核版本
// ============================================================

extern "C" __global__ __aicore__ void topk_custom_multicore_half(
    LocalTensor<half>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    uint32_t tilingData
) {
    // 将 tilingData 转换为 TopKTilingData 结构
    TopKTilingData* tiling = reinterpret_cast<TopKTilingData*>(tilingData);

    TopKCustomMultiCore<half> op;
    op.process(inputTensor, outputTensor, sharedTensor, *tiling);
}

extern "C" __global__ __aicore__ void topk_custom_multicore_float(
    LocalTensor<float>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    uint32_t tilingData
) {
    // 将 tilingData 转换为 TopKTilingData 结构
    TopKTilingData* tiling = reinterpret_cast<TopKTilingData*>(tilingData);

    TopKCustomMultiCore<float> op;
    op.process(inputTensor, outputTensor, sharedTensor, *tiling);
}

} // namespace acl
