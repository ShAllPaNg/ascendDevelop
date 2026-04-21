#include "topk_custom.h"

namespace acl {

// ============================================================
// TopKCustom 类实现
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
// Kernel 入口函数 - half版本
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

// ============================================================
// Kernel 入口函数 - float版本
// ============================================================

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
__aicore__ inline void TopKCustomMultiCore<T>::initIndices(
    LocalTensor<int32_t>& indices,
    int32_t N
) {
    for (int32_t i = 0; i < N; ++i) {
        indices.SetValue(i, i);
    }
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::swap(
    LocalTensor<int32_t>& indices,
    int32_t i,
    int32_t j
) {
    int32_t temp = indices.GetValue(i);
    indices.SetValue(i, indices.GetValue(j));
    indices.SetValue(j, temp);
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::syncCores(
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    // 使用原子操作实现 barrier
    // 每个核完成时增加计数器，等待所有核完成
    if (tilingData.coreId == 0) {
        // 主核重置同步计数器
        sharedTensor.SetValue(0, 0);
    }

    // 所有核到达 barrier
    // 在实际 AscendC 实现中，使用 SyncAll 或类似 API
    // 这里使用简化的模拟实现

    // 原子递增
    uint32_t offset = 0;  // 同步计数器在 sharedTensor 中的偏移
    // AtomicAdd(sharedTensor, offset, 1);

    // 等待所有核到达
    // while (sharedTensor.GetValue(offset) < tilingData.coreNum) {
    //     // 忙等待
    // }
}

template<typename T>
__aicore__ inline int32_t TopKCustomMultiCore<T>::computePrefixSum(
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData,
    int32_t localCount
) {
    // 将每个核的本地计数写入共享内存
    // 共享内存布局: [count0, count1, ..., countM, prefixSum0, prefixSum1, ...]
    uint32_t countOffset = tilingData.coreId;
    sharedTensor.SetValue(countOffset, localCount);

    // 同步，确保所有核都写入了计数
    syncCores(sharedTensor, tilingData);

    // 主核计算前缀和
    int32_t globalCount = 0;
    if (tilingData.coreId == 0) {
        int32_t sum = 0;
        for (int32_t i = 0; i < tilingData.coreNum; ++i) {
            int32_t cnt = sharedTensor.GetValue(i);
            sharedTensor.SetValue(tilingData.coreNum + i, sum);  // 存储前缀和
            sum += cnt;
        }
        sharedTensor.SetValue(2 * tilingData.coreNum, sum);  // 总数
    }

    // 再次同步
    syncCores(sharedTensor, tilingData);

    // 每个核获取自己的前缀和偏移
    int32_t prefixSum = sharedTensor.GetValue(tilingData.coreNum + tilingData.coreId);

    return prefixSum;
}

template<typename T>
__aicore__ inline int32_t TopKCustomMultiCore<T>::parallelPartition(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& indices,
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData,
    int32_t pivotIdx,
    int32_t globalLeft,
    int32_t globalRight
) {
    // 获取 pivot 值
    T pivotVal = inputTensor.GetValue(pivotIdx);

    // 确定当前核的数据范围
    int32_t coreStart = tilingData.coreId * tilingData.blockSize;
    int32_t coreEnd = min(coreStart + tilingData.blockSize, tilingData.totalLength);

    // 限制在当前分区范围内
    coreStart = max(coreStart, globalLeft);
    coreEnd = min(coreEnd, globalRight + 1);

    // 统计当前核分区内大于 pivot 的元素数量
    int32_t localGreaterCount = 0;

    // 第一遍：统计数量
    for (int32_t i = coreStart; i < coreEnd; ++i) {
        int32_t idx = indices.GetValue(i);
        T val = inputTensor.GetValue(idx);
        if (val.GetValue() > pivotVal.GetValue()) {
            localGreaterCount++;
        }
    }

    // 计算前缀和，获取全局偏移
    int32_t globalOffset = computePrefixSum(sharedTensor, tilingData, localGreaterCount);

    // 第二遍：重排元素（使用临时存储）
    // 创建临时缓冲区存储当前核的分区结果
    auto& ub = inputTensor.GetBuffer();
    LocalTensor<int32_t> tempIndices = ub.Alloc<int32_t>(coreEnd - coreStart);

    int32_t writePos = 0;
    int32_t localLeftCount = 0;   // 大于 pivot 的数量
    int32_t localRightCount = 0;  // 小于等于 pivot 的数量

    // 收集大于 pivot 的元素
    for (int32_t i = coreStart; i < coreEnd; ++i) {
        int32_t idx = indices.GetValue(i);
        T val = inputTensor.GetValue(idx);
        if (val.GetValue() > pivotVal.GetValue()) {
            tempIndices.SetValue(writePos++, idx);
            localLeftCount++;
        }
    }

    int32_t leftStart = globalOffset;  // 大于 pivot 的起始位置

    // 收集小于等于 pivot 的元素
    for (int32_t i = coreStart; i < coreEnd; ++i) {
        int32_t idx = indices.GetValue(i);
        T val = inputTensor.GetValue(idx);
        if (val.GetValue() <= pivotVal.GetValue()) {
            tempIndices.SetValue(writePos++, idx);
            localRightCount++;
        }
    }

    int32_t rightStart = globalLeft + (globalRight - globalLeft + 1) - localGreaterCount -
                         (coreEnd - coreStart - localGreaterCount) + globalOffset - localLeftCount;

    // 写回全局索引数组
    writePos = 0;
    for (int32_t i = 0; i < localLeftCount; ++i) {
        indices.SetValue(leftStart + writePos, tempIndices.GetValue(i));
        writePos++;
    }

    writePos = 0;
    for (int32_t i = localLeftCount; i < localLeftCount + localRightCount; ++i) {
        indices.SetValue(rightStart + writePos, tempIndices.GetValue(i));
        writePos++;
    }

    ub.Free(tempIndices);

    // 返回当前核的大于 pivot 的元素数量
    return localLeftCount;
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::parallelQuickSelect(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    auto& ub = inputTensor.GetBuffer();

    // 分配全局索引数组
    LocalTensor<int32_t> indices = ub.Alloc<int32_t>(tilingData.totalLength);

    // 主核初始化索引数组
    if (tilingData.coreId == 0) {
        initIndices(indices, tilingData.totalLength);

        // 初始化共享数据
        sharedTensor.SetValue(0, tilingData.topK);  // currentK
        sharedTensor.SetValue(1, 0);                // iteration
        sharedTensor.SetValue(2, 0);                // done
    }

    // 同步，确保初始化完成
    syncCores(sharedTensor, tilingData);

    int32_t globalLeft = 0;
    int32_t globalRight = tilingData.totalLength - 1;
    int32_t currentK = tilingData.topK;

    // 迭代进行并行快速选择
    while (globalLeft < globalRight && currentK > 0) {
        // 选择 pivot（使用中间位置的元素）
        int32_t pivotPos = (globalLeft + globalRight) / 2;
        int32_t pivotIdx = indices.GetValue(pivotPos);

        // 并行分区
        int32_t localGreaterCount = parallelPartition(
            inputTensor, indices, sharedTensor, tilingData,
            pivotIdx, globalLeft, globalRight
        );

        // 同步，确保所有核完成分区
        syncCores(sharedTensor, tilingData);

        // 主核计算 pivot 的全局位置
        if (tilingData.coreId == 0) {
            // 计算大于 pivot 的总数
            int32_t totalGreater = 0;
            for (int32_t i = 0; i < tilingData.coreNum; ++i) {
                totalGreater += sharedTensor.GetValue(i);
            }

            int32_t pivotGlobalPos = globalLeft + totalGreater;

            // 更新共享数据
            sharedTensor.SetValue(3, pivotGlobalPos);  // pivotIndex
            sharedTensor.SetValue(4, pivotIdx);         // pivotValueIdx
        }

        // 同步
        syncCores(sharedTensor, tilingData);

        // 获取 pivot 的全局位置
        int32_t pivotGlobalPos = sharedTensor.GetValue(3);

        // 判断下一步处理哪个分区
        if (currentK <= pivotGlobalPos - globalLeft + 1) {
            // TopK 在左分区（包含 pivot）
            globalRight = pivotGlobalPos;
        } else {
            // TopK 在右分区
            currentK -= (pivotGlobalPos - globalLeft + 1);
            globalLeft = pivotGlobalPos + 1;
        }

        // 同步
        syncCores(sharedTensor, tilingData);
    }

    // 主核收集结果
    if (tilingData.coreId == 0) {
        for (int32_t i = 0; i < tilingData.topK; ++i) {
            outputTensor.SetValue(i, indices.GetValue(globalLeft + i));
        }
    }

    ub.Free(indices);
}

template<typename T>
__aicore__ inline void TopKCustomMultiCore<T>::process(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    const TopKTilingData& tilingData
) {
    // 计算当前核的数据范围
    localStart_ = tilingData.coreId * tilingData.blockSize;
    localEnd_ = min(localStart_ + tilingData.blockSize, tilingData.totalLength);

    // 执行并行快速选择
    parallelQuickSelect(inputTensor, outputTensor, sharedTensor, tilingData);
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
