#include "topk_custom.h"

namespace acl {

// ============================================================
// TopKCustom 类实现（使用 AscendC 高效接口）
// ============================================================

template<typename T>
__aicore__ inline void TopKCustom<T>::initIndices(
    LocalTensor<int32_t>& indices,
    int32_t N
) {
    // 使用 Vector 接口初始化索引
    for (int32_t i = 0; i < N; ++i) {
        indices.SetValue(i, i);
    }
}

template<typename T>
__aicore__ inline int32_t TopKCustom<T>::partition(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& indices,
    LocalTensor<T>& valueBuffer,
    LocalTensor<uint8_t>& maskBuffer,
    LocalTensor<T>& tempBuffer,
    int32_t left,
    int32_t right
) {
    // 使用最右侧元素作为 pivot
    int32_t pivot_idx = indices.GetValue(right);
    T pivot_val = inputTensor.GetValue(pivot_idx);

    int32_t partition_size = right - left + 1;

    // 步骤1: 使用 CompareScalar 向量化比较
    // 将 input[left:right] 与 pivot 比较，大于等于 pivot 的为 1
    LocalTensor<T> src = inputTensor[left];
    CompareScalar(
        maskBuffer,                           // 输出 mask (uint8_t)
        src,                                   // 源数据
        pivot_val,                            // pivot scalar
        CMPMODE::GE,                          // 大于等于（降序分区）
        partition_size                        // 元素个数
    );

    // 步骤2: 使用 Select 根据 mask 分离大于等于和小于 pivot 的元素
    LocalTensor<T> greaterTemp = valueBuffer;      // 重用缓冲区存储大于等于 pivot 的值
    LocalTensor<T> lessTemp = valueBuffer[partition_size];  // 存储小于 pivot 的值
    LocalTensor<int32_t> greaterIdx = tempBuffer;   // 转换为索引类型
    LocalTensor<int32_t> lessIdx = greaterIdx[partition_size];

    // 创建全1和全0的 mask 用于 Select
    LocalTensor<uint8_t> maskOnes;
    LocalTensor<uint8_t> maskZeros;
    maskOnes = maskBuffer[partition_size];
    maskZeros = maskOnes[partition_size];

    // 简单初始化（实际应该用 Dup 接口）
    for (int32_t i = 0; i < partition_size; ++i) {
        maskOnes.SetValue(i, 0xFF);
        maskZeros.SetValue(i, 0);
    }

    // 收集大于等于 pivot 的元素（使用 mask 作为选择器）
    // 模式1: 根据 mask 在 tensor 和 scalar 之间选择
    LocalTensor<T> originalSrc = src;
    T zeroScalar = static_cast<T>(0);

    Select(
        greaterTemp,                          // 目标
        maskBuffer,                            // 选择 mask
        originalSrc,                           // 源0 (mask=1时选择)
        zeroScalar,                            // 源1 (mask=0时选择，这里用0占位)
        SELMODE::VSEL_TENSOR_SCALAR_MODE,       // 模式1
        partition_size
    );

    // 收集索引（同样的方式）
    Select(
        greaterIdx,                            // 目标
        maskBuffer,                            // 选择 mask
        indices[left],                          // 源0
        static_cast<int32_t>(-1),              // 源1 (mask=0时选择-1占位)
        SELMODE::VSEL_TENSOR_SCALAR_MODE,
        partition_size
    );

    // 计算 mask 中 1 的个数（大于等于 pivot 的元素个数）
    int32_t greater_count = 0;
    for (int32_t i = 0; i < partition_size; ++i) {
        uint8_t mask_val = maskBuffer.GetValue(i);
        if (mask_val != 0) {
            greater_count++;
        }
    }

    // 收集小于 pivot 的元素（反向 mask）
    LocalTensor<uint8_t> invertedMask = maskBuffer[partition_size * 2];
    for (int32_t i = 0; i < partition_size; ++i) {
        invertedMask.SetValue(i, ~maskBuffer.GetValue(i));
    }

    Select(
        lessTemp,
        invertedMask,
        originalSrc,
        zeroScalar,
        SELMODE::VSEL_TENSOR_SCALAR_MODE,
        partition_size
    );

    Select(
        lessIdx,
        invertedMask,
        indices[left],
        static_cast<int32_t>(-1),
        SELMODE::VSEL_TENSOR_SCALAR_MODE,
        partition_size
    );

    // 步骤3: 写回结果
    // 大于等于 pivot 的元素放左边
    for (int32_t i = 0; i < greater_count; ++i) {
        indices.SetValue(left + i, greaterIdx.GetValue(i));
    }

    // 小于 pivot 的元素放右边
    for (int32_t i = 0; i < partition_size - greater_count; ++i) {
        if (lessIdx.GetValue(i) != -1) {
            indices.SetValue(left + greater_count + i, lessIdx.GetValue(i));
        }
    }

    return left + greater_count - 1;  // pivot 的最终位置
}

template<typename T>
__aicore__ inline void TopKCustom<T>::quickselect(
    LocalTensor<T>& inputTensor,
    LocalTensor<int32_t>& indices,
    LocalTensor<T>& valueBuffer,
    LocalTensor<uint8_t>& maskBuffer,
    int32_t left,
    int32_t right,
    int32_t K
) {
    while (left < right) {
        int32_t pivot_pos = partition(
            inputTensor, indices, valueBuffer, maskBuffer,
            valueBuffer,  // tempBuffer 重用
            left, right
        );

        if (K <= pivot_pos) {
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
    auto& ub = inputTensor.GetBuffer();

    // 分配工作数组
    LocalTensor<int32_t> indices = ub.Alloc<int32_t>(N);
    LocalTensor<T> valueBuffer = ub.Alloc<T>(N * 3);  // 分配足够的空间
    LocalTensor<uint8_t> maskBuffer = ub.Alloc<uint8_t>(N * 2);

    // 初始化索引数组
    initIndices(indices, N);

    // 执行快速选择算法
    quickselect(inputTensor, indices, valueBuffer, maskBuffer, 0, N - 1, K);

    // 复制前 K 个索引到输出
    for (int32_t i = 0; i < K; ++i) {
        outputTensor.SetValue(i, indices.GetValue(i));
    }

    // 释放工作数组
    ub.Free(indices);
    ub.Free(valueBuffer);
    ub.Free(maskBuffer);
}

// ============================================================
// Kernel 入口函数
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

} // namespace acl
