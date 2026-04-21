#ifndef TOPK_CUSTOM_H
#define TOPK_CUSTOM_H

#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_tensor.h"
#include "kernel_launch_config.h"

namespace acl {

// ============================================================
// 多核 Tiling 结构体
// ============================================================

struct TopKTilingData {
    int32_t totalLength;    // 总数据长度 N
    int32_t topK;           // 需要获取的 TopK 个数
    int32_t blockSize;      // 每个核处理的数据块大小
    int32_t coreNum;        // 使用的核心数
    int32_t coreId;         // 当前核 ID
};

// ============================================================
// 全局共享数据（用于核间通信）
// ============================================================

struct TopKSharedData {
    int32_t coreResultsOffset;  // 每个核结果的起始偏移
    int32_t barrierCounter;     // barrier 同步计数器
    int32_t phase;              // barrier 阶段
};

/**
 * @brief TopK Custom算子实现（多核并行版本）
 *
 * 从输入数据中找出最大的K个元素对应的索引
 * 策略：每个核对自己的数据排序选出TopK，主核合并结果
 *
 * @tparam T 数据类型 (half 或 float)
 */
template<typename T>
class TopKCustomMultiCore {
public:
    __aicore__ inline TopKCustomMultiCore() = default;

    __aicore__ inline ~TopKCustomMultiCore() = default;

    /**
     * @brief 执行TopK操作（多核版本）
     * @param inputTensor 输入数据张量
     * @param outputTensor 输出索引张量
     * @param sharedTensor 共享数据张量（用于核间通信）
     * @param tilingData Tiling 数据
     */
    __aicore__ inline void process(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& outputTensor,
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData
    );

private:
    /**
     * @brief 本地排序并选出TopK
     * @return 返回选出的元素数量
     */
    __aicore__ inline int32_t localSortAndSelect(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& outputTensor,
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData
    );

    /**
     * @brief 交换两个元素
     */
    __aicore__ inline void swap(
        LocalTensor<int32_t>& indices,
        LocalTensor<T>& values,
        int32_t i,
        int32_t j
    );

    /**
     * @brief 本地排序（降序）
     */
    __aicore__ inline void localSort(
        LocalTensor<int32_t>& indices,
        LocalTensor<T>& values,
        int32_t n
    );

    /**
     * @brief 核间同步（barrier）
     */
    __aicore__ inline void syncCores(
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData
    );

    // 每个核的数据范围
    int32_t localStart_;
    int32_t localEnd_;
};

// ============================================================
// 单核版本（兼容旧接口）
// ============================================================

template<typename T>
class TopKCustom {
public:
    __aicore__ inline TopKCustom() = default;

    __aicore__ inline ~TopKCustom() = default;

    __aicore__ inline void process(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& outputTensor,
        int32_t N,
        int32_t K
    );

private:
    __aicore__ inline void quickselect(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& indices,
        int32_t left,
        int32_t right,
        int32_t k
    );

    __aicore__ inline int32_t partition(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& indices,
        int32_t left,
        int32_t right
    );

    __aicore__ inline void swap(
        LocalTensor<int32_t>& indices,
        int32_t i,
        int32_t j
    );

    __aicore__ inline void initIndices(
        LocalTensor<int32_t>& indices,
        int32_t N
    );
};

// ============================================================
// Kernel 入口函数
// ============================================================

// 多核版本 - half
extern "C" __global__ __aicore__ void topk_custom_multicore_half(
    LocalTensor<half>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    uint32_t tilingData
);

// 多核版本 - float
extern "C" __global__ __aicore__ void topk_custom_multicore_float(
    LocalTensor<float>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<int32_t>& sharedTensor,
    uint32_t tilingData
);

// 单核版本 - half（兼容）
extern "C" __global__ __aicore__ void topk_custom_half(
    LocalTensor<half>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<half>& tempTensor,
    int32_t N,
    int32_t K
);

// 单核版本 - float（兼容）
extern "C" __global__ __aicore__ void topk_custom_float(
    LocalTensor<float>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<float>& tempTensor,
    int32_t N,
    int32_t K
);

} // namespace acl

#endif // TOPK_CUSTOM_H
