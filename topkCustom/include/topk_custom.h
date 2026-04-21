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
    int32_t pivotIndex;     // pivot 的全局位置
    int32_t pivotValueIdx;  // pivot 对应的索引值
    int32_t leftCount;      // 左侧（大于pivot）的元素总数
    int32_t rightCount;     // 右侧（小于等于pivot）的元素总数
    int32_t currentK;       // 当前需要找的第 K 大
    int32_t iteration;      // 迭代次数（用于同步）
    int32_t done;           // 完成标志
};

/**
 * @brief TopK Custom算子实现（多核并行版本）
 *
 * 从输入数据中找出最大的K个元素对应的索引
 * 使用并行快速选择(Parallel QuickSelect)算法
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
     * @brief 并行快速选择算法主函数
     */
    __aicore__ inline void parallelQuickSelect(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& outputTensor,
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData
    );

    /**
     * @brief 并行分区函数（多核协作）
     * @return 当前核的分区内，大于 pivot 的元素数量
     */
    __aicore__ inline int32_t parallelPartition(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& indices,
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData,
        int32_t pivotIdx,
        int32_t globalLeft,
        int32_t globalRight
    );

    /**
     * @brief 前缀和计算（计算全局 pivot 位置）
     */
    __aicore__ inline int32_t computePrefixSum(
        LocalTensor<int32_t>& sharedTensor,
        const TopKTilingData& tilingData,
        int32_t localCount
    );

    /**
     * @brief 交换索引数组中的两个元素
     */
    __aicore__ inline void swap(
        LocalTensor<int32_t>& indices,
        int32_t i,
        int32_t j
    );

    /**
     * @brief 初始化索引数组为 [0, 1, 2, ..., N-1]
     */
    __aicore__ inline void initIndices(
        LocalTensor<int32_t>& indices,
        int32_t N
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
