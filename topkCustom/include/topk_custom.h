#ifndef TOPK_CUSTOM_H
#define TOPK_CUSTOM_H

#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_tensor.h"
#include "kernel_launch_config.h"
#include "kernel_operator.h"
#include "kernel_scalar_ops.h"
#include "kernel_cmp_ops.h"
#include "kernel_select_op.h"

namespace acl {

// ============================================================
// 单核版本 - 使用 AscendC 高效接口
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
    /**
     * @brief 快速选择算法主函数（使用向量化接口）
     */
    __aicore__ inline void quickselect(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& indices,
        LocalTensor<T>& valueBuffer,
        LocalTensor<uint8_t>& maskBuffer,
        int32_t left,
        int32_t right,
        int32_t K
    );

    /**
     * @brief 向量化分区函数
     * @return pivot 的最终位置
     */
    __aicore__ inline int32_t partition(
        LocalTensor<T>& inputTensor,
        LocalTensor<int32_t>& indices,
        LocalTensor<T>& valueBuffer,
        LocalTensor<uint8_t>& maskBuffer,
        LocalTensor<T>& tempBuffer,
        int32_t left,
        int32_t right
    );

    /**
     * @brief 初始化索引数组
     */
    __aicore__ inline void initIndices(
        LocalTensor<int32_t>& indices,
        int32_t N
    );
};

// ============================================================
// Kernel 入口函数
// ============================================================

extern "C" __global__ __aicore__ void topk_custom_half(
    LocalTensor<half>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<half>& tempTensor,
    int32_t N,
    int32_t K
);

extern "C" __global__ __aicore__ void topk_custom_float(
    LocalTensor<float>& inputTensor,
    LocalTensor<int32_t>& outputTensor,
    LocalTensor<float>& tempTensor,
    int32_t N,
    int32_t K
);

} // namespace acl

#endif // TOPK_CUSTOM_H
