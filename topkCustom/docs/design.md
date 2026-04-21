# TopK Custom 算子设计方案

## 1. 算子概述

### 1.1 功能描述
实现一个 TopK 算子，从输入数据中找出最大的 K 个元素对应的索引（indice），不要求对结果进行排序。

### 1.2 主要特性
- **算法选择**: 基于快速选择（QuickSelect）算法，平均时间复杂度 O(n)
- **返回内容**: 返回 TopK 元素的索引（indice），而非数值本身
- **精度支持**: 支持 half(float16) 和 float32
- **执行模式**: 支持 CPU 模式（用于测试）和 NPU 模式

---

## 2. 接口设计

### 2.1 算子原型
```cpp
template<typename T>
__global__ __aicore__ inline void TopKCustom(
    LocalTensor<T> inputTensor,      // 输入数据 [N]
    LocalTensor<int32_t> outputTensor, // 输出索引 [K]
    int32_t N,                        // 输入数据个数
    int32_t K,                        // 需要获取的TopK个数
    uint32_t reductionAxis = 0        // reduction维度（预留）
);
```

### 2.2 输入输出参数

| 参数名 | 类型 | 方向 | 形状 | 描述 |
|--------|------|------|------|------|
| inputTensor | LocalTensor\<T\> | 输入 | [N] | 输入数据张量 |
| outputTensor | LocalTensor\<int32_t\> | 输出 | [K] | 输出TopK对应的索引 |
| N | int32_t | 标量 | - | 输入数据元素个数 |
| K | int32_t | 标量 | - | 要获取的最大元素个数 |

### 2.3 支持的数据类型
- `half` (float16)
- `float`

---

## 3. 算法设计

### 3.1 快速选择算法（QuickSelect）

```
快速选择是快速排序的变种，用于在未排序数组中找到第K大/小的元素。

核心思想：
1. 选择一个 pivot（基准元素）
2. 将数组分为三部分：大于 pivot、等于 pivot、小于 pivot
3. 根据 K 所在的分区决定下一步处理哪个分区
4. 递归处理，直到找到 TopK 元素

平均时间复杂度: O(n)
最坏时间复杂度: O(n²) - 可通过随机 pivot 优化
```

### 3.2 算法流程

```
输入: input[N], K
输出: indices[K] (TopK的索引，不排序)

算法步骤:
1. 初始化索引数组 idx[0..N-1] = {0, 1, 2, ..., N-1}
2. 使用快速选择在 idx 上操作，比较 input[idx[i]]:
   
   function quickselect(input, idx, left, right, k):
       if left == right: return
       
       pivot_idx = partition(input, idx, left, right)
       
       if k <= pivot_idx:
           quickselect(input, idx, left, pivot_idx - 1, k)
       else:
           quickselect(input, idx, pivot_idx + 1, right, k)

3. partition 函数: 按 input[idx[i]] 降序分区

4. 最终 idx[0..K-1] 即为 TopK 的索引
```

### 3.3 分区策略（降序）

```
分区目标：大的元素在左边，小的元素在右边

partition(input, idx, left, right):
    pivot = input[idx[right]]
    i = left - 1
    
    for j from left to right-1:
        if input[idx[j]] >= pivot:  // 降序，大的放左边
            i++
            swap(idx[i], idx[j])
    
    swap(idx[i+1], idx[right])
    return i+1
```

---

## 4. AscendC 算子实现设计

### 4.1 核心类结构

```cpp
template<typename T>
class TopKKernel {
public:
    __aicore__ inline TopKKernel() {}
    
    __aicore__ inline void process(
        LocalTensor<T>& input,
        LocalTensor<int32_t>& output,
        int32_t N, 
        int32_t K
    );

private:
    // 快速选择主函数
    __aicore__ inline void quickselect(
        LocalTensor<T>& input,
        LocalTensor<int32_t>& indices,
        int32_t left,
        int32_t right,
        int32_t k
    );
    
    // 分区函数
    __aicore__ inline int32_t partition(
        LocalTensor<T>& input,
        LocalTensor<int32_t>& indices,
        int32_t left,
        int32_t right
    );
    
    // 交换索引
    __aicore__ inline void swap(
        LocalTensor<int32_t>& indices,
        int32_t i,
        int32_t j
    );
    
    // 初始化索引数组
    __aicore__ inline void initIndices(
        LocalTensor<int32_t>& indices,
        int32_t N
    );
};
```

### 4.2 内存使用规划

| 缓冲区 | 大小 | 用途 |
|--------|------|------|
| input | N 个元素 | 输入数据（通过 GlobalTensor → LocalTensor 搬运） |
| output | K 个元素 | 输出 TopK 索引 |
| indices | N 个元素 | 工作数组，存储索引 |

### 4.3 数据流

```
GM (Global Memory)          UB (Unified Buffer)
┌─────────────┐            ┌─────────────┐
│ input[N]    │───Copy────>│ input[N]    │
└─────────────┘            └─────────────┘
                                  │
                                  ▼
                           QuickSelect 算法
                           在 indices 上操作
                                  │
                                  ▼
┌─────────────┐            ┌─────────────┐
│ output[K]   │<───Copy────│ indices[0:K]│
└─────────────┘            └─────────────┘
```

---

## 5. 项目结构

```
topkCustom/
├── CMakeLists.txt              # CMake 构建配置
├── build.sh                    # 编译脚本
├── run.sh                      # 运行脚本
├── docs/
│   └── design.md               # 本设计方案文档
├── include/
│   └── topk_custom.h           # 算子头文件
├── src/
│   └── topk_custom.cpp         # 算子实现
├── cpu_op/
│   ├── topk_custom_cpu.cpp     # CPU参考实现
│   └── topk_custom_cpu.h       # CPU参考头文件
└── tests/
    └── topk_test.cpp           # 测试代码
```

---

## 6. CPU 模式测试方案

### 6.1 测试配置
- 使用 `ASCEND_CPU_EXECUTE` 环境变量启用 CPU 模式
- 不需要实际的 NPU 硬件

### 6.2 测试用例

| 用例 | N | K | 描述 |
|------|---|---|------|
| 基本用例 | 16 | 4 | 正常情况 |
| 边界情况 | 8 | 8 | K = N |
| 边界情况 | 8 | 1 | K = 1 |
| 大数据量 | 1024 | 10 | 较大数组 |
| 重复元素 | 16 | 4 | 包含重复值 |
| 负数处理 | 16 | 4 | 包含负数 |

### 6.3 验证方法
```cpp
// CPU 参考实现
void topk_cpu_ref(const float* input, int32_t* output, int N, int K);

// 对比 AscendC 结果和 CPU 参考结果
bool verify_results(int32_t* npu_result, int32_t* cpu_result, int K);
```

---

## 7. 编译和运行流程

### 7.1 编译流程
```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 编译
./build.sh

# 生成可执行文件: build/tests/topk_test
```

### 7.2 运行流程
```bash
# CPU 模式运行（无 NPU 环境）
export ASCEND_CPU_EXECUTE=1
./build/tests/topk_test

# NPU 模式运行（有 NPU 环境，预留）
unset ASCEND_CPU_EXECUTE
./build/tests/topk_test
```

---

## 8. 待确认事项

- [ ] K 的最大值限制（建议不超过 2048）
- [ ] N 的最大值限制（根据 UB 大小确定）
- [ ] 是否需要支持批量处理（batch 维度）
- [ ] 精度要求：half 和 float 是否都需要支持
- [ ] 升腾 CANN 版本号（用于确定 API）

---

## 9. 参考资料

- AscendC 开发指南
- 快速选择算法 (QuickSelect)
- Ascend CANN API 文档
