#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <functional>
#include <type_traits>

#include "../cpu_op/topk_custom_cpu.h"
#include "../cpu_op/topk_custom_multicore_cpu.h"

using namespace std;

// ============================================================
// 测试辅助类
// ============================================================

class TopKTester {
public:
    struct TestConfig {
        int32_t N;
        int32_t K;
        string description;
    };

    struct TestCase {
        vector<float> float_input;
        vector<half> half_input;
        vector<int32_t> output;
        vector<int32_t> cpu_output;
        int32_t N;
        int32_t K;
        string name;
    };

    // 生成测试数据
    static vector<float> generateRandomData(int32_t N, float min_val = -100.0f, float max_val = 100.0f) {
        vector<float> data(N);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dis(min_val, max_val);

        for (int32_t i = 0; i < N; ++i) {
            data[i] = dis(gen);
        }
        return data;
    }

    // float转half
    static vector<half> floatToHalf(const vector<float>& input) {
        vector<half> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = half(input[i]);
        }
        return output;
    }

    // 打印数组
    static void printArray(const char* label, const vector<float>& data, int32_t max_items = 20) {
        cout << label << ": [";
        int32_t n = min((int32_t)data.size(), max_items);
        for (int32_t i = 0; i < n; ++i) {
            cout << fixed << setprecision(2) << data[i];
            if (i < n - 1) cout << ", ";
        }
        if (data.size() > (size_t)max_items) {
            cout << ", ... (" << data.size() << " total)";
        }
        cout << "]" << endl;
    }

    static void printArray(const char* label, const vector<int32_t>& data, int32_t max_items = 20) {
        cout << label << ": [";
        int32_t n = min((int32_t)data.size(), max_items);
        for (int32_t i = 0; i < n; ++i) {
            cout << data[i];
            if (i < n - 1) cout << ", ";
        }
        if (data.size() > (size_t)max_items) {
            cout << ", ... (" << data.size() << " total)";
        }
        cout << "]" << endl;
    }

    // 验证结果
    static bool verifyResults(
        const vector<float>& input,
        const vector<int32_t>& output,
        int32_t N,
        int32_t K
    ) {
        // 检查索引有效性
        for (int32_t i = 0; i < K; ++i) {
            if (output[i] < 0 || output[i] >= N) {
                cout << "  [FAIL] Invalid index at position " << i << ": " << output[i] << endl;
                return false;
            }
        }

        // 检查重复索引
        for (int32_t i = 0; i < K; ++i) {
            for (int32_t j = i + 1; j < K; ++j) {
                if (output[i] == output[j]) {
                    cout << "  [FAIL] Duplicate index: " << output[i] << endl;
                    return false;
                }
            }
        }

        // 检查输出的K个索引对应的值是否都是最大的
        float min_in_output = input[output[0]];
        for (int32_t i = 1; i < K; ++i) {
            min_in_output = min(min_in_output, input[output[i]]);
        }

        for (int32_t i = 0; i < N; ++i) {
            bool in_output = false;
            for (int32_t j = 0; j < K; ++j) {
                if (output[j] == i) {
                    in_output = true;
                    break;
                }
            }

            if (!in_output && input[i] > min_in_output + 1e-5f) {
                cout << "  [FAIL] Element at index " << i << " (" << input[i]
                     << ") is larger than min in output (" << min_in_output << ")" << endl;
                return false;
            }
        }

        return true;
    }

    // 比较两个结果是否一致
    static bool compareResults(
        const vector<int32_t>& result1,
        const vector<int32_t>& result2,
        const vector<float>& input,
        int32_t K
    ) {
        // 提取两个结果的值
        vector<float> values1(K), values2(K);
        for (int32_t i = 0; i < K; ++i) {
            values1[i] = input[result1[i]];
            values2[i] = input[result2[i]];
        }

        // 排序后比较
        sort(values1.begin(), values1.end(), greater<float>());
        sort(values2.begin(), values2.end(), greater<float>());

        for (int32_t i = 0; i < K; ++i) {
            if (abs(values1[i] - values2[i]) > 1e-5f) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================
// AscendC 算子调用接口（CPU模式模拟）
// ============================================================

// float版本
inline void topk_ascendc_cpu(const vector<float>& input, vector<int32_t>& output, int32_t N, int32_t K) {
    topk_cpu_float(input.data(), output.data(), N, K);
}

// half版本
inline void topk_ascendc_cpu(const vector<half>& input, vector<int32_t>& output, int32_t N, int32_t K) {
    topk_cpu_half(input.data(), output.data(), N, K);
}

// ============================================================
// 测试用例
// ============================================================

bool runBasicTestFloat() {
    cout << "\n=== Test: Basic Float Test ===" << endl;

    vector<float> input = {3.5f, 1.2f, 8.7f, 4.3f, 9.1f, 2.8f, 7.4f, 5.6f};
    int32_t N = input.size();
    int32_t K = 4;

    TopKTester::printArray("Input", input);

    vector<int32_t> output(K);
    vector<int32_t> cpu_output(K);

    // 调用算子
    topk_ascendc_cpu(input, output, N, K);
    topk_cpu_float(input.data(), cpu_output.data(), N, K);

    TopKTester::printArray("Output", output);
    TopKTester::printArray("CPU Output", cpu_output);

    bool passed = TopKTester::verifyResults(input, output, N, K);
    bool matched = TopKTester::compareResults(output, cpu_output, input, K);

    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;
    cout << "Match CPU: " << (matched ? "PASS" : "FAIL") << endl;

    // 打印TopK对应的值
    cout << "TopK values: ";
    for (int32_t i = 0; i < K; ++i) {
        cout << input[output[i]] << " ";
    }
    cout << endl;

    return passed;
}

bool runBasicTestHalf() {
    cout << "\n=== Test: Basic Half Test ===" << endl;

    vector<float> float_input = {3.5f, 1.2f, 8.7f, 4.3f, 9.1f, 2.8f, 7.4f, 5.6f};
    vector<half> input = TopKTester::floatToHalf(float_input);
    int32_t N = input.size();
    int32_t K = 4;

    TopKTester::printArray("Input (float view)", float_input);

    vector<int32_t> output(K);
    vector<int32_t> cpu_output(K);

    // 调用算子
    topk_ascendc_cpu(input, output, N, K);
    topk_cpu_half(input.data(), cpu_output.data(), N, K);

    TopKTester::printArray("Output", output);
    TopKTester::printArray("CPU Output", cpu_output);

    bool passed = TopKTester::verifyResults(float_input, output, N, K);
    bool matched = TopKTester::compareResults(output, cpu_output, float_input, K);

    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;
    cout << "Match CPU: " << (matched ? "PASS" : "FAIL") << endl;

    return passed;
}

bool runEdgeTest_K_Equals_N() {
    cout << "\n=== Test: Edge Case K = N ===" << endl;

    int32_t N = 8;
    int32_t K = N;
    auto input = TopKTester::generateRandomData(N, 0.0f, 100.0f);

    TopKTester::printArray("Input", input);

    vector<int32_t> output(K);
    topk_ascendc_cpu(input, output, N, K);

    TopKTester::printArray("Output", output);

    bool passed = TopKTester::verifyResults(input, output, N, K);
    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;

    return passed;
}

bool runEdgeTest_K_Equals_1() {
    cout << "\n=== Test: Edge Case K = 1 ===" << endl;

    vector<float> input = {3.5f, 1.2f, 8.7f, 4.3f, 9.1f, 2.8f, 7.4f, 5.6f};
    int32_t N = input.size();
    int32_t K = 1;

    TopKTester::printArray("Input", input);

    vector<int32_t> output(K);
    topk_ascendc_cpu(input, output, N, K);

    cout << "Output: [" << output[0] << "]" << endl;
    cout << "Value: " << input[output[0]] << endl;

    // 找到真实最大值
    int32_t true_max_idx = 0;
    for (int32_t i = 1; i < N; ++i) {
        if (input[i] > input[true_max_idx]) {
            true_max_idx = i;
        }
    }

    bool passed = (output[0] == true_max_idx);
    cout << "Verify: " << (passed ? "PASS" : "FAIL")
         << " (expected: " << true_max_idx << ")" << endl;

    return passed;
}

bool runDuplicateTest() {
    cout << "\n=== Test: Duplicate Values ===" << endl;

    vector<float> input = {5.0f, 3.0f, 5.0f, 2.0f, 5.0f, 1.0f, 3.0f, 4.0f};
    int32_t N = input.size();
    int32_t K = 5;

    TopKTester::printArray("Input", input);

    vector<int32_t> output(K);
    topk_ascendc_cpu(input, output, N, K);

    TopKTester::printArray("Output", output);

    bool passed = TopKTester::verifyResults(input, output, N, K);
    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;

    cout << "TopK values: ";
    for (int32_t i = 0; i < K; ++i) {
        cout << input[output[i]] << " ";
    }
    cout << endl;

    return passed;
}

bool runNegativeTest() {
    cout << "\n=== Test: Negative Values ===" << endl;

    vector<float> input = {-3.5f, 1.2f, -8.7f, 4.3f, -9.1f, 2.8f, -7.4f, 5.6f};
    int32_t N = input.size();
    int32_t K = 4;

    TopKTester::printArray("Input", input);

    vector<int32_t> output(K);
    topk_ascendc_cpu(input, output, N, K);

    TopKTester::printArray("Output", output);

    bool passed = TopKTester::verifyResults(input, output, N, K);
    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;

    cout << "TopK values: ";
    for (int32_t i = 0; i < K; ++i) {
        cout << input[output[i]] << " ";
    }
    cout << endl;

    return passed;
}

bool runLargeDataTest() {
    cout << "\n=== Test: Large Data (N=1024, K=10) ===" << endl;

    int32_t N = 1024;
    int32_t K = 10;
    auto input = TopKTester::generateRandomData(N, -1000.0f, 1000.0f);

    vector<int32_t> output(K);
    auto start = chrono::high_resolution_clock::now();
    topk_ascendc_cpu(input, output, N, K);
    auto end = chrono::high_resolution_clock::now();

    bool passed = TopKTester::verifyResults(input, output, N, K);

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Time: " << duration.count() << " us" << endl;
    cout << "Verify: " << (passed ? "PASS" : "FAIL") << endl;

    cout << "TopK values: ";
    for (int32_t i = 0; i < K; ++i) {
        cout << fixed << setprecision(2) << input[output[i]] << " ";
    }
    cout << endl;

    return passed;
}

// ============================================================
// 多核测试用例
// ============================================================

bool runMultiCoreTest() {
    cout << "\n=== Test: Multi-Core (4 cores) ===" << endl;

    int32_t N = 1024;
    int32_t K = 10;
    int32_t numCores = 4;
    auto input = TopKTester::generateRandomData(N, -1000.0f, 1000.0f);

    vector<int32_t> singleCoreOutput(K);
    vector<int32_t> multiCoreOutput(K);

    // 单核版本
    auto start1 = chrono::high_resolution_clock::now();
    topk_cpu_float(input.data(), singleCoreOutput.data(), N, K);
    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);

    // 多核版本
    auto start2 = chrono::high_resolution_clock::now();
    topk_multicore_cpu_float(input.data(), multiCoreOutput.data(), N, K, numCores);
    auto end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);

    cout << "Single-core time: " << duration1.count() << " us" << endl;
    cout << "Multi-core time:  " << duration2.count() << " us" << endl;
    cout << "Speedup:         " << fixed << setprecision(2)
         << (float)duration1.count() / duration2.count() << "x" << endl;

    // 验证多核结果正确性
    bool passed = TopKTester::verifyResults(input, multiCoreOutput, N, K);

    // 比较单核和多核结果
    bool matched = TopKTester::compareResults(singleCoreOutput, multiCoreOutput, input, K);

    cout << "Verify:   " << (passed ? "PASS" : "FAIL") << endl;
    cout << "Match single-core: " << (matched ? "PASS" : "FAIL") << endl;

    return passed && matched;
}

bool runMultiCoreCompareTest() {
    cout << "\n=== Test: Multi-Core Performance Comparison ===" << endl;

    int32_t N = 4096;
    int32_t K = 100;
    auto input = TopKTester::generateRandomData(N, -1000.0f, 1000.0f);

    struct PerfResult {
        int32_t cores;
        int64_t timeUs;
        float speedup;
        bool correct;
    };

    vector<PerfResult> results;

    // 单核基准
    {
        vector<int32_t> output(K);
        auto start = chrono::high_resolution_clock::now();
        topk_cpu_float(input.data(), output.data(), N, K);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        bool correct = TopKTester::verifyResults(input, output, N, K);
        results.push_back({1, duration.count(), 1.0f, correct});
    }

    // 多核测试
    for (int32_t cores : {2, 4, 8}) {
        vector<int32_t> output(K);
        auto start = chrono::high_resolution_clock::now();
        topk_multicore_cpu_float(input.data(), output.data(), N, K, cores);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        bool correct = TopKTester::verifyResults(input, output, N, K);
        float speedup = (float)results[0].timeUs / duration.count();
        results.push_back({cores, duration.count(), speedup, correct});
    }

    cout << "\nPerformance Results:" << endl;
    cout << "Cores | Time (us) | Speedup | Correct" << endl;
    cout << "------|-----------|---------|--------" << endl;
    for (const auto& r : results) {
        cout << setw(5) << r.cores << " | "
             << setw(9) << r.timeUs << " | "
             << setw(7) << fixed << setprecision(2) << r.speedup << " | "
             << (r.correct ? "YES" : "NO") << endl;
    }

    // 检查所有结果是否正确
    bool allCorrect = std::all_of(results.begin(), results.end(),
        [](const PerfResult& r) { return r.correct; });

    return allCorrect;
}

// ============================================================
// 主函数
// ============================================================

int main() {
    cout << "========================================" << endl;
    cout << "TopK Custom Operator Test" << endl;
    cout << "Mode: CPU Simulation" << endl;
    cout << "========================================" << endl;

    int passed = 0;
    int total = 0;

    struct Test {
        string name;
        function<bool()> func;
    };

    vector<Test> tests = {
        {"Basic Float Test", runBasicTestFloat},
        {"Basic Half Test", runBasicTestHalf},
        {"Edge Case K = N", runEdgeTest_K_Equals_N},
        {"Edge Case K = 1", runEdgeTest_K_Equals_1},
        {"Duplicate Values", runDuplicateTest},
        {"Negative Values", runNegativeTest},
        {"Large Data Test", runLargeDataTest},
        {"Multi-Core Test", runMultiCoreTest},
        {"Multi-Core Performance", runMultiCoreCompareTest},
    };

    for (const auto& test : tests) {
        total++;
        if (test.func()) {
            passed++;
            cout << "[" << test.name << "] PASSED" << endl;
        } else {
            cout << "[" << test.name << "] FAILED" << endl;
        }
    }

    cout << "\n========================================" << endl;
    cout << "Test Summary: " << passed << "/" << total << " passed" << endl;
    cout << "========================================" << endl;

    return (passed == total) ? 0 : 1;
}
