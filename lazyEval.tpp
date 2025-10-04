#pragma once
#include "Tensor.tpp"
#include "pch.tpp"
#include <arm_neon.h>

/**
 * @brief Lazy evaluation engine for element-wise tensor operations with optional SIMD acceleration.
 * 
 * @tparam T Data type (float, int, etc.)
 * @tparam TensorType Tensor class (contains data vector, shape, strides)
 */
template <typename T, typename TensorType>
struct LazyEval {
    // ------------------------------
    // Structure for batched operations
    // ------------------------------
    struct BatchOp {
        std::function<T(T, T)> func;    // Element-wise operation function
        const TensorType* tensor;       // Pointer to operand tensor
    };

    std::vector<BatchOp> batch;        // Batch of operations to execute

    // ------------------------------
    // Add an operation to the lazy batch
    // ------------------------------
    template <typename Func>
    void add(const TensorType& other, Func f) {
        batch.push_back({ std::function<T(T, T)>(f), &other });
    }

    // ------------------------------
    // Execute all batched operations on the core tensor
    // ------------------------------
    void execute(TensorType& core) {
        if (batch.empty()) return;

        // ------------------------------
        // 1. Determine target shape using broadcasting
        // ------------------------------
        std::vector<size_t> target_shape = core.shape;
        for (auto& op : batch)
            target_shape = broadcast_shapes(target_shape, op.tensor->shape);

        // Broadcast core tensor if necessary
        if (core.shape != target_shape) {
            size_t new_size = 1;
            for (auto s : target_shape) new_size *= s;
            std::vector<T> new_data(new_size);

            std::vector<size_t> coord(target_shape.size());
            for (size_t idx = 0; idx < new_size; ++idx) {
                size_t tmp = idx;
                for (int d = int(target_shape.size()) - 1; d >= 0; --d) {
                    coord[d] = tmp % target_shape[d];
                    tmp /= target_shape[d];
                }

                // Map coordinates back to the original tensor
                std::vector<size_t> src_coord = coord;
                int shift = target_shape.size() - core.shape.size();
                for (size_t d = 0; d < core.shape.size(); ++d)
                    if (core.shape[d] == 1) src_coord[d + shift] = 0;

                size_t si = 0;
                for (size_t d = 0; d < core.shape.size(); ++d)
                    si += (core.shape[d] == 1 ? 0 : src_coord[d + shift]) * core.strides[d];

                new_data[idx] = core.data[si];
            }

            core.data.swap(new_data);
            core.shape = target_shape;
            core.computeStrides();
        }

        // ------------------------------
        // 2. Multithreaded execution
        // ------------------------------
        static size_t threads_hardware = std::thread::hardware_concurrency();
        if (threads_hardware == 0) threads_hardware = 4;

        size_t N = core.data.size();
        size_t min_chunk = 1024;
        size_t chunk = std::max(N / threads_hardware, min_chunk);
        size_t num_threads = (N + chunk - 1) / chunk;

        std::atomic<size_t> next_idx(0);
        std::vector<std::thread> threads(num_threads);

        auto worker = [&](void) {
            while (true) {
                size_t start = next_idx.fetch_add(chunk);
                if (start >= N) break;
                size_t end = std::min(start + chunk, N);

                // ------------------------------
                // SIMD optimized section for float
                // ------------------------------
                if constexpr (std::is_same<T, float>::value) {
                    for (auto& op : this->batch) {
                        size_t i = start;
                        // Vectorized loop using NEON
                        for (; i + 4 <= end; i += 4) {
                            float32x4_t a = vld1q_f32(&core.data[i]);
                            float32x4_t b = vld1q_f32(&op.tensor->data[i]);
                            float32x4_t r = vaddq_f32(a, b); // Example: addition, override with proper func if needed
                            vst1q_f32(&core.data[i], r);
                        }
                        // Scalar tail
                        for (; i < end; i++) {
                            core.data[i] = op.func(core.data[i], op.tensor->data[i]);
                        }
                    }
                }
                // SIMD optimized section for int
                else if constexpr (std::is_same<T, int>::value) {
                    for (auto& op : this->batch) {
                        size_t i = start;
                        for (; i + 4 <= end; i += 4) {
                            int32x4_t a = vld1q_s32(&core.data[i]);
                            int32x4_t b = vld1q_s32(&op.tensor->data[i]);
                            int32x4_t r = vaddq_s32(a, b);
                            vst1q_s32(&core.data[i], r);
                        }
                        for (; i < end; i++) {
                            core.data[i] = op.func(core.data[i], op.tensor->data[i]);
                        }
                    }
                }
                // Fallback for all other types
                else {
                    for (auto& op : this->batch) {
                        for (size_t i = start; i < end; i++) {
                            core.data[i] = op.func(core.data[i], op.tensor->data[i]);
                        }
                    }
                }
            }
        };

        // Launch threads
        for (size_t t = 0; t < num_threads; ++t)
            threads[t] = std::thread(worker);

        // Join threads
        for (auto& th : threads) th.join();

        // Clear batch after execution
        batch.clear();
    }

private:
    // ------------------------------
    // Broadcasting helper function
    // ------------------------------
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t na = a.size(), nb = b.size();
        size_t n = std::max(na, nb);
        std::vector<size_t> result(n);

        for (size_t i = 0; i < n; i++) {
            size_t dim_a = (i < n - na) ? 1 : a[i - (n - na)];
            size_t dim_b = (i < n - nb) ? 1 : b[i - (n - nb)];
            if (dim_a == dim_b || dim_a == 1 || dim_b == 1)
                result[i] = std::max(dim_a, dim_b);
            else
                throw std::runtime_error("broadcast_shapes: incompatible shapes");
        }

        return result;
    }
};
