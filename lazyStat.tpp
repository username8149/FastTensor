#pragma once

#include "pch.tpp"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <queue>
#include <future>
#include <condition_variable>

// =====================
// Simple ThreadPool
// =====================
struct ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool stop = false;

    ThreadPool(size_t n_threads = std::thread::hardware_concurrency()) {
        if (n_threads == 0) n_threads = 4;
        for (size_t i = 0; i < n_threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->cv.wait(lock, [this]() { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto &w : workers) w.join();
    }
};

// =====================
// LazyEvalStat
// =====================
template<typename T, typename PvType>
struct LazyEvalStat {
    using StatFunc = std::function<T(const PvType&)>;

    std::vector<StatFunc> batch;
    ThreadPool pool;

    LazyEvalStat(size_t threads = std::thread::hardware_concurrency()) : pool(threads) {}

    void add(StatFunc f) {
        batch.push_back(f);
    }

    // Helper functions
    void add_sum() {
        add([](const PvType& core) {
            T result = 0;
            const auto& data = core.data;
            size_t N = data.size();

#ifdef __ARM_NEON
            if constexpr (std::is_same<T,float>::value) {
                size_t i=0;
                float32x4_t acc = vdupq_n_f32(0.f);
                for (; i+4<=N; i+=4) {
                    float32x4_t v = vld1q_f32(&data[i]);
                    acc = vaddq_f32(acc, v);
                }
                float temp[4]; vst1q_f32(temp, acc);
                result += temp[0]+temp[1]+temp[2]+temp[3];
                for (; i<N; i++) result += data[i];
                return result;
            }
#  if defined(__aarch64__)
            else if constexpr (std::is_same<T,double>::value) {
                size_t i=0;
                float64x2_t acc = vdupq_n_f64(0.0);
                for (; i+2<=N; i+=2) {
                    float64x2_t v = vld1q_f64(&data[i]);
                    acc = vaddq_f64(acc, v);
                }
                double temp[2]; vst1q_f64(temp, acc);
                result += temp[0]+temp[1];
                for (; i<N; i++) result += data[i];
                return result;
            }
#  endif
#endif
            // fallback scalar
            for (auto v : data) result += v;
            return result;
        });
    }

    // Execute all stats
    std::vector<T> execute(const PvType& core, bool clear_after=true) {
        std::vector<std::future<void>> futures;
        std::vector<T> results(batch.size());

        for (size_t i = 0; i < batch.size(); ++i) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                results[i] = batch[i](core);
            }));
        }

        for (auto &f : futures) f.wait();

        if (clear_after) batch.clear();
        return results;
    }
};
