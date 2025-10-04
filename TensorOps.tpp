#pragma once
#include "LazyStat.tpp"
#include "pch.tpp"
#include "Tensor.tpp"

template<typename T>
struct Ops {
    using PvT = Pv<T>;

    // SUM
    static T sum(const PvT& a) {
        LazyEvalStat<T, PvT> lazy;
        lazy.add_sum();
        return lazy.execute(a)[0];
    }

    // LEN
    static T len(const PvT& a) {
        LazyEvalStat<T, PvT> lazy;
        lazy.add([](const PvT& core) {
            return static_cast<T>(core.data.size());
        });
        return lazy.execute(a)[0];
    }
    
    static std::vector<size_t> shape(const PvT& a) {
    return a.shape;
    }

    // MAX
    static T max(const PvT& a) {
        LazyEvalStat<T, PvT> lazy;
        lazy.add([](const PvT& core) {
            return *std::max_element(core.data.begin(), core.data.end());
        });
        return lazy.execute(a)[0];
    }

    // MIN
    static T min(const PvT& a) {
        LazyEvalStat<T, PvT> lazy;
        lazy.add([](const PvT& core) {
            return *std::min_element(core.data.begin(), core.data.end());
        });
        return lazy.execute(a)[0];
    }

    // MEAN
    static T mean(const PvT& a) {
        LazyEvalStat<T, PvT> lazy;
        lazy.add([](const PvT& core) {
            T s = 0;
            for (auto v : core.data) s += v;
            return s / static_cast<T>(core.data.size());
        });
        return lazy.execute(a)[0];
    }

    // ARGMAX (lazy)
static size_t argmax(const PvT& a) {
    LazyEvalStat<size_t, PvT> lazy;
    lazy.add([](const PvT& core) -> size_t {
        return std::distance(core.data.begin(),
                             std::max_element(core.data.begin(), core.data.end()));
    });
    return lazy.execute(a)[0];
}

// ARGMIN (lazy)
static size_t argmin(const PvT& a) {
    LazyEvalStat<size_t, PvT> lazy;
    lazy.add([](const PvT& core) -> size_t {
        return std::distance(core.data.begin(),
                             std::min_element(core.data.begin(), core.data.end()));
    });
    return lazy.execute(a)[0];
}

    static PvT reshape(const PvT& a, const std::vector<size_t>& nw_shp) {
    size_t nw_tlt = 1;
    for (auto d : nw_shp) nw_tlt *= d;

    size_t old_tlt = 1;
    for (auto d : a.shape) old_tlt *= d;
    
    if (nw_tlt != old_tlt)
        throw std::runtime_error("Reshape failed: element count mismatch");

    PvT result;
    result.data = a.data;
    result.shape = nw_shp;  // ← fix
    result.computeStrides();
    return result;
}

static void reshape(PvT& a, const std::vector<size_t>& nw_shp) {
    size_t nw_tlt = 1;
    for (auto d : nw_shp) nw_tlt *= d;

    size_t old_tlt = 1;
    for (auto d : a.shape) old_tlt *= d;
    
    if (nw_tlt != old_tlt)
        throw std::runtime_error("Reshape failed: element count mismatch");

    a.shape = nw_shp;        
    a.computeStrides();
}

static void assign(PvT& a, T value) {
    std::fill(a.data.begin(), a.data.end(), value);
}

static PvT assign(const PvT& a, T value) {
    PvT result = a; 
    std::fill(result.data.begin(), result.data.end(), value);
    return result;
}
// In-place
static void assign(PvT& a,
                   const std::vector<size_t>& start,
                   const std::vector<size_t>& end,
                   T value) {
    // ... validasi start/end seperti sebelumnya
    for (size_t idx = 0; idx < a.data.size(); ++idx) {
        bool inside = true;
        size_t tmp = idx;
        for (size_t d = 0; d < a.shape.size(); ++d) {
            size_t pos = tmp / a.strides[d];
            tmp %= a.strides[d];
            if (pos < start[d] || pos >= end[d]) { inside = false; break; }
        }
        if (inside) a.data[idx] = value;
    }
}

// Return new PvT
static PvT assign(const PvT& a,
                   const std::vector<size_t>& start,
                   const std::vector<size_t>& end,
                   T value) {
    PvT result = a;   // copy
    assign(result, start, end, value); // reuse in-place
    return result;
}

};
