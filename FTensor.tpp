#pragma once

/*Front End, Or wrapper*/

#include "Tensor.tpp"
#include "LazyEval.tpp"
#include "TensorOps.tpp"
template<typename T>
class Tensor {
private:
    Pv<T> storage;
    LazyEval<T, Pv<T>> lazy;

public:
    Tensor() {}
    Tensor(const std::vector<size_t>& shp, T init=T()) : storage(shp, init) {}

    // Factory
    static Tensor<T> zeros(const std::vector<size_t>& shp) { return Tensor<T>(shp, T(0)); }
    static Tensor<T> ones(const std::vector<size_t>& shp)  { return Tensor<T>(shp, T(1)); }
    static Tensor<T> fill(const std::vector<size_t>& shp, T init) { return Tensor<T>(shp, init); }
  //Random________  
    static Tensor<T> random(T min_val, T max_val, const std::vector<size_t>& shape) {
        Tensor<T> result;
        result.storage.shape = shape;
        size_t total = 1;
        for (auto d : shape) total *= d;
        result.storage.data.resize(total);
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (auto &v : result.storage.data) v = dist(gen);
        } 
        else if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            for (auto &v : result.storage.data) v = dist(gen);}
        else {
            throw std::runtime_error("Unsupported type for random()");
        }
        result.storage.computeStrides();
        return result;
    }

    // ============================
    // Elementwise operators (lazy)
    // ============================
    Tensor<T>& add(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x + y; });
    return *this;
    }
    Tensor<T>& sub(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x - y; });
    return *this;
    }
    Tensor<T>& mul(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x * y; });
    return *this;
    }
    Tensor<T>& div(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x / y; });
    return *this;
    }
    
    Tensor<T>& operator+=(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x + y; });
    return *this;
    }
    Tensor<T>& operator-=(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x - y; });
    return *this;
    }
    Tensor<T>& operator*=(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x * y; });
    return *this;
    }
    Tensor<T>& operator/=(const Tensor<T>& other) {
    lazy.add(other.storage, [](T x, T y){ return x / y; });
    return *this;
    }

    Tensor<T> operator+(const Tensor<T>& other) const {
        Tensor<T> result(*this);
        result += other;
        return result;}
    Tensor<T> operator-(const Tensor<T>& other) const {
        Tensor<T> result(*this);
        result -= other;
        return result;}
    Tensor<T> operator*(const Tensor<T>& other) const {
        Tensor<T> result(*this);
        result *= other;
        return result;}
    Tensor<T> operator/(const Tensor<T>& other) const {
        Tensor<T> result(*this);
        result /= other;
        return result;}

    // ============================
    // Unary ops (lazy)
    // ============================
    Tensor<T>& sqrt() {
    lazy.add(storage, [](T x, T){ return std::sqrt(x); });
    return *this;
}

Tensor<T>& pow(T v) {
    lazy.add(storage, [v](T x, T){ return std::pow(x, v); });
    return *this;
}

Tensor<T>& sin() {
    lazy.add(storage, [](T x, T){ return std::sin(x); });
    return *this;
}

Tensor<T>& cos() {
    lazy.add(storage, [](T x, T){ return std::cos(x); });
    return *this;
}
    //============================
    //Statistic operator
    //============================
   T sum() const { return Ops<T>::sum(storage); }
    T len() const { return Ops<T>::len(storage); }
    T max() const { return Ops<T>::max(storage); }
    T mean() const { return Ops<T>::mean(storage); }
    T min() const { return Ops<T>::min(storage); }
    std::vector<size_t> shape() const {return Ops<T>::shape(storage);}
    size_t argmax() const { return Ops<T>::argmax(storage); }
    size_t argmin() const { return Ops<T>::argmin(storage);}
    T dot(const Tensor<T>& other) const { return Ops<T>::dot(storage, other.storage); }
    //========================
    //=====Tensor Manipulation===
    //===================≠====
    Tensor<T>& reshape(const std::vector<size_t>& nw) {
    Ops<T>::reshape(storage, nw);
    return *this;}
    Tensor<T> reshaped(const std::vector<size_t>& nw)     const {
        return Ops<T>::reshape(storage, nw);}

Tensor<T>& assign(T value) {
    Ops<T>::assign(storage, value);
    return *this;
}
Tensor<T> assign(T value) const {
    return Ops<T>::assign(storage, value);
}
Tensor<T>& assign(const std::vector<size_t>& start,
                  const std::vector<size_t>& end,
                  T value) {
    Ops<T>::assign(storage, start, end, value);
    return *this;
}
Tensor<T> assign(const std::vector<size_t>& start,
                   const std::vector<size_t>& end,
                   T value) const {
    return Ops<T>::assign(storage, start, end, value);
}

    // ============================
    // evaluation
    // ============================
    Tensor<T>& evaluate() {
        lazy.execute(storage);
        return *this;
    }

    void print() const {
        for (size_t i = 0; i < storage.data.size(); ++i) {
            std::cout << storage.data[i];
            if (i + 1 != storage.data.size()) std::cout << ", ";
        }
        std::cout << "\n";
    }
    
};

template<typename T>
void print(const std::vector<T>& vec, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i + 1 < vec.size()) std::cout << " ";
    }
    std::cout << "\n";
}
