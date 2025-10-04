#pragma once

#include <vector>
#include <cstddef>

/* Tensor Core tpp*/

template<typename T>
struct Pv {
    //Data 
    std::vector<T> data; //flat data
    std::vector<size_t> shape; //shape data
    std::vector<size_t> strides; //Strides data
    //constructor 
    Pv() {}
    //constructor 
    Pv(const std::vector<size_t>& shp, T init = T()) : shape(shp) {
        computeStrides();
        data.resize(size(), init);
    }
    //Constructor 
    Pv(const std::vector<T>& values) : shape({values.size()}), data(values) {
        computeStrides();
    }
    //Strides compute for Cordinate
    void computeStrides() {
        strides.resize(shape.size(), 1);
        for (int i = int(shape.size()) - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    //Get a flat cordinate
    size_t flattenIndex(const std::vector<size_t>& coord) const {
        size_t idx = 0;
        for (size_t i = 0; i < coord.size(); i++)
            idx += (shape[i] == 1 ? 0 : coord[i]) * strides[i];
        return idx;
    }

    T& operator[](size_t idx) { return data[idx]; }
    const T& operator[](size_t idx) const { return data[idx]; }

    //Return size
    size_t size() const {
        size_t tot = 1;
        for (auto s : shape) tot *= s;
        return tot;
    }
};
