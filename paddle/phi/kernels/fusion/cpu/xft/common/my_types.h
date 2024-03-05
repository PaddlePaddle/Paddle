// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#ifndef MY_TYPES_H
#define MY_TYPES_H
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>

typedef int8_t s8;
typedef uint8_t u8;

#define unlikely(x) __builtin_expect((x), 0)

namespace hpj {

// void *xft_numa_alloc(size_t size) { return NULL;}
// void xft_numa_free(void *start, size_t size){}

template <typename T>
struct is_quantization_type {
    static const bool value = false;
};
template <>
struct is_quantization_type<u8> {
    static const bool value = true;
};
template <>
struct is_quantization_type<s8> {
    static const bool value = true;
};

template <typename T, bool _IS_QUANTIZED = is_quantization_type<T>::value>
struct MatData {
    // A sub matrix of others, if true
    bool shadow;

    int buf_alloc_size;
    T *buf;

    MatData() {
        this->shadow = false;
        this->buf = NULL;
        this->buf_alloc_size = 0;
    }
    MatData(T *buf) {
        this->shadow = true;
        this->buf = buf;
        this->buf_alloc_size = 0;
    }
    void Assign(T *buf) {
        if (shadow) {
            this->buf = buf;
        } else {
            Release();
            this->shadow = true;
            this->buf = buf;
            this->buf_alloc_size = 0;
        }
    }
    void Resize(int rows, int cols, int stride) {
        assert(!shadow);
        int size = rows * stride;
        if (this->buf_alloc_size >= size) {
            return;
        } else {
            // if (buf) { xft_numa_free(buf, sizeof(T) * buf_alloc_size); }
            this->buf_alloc_size = size;
            // buf = (T *)xft_numa_alloc(sizeof(T) * size);
            if (buf == NULL) { throw std::bad_alloc(); }
        }
    }
    void Release() {
        if (!shadow && buf) {
            // xft_numa_free(buf, sizeof(T) *buf_alloc_size);
            buf = NULL;
        }
        buf_alloc_size = 0;
    }
    ~MatData() { Release(); }
};

enum QuantizationScheme {
    qscheme_undefined = 0,
    per_tensor_symmetric = 1,
    per_tensor_affine = 2,
    per_channel_symmetric = 3,
    per_channel_affine = 4,
};

// The matrix is quantized per row/channel
template <typename T>
struct MatData<T, true> {
    // A sub matrix of others, if true
    bool shadow;

    int buf_alloc_size;

    T *buf;

    QuantizationScheme qscheme;

    union QParam {
        struct QParamPerTensor {
            float scale;
            int32_t zp;
        } per_t;
        struct QParamPerChannel {
            float *scales;
            int32_t *zps;
            int alloc_size;
        } per_c;
    } qparam;

    MatData() {
        this->shadow = false;
        this->buf = NULL;
        this->buf_alloc_size = 0;
        this->qscheme = qscheme_undefined;
    }
    void Resize(int rows, int cols, int stride) {
        assert(!shadow);
        int size = rows * stride;
        if (this->buf_alloc_size < size) {
            // if (buf) { xft_numa_free(buf, sizeof(T) * buf_alloc_size); }
            this->buf_alloc_size = size;
            // buf = (T *)xft_numa_alloc(sizeof(T) * size);
            if (buf == NULL) { throw std::bad_alloc(); }
        }
        // Check the scale and zero point buffer
        if ((this->qscheme == per_channel_symmetric || this->qscheme == per_channel_affine)
                && this->qparam.per_c.alloc_size < rows) {
            if (this->qparam.per_c.scales) { free(this->qparam.per_c.scales); }
            this->qparam.per_c.scales = (float *)aligned_alloc(64, sizeof(float) * rows);
            if (this->qparam.per_c.scales == NULL) { throw std::bad_alloc(); }
            this->qparam.per_c.alloc_size = rows;
            // For per_channel_affine, need to check buffer for zero point
            if (this->qscheme == per_channel_affine) {
                if (this->qparam.per_c.zps) { free(this->qparam.per_c.zps); }
                this->qparam.per_c.zps = (int32_t *)aligned_alloc(64, sizeof(int32_t) * rows);
                if (this->qparam.per_c.zps == NULL) { throw std::bad_alloc(); }
            }
        }
    }
    void Release() {
        if (!shadow && buf) {
            // xft_numa_free(buf, sizeof(T) * buf_alloc_size);
            buf = NULL;
        }
        if (!shadow && (this->qscheme == per_channel_symmetric || this->qscheme == per_channel_affine)) {
            free(this->qparam.per_c.scales);
            this->qparam.per_c.scales = NULL;
            if (this->qscheme == per_channel_affine) {
                free(this->qparam.per_c.zps);
                this->qparam.per_c.zps = NULL;
            }
            this->qparam.per_c.alloc_size = 0;
        }
        buf_alloc_size = 0;
    }
    void SetQScheme(QuantizationScheme scheme) {
        if (scheme == per_tensor_symmetric || scheme == per_tensor_affine) {
            // From per_channel to per_tensor
            if (unlikely(this->qscheme == per_channel_symmetric || this->qscheme == per_channel_affine)) {
                if (this->qparam.per_c.scales) { free(this->qparam.per_c.scales); }
                if (this->qparam.per_c.zps) { free(this->qparam.per_c.zps); }
            }
            this->qparam.per_t.scale = 1.0;
            this->qparam.per_t.zp = 0;
        } else if (scheme == per_channel_symmetric || scheme == per_channel_affine) {
            // From non_per_channel to per_channel
            if (this->qscheme != per_channel_symmetric && this->qscheme != per_channel_affine) {
                this->qparam.per_c.scales = NULL;
                this->qparam.per_c.zps = NULL;
                this->qparam.per_c.alloc_size = 0;
            }
        }
        this->qscheme = scheme;
    }
    float *Scales() {
        if (this->qscheme == per_tensor_symmetric || this->qscheme == per_tensor_affine) {
            return &qparam.per_t.scale;
        } else if (this->qscheme == per_channel_symmetric || this->qscheme == per_channel_affine) {
            return qparam.per_c.scales;
        } else {
            return NULL;
        }
    }
    int32_t *ZeroPoint() {
        if (this->qscheme == per_tensor_symmetric || this->qscheme == per_tensor_affine) {
            return &qparam.per_t.zp;
        } else if (this->qscheme == per_channel_symmetric || this->qscheme == per_channel_affine) {
            return qparam.per_c.zps;
        } else {
            return NULL;
        }
    }
    ~MatData() { Release(); }
};

template <typename T>
class Matrix {
public:
    int rows;
    int cols;
    int stride;

    MatData<T> data;

    Matrix &operator=(const Matrix &m);

public:
    Matrix() {
        this->rows = 0;
        this->cols = 0;
        this->stride = 0;
    }

    Matrix(Matrix &m, int start_row, int rows, int start_col, int cols)
        : data(m.data.buf + start_row * m.stride + start_col) {
        this->rows = rows;
        this->cols = cols;
        this->stride = m.stride;
    }

    Matrix(Matrix &m) : data(m.data) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->stride = m.stride;
    }

    // Create dilated matrix, for example, if dilation = 2, then select the 1st, 3rd, 5th, ... lines
    Matrix(Matrix &m, int start_row, int dilation, bool unused) : data(m.data.buf + start_row * m.stride) {
        this->rows = m.rows / dilation;
        this->cols = m.cols;
        this->stride = m.stride * dilation;
    }

    Matrix(Matrix &m, int start_row, int rows) : data(m.data.buf + start_row * m.stride) {
        this->rows = rows;
        this->cols = m.cols;
        this->stride = m.stride;
    }

    Matrix(T *buf, int rows, int cols, int stride) : data(buf) {
        this->rows = rows;
        this->cols = cols;
        this->stride = stride;
    }

    ~Matrix() { this->Release(); }

    void Assign(T *buf, int rows, int cols, int stride) {
        this->data.Assign(buf);
        this->rows = rows;
        this->cols = cols;
        this->stride = stride;
    }

    void Resize(int rows, int cols) {
        assert(!data.shadow);

        if (this->rows == rows && this->cols == cols) { return; }
        if (rows <= 0 || cols <= 0) {
            this->Release();
            return;
        }

        // Previously, we used to pad the matrix when the columns aligned with the boundary of 1024.
        // However, we discovered that this did not enhance the performance. 
        // As a result, we have decided to remove this approach.
        this->stride = cols;
        this->rows = rows;
        this->cols = cols;
        this->data.Resize(rows, cols, stride);
    }
    void Resize(int rows, int cols, int stride) {
        assert(!data.shadow);

        if (this->rows == rows && this->cols == cols && this->stride == stride) { return; }
        if (rows <= 0 || cols <= 0 || stride <= 0) {
            this->Release();
            return;
        }
        this->rows = rows;
        this->cols = cols;
        this->stride = stride;
        this->data.Resize(rows, cols, stride);
    }
    T *Data() { return data.buf; }
    const T *Data() const { return data.buf; }
    void SetQScheme(QuantizationScheme qscheme) { data.SetQScheme(qscheme); }
    float *Scales() { return data.Scales(); }
    int32_t *ZeroPoint() { return data.ZeroPoint(); }
    void Release() {
        this->data.Release();
        this->rows = 0;
        this->cols = 0;
        this->stride = 0;
    }
    int Rows() const { return this->rows; }
    int Cols() const { return this->cols; }
    int Stride() const { return this->stride; }
    T *Row(const int idx) {
        // assert(idx < rows_ && idx >= 0);
        return this->data.buf + this->stride * idx;
    }
    const T *Row(const int idx) const { return this->data.buf + this->stride * idx; }
    T &operator()(int r, int c) {
        // assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return *(this->data.buf + r * this->stride + c);
    }
};

template <typename T>
class Vector {
public:
    T *data;
    int size;
    int alloc_size;

public:
    Vector() {
        data = NULL;
        size = 0;
        alloc_size = 0;
    }
    ~Vector() { this->Release(); }
    void Resize(int size) {
        if (size <= 0) {
            this->Release();
            return;
        }
        if (this->alloc_size >= size) { // space is enough
            this->size = size;
            return;
        }
        // if (this->data) { xft_numa_free(this->data, sizeof(T) * alloc_size); }
        this->alloc_size = size + (16 - (size % 16)) % 16;
        this->size = size;
        // this->data = (T *)xft_numa_alloc(sizeof(T) * alloc_size);
        if (this->data == NULL) { throw std::bad_alloc(); }
    }
    void SetZero() { memset(data, 0, sizeof(T) * size); }
    T *Data() { return data; }
    void Release() {
        if (data) {
            // xft_numa_free(data, sizeof(T) * alloc_size);
            data = NULL;
        }
        size = 0;
        alloc_size = 0;
    }
    int Size() { return size; }
};
} // namespace hpj
#endif