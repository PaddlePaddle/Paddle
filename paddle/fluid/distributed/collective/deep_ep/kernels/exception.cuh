// MIT License

// Copyright (c) 2025 DeepSeek

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <string>
#include <exception>

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException: public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd) \
do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif
