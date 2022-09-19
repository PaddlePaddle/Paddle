/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PLUGIN_H
#define PLUGIN_H
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "NvInfer.h"
#include "NvInferPlugin.h"

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {

class BasePlugin : public IPluginV2 {
 protected:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

  std::string mNamespace;
};

class BaseCreator : public IPluginCreator {
 public:
  void setPluginNamespace(const char* libNamespace) noexcept override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

 protected:
  std::string mNamespace;
};

namespace plugin {

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer) {
  T val{};
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}

}  // namespace plugin
}  // namespace nvinfer1

#ifndef DEBUG

#define PLUGIN_CHECK(status)  \
  do {                        \
    if (status != 0) abort(); \
  } while (0)

#define ASSERT_PARAM(exp)                \
  do {                                   \
    if (!(exp)) return STATUS_BAD_PARAM; \
  } while (0)

#define ASSERT_FAILURE(exp)            \
  do {                                 \
    if (!(exp)) return STATUS_FAILURE; \
  } while (0)

#define CSC(call, err)               \
  do {                               \
    cudaError_t cudaStatus = call;   \
    if (cudaStatus != cudaSuccess) { \
      return err;                    \
    }                                \
  } while (0)

#define DEBUG_PRINTF(...) \
  do {                    \
  } while (0)

#else

#define ASSERT_PARAM(exp)                                                   \
  do {                                                                      \
    if (!(exp)) {                                                           \
      fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
      return STATUS_BAD_PARAM;                                              \
    }                                                                       \
  } while (0)

#define ASSERT_FAILURE(exp)                                               \
  do {                                                                    \
    if (!(exp)) {                                                         \
      fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
      return STATUS_FAILURE;                                              \
    }                                                                     \
  } while (0)

#define CSC(call, err)                        \
  do {                                        \
    cudaError_t cudaStatus = call;            \
    if (cudaStatus != cudaSuccess) {          \
      printf("%s %d CUDA FAIL %s\n",          \
             __FILE__,                        \
             __LINE__,                        \
             cudaGetErrorString(cudaStatus)); \
      return err;                             \
    }                                         \
  } while (0)

#define PLUGIN_CHECK(status)                    \
  {                                             \
    if (status != 0) {                          \
      DEBUG_PRINTF("%s %d CUDA FAIL %s\n",      \
                   __FILE__,                    \
                   __LINE__,                    \
                   cudaGetErrorString(status)); \
      abort();                                  \
    }                                           \
  }

#define DEBUG_PRINTF(...) \
  do {                    \
    printf(__VA_ARGS__);  \
  } while (0)

#endif  // DEBUG

#endif  // TRT_PLUGIN_H
