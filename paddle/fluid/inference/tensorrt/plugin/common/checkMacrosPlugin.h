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

#ifndef CHECK_MACROS_PLUGIN_H
#define CHECK_MACROS_PLUGIN_H

#include <mutex>
#include <sstream>
#include "NvInfer.h"

#ifndef TRT_CHECK_MACROS_H
#ifndef TRT_TUT_HELPERS_H

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#endif  // TRT_TUT_HELPERS_H
#endif  // TRT_CHECK_MACROS_H

namespace nvinfer1 {
namespace plugin {
template <ILogger::Severity kSeverity>
class LogStream : public std::ostream {
  class Buf : public std::stringbuf {
   public:
    int sync() override;
  };

  Buf buffer;
  std::mutex mLogStreamMutex;

 public:
  std::mutex& getMutex() { return mLogStreamMutex; }
  LogStream() : std::ostream(&buffer){};
};

// Use mutex to protect multi-stream write to buffer
template <ILogger::Severity kSeverity, typename T>
LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, T const& msg) {
  std::lock_guard<std::mutex> guard(stream.getMutex());
  auto& os = static_cast<std::ostream&>(stream);
  os << msg;
  return stream;
}

// Special handling static numbers
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream,
                                        int32_t num) {
  std::lock_guard<std::mutex> guard(stream.getMutex());
  auto& os = static_cast<std::ostream&>(stream);
  os << num;
  return stream;
}

// Special handling std::endl
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream,
                                        std::ostream& (*f)(std::ostream&)) {
  std::lock_guard<std::mutex> guard(stream.getMutex());
  auto& os = static_cast<std::ostream&>(stream);
  os << f;
  return stream;
}

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;
extern LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

void reportValidationFailure(char const* msg, char const* file, int line);
void reportAssertion(const char* msg, const char* file, int line);
void logError(const char* msg, const char* file, const char* fn, int line);

[[noreturn]] void throwCudaError(const char* file,
                                 const char* function,
                                 int line,
                                 int status,
                                 const char* msg = nullptr);
[[noreturn]] void throwCudnnError(const char* file,
                                  const char* function,
                                  int line,
                                  int status,
                                  const char* msg = nullptr);
[[noreturn]] void throwCublasError(const char* file,
                                   const char* function,
                                   int line,
                                   int status,
                                   const char* msg = nullptr);
[[noreturn]] void throwPluginError(char const* file,
                                   char const* function,
                                   int line,
                                   int status,
                                   char const* msg = nullptr);

class TRTException : public std::exception {
 public:
  TRTException(const char* fl,
               const char* fn,
               int ln,
               int st,
               const char* msg,
               const char* nm)
      : file(fl), function(fn), line(ln), status(st), message(msg), name(nm) {}
  virtual void log(std::ostream& logStream) const;
  void setMessage(const char* msg) { message = msg; }

 protected:
  const char* file{nullptr};
  const char* function{nullptr};
  int line{0};
  int status{0};
  const char* message{nullptr};
  const char* name{nullptr};
};

class CudaError : public TRTException {
 public:
  CudaError(const char* fl,
            const char* fn,
            int ln,
            int stat,
            const char* msg = nullptr)
      : TRTException(fl, fn, ln, stat, msg, "Cuda") {}
};

class CudnnError : public TRTException {
 public:
  CudnnError(const char* fl,
             const char* fn,
             int ln,
             int stat,
             const char* msg = nullptr)
      : TRTException(fl, fn, ln, stat, msg, "Cudnn") {}
};

class CublasError : public TRTException {
 public:
  CublasError(const char* fl,
              const char* fn,
              int ln,
              int stat,
              const char* msg = nullptr)
      : TRTException(fl, fn, ln, stat, msg, "cuBLAS") {}
};

class PluginError : public TRTException {
 public:
  PluginError(char const* fl,
              char const* fn,
              int ln,
              int stat,
              char const* msg = nullptr)
      : TRTException(fl, fn, ln, stat, msg, "Plugin") {}
};

inline void caughtError(const std::exception& e) {
  gLogError << e.what() << std::endl;
}
}  // namespace plugin

}  // namespace nvinfer1

#ifndef TRT_CHECK_MACROS_H
#ifndef TRT_TUT_HELPERS_H

#define PLUGIN_API_CHECK(condition)                                        \
  {                                                                        \
    if ((condition) == false) {                                            \
      nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__); \
      return;                                                              \
    }                                                                      \
  }

#define PLUGIN_API_CHECK_RETVAL(condition, retval)                         \
  {                                                                        \
    if ((condition) == false) {                                            \
      nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__); \
      return retval;                                                       \
    }                                                                      \
  }

#define PLUGIN_API_CHECK_ENUM_RANGE(Type, val) \
  PLUGIN_API_CHECK(int(val) >= 0 && int(val) < EnumMax<Type>())
#define PLUGIN_API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval) \
  PLUGIN_API_CHECK_RETVAL(int(val) >= 0 && int(val) < EnumMax<Type>(), retval)

#define PLUGIN_CHECK_CUDA(call)  \
  do {                           \
    cudaError_t status = call;   \
    if (status != cudaSuccess) { \
      return status;             \
    }                            \
  } while (0)

#define PLUGIN_CHECK_CUDNN(call)          \
  do {                                    \
    cudnnStatus_t status = call;          \
    if (status != CUDNN_STATUS_SUCCESS) { \
      return status;                      \
    }                                     \
  } while (0)

#define PLUGIN_CUBLASASSERT(status_)                                       \
  {                                                                        \
    auto s_ = status_;                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
      nvinfer1::plugin::throwCublasError(__FILE__, FN_NAME, __LINE__, s_); \
    }                                                                      \
  }

#define PLUGIN_CUDNNASSERT(status_)                                            \
  {                                                                            \
    auto s_ = status_;                                                         \
    if (s_ != CUDNN_STATUS_SUCCESS) {                                          \
      const char* msg = cudnnGetErrorString(s_);                               \
      nvinfer1::plugin::throwCudnnError(__FILE__, FN_NAME, __LINE__, s_, msg); \
    }                                                                          \
  }

#define PLUGIN_CUASSERT(status_)                                              \
  {                                                                           \
    auto s_ = status_;                                                        \
    if (s_ != cudaSuccess) {                                                  \
      const char* msg = cudaGetErrorString(s_);                               \
      nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg); \
    }                                                                         \
  }

// Logs failed condition and throws a PluginError.
// PLUGIN_ASSERT will eventually perform this function, at which point
// PLUGIN_VALIDATE will be removed.
#define PLUGIN_VALIDATE(condition)                     \
  {                                                    \
    if (!(condition)) {                                \
      nvinfer1::plugin::throwPluginError(              \
          __FILE__, FN_NAME, __LINE__, 0, #condition); \
    }                                                  \
  }

// Logs failed assertion and aborts.
// Aborting is undesirable and will be phased-out from the plugin module, at
// which point PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
#define PLUGIN_ASSERT(assertion)                                         \
  {                                                                      \
    if (!(assertion)) {                                                  \
      nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__); \
    }                                                                    \
  }

#define PLUGIN_FAIL(msg) \
  { nvinfer1::plugin::reportAssertion(msg, __FILE__, __LINE__); }

#define PLUGIN_CUERROR(status_)                               \
  {                                                           \
    auto s_ = status_;                                        \
    if (s_ != 0)                                              \
      nvinfer1::plugin::logError(                             \
          #status_ " failure.", __FILE__, FN_NAME, __LINE__); \
  }

#endif  // TRT_TUT_HELPERS_H
#endif  // TRT_CHECK_MACROS_H

#endif /*CHECK_MACROS_PLUGIN_H*/
