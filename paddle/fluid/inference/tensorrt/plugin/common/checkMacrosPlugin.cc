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

#include "checkMacrosPlugin.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace nvinfer1 {
namespace plugin {

// This will be populated by the logger supplied by the user to
// initLibNvInferPlugins()
ILogger* gLogger{};

template <ILogger::Severity kSeverity>
int LogStream<kSeverity>::Buf::sync() {
  std::string s = str();
  while (!s.empty() && s.back() == '\n') {
    s.pop_back();
  }
  if (gLogger != nullptr) {
    gLogger->log(kSeverity, s.c_str());
  }
  str("");
  return 0;
}

// These use gLogger, and therefore require initLibNvInferPlugins() to be called
// with a logger (otherwise, it will not log)
LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;
LogStream<ILogger::Severity::kINFO> gLogInfo;
LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

// break-pointable
void throwCudaError(const char* file,
                    const char* function,
                    int line,
                    int status,
                    const char* msg) {
  CudaError error(file, function, line, status, msg);
  error.log(gLogError);
  throw error;
}

// break-pointable
void throwCublasError(const char* file,
                      const char* function,
                      int line,
                      int status,
                      const char* msg) {
  if (msg == nullptr) {
    auto s_ = static_cast<cublasStatus_t>(status);
    switch (s_) {
      case CUBLAS_STATUS_SUCCESS:
        msg = "CUBLAS_STATUS_SUCCESS";
        break;
      case CUBLAS_STATUS_NOT_INITIALIZED:
        msg = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        msg = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        msg = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        msg = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        msg = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        msg = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        msg = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        msg = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        msg = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    }
  }
  CublasError error(file, function, line, status, msg);
  error.log(gLogError);
  throw error;
}

// break-pointable
void throwCudnnError(const char* file,
                     const char* function,
                     int line,
                     int status,
                     const char* msg) {
  CudnnError error(file, function, line, status, msg);
  error.log(gLogError);
  throw error;
}

// break-pointable
void throwPluginError(char const* file,
                      char const* function,
                      int line,
                      int status,
                      char const* msg) {
  PluginError error(file, function, line, status, msg);
  reportValidationFailure(msg, file, line);
  throw error;
}

void logError(const char* msg, const char* file, const char* fn, int line) {
  gLogError << "Parameter check failed at: " << file << "::" << fn
            << "::" << line;
  gLogError << ", condition: " << msg << std::endl;
}

void reportValidationFailure(char const* msg, char const* file, int line) {
  std::ostringstream stream;
  stream << "Validation failed: " << msg << std::endl
         << file << ':' << line << std::endl;
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                   stream.str().c_str());
}

// break-pointable
void reportAssertion(const char* msg, const char* file, int line) {
  std::ostringstream stream;
  stream << "Assertion failed: " << msg << std::endl
         << file << ':' << line << std::endl
         << "Aborting..." << std::endl;
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                   stream.str().c_str());
  PLUGIN_CUASSERT(cudaDeviceReset());
  abort();
}

void TRTException::log(std::ostream& logStream) const {
  logStream << file << " (" << line << ") - " << name << " Error in "
            << function << ": " << status;
  if (message != nullptr) {
    logStream << " (" << message << ")";
  }
  logStream << std::endl;
}

}  // namespace plugin

}  // namespace nvinfer1
