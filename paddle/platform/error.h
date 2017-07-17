#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#ifndef PADDLE_ONLY_CPU

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#endif  // PADDLE_ONLY_CPU

namespace paddle {
namespace platform {

#ifndef PADDLE_ONLY_CPU

inline void throw_on_error(cudaError_t e, const char* message) {
  if (e) {
    throw thrust::system_error(e, thrust::cuda_category(), message);
  }
}

inline void throw_on_error(curandStatus_t stat, const char* message) {
  if (stat != CURAND_STATUS_SUCCESS) {
    throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                               message);
  }
}

inline void throw_on_error(cudnnStatus_t stat, const char* message) {
  std::stringstream ss;
  if (stat == CUDNN_STATUS_SUCCESS) {
    return;
  } else {
    ss << cudnnGetErrorString(stat);
    ss << ", " << message;
    throw std::runtime_error(ss.str());
  }
}

inline void throw_on_error(cublasStatus_t stat, const char* message) {
  std::stringstream ss;
  if (stat == CUBLAS_STATUS_SUCCESS) {
    return;
  } else if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
    ss << "CUBLAS: not initialized";
  } else if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
    ss << "CUBLAS: alloc failed";
  } else if (stat == CUBLAS_STATUS_INVALID_VALUE) {
    ss << "CUBLAS: invalid value";
  } else if (stat == CUBLAS_STATUS_ARCH_MISMATCH) {
    ss << "CUBLAS: arch mismatch";
  } else if (stat == CUBLAS_STATUS_MAPPING_ERROR) {
    ss << "CUBLAS: mapping error";
  } else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
    ss << "CUBLAS: execution failed";
  } else if (stat == CUBLAS_STATUS_INTERNAL_ERROR) {
    ss << "CUBLAS: internal error";
  } else if (stat == CUBLAS_STATUS_NOT_SUPPORTED) {
    ss << "CUBLAS: not supported";
  } else if (stat == CUBLAS_STATUS_LICENSE_ERROR) {
    ss << "CUBLAS: license error";
  }
  ss << ", " << message;
  throw std::runtime_error(ss.str());
}

inline void throw_on_error(cublasStatus_t stat) {
  const char* message = "";
  throw_on_error(stat, message);
}

#endif  // PADDLE_ONLY_CPU

inline void throw_on_error(int stat, const char* message) {
  if (stat) {
    throw std::runtime_error(message + (", stat = " + std::to_string(stat)));
  }
}

}  // namespace platform
}  // namespace paddle
