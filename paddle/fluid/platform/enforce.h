/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__

#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include "paddle/fluid/platform/external_error.pb.h"
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hiprand.h>
#include <miopen/miopen.h>
#include <rocblas.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>  // NOLINT
#endif

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/to_string.h"
#include "paddle/phi/backends/dynload/port.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/curand.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include <error.h>
#include "paddle/phi/backends/dynload/nccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/hipfft.h"
#include "paddle/phi/backends/dynload/hiprand.h"
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include <error.h>  // NOLINT
#include "paddle/phi/backends/dynload/rccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_HIP

// Note: these headers for simplify demangle type string
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/phi/core/enforce.h"
// Note: this header for simplify HIP and CUDA type string
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#endif
#include "paddle/fluid/platform/flags.h"

namespace phi {
class ErrorSummary;
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
DECLARE_int64(gpu_allocator_retry_time);
#endif
DECLARE_int32(call_stack_level);

namespace paddle {
namespace platform {
using namespace ::phi::enforce;  // NOLINT

/** HELPER MACROS AND FUNCTIONS **/

#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
#endif

/*
 * Summary: This BOOST_GET(_**) series macros are used to call boost::get
 *   safely. boost::get is not a completely safe api, although it will not
 *   go wrong in most cases, but in extreme cases, it may fail and directly
 *   throw a boost::bad_get exception, without any stack information.
 *   This kind of problems is difficult to debug, so add these macros to
 *   enrich boost::get error information. At the same time, we restrict
 *   the direct use of boost::get by CI rule.
 *
 * Parameters:
 *     __TYPE: the target variable type
 *     __VALUE: the target variable to get
 *
 * Examples:
 *     - unsafe writing: int x = boost::get<int>(y);
 *     - safe writing: int x = BOOST_GET(int, y);
 *
 * Note: GCC 4.8 cannot select right overloaded function here, so need
 *    to define different functions and macros here, after we upgreade
 *    CI gcc version, we can only define one BOOST_GET macro.
 */
namespace details {

using namespace phi::enforce::details;  // NOLINT

#define DEFINE_SAFE_BOOST_GET(__InputType, __OutputType, __OutputTypePtr,      \
                              __FuncName)                                      \
  template <typename OutputType, typename InputType>                           \
  auto __FuncName(__InputType input, const char* expression, const char* file, \
                  int line)                                                    \
      ->typename std::conditional<std::is_pointer<InputType>::value,           \
                                  __OutputTypePtr, __OutputType>::type {       \
    try {                                                                      \
      return boost::get<OutputType>(input);                                    \
    } catch (boost::bad_get&) {                                                \
      HANDLE_THE_ERROR                                                         \
      throw ::phi::enforce::EnforceNotMet(                                     \
          phi::errors::InvalidArgument(                                        \
              "boost::get failed, cannot get value "                           \
              "(%s) by type %s, its type is %s.",                              \
              expression, phi::enforce::demangle(typeid(OutputType).name()),   \
              phi::enforce::demangle(input.type().name())),                    \
          file, line);                                                         \
      END_HANDLE_THE_ERROR                                                     \
    }                                                                          \
  }

DEFINE_SAFE_BOOST_GET(InputType&, OutputType&, OutputType*, SafeBoostGet);
DEFINE_SAFE_BOOST_GET(const InputType&, const OutputType&, const OutputType*,
                      SafeBoostGetConst);
DEFINE_SAFE_BOOST_GET(InputType&&, OutputType, OutputType*,
                      SafeBoostGetMutable);

}  // namespace details

#define BOOST_GET(__TYPE, __VALUE)                                             \
  paddle::platform::details::SafeBoostGet<__TYPE>(__VALUE, #__VALUE, __FILE__, \
                                                  __LINE__)
#define BOOST_GET_CONST(__TYPE, __VALUE)                                  \
  paddle::platform::details::SafeBoostGetConst<__TYPE>(__VALUE, #__VALUE, \
                                                       __FILE__, __LINE__)
#define BOOST_GET_MUTABLE(__TYPE, __VALUE)                                  \
  paddle::platform::details::SafeBoostGetMutable<__TYPE>(__VALUE, #__VALUE, \
                                                         __FILE__, __LINE__)

/** OTHER EXCEPTION AND ENFORCE **/

struct EOFException : public std::exception {
  std::string err_str_;
  EOFException(const char* err_msg, const char* file, int line) {
    err_str_ = paddle::string::Sprintf("%s at [%s:%d]", err_msg, file, line);
  }

  const char* what() const noexcept override { return err_str_.c_str(); }
};

#define PADDLE_THROW_EOF()                                                   \
  do {                                                                       \
    HANDLE_THE_ERROR                                                         \
    throw paddle::platform::EOFException("There is no next data.", __FILE__, \
                                         __LINE__);                          \
    END_HANDLE_THE_ERROR                                                     \
  } while (0)

#define PADDLE_THROW_BAD_ALLOC(...)                                      \
  do {                                                                   \
    HANDLE_THE_ERROR                                                     \
    throw ::paddle::memory::allocation::BadAlloc(                        \
        phi::ErrorSummary(__VA_ARGS__).to_string(), __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                                 \
  } while (0)

/**************************************************************************/
/**************************** NVIDIA ERROR ********************************/
#ifdef PADDLE_WITH_CUDA

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value, proto_type) \
  template <>                                                     \
  struct ExternalApiType<type> {                                  \
    using Type = type;                                            \
    static constexpr Type kSuccess = success_value;               \
    static constexpr const char* kTypeString = #proto_type;       \
    static constexpr platform::proto::ApiType kProtoType =        \
        platform::proto::ApiType::proto_type;                     \
  }

DEFINE_EXTERNAL_API_TYPE(cudaError_t, cudaSuccess, CUDA);
DEFINE_EXTERNAL_API_TYPE(curandStatus_t, CURAND_STATUS_SUCCESS, CURAND);
DEFINE_EXTERNAL_API_TYPE(cudnnStatus_t, CUDNN_STATUS_SUCCESS, CUDNN);
DEFINE_EXTERNAL_API_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS, CUBLAS);
DEFINE_EXTERNAL_API_TYPE(cusparseStatus_t, CUSPARSE_STATUS_SUCCESS, CUSPARSE);
DEFINE_EXTERNAL_API_TYPE(cusolverStatus_t, CUSOLVER_STATUS_SUCCESS, CUSOLVER);
DEFINE_EXTERNAL_API_TYPE(cufftResult_t, CUFFT_SUCCESS, CUFFT);
DEFINE_EXTERNAL_API_TYPE(CUresult, CUDA_SUCCESS, CU);

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess, NCCL);
#endif

}  // namespace details

template <typename T>
inline const char* GetErrorMsgUrl(T status) {
  using __CUDA_STATUS_TYPE__ = decltype(status);
  platform::proto::ApiType proto_type =
      details::ExternalApiType<__CUDA_STATUS_TYPE__>::kProtoType;
  switch (proto_type) {
    case platform::proto::ApiType::CUDA:
    case platform::proto::ApiType::CU:
      return "https://docs.nvidia.com/cuda/cuda-runtime-api/"
             "group__CUDART__TYPES.html#group__CUDART__TYPES_"
             "1g3f51e3575c2178246db0a94a430e0038";
      break;
    case platform::proto::ApiType::CURAND:
      return "https://docs.nvidia.com/cuda/curand/"
             "group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437";
      break;
    case platform::proto::ApiType::CUDNN:
      return "https://docs.nvidia.com/deeplearning/cudnn/api/"
             "index.html#cudnnStatus_t";
      break;
    case platform::proto::ApiType::CUBLAS:
      return "https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t";
      break;
    case platform::proto::ApiType::CUSOLVER:
      return "https://docs.nvidia.com/cuda/cusolver/"
             "index.html#cuSolverSPstatus";
      break;
    case platform::proto::ApiType::NCCL:
      return "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/"
             "types.html#ncclresult-t";
      break;
    case platform::proto::ApiType::CUFFT:
      return "https://docs.nvidia.com/cuda/cufft/index.html#cufftresult";
    case platform::proto::ApiType::CUSPARSE:
      return "https://docs.nvidia.com/cuda/cusparse/"
             "index.html#cusparseStatus_t";
      break;
    default:
      return "Unknown type of External API, can't get error message URL!";
      break;
  }
}

template <typename T>
inline std::string GetExternalErrorMsg(T status) {
  std::ostringstream sout;
  bool _initSucceed = false;
  platform::proto::ExternalErrorDesc externalError;
  if (externalError.ByteSizeLong() == 0) {
    std::string filePath;
#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCurrentTraceBackString), &info)) {
      std::string strModule(info.dli_fname);
      const size_t last_slash_idx = strModule.find_last_of("/");
      std::string compare_path = strModule.substr(strModule.length() - 6);
      if (std::string::npos != last_slash_idx) {
        strModule.erase(last_slash_idx, std::string::npos);
      }
      if (compare_path.compare("avx.so") == 0) {
        filePath =
            strModule +
            "/../include/third_party/externalError/data/externalErrorMsg.pb";
      } else {
        filePath = strModule +
                   "/../../third_party/externalError/data/externalErrorMsg.pb";
      }
    }
#else
    char buf[512];
    MEMORY_BASIC_INFORMATION mbi;
    HMODULE h_module =
        (::VirtualQuery(GetCurrentTraceBackString, &mbi, sizeof(mbi)) != 0)
            ? (HMODULE)mbi.AllocationBase
            : NULL;
    GetModuleFileName(h_module, buf, 512);
    std::string strModule(buf);
    const size_t last_slash_idx = strModule.find_last_of("\\");
    std::string compare_path = strModule.substr(strModule.length() - 7);
    if (std::string::npos != last_slash_idx) {
      strModule.erase(last_slash_idx, std::string::npos);
    }
    if (compare_path.compare("avx.pyd") == 0) {
      filePath = strModule +
                 "\\..\\include\\third_"
                 "party\\externalerror\\data\\externalErrorMsg.pb";
    } else {
      filePath =
          strModule +
          "\\..\\..\\third_party\\externalerror\\data\\externalErrorMsg.pb";
    }
#endif
    std::ifstream fin(filePath, std::ios::in | std::ios::binary);
    _initSucceed = externalError.ParseFromIstream(&fin);
  }
  using __CUDA_STATUS_TYPE__ = decltype(status);
  platform::proto::ApiType proto_type =
      details::ExternalApiType<__CUDA_STATUS_TYPE__>::kProtoType;
  if (_initSucceed) {
    for (int i = 0; i < externalError.errors_size(); ++i) {
      if (proto_type == externalError.errors(i).type()) {
        for (int j = 0; j < externalError.errors(i).messages_size(); ++j) {
          if (status == externalError.errors(i).messages(j).code()) {
            sout << "\n  [Hint: "
                 << externalError.errors(i).messages(j).message() << "]";
            return sout.str();
          }
        }
      }
    }
  }

  sout << "\n  [Hint: Please search for the error code(" << status
       << ") on website (" << GetErrorMsgUrl(status)
       << ") to get Nvidia's official solution and advice about "
       << details::ExternalApiType<__CUDA_STATUS_TYPE__>::kTypeString
       << " Error.]";
  return sout.str();
}

template std::string GetExternalErrorMsg<cudaError_t>(cudaError_t);
template std::string GetExternalErrorMsg<curandStatus_t>(curandStatus_t);
template std::string GetExternalErrorMsg<cudnnStatus_t>(cudnnStatus_t);
template std::string GetExternalErrorMsg<cublasStatus_t>(cublasStatus_t);
template std::string GetExternalErrorMsg<cusparseStatus_t>(cusparseStatus_t);
template std::string GetExternalErrorMsg<cusolverStatus_t>(cusolverStatus_t);
template std::string GetExternalErrorMsg<cufftResult_t>(cufftResult_t);
template std::string GetExternalErrorMsg<CUresult>(CUresult);
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
template std::string GetExternalErrorMsg<ncclResult_t>(ncclResult_t);
#endif

/*************** CUDA ERROR ***************/
inline bool is_error(cudaError_t e) { return e != cudaSuccess; }

inline std::string build_nvidia_error_msg(cudaError_t e) {
  std::ostringstream sout;
  sout << "CUDA error(" << e << "), " << cudaGetErrorString(e) << ". "
       << GetExternalErrorMsg(e);
  return sout.str();
}

/*************** CURAND ERROR ***************/
inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(curandStatus_t stat) {
  std::ostringstream sout;
  sout << "CURAND error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUDNN ERROR ***************/
inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cudnnStatus_t stat) {
  std::ostringstream sout;
  sout << "CUDNN error(" << stat << "), "
       << phi::dynload::cudnnGetErrorString(stat) << ". "
       << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUBLAS ERROR ***************/
inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cublasStatus_t stat) {
  std::ostringstream sout;
  sout << "CUBLAS error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUSPARSE ERROR ***************/
inline bool is_error(cusparseStatus_t stat) {
  return stat != CUSPARSE_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cusparseStatus_t stat) {
  std::ostringstream sout;
  sout << "CUSparse error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUSOLVER ERROR ***************/
inline bool is_error(cusolverStatus_t stat) {
  return stat != CUSOLVER_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cusolverStatus_t stat) {
  std::ostringstream sout;
  sout << "CUSOLVER error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUFFT ERROR ***************/
inline bool is_error(cufftResult_t stat) { return stat != CUFFT_SUCCESS; }

inline std::string build_nvidia_error_msg(cufftResult_t stat) {
  std::ostringstream sout;
  sout << "CUFFT error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUresult ERROR ***************/
inline bool is_error(CUresult stat) { return stat != CUDA_SUCCESS; }

inline std::string build_nvidia_error_msg(CUresult stat) {
  std::ostringstream sout;
  sout << "CU error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/**************** NCCL ERROR ****************/
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_nvidia_error_msg(ncclResult_t nccl_result) {
  std::ostringstream sout;
  sout << "NCCL error(" << nccl_result << "), "
       << phi::dynload::ncclGetErrorString(nccl_result) << ". ";
  if (errno == ENOSPC || errno == EAGAIN) {
    std::string detail(strerror(errno));
    detail += "\nPlease try one of the following solutions:";
    detail += "\n1. export NCCL_SHM_DISABLE=1;";
    detail += "\n2. export NCCL_P2P_LEVEL=SYS;";
    detail +=
        "\n3. Increase shared memory by setting the -shm-size "
        "option when starting docker container, e.g., setting "
        " -shm-size=2g.\n";
    sout << " Detail: " + detail;
  }
  sout << GetExternalErrorMsg(nccl_result);
  return sout.str();
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

#define PADDLE_ENFORCE_GPU_SUCCESS(COND)                         \
  do {                                                           \
    auto __cond__ = (COND);                                      \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);             \
    constexpr auto __success_type__ =                            \
        ::paddle::platform::details::ExternalApiType<            \
            __CUDA_STATUS_TYPE__>::kSuccess;                     \
    if (UNLIKELY(__cond__ != __success_type__)) {                \
      auto __summary__ = phi::errors::External(                  \
          ::paddle::platform::build_nvidia_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                     \
    }                                                            \
  } while (0)

#define PADDLE_ENFORCE_CUDA_LAUNCH_SUCCESS(OP)                                 \
  do {                                                                         \
    auto res = cudaGetLastError();                                             \
    if (UNLIKELY(res != cudaSuccess)) {                                        \
      auto msg = ::paddle::platform::build_nvidia_error_msg(res);              \
      PADDLE_THROW(platform::errors::Fatal("CUDA error after kernel (%s): %s", \
                                           OP, msg));                          \
    }                                                                          \
  } while (0)

inline void retry_sleep(unsigned milliseconds) {
#ifdef _WIN32
  Sleep(milliseconds);
#else
  if (milliseconds < 1000) {
    // usleep argument must be less than 1,000,000. Reference:
    // https://pubs.opengroup.org/onlinepubs/7908799/xsh/usleep.html
    usleep(milliseconds * 1000);
  } else {
    // clip to sleep in seconds because we can not and don't have to
    // sleep for exact milliseconds
    sleep(milliseconds / 1000);
  }
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::paddle::platform::details::ExternalApiType<                   \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      paddle::platform::retry_sleep(FLAGS_gpu_allocator_retry_time);    \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = phi::errors::External(                         \
          ::paddle::platform::build_nvidia_error_msg(__cond__));        \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_CUDA

/**************************************************************************/
/***************************** HIP ERROR **********************************/
#ifdef PADDLE_WITH_HIP

/***** HIP ERROR *****/
inline bool is_error(hipError_t e) { return e != hipSuccess; }

inline std::string build_rocm_error_msg(hipError_t e) {
  std::ostringstream sout;
  sout << " Hip error(" << e << "), " << hipGetErrorString(e) << ".";
  return sout.str();
}

/***** HIPRAND ERROR *****/
inline bool is_error(hiprandStatus_t stat) {
  return stat != HIPRAND_STATUS_SUCCESS;
}

inline const char* hiprandGetErrorString(hiprandStatus_t stat) {
  switch (stat) {
    case HIPRAND_STATUS_SUCCESS:
      return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH:
      return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED:
      return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED:
      return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR:
      return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE:
      return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE:
      return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE:
      return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED:
      return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH:
      return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR:
      return "HIPRAND_STATUS_INTERNAL_ERROR";
    case HIPRAND_STATUS_NOT_IMPLEMENTED:
      return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    default:
      return "Unknown hiprand status";
  }
}

inline std::string build_rocm_error_msg(hiprandStatus_t stat) {
  std::string msg(" Hiprand error, ");
  return msg + hiprandGetErrorString(stat) + " ";
}

/***** MIOPEN ERROR *****/
inline bool is_error(miopenStatus_t stat) {
  return stat != miopenStatusSuccess;
}

inline std::string build_rocm_error_msg(miopenStatus_t stat) {
  std::string msg(" Miopen error, ");
  return msg + phi::dynload::miopenGetErrorString(stat) + " ";
}

/***** ROCBLAS ERROR *****/
inline bool is_error(rocblas_status stat) {
  return stat != rocblas_status_success;
}

inline const char* rocblasGetErrorString(rocblas_status stat) {
  switch (stat) {
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return "Unknown cublas status";
  }
}

inline std::string build_rocm_error_msg(rocblas_status stat) {
  std::string msg(" Rocblas error, ");
  return msg + rocblasGetErrorString(stat) + " ";
}

/****** RCCL ERROR ******/
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_rocm_error_msg(ncclResult_t nccl_result) {
  std::string msg(" Rccl error, ");
  return msg + phi::dynload::ncclGetErrorString(nccl_result) + " ";
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

/***** HIPFFT ERROR *****/
inline bool is_error(hipfftResult_t stat) { return stat != HIPFFT_SUCCESS; }

inline std::string build_rocm_error_msg(hipfftResult_t stat) {
  std::string msg(" HIPFFT error, ");
  return msg + phi::dynload::hipfftGetErrorString(stat) + " ";
}

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

DEFINE_EXTERNAL_API_TYPE(hipError_t, hipSuccess);
DEFINE_EXTERNAL_API_TYPE(hiprandStatus_t, HIPRAND_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(miopenStatus_t, miopenStatusSuccess);
DEFINE_EXTERNAL_API_TYPE(rocblas_status, rocblas_status_success);
DEFINE_EXTERNAL_API_TYPE(hipfftResult_t, HIPFFT_SUCCESS);

#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

#define PADDLE_ENFORCE_GPU_SUCCESS(COND)                       \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);           \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::ExternalApiType<          \
            __CUDA_STATUS_TYPE__>::kSuccess;                   \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      auto __summary__ = phi::errors::External(                \
          ::paddle::platform::build_rocm_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                   \
    }                                                          \
  } while (0)

inline void retry_sleep(unsigned millisecond) {
#ifdef _WIN32
  Sleep(millisecond);
#else
  sleep(millisecond);
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::paddle::platform::details::ExternalApiType<                   \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      ::paddle::platform::retry_sleep(FLAGS_gpu_allocator_retry_time);  \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = phi::errors::External(                         \
          ::paddle::platform::build_rocm_error_msg(__cond__));          \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_HIP

}  // namespace platform
}  // namespace paddle
