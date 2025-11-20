// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xblas/xblas_api.h"
namespace phi {
namespace backends {
namespace xpu {

inline const char* xblasGetErrorString(int stat) {
  switch (stat) {
    case xblasStatus_t::CUBLAS_STATUS_SUCCESS:
      return "XBLAS_STATUS_SUCCESS";
    case xblasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED:
      return "XBLAS_STATUS_NOT_INITIALIZED";
    case xblasStatus_t::CUBLAS_STATUS_ALLOC_FAILED:
      return "XBLAS_STATUS_ALLOC_FAILED";
    case xblasStatus_t::CUBLAS_STATUS_INVALID_VALUE:
      return "XBLAS_STATUS_INVALID_VALUE";
    case xblasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH:
      return "XBLAS_STATUS_ARCH_MISMATCH";
    case xblasStatus_t::CUBLAS_STATUS_MAPPING_ERROR:
      return "XBLAS_STATUS_MAPPING_ERROR";
    case xblasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED:
      return "XBLAS_STATUS_EXECUTION_FAILED";
    case xblasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR:
      return "XBLAS_STATUS_INTERNAL_ERROR";
    case xblasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED:
      return "XBLAS_STATUS_NOT_SUPPORTED";
    case xblasStatus_t::CUBLAS_STATUS_LICENSE_ERROR:
      return "XBLAS_STATUS_LICENSE_ERROR";
    default:
      return "Unknown XBLAS status";
  }
}

#ifdef PADDLE_WITH_XPU_FFT
inline const char* fftGetErrorString(cufftResult_t stat) {
  switch (stat) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
    default:
      return "CUFFT_NOT_SUPPORTED";
  }
}
#endif

// Note: XPU runtime api return int, not XPUError_t
inline const char* xpuGetErrorString(int stat) {
  switch (stat) {
    case XPU_SUCCESS:
      return "Success";
    case XPUERR_INVALID_DEVICE:
      return "Invalid XPU device";
    case XPUERR_UNINIT:
      return "XPU runtime not properly inited";
    case XPUERR_NOMEM:
      return "Device memory not enough";
    case XPUERR_NOCPUMEM:
      return "CPU memory not enough";
    case XPUERR_INVALID_PARAM:
      return "Invalid parameter";
    case XPUERR_NOXPUFUNC:
      return "Cannot get XPU Func";
    case XPUERR_LDSO:
      return "Error loading dynamic library";
    case XPUERR_LDSYM:
      return "Error loading func from dynamic library";
    case XPUERR_SIMULATOR:
      return "Error from XPU Simulator";
    case XPUERR_NOSUPPORT:
      return "Operation not supported";
    case XPUERR_ABNORMAL:
      return "Device abnormal due to previous error";
    case XPUERR_KEXCEPTION:
      return "Exception in kernel execution";
    case XPUERR_TIMEOUT:
      return "Kernel execution timed out";
    case XPUERR_BUSY:
      return "Resource busy";
    case XPUERR_USEAFCLOSE:
      return "Use a stream after closed";
    case XPUERR_UCECC:
      return "Uncorrectable ECC";
    case XPUERR_OVERHEAT:
      return "Overheat";
    case XPUERR_UNEXPECT:
      return "Execution error, reach unexpected control flow";
    case XPUERR_DEVRESET:
      return "Device is being reset, try again later";
    case XPUERR_HWEXCEPTION:
      return "Hardware module exception";
    case XPUERR_HBM_INIT:
      return "Error init HBM";
    case XPUERR_DEVINIT:
      return "Error init device";
    case XPUERR_PEERRESET:
      return "Device is being reset, try again later";
    case XPUERR_MAXDEV:
      return "Device count exceed limit";
    case XPUERR_NOIOC:
      return "Unknown IOCTL command";
    case XPUERR_DMATIMEOUT:
      return "DMA timed out, a reboot maybe needed";
    case XPUERR_DMAABORT:
      return "DMA aborted due to error, possibly wrong address or hardware "
             "state";
    case XPUERR_MCUUNINIT:
      return "Firmware not initialized";
    case XPUERR_OLDFW:
      return "Firmware version too old (<15), please update.";
    case XPUERR_PCIE:
      return "Error in PCIE";
    case XPUERR_FAULT:
      return "Error copy between kernel and user space";
    case XPUERR_INTERRUPTED:
      return "Execution interrupted by user";
    default:
      return "Unknown error";
  }
}

#ifdef PADDLE_WITH_XPU_BKCL
inline const char* bkclGetErrorString(BKCLResult_t stat) {
  switch (stat) {
    case BKCLResult_t::BKCL_SUCCESS:
      return "BKCL_SUCCESS";
    case BKCLResult_t::BKCL_INVALID_ARGUMENT:
      return "BKCL_INVALID_ARGUMENT";
    case BKCLResult_t::BKCL_RUNTIME_ERROR:
      return "BKCL_RUNTIME_ERROR";
    case BKCLResult_t::BKCL_SYSTEM_ERROR:
      return "BKCL_SYSTEM_ERROR";
    case BKCLResult_t::BKCL_INTERNAL_ERROR:
      return "BKCL_INTERNAL_ERROR";
    default:
      return "Unknown BKCL status";
  }
}
#endif

inline const char* xdnnGetErrorString(int stat) {
  // Also reused by xfa and xpudnn apis.
  switch (stat) {
    case baidu::xpu::api::Error_t::SUCCESS:
      return "XDNN_SUCCESS";
    case baidu::xpu::api::Error_t::INVALID_PARAM:
      return "XDNN_INVALID_PARAM";
    case baidu::xpu::api::Error_t::RUNTIME_ERROR:
      return "XDNN_RUNTIME_ERROR";
    case baidu::xpu::api::Error_t::NO_ENOUGH_WORKSPACE:
      return "XDNN_NO_ENOUGH_WORKSPACE";
    case baidu::xpu::api::Error_t::NOT_IMPLEMENT:
      return "XDNN_NOT_IMPLEMENT";
    default:
      return "Unknown XDNN status";
  }
}

inline std::string build_xpu_error_msg(int stat) {
  std::string error_msg = "XPU Error <" + std::to_string(stat) + ">, " +
                          xpuGetErrorString(stat) + " ";
  return error_msg;
}

#ifdef PADDLE_WITH_XPU_BKCL
inline std::string build_bkcl_error_msg(BKCLResult_t stat) {
  std::string error_msg = "BKCL Error <" + std::to_string(stat) + ">, " +
                          bkclGetErrorString(stat) + " ";
  return error_msg;
}
#endif

#ifdef PADDLE_WITH_XPU_FFT
inline std::string build_fft_error_msg(cufftResult_t stat) {
  std::string error_msg = "FFT Error <" + std::to_string(stat) + ">, " +
                          fftGetErrorString(stat) + " ";
  return error_msg;
}
#endif

inline std::string build_xdnn_error_msg(int stat, std::string msg) {
  std::string error_msg = msg + " XDNN Error <" + std::to_string(stat) + ">, " +
                          xdnnGetErrorString(stat) + " ";
  return error_msg;
}

inline std::string build_xblas_error_msg(int stat, std::string msg) {
  std::string error_msg = msg + " XBLAS Error <" + std::to_string(stat) +
                          ">, " + xblasGetErrorString(stat) + " ";
  return error_msg;
}

inline std::string build_runtime_error_msg() {
  auto rt_error_code = cudaGetLastError();
  std::string error_msg = "XPU Runtime Error <" +
                          std::to_string(rt_error_code) + ">, " +
                          std::string(cudaGetErrorString(rt_error_code)) + " ";
  return error_msg;
}

#ifdef PADDLE_WITH_XPU_BKCL
std::string get_bkcl_error_msg(BKCLResult_t stat) {
  std::string error_msg;
  if (stat == BKCLResult_t::BKCL_RUNTIME_ERROR) {
    error_msg = ::phi::backends::xpu::build_bkcl_error_msg(stat) + "\n" +
                ::phi::backends::xpu::build_runtime_error_msg();
  } else {
    error_msg = ::phi::backends::xpu::build_bkcl_error_msg(stat);
  }
  return error_msg;
}
#endif

#ifdef PADDLE_WITH_XPU_FFT
std::string get_fft_error_msg(cufftResult_t stat) {
  std::string error_msg = ::phi::backends::xpu::build_fft_error_msg(stat);
  return error_msg;
}
#endif

std::string get_xpu_error_msg(int stat) {
  std::string error_msg = ::phi::backends::xpu::build_xpu_error_msg(stat);
  return error_msg;
}

std::string get_xdnn_error_msg(int stat, std::string msg) {
  std::string error_msg = "";
  if (stat == baidu::xpu::api::Error_t::RUNTIME_ERROR) {
    error_msg = ::phi::backends::xpu::build_xdnn_error_msg(stat, msg) + "\n" +
                ::phi::backends::xpu::build_runtime_error_msg();
  } else {
    error_msg = ::phi::backends::xpu::build_xdnn_error_msg(stat, msg);
  }
  return error_msg;
}

std::string get_xblas_error_msg(int stat, std::string msg) {
  std::string error_msg = "";
  if (stat == xblasStatus_t::CUBLAS_STATUS_SUCCESS) {
    return error_msg;
  } else {
    if (stat == xblasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR) {
      error_msg = ::phi::backends::xpu::build_xblas_error_msg(stat, msg) +
                  "\n" + ::phi::backends::xpu::build_runtime_error_msg();
    } else {
      error_msg = ::phi::backends::xpu::build_xblas_error_msg(stat, msg);
    }
  }
  return error_msg;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
