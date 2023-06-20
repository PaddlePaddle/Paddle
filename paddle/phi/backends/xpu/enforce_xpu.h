/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/enforce.h"
#ifdef PADDLE_WITH_XPU_BKCL
#include "xpu/bkcl.h"
#endif
#include <sys/syscall.h>
#include <sys/types.h>
#define gettid() syscall(__NR_gettid)

namespace phi {
namespace backends {
namespace xpu {

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
      return "unknown error";
  }
}

#ifdef PADDLE_WITH_XPU_BKCL
inline const char* bkclGetErrorString(BKCLResult_t stat) {
  switch (stat) {
    case BKCL_SUCCESS:
      return "BKCL_SUCCESS";
    case BKCL_INVALID_ARGUMENT:
      return "BKCL_INVALID_ARGUMENT";
    case BKCL_RUNTIME_ERROR:
      return "BKCL_RUNTIME_ERROR";
    case BKCL_SYSTEM_ERROR:
      return "BKCL_SYSTEM_ERROR";
    case BKCL_INTERNAL_ERROR:
      return "BKCL_INTERNAL_ERROR";
    default:
      return "Unknown BKCL status";
  }
}
#endif

inline const char* xdnnGetErrorString(int stat) {
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
  std::string msg("XPU Error <" + std::to_string(stat) + ">, ");
  return msg + xpuGetErrorString(stat) + " ";
}

#ifdef PADDLE_WITH_XPU_BKCL
inline std::string build_xpu_error_msg(BKCLResult_t stat) {
  std::string msg("BKCL Error, ");
  return msg + bkclGetErrorString(stat) + " ";
}
#endif

inline std::string build_xpu_xdnn_error_msg(int stat, std::string msg) {
  return msg + " XDNN Error, " + xdnnGetErrorString(stat) + " ";
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

DEFINE_EXTERNAL_API_TYPE(int, XPU_SUCCESS);
#ifdef PADDLE_WITH_XPU_BKCL
DEFINE_EXTERNAL_API_TYPE(BKCLResult_t, BKCL_SUCCESS);
#endif

#undef DEFINE_EXTERNAL_API_TYPE

}  // namespace details

#define PADDLE_ENFORCE_XPU_SUCCESS(COND)                        \
  do {                                                          \
    auto __cond__ = (COND);                                     \
    using __XPU_STATUS_TYPE__ = decltype(__cond__);             \
    constexpr auto __success_type__ =                           \
        ::phi::backends::xpu::details::ExternalApiType<         \
            __XPU_STATUS_TYPE__>::kSuccess;                     \
    if (UNLIKELY(__cond__ != __success_type__)) {               \
      auto __summary__ = phi::errors::External(                 \
          ::phi::backends::xpu::build_xpu_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                    \
    }                                                           \
  } while (0)

#define PADDLE_ENFORCE_XDNN_SUCCESS(COND, MSG)                            \
  do {                                                                    \
    auto __cond__ = (COND);                                               \
    if (UNLIKELY(__cond__ != baidu::xpu::api::Error_t::SUCCESS)) {        \
      auto __summary__ = phi::errors::External(                           \
          ::phi::backends::xpu::build_xpu_xdnn_error_msg(__cond__, MSG)); \
      __THROW_ERROR_INTERNAL__(__summary__);                              \
    }                                                                     \
  } while (0)

#define PADDLE_ENFORCE_XDNN_NOT_NULL(ptr)                    \
  do {                                                       \
    if (UNLIKELY(ptr == nullptr)) {                          \
      auto __summary__ = phi::errors::External(              \
          ::phi::backends::xpu::build_xpu_xdnn_error_msg(    \
              baidu::xpu::api::Error_t::NO_ENOUGH_WORKSPACE, \
              "XPU memory is not enough"));                  \
      __THROW_ERROR_INTERNAL__(__summary__);                 \
    }                                                        \
  } while (0)
#define PADDLE_ENFORCE_XRE_SUCCESS(COND)                         \
  do {                                                           \
    auto __cond__ = (COND);                                      \
    auto xre_msg = xpu_strerror(__cond__);                       \
    if (UNLIKELY(__cond__ != XPU_SUCCESS)) {                     \
      auto __summary__ =                                         \
          phi::errors::External("XPU Runtime Error: ", xre_msg); \
      __THROW_ERROR_INTERNAL__(__summary__);                     \
    }                                                            \
  } while (0)

#define XPU_CHECK_SIZE 4
inline int xpu_mem_check(void* dev_ptr, uint64_t size) {
#ifdef XPU_CHECK_SIZE
  int ret = xpu_set_device(0);
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_set_device(0) failed(" << ret << ")!";
  uint8_t* host_ptr = new uint8_t[2 * (size + XPU_CHECK_SIZE)];
  memset(host_ptr, 1, size + XPU_CHECK_SIZE);
  // usleep(50000);
  ret = xpu_wait();
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_wait(0) failed(" << ret << ")!";
  ret = xpu_memcpy(host_ptr,
                   dev_ptr,
                   size + XPU_CHECK_SIZE,
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_memcpy(d2h) failed(" << ret << ")!";
  // usleep(50000);
  ret = xpu_wait();
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_wait(0) failed(" << ret << ")!";
  std::stringstream os;
  for (int i = 0; i < 5 && i < size; i++) {
    uint8_t val = host_ptr[i];
    char str[255];
    sprintf(str, "0x%X", val);  // NOLINT
    os << str << ",";
  }
  os << " ";
  int sum = 0;
  for (int i = 0; i < XPU_CHECK_SIZE; i++) {
    uint8_t val = host_ptr[size + i];
    sum += static_cast<int32_t>(val);
    char str[255];
    sprintf(str, "0x%X", val);  // NOLINT
    os << str << ",";
  }
  CHECK_EQ(sum, 0) << "tid=" << gettid() << ": xpu_mem_check(sum=" << sum
                   << ") failed! dev_ptr=" << dev_ptr << " size=" << size
                   << " data=[" << os.str() << "].";
  // if(sum !=0) {
  //   LOG(INFO) << "tid=" << gettid() << ": xpu_mem_check(sum=" << sum << ")
  //   failed! dev_ptr=" << dev_ptr << " size=" << size <<  " data=[" <<
  //   os.str() << "].";
  // }
  delete[] host_ptr;
  return XPU_SUCCESS;
#else
  return XPU_SUCCESS;
#endif
}

inline int xpu_mem_alloc(void** p_dev_ptr,
                         uint64_t size,
                         XPUMemoryKind kind = XPU_MEM_MAIN) {
#ifdef XPU_CHECK_SIZE
  int ret = xpu_set_device(0);
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_set_device(0) failed(" << ret << ")!";
  ret = xpu_malloc(p_dev_ptr, size + XPU_CHECK_SIZE, kind);
  if (ret != XPU_SUCCESS) {
    LOG(WARNING) << "xpu memory malloc(" << size << ") failed, try again";
    xpu_wait();
    ret = xpu_malloc(p_dev_ptr, size + XPU_CHECK_SIZE);
  }
  CHECK_EQ(ret, XPU_SUCCESS) << "tid=" << gettid() << ": xpu_malloc(" << size
                             << ") failed(" << ret << ")! no enough memory!";
  LOG(INFO) << "tid=" << gettid() << ": xpu_mem_alloc(size=" << size
            << ", dev_ptr=" << *p_dev_ptr << ")";
  uint8_t* host_ptr = new uint8_t[size + XPU_CHECK_SIZE];
  memset(host_ptr, 0, size + XPU_CHECK_SIZE);
  // usleep(50000);
  ret = xpu_wait();
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_wait(0) failed(" << ret << ")!";
  ret = xpu_memcpy(*p_dev_ptr,
                   host_ptr,
                   size + XPU_CHECK_SIZE,
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_memcpy(h2d) failed(" << ret << ")!";
  // usleep(50000);
  ret = xpu_wait();
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_wait(0) failed(" << ret << ")!";
  delete[] host_ptr;
  return xpu_mem_check(*p_dev_ptr, size);
#else
  int ret = xpu_malloc(p_dev_ptr, size, kind);
  // LOG(INFO) << "tid=" << gettid() <<  ": xpu_mem_alloc(size=" << size << ",
  // dev_ptr=" << *p_dev_ptr << ")";
  return ret;
#endif
}

inline int xpu_mem_free(void* dev_ptr, uint64_t size) {
#ifdef XPU_CHECK_SIZE
  int ret = xpu_set_device(0);
  CHECK_EQ(ret, XPU_SUCCESS)
      << "tid=" << gettid() << ": xpu_set_device(0) failed(" << ret << ")!";
  LOG(INFO) << "tid=" << gettid() << ": xpu_mem_free(size=" << size
            << ", dev_ptr=" << dev_ptr << ")";
  xpu_mem_check(dev_ptr, size);
  return xpu_free(dev_ptr);
#else
  // LOG(INFO) << "tid=" << gettid() <<  ": xpu_mem_free(size=" << size << ",
  // dev_ptr=" << dev_ptr << ")";
  return xpu_free(dev_ptr);
#endif
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
