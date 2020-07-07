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
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/dynload/cupti_lib_path.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/port.h"

DEFINE_string(cudnn_dir, "",
              "Specify path for loading libcudnn.so. For instance, "
              "/usr/local/cudnn/lib. If empty [default], dlopen "
              "will search cudnn from LD_LIBRARY_PATH");

DEFINE_string(cuda_dir, "",
              "Specify path for loading cuda library, such as libcublas, "
              "libcurand, libcusolver. For instance, /usr/local/cuda/lib64. "
              "If default, dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(nccl_dir, "",
              "Specify path for loading nccl library, such as libnccl.so. "
              "For instance, /usr/local/cuda/lib64. If default, "
              "dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(cupti_dir, "", "Specify path for loading cupti.so.");

DEFINE_string(
    tensorrt_dir, "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

DEFINE_string(mklml_dir, "", "Specify path for loading libmklml_intel.so.");

DEFINE_string(op_dir, "", "Specify path for loading user-defined op library.");

namespace paddle {
namespace platform {
namespace dynload {

struct PathNode {
  PathNode() {}
  std::string path = "";
};

static constexpr char cupti_lib_path[] = CUPTI_LIB_PATH;

// NOTE: In order to adapt to the default installation path of cuda on linux
static constexpr char linux_cudnn_lib_path[] = "/usr/local/cuda/lib64";

static PathNode s_py_site_pkg_path;

#if defined(_WIN32) && defined(PADDLE_WITH_CUDA)
static constexpr char* win_cublas_lib = "cublas64_" PADDLE_CUDA_BINVER ".dll";
static constexpr char* win_curand_lib = "curand64_" PADDLE_CUDA_BINVER ".dll";
static constexpr char* win_cudnn_lib = "cudnn64_" PADDLE_CUDNN_BINVER ".dll";
static constexpr char* win_cusolver_lib =
    "cusolver64_" PADDLE_CUDA_BINVER ".dll";
#endif

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
  // directory separator
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

void SetPaddleLibPath(const std::string& py_site_pkg_path) {
  s_py_site_pkg_path.path = py_site_pkg_path;
  VLOG(3) << "Set paddle lib path : " << py_site_pkg_path;
}

static inline void* GetDsoHandleFromSpecificPath(const std::string& spec_path,
                                                 const std::string& dso_name,
                                                 int dynload_flags) {
  void* dso_handle = nullptr;
  if (!spec_path.empty()) {
    // search xxx.so from custom path
    VLOG(3) << "Try to find library: " << dso_name
            << " from specific path: " << spec_path;
    std::string dso_path = join(spec_path, dso_name);
    dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  }
  return dso_handle;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";

// TODO(chenweihang): This path is used to search which libs?
// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(join("/usr/local/cuda/lib/", dso_path).c_str(), dynload_flags);
  }
#endif

  return dso_handle;
}

/*
 * We define three priorities for dynamic library search:
 *
 * First: Search for the path specified by the user
 * Second: Search the system default path
 * Third: Search for a special path corresponding to
 *        a specific library to adapt to changes and easy to expand.
 */

static inline void* GetDsoHandleFromSearchPath(
    const std::string& config_path, const std::string& dso_name,
    bool throw_on_error = true,
    const std::vector<std::string>& extra_paths = std::vector<std::string>(),
    const std::string& warning_msg = std::string()) {
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32
  // 1. search in user config path by FLAGS
  void* dso_handle =
      GetDsoHandleFromSpecificPath(config_path, dso_name, dynload_flags);
  // 2. search in system default path
  if (nullptr == dso_handle) {
    dso_handle = GetDsoHandleFromDefaultPath(dso_name, dynload_flags);
  }
  // 3. search in extra paths
  if (nullptr == dso_handle) {
    for (auto path : extra_paths) {
      dso_handle = GetDsoHandleFromSpecificPath(path, dso_name, dynload_flags);
    }
  }

  // 4. [If Failed] logging warning if exists
  if (nullptr == dso_handle && !warning_msg.empty()) {
    LOG(WARNING) << warning_msg;
  }

  // 5. [If Failed] logging or throw error info
  if (nullptr == dso_handle) {
    auto error_msg =
        "Failed to find dynamic library: %s ( %s ) \n"
        "Please specify its path correctly using following ways: \n"
        "  set environment variable LD_LIBRARY_PATH on Linux or "
        "DYLD_LIBRARY_PATH on Mac OS. \n"
        "  For instance, issue command: export LD_LIBRARY_PATH=... \n"
        "  Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is "
        "impossible unless System Integrity Protection (SIP) is disabled.";
#if !defined(_WIN32)
    auto errorno = dlerror();
#else
    auto errorno = GetLastError();
#endif  // !_WIN32
    if (throw_on_error) {
      // NOTE: Special error report case, no need to change its format
      PADDLE_THROW(platform::errors::NotFound(error_msg, dso_name, errorno));
    } else {
      LOG(WARNING) << string::Sprintf(error_msg, dso_name, errorno);
    }
  }

  return dso_handle;
}

void* GetCublasDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_cublas_lib);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.so");
#endif
}

void* GetCUDNNDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string mac_warn_meg(
      "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n "
      "For instance, sudo tar -xzf "
      "cudnn-7.5-osx-x64-v5.0-ga.tgz -C /usr/local \n sudo "
      "chmod a+r /usr/local/cuda/include/cudnn.h "
      "/usr/local/cuda/lib/libcudnn*");
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.dylib", false,
                                    {}, mac_warn_meg);
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, win_cudnn_lib);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.so", false,
                                    {linux_cudnn_lib_path});
#endif
}

void* GetCUPTIDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cupti_dir, "libcupti.dylib", false,
                                    {cupti_lib_path});
#else
  return GetDsoHandleFromSearchPath(FLAGS_cupti_dir, "libcupti.so", false,
                                    {cupti_lib_path});
#endif
}

void* GetCurandDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_curand_lib);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.so");
#endif
}

void* GetCusolverDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_cusolver_lib);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.so");
#endif
}

void* GetNVRTCDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.dylib", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.so", false);
#endif
}

void* GetCUDADsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.dylib", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.so", false);
#endif
}

void* GetWarpCTCDsoHandle() {
  std::string warpctc_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    warpctc_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(warpctc_dir, "warpctc.dll");
#else
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.so");
#endif
}

void* GetNCCLDsoHandle() {
  std::string warning_msg(
      "You may need to install 'nccl2' from NVIDIA official website: "
      "https://developer.nvidia.com/nccl/nccl-download"
      "before install PaddlePaddle.");
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.dylib", true, {},
                                    warning_msg);
#else
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.so", true, {},
                                    warning_msg);
#endif
}

void* GetTensorRtDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "nvinfer.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.so");
#endif
}

void* GetMKLMLDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "mklml.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.so");
#endif
}

void* GetOpDsoHandle(const std::string& dso_name) {
#if defined(__APPLE__) || defined(__OSX__)
  PADDLE_THROW(platform::errors::Unimplemented(
      "Create custom cpp op outside framework do not support Apple."));
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  PADDLE_THROW(platform::errors::Unimplemented(
      "Create custom cpp op outside framework do not support Windows."));
#else
  return GetDsoHandleFromSearchPath(FLAGS_op_dir, dso_name);
#endif
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
