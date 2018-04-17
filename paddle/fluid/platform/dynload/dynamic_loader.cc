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

#include <dlfcn.h>

#include <memory>
#include <mutex>  // NOLINT
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/dynload/cupti_lib_path.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_string(cudnn_dir, "",
              "Specify path for loading libcudnn.so. For instance, "
              "/usr/local/cudnn/lib. If empty [default], dlopen "
              "will search cudnn from LD_LIBRARY_PATH");

DEFINE_string(cuda_dir, "",
              "Specify path for loading cuda library, such as libcublas, "
              "libcurand. For instance, /usr/local/cuda/lib64. If default, "
              "dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(warpctc_dir, "", "Specify path for loading libwarpctc.so.");

DEFINE_string(lapack_dir, "", "Specify path for loading liblapack.so.");

DEFINE_string(nccl_dir, "",
              "Specify path for loading nccl library, such as libcublas, "
              "libcurand. For instance, /usr/local/cuda/lib64. If default, "
              "dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(cupti_dir, "", "Specify path for loading cupti.so.");

DEFINE_string(
    tensorrt_dir, "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

namespace paddle {
namespace platform {
namespace dynload {
static constexpr char cupti_lib_path[] = CUPTI_LIB_PATH;

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

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);

// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(join("/usr/local/cuda/lib/", dso_path).c_str(), dynload_flags);
    if (nullptr == dso_handle) {
      if (dso_path == "libcudnn.dylib") {
        LOG(WARNING) << "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n "
                        "For instance, sudo tar -xzf "
                        "cudnn-7.5-osx-x64-v5.0-ga.tgz -C /usr/local \n sudo "
                        "chmod a+r /usr/local/cuda/include/cudnn.h "
                        "/usr/local/cuda/lib/libcudnn*";
      }
    }
  }
#endif

  return dso_handle;
}

static inline void* GetDsoHandleFromSearchPath(const std::string& search_root,
                                               const std::string& dso_name,
                                               bool throw_on_error = true) {
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
  void* dso_handle = nullptr;

  std::string dlPath = dso_name;
  if (search_root.empty()) {
    dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
  } else {
    // search xxx.so from custom path
    dlPath = join(search_root, dso_name);
    dso_handle = dlopen(dlPath.c_str(), dynload_flags);
    // if not found, search from default path
    if (nullptr == dso_handle) {
      LOG(WARNING) << "Failed to find dynamic library: " << dlPath << " ("
                   << dlerror() << ")";
      dlPath = dso_name;
      dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
    }
  }
  auto error_msg =
      "Failed to find dynamic library: %s ( %s ) \n Please specify "
      "its path correctly using following ways: \n Method. set "
      "environment variable LD_LIBRARY_PATH on Linux or "
      "DYLD_LIBRARY_PATH on Mac OS. \n For instance, issue command: "
      "export LD_LIBRARY_PATH=... \n Note: After Mac OS 10.11, "
      "using the DYLD_LIBRARY_PATH is impossible unless System "
      "Integrity Protection (SIP) is disabled.";
  if (throw_on_error) {
    PADDLE_ENFORCE(nullptr != dso_handle, error_msg, dlPath, dlerror());
  } else if (nullptr == dso_handle) {
    LOG(WARNING) << string::Sprintf(error_msg, dlPath, dlerror());
  }

  return dso_handle;
}

void* GetCublasDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.so");
#endif
}

void* GetCUDNNDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.dylib", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.so", false);
#endif
}

void* GetCUPTIDsoHandle() {
  std::string cupti_path = cupti_lib_path;
  if (!FLAGS_cupti_dir.empty()) {
    cupti_path = FLAGS_cupti_dir;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(cupti_path, "libcupti.dylib", false);
#else
  return GetDsoHandleFromSearchPath(cupti_path, "libcupti.so", false);
#endif
}

void* GetCurandDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.so");
#endif
}

void* GetWarpCTCDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_warpctc_dir, "libwarpctc.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_warpctc_dir, "libwarpctc.so");
#endif
}

void* GetLapackDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapacke.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapacke.so");
#endif
}

void* GetNCCLDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.so");
#endif
}

void* GetTensorRtDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.so");
#endif
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
