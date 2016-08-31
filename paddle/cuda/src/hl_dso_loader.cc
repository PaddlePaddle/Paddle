/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "hl_dso_loader.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/CommandLineParser.h"

P_DEFINE_string(cudnn_dir, "",
                "Specify path for loading libcudnn.so. For instance, "
                "/usr/local/cudnn/lib64. If empty [default], dlopen will search "
                "cudnn from LD_LIBRARY_PATH");

P_DEFINE_string(cuda_dir, "",
                "Specify path for loading cuda library, such as libcublas, "
                "libcurand. For instance, /usr/local/cuda/lib64. "
                "(Note: libcudart can not be specified by cuda_dir, since some "
                "build-in function in cudart already ran before main entry). "
                "If empty [default], dlopen will search cuda from LD_LIBRARY_PATH");

static inline std::string join(const std::string& part1, const std::string& part2) {
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

static inline void GetDsoHandleWithSearchPath(
        const std::string& search_root,
        const std::string& dso_path,
        void** dso_handle) {
    int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
    *dso_handle = nullptr;

    std::string dlPath = dso_path;
    if (search_root.empty()) {
        // default search xxx.so from LD_LIBRARY_PATH
        *dso_handle = dlopen(dlPath.c_str(), dynload_flags);
    } else {
        // search xxx.so from custom path
        dlPath = join(search_root, dso_path);
        *dso_handle = dlopen(dlPath.c_str(), dynload_flags);
        // then, search xxx.so from LD_LIBRARY_PATH
        if (nullptr == *dso_handle) {
            *dso_handle = dlopen(dso_path.c_str(), dynload_flags);
        }
    }

    CHECK(nullptr != *dso_handle)
      << "For Gpu version of PaddlePaddle, it couldn't find CUDA library: "
      << dlPath.c_str() << " Please make sure you already specify its path."
      << "Note: for training data on Cpu using Gpu version of PaddlePaddle,"
      << "you must specify libcudart.so via LD_LIBRARY_PATH.";
}

void GetCublasDsoHandle(void** dso_handle) {
    GetDsoHandleWithSearchPath(FLAGS_cuda_dir, "libcublas.so", dso_handle);
}

void GetCudnnDsoHandle(void** dso_handle) {
    GetDsoHandleWithSearchPath(FLAGS_cudnn_dir, "libcudnn.so", dso_handle);
}

void GetCudartDsoHandle(void** dso_handle) {
    GetDsoHandleWithSearchPath("", "libcudart.so", dso_handle);
}

void GetCurandDsoHandle(void** dso_handle) {
    GetDsoHandleWithSearchPath(FLAGS_cuda_dir, "libcurand.so", dso_handle);
}
