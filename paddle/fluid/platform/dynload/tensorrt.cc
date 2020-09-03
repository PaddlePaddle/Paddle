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

#include "paddle/fluid/platform/dynload/tensorrt.h"
#include <string>

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag tensorrt_dso_flag;
void* tensorrt_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

TENSORRT_RAND_ROUTINE_EACH(DEFINE_WRAP);

void* GetTensorRtHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string dso_name = "libnvinfer.dylib";
#elif defined(_WIN32)
  std::string dso_name = "nvinfer.dll";
#else
  std::string dso_name = "libnvinfer.so";
#endif

#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32

  void* dso_handle = dlopen(dso_name.c_str(), dynload_flags);

  if (nullptr == dso_handle) {
    auto error_msg =
        "TensorRT dynamic library (%s) that Paddle depends on is not "
        "configured correctly. (error code is %s)\n"
        "  Suggestions:\n"
        "  1. Check if TensorRT "
        "is installed correctly and its version is matched with paddlepaddle "
        "you installed.\n"
        "  2. Configure TensorRT dynamic library environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n"
        "  - Windows: set PATH by `set PATH=XXX;%PATH%`\n"
        "  - Mac: set  DYLD_LIBRARY_PATH by `export DYLD_LIBRARY_PATH=...` "
        "[Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is "
        "impossible unless System Integrity Protection (SIP) is disabled.]";
#if !defined(_WIN32)
    auto errorno = dlerror();
#else
    auto errorno = GetLastError();
#endif  // !_WIN32
    std::cerr << string::Sprintf(error_msg, dso_name, errorno);
  }

  return dso_handle;
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
