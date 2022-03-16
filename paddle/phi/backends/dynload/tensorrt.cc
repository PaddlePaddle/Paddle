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

#include "paddle/phi/backends/dynload/tensorrt.h"
#include <string>

namespace phi {
namespace dynload {

std::once_flag tensorrt_dso_flag;
void* tensorrt_dso_handle;

std::once_flag tensorrt_plugin_dso_flag;
void* tensorrt_plugin_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

TENSORRT_RAND_ROUTINE_EACH_POINTER(DEFINE_WRAP);
TENSORRT_RAND_ROUTINE_EACH_NON_POINTER(DEFINE_WRAP);
TENSORRT_PLUGIN_RAND_ROUTINE_EACH(DEFINE_WRAP);

void* GetDsoHandle(const std::string& dso_name) {
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32

  void* dso_handle = dlopen(dso_name.c_str(), dynload_flags);

  if (nullptr == dso_handle) {
    auto error_msg =
        "You are using Paddle compiled with TensorRT, but TensorRT dynamic "
        "library is not found. Ignore this if TensorRT is not needed.\n"
        "The TensorRT that Paddle depends on is not configured correctly.\n"
        "  Suggestions:\n"
        "  1. Check if the TensorRT is installed correctly and its version"
        " is matched with paddlepaddle you installed.\n"
        "  2. Configure environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n"
        "  - Windows: set PATH by `set PATH=XXX;%PATH%`\n"
        "  - Mac: set  DYLD_LIBRARY_PATH by `export DYLD_LIBRARY_PATH=...`\n";
    LOG(WARNING) << error_msg;
  }
  return dso_handle;
}

void* GetTensorRtHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string dso_name = "libnvinfer.dylib";
#elif defined(_WIN32)
  std::string dso_name = "nvinfer.dll";
#else
  std::string dso_name = "libnvinfer.so";
#endif
  return GetDsoHandle(dso_name);
}

void* GetTensorRtPluginHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string dso_name = "libnvinfer_plugin.dylib";
#elif defined(_WIN32)
  std::string dso_name = "nvinfer_plugin.dll";
#else
  std::string dso_name = "libnvinfer_plugin.so";
#endif
  return GetDsoHandle(dso_name);
}

}  // namespace dynload
}  // namespace phi
