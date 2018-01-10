/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <algorithm>
#include <string>

#include "paddle/framework/init.h"
#include "paddle/framework/operator.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/place.h"
#include "paddle/string/piece.h"

namespace paddle {
namespace framework {

std::once_flag gflags_init_flag;

void InitGflags(std::vector<std::string> &argv) {
  std::call_once(gflags_init_flag, [&]() {
    int argc = argv.size();
    char **arr = new char *[argv.size()];
    std::string line;
    for (size_t i = 0; i < argv.size(); i++) {
      arr[i] = &argv[i][0];
      line += argv[i];
      line += ' ';
    }
    google::ParseCommandLineFlags(&argc, &arr, true);
    VLOG(1) << "Init commandline: " << line;
  });
}

void InitDevices() {
  /*Init all avaiable devices by default */
  std::vector<std::pair<platform::Place, framework::LibraryType>> pairs;
  pairs.emplace_back(std::make_pair(platform::CPUPlace(), LibraryType::kPlain));

#ifdef PADDLE_WITH_MKLDNN
  pairs.emplace_back(
      std::make_pair(platform::CPUPlace(), LibraryType::kMKLDNN));
#endif

#ifdef PADDLE_WITH_CUDA
  int count = platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    places.emplace_back(
        std::make_pair(platform::CUDAPlace(i), LibraryType::kPlain));
  }
#else
  LOG(WARNING)
      << "'GPU' is not supported, Please re-compile with WITH_GPU option";
#endif

  platform::DeviceContextPool::Init(pairs);
}

void InitGLOG(const std::string &prog_name) {
  google::InitGoogleLogging(prog_name.c_str());
  google::InstallFailureSignalHandler();
}

}  // namespace framework
}  // namespace paddle
