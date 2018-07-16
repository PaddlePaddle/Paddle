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
#include <string.h>  // for strdup
#include <algorithm>
#include <stdexcept>
#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/piece.h"

namespace paddle {
namespace framework {

std::once_flag gflags_init_flag;
std::once_flag p2p_init_flag;

void InitGflags(std::vector<std::string> argv) {
  std::call_once(gflags_init_flag, [&]() {
    argv.insert(argv.begin(), "dummy");
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

void InitP2P(std::vector<int> devices) {
#ifdef PADDLE_WITH_CUDA
  std::call_once(p2p_init_flag, [&]() {
    int count = devices.size();
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < count; ++j) {
        if (devices[i] == devices[j]) continue;
        int can_acess = -1;
        PADDLE_ENFORCE(
            cudaDeviceCanAccessPeer(&can_acess, devices[i], devices[j]),
            "Failed to test P2P access.");
        if (can_acess != 1) {
          LOG(WARNING) << "Cannot enable P2P access from " << devices[i]
                       << " to " << devices[j];
        } else {
          cudaSetDevice(devices[i]);
          cudaDeviceEnablePeerAccess(devices[j], 0);
        }
      }
    }
  });
#endif
}

#ifdef PADDLE_WITH_CUDA
static void GetCUDAPlacesAndInitP2P(bool init_p2p) {
  /*Init all available devices by default */
  std::vector<int> devices;
  try {
    int count = platform::GetCUDADeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  } catch (const std::exception &exp) {
    LOG(WARNING) << "Compiled with WITH_GPU, but no GPU found in runtime.";
  }
  return GetCUDAPlacesAndInitP2P(init_p2p, devices);
}
static std::vector<platform::Place> GetCUDAPlacesAndInitP2P(
    bool init_p2p, const std::vector<int> &devices) {
  std::vector<platform::Place> places;
  int count = 0;
  try {
    count = platform::GetCUDADeviceCount();
  } catch (const std::exception &exp) {
    LOG(WARNING) << "Compiled with WITH_GPU, but no GPU found in runtime.";
  }

  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] >= count || devices[i] < 0) {
      LOG(WARNING) << "Invalid devices id.";
      continue;
    }
    places.emplace_back(platform::CUDAPlace(devices[i]));
  }
  if (init_p2p) {
    InitP2P(devices);
  }
  return places;
}
#endif

static std::vector<platform::Place> GetCPUPlaces() {
#ifndef PADDLE_WITH_MKLDNN
  platform::SetNumThreads(1);
#endif
  return {platform::CPUPlace()};
}

void InitDevices(bool init_p2p) {
  auto places = GetCPUPlaces();
#ifdef PADDLE_WITH_CUDA
  for (auto &p : GetCUDAPlacesAndInitP2P(init_p2p)) {
    places.emplace_back(p);
  }
#endif
  platform::DeviceContextPool::Init(places);
}

void InitDevices(bool init_p2p, const std::vector<int> &devices) {
  auto places = GetCPUPlaces();
#ifdef PADDLE_WITH_CUDA
  for (auto &p : GetCUDAPlacesAndInitP2P(init_p2p, devices)) {
    places.emplace_back(p);
  }
#endif
  platform::DeviceContextPool::Init(places);
}

void InitGLOG(const std::string &prog_name) {
  // glog will not hold the ARGV[0] inside.
  // Use strdup to alloc a new string.
  google::InitGoogleLogging(strdup(prog_name.c_str()));
  google::InstallFailureSignalHandler();
}

}  // namespace framework
}  // namespace paddle
