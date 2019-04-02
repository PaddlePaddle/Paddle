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
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/string/split.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/cupti.h"
#endif
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/piece.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "dgc/dgc.h"
#endif

DEFINE_int32(paddle_num_threads, 1,
             "Number of threads for each paddle instance.");
DEFINE_int32(multiple_of_cupti_buffer_size, 1,
             "Multiple of the CUPTI device buffer size. If the timestamps have "
             "been dropped when you are profiling, try increasing this value.");

namespace paddle {
namespace framework {

std::once_flag gflags_init_flag;
std::once_flag p2p_init_flag;

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
std::once_flag dgc_init_flag;
#endif

void InitGflags(std::vector<std::string> argv) {
  std::call_once(gflags_init_flag, [&]() {
    FLAGS_logtostderr = true;
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
          platform::CUDADeviceGuard guard(devices[i]);
          cudaDeviceEnablePeerAccess(devices[j], 0);
        }
      }
    }
  });
#endif
}

void InitCupti() {
#ifdef PADDLE_WITH_CUPTI
  if (FLAGS_multiple_of_cupti_buffer_size == 1) return;
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
#define MULTIPLY_ATTR_VALUE(attr)                                 \
  {                                                               \
    PADDLE_ENFORCE(!platform::dynload::cuptiActivityGetAttribute( \
        attr, &attrValueSize, &attrValue));                       \
    attrValue *= FLAGS_multiple_of_cupti_buffer_size;             \
    LOG(WARNING) << "Set " #attr " " << attrValue << " byte";     \
    PADDLE_ENFORCE(!platform::dynload::cuptiActivitySetAttribute( \
        attr, &attrValueSize, &attrValue));                       \
  }
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE);
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP);
#if CUDA_VERSION >= 9000
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE);
#endif
#undef MULTIPLY_ATTR_VALUE
#endif
}

void InitDevices(bool init_p2p) {
  // CUPTI attribute should be set before any CUDA context is created (see CUPTI
  // documentation about CUpti_ActivityAttribute).
  InitCupti();
  /*Init all available devices by default */
  std::vector<int> devices;
#ifdef PADDLE_WITH_CUDA
  try {
    // use user specified GPUs in single-node multi-process mode.
    devices = platform::GetSelectedDevices();
  } catch (const std::exception &exp) {
    LOG(WARNING) << "Compiled with WITH_GPU, but no GPU found in runtime.";
  }
#endif
  InitDevices(init_p2p, devices);
}

void InitDevices(bool init_p2p, const std::vector<int> devices) {
  std::vector<platform::Place> places;

  for (size_t i = 0; i < devices.size(); ++i) {
    // In multi process multi gpu mode, we may have gpuid = 7
    // but count = 1.
    if (devices[i] < 0) {
      LOG(WARNING) << "Invalid devices id.";
      continue;
    }

    places.emplace_back(platform::CUDAPlace(devices[i]));
  }
  if (init_p2p) {
    InitP2P(devices);
  }
  places.emplace_back(platform::CPUPlace());
  platform::DeviceContextPool::Init(places);
  platform::DeviceTemporaryAllocator::Init();

#ifndef PADDLE_WITH_MKLDNN
  platform::SetNumThreads(FLAGS_paddle_num_threads);
#endif

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__OSX__)
  if (platform::MayIUse(platform::avx)) {
#ifndef __AVX__
    LOG(WARNING) << "AVX is available, Please re-compile on local machine";
#endif
  }

// Throw some informations when CPU instructions mismatch.
#define AVX_GUIDE(compiletime, runtime)                                     \
  LOG(FATAL)                                                                \
      << "This version is compiled on higher instruction(" #compiletime     \
         ") system, you may encounter illegal instruction error running on" \
         " your local CPU machine. Please reinstall the " #runtime          \
         " version or compile from source code."

#ifdef __AVX512F__
  if (!platform::MayIUse(platform::avx512f)) {
    if (platform::MayIUse(platform::avx2)) {
      AVX_GUIDE(AVX512, AVX2);
    } else if (platform::MayIUse(platform::avx)) {
      AVX_GUIDE(AVX512, AVX);
    } else {
      AVX_GUIDE(AVX512, NonAVX);
    }
  }
#endif

#ifdef __AVX2__
  if (!platform::MayIUse(platform::avx2)) {
    if (platform::MayIUse(platform::avx)) {
      AVX_GUIDE(AVX2, AVX);
    } else {
      AVX_GUIDE(AVX2, NonAVX);
    }
  }
#endif

#ifdef __AVX__
  if (!platform::MayIUse(platform::avx)) {
    AVX_GUIDE(AVX, NonAVX);
  }
#endif
#undef AVX_GUIDE

#endif
}

void InitGLOG(const std::string &prog_name) {
  // glog will not hold the ARGV[0] inside.
  // Use strdup to alloc a new string.
  google::InitGoogleLogging(strdup(prog_name.c_str()));
#ifndef _WIN32
  google::InstallFailureSignalHandler();
#endif
}

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
void InitDGC() {
  std::call_once(dgc_init_flag, []() {
    PADDLE_ENFORCE(paddle::communication::dgc::dynloadNcclLib());
  });
}
#else
void InitDGC() {}
#endif

}  // namespace framework
}  // namespace paddle
