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
#include <fstream>
#include <iostream>
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

DECLARE_int32(paddle_num_threads);
DEFINE_int32(multiple_of_cupti_buffer_size, 1,
             "Multiple of the CUPTI device buffer size. If the timestamps have "
             "been dropped when you are profiling, try increasing this value.");

namespace paddle {
namespace framework {

#ifdef _WIN32
#define strdup _strdup
#endif

std::once_flag gflags_init_flag;
std::once_flag glog_init_flag;
std::once_flag p2p_init_flag;
std::once_flag glog_warning_once_flag;

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

#ifndef _WIN32
void SignalHandle(const char *data, int size) {
  auto file_path = string::Sprintf("/tmp/paddle.%d.dump_info", ::getpid());
  try {
    // The signal is coming line by line but we print general guide just once
    std::call_once(glog_warning_once_flag, [&]() {
      LOG(WARNING) << "Warning: PaddlePaddle catches a failure signal, it may "
                      "not work properly\n";
      LOG(WARNING) << "You could check whether you killed PaddlePaddle "
                      "thread/process accidentally or report the case to "
                      "PaddlePaddle\n";
      LOG(WARNING) << "The detail failure signal is:\n\n";
    });

    LOG(WARNING) << std::string(data, size);
    std::ofstream dump_info;
    dump_info.open(file_path, std::ios::app);
    dump_info << std::string(data, size);
    dump_info.close();
  } catch (...) {
  }
}
#endif

void InitGLOG(const std::string &prog_name) {
  std::call_once(glog_init_flag, [&]() {
    // glog will not hold the ARGV[0] inside.
    // Use strdup to alloc a new string.
    google::InitGoogleLogging(strdup(prog_name.c_str()));
#ifndef _WIN32
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter(&SignalHandle);
#endif
  });
}

}  // namespace framework
}  // namespace paddle
