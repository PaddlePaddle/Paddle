// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <glog/logging.h>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/lite/cl/cl2_header.h"
#include "paddle/fluid/lite/cl/cl_extension.h"
#include "paddle/fluid/lite/cl/helper.h"
#include "paddle/fluid/lite/utils/blob_map.h"
#include "paddle/fluid/lite/utils/string.h"

// We borrowed some ideas about managing OpenCL code building from MACE
// project(https://github.com/XiaoMi/mace), great thanks to the MACE team.
namespace paddle {
namespace lite {

enum class ClGpuType {
  QUALCOMM_ADRENO,
  MALI,
  UNK,
};

enum class ClVersion {
  VER_1_0,
  VER_1_1,
  VER_1_2,
  VER_2_0,
  UNK,
};

enum class ClGpuPerfHint {
  DEFAULT = 0,
  LOW,
  NORMAL,
  HIGH,
};

enum class ClGpuPriorityHint {
  DEFAULT = 0,
  LOW,
  NORMAL,
  HIGH,
};

const char *kOpenCLPlatformInfoKey = "__opencl_platform_info__";

ClGpuType ParseGPUType(const std::string &device_name) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";

  if (device_name == kQualcommAdrenoGPUStr) {
    return ClGpuType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
    return ClGpuType::MALI;
  } else {
    return ClGpuType::UNK;
  }
}

ClVersion ParseDeviceVersion(const std::string &device_version) {
  // OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <vendor-specific information>
  auto words = Split(device_version, ' ');
  if (words[1] == "2.0") {
    return ClVersion::VER_2_0;
  } else if (words[1] == "1.2") {
    return ClVersion::VER_1_2;
  } else if (words[1] == "1.1") {
    return ClVersion::VER_1_1;
  } else if (words[1] == "1.0") {
    return ClVersion::VER_1_0;
  } else {
    LOG(ERROR) << "Do not support OpenCL version: " << words[1];
    return ClVersion::UNK;
  }
}

void GetAdrenoContextProperties(std::vector<cl_context_properties> *properties,
                                ClGpuPerfHint gpu_perf_hint,
                                ClGpuPriorityHint gpu_priority_hint) {
  CHECK(properties);
  switch (gpu_perf_hint) {
    case ClGpuPerfHint::LOW:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_LOW_QCOM);
      break;
    case ClGpuPerfHint::NORMAL:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_NORMAL_QCOM);
      break;
    case ClGpuPerfHint::HIGH:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  switch (gpu_priority_hint) {
    case ClGpuPriorityHint::LOW:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_LOW_QCOM);
      break;
    case ClGpuPriorityHint::NORMAL:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
      break;
    case ClGpuPriorityHint::HIGH:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  // The properties list should be terminated with 0
  properties->push_back(0);
}

/*
 * The OpenclContext encapsulate all the information and operations related to
 * OpenCL Context to simplify the usage.
 */
class OpenclContext {
 public:
  void Build() {}

  bool CollectPlatform() {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.empty()) {
      LOG(ERROR) << "No OpenCL platforms found";
      return false;
    }

    auto default_platform = all_platforms.front();
    std::stringstream ss;
    ss << default_platform.getInfo<CL_PLATFORM_NAME>() << ", "
       << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
       << default_platform.getInfo<CL_PLATFORM_VERSION>();
    platform_info_ = ss.str();
    VLOG(1) << "Platform found: " << platform_info_;
    return true;
  }

  bool CollectDevice(const cl::Platform &platform) {
    std::vector<cl::Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.empty()) {
      LOG(ERROR) << "No OpenCL devices found";
      return true;
    }

    bool gpu_detected = false;
    device_ = std::make_shared<cl::Device>();
    for (auto &device : all_devices) {
      if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
        *device_ = device;
        gpu_detected = true;

        auto device_name = device.getInfo<CL_DEVICE_NAME>();
        gpu_type_ = ParseGPUType(device_name);

        auto device_version = device.getInfo<CL_DEVICE_VERSION>();
        cl_version_ = ParseDeviceVersion(device_version);
        if (cl_version_ == ClVersion::UNK) return false;

        VLOG(1) << "Using device: " << device_name;
      }
    }

    if (!gpu_detected) {
      LOG(ERROR) << "No GPU device found";
      return false;
    }

    return true;
  }

  bool CreateContext(ClGpuPerfHint perf_hint, ClGpuPriorityHint priority_hint) {
    cl_int err;
    if (gpu_type_ == ClGpuType::QUALCOMM_ADRENO &&
        cl_version_ == ClVersion::VER_2_0) {
      std::vector<cl_context_properties> context_properties;
      context_properties.reserve(5);
      GetAdrenoContextProperties(&context_properties, perf_hint, priority_hint);
      cl_ctx_ = std::shared_ptr<cl::Context>(new cl::Context(
          {*device_}, context_properties.data(), nullptr, nullptr, &err));
    } else {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
      cl_ctx_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
#else
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
#endif
    }
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "Failed to create OpenCL Context: "
                 << OpenclErrorToStr(err);
      return false;
    }
    return true;
  }

  bool CreateCommandQueue() {
    cl_command_queue_properties properties = 0;
    cl_int err;
    cl_command_queue_ = std::make_shared<cl::CommandQueue>(*cl_ctx_, *device_,
                                                           properties, &err);

    if (err != CL_SUCCESS) {
      LOG(ERROR) << "Failed to create OpenCL CommandQueue: "
                 << OpenclErrorToStr(err);
      return false;
    }
    return true;
  }

  bool LoadProgramCache() {
    std::string cached_binary_platform_info;
    if (program_cache_ != nullptr) {
      if (!program_cache_->Load()) {
        LOG(WARNING) << "Load OpenCL cached compiled kernel file failed. "
                     << "Please make sure the storage directory exist "
                     << "and you have Write&Read permission";
      }
      auto platform_info_array =
          this->program_cache_->Find(kOpenCLPlatformInfoKey);
      if (platform_info_array != nullptr) {
        cached_binary_platform_info = *platform_info_array;
        if (cached_binary_platform_info != platform_info_) {
          program_cache_->Clear();
        }
      }
    }

    /*
    if (cached_binary_platform_info != platform_info_) {
      if (precompiled_binary_storage_ == nullptr) {
        VLOG(1) << "There is no precompiled OpenCL binary in"
                   " all OpenCL binary paths.";
      } else {
        if (precompiled_binary_storage_->Load() != 0) {
          LOG(WARNING) << "Load OpenCL precompiled kernel file failed. "
                       << "Please make sure the storage directory exist "
                       << "and you have Write&Read permission";
        }

        auto platform_info_array =
            this->precompiled_binary_storage_->Find(kOpenCLPlatformInfoKey);
        if (platform_info_array != nullptr) {
          precompiled_binary_platform_info_ = std::string(
              platform_info_array->begin(), platform_info_array->end());
        }
      }
    }
     */
  }

  bool CollectOtherInfo() {
    device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                     &device_global_mem_cache_size_);
    device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_compute_units_);
    is_opencl_avaliable_ = true;
  }

  cl::Context *context() { return cl_ctx_.get(); }
  const cl::Device &device() const {
    CHECK(device_);
    return *device_;
  }

  std::vector<uint64_t> GetMaxImage2DSize() {
    size_t max_height, max_width;
    cl_int err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenclErrorToStr(err);
      return {};
    }
    err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenclErrorToStr(err);
      return {};
    }
    return {max_width, max_height};
  }

  uint64_t GetDeviceMaxMemAllocSize() const {
    uint64_t size = 0;
    cl_int err = device_->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenclErrorToStr(err);
      size = 0;
    }
    return size;
  }

  uint64_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) const {
    uint64_t size = 0;
    cl_int err =
        kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &size);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenclErrorToStr(err);
      size = 0;
    }
    return size;
  }

  /*
  uint64_t GetKernelWaveSize(const cl::Kernel &kernel) const {
    uint64_t size = 0;
    cl_int err =
        kernel.getWorkGroupInfo(*device_, CL_KERNEL_WAVE_SIZE_QCOM, &size);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "error: " << OpenCLErrorToString(err);
      size = 0;
    }
    return size;
  }
   */

 private:
  bool GetProgramSourceByName(const std::string &program_name,
                              std::string *source);

  bool BuildProgramFromSource(const std::string &program_name,
                              const std::string &built_program_key,
                              const std::string &build_options_str,
                              cl::Program *program) {
    std::string kernel_source;
    bool status = GetProgramSourceByName(program_name, &kernel_source);
    if (status && !kernel_source.empty()) {
      cl::Program::Sources sources;
      sources.push_back(kernel_source);
      *program = cl::Program(*context(), sources);
      cl_int ret = program->build({*device_}, build_options_str.c_str());
      if (ret != CL_SUCCESS) {
        if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device_) ==
            CL_BUILD_ERROR) {
          std::string build_log =
              program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device_);
          LOG(INFO) << "Program build log: " << build_log;
        }
        return false;
      }

      // Keep built program binary
      size_t device_list_size = 1;
      std::unique_ptr<size_t[]> program_binary_sizes(
          new size_t[device_list_size]);
      cl_int err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARY_SIZES,
                                    sizeof(size_t) * device_list_size,
                                    program_binary_sizes.get(), nullptr);
      if (err != CL_SUCCESS) {
        LOG(ERROR) << "error: " << OpenclErrorToStr(err);
        return false;
      }
      std::unique_ptr<std::unique_ptr<unsigned char[]>[]> program_binaries(
          new std::unique_ptr<unsigned char[]>[device_list_size]);
      for (cl_uint i = 0; i < device_list_size; ++i) {
        program_binaries[i] = std::unique_ptr<unsigned char[]>(
            new unsigned char[program_binary_sizes[i]]);
      }

      err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARIES,
                             sizeof(unsigned char *) * device_list_size,
                             program_binaries.get(), nullptr);
      if (err != CL_SUCCESS) {
        LOG(ERROR) << "error: " << OpenclErrorToStr(err);
        return false;
      }
      std::vector<unsigned char> content(
          reinterpret_cast<unsigned char const *>(program_binaries[0].get()),
          reinterpret_cast<unsigned char const *>(program_binaries[0].get()) +
              program_binary_sizes[0]);

      if (program_cache_) {
        program_cache_->Insert(built_program_key, content);
        // update platform info
        /*
        this->cache_storage_->Insert(
            kOpenCLPlatformInfoKey,
            std::vector<unsigned char>(platform_info_.begin(),
                                       platform_info_.end()));
                                       */
      }

      VLOG(3) << "Program from source: " << built_program_key;
    }
    return true;
  }

  bool BuildProgram(const std::string &program_name,
                    const std::string &built_program_key,
                    const std::string &build_options, cl::Program *program) {
    CHECK(program);

    std::string build_options_str =
        build_options + " -Werror -cl-mad-enable -cl-fast-relaxed-math";
    // Build flow: cache -> precompiled binary -> source
    bool ret =
        BuildProgramFromCache(built_program_key, build_options_str, program);
    if (!ret) {
      ret = BuildProgramFromPrecompiledBinary(built_program_key,
                                              build_options_str, program);
      if (!ret) {
        ret = BuildProgramFromSource(program_name, built_program_key,
                                     build_options_str, program);
      }
    }
    return ret;
  }

  bool BuildProgramFromCache(const std::string &built_program_key,
                             const std::string &build_options_str,
                             cl::Program *program) {
    // Find from binary
    if (!program_cache_) return false;
    auto content = program_cache_->Find(built_program_key);
    if (!content) return false;

    *program = cl::Program(*context(), {device()}, {*content});
    cl_int ret = program->build({device()}, build_options_str.c_str());
    if (ret != CL_SUCCESS) {
      if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
          CL_BUILD_ERROR) {
        std::string build_log =
            program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
        LOG(INFO) << "Program build log: " << build_log;
      }
      LOG(WARNING) << "Build program " << built_program_key
                   << " from Cache failed";
      return false;
    }
    VLOG(3) << "Program from Cache: " << built_program_key;
    return true;
  }

  void BuildKernel(const std::string &program_name,
                   const std::string &kernel_name,
                   const std::set<std::string> &build_options,
                   cl::Kernel *kernel) {
    std::string build_options_str;
    for (auto &option : build_options) {
      build_options_str += " " + option;
    }
    std::string built_program_key = program_name + build_options_str;

    std::lock_guard<std::mutex> lock(program_build_mutex_);
    auto built_program_it = built_program_map_.find(built_program_key);
    cl::Program program;
    if (built_program_it != built_program_map_.end()) {
      program = built_program_it->second;
    } else {
      bool ret = this->BuildProgram(program_name, built_program_key,
                                    build_options_str, &program);
      if (!ret) {
        return;
      }
      built_program_map_.emplace(built_program_key, program);
    }
    cl_int err;
    *kernel = cl::Kernel(program, kernel_name.c_str(), &err);
    CHECK(err == CL_SUCCESS);
  }

 private:
  std::shared_ptr<cl::Context> cl_ctx_;
  std::shared_ptr<cl::Device> device_;
  std::mutex program_build_mutex_;
  std::map<std::string, cl::Program> built_program_map_;
  // program_key, program binary content
  std::shared_ptr<BlobMap> program_cache_;
  std::shared_ptr<cl::CommandQueue> cl_command_queue_;
  uint64_t device_global_mem_cache_size_;
  uint32_t device_compute_units_;
  std::string platform_info_;
  bool is_opencl_avaliable_{false};
  ClGpuType gpu_type_;
  ClVersion cl_version_;
};

class OpenclRuntimeContext {
 public:
 private:
  std::shared_ptr<OpenclContext> opencl_context_;
};

}  // namespace lite
}  // namespace paddle
