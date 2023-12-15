// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#ifdef CINN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif
#ifdef CINN_WITH_SYCL
#include "paddle/cinn/runtime/sycl/sycl_runtime.h"
#endif

#include <glog/logging.h>

#include <sstream>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace common {

Target::Target(OS o,
               Arch a,
               Language l,
               Bit b,
               const std::vector<Feature> &features,
               const std::vector<Lib> &libs)
    : os(o), arch(a), language(l), bits(b), features(features), libs(libs) {}

bool Target::operator==(const Target &other) const {
  // set SYCLTarget to NVGPUTarget temporary
  if(language == Target::Language::cuda || language == Target::Language::sycl){
    if(other.language==Target::Language::cuda || other.language==Target::Language::sycl){
      return true;
    }
  }

  return os == other.os &&            //
         arch == other.arch &&        //
         language == other.language && //
         bits == other.bits &&        //
         features == other.features;
}

int Target::runtime_arch() const {
  switch (arch) {
    case Arch::Unk:
      return cinn_unk_device;
    case Arch::X86:
      return cinn_x86_device;
    case Arch::ARM:
      return cinn_arm_device;
    default:
      LOG(FATAL) << "Not supported arch";
  }
  return -1;
}

int Target::max_num_threads() const {
  CHECK(arch == Arch::NVGPU)
      << "The target is not NVGPU! Cannot get max number of threads.";
  return 1024;
}

int Target::get_multi_processor_count() const {
  CHECK(arch == Arch::NVGPU)
      << "The target is not NVGPU! Cannot get multi processor count";
  int num_sm = 0;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
#endif
  return num_sm;
}

int Target::get_max_threads_per_sm() const {
  CHECK(arch == Arch::NVGPU)
      << "The target is not NVGPU! Cannot get max threads per stream processor";
  int max_thread = 0;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &max_thread, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, 0);
#endif
  return max_thread;
}

int Target::get_max_blocks_per_sm() const {
  CHECK(arch == Arch::NVGPU)
      << "The target is not NVGPU! Cannot get max blocks per stream processor";
  int max_blocks = 1;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &max_blocks, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, 0);
#endif
  return max_blocks;
}

std::vector<Target::Lib> Target::get_target_libs() const { return libs; }

int Target::get_target_bits() const {
  switch (bits) {
    case Bit::k32:
      return 32;
    case Bit::k64:
      return 64;
    case Bit::Unk:
      return 0;
    default:
      LOG(FATAL) << "Not supported Bit";
  }
  return -1;
}

std::string Target::arch_str() const {
  std::ostringstream oss;
  oss << arch;
  return oss.str();
}

void Target::SetActiveDevices(std::vector<int> deviceIds) {
  if(language != Target::Language::sycl){
    LOG(ERROR) << "set device only supported for sycl backend!";
  }
  SYCLWorkspace::Global()->SetActiveDevices(deviceIds);
}

std::ostream &operator<<(std::ostream &os, const Target &target) {
  os << "Target<";
  switch (target.os) {
    case Target::OS::Linux:
      os << "linux";
      break;
    case Target::OS::Windows:
      os << "windows";
      break;
    case Target::OS::Unk:
      os << "unk";
      break;
  }

  os << ",";
  os << target.arch;
  os << ",";
  //if(target.language==Target::Language::sycl){
  //  os <<"cuda";
 // }else{
    os << target.language;
  //}
  os << ",";

  switch (target.bits) {
    case Target::Bit::k32:
      os << "32";
      break;
    case Target::Bit::k64:
      os << "64";
      break;
    case Target::Bit::Unk:
      os << "unk";
      break;
  }
  os << ">";

  return os;
}

std::ostream &operator<<(std::ostream &os, Target::Arch arch) {
  switch (arch) {
    case Target::Arch::Unk:
      os << "Unk";
      break;
    case Target::Arch::X86:
      os << "X86";
      break;
    case Target::Arch::ARM:
      os << "ARM";
      break;
    case Target::Arch::NVGPU:
      os << "NVGPU";
      break;
    case Target::Arch::AMDGPU:
      os << "AMDGPU";
      break;
    case Target::Arch::IntelGPU:
      os << "IntelGPU";
      break;
    case Target::Arch::HygonDCU:
      os << "HygonDCU";
      break;
    case Target::Arch::CambrianMLU:
      os << "CambrianMLU";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, Target::Language language) {
  switch(language){
    case Target::Language::Unk:
      os << "Unk";
      break;
    case Target::Language::llvm:
      os << "llvm";
      break;
    case Target::Language::cuda:
      os << "cuda";
      break;
    case Target::Language::hip:
      os << "hip";
      break;
    case Target::Language::sycl:
      os << "sycl";
      break;
  }
  return os;
}

const Target &UnkTarget() {
  static Target target(
      Target::OS::Unk, Target::Arch::Unk, Target::Language::Unk, Target::Bit::Unk, {}, {});
  return target;
}
const Target &DefaultHostTarget() {
  static Target target(
      Target::OS::Linux, Target::Arch::X86, Target::Language::llvm, Target::Bit::k64, {}, {});
  return target;
}

const Target &DefaultNVGPUTarget() {
  static Target target(
      Target::OS::Linux, Target::Arch::NVGPU, Target::Language::cuda, Target::Bit::k64, {}, {});
  return target;
}

const Target &SYCLTarget(Target::Arch arch) {
  static Target target(
      Target::OS::Linux, arch, Target::Language::sycl, Target::Bit::k64, {}, {});
  SYCLWorkspace::Global()->Init(arch);
  return target;
}

int GetMaxThreads() {
  // cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )
  int max_threads = 1;
#ifdef CINN_WITH_CUDA
  int num_sm = 1;
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(
      &max_threads, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  // multiplication num_sm
  max_threads *= (num_sm * 4);
#endif
  return max_threads;
}

int GetMaxBlocks() {
  // cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )
  int max_blocks = 1;
#ifdef CINN_WITH_CUDA
  int num_sm = 1;
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(
      &max_blocks, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, 0);

  // multiplication num_sm
  max_blocks *= num_sm;
#endif
  return max_blocks;
}

const Target &DefaultTarget() {
#ifdef CINN_WITH_CUDA
  return DefaultNVGPUTarget();
#else
  return DefaultHostTarget();
#endif
}

}  // namespace common
}  // namespace cinn
