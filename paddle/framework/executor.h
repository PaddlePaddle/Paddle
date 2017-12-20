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

#pragma once

#include <map>
#include <unordered_map>

#include "paddle/framework/op_info.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

class DeviceContextPool {
 public:
  static DeviceContextPool& Get() {
    PADDLE_ENFORCE_NOT_NULL(pool, "Need to Create DeviceContextPool first!");
    return *pool;
  }

  static DeviceContextPool& Create(const std::vector<platform::Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  const platform::DeviceContext* Borrow(const platform::Place& place) {
    auto range = device_contexts_.equal_range(place);
    if (range.first == range.second) {
      PADDLE_THROW(
          "'Place' is not supported, Please re-compile with WITH_GPU "
          "option");
    }
    return range.first->second;
  }

  std::vector<const platform::DeviceContext*> Borrow(
      const std::vector<platform::Place>& places) {
    PADDLE_ENFORCE_GT(places.size(), 0);
    PADDLE_ENFORCE_LE(places.size(), device_contexts_.size());
    std::vector<const platform::DeviceContext*> borrowed_contexts;
    for (auto& place : places) {
      auto range = device_contexts_.equal_range(place);
      if (range.first == range.second) {
        PADDLE_THROW(
            "'Place' is not supported, Please re-compile with WITH_GPU "
            "option");
      }
      // TODO(dzhwinter) : assign the first found device. Will enhanced later.
      // device load balancer maybe useful here.
      borrowed_contexts.emplace_back(range.first->second);
    }
    return borrowed_contexts;
  }

  explicit DeviceContextPool(const std::vector<platform::Place>& places) {
    PADDLE_ENFORCE_GT(places.size(), 0);
    for (size_t i = 0; i < places.size(); i++) {
      if (platform::is_cpu_place(places[i])) {
        device_contexts_.emplace(
            places[i], new platform::CPUDeviceContext(
                           boost::get<platform::CPUPlace>(places[i])));
      } else if (platform::is_gpu_place(places[i])) {
#ifdef PADDLE_WITH_CUDA
        device_contexts_.emplace(
            places[i], new platform::CUDADeviceContext(
                           boost::get<platform::GPUPlace>(places[i])));
#else
        PADDLE_THROW(
            "'GPUPlace' is not supported, Please re-compile with WITH_GPU "
            "option");
#endif
      }
    }
  }

  ~DeviceContextPool() {}

 private:
  static DeviceContextPool* pool;
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const platform::Place& place) const {
      return hash_(place.which());
    }
  };
  std::unordered_multimap<const platform::Place, const platform::DeviceContext*,
                          Hash>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

class Executor {
 public:
  // TODO(dzhwinter) : Do not rely on this function, it will be removed
  explicit Executor(const platform::DeviceContext& device)
      : Executor(std::vector<platform::Place>({device.GetPlace()})) {}

  explicit Executor(const platform::Place& place)
      : Executor(std::vector<platform::Place>({place})) {}

  explicit Executor(const std::vector<platform::Place>& places);

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDescBind&, Scope*, int, bool create_local_scope = true);

 private:
  std::vector<const platform::DeviceContext*> device_contexts_;
};

}  // namespace framework
}  // namespace paddle
