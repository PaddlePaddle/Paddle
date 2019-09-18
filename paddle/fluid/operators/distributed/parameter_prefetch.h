//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
namespace distributed {

void prefetch(const std::string& id_name, const std::string& out_name,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& epmap,
              const std::vector<int64_t>& height_sections,
              const framework::ExecutionContext& context,
              const framework::Scope& scope);

template <typename T>
void prefetch_with_reconstruct(const std::string& id_name,
                               const std::string& out_name,
                               const std::vector<std::string>& table_names,
                               const std::vector<std::string>& epmap,
                               const std::vector<int64_t>& height_sections,
                               const framework::ExecutionContext& context,
                               const framework::Scope& scope,
                               framework::LoDTensor* original) {
  prefetch(id_name, out_name, table_names, epmap, height_sections, context,
           scope);
  auto& out = scope.FindVar(out_name)->Get<framework::LoDTensor>();
  auto& ids = scope.FindVar(id_name)->Get<framework::LoDTensor>();
  auto* original_value = original->data<T>();
  auto* out_value = out.data<T>();
  size_t original_width = original->numel() / original->dims()[0];

  bool is_on_cpu_place = true;
  if (!platform::is_cpu_place(ids.place())) {
    is_on_cpu_place = false;
  }
  if (is_on_cpu_place) {
    for (int64_t i = 0; i < ids.numel(); i++) {
      const T* out_rows = out_value + original_width * i;
      T* original_row =
          original_value + original_width * ids.data<int64_t>()[i];
      std::memcpy(original_row, out_rows, original_width * sizeof(T));
    }
  } else {
#ifndef PADDLE_WITH_CUDA
    PADDLE_THROW("paddle is not compiled with CUDA!");
#else
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& actual_ctx = *pool.Get(context.GetPlace());
    for (int64_t i = 0; i < ids.numel(); i++) {
      const T* out_rows = out_value + original_width * i;
      T* original_row =
          original_value + original_width * ids.data<int64_t>()[i];
      auto stream =
          static_cast<platform::CUDADeviceContext*>(&actual_ctx)->stream();
      memory::Copy(boost::get<platform::CUDAPlace>(ids.place()), original_row,
                   platform::CPUPlace(), out_rows, original_width * sizeof(T),
                   stream);
    }
#endif
  }
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
