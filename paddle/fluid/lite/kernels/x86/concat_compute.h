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

#include <Eigen/Core>
#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

inline int count(int start_axis, int end_axis, const lite::DDim& dim) {
  int count = 1;
  for (int i = start_axis; i < end_axis; ++i) {
    count *= dim[i];
  }
  return count;
}

template <typename T>
class ConcatCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConcatParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    int64_t axis = static_cast<int64_t>(param.axis);
    auto x_dims = param.x[0]->dims();
    auto out = param.output;
    if (param.x.size() == 1) return;

    auto output_data = param.output->template mutable_data<T>();
    int offset_concat_axis = 0;
    int num_concat = count(0, axis, x_dims);
    int concat_input_size = count(axis + 1, x_dims.size(), x_dims);
    const int top_concat_axis = out->dims()[axis];
    for (size_t i = 0; i < param.x.size(); ++i) {
      auto bottom_data = param.x[i]->data<T>();
      const int64_t bottom_concat_axis = param.x[i]->dims()[axis];
      for (int n = 0; n < num_concat; ++n) {
        std::memcpy(
            output_data +
                (n * top_concat_axis + offset_concat_axis) * concat_input_size,
            bottom_data + n * bottom_concat_axis * concat_input_size,
            (bottom_concat_axis * concat_input_size) * sizeof(T));
      }
      offset_concat_axis += bottom_concat_axis;
    }
  }
  virtual ~ConcatCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
