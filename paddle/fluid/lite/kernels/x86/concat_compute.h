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

template <typename T>
class ConcatCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConcatParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    int64_t axis = static_cast<int64_t>(param.axis);
    auto out = param.output;

    if (axis == 0 && param.x.size() < 10) {
      size_t output_offset = 0;
      for (auto* in : param.x) {
        if (!in || in->dims().production() == 0UL) {
          continue;
        }
        auto in_stride = framework::stride_numel(in->dims().data());
        auto out_stride = framework::stride_numel(out->dims().data());
        paddle::operators::StridedNumelCopyWithAxis<T>(
            platform::CPUDeviceContext(), axis,
            out->mutable_data<T>() + output_offset, out_stride, in->data<T>(),
            in_stride, in_stride[axis]);

        output_offset += in_stride[axis];
      }
    } else {
      std::vector<lite::Tensor> inputs;
      for (size_t j = 0; j < param.x.size(); ++j) {
        if (param.x[j] && param.x[j]->dims().production() > 0) {
          inputs.push_back(*param.x[j]);
        } else {
          continue;
        }
      }

      int num = inputs.size();
      int rows = 1;
      auto dim_0 = inputs[0].dims();
      for (int i = 0; i < axis; ++i) {
        rows *= dim_0[i];
      }
      int out_rows = rows, out_cols = 0;

      std::vector<int64_t> input_cols(inputs.size());
      for (int i = 0; i < num; ++i) {
        int t_cols = inputs[i].dims().production() / rows;
        out_cols += t_cols;
        input_cols[i] = t_cols;
      }
      // computation
      auto output_data = param.output->template mutable_data<T>();
      int col_idx = 0;
      for (int j = 0; j < num; ++j) {
        int col_len = input_cols[j];
        auto input_data = inputs[j].data<float>();
        for (int k = 0; k < out_rows; ++k) {
          std::memcpy(output_data + k * out_cols + col_idx,
                      input_data + k * col_len, sizeof(T) * col_len);
        }
        col_idx += col_len;
      }
    }
  }

  virtual ~ConcatCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
