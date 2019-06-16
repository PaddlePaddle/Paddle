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

#include "paddle/fluid/lite/kernels/arm/split_compute.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void splite_resize_out(const lite::Tensor* din,
                       const std::vector<lite::Tensor*>& dout, int axis,
                       int num, const std::vector<int>& sections) {
  auto in_dims = din->dims();
  int outs_number = dout.size();

  std::vector<lite::DDimLite> outs_dims;
  outs_dims.reserve(outs_number);

  if (num > 0) {
    int out_axis_dim = in_dims[axis] / num;
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = out_axis_dim;
      outs_dims.push_back(dim);
    }
  } else if (sections.size() > 0) {
    for (size_t i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = sections[i];
      outs_dims.push_back(dim);
    }
  }

  for (int j = 0; j < outs_dims.size(); ++j) {
    dout[j]->Resize(outs_dims[j]);
  }
}

template <typename dtype>
void split_compute_ref(const operators::SplitParam& param) {
  const dtype* din = param.x->mutable_data<const dtype>();
  auto& dout = param.output;
  auto in_dim = param.x->dims();
  int axis = param.axis;
  std::vector<int> in_strides(in_dim.size());
  in_strides[in_dim.size() - 1] = in_dim[in_dim.size() - 1];
  for (int i = in_dim.size() - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * in_dim[i];
  }

  int input_offset = 0;
  for (auto out : dout) {
    auto out_dim = out->dims();
    std::vector<int> out_strides(out_dim.size());
    out_strides[out_dim.size() - 1] = out_dim[out_dim.size() - 1];
    for (int i = out_dim.size() - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

    dtype* out_data = out->mutable_data<dtype>();
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    for (int i = 0; i < before; ++i) {
      std::memcpy(out_data + i * out_after, din + input_offset + i * in_after,
                  sizeof(dtype) * out_after);
    }
    input_offset += out_strides[axis];
  }
}

TEST(split_arm, init) {
  SplitCompute split;
  ASSERT_EQ(split.precision(), PRECISION(kFloat));
  ASSERT_EQ(split.target(), TARGET(kARM));
}

TEST(split_arm, compute) {
  SplitCompute split;
  operators::SplitParam param;

  lite::Tensor x;
  std::vector<lite::Tensor*> output;
  std::vector<lite::Tensor*> output_ref;

  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto axis : {0, 1, 2, 3}) {
            for (auto num : {0, 1, 2, 3}) {
              for (auto sections :
                   {std::vector<int>{1, 1, 1}, std::vector<int>{2, 2},
                    std::vector<int>{1, 2}}) {
                auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
                x.Resize(x_dim);
                if ((num != 0 && x_dim[axis] % num != 0) ||
                    (num == 0 && x_dim[axis] % sections.size() != 0))
                  continue;
                auto* x_data = x.mutable_data<float>();
                for (int i = 0; i < x.dims().production(); i++) {
                  x_data[i] = i;
                }
                for (auto out : output) delete out;
                for (auto out : output_ref) delete out;
                output.clear();
                output_ref.clear();

                int outs_number;
                if (num > 0) {
                  outs_number = num;
                } else {
                  outs_number = sections.size();
                }
                for (int i = 0; i < outs_number; i++) {
                  output.push_back(new lite::Tensor);
                  output_ref.push_back(new lite::Tensor);
                }
                splite_resize_out(&x, output, axis, num, sections);
                splite_resize_out(&x, output_ref, axis, num, sections);
                param.x = &x;
                param.axis = axis;
                param.num = num;
                param.sections = sections;
                param.output = output;
                split.SetParam(param);
                split.Run();
                param.output = output_ref;
                split_compute_ref<float>(param);
                for (int i = 0; i < output.size(); i++) {
                  float* output_data = output[i]->mutable_data<float>();
                  float* output_ref_data = output_ref[i]->mutable_data<float>();
                  for (int j = 0; j < output[i]->dims().production(); j++) {
                    EXPECT_NEAR(output_data[j], output_ref_data[j], 1e-5);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(split, retrive_op) {
  auto split =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("split");
  ASSERT_FALSE(split.empty());
  ASSERT_TRUE(split.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(split, kARM, kFloat, kNCHW, def);
