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

#include "paddle/fluid/lite/kernels/arm/transpose_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/lite_tensor.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]
void transpose_compute_ref(const operators::TransposeParam& param) {
  const lite::Tensor* input = param.x;
  lite::Tensor* output = param.output;
  std::vector<int> axis = param.axis;

  auto* input_data = input->data<float>();
  auto* output_data = output->mutable_data<float>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_n = output->dims()[0];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}

// Transpose
TEST(transpose_arm, init) {
  TransposeCompute transpose;
  ASSERT_EQ(transpose.precision(), PRECISION(kFloat));
  ASSERT_EQ(transpose.target(), TARGET(kARM));
}

TEST(transpose_arm, compute_shape_nchw) {
  TransposeCompute transpose;
  operators::TransposeParam param;

  std::vector<int> axis{0, 2, 3, 1};
  param.axis = axis;

  lite::Tensor input;
  lite::Tensor output;
  lite::Tensor output_ref;

  const std::vector<int64_t> input_shape{1, 24, 2, 2};
  const std::vector<int64_t> output_shape{1, 2, 2, 24};

  DDimLite ddimInput(input_shape);
  DDimLite ddimOutput(output_shape);

  input.Resize(ddimInput);
  output.Resize(ddimOutput);
  output_ref.Resize(ddimOutput);

  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
    input.mutable_data<float>()[i] = i;
    input.mutable_data<float>()[i + 1] = i + 1;
    input.mutable_data<float>()[i + 2] = i + 2;
    input.mutable_data<float>()[i + 3] = i + 3;
  }
  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
  }
  param.x = &input;
  param.output = &output;

  // run transpose_compute
  transpose.SetParam(param);
  transpose.Run();

  // run transpose_compute_ref
  param.output = &output_ref;
  transpose_compute_ref(param);

  auto* output_data = output.data<float>();
  auto* output_ref_data = output_ref.data<float>();
  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
  }
}

TEST(transpose, retrive_op) {
  auto transpose =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "transpose");
  ASSERT_FALSE(transpose.empty());
  ASSERT_TRUE(transpose.front());
}

// Transpose2
TEST(transpose2_arm, init) {
  Transpose2Compute transpose2;
  ASSERT_EQ(transpose2.precision(), PRECISION(kFloat));
  ASSERT_EQ(transpose2.target(), TARGET(kARM));
}

TEST(transpose2_arm, compute_shape_nchw) {
  Transpose2Compute transpose2;
  operators::TransposeParam param;

  std::vector<int> axis{0, 2, 3, 1};
  param.axis = axis;

  lite::Tensor input;
  lite::Tensor output;
  lite::Tensor output_ref;

  const std::vector<int64_t> input_shape{1, 24, 2, 2};
  const std::vector<int64_t> output_shape{1, 2, 2, 24};

  DDimLite ddimInput(input_shape);
  DDimLite ddimOutput(output_shape);

  input.Resize(ddimInput);
  output.Resize(ddimOutput);
  output_ref.Resize(ddimOutput);

  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
    input.mutable_data<float>()[i] = i;
    input.mutable_data<float>()[i + 1] = i + 1;
    input.mutable_data<float>()[i + 2] = i + 2;
    input.mutable_data<float>()[i + 3] = i + 3;
  }
  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
  }
  param.x = &input;
  param.output = &output;

  // run transpose_compute
  transpose2.SetParam(param);
  transpose2.Run();

  // run transpose_compute_ref
  param.output = &output_ref;
  transpose_compute_ref(param);

  auto* output_data = output.data<float>();
  auto* output_ref_data = output_ref.data<float>();
  for (int i = 0;
       i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
       i += 4) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
  }
}

TEST(transpose2, retrive_op) {
  auto transpose2 =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "transpose2");
  ASSERT_FALSE(transpose2.empty());
  ASSERT_TRUE(transpose2.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kARM, kFloat, kNCHW, def);
