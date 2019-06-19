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

#include "paddle/fluid/lite/kernels/arm/concat_compute.h"
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

bool infer_shape(const operators::ConcatParam& param) {
  std::vector<lite::DDim> input_dims;
  for (auto p : param.x) {
    input_dims.push_back(p->dims());
  }
  size_t axis = static_cast<size_t>(param.axis);
  const size_t n = input_dims.size();
  CHECK_GT_OR_FALSE(n, 0);
  auto& out_dims = input_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += input_dims[i][j];
      } else {
        CHECK_EQ_OR_FALSE(out_dims[j], input_dims[i][j]);
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }
  // Set output dims
  param.output->Resize(lite::DDim(out_dims));
  return true;
}

void concat_compute_ref(const operators::ConcatParam& param) {
  std::vector<lite::Tensor*> input = param.x;
  int axis = param.axis;
  infer_shape(param);

  lite::Tensor* output = param.output;
  int num = input.size();
  int rows = 1;
  auto dim_0 = input[0]->dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;

  std::vector<int> input_cols(input.size());
  for (int i = 0; i < num; ++i) {
    int input_i_numel = input[i]->dims().size() == 0 ? 0 : 1;
    for (int didx = 0; didx < input[i]->dims().size(); ++didx) {
      input_i_numel *= input[i]->dims()[didx];
    }
    int t_cols = input_i_numel / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }

  // computation
  auto output_data = output->mutable_data<float>();
  int col_idx = 0;
  for (int j = 0; j < num; ++j) {
    int col_len = input_cols[j];
    auto input_data = input[j]->data<float>();
    for (int k = 0; k < out_rows; ++k) {
      memcpy(output_data + k * out_cols + col_idx, input_data + k * col_len,
             sizeof(float) * col_len);
    }
    col_idx += col_len;
  }
}

TEST(concat_arm, init) {
  ConcatCompute concat;
  ASSERT_EQ(concat.precision(), PRECISION(kFloat));
  ASSERT_EQ(concat.target(), TARGET(kARM));
}

TEST(concat_arm, compute_input_single) {
  ConcatCompute concat;
  operators::ConcatParam param;

  LOG(INFO) << "test concat start";
  lite::Tensor output;
  lite::Tensor output_ref;
  lite::Tensor tensorA;
  DDimLite ddimA({10, 4, 3, 2});
  tensorA.Resize(ddimA);

  for (int i = 0; i < ddimA.data()[0] * ddimA.data()[1] * ddimA.data()[2] *
                          ddimA.data()[3];
       i++) {
    tensorA.mutable_data<float>()[i] = i;
  }

  param.x.push_back(&tensorA);
  for (int cur_axis : {0, 1}) {
    param.output = &output;
    param.axis = cur_axis;
    CHECK(infer_shape(param));
    concat.SetParam(param);
    LOG(INFO) << "test concat start cur_axis:" << cur_axis;

    concat.Run();
    LOG(INFO) << "concat.Run end";
    param.output = &output_ref;
    LOG(INFO) << "concat_compute_ref start";
    concat_compute_ref(param);
    LOG(INFO) << "concat_compute_ref end";

    auto* output_data = output.data<float>();
    auto* output_ref_data = output_ref.data<float>();
    for (int i = 0; i < (ddimA.data()[0]) * ddimA.data()[1] * ddimA.data()[2] *
                            ddimA.data()[3];
         i++) {
      // LOG(INFO) << "output[" << i << "]:" << output_data[i] << "
      // output_ref_data[" << i << "]:" << output_ref_data[i];
      EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
    }
  }
}

TEST(concat_arm, compute_input_multi) {
  ConcatCompute concat;
  operators::ConcatParam param;

  LOG(INFO) << "test concat start";
  // init param
  // x: tensorA, tensorB, tensorC, tensorD
  // axis: 0
  lite::Tensor output;
  lite::Tensor output_ref;
  lite::Tensor tensorA;
  lite::Tensor tensorB;
  lite::Tensor tensorC;
  lite::Tensor tensorD;

  DDimLite ddimA({10, 4, 3, 2});
  DDimLite ddimB({20, 4, 3, 2});
  DDimLite ddimC({30, 4, 3, 2});
  DDimLite ddimD({40, 4, 3, 2});

  tensorA.Resize(ddimA);
  tensorB.Resize(ddimB);
  tensorC.Resize(ddimC);
  tensorD.Resize(ddimD);

  for (int i = 0; i < ddimA.data()[0] * ddimA.data()[1] * ddimA.data()[2] *
                          ddimA.data()[3];
       i++) {
    tensorA.mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < ddimB.data()[0] * ddimB.data()[1] * ddimB.data()[2] *
                          ddimB.data()[3];
       i++) {
    tensorB.mutable_data<float>()[i] = i + 1;
  }
  for (int i = 0; i < ddimC.data()[0] * ddimC.data()[1] * ddimC.data()[2] *
                          ddimC.data()[3];
       i++) {
    tensorC.mutable_data<float>()[i] = i + 2;
  }
  for (int i = 0; i < ddimD.data()[0] * ddimD.data()[1] * ddimD.data()[2] *
                          ddimD.data()[3];
       i++) {
    tensorD.mutable_data<float>()[i] = i + 3;
  }

  param.x.push_back(&tensorA);
  param.x.push_back(&tensorB);
  param.x.push_back(&tensorC);
  param.x.push_back(&tensorD);
  for (int cur_axis : {0}) {
    param.output = &output;
    param.axis = cur_axis;
    CHECK(infer_shape(param));
    concat.SetParam(param);
    LOG(INFO) << "test concat start cur_axis:" << cur_axis;

    concat.Run();
    LOG(INFO) << "concat.Run end";
    param.output = &output_ref;
    LOG(INFO) << "concat_compute_ref start";
    concat_compute_ref(param);
    LOG(INFO) << "concat_compute_ref end";

    auto* output_data = output.data<float>();
    auto* output_ref_data = output_ref.data<float>();
    int elem_num = (ddimA.data()[0] + ddimB.data()[0] + ddimC.data()[0] +
                    ddimD.data()[0]) *
                   ddimA.data()[1] * ddimA.data()[2] * ddimA.data()[3];
    for (int i = 0; i < elem_num; i++) {
      // LOG(INFO) << "output[" << i << "]:" << output_data[i] << "
      // output_ref_data[" << i << "]:" << output_ref_data[i];
      EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
    }
  }
}

TEST(concat, retrive_op) {
  auto concat =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "concat");
  ASSERT_FALSE(concat.empty());
  ASSERT_TRUE(concat.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(concat, kARM, kFloat, kNCHW, def);
