// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/anakin/convert/dropout.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/inference/anakin/convert/helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void DropoutOpConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Mask").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  auto out_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Scale", {x_name}, {out_name});

  auto dropout_prob = boost::get<float>(op_desc.GetAttr("dropout_prob"));
  auto factor = 1 - dropout_prob;
  auto *weight1 = pblock_from_vector<TargetT>(std::vector<float>({factor}));

  this->engine_->AddOpAttr(op_name, "weight_1", *weight1);
  this->engine_->AddOpAttr(op_name, "axis", 0);
  this->engine_->AddOpAttr(op_name, "num_axes", 0);
  this->engine_->AddOpAttr(op_name, "bias_term", false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
using dropout_nv_fp32 =
    ::paddle::inference::anakin::DropoutOpConverter<::anakin::saber::NV,
                                                    ::anakin::Precision::FP32>;
using dropout_nv_int8 =
    ::paddle::inference::anakin::DropoutOpConverter<::anakin::saber::NV,
                                                    ::anakin::Precision::INT8>;
REGISTER_CUDA_ANAKIN_OP_CONVERTER(dropout, dropout_nv_fp32);
REGISTER_CUDA_INT8_ANAKIN_OP_CONVERTER(dropout, dropout_nv_int8);
#endif

using dropout_cpu_fp32 =
    ::paddle::inference::anakin::DropoutOpConverter<::anakin::saber::X86,
                                                    ::anakin::Precision::FP32>;
using dropout_cpu_int8 =
    ::paddle::inference::anakin::DropoutOpConverter<::anakin::saber::X86,
                                                    ::anakin::Precision::INT8>;
REGISTER_CPU_ANAKIN_OP_CONVERTER(dropout, dropout_cpu_fp32);
REGISTER_CPU_INT8_ANAKIN_OP_CONVERTER(dropout, dropout_cpu_int8);
