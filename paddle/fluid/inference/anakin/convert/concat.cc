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

#include "paddle/fluid/inference/anakin/convert/concat.h"
#include <algorithm>

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void ConcatOpConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  int axis = boost::get<int>(op_desc.GetAttr("axis"));
  auto input_names = op_desc.Input("X");

  auto y_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Concat", input_names, {y_name});
  this->engine_->AddOpAttr(op_name, "axis", axis);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
using concat_nv_fp32 =
    ::paddle::inference::anakin::ConcatOpConverter<::anakin::saber::NV,
                                                   ::anakin::Precision::FP32>;
using concat_nv_int8 =
    ::paddle::inference::anakin::ConcatOpConverter<::anakin::saber::NV,
                                                   ::anakin::Precision::INT8>;
REGISTER_CUDA_ANAKIN_OP_CONVERTER(concat, concat_nv_fp32);
REGISTER_CUDA_INT8_ANAKIN_OP_CONVERTER(concat, concat_nv_int8);

#endif
using concat_cpu_fp32 =
    ::paddle::inference::anakin::ConcatOpConverter<::anakin::saber::X86,
                                                   ::anakin::Precision::FP32>;
using concat_cpu_int8 =
    ::paddle::inference::anakin::ConcatOpConverter<::anakin::saber::X86,
                                                   ::anakin::Precision::INT8>;
REGISTER_CPU_ANAKIN_OP_CONVERTER(concat, concat_cpu_fp32);
REGISTER_CPU_INT8_ANAKIN_OP_CONVERTER(concat, concat_cpu_int8);
