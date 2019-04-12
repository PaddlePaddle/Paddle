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

#include "paddle/fluid/inference/anakin/convert/fc.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/inference/anakin/convert/helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void FcBaseOpConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  auto input_names = op_desc.InputNames();
  bool with_bias = input_names.size() >= 3;

  std::string w_name = "Y";
  std::string i_name = "X";
  if (with_bias) {
    w_name = "W";
    i_name = "Input";
  }

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  // get weights
  auto *y_v = scope.FindVar(op_desc.Input(w_name).front());
  PADDLE_ENFORCE_NOT_NULL(y_v);
  auto weight_tensor = tensor_from_var(*y_v, platform::CPUPlace());
  auto weight_shape = framework::vectorize2int(weight_tensor->dims());

  int out_dim = weight_shape[1];
  const int w_m = weight_shape[0];
  const int w_k = weight_shape[1];

  auto input_name = op_desc.Input(i_name).front();
  auto output_name = op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Dense", {input_name}, {output_name});
  this->engine_->AddOpAttr(op_name, "bias_term", with_bias);
  this->engine_->AddOpAttr(op_name, "axis", 1);
  this->engine_->AddOpAttr(op_name, "out_dim", out_dim);

  auto *weight_data = weight_tensor->data<float>();
  PADDLE_ENFORCE(w_m * w_k == weight_tensor->numel());

  std::vector<float> trans_weight_data(weight_tensor->numel());
  for (int i = 0; i < w_m; i++) {
    for (int j = 0; j < w_k; j++) {
      trans_weight_data[i + j * w_m] = weight_data[i * w_k + j];
    }
  }

  int weight_num = weight_tensor->numel();
  bool enable_int8 = boost::get<bool>(op_desc.HasAttr("enable_int8"));
  if (enable_int8) {
    if (weight_shape.size() < 4UL) {
      weight_shape.insert(weight_shape.begin(), 4UL - weight_shape.size(), 1);
    }
    ::anakin::saber::Shape anakin_shape(weight_shape);
    const float int8_range = 127.;
    float in_scale = boost::get<float>(op_desc.GetAttr("input_scale"));
    float weight_scale = boost::get<float>(op_desc.GetAttr("weight_scale"));
    PBlock<TargetT> *weight1 =
        new PBlock<TargetT>(anakin_shape, ::anakin::AK_INT8);
    this->engine_->RegistBlock(weight1);
    std::vector<char> weight_int8;
    for (int i = 0; i < weight_num; i++) {
      bool is_valid_int8 =
          ((trans_weight_data[i] >= -128) && (trans_weight_data[i] <= 127));
      PADDLE_ENFORCE(is_valid_int8,
                     "We are in anakin subgraph int8 mode, the weight of fc "
                     "should be in range [-128, 127]");
      weight_int8.push_back(static_cast<char>(trans_weight_data[i]));
    }
    memcpy(static_cast<void *>(weight1->h_tensor().mutable_data()),
           static_cast<void *>(weight_int8.data()), sizeof(char) * weight_num);
    weight1->d_tensor().set_shape(anakin_shape);
    weight1->d_tensor().copy_from(weight1->h_tensor());
    this->engine_->AddOpAttr(op_name, "weight_1", *weight1);
    this->engine_->Graph()->SetOpPrec(op_name, ::anakin::AK_INT8);
    this->engine_->Graph()->SetWeightsScale(op_name,
                                            {weight_scale / int8_range}, false);
    this->engine_->AddTensorScale(input_name, in_scale / int8_range);
  } else {
    auto *weight1 = pblock_from_vector<TargetT, PrecisionT>(trans_weight_data,
                                                            this->engine_);
    this->engine_->AddOpAttr(op_name, "weight_1", *weight1);
  }

  // get bias
  if (with_bias) {
    auto *b_v = scope.FindVar(op_desc.Input("Bias").front());
    PADDLE_ENFORCE_NOT_NULL(b_v);
    auto weight2 = pblock_from_var<TargetT, PrecisionT>(*b_v, this->engine_);
    this->engine_->AddOpAttr(op_name, "weight_2", *weight2);
  }
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
using mul_nv_fp32 =
    ::paddle::inference::anakin::MulOpConverter<::anakin::saber::NV,
                                                ::anakin::Precision::FP32>;
using fc_nv_fp32 =
    ::paddle::inference::anakin::FcOpConverter<::anakin::saber::NV,
                                               ::anakin::Precision::FP32>;
using mul_nv_int8 =
    ::paddle::inference::anakin::MulOpConverter<::anakin::saber::NV,
                                                ::anakin::Precision::INT8>;
using fc_nv_int8 =
    ::paddle::inference::anakin::FcOpConverter<::anakin::saber::NV,
                                               ::anakin::Precision::INT8>;

REGISTER_CUDA_ANAKIN_OP_CONVERTER(mul, mul_nv_fp32);
REGISTER_CUDA_ANAKIN_OP_CONVERTER(fc, fc_nv_fp32);
REGISTER_CUDA_INT8_ANAKIN_OP_CONVERTER(mul, mul_nv_int8);
REGISTER_CUDA_INT8_ANAKIN_OP_CONVERTER(fc, fc_nv_int8);
#endif

using mul_cpu_fp32 =
    ::paddle::inference::anakin::MulOpConverter<::anakin::saber::X86,
                                                ::anakin::Precision::FP32>;
using fc_cpu_fp32 =
    ::paddle::inference::anakin::FcOpConverter<::anakin::saber::X86,
                                               ::anakin::Precision::FP32>;
using mul_cpu_int8 =
    ::paddle::inference::anakin::MulOpConverter<::anakin::saber::X86,
                                                ::anakin::Precision::INT8>;
using fc_cpu_int8 =
    ::paddle::inference::anakin::FcOpConverter<::anakin::saber::X86,
                                               ::anakin::Precision::INT8>;

REGISTER_CPU_ANAKIN_OP_CONVERTER(mul, mul_cpu_fp32);
REGISTER_CPU_ANAKIN_OP_CONVERTER(fc, fc_cpu_fp32);
REGISTER_CPU_INT8_ANAKIN_OP_CONVERTER(mul, mul_cpu_int8);
REGISTER_CPU_INT8_ANAKIN_OP_CONVERTER(fc, fc_cpu_int8);
