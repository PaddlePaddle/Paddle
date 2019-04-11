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

#include "paddle/fluid/inference/anakin/convert/conv2d.h"
#include <algorithm>
#include <memory>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::PTuple;
using anakin::AK_FLOAT;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void Conv2dOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(), 1UL);

  auto input_name = op_desc.Input("Input").front();
  auto output_name = op_desc.Output("Output").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Output").front();
  this->engine_->AddOp(op_name, "Convolution", {input_name}, {output_name});

  auto *filter_v = scope.FindVar(op_desc.Input("Filter").front());
  PADDLE_ENFORCE_NOT_NULL(filter_v);
  auto *filter_t = filter_v->GetMutable<framework::LoDTensor>();
  std::unique_ptr<framework::LoDTensor> weight_tensor(
      new framework::LoDTensor());
  weight_tensor->Resize(filter_t->dims());
  TensorCopySync((*filter_t), platform::CPUPlace(), weight_tensor.get());

  PADDLE_ENFORCE_EQ(weight_tensor->dims().size(), 4UL);

  const int filter_h = weight_tensor->dims()[2];
  const int filter_w = weight_tensor->dims()[3];

  auto filter_num = weight_tensor->dims()[0];
  this->engine_->template AddOpAttr<int>(op_name, "filter_num", filter_num);
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "kernel_size",
                                                 {filter_h, filter_w});
  auto strides = boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "strides", strides);
  auto paddings = boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "padding", paddings);
  auto dilations = boost::get<std::vector<int>>(op_desc.GetAttr("dilations"));
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "dilation_rate",
                                                 dilations);
  const int groups = boost::get<int>(op_desc.GetAttr("groups"));
  this->engine_->AddOpAttr(op_name, "group", groups);
  this->engine_->AddOpAttr(op_name, "axis", 1);
  this->engine_->AddOpAttr(op_name, "bias_term", false);

  auto weight_shape = framework::vectorize2int(filter_t->dims());
  Shape anakin_shape(weight_shape);
  auto *weight1 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(
          anakin_shape);
  float *cpu_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(weight_tensor->data<float>(), weight_tensor->numel(), cpu_data);
  weight1->d_tensor().set_shape(anakin_shape);
  weight1->d_tensor().copy_from(weight1->h_tensor());
  this->engine_->AddOpAttr(op_name, "weight_1", *weight1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_CPU_ANAKIN_OP_CONVERTER(conv2d,
                                 Conv2dOpConverter<::anakin::saber::X86>);
#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(conv2d,
                                  Conv2dOpConverter<::anakin::saber::NV>);
#endif
