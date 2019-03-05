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

#include "paddle/fluid/inference/anakin/convert/batch_norm.h"
#include <algorithm>
#include <map>
#include <string>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

void BatchNormOpConverter::operator()(const framework::proto::OpDesc &op,
                                      const framework::Scope &scope,
                                      bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);
  std::map<std::string, std::string> inputs;
  for (auto k : {"X", "Scale", "Bias", "Mean", "Variance"}) {
    PADDLE_ENFORCE_EQ(op_desc.Input(k).size(), 1UL);
    auto v = op_desc.Input(k).front();
    inputs.insert({k, v});
  }

  auto output = op_desc.Output("Y").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Y").front();
  engine_->AddOp(op_name, "BatchNorm", {inputs["X"]}, {output});

  bool is_test = boost::get<bool>(op_desc.GetAttr("is_test"));
  PADDLE_ENFORCE(is_test);
  float momentum = boost::get<float>(op_desc.GetAttr("momentum"));
  engine_->AddOpAttr(op_name, "momentum", momentum);
  float epsilon = boost::get<float>(op_desc.GetAttr("epsilon"));
  engine_->AddOpAttr(op_name, "epsilon", epsilon);

  auto set_anakin_weight = [this, &scope, &op_desc, &op_name](
      const std::string &weight_name, const std::string &var_name) {
    auto *v = scope.FindVar(var_name); //op_desc.Input(var_name).front());
    PADDLE_ENFORCE_NOT_NULL(v);
    auto *t = v->GetMutable<framework::LoDTensor>();
    framework::LoDTensor tensor;
    tensor.Resize(t->dims());
    TensorCopySync((*t), platform::CPUPlace(), &tensor);
    auto shape = framework::vectorize2int(t->dims());
    if (shape.size() < 4UL) {
      shape.insert(shape.end(), 4UL - shape.size(), 1);
    }
    Shape anakin_shape(shape);
    //Shape shape(framework::vectorize2int(t->dims()));
    auto *weight =
        GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(anakin_shape);
    float *cpu_data = static_cast<float *>(weight->h_tensor().mutable_data());
    std::copy_n(tensor.data<float>(), tensor.numel(), cpu_data);
    this->engine_->AddOpAttr(op_name, weight_name, *weight);
  };
  set_anakin_weight("weight_1", inputs["Mean"]);
  set_anakin_weight("weight_2", inputs["Variance"]);
  set_anakin_weight("weight_3", inputs["Scale"]);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(batch_norm, BatchNormOpConverter);
