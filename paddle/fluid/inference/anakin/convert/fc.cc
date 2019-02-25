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

#include <string>
#include <vector>
#include "framework/core/types.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/registrar.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "saber/saber_types.h"

namespace paddle {
namespace inference {
namespace anakin {

class FcOpConverter : OpConverter {
 public:
  FcOpConverter() {}

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::Scope &scope);
  void ConvertOp(const framework::proto::OpDesc &op,
                 const std::unordered_set<std::string> &parameters,
                 const framework::Scope &scope, AnakinNvEngine *engine);
  virtual ~FcOpConverter() {}

 private:
};

void FcOpConverter::ConvertOp(const framework::proto::OpDesc &op,
                              const std::unordered_set<std::string> &parameters,
                              const framework::Scope &scope,
                              AnakinNvEngine *engine) {
  framework::OpDesc op_desc(op, nullptr);
  auto *Y_v = scope.FindVar(op_desc.Input("Y").front());
  PADDLE_ENFORCE_NOT_NULL(Y_v);
  auto *Y_t = Y_v.GetMutable<framework::LoDTensor>();
  platform::CPUPlace cpu_place;
  framwork::LoDTensor weight_tensor;
  weight_tensor.Resize(Y_t->dims());
  auto op_name = op.attrs(0).name();

  engine->AddOp(op_name, "dense", op_desc.Input("X"), op_desc.Output("Out"));
  // engine->AddOpAttr(op_name, "out_dim", )
  // engine->AddOpAttr(op_name, "bias_term", false);
  // engine->AddOpAttr(op_name, "axis", );
  // std::vector<int> shape = {1, 1, 3, 100};
  //::anakin::saber::Shape tmp_shape{shape};
  //::anakin::saber::Tensor<::anakin::saber::NV> weight1(tmp_shape);
  // engine->AddOpAttr(op_name, "weight_1", weight1);

  // engine->AddOp();
  // engine->AddOpAttr();
}

static Registrar<FcOpConverter> registrar_fc("fc");

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
