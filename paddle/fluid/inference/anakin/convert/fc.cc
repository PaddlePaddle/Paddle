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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/registrar.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "saber/saber_types.h"

namespace paddle {
namespace inference {
namespace anakin {

class FcOpConverter : public OpConverter {
 public:
  FcOpConverter() = default;

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::Scope &scope);
  virtual ~FcOpConverter() {}

 private:
};

void FcOpConverter::operator()(const framework::proto::OpDesc &op,
                               const framework::Scope &scope) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Input("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE(x_name.size() > 0);
  auto *y_v = scope.FindVar(op_desc.Input("Y").front());
  PADDLE_ENFORCE_NOT_NULL(y_v);
  auto *y_t = y_v->GetMutable<framework::LoDTensor>();

  auto shape = framework::vectorize2int(y_t->dims());
  ::anakin::saber::Shape anakin_shape(shape);

  PADDLE_ENFORCE_NOT_NULL(y_t);
  framework::LoDTensor weight;
  weight.Resize(y_t->dims());
  TensorCopySync(*y_t, platform::CUDAPlace(), &weight);

  auto *weight_data = weight.mutable_data<float>(platform::CUDAPlace());
  PADDLE_ENFORCE_EQ(weight.dims().size(), 2UL);
  auto n_output = weight.dims()[1];  // out_dim

  engine->AddOpAttr(x_name, "out_dim", n_output);

  PADDLE_ENFORCE_NOT_NULL(weight_data);
  PADDLE_ENFORCE(n_output > 0);

  /*
  std::unique_ptr<framework::Tensor> tmp(new framework::LoDTensor());
  tmp->Resize(weight.dims());

  std::memcpy(tmp->mutable_data<float>(platform::CPUPlace()), weight,
  y_t->dims()[0] * y_t->dims()[1] * sizeof(float));
  ::anakin::saber::Shape tmp_shape(shape);
  ::anakin::PBlock<::anakin::saber::NV> weight1(tmp_shape);
  */
}

static Registrar<FcOpConverter> registrar_fc("fc");

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
