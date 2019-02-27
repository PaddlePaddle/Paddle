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

namespace paddle {
namespace inference {
namespace anakin {

void FcOpConverter::operator()(const framework::proto::OpDesc &op,
                               const framework::Scope &scope, bool test_mode) {
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
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
