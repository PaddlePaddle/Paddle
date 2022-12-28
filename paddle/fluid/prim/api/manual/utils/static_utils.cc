// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/utils/data_type.h"
namespace paddle {
namespace prim {
using Tensor = paddle::experimental::Tensor;
template <>
Tensor empty<DescTensor>(const paddle::experimental::IntArray& shape,
                         paddle::experimental::DataType dtype,
                         const paddle::Place& place) {
  framework::VarDesc* new_var =
      StaticCompositeContext::Instance().GetBlock()->Var(
          std::move(StaticCompositeContext::Instance().GenerateUniqueName()));
  new_var->SetShape(shape.GetData());
  new_var->SetDataType(framework::TransToProtoVarType(dtype));
  // Place is not supported in static mode
  return Tensor(std::make_shared<prim::DescTensor>(new_var));
}

template <>
Tensor empty_like<DescTensor>(const Tensor& x,
                              paddle::experimental::DataType dtype,
                              const paddle::Place& place) {
  return empty<prim::DescTensor>(
      paddle::experimental::IntArray(x.shape()), x.dtype(), paddle::Place());
}

}  // namespace prim
}  // namespace paddle
