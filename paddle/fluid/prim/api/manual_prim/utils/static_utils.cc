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

#include "paddle/common/macros.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/prim/api/manual_prim/utils/utils.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/utils/data_type.h"
namespace paddle::prim {
using Tensor = paddle::Tensor;
template <>
TEST_API Tensor empty<DescTensor>(const paddle::experimental::IntArray& shape,
                                  phi::DataType dtype,
                                  const paddle::Place& place) {
  framework::VarDesc* new_var =
      StaticCompositeContext::Instance().GetBlock()->Var(
          StaticCompositeContext::Instance().GenerateUniqueName());
  new_var->SetShape(shape.GetData());
  new_var->SetDataType(framework::TransToProtoVarType(dtype));
  // Place is not supported in static mode
  return Tensor(std::make_shared<prim::DescTensor>(new_var));
}

template <>
Tensor empty_like<DescTensor>(const Tensor& x,
                              phi::DataType dtype,
                              const paddle::Place& place) {
  return empty<prim::DescTensor>(
      paddle::experimental::IntArray(x.shape()), x.dtype(), paddle::Place());
}

template <>
void set_output<DescTensor>(const paddle::Tensor& x_tmp, paddle::Tensor* x) {
  x->set_impl(x_tmp.impl());
}

template <>
void by_pass<DescTensor>(const paddle::Tensor& x, paddle::Tensor* real_out) {
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("assign");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  auto out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);

  set_output<DescTensor>(out, real_out);
}

}  // namespace paddle::prim
