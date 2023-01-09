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

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/prim/api/manual/prim_api/prim_api.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/api/include/tensor.h"
namespace paddle {
namespace prim {

template <>
Tensor pow<DescTensor>(const Tensor& x, const paddle::experimental::Scalar& y) {
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("pow");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("factor", y.to<float>());
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor scale<DescTensor>(const Tensor& x,
                         const paddle::experimental::Scalar& scale,
                         float bias,
                         bool bias_after_scale) {
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("scale");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("scale", scale.to<float>());
  op->SetAttr("bias", bias);
  op->SetAttr("bias_after_scale", bias_after_scale);
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor multiply<DescTensor>(const Tensor& x, const Tensor& y) {
  // Grad infershape
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("elementwise_mul");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetInput("Y",
               {std::static_pointer_cast<prim::DescTensor>(y.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

}  // namespace prim
}  // namespace paddle
