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

#include <string.h>
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
#include "paddle/fluid/prim/api/manual_prim/prim_manual_api.h"
#include "paddle/fluid/prim/api/manual_prim/utils/utils.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
namespace paddle {
namespace prim {

template <>
Tensor full<DescTensor>(const IntArray& shape,
                        const Scalar& value,
                        DataType dtype,
                        const Place& place) {
  // Grad infershape
  Tensor out = empty<DescTensor>({}, dtype, place);
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("fill_constant");
  op->SetAttr("shape", shape.GetData());
  op->SetAttr("value", value);
  op->SetAttr("dtype", paddle::framework::TransToProtoVarType(dtype));
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor cast<DescTensor>(const Tensor& x, DataType dtype) {
  Tensor out = empty<DescTensor>({}, DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("cast");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("in_dtype", paddle::framework::TransToProtoVarType(x.dtype()));
  op->SetAttr("out_dtype", paddle::framework::TransToProtoVarType(dtype));
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor reshape<DescTensor>(const Tensor& x, const IntArray& shape) {
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("reshape2");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  auto out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("shape", unsafe_vector_cast<int64_t, int>(shape.GetData()));
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

}  // namespace prim
}  // namespace paddle
