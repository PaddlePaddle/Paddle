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
Tensor transpose<DescTensor>(const Tensor& x, const std::vector<int>& perm) {
  Tensor out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  Tensor xshape = empty<DescTensor>({}, x.dtype(), paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("transpose2");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetAttr("axis", perm);
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetOutput(
      "XShape",
      {std::static_pointer_cast<prim::DescTensor>(xshape.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor pow<DescTensor>(const Tensor& x, const Scalar& y) {
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
                         const Scalar& scale,
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

template <>
Tensor unsqueeze<DescTensor>(const Tensor& x, const IntArray& axis) {
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("unsqueeze2");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  std::vector<int> new_shape(axis.GetData().begin(), axis.GetData().end());
  op->SetAttr("axes", new_shape);
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor expand<DescTensor>(const Tensor& x, const IntArray& shape) {
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("expand_v2");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  std::vector<int> new_shape(shape.GetData().begin(), shape.GetData().end());
  op->SetAttr("shape", new_shape);
  op->CheckAttrs();
  op->InferVarType(block);
  return out;
}

template <>
Tensor divide<DescTensor>(const Tensor& x, const Tensor& y) {
  // Grad infershape
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("elementwise_div");
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
  PADDLE_ENFORCE_EQ(
      ((dtype == DataType::FLOAT32) || (dtype == DataType::FLOAT64) ||
       (dtype == DataType::FLOAT16)),
      true,
      phi::errors::InvalidArgument(
          "We only support float32/float16 for full, but we got data type: %s",
          phi::DataTypeToString(dtype)));
  if (dtype == phi::DataType::FLOAT32) {
    op->SetAttr("value", value.to<float>());
  } else if (dtype == phi::DataType::FLOAT64) {
    op->SetAttr("str_value", std::to_string(value.to<double>()));
  } else if (dtype == phi::DataType::FLOAT16) {
    op->SetAttr("str_value", std::to_string(value.to<float>()));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "We only support float64/float32/float16 for full"));
  }
  op->SetAttr("dtype", paddle::framework::TransToProtoVarType(dtype));
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor sum<DescTensor>(const Tensor& x,
                       const IntArray& axis,
                       DataType dtype,
                       bool keepdim) {
  // Grad infershape
  Tensor out = empty<DescTensor>({}, dtype, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("reduce_sum");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  std::vector<int> res;
  for (auto value : axis.GetData()) {
    res.push_back(static_cast<int>(value));
  }
  op->SetAttr("dim", res);
  op->SetAttr("keep_dim", keepdim);
  op->SetAttr("dtype", paddle::framework::TransToProtoVarType(dtype));
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor reshape<DescTensor>(const Tensor& x, const IntArray& shape) {
  // Grad infershape
  Tensor out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  Tensor xshape = empty<DescTensor>({}, x.dtype(), paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("reshape2");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  std::vector<int> res;
  for (auto value : shape.GetData()) {
    // TODO(jiabin): This cast is not safe for now, find a way to handle this.
    res.push_back(static_cast<int>(value));
  }
  op->SetAttr("shape", res);
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetOutput(
      "XShape",
      {std::static_pointer_cast<prim::DescTensor>(xshape.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

template <>
Tensor exp<DescTensor>(const Tensor& x) {
  Tensor out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("exp");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}
}  // namespace prim
}  // namespace paddle
