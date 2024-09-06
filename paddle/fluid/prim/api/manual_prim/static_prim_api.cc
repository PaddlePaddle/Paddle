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

#include <cstring>
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
namespace paddle::prim {

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
  switch (dtype) {
    case phi::DataType::FLOAT16:
    case phi::DataType::BFLOAT16:
      op->SetAttr("str_value", std::to_string(value.to<float>()));
      break;
    case phi::DataType::FLOAT32:
      op->SetAttr("value", value.to<float>());
      break;
    case phi::DataType::FLOAT64: {
      std::stringstream ss;
      ss << std::setprecision(20) << value.to<double>();
      op->SetAttr("str_value", ss.str());
      break;
    }
    case phi::DataType::BOOL:
      op->SetAttr("str_value", std::to_string(value.to<bool>()));
      break;
    case phi::DataType::INT8:
      op->SetAttr("str_value", std::to_string(value.to<int8_t>()));
      break;
    case phi::DataType::UINT8:
      op->SetAttr("str_value", std::to_string(value.to<uint8_t>()));
      break;
    case phi::DataType::INT16:
      op->SetAttr("str_value", std::to_string(value.to<int16_t>()));
      break;
    case phi::DataType::UINT16:
      op->SetAttr("str_value", std::to_string(value.to<uint16_t>()));
      break;
    case phi::DataType::INT32:
      op->SetAttr("str_value", std::to_string(value.to<int32_t>()));
      break;
    case phi::DataType::UINT32:
      op->SetAttr("str_value", std::to_string(value.to<uint32_t>()));
      break;
    case phi::DataType::INT64:
      op->SetAttr("str_value", std::to_string(value.to<int64_t>()));
      break;
    case phi::DataType::UINT64:
      op->SetAttr("str_value", std::to_string(value.to<uint64_t>()));
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "We support "
          "bool/float16/bfloat16/float32/float64/int8/int16/int32/int64/uint8/"
          "uint16/"
          "uint32/uint64 for full, but we got data type: %s",
          phi::DataTypeToString(dtype)));
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
Tensor slice<DescTensor>(const Tensor& input,
                         const std::vector<int64_t>& axes,
                         const IntArray& starts,
                         const IntArray& ends,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis) {
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("slice");
  op->SetInput(
      "Input",
      {std::static_pointer_cast<prim::DescTensor>(input.impl())->Name()});
  auto out = empty<DescTensor>({}, phi::DataType::FLOAT32, paddle::Place());
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("axes", unsafe_vector_cast<int64_t, int>(axes));
  op->SetAttr("starts", unsafe_vector_cast<int64_t, int>(starts.GetData()));
  op->SetAttr("ends", unsafe_vector_cast<int64_t, int>(ends.GetData()));
  op->SetAttr("infer_flags", unsafe_vector_cast<int64_t, int>(infer_flags));
  op->SetAttr("decrease_axis", unsafe_vector_cast<int64_t, int>(decrease_axis));
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}

}  // namespace paddle::prim
