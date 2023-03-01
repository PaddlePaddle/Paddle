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
Tensor reshape<DescTensor>(const Tensor& x, const IntArray& shape) {
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  // TODO(cxxly): Fix test_resnet_prim_cinn error when SetType("reshape2")
  op->SetType("reshape");
  op->SetInput("X",
               {std::static_pointer_cast<prim::DescTensor>(x.impl())->Name()});
  // Tensor out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  auto out = empty<DescTensor>({}, x.dtype(), paddle::Place());
  op->SetOutput(
      "Out", {std::static_pointer_cast<prim::DescTensor>(out.impl())->Name()});
  op->SetAttr("shape", unsafe_vector_cast<int64_t, int>(shape.GetData()));
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
  switch (dtype) {
    case phi::DataType::FLOAT16:
      op->SetAttr("str_value", std::to_string(value.to<float>()));
      break;
    case phi::DataType::FLOAT32:
      op->SetAttr("value", value.to<float>());
      break;
    case phi::DataType::FLOAT64:
      op->SetAttr("str_value", std::to_string(value.to<double>()));
      break;
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
      PADDLE_THROW(phi::errors::Unimplemented(
          "We support "
          "bool/float16/float32/float64/int8/int16/int32/int64/uint8/uint16/"
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
std::vector<Tensor> split<DescTensor>(const Tensor& x,
                                      const IntArray& sections,
                                      const Scalar& axis) {
  int elem_num = sections.size();
  std::vector<std::string> outs_name;
  std::vector<Tensor> outs;
  for (int i = 0; i < elem_num; ++i) {
    Tensor out = empty<DescTensor>({}, x.dtype(), paddle::Place());
    std::string out_name =
        std::static_pointer_cast<prim::DescTensor>(out.impl())->Name();
    outs_name.push_back(std::move(out_name));
    outs.push_back(out);
  }
  framework::BlockDesc* block = StaticCompositeContext::Instance().GetBlock();
  framework::OpDesc* op = block->AppendOp();
  op->SetType("split");
  op->SetAttr("sections", sections.GetData());
  op->SetAttr("axis", axis.to<int>());
  op->SetOutput("Out", outs_name);
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return outs;
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
  op->SetAttr("in_dtype", static_cast<int>(x.dtype()));
  op->SetAttr("out_dtype", static_cast<int>(dtype));
  op->CheckAttrs();
  op->InferVarType(block);
  op->InferShape(*block);
  return out;
}
}  // namespace prim
}  // namespace paddle
