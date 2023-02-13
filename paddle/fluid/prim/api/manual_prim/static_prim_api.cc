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

}  // namespace prim
}  // namespace paddle
