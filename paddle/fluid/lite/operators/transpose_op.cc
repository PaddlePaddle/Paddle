// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/operators/transpose_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

// Transpose
bool TransposeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  std::vector<int> axis = param_.axis;
  size_t axis_size = axis.size();
  // "The input tensor's rank(%d) should be equal to the axis's size(%d)",
  // x_rank, axis_size
  CHECK_OR_FALSE(x_rank == axis_size);

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    // Each element of Attribute axis should be a unique value
    // range from 0 to (dims - 1),
    // where the dims is the axis's size
    CHECK_OR_FALSE(axis[i] < static_cast<int>(axis_size) &&
                   ++count[axis[i]] == 1);
  }
  return true;
}

bool TransposeOp::InferShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  std::vector<int> axis = param_.axis;
  size_t axis_size = axis.size();
  // "The input tensor's rank(%d) should be equal to the axis's size(%d)",
  // x_rank, axis_size
  CHECK_OR_FALSE(x_rank == axis_size);

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    // Each element of Attribute axis should be a unique value
    // range from 0 to (dims - 1),
    // where the dims is the axis's size
    CHECK_OR_FALSE(axis[i] < static_cast<int>(axis_size) &&
                   ++count[axis[i]] == 1);
  }
  lite::DDim out_dims(x_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = x_dims[axis[i]];
  }
  param_.output->Resize(out_dims);
  return true;
}

bool TransposeOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();

  CHECK(scope->FindVar(x));
  CHECK(scope->FindVar(out));
  param_.x = GetVar<lite::Tensor>(scope, x);
  param_.output = GetMutableVar<lite::Tensor>(scope, out);

  param_.axis = op_desc.GetAttr<std::vector<int>>("axis");
  if (op_desc.HasAttr("use_mkldnn")) {
    param_.use_mkldnn = op_desc.GetAttr<bool>("use_mkldnn");
  }
  if (op_desc.HasAttr("data_format")) {
    param_.data_format = op_desc.GetAttr<std::string>("data_format");
  }
  return true;
}

// Transpose2
bool Transpose2Op::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  std::vector<int> axis = param_.axis;
  size_t axis_size = axis.size();
  // "The input tensor's rank(%d) should be equal to the axis's size(%d)",
  // x_rank, axis_size
  CHECK_OR_FALSE(x_rank == axis_size);

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    // Each element of Attribute axis should be a unique value
    // range from 0 to (dims - 1),
    // where the dims is the axis's size
    CHECK_OR_FALSE(axis[i] < static_cast<int>(axis_size) &&
                   ++count[axis[i]] == 1);
  }
  return true;
}

bool Transpose2Op::InferShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  std::vector<int> axis = param_.axis;
  size_t axis_size = axis.size();
  // "The input tensor's rank(%d) should be equal to the axis's size(%d)",
  // x_rank, axis_size
  CHECK_OR_FALSE(x_rank == axis_size);

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    // Each element of Attribute axis should be a unique value
    // range from 0 to (dims - 1),
    // where the dims is the axis's size
    CHECK_OR_FALSE(axis[i] < static_cast<int>(axis_size) &&
                   ++count[axis[i]] == 1);
  }
  lite::DDim out_dims(x_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = x_dims[axis[i]];
  }
  param_.output->Resize(out_dims);
  return true;
}

bool Transpose2Op::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();

  CHECK(scope->FindVar(x));
  CHECK(scope->FindVar(out));
  param_.x = GetVar<lite::Tensor>(scope, x);
  param_.output = GetMutableVar<lite::Tensor>(scope, out);

  param_.axis = op_desc.GetAttr<std::vector<int>>("axis");
  if (op_desc.HasAttr("use_mkldnn")) {
    param_.use_mkldnn = op_desc.GetAttr<bool>("use_mkldnn");
  }
  if (op_desc.HasAttr("data_format")) {
    param_.data_format = op_desc.GetAttr<std::string>("data_format");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(transpose, paddle::lite::operators::TransposeOp);
REGISTER_LITE_OP(transpose2, paddle::lite::operators::Transpose2Op);
