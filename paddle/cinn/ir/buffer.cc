// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/buffer.h"

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_visitor.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

std::string TensorGetBufferName(const _Tensor_ *tensor) {
  CHECK(!tensor->name.empty());
  CHECK(!utils::Startswith(tensor->name, "_"))
      << "the name with prefix _ is not allowed for tensor. Current tensor's "
         "name is: "
      << tensor->name;
  return "_" + tensor->name;
}
std::string BufferGetTensorName(const _Buffer_ *buffer) {
  CHECK(!buffer->name.empty());
  CHECK(utils::Startswith(buffer->name, "_"))
      << "buffer's name should start with _";
  return buffer->name.substr(1);
}

const _Buffer_ *Buffer::operator->() const { return IrNodeRef::As<_Buffer_>(); }
_Buffer_ *Buffer::operator->() { return IrNodeRef::As<_Buffer_>(); }

Buffer _Buffer_::Make(Var data,
                      Type dtype,
                      const std::vector<Expr> &shape,
                      const std::vector<Expr> &strides,
                      Expr elem_offset,
                      const std::string &name,
                      const std::string &scope,
                      int data_alignment,
                      int offset_factor,
                      Target target) {
  CHECK(dtype.valid());
  CHECK(!dtype.is_unk());
  CHECK(!dtype.is_void());
  auto *node = common::make_shared<_Buffer_>();
  node->shape = shape;
  node->strides = strides;
  node->elem_offset = elem_offset;
  node->name = name;
  node->scope = scope;
  node->data_alignment = data_alignment;
  node->offset_factor = offset_factor;
  node->target = target;
  node->dtype = dtype;
  return Buffer(node);
}

Buffer _Buffer_::Make(const std::string &name, const std::vector<Expr> &shape) {
  auto *node = common::make_shared<_Buffer_>();
  node->name = name;
  node->shape = shape;
  node->dtype = Void();
  return Buffer(node);
}

Buffer _Buffer_::Make() {
  auto *node = common::make_shared<_Buffer_>();
  node->dtype = Void();
  return Buffer(node);
}

IrNodeTy _Buffer_::node_type() const { return _node_type_; }

void _Buffer_::BindTo(const Tensor &tensor) { BindTo(tensor.As<_Tensor_>()); }
void _Buffer_::BindTo(const _Tensor_ *tensor) {
  if (name.empty()) name = TensorGetBufferName(tensor);
  if (type().is_unk()) set_type(tensor->type());
  CHECK(!tensor->shape.empty())
      << "Tensor should have shape to bind to a Buffer";
  shape = tensor->shape;
  binded_tensors_names_.insert(tensor->name);
}
void _Buffer_::Unbind(const _Tensor_ *tensor) {
  binded_tensors_names_.erase(tensor->name);
}

Var _Buffer_::buffer_addr() const {
  auto thetype = type().ElementOf();
  thetype.set_cpp_handle();
  return _Var_::Make(name, thetype);
}

int _Buffer_::numel() const {
  int res = 1;
  for (auto &i : shape) {
    CHECK(i.is_constant());
    res *= i.as_int32();
  }
  return res;
}

void _Buffer_::Verify() const {
  CHECK(!shape.empty());
  CHECK(!name.empty());
  CHECK(dtype.valid());
}

Expr Buffer::DestroyExpr() const {
  auto *node = operator->();
  return runtime::IntrinsicCall(Void(),
                                runtime::intrinsic::buffer_destroy,
                                {ir::_Var_::Make(node->name, node->type())});
}

Expr _BufferRange_::Make(const Expr &buffer, const std::vector<Var> &ranges) {
  auto node = make_shared<_BufferRange_>();
  node->buffer = buffer;
  node->ranges = ranges;
  return Expr(node);
}
void _BufferRange_::Verify() const {
  auto *buffer_ptr = buffer.As<_Buffer_>();
  CHECK(buffer_ptr);
}
Expr _BufferRange_::Copy() const {
  auto node = make_shared<_BufferRange_>();
  node->buffer = buffer;
  node->ranges = ranges;
  node->set_type(type());
  return Expr(node);
}

bool BufferRange::operator==(const BufferRange &x) const {
  auto this_buffer = operator->()->buffer.As<_Buffer_>();
  auto other_buffer = x->buffer.As<_Buffer_>();
  CHECK(this_buffer);
  CHECK(other_buffer);
  if (this_buffer != other_buffer) return false;
  if (x->ranges.size() != operator->()->ranges.size()) return false;
  for (int i = 0; i < x->ranges.size(); i++) {
    Var this_range = operator->()->ranges[i];
    Var other_range = x->ranges[i];
    if (!is_zero(this_range->lower_bound - other_range->lower_bound))
      return false;
    if (!is_zero(this_range->upper_bound - other_range->upper_bound))
      return false;
  }
  return true;
}
bool BufferRange::operator!=(const BufferRange &x) const {
  return !(*this == x);
}
BufferRange &BufferRange::operator=(_BufferRange_ *x) {
  *this = BufferRange(x);
  return *this;
}
BufferRange &BufferRange::operator=(const _BufferRange_ *x) {
  auto node = make_shared<_BufferRange_>();
  node->buffer = x->buffer;
  node->ranges = x->ranges;
  node->set_type(x->type());
  *this = BufferRange(node);
  return *this;
}

}  // namespace ir
}  // namespace cinn
