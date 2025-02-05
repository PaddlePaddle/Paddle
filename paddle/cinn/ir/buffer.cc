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

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

std::string TensorGetBufferName(const _Tensor_ *tensor) {
  PADDLE_ENFORCE_EQ(
      !tensor->name.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Tensor name is empty. Ensure that the tensor has a valid name."));
  PADDLE_ENFORCE_EQ(utils::StartsWith(tensor->name, "_"),
                    false,
                    ::common::errors::InvalidArgument(
                        "The name with prefix '_' is not allowed "
                        "for tensor. Current tensor's name is: %s",
                        tensor->name));
  return "_" + tensor->name;
}

std::string BufferGetTensorName(const _Buffer_ *buffer) {
  PADDLE_ENFORCE_EQ(
      !buffer->name.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Buffer name is empty. Ensure that the buffer has a valid name."));
  PADDLE_ENFORCE_EQ(
      utils::StartsWith(buffer->name, "_"),
      true,
      ::common::errors::InvalidArgument(
          "Buffer's name should start with '_'. Current buffer's name is: %s",
          buffer->name));
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
  PADDLE_ENFORCE_EQ(dtype.valid(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The data type (dtype) is invalid. Ensure that dtype "
                        "is properly initialized and valid."));
  PADDLE_ENFORCE_EQ(dtype.is_unk(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The data type (dtype) is unknown. Ensure that dtype "
                        "is properly initialized and known."));
  PADDLE_ENFORCE_EQ(dtype.is_void(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The data type (dtype) is void. Ensure that dtype is "
                        "properly initialized and not void."));

  auto *node = cinn::common::make_shared<_Buffer_>();
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
  auto *node = cinn::common::make_shared<_Buffer_>();
  node->name = name;
  node->shape = shape;
  node->dtype = Void();
  return Buffer(node);
}

Buffer _Buffer_::Make() {
  auto *node = cinn::common::make_shared<_Buffer_>();
  node->dtype = Void();
  return Buffer(node);
}

IrNodeTy _Buffer_::node_type() const { return _node_type_; }

void _Buffer_::BindTo(const Tensor &tensor) { BindTo(tensor.As<_Tensor_>()); }
void _Buffer_::BindTo(const _Tensor_ *tensor) {
  if (name.empty()) name = TensorGetBufferName(tensor);
  if (type().is_unk()) set_type(tensor->type());
  PADDLE_ENFORCE_EQ(
      !tensor->shape.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Tensor should have a shape to bind to a Buffer. Ensure that the "
          "tensor's shape is properly initialized and not empty."));

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

int64_t _Buffer_::numel() const {
  int64_t res = 1;
  for (auto &i : shape) {
    PADDLE_ENFORCE_EQ(i.is_constant(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The value of 'i' is not constant. Ensure that 'i' "
                          "is a constant value before proceeding."));

    if (i->type() == Int(64)) {
      res *= i.as_int64();
    } else {
      res *= i.as_int32();
    }
  }
  return res;
}

ir::Expr _Buffer_::SymbolicNumel() const {
  ir::Expr res{1};
  for (auto &i : shape) {
    res = res * i;
  }
  return optim::ArithSimplify(res);
}

void _Buffer_::Verify() const {
  PADDLE_ENFORCE_EQ(
      !shape.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Buffer shape is empty. Ensure that the buffer has a valid shape."));
  PADDLE_ENFORCE_EQ(
      !name.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Buffer name is empty. Ensure that the buffer has a valid name."));
  PADDLE_ENFORCE_EQ(dtype.valid(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Buffer data type (dtype) is invalid. Ensure that "
                        "dtype is properly initialized and valid."));
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
  PADDLE_ENFORCE_NOT_NULL(buffer_ptr,
                          ::common::errors::InvalidArgument(
                              "buffer_ptr is null. Ensure that buffer_ptr "
                              "is properly initialized and not null."));
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
  PADDLE_ENFORCE_NOT_NULL(this_buffer,
                          ::common::errors::InvalidArgument(
                              "this_buffer is null. Ensure that this_buffer is "
                              "properly initialized and not null."));
  PADDLE_ENFORCE_NOT_NULL(other_buffer,
                          ::common::errors::InvalidArgument(
                              "other_buffer is null. Ensure that other_buffer "
                              "is properly initialized and not null."));

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
