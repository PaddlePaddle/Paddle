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

#include "paddle/cinn/ir/operation.h"

#include <memory>

#include "paddle/cinn/common/common.h"

namespace cinn {
namespace ir {

Operation PlaceholderOp::Make(const std::string &name,
                              const std::vector<Expr> &shape,
                              Type dtype) {
  auto n = make_shared<PlaceholderOp>();
  n->name = name;
  n->shape = shape;
  n->set_type(dtype);
  return Operation(n);
}

Operation PlaceholderOp::Make(const std::string &name,
                              const std::vector<Dim> &sym_shape,
                              Type dtype) {
  auto n = make_shared<PlaceholderOp>();
  n->name = name;
  n->sym_shape = sym_shape;
  for (int i = 0; i < sym_shape.size(); i++) {
    n->shape.emplace_back(sym_shape[i]->dim_expr);
  }
  n->set_type(dtype);
  return Operation(n);
}

const char *PlaceholderOp::func_type() const { return "placeholder_op"; }

const char *ComputeOp::func_type() const { return "compute_op"; }

Operation ComputeOp::Make(const std::string &name,
                          ComputeOp::handle_t handle,
                          const std::vector<Expr> &shape,
                          const std::vector<Expr> &domain,
                          const std::vector<Var> &reduce_axis,
                          const std::map<std::string, IrNodeRef> &attrs,
                          const std::string &tag) {
  auto n = make_shared<ComputeOp>();
  n->name = name;
  n->producer_fn = handle;
  n->shape = domain;
  n->reduce_axis = reduce_axis;
  n->tag = tag;
  n->attrs = attrs;
  n->axis = cinn::common::GenDefaultAxis(domain.size());
  std::vector<Expr> tmp_axis;
  for (auto &x : n->axis) {
    tmp_axis.push_back(x);
  }
  n->body = {handle(tmp_axis)};
  n->reduce_axis = reduce_axis;
  return Operation(n);
}

Operation CallOp::Make(const std::string &call_target, Expr call_op) {
  auto n = make_shared<CallOp>();
  n->call_expr = call_op;
  return Operation(n);
}

Operation PrecedingViewOp::Make(const Tensor &tensor, int preceding_axis) {
  return Operation();
}

const char *PrecedingViewOp::func_type() const {
  return PrecedingViewOp::__func_type__;
}

const char *CallOp::func_type() const { return __func_type__; }

const char *ComputeOp::__func_type__ = "compute_op";
const char *PlaceholderOp::__func_type__ = "placeholder_op";
const char *CallOp::__func_type__ = "call_op";

const std::string &CallOp::target() const {
  auto *call = call_expr.As<ir::Call>();
  PADDLE_ENFORCE_NOT_NULL(call,
                          ::common::errors::InvalidArgument(
                              "The 'call_expr' must be of type 'ir::Call'."));
  return call->name;
}
std::vector<Expr> &CallOp::write_args() {
  auto *call = call_expr.As<ir::Call>();
  PADDLE_ENFORCE_NOT_NULL(call,
                          ::common::errors::InvalidArgument(
                              "The 'call_expr' must be of type 'ir::Call'."));
  return call->write_args;
}
std::vector<Expr> &CallOp::read_args() {
  auto *call = call_expr.As<ir::Call>();
  PADDLE_ENFORCE_NOT_NULL(call,
                          ::common::errors::InvalidArgument(
                              "The 'call_expr' must be of type 'ir::Call'."));
  return call->read_args;
}
const std::vector<Expr> &CallOp::write_args() const {
  auto *call = call_expr.As<ir::Call>();
  PADDLE_ENFORCE_NOT_NULL(call,
                          ::common::errors::InvalidArgument(
                              "The 'call_expr' must be of type 'ir::Call'."));
  return call->write_args;
}
const std::vector<Expr> &CallOp::read_args() const {
  auto *call = call_expr.As<ir::Call>();
  PADDLE_ENFORCE_NOT_NULL(call,
                          ::common::errors::InvalidArgument(
                              "The 'call_expr' must be of type 'ir::Call'."));
  return call->read_args;
}
std::vector<Expr> CallOp::args() const {
  std::vector<Expr> args;
  auto &rargs = read_args();
  auto &wargs = write_args();
  args.insert(std::end(args), rargs.begin(), rargs.end());
  args.insert(std::end(args), wargs.begin(), wargs.end());
  return args;
}
const char *PrecedingViewOp::__func_type__ = "preceding_view_op";

const char *BufferShareOp::__func_type__ = "buffer_share_op";
const char *BufferShareOp::func_type() const { return __func_type__; }

}  // namespace ir
}  // namespace cinn
