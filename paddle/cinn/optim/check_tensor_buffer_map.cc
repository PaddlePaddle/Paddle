// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/check_tensor_buffer_map.h"

namespace cinn {
namespace optim {

struct TensorBufferMapChecker : public ir::IRVisitorRequireReImpl<void> {
  TensorBufferMapChecker() : error_flag_(true) { tensor2buffer_.clear(); }
  void Visit(const ir::Expr x) { IRVisitorRequireReImpl::Visit(&x); }
  void Visit(const ir::Expr *x) { IRVisitorRequireReImpl::Visit(x); }

  void Visit(const ir::_Tensor_ *x) override;

  bool get_debug_status() {
    bool error_flag = error_flag_;
    error_flag_ = true;
    tensor2buffer_.clear();
    return error_flag;
  }
  bool operator()(const Expr *expr) {
    Visit(expr);
    return get_debug_status();
  }

#define VisitImpl(_TYPE) void Visit(const ir::_TYPE *x) override;
  VisitImpl(Minus);
  VisitImpl(Not);

  VisitImpl(Add);
  VisitImpl(Sub);
  VisitImpl(Mul);
  VisitImpl(Div);
  VisitImpl(Mod);
  VisitImpl(EQ);
  VisitImpl(NE);
  VisitImpl(LT);
  VisitImpl(LE);
  VisitImpl(GT);
  VisitImpl(GE);
  VisitImpl(And);
  VisitImpl(Or);
  VisitImpl(Min);
  VisitImpl(Max);

  VisitImpl(IfThenElse);
  VisitImpl(Block);
  VisitImpl(ScheduleBlock);
  VisitImpl(ScheduleBlockRealize);
  VisitImpl(For);
  VisitImpl(IntImm);
  VisitImpl(UIntImm);
  VisitImpl(FloatImm);
  VisitImpl(StringImm);
  VisitImpl(Cast);
  VisitImpl(PolyFor);
  VisitImpl(Select);
  VisitImpl(Call);
  VisitImpl(_Var_);
  VisitImpl(Load);
  VisitImpl(Store);
  VisitImpl(Alloc);
  VisitImpl(Free);
  VisitImpl(_Buffer_);
  VisitImpl(Let);
  VisitImpl(Reduce);
  VisitImpl(Ramp);
  VisitImpl(Broadcast);
  VisitImpl(FracOp);
  VisitImpl(Product);
  VisitImpl(Sum);
  VisitImpl(PrimitiveNode);
  VisitImpl(IntrinsicOp);
  VisitImpl(_BufferRange_);
  VisitImpl(_Dim_);

 private:
  bool error_flag_;
  std::map<std::string, const ir::IrNode *> tensor2buffer_;
};

void TensorBufferMapChecker::Visit(const ir::_Tensor_ *x) {
  VLOG(3) << "step into tensor buffer map check";
  std::string tensor_name = x->name;
  const ir::IrNode *buffer_ptr = x->buffer.ptr();
  if (this->tensor2buffer_.find(tensor_name) != this->tensor2buffer_.end()) {
    if (tensor2buffer_[tensor_name] != buffer_ptr) {
      this->error_flag_ = false;
      VLOG(3) << "tensor name [" << tensor_name
              << "] maps multiple buffer_ptrs [" << buffer_ptr << "] and ["
              << tensor2buffer_[tensor_name] << "]";
    } else {
      VLOG(3) << "tensor name [" << tensor_name << "] maps with buffer ptr["
              << buffer_ptr << "]";
    }
  } else {
    this->tensor2buffer_[tensor_name] = buffer_ptr;
    VLOG(3) << "add tensor: [" << tensor_name
            << "] with buffer_ptr:" << buffer_ptr;
  }
}

void TensorBufferMapChecker::Visit(const ir::Add *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Sub *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Mul *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Div *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Mod *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::EQ *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::NE *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::LT *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::LE *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::GT *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::GE *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::And *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Or *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Min *x) {
  Visit(x->a());
  Visit(x->b());
}
void TensorBufferMapChecker::Visit(const ir::Max *x) {
  Visit(x->a());
  Visit(x->b());
}

void TensorBufferMapChecker::Visit(const ir::Minus *x) { Visit(x->v()); }
void TensorBufferMapChecker::Visit(const ir::Not *x) { Visit(x->v()); }

void TensorBufferMapChecker::Visit(const ir::IntImm *x) {}
void TensorBufferMapChecker::Visit(const ir::UIntImm *x) {}
void TensorBufferMapChecker::Visit(const ir::FloatImm *x) {}
void TensorBufferMapChecker::Visit(const ir::StringImm *x) {}
void TensorBufferMapChecker::Visit(const ir::Cast *x) {
  TensorBufferMapChecker::Visit(x->v());
}
void TensorBufferMapChecker::Visit(const ir::For *x) {
  TensorBufferMapChecker::Visit(x->min);
  TensorBufferMapChecker::Visit(x->extent);
  TensorBufferMapChecker::Visit(x->body);
}
void TensorBufferMapChecker::Visit(const ir::PolyFor *x) {
  TensorBufferMapChecker::Visit(x->init);
  TensorBufferMapChecker::Visit(x->condition);
  TensorBufferMapChecker::Visit(x->inc);
  TensorBufferMapChecker::Visit(x->body);
}
void TensorBufferMapChecker::Visit(const ir::Select *x) {
  TensorBufferMapChecker::Visit(x->condition);
  TensorBufferMapChecker::Visit(x->true_value);
  TensorBufferMapChecker::Visit(x->false_value);
}
void TensorBufferMapChecker::Visit(const ir::IfThenElse *x) {
  TensorBufferMapChecker::Visit(x->condition);
  TensorBufferMapChecker::Visit(x->true_case);
  if (x->false_case.defined()) TensorBufferMapChecker::Visit(x->false_case);
}
void TensorBufferMapChecker::Visit(const ir::Block *x) {
  for (std::size_t i = 0; !x->stmts.empty() && i + 1 < x->stmts.size(); i++) {
    Visit(x->stmts[i]);
  }
  if (!x->stmts.empty()) {
    Visit(x->stmts.back());
  }
}
void TensorBufferMapChecker::Visit(const ir::Call *x) {
  if (!x->read_args.empty()) {
    for (std::size_t i = 0; i + 1 < x->read_args.size(); i++) {
      Visit(x->read_args[i]);
    }
    Visit(x->read_args.back());
  }

  if (!x->write_args.empty()) {
    for (std::size_t i = 0; i + 1 < x->write_args.size(); i++) {
      Visit(x->write_args[i]);
    }
    Visit(x->write_args.back());
  }
}

void TensorBufferMapChecker::Visit(const ir::_Var_ *x) {
  if (x->lower_bound.defined()) {
    TensorBufferMapChecker::Visit(x->lower_bound);
  }
  if (x->upper_bound.defined()) {
    TensorBufferMapChecker::Visit(x->upper_bound);
  }
}
void TensorBufferMapChecker::Visit(const ir::Load *x) {
  for (auto &idx : x->indices) TensorBufferMapChecker::Visit(&idx);
  TensorBufferMapChecker::Visit(x->tensor);
}
void TensorBufferMapChecker::Visit(const ir::Store *x) {
  TensorBufferMapChecker::Visit(x->value);
  TensorBufferMapChecker::Visit(x->tensor);
  for (auto &idx : x->indices) TensorBufferMapChecker::Visit(&idx);
}
void TensorBufferMapChecker::Visit(const ir::Alloc *x) {
  for (auto &e : x->extents) {
    TensorBufferMapChecker::Visit(&e);
  }
  if (x->condition.defined()) TensorBufferMapChecker::Visit(x->condition);
  if (x->body.defined()) {
    TensorBufferMapChecker::Visit(x->body);
  }
}
void TensorBufferMapChecker::Visit(const ir::Free *x) {
  TensorBufferMapChecker::Visit(x->destination);
}
void TensorBufferMapChecker::Visit(const ir::_Buffer_ *x) {
  for (auto &e : x->shape) {
    TensorBufferMapChecker::Visit(&e);
  }
  for (auto &e : x->strides) {
    TensorBufferMapChecker::Visit(&e);
  }
  TensorBufferMapChecker::Visit(x->elem_offset);
}
void TensorBufferMapChecker::Visit(const ir::Let *x) {
  TensorBufferMapChecker::Visit(x->symbol);
  if (x->body.defined()) TensorBufferMapChecker::Visit(x->body);
}
void TensorBufferMapChecker::Visit(const ir::Reduce *x) {
  if (x->init.defined()) TensorBufferMapChecker::Visit(x->init);
  PADDLE_ENFORCE_EQ(
      x->body.defined(),
      true,
      ::common::errors::InvalidArgument("The x->body is not defined"));
  TensorBufferMapChecker::Visit(x->body);
}

void TensorBufferMapChecker::Visit(const ir::Ramp *x) {
  TensorBufferMapChecker::Visit(x->base);
  TensorBufferMapChecker::Visit(x->stride);
}

void TensorBufferMapChecker::Visit(const ir::Broadcast *x) {
  TensorBufferMapChecker::Visit(x->value);
}

void TensorBufferMapChecker::Visit(const ir::FracOp *x) {
  TensorBufferMapChecker::Visit(x->a());
  TensorBufferMapChecker::Visit(x->b());
}

void TensorBufferMapChecker::Visit(const ir::Product *x) {
  for (auto &op : x->operands()) {
    TensorBufferMapChecker::Visit(&op);
  }
}

void TensorBufferMapChecker::Visit(const ir::Sum *x) {
  for (auto &op : x->operands()) {
    TensorBufferMapChecker::Visit(&op);
  }
}
void TensorBufferMapChecker::Visit(const ir::PrimitiveNode *x) {
  for (auto &args : x->arguments) {
    for (auto &arg : args) {
      TensorBufferMapChecker::Visit(&arg);
    }
  }
}

void TensorBufferMapChecker::Visit(const ir::IntrinsicOp *x) {
  switch (x->getKind()) {
    case ir::IntrinsicKind::kBufferGetDataHandle: {
      auto *n = llvm::dyn_cast<ir::intrinsics::BufferGetDataHandle>(x);
      Visit(&n->buffer);
    } break;
    case ir::IntrinsicKind::kBufferGetDataConstHandle: {
      auto *n = llvm::dyn_cast<ir::intrinsics::BufferGetDataConstHandle>(x);
      Visit(&n->buffer);
    } break;
    case ir::IntrinsicKind::kPodValueToX: {
      auto *n = llvm::dyn_cast<ir::intrinsics::PodValueToX>(x);
      Visit(&n->pod_value_ptr);
    } break;
    case ir::IntrinsicKind::kBuiltinIntrin: {
      auto *n = llvm::dyn_cast<ir::intrinsics::BuiltinIntrin>(x);
      for (auto &x : n->args) {
        Visit(&x);
      }
    } break;
  }
}

void TensorBufferMapChecker::Visit(const ir::_BufferRange_ *x) {
  PADDLE_ENFORCE_NOT_NULL(
      x, ::common::errors::InvalidArgument("Check that _BufferRange_ is null"));
  TensorBufferMapChecker::Visit(x->buffer);
  for (auto &var : x->ranges) {
    if (var->lower_bound.defined()) {
      TensorBufferMapChecker::Visit(&var->lower_bound);
    }
    if (var->upper_bound.defined()) {
      TensorBufferMapChecker::Visit(&var->upper_bound);
    }
  }
}

void TensorBufferMapChecker::Visit(const ir::ScheduleBlock *x) {
  PADDLE_ENFORCE_NOT_NULL(
      x, ::common::errors::InvalidArgument("ScheduleBlock is null"));
  for (auto &var : x->iter_vars) {
    if (var->lower_bound.defined()) {
      TensorBufferMapChecker::Visit(&var->lower_bound);
    }
    if (var->upper_bound.defined()) {
      TensorBufferMapChecker::Visit(&var->upper_bound);
    }
  }
  for (auto &buffer_region : x->read_buffers) {
    TensorBufferMapChecker::Visit(&buffer_region);
  }
  for (auto &buffer_region : x->write_buffers) {
    TensorBufferMapChecker::Visit(&buffer_region);
  }
  TensorBufferMapChecker::Visit(&(x->body));
}

void TensorBufferMapChecker::Visit(const ir::ScheduleBlockRealize *x) {
  PADDLE_ENFORCE_NOT_NULL(
      x, ::common::errors::InvalidArgument("ScheduleBlockRealize is null"));
  for (auto &value : x->iter_values) {
    TensorBufferMapChecker::Visit(&value);
  }
  TensorBufferMapChecker::Visit(x->schedule_block);
}

void TensorBufferMapChecker::Visit(const ir::_Dim_ *x) {}

bool CheckTensorBufferMap(const Expr *expr) {
  return TensorBufferMapChecker()(expr);
}
bool CheckTensorBufferMap(const Expr &expr) {
  return TensorBufferMapChecker()(&expr);
}

void CheckTensorBufferMap(const std::vector<ir::Expr> &expr,
                          const std::string &process) {
  for (auto e : expr) {
    bool flag = CheckTensorBufferMap(e);
    if (!flag) {
      VLOG(3) << "process [" << process << "]"
              << " has wrong tensor-buffer map in " << e;
    }
    PADDLE_ENFORCE_EQ(
        flag,
        true,
        ::common::errors::InvalidArgument("CheckTensorBufferMap fail"));
  }
}

void CheckTensorBufferMap(const std::vector<ir::Expr *> &expr,
                          const std::string &process) {
  for (auto e : expr) {
    bool flag = CheckTensorBufferMap(e);
    if (!flag) {
      VLOG(3) << "process [" << process << "]"
              << " has wrong tensor-buffer map in " << e;
    }
    PADDLE_ENFORCE_EQ(
        flag,
        true,
        ::common::errors::InvalidArgument("CheckTensorBufferMap fail"));
  }
}

void CheckTensorBufferMap(const Expr *expr, const std::string &process) {
  bool flag = CheckTensorBufferMap(expr);
  if (!flag) {
    VLOG(3) << "process [" << process << "]"
            << " has wrong tensor-buffer map in " << expr;
  }
  PADDLE_ENFORCE_EQ(
      flag,
      true,
      ::common::errors::InvalidArgument("CheckTensorBufferMap fail"));
}

void CheckTensorBufferMap(const Expr &expr, const std::string &process) {
  bool flag = CheckTensorBufferMap(expr);
  if (!flag) {
    VLOG(3) << "process [" << process << "]"
            << " has wrong tensor-buffer map in " << expr;
  }
  PADDLE_ENFORCE_EQ(
      flag,
      true,
      ::common::errors::InvalidArgument("CheckTensorBufferMap fail"));
}

}  // namespace optim
}  // namespace cinn
