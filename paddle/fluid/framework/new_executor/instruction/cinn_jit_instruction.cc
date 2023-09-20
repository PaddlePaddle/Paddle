// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"

namespace paddle {
namespace framework {

// TODO(Aurelius84): Think deeply what's the responsibility is it.
// Currently it assumes CinnLaunchContext role.
class JitContext {
 public:
  cinn_buffer_t* GetCinnBufferOfVar(const std::string& name) {
    auto res = paddle2argument_.find(name);
    PADDLE_ENFORCE_NE(
        res,
        paddle2argument_.end(),
        platform::errors::NotFound(
            "Variable(%s) not found in compilation result", name));
    return static_cast<cinn_buffer_t*>(res->second);
  }

  // NOTE(Aurelius84): Before running each instruction, we should share Tensor
  // memory from paddle scope with cinn_buffer_t from cinn scope including
  // inputs and outputs.
  void ShareMemToCinn(const std::string& var_name,
                      const phi::Place& place,
                      Scope* scope) {
    cinn_buffer_t* buffer = GetCinnBufferOfVar(var_name);
    auto* tensor = scope->GetVar(var_name)->GetMutable<phi::DenseTensor>();
    // TODO(Aurelius84): Maybe we should consider to unify the Scope
    // structure between paddle and cinn, so that we don't need to develop
    // the glue code.
    buffer->memory = reinterpret_cast<uint8_t*>(tensor->mutable_data(
        place, paddle2cinn::TransToPaddleDataType(buffer->type)));
  }

  // TODO(Aurelius84): Add logic to parse stream for different device.
  void* GetStream() { return nullptr; }

 private:
  // because a cinn_pod_value_t does not own a cinn_buffer_t object,
  // an extra stroage is necessary to keep those objects and they can
  // not be released until the runtime program finish execution.
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers_;
  // this map saves all execution arguments with their cinn names as key,
  // and it is passed to the Execute interface of a cinn runtime program.
  std::map<std::string, cinn_pod_value_t> name2argument_;
  // this map saves all execution arguments with paddle variables as key,
  // this map conbine name2argument_ and paddle2cinn_varmap_
  std::map<std::string, cinn_pod_value_t> paddle2argument_;
};

// TODO(Aurelius84): Impl should hold JitContext instance to
// deliver the device context for 'instr->Run' and responsible
// to deal with inner buffer_t shareing between framework::Scope
// and cinn::Scope.
class CinnJitInstruction::Impl {
  using Instruction = cinn::hlir::framework::Instruction;

 public:
  explicit Impl(Instruction* instr) : instr_(instr) {}
  // TODO(Aurelus84): Support to specify name2podargs and stream arguments.
  void Run() {
    PADDLE_ENFORCE_NOT_NULL(
        instr_, platform::errors::NotFound("instr_ should not be NULL"));
    instr_->Run(/*name2podargs=*/nullptr,
                false,
                /*stream=*/nullptr,
                /*use_cache=*/true);
  }
  const Instruction* pointer() const { return instr_; }

 private:
  Instruction* instr_{nullptr};
};

CinnJitInstruction::CinnJitInstruction(size_t id,
                                       const platform::Place& place,
                                       ::pir::Operation* op,
                                       Scope* scope)
    : InstructionBase(id, place) {
  // TODO(Aurelius84): We shall simplify members of JitKernelOp to make it
  // only hold related function ptrs. Impl is the real runtime data structure
  // responsible to construct hlir::framework::Instruction.
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  impl_ = std::make_shared<Impl>(jit_kernel_op.instruction());
  op_ = op;
}

void CinnJitInstruction::Run() {
  VLOG(6) << "Run cinn jit_kernel_op : " << Name();
  impl_->Run();
}

const std::string& CinnJitInstruction::Name() const {
  // TODO(Aurelius84): Consider the case for instrucitons constaning
  // multipule function ptrs and function names.
  return impl_->pointer()->function_name();
}

}  // namespace framework
}  // namespace paddle
