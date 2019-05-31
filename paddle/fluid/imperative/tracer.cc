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

#include "paddle/fluid/imperative/tracer.h"
#include <string>
#include <utility>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward) {
  platform::RecordEvent event(type);

  size_t op_id = GenerateUniqueId();
  auto op =
      OpBase::Create(this, op_id, type, ins, outs, std::move(attrs), place);
  op->Run(ins, outs);

  if (trace_backward) {
    framework::OpDesc fwd_op(op->Type(), op->InputNameMap(),
                             op->OutputNameMap(), op->Attrs());
    op->TraceBackward(std::move(fwd_op), ins, outs);
    ops_.emplace(op_id, std::move(op));
  }
}

void Tracer::TraceOp(const framework::OpDesc& op_desc,
                     const NameVarBaseMap& ins, const NameVarBaseMap& outs,
                     const platform::Place& place, bool trace_backward) {
  platform::RecordEvent event(op_desc.Type());

  size_t op_id = GenerateUniqueId();
  auto op = OpBase::Create(this, op_id, op_desc, ins, outs, place);

  op->Run(ins, outs);

  if (trace_backward) {
    op->TraceBackward(op_desc, ins, outs);
    ops_.emplace(op_id, std::move(op));
  }
}

}  // namespace imperative
}  // namespace paddle
