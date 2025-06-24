/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <string>
#include <utility>

#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_tracer.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/api/profiler/common_event.h"
#include "paddle/phi/api/profiler/device_tracer.h"
#include "paddle/phi/api/profiler/event.h"  // import EventRole, TODO(TIEXING): remove later
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/profiler/host_event_recorder.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/nvtx.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/os_info.h"

COMMON_DECLARE_bool(enable_record_memory);
PHI_DECLARE_bool(enable_host_event_recorder_hook);

namespace paddle {

namespace framework {
class RuntimeContext;
}
namespace platform {

RecordOpInfoSupplement::RecordOpInfoSupplement(
    const std::string &type,
    const framework::AttributeMap &attrs,
    const framework::InferShapeContext &shape_ctx,
    const framework::RuntimeContext &ctx,
    uint64_t op_id) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  if (IsEnabled() == false) {
    return;
  }
  std::map<std::string, std::vector<phi::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  for (const auto &input : ctx.inputs) {
    input_shapes[input.first] = shape_ctx.GetInputsDim(input.first);
    dtypes[input.first] = shape_ctx.GetInputsVarType(input.first);
  }

  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance().RecordEvent(
      phi::PosixInNsec(), type, input_shapes, dtypes, attrs, op_id);
}

RecordOpInfoSupplement::RecordOpInfoSupplement(
    const std::string &type,
    const framework::AttributeMap &attrs,
    const framework::InferShapeContext &shape_ctx,
    const phi::KernelSignature &kernel_signature) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  if (IsEnabled() == false) {
    return;
  }
  std::map<std::string, std::vector<phi::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  for (auto input_name_char : kernel_signature.input_names) {
    std::string input_name(input_name_char);
    if (shape_ctx.HasInputs(input_name)) {
      input_shapes[input_name] = shape_ctx.GetInputsDim(input_name);
      dtypes[input_name] = shape_ctx.GetInputsVarType(input_name);
    }
  }
  uint64_t op_id = 0;
  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance().RecordEvent(
      phi::PosixInNsec(), type, input_shapes, dtypes, attrs, op_id);
}

}  // namespace platform
}  // namespace paddle
