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

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/event.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace framework {

std::vector<int> GetValueIds(
    pir::Value value,
    Scope* inner_scope,
    const std::unordered_map<pir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  std::vector<int> ids;
  auto& var_name = value_2_var_name.at(value);
  ids.push_back(var_name_2_id.at(var_name));
  // NOTE(zhangbo): Value maybe a VariableRefArray
  auto var = inner_scope->FindVar(var_name);
  if (var->IsType<paddle::framework::VariableRefArray>()) {
    auto& var_array = var->Get<paddle::framework::VariableRefArray>();
    for (auto item : var_array) {
      ids.push_back(var_name_2_id.at(variable_2_var_name.at(item)));
    }
  }
  return ids;
}

platform::DeviceContext* ParseDeviceContext(
    pir::Operation* op,
    platform::DeviceContext* origin_dev_ctx,
    const platform::Place& place,
    const std::string& execution_stream,
    const int stream_priority) {
  auto& op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  interpreter::ContextManager& ctx_manager =
      interpreter::ContextManager::Instance();

  platform::DeviceContext* dev_ctx = nullptr;

  // only gpu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (platform::is_gpu_place(place) || platform::is_custom_place(place)) {
    VLOG(6) << "Parse DeviceContext for " << op_name
            << ", execution stream = " << execution_stream;
    if (execution_stream != kDefaultStream) {
      dev_ctx = ctx_manager
                    .Get(std::string(kCustomStream) + "-" + execution_stream,
                         place,
                         stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

    if (op_name == interpreter::kMemcpyD2H) {
      dev_ctx = ctx_manager.Get(std::string(kD2HStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    } else if (op_name == interpreter::kMemcpyH2D) {
      dev_ctx = ctx_manager.Get(std::string(kH2DStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // NOTE(Ruibiao): Here supports multi-stream overlap for c_allreduce_sum
    // with use_cal_stream==false by returning a device context getting from the
    // global NCCLCommContext instance. Because when use_calc_stream==false, in
    // OP kernel, the NCCL communication will be launched to the stream directly
    // getting from the global NCCLCommContext instance rather than the
    // DeviceContext passed from executor (see CAllReduceOpCUDAKernel in
    // c_allreduce_op.h). Now it is just a temporary solution for ONLY
    // c_allreduce_sum which is used in ResNet50 distributed training.
    if (op_name == "c_allreduce_sum" && op_attributes.at("use_calc_stream")
                                                .dyn_cast<pir::BoolAttribute>()
                                                .data() == false) {
      int ring_id =
          op_attributes.at("ring_id").dyn_cast<pir::Int32Attribute>().data();
      if (FLAGS_dynamic_static_unified_comm) {
        const auto& comm_context_manager =
            phi::distributed::CommContextManager::GetInstance();
        dev_ctx = static_cast<platform::DeviceContext*>(
            static_cast<phi::distributed::NCCLCommContext*>(
                comm_context_manager.Get(std::to_string(ring_id)))
                ->GetDevContext());
      } else {
        dev_ctx = platform::NCCLCommContext::Instance()
                      .Get(ring_id, place)
                      ->dev_context();
      }
      return dev_ctx;
    }
#endif
  }

  if (origin_dev_ctx != nullptr) {
    interpreter::SetDeviceCommContext(op, origin_dev_ctx);
  }
  return origin_dev_ctx;
}

OpFuncType AnalyseOpFuncType(pir::Operation* op, const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  auto kernel_key = op->attributes()
                        .at("kernel_key")
                        .dyn_cast<dialect::KernelAttribute>()
                        .data();
  if (phi::TransToPhiPlace(kernel_key.backend()).GetType() ==
      phi::AllocationType::CPU) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(interpreter::IsSupportedHeterPlace(place),
                    true,
                    phi::errors::Fatal("Unsupported current place %s", place));

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  auto& op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  if (op_name == "pd_op.coalesce_tensor" &&
      (!platform::is_xpu_place(place) ||
       op->attribute<pir::BoolAttribute>("persist_output").data() == false) &&
      op->attribute<pir::BoolAttribute>("set_constant").data() == false &&
      op->attribute<pir::BoolAttribute>("copy_data").data() == false) {
    return OpFuncType::kGpuSync;
  }

  // for memcpy explicitly called by user
  if (platform::is_gpu_place(place) && op_name == "pd_op.memcpy_d2h") {
    return OpFuncType::kGpuSync;
  }

  if (op_name == "pd_op.shape") {
    return OpFuncType::kGpuSync;
  }
  return OpFuncType::kGpuAsync;
}

}  // namespace framework
}  // namespace paddle
