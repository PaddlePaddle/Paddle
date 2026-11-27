// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/op_schedule_analyzer.h"

#include <algorithm>
#include <string>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"

namespace paddle {
namespace framework {
namespace interpreter {

namespace {

// Tier 1a: Ops that perform blocking device/host synchronization, file/network
// I/O, or interactive side effects; always classified as kBlockingDeviceIO.
const std::unordered_set<std::string>& BlockingDeviceIOOpNames() {
  static const std::unordered_set<std::string> kBlockingOps = {
      "pd_op.memcpy_d2h",
      "pd_op.memcpy_d2h_multi_io",
      "pd_op.memcpy_h2d",
      "pd_op.memcpy",
      "pd_op.coalesce_tensor",
      "pd_op.read_file",
      "pd_op.print",
      "pd_op.assert",
      "pd_op.send_v2",
      "pd_op.recv_v2",
      "pd_op.p_send",
      "pd_op.p_recv",
      "pd_op.barrier",
      "pd_op.all_reduce",
      "pd_op.all_gather",
      "pd_op.broadcast",
      "pd_op.reduce",
      "pd_op.reduce_scatter",
      "pd_op.all_to_all",
      "pd_op.c_allreduce_sum",
      "pd_op.c_sync_calc_stream",
      "pd_op.c_sync_comm_stream",
      "pd_op.sync_batch_norm",
      "pd_op.check_finite_and_unscale",
      "pd_op.update_loss_scaling",
      "pd_op.fetch",
      "pd_op.feed",
      "pd_op.shadow_feed",
      "pd_op.shadow_feed_tensors",
      "pd_op.share_data",
      "pd_op.load_combine",
      "pd_op.save_combine",
  };
  return kBlockingOps;
}

// Tier 1b: CPU ops whose cost can be superlinear, data-dependent, or heavy
// even for small tensors; always classified as kCompute.
const std::unordered_set<std::string>& HeavyOrSuperlinearCpuOpNames() {
  static const std::unordered_set<std::string> kHeavyOps = {
      "pd_op.svd",
      "pd_op.eig",
      "pd_op.eigh",
      "pd_op.eigvals",
      "pd_op.eigvalsh",
      "pd_op.cholesky",
      "pd_op.cholesky_solve",
      "pd_op.qr",
      "pd_op.lu",
      "pd_op.lu_unpack",
      "pd_op.solve",
      "pd_op.inverse",
      "pd_op.matrix_power",
      "pd_op.pinv",
      "pd_op.lstsq",
      "pd_op.det",
      "pd_op.slogdet",
      "pd_op.matmul",
      "pd_op.bmm",
      "pd_op.einsum",
      "pd_op.conv2d",
      "pd_op.conv3d",
      "pd_op.conv2d_transpose",
      "pd_op.conv3d_transpose",
      "pd_op.fft_c2c",
      "pd_op.fft_c2r",
      "pd_op.fft_r2c",
      "pd_op.sort",
      "pd_op.argsort",
      "pd_op.topk",
      "pd_op.unique",
      "pd_op.unique_consecutive",
      "pd_op.nonzero",
      "pd_op.masked_select",
      "pd_op.nms",
      "pd_op.multiclass_nms3",
      "pd_op.generate_proposals",
      "pd_op.box_coder",
      "pd_op.searchsorted",
      "pd_op.histogram",
      "pd_op.bincount",
      "pd_op.kthvalue",
      "pd_op.mode",
      "pd_op.median",
      "pd_op.nanmedian",
      "pd_op.quantile",
      "pd_op.nanquantile",
      "pd_op.edit_distance",
      "pd_op.ctc_align",
      "pd_op.warprnnt",
  };
  return kHeavyOps;
}

// Tier 2: Intrinsic O(1) metadata ops that only read/write tensor shape or
// small shape-vector metadata without touching bulk device buffers.
const std::unordered_set<std::string>& IntrinsicMetadataOpNames() {
  static const std::unordered_set<std::string> kIntrinsicOps = {
      paddle::dialect::FullIntArrayOp::name(),
      paddle::dialect::ShapeOp::name(),
      paddle::dialect::Shape64Op::name(),
      paddle::dialect::NumelOp::name(),
  };
  return kIntrinsicOps;
}

// Computes exact static element count of a single AllocatedDenseTensorType.
// Returns false if the type is not an AllocatedDenseTensorType, contains any
// dynamic dimension (< 0), or exceeds `limit` (checked overflow-safely).
bool TryComputeStaticTensorNumel(pir::Type type,
                                 int64_t limit,
                                 int64_t* out_numel) {
  auto tensor_type = type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
  if (!tensor_type) {
    return false;
  }
  const auto& dims = tensor_type.dims();
  int64_t numel = 1;
  bool has_zero_dim = false;
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      return false;
    }
    if (dims[i] == 0) {
      has_zero_dim = true;
      continue;
    }
    if (!has_zero_dim && numel > limit / dims[i]) {
      return false;
    }
    if (!has_zero_dim) {
      numel *= dims[i];
    }
  }
  *out_numel = has_zero_dim ? 0 : numel;
  return true;
}

// Evaluates Tier 3 generic static I/O cost for a standard CPU PhiKernelOp:
// every non-null operand and result must be an AllocatedDenseTensorType with
// statically known dims and all inputs residing on CPU, and the total element
// count across inputs + outputs must be <= kInlineMetadataMaxOpNumel.
bool TryEstimateSmallStaticCpuOpNumel(const pir::Operation* op,
                                      int64_t* total_numel) {
  if (op == nullptr || op->num_results() == 0) {
    return false;
  }
  int64_t sum = 0;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    pir::Value v = op->operand_source(i);
    if (!v || !v.type()) {
      continue;
    }
    auto tensor_type =
        v.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    if (!tensor_type || !phi::is_cpu_place(tensor_type.place())) {
      return false;
    }
    int64_t n = 0;
    if (!TryComputeStaticTensorNumel(
            v.type(), kInlineMetadataMaxOpNumel - sum, &n)) {
      return false;
    }
    sum += n;
  }
  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Type res_type = op->result(i).type();
    if (!res_type) {
      return false;
    }
    int64_t n = 0;
    if (!TryComputeStaticTensorNumel(
            res_type, kInlineMetadataMaxOpNumel - sum, &n)) {
      return false;
    }
    sum += n;
  }
  *total_numel = sum;
  return true;
}

std::string ExtractLogicalOpName(const pir::Operation* op) {
  const auto& attrs = op->attributes();
  auto it = attrs.find("op_name");
  if (it != attrs.end()) {
    if (auto str_attr = it->second.dyn_cast<pir::StrAttribute>()) {
      return str_attr.AsString();
    }
  }
  return op->name();
}

OpScheduleInfo ClassifyOpScheduleInfo(pir::Operation* op,
                                      OpFuncType func_type,
                                      const std::string& fallback_name) {
  OpScheduleInfo info;
  info.func_type = func_type;

  // Non-sync device ops never block the host launch thread.
  if (func_type == OpFuncType::kGpuAsync) {
    info.host_cost = HostSyncCost::kNone;
    info.estimated_host_numel = 0;
    return info;
  }

  // Without a backing Operation we cannot verify static shapes/dialect;
  // fall back to Tier 2 intrinsic check on fallback_name, otherwise kCompute.
  if (op == nullptr) {
    if (IntrinsicMetadataOpNames().count(fallback_name) > 0) {
      info.host_cost = HostSyncCost::kInlineMetadata;
      info.estimated_host_numel = 1;
    } else if (BlockingDeviceIOOpNames().count(fallback_name) > 0) {
      info.host_cost = HostSyncCost::kBlockingDeviceIO;
    } else {
      info.host_cost = HostSyncCost::kCompute;
    }
    return info;
  }

  const std::string dialect_name = op->dialect()->name();
  const std::string op_name = ExtractLogicalOpName(op);

  // Tier 2 (builtin.combine): pointer-packing of CPU metadata tensors only.
  if (auto combine_op = op->dyn_cast<pir::CombineOp>()) {
    info.host_cost = HostSyncCost::kInlineMetadata;
    info.estimated_host_numel =
        static_cast<int64_t>(std::max<size_t>(1, combine_op.num_operands()));
    return info;
  }

  // Tier 1 (Dialect blocklist): non-pd_kernel dialects (control flow, py_func,
  // custom_kernel, onednn_kernel, etc.) are conservatively treated as
  // blocking or compute.
  if (dialect_name == "py_func" || dialect_name == "pd_op" ||
      dialect_name == "cf") {
    info.host_cost = HostSyncCost::kBlockingDeviceIO;
    return info;
  }
  if (dialect_name != paddle::dialect::KernelDialect::name()) {
    info.host_cost = HostSyncCost::kCompute;
    return info;
  }

  // Tier 1 (Op blocklist): explicit blocking/sync/I/O or heavy CPU ops.
  if (BlockingDeviceIOOpNames().count(op_name) > 0) {
    info.host_cost = HostSyncCost::kBlockingDeviceIO;
    return info;
  }
  if (HeavyOrSuperlinearCpuOpNames().count(op_name) > 0) {
    info.host_cost = HostSyncCost::kCompute;
    return info;
  }

  // Tier 2 (Intrinsic O(1) metadata ops).
  if (IntrinsicMetadataOpNames().count(op_name) > 0) {
    info.host_cost = HostSyncCost::kInlineMetadata;
    info.estimated_host_numel = 1;
    return info;
  }

  // Tier 3 (Generic static I/O cost model): standard PhiKernelOp running on
  // CPU with statically small dense-tensor inputs and outputs.
  if (func_type == OpFuncType::kCpuSync &&
      op->isa<paddle::dialect::PhiKernelOp>()) {
    int64_t total_numel = 0;
    if (TryEstimateSmallStaticCpuOpNumel(op, &total_numel)) {
      info.host_cost = HostSyncCost::kInlineMetadata;
      info.estimated_host_numel = std::max<int64_t>(1, total_numel);
      return info;
    }
  }

  info.host_cost = (func_type == OpFuncType::kGpuSync)
                       ? HostSyncCost::kBlockingDeviceIO
                       : HostSyncCost::kCompute;
  return info;
}

}  // namespace

OpFuncType AnalyzeOpFuncType(pir::Operation* op, const Place& place) {
  if (phi::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(
      IsSupportedHeterPlace(place),
      true,
      common::errors::Fatal("Unsupported current place %s", place));

  auto& op_attributes = op->attributes();

  if ((op->dialect()->name() == paddle::dialect::KernelDialect::name()) &&
      (op_attributes.count("kernel_key") > 0)) {
    auto kernel_key = op_attributes.at("kernel_key")
                          .dyn_cast<dialect::KernelAttribute>()
                          .data();
    if (phi::TransToPhiPlace(kernel_key.backend()).GetType() ==
        phi::AllocationType::CPU) {
      return OpFuncType::kCpuSync;
    }
  }

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  if ((op->dialect()->name() == "pd_kernel") &&
      (op_attributes.count("op_name") > 0)) {
    auto op_name =
        op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
    if (op_name == "pd_op.coalesce_tensor" &&
        (!phi::is_xpu_place(place) ||
         op->attribute<pir::BoolAttribute>("persist_output").data() == false) &&
        op->attribute<pir::BoolAttribute>("set_constant").data() == false &&
        op->attribute<pir::BoolAttribute>("copy_data").data() == false) {
      return OpFuncType::kGpuSync;
    }

    if (phi::is_gpu_place(place) && (op_name == "pd_op.memcpy_d2h" ||
                                     op_name == "pd_op.memcpy_d2h_multi_io")) {
      return OpFuncType::kGpuSync;
    }

    if (op_name.compare(paddle::dialect::ShapeOp::name()) == 0 ||
        op_name.compare(paddle::dialect::Shape64Op::name()) == 0) {
      return OpFuncType::kGpuSync;
    }
  }

  if (auto combine_op = op->dyn_cast<pir::CombineOp>()) {
    for (size_t i = 0; i < combine_op.num_operands(); ++i) {
      if (auto combine_operand_type =
              combine_op.operand_source(i)
                  .type()
                  .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()) {
        if (phi::is_cpu_place(combine_operand_type.place())) {
          return OpFuncType::kCpuSync;
        }
      }
    }
  }

  return OpFuncType::kGpuAsync;
}

OpScheduleInfo AnalyzeOpScheduleInfo(pir::Operation* op, const Place& place) {
  OpFuncType func_type =
      op == nullptr ? (phi::is_cpu_place(place) ? OpFuncType::kCpuSync
                                                : OpFuncType::kGpuAsync)
                    : AnalyzeOpFuncType(op, place);
  return ClassifyOpScheduleInfo(op, func_type, "");
}

OpScheduleInfo AnalyzeOpScheduleInfo(const InstructionBase* ins,
                                     const Place& place) {
  if (ins == nullptr) {
    return OpScheduleInfo{};
  }
  pir::Operation* op = ins->Operation();
  OpFuncType func_type = ins->KernelType();
  return ClassifyOpScheduleInfo(op, func_type, ins->Name());
}

int64_t CountTraceBlockingSyncOps(
    const std::vector<std::unique_ptr<InstructionBase>>& instructions) {
  int64_t trace_blocking_sync_op_num = 0;
  int64_t cumulative_inline_numel = 0;
  for (const auto& ins : instructions) {
    const OpScheduleInfo& info = ins->ScheduleInfo();
    if (!info.IsSync()) {
      continue;
    }
    if (!info.CanInlineInTraceRun()) {
      ++trace_blocking_sync_op_num;
      continue;
    }
    cumulative_inline_numel += info.estimated_host_numel;
    if (cumulative_inline_numel > kGraphInlineMetadataNumelBudget) {
      ++trace_blocking_sync_op_num;
    }
  }
  return trace_blocking_sync_op_num;
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
