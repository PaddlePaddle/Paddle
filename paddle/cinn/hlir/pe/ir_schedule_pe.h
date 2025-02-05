// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "paddle/cinn/hlir/pe/schedule_param.pb.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {

void IRElementwiseSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                           const std::vector<int> &output_shape,
                           const cinn::common::Target &target);

void IRInjectiveSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                         const std::vector<int> &output_shape,
                         const cinn::common::Target &target);

void IRScheduleInjectiveCPU(ir::IRSchedule &ir_sch,  // NOLINT
                            const std::vector<int> &output_shape,
                            const cinn::common::Target &target,
                            bool vectorizable = true);

void IRGpuScheduleInjective(ir::IRSchedule &ir_sch,  // NOLINT
                            const std::vector<int> &output_shape,
                            const cinn::common::Target &target);

std::vector<cinn::common::CINNValue> IRGpuScheduleMatMul(
    const cinn::common::CINNValuePack &arg_pack,
    const std::vector<int> &output_shape,
    const cinn::common::Target &target);

void IRCudaScheduleMul(ir::IRSchedule &ir_sch,  // NOLINT
                       const std::vector<int> &output_shape,
                       const cinn::common::Target &target);

void IRMulScheduleCPU(ir::IRSchedule &ir_sch,  // NOLINT
                      const std::vector<int> &reduce_first_shape,
                      const cinn::common::Target &target);

void IRCudaSplitSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                         const std::vector<std::vector<int>> &output_shapes,
                         int axis,
                         const cinn::common::Target &target);

void IRGpuScheduleReduce(ir::IRSchedule &ir_sch,  // NOLINT
                         ir::Tensor out,
                         int last_dimension_num,
                         const cinn::common::Target &target);

void IRGpuScheduleBlockReduce(ir::IRSchedule &ir_sch,  // NOLINT
                              ir::Tensor reduce_tmp_out,
                              ir::Tensor tmp_out,
                              ir::Tensor out,
                              const cinn::common::Target &target);

void IRGpuScheduleBlockReduceInternal(ir::IRSchedule &ir_sch,  // NOLINT
                                      ir::Tensor tmp_out,
                                      ir::Tensor out,
                                      const cinn::common::Target &target);

void IRGpuScheduleBlockShuffleReduce(ir::IRSchedule &ir_sch,  // NOLINT
                                     ir::Tensor reshape,
                                     ir::Tensor internal,
                                     ir::Tensor out,
                                     const cinn::common::Target &target);

void IRGpuTwoStepReduceSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                                ir::Tensor reshape,
                                ir::Tensor internal,
                                ir::Tensor tmp_out,
                                ir::Tensor out,
                                const cinn::common::Target &target);

void IRSoftmaxScheduleCPU(ir::IRSchedule &ir_sch, int axis = -1);  // NOLINT

void IRPoolScheduleGPU(ir::IRSchedule &ir_sch,  // NOLINT
                       const cinn::common::Target &target,
                       int arg_pack_size = 3);

void IRCudaScheduleDepthwiseConv(ir::IRSchedule &ir_sch,  // NOLINT
                                 const std::vector<ir::Expr> &tensors);

void IRGlobalPoolScheduleGPU(ir::IRSchedule &ir_sch,  // NOLINT
                             const cinn::common::Target &target);

void IRCudaScheduleConv2(ir::IRSchedule &ir_sch,  // NOLINT
                         ir::Tensor &input_pad,   // NOLINT
                         ir::Tensor &weights,     // NOLINT
                         ir::Tensor &output,      // NOLINT
                         const cinn::common::Target &target,
                         const std::string &key);

void IRCudaScheduleConv(ir::IRSchedule &ir_sch,  // NOLINT
                        const cinn::common::Target &target);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
