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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/pe/schedule_param.pb.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {
class ScheduleParam {
 public:
  ~ScheduleParam();
  ScheduleParam(const ScheduleParam &) = delete;
  ScheduleParam &operator=(const ScheduleParam &) = delete;
  static ScheduleParam &get_cuda_instance() {
    static ScheduleParam instance{common::Target::Arch::NVGPU};
    return instance;
  }
  static ScheduleParam &get_x86_instance() {
    static ScheduleParam instance{common::Target::Arch::X86};
    return instance;
  }
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::vector<int>>>
      &GetParam() {
    return param_data;
  }
  absl::flat_hash_map<std::string, std::vector<int>> &operator[](
      const std::string &key) {
    return param_data[key];
  }
  int Count(const std::string &key) { return param_data.count(key); }

 private:
  explicit ScheduleParam(common::Target::Arch arch);
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::vector<int>>>
      param_data;
};

int GetInnerSplitter(int origin, int other_axis);

int GetVectorizeFactor(int shape, int split_factor);

int SplitEven(int origin);

int GetBasicFactor(const Type &type, const common::Target &target);

int GetBetterSplitFactor(int shape, int split_factor);

int GetArrayPackingFactor(int shape,
                          const Type &type,
                          const common::Target &target);

void ScheduleInjectiveCPU(poly::Stage *stage,
                          const std::vector<int> &output_shape,
                          const common::Target &target,
                          bool vectorizable = true);
// to deprecate
void ScheduleInjectiveCPU1(poly::Stage *stage,
                           const std::vector<int> &output_shape,
                           const common::Target &target,
                           bool vectorizable = true);

void MatmulScheduleCUDA(poly::StageMap stages,
                        const ir::Tensor &output,
                        const common::Target &target);

void MatmulScheduleCPU(poly::StageMap stage,
                       const ir::Tensor &output,
                       const ir::Tensor &packedB,
                       const common::Target &target);

void MulScheduleCPU(poly::StageMap stage,
                    const ir::Tensor &output,
                    const ir::Tensor &input_tensor,
                    const common::Target &target);

void SoftmaxScheduleCPU(poly::StageMap stage,
                        const ir::Tensor &output,
                        const ir::Tensor &temp,
                        int axis = -1);

void GetConv2dFactors(absl::flat_hash_map<std::string, int> *factors,
                      int oc,
                      int ic,
                      int fc,
                      int oh,
                      int ow,
                      const Type &type,
                      const common::Target &target,
                      const std::string &key = "",
                      bool import_params = true);

void GetConv2d1x1Factors(absl::flat_hash_map<std::string, int> *factors,
                         int oc,
                         int ic,
                         int oh,
                         int ow,
                         const Type &type,
                         const common::Target &target);

void Conv2d_NCHWc_Schedule_CPU(poly::StageMap stages,
                               const ir::Tensor &res,
                               ir::Tensor &packed_out,  // NOLINT
                               const ir::Tensor &input_pad,
                               const ir::Tensor &weights_dilation,
                               const ir::Tensor &data,
                               const common::Target &target,
                               const std::string &key,
                               bool do_padding);
void GlobalPoolScheduleGPU(poly::StageMap stages,
                           const std::vector<ir::Tensor> &output,
                           const common::Target &target);
void PoolScheduleCPU(poly::StageMap stages,
                     const ir::Tensor &output,
                     const common::Target &target);
void PoolScheduleGPU(poly::StageMap stages,
                     const ir::Tensor &output,
                     const common::Target &target);

void Conv2d_NCHWc_Schedule_CPU_Nofuse(poly::StageMap stages,
                                      const ir::Tensor &res,
                                      ir::Tensor &packed_out,  // NOLINT
                                      const ir::Tensor &input_pad,
                                      const ir::Tensor &weights_dilation,
                                      const ir::Tensor &data,
                                      const common::Target &target);

void Conv2d_NCHWc_1X1_Schedule_CPU(poly::StageMap stages,
                                   const ir::Tensor &res,
                                   ir::Tensor &packed_out,  // NOLINT
                                   const ir::Tensor &input_pad,
                                   const ir::Tensor &weights_dilation,
                                   const ir::Tensor &data,
                                   const common::Target &target,
                                   const std::string &key,
                                   bool do_padding);

void Conv2d_NCHWc_1X1_Schedule_CPU_Nofuse(poly::StageMap stages,
                                          const ir::Tensor &res,
                                          ir::Tensor &packed_out,  // NOLINT
                                          const ir::Tensor &input_pad,
                                          const ir::Tensor &weights_dilation,
                                          const ir::Tensor &data,
                                          const common::Target &target);

void Depthwise_Conv2d_NCHWc_Schedule_CPU_Nofuse(
    poly::StageMap stages,
    const ir::Tensor &res,
    ir::Tensor &packed_out,  // NOLINT
    const ir::Tensor &input_pad,
    const ir::Tensor &weights_dilation,
    const ir::Tensor &data,
    const common::Target &target,
    bool do_padding);

void CudaScheduleMul(poly::StageMap stages,
                     ir::Tensor output,
                     const std::vector<int> &output_shape,
                     const common::Target &target);

// reduce shedules.
void CudaReduceSchedule(poly::StageMap stages,
                        ir::Tensor output,
                        int last_dimension_num,
                        const common::Target &target);

void CudaWarpReduceSchedule(poly::StageMap stages,
                            ir::Tensor tmp_out,
                            ir::Tensor out,
                            const common::Target &target);

void CudaBlockReduceInternalSchedule(poly::StageMap stages,
                                     ir::Tensor tmp_out,
                                     ir::Tensor out,
                                     const common::Target &target);

void CudaBlockReduceSchedule(poly::StageMap stages,
                             ir::Tensor reduce_tmp_out,
                             ir::Tensor tmp_out,
                             ir::Tensor out,
                             const common::Target &target);

void CudaBlockShuffleReduceSchedule(poly::StageMap stages,
                                    ir::Tensor reduce_reshape,
                                    ir::Tensor reduce_internal,
                                    ir::Tensor reduce_out,
                                    const common::Target &target);

void CudaTwoStepReduceSchedule(poly::StageMap stages,
                               ir::Tensor reshape,
                               ir::Tensor internal,
                               ir::Tensor tmp_out,
                               ir::Tensor out,
                               const common::Target &target);

void CudaScheduleDepthwiseConv(poly::StageMap stages,
                               ir::Tensor &output,  // NOLINT
                               const common::Target &target);

void CudaScheduleConv(poly::StageMap stages,
                      ir::Tensor &input_pad,  // NOLINT
                      ir::Tensor &weights,    // NOLINT
                      ir::Tensor &output,     // NOLINT
                      const common::Target &target);

void CudaScheduleWinogradConv(poly::StageMap wino_stages,
                              std::vector<ir::Tensor> &all_tensors,  // NOLINT
                              const common::Target &target);

void CudaScheduleConv2(poly::StageMap stages,
                       ir::Tensor &input_pad,  // NOLINT
                       ir::Tensor &weights,    // NOLINT
                       ir::Tensor &output,     // NOLINT
                       const common::Target &target,
                       const std::string &key);

void CudaScheduleInjective(poly::Stage *stage,
                           const std::vector<int> &output_shape,
                           const common::Target &target);

void CudaSplitSchedule(common::CINNValuePack *arg_pack,
                       const std::vector<std::vector<int>> &output_shapes,
                       int axis,
                       const common::Target &target);

void CreateCudaSerialData(const std::string &file_name = "default_serial.log");

std::string GenerateX86ConvKey(const std::vector<Expr> &input_shape,
                               const std::vector<Expr> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations,
                               const int &index = 0,
                               const std::string &model_name = "");

std::string GenerateX86ConvKey(const std::vector<int> &input_shape,
                               const std::vector<int> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations,
                               const int &index = 0,
                               const std::string &model_name = "");
void CreateX86SerialData(const std::string &file_name = "default_serial.log");

void LoadSerialData(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *params,
    const std::string &file_name = "default_serial.log");

void SaveSerialData(
    const absl::flat_hash_map<
        std::string,
        absl::flat_hash_map<std::string, std::vector<int>>> &model_data,
    const std::string &file_name = "default_serial.log");

int GetMaxSplitter(int a, int b);

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<std::string, std::vector<int>>>
CreateCudaParams();

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
