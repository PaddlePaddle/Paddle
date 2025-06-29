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

#include <string>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/pe/schedule_param.pb.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/utils/flat_hash_map.h"

namespace cinn {
namespace hlir {
namespace pe {
class ScheduleParam {
 public:
  ~ScheduleParam();
  ScheduleParam(const ScheduleParam &) = delete;
  ScheduleParam &operator=(const ScheduleParam &) = delete;
  static ScheduleParam &get_cuda_instance() {
    static ScheduleParam instance{cinn::common::NVGPUArch{}};
    return instance;
  }
  static ScheduleParam &get_hip_instance() {
    static ScheduleParam instance{cinn::common::HygonDCUArchHIP{}};
    return instance;
  }
  static ScheduleParam &get_sycl_instance() {
    static ScheduleParam instance{cinn::common::HygonDCUArchSYCL{}};
    return instance;
  }
  static ScheduleParam &get_x86_instance() {
    static ScheduleParam instance{cinn::common::X86Arch{}};
    return instance;
  }
  paddle::flat_hash_map<std::string,
                        paddle::flat_hash_map<std::string, std::vector<int>>>
      &GetParam() {
    return param_data;
  }
  paddle::flat_hash_map<std::string, std::vector<int>> &operator[](
      const std::string &key) {
    return param_data[key];
  }
  int Count(const std::string &key) { return param_data.count(key); }

 private:
  explicit ScheduleParam(cinn::common::Arch arch);
  paddle::flat_hash_map<std::string,
                        paddle::flat_hash_map<std::string, std::vector<int>>>
      param_data;
};

int GetInnerSplitter(int origin, int other_axis);

int GetVectorizeFactor(int shape, int split_factor);

int SplitEven(int origin);

int GetBasicFactor(const Type &type, const cinn::common::Target &target);

int GetBetterSplitFactor(int shape, int split_factor);

int GetArrayPackingFactor(int shape,
                          const Type &type,
                          const cinn::common::Target &target);

void GetConv2dFactors(paddle::flat_hash_map<std::string, int> *factors,
                      int oc,
                      int ic,
                      int fc,
                      int oh,
                      int ow,
                      const Type &type,
                      const cinn::common::Target &target,
                      const std::string &key = "",
                      bool import_params = true);

void GetConv2d1x1Factors(paddle::flat_hash_map<std::string, int> *factors,
                         int oc,
                         int ic,
                         int oh,
                         int ow,
                         const Type &type,
                         const cinn::common::Target &target);

void CudaSplitSchedule(cinn::common::CINNValuePack *arg_pack,
                       const std::vector<std::vector<int>> &output_shapes,
                       int axis,
                       const cinn::common::Target &target);

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
    paddle::flat_hash_map<std::string,
                          paddle::flat_hash_map<std::string, std::vector<int>>>
        *params,
    const std::string &file_name = "default_serial.log");

void SaveSerialData(
    const paddle::flat_hash_map<
        std::string,
        paddle::flat_hash_map<std::string, std::vector<int>>> &model_data,
    const std::string &file_name = "default_serial.log");

int GetMaxSplitter(int a, int b);

paddle::flat_hash_map<std::string,
                      paddle::flat_hash_map<std::string, std::vector<int>>>
CreateCudaParams();

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
