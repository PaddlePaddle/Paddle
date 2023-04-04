/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/xpu/xpu_op_list.h"
#include "paddle/phi/common/backend.h"
#include "paddle/utils/optional.h"

namespace phi {
class DeviceContext;
}

namespace paddle {
template <typename T>
class optional;
}  // namespace paddle

namespace paddle {
namespace experimental {

// phi::Place& xpu_debug_run_dev2();
int64_t OpId();
int64_t OpIdAdd();
bool ContinueOrNot(const std::string& op_name);
bool ContinueRunDev2OrNot(const std::string& op_name);
bool DebugOrNot();
phi::Place& xpu_debug_run_dev2();
std::string GetDebugStartStr();
void SetDebugStartStr(const std::string& str);
std::string XPUDebugStartString(const std::string& op_name,
                                const Backend& dev_place,
                                const DataType& kernel_data_type);
std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const Tensor& a,
                           const Tensor& b);
std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const optional<Tensor>& a,
                           const optional<Tensor>& b);
std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<Tensor>& a,
                           const std::vector<Tensor>& b);
std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<Tensor*>& a,
                           const std::vector<Tensor*>& b);
std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const optional<std::vector<Tensor>>& a,
                           const optional<std::vector<Tensor>>& b);
void XPUPaddleOpTimeTik();
void XPUPaddleOpTimeTok(const std::string& op_name,
                        const phi::DeviceContext* dev_ctx,
                        const Backend& dev_place,
                        const DataType& kernel_data_type);

void copy(const Tensor& src, const Place& place, bool blocking, Tensor* dst);
void copy(const std::vector<Tensor>& src,
          const Place& place,
          bool blocking,
          std::vector<Tensor>* dst);
// void copy(const std::vector<Tensor*>& src,
//           const Place& place,
//           bool blocking,
//           std::vector<Tensor*> dst);
void copy(const optional<Tensor>& src,
          const Place& place,
          bool blocking,
          optional<Tensor>* dst);
void copy(const optional<std::vector<Tensor>>& src,
          const Place& place,
          bool blocking,
          optional<std::vector<Tensor>>* dst);
std::shared_ptr<Tensor> copy(const Tensor& src,
                             const Place& place,
                             bool blocking);
// std::shared_ptr<Tensor> copy(const Tensor* src, const Place& place, bool
// blocking);
std::shared_ptr<std::vector<Tensor>> copy(const std::vector<Tensor>& src,
                                          const Place& place,
                                          bool blocking);
// std::shared_ptr<std::vector<Tensor*>> copy(const std::vector<Tensor*> src,
// const Place& place, bool blocking);
std::shared_ptr<paddle::optional<Tensor>> copy(
    const paddle::optional<Tensor>& src, const Place& place, bool blocking);
std::shared_ptr<paddle::optional<std::vector<Tensor>>> copy(
    const optional<std::vector<Tensor>>& src,
    const Place& place,
    bool blocking);

double check_mse(const Tensor& a, const Tensor& b);
double check_mse(const paddle::optional<Tensor>& a,
                 const paddle::optional<Tensor>& b);
double check_mse(const std::vector<Tensor>& a,
                 const paddle::optional<Tensor>& b);
double check_mse(const optional<std::vector<Tensor>>& a,
                 const optional<std::vector<Tensor>>& b);
}  // namespace experimental
}  // namespace paddle
