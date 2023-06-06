/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
// #include "paddle/utils/optional.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */
// Copy for DenseTensor
std::shared_ptr<phi::DenseTensor> CopyDenseTensor(
    const std::shared_ptr<phi::DenseTensor>& in, Place dst_place);

paddle::optional<phi::DenseTensor> CopyDenseTensor(
    const paddle::optional<phi::DenseTensor>& in, Place dst_place);

// Copy for SelectedRows
std::shared_ptr<phi::SelectedRows> CopySelectedRows(
    const std::shared_ptr<phi::SelectedRows>& in, Place dst_place);

paddle::optional<phi::SelectedRows> CopySelectedRows(
    const paddle::optional<phi::SelectedRows>& in, Place dst_place);

// std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
//     const std::unique_ptr<std::vector<phi::DenseTensor>>& ins,
//     Place dst_place);

std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
    const std::vector<const phi::DenseTensor*>& ins, Place dst_place);

// paddle::optional<std::vector<phi::DenseTensor>> Copy(
//     const paddle::optional<std::vector<phi::DenseTensor>>& ins,
//     Place dst_place);

paddle::optional<std::vector<phi::DenseTensor>> CopyOptionalVector(
    const paddle::optional<std::vector<const phi::DenseTensor*>>& ins,
    Place dst_place);

std::vector<const phi::DenseTensor*> DenseTensorToConstDenseTensorPtr(
    const std::vector<phi::DenseTensor>& tensors,
    const std::vector<const phi::DenseTensor*>& ins);

paddle::optional<std::vector<const phi::DenseTensor*>>
DenseTensorToConstDenseTensorPtr(
    const paddle::optional<std::vector<phi::DenseTensor>>& tensors);

/* ------------------ for output ----------------------- */
std::shared_ptr<phi::DenseTensor> CopyDenseTensor(const phi::DenseTensor* in,
                                                  Place dst_place);

std::shared_ptr<phi::SelectedRows> CopySelectedRows(const phi::SelectedRows* in,
                                                    Place dst_place);

std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
    const std::vector<phi::DenseTensor*>& ins, Place dst_place);

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    std::vector<phi::DenseTensor>* tensors,
    const std::vector<phi::DenseTensor*>& ins);

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    std::vector<phi::DenseTensor>* tensors,
    const std::vector<const phi::DenseTensor*>& ins);

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    const paddle::optional<std::vector<phi::DenseTensor>>& tensors);

/* ------------------ for device 2 ----------------------- */
phi::Place& GetDebugDev2Type();

/* ------------------ for parsing environment variables -----------------------
 */
int64_t OpId();
int64_t OpIdAdd();

bool ContinueOrNot(const std::string& op_name);

bool DebugOrNot();

/* ------------------ for log acc ----------------------- */
std::string XPUDebugStartString(const std::string& op_name,
                                const Backend& dev_place,
                                const DataType& kernel_data_type);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::DenseTensor& a,
                           const phi::DenseTensor& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const paddle::optional<phi::DenseTensor>& a,
                           const paddle::optional<phi::DenseTensor>& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::SelectedRows& a,
                           const phi::SelectedRows& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const paddle::optional<phi::SelectedRows>& a,
                           const paddle::optional<phi::SelectedRows>& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<const phi::DenseTensor*>& a,
                           const std::vector<const phi::DenseTensor*>& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<const phi::TensorBase*>& a,
                           const std::vector<const phi::TensorBase*>& b);

std::string XPUDebugString(
    const std::string& op_name,
    const std::string& tensor_name,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& a,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::DenseTensor* a,
                           const phi::DenseTensor* b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::SelectedRows* a,
                           const phi::SelectedRows* b);

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<phi::DenseTensor*>& a,
                           const std::vector<phi::DenseTensor*>& b);
}  // namespace experimental
}  // namespace paddle
