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
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
class DeviceContext;
namespace distributed {
class DistTensor;
class TensorDistAttr;
}  // namespace distributed
}  // namespace phi

namespace paddle {
namespace experimental {

class TransformFlag {
 public:
  TransformFlag(bool stop_transform = false,
                bool trans_dtype = false,
                bool trans_backend = true,
                bool trans_layout = true)
      : stop_transform_(stop_transform),
        trans_data_type_(trans_dtype),
        trans_backend_(trans_backend),
        trans_layout_(trans_layout) {}

  bool NeedTransform() const {
    return !stop_transform_ &&
           (trans_data_type_ || trans_backend_ || trans_layout_);
  }

  bool need_trans_data_type() const {
    return !stop_transform_ && trans_data_type_;
  }

  bool need_trans_backend() const { return !stop_transform_ && trans_backend_; }

  bool need_trans_layout() const { return !stop_transform_ && trans_layout_; }

 private:
  // This is the highest priority in flags,
  // and can be setted by api[data_transform->skip_transform] in the yaml file.
  bool stop_transform_ = false;

  // trans_data_type_ can be setted by api[data_transform->support_trans_dtype]
  // in the yaml file.
  // trans_data_type_ only affect the non complex types,
  // the complex is always transferd, except stop_transform_ is true.
  bool trans_data_type_ = false;

  // trans_backend_ and trans_layout_ are true defaultly,
  // and they can only be setted by global flag.
  bool trans_backend_ = true;
  bool trans_layout_ = true;
};

static inline phi::TensorArgDef GetKernelInputArgDef(
    const phi::TensorArgDef& input_def, phi::Backend kernel_backend) {
  phi::TensorArgDef input_actual_def = input_def;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  // When the backend of input tensor arg_def is CUSTOM, we need to set it to
  // the actual backend by expected_kernel_key.
  if (input_actual_def.backend == phi::Backend::CUSTOM) {
    input_actual_def.SetBackend(kernel_backend);
  }
#endif
  return input_actual_def;
}

std::shared_ptr<phi::DenseTensor> PrepareData(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<phi::DenseTensor> PrepareData(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::unique_ptr<std::vector<phi::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

paddle::optional<std::vector<phi::DenseTensor>> PrepareData(
    const paddle::optional<std::vector<Tensor>>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

// Only support transfering place for SelectedRows
std::shared_ptr<phi::SelectedRows> PrepareDataForSelectedRows(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

paddle::optional<phi::SelectedRows> PrepareDataForSelectedRows(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag);

// Only support transfering contiguous for SparseCooTensor
std::shared_ptr<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const Tensor& input);

paddle::optional<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const paddle::optional<Tensor>& input);

// Only support transfering contiguous for SparseCsrTensor
std::shared_ptr<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const Tensor& input);

paddle::optional<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const paddle::optional<Tensor>& input);

// Only support transfering contiguous
std::shared_ptr<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const Tensor& input);

paddle::optional<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const paddle::optional<Tensor>& input);

void TransDataBackend(const phi::DenseTensor* tensor,
                      Backend target_backend,
                      phi::DenseTensor* out);

void TransDataBackend(const std::vector<phi::DenseTensor*>& tensor,
                      Backend target_backend,
                      std::vector<phi::DenseTensor*> out);

void TransDataBackend(const phi::SelectedRows* tensor,
                      Backend target_backend,
                      phi::SelectedRows* out);

phi::DenseTensor Trans2Contiguous(const phi::DenseTensor& tensor);

void CheckAndTrans2Contiguous(phi::DenseTensor* tensor);
inline bool NeedTransformPlace(const phi::Place& src_place,
                               const Backend& target,
                               const TransformFlag& transform_flag) {
  // NOTE(dev): The default value of TransformFlag is True, if it is set with
  // False
  // somewhere such as ops.yaml or backward.yaml that means we should skip data
  // transform. Because "stop_transform_" has highest priority.
  if (!transform_flag.need_trans_backend()) {
    return false;
  }
  bool ret = src_place.GetType() == AllocationType::GPUPINNED ||
             (target != Backend::ALL_BACKEND &&
              phi::TransToPhiBackend(src_place) !=
                  (target != Backend::GPUDNN ? target : Backend::GPU));
  return ret;
}

/* ------------------ for auto parallel ----------------------- */

std::shared_ptr<phi::distributed::DistTensor> ReshardDistTensor(
    phi::DeviceContext* dev_ctx,
    const Tensor& tensor,
    const phi::distributed::TensorDistAttr& dist_attr);

std::shared_ptr<phi::distributed::DistTensor> PrepareDataForDistTensor(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::shared_ptr<phi::distributed::DistTensor> PrepareDataForDistTensor(
    const std::shared_ptr<phi::distributed::DistTensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel);

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
PrepareDataForDistTensor(const std::vector<Tensor>& input,
                         const phi::TensorArgDef& target_args_def,
                         const TransformFlag& transform_flag,
                         bool is_stride_kernel);

}  // namespace experimental
}  // namespace paddle
