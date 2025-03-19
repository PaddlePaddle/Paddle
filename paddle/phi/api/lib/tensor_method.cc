/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/include/tensor.h"

#include "glog/logging.h"

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_base.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/unary.h"
// clang-format off
#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#endif
namespace paddle {
namespace experimental {
// declare cast api
Tensor cast(const Tensor &x, DataType out_dtype);
Tensor copy_to(const Tensor &x, const Place &place, bool blocking);
}  // namespace experimental

// TODO(chenweihang): Remove this namespace using-directives later
using namespace experimental;  // NOLINT

Tensor Tensor::cast(DataType target_type) const {
  return experimental::cast(*this, target_type);
}

Tensor Tensor::copy_to(const Place &place, bool blocking) const {
  return experimental::copy_to(*this, place, blocking);
}

template <typename T>
Tensor Tensor::copy_to(const Place &target_place) const {
  LOG_FIRST_N(WARNING, 1)
      << "The Tensor's `copy_to` method is deprecated since version "
         "2.3, and will be removed in version 2.4, please use "
         "`copy_to` method without template argument instead. "
         "reason: copying a Tensor to another device does not need "
         "to specify the data type template argument.";
  return copy_to(target_place, /*blocking=*/false);
}

template PADDLE_API Tensor
Tensor::copy_to<float>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<double>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int64_t>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int32_t>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<uint8_t>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int8_t>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int16_t>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<bool>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::complex<float>>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::complex<double>>(const Place &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::float16>(const Place &target_place) const;

void Tensor::copy_(const Tensor &src,
                   const phi::Place &target_place,
                   bool blocking) {
  if (!src.has_allocation()) {
    VLOG(8) << "Src is empty, skip copy";
    return;
  }

  if (src.place().GetType() == AllocationType::UNDEFINED) {
    VLOG(8) << "Src place is UNDEFINED, skip copy";
    return;
  }

  VLOG(3) << "Deep copy Tensor from " << src.name() << " to " << name();
  if (initialized()) {
    PADDLE_ENFORCE_EQ(dtype(),
                      src.dtype(),
                      common::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s, "
                          "Tensor Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      common::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(target_place,
                      place(),
                      common::errors::PreconditionNotMet(
                          "Place is different of dst tensor and args %s, which "
                          "current tensor holds %s "
                          "Copy cannot be performed!",
                          target_place,
                          place()));
  }

  // Prepare copy kernel key and outputs
  auto kernel_key_set = ParseKernelKeyByInputArgs(src);
  KernelType kernel_type = ParseKernelTypeByInputArgs(src);
  if (initialized()) {
    kernel_key_set.backend_set = kernel_key_set.backend_set |
                                 BackendSet(phi::TransToPhiBackend(place()));
  } else {
    // Deep Copy AutoGrad info from src to self.
    *autograd_meta_ = *(src.autograd_meta_);
  }
  kernel_key_set.backend_set = kernel_key_set.backend_set |
                               BackendSet(phi::TransToPhiBackend(target_place));
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  auto place = phi::TransToPhiPlace(kernel_key.backend());
  auto &pool = paddle::experimental::DeviceContextPool::Instance();
  auto *dev_ctx = pool.GetMutable(
      place.GetType() == target_place.GetType() ? target_place : place);

  if (kernel_type == KernelType::DENSE_TENSOR_KERNEL) {
#ifdef PADDLE_WITH_DISTRIBUTE
  bool run_auto_parallel = AllInputsAreDistTensor(src);
  bool rank_is_in_current_mesh = false;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(
                    src.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);

    auto meta_dist_input_x = MakeDistMetaTensor(*src.impl());

    if (this->initialized()) {
      auto this_dist_attr =
                std::static_pointer_cast<phi::distributed::DistTensor>(
                this->impl())->dist_attr();
      PADDLE_ENFORCE_EQ((meta_dist_input_x.dist_attr() == this_dist_attr
                        || this_dist_attr.empty()),
                        true,
                        common::errors::PreconditionNotMet(
                            "DistAttr is different of dst "
                            "tensor and args %s, which "
                            "current tensor holds %s "
                            "Copy cannot be performed!",
                            meta_dist_input_x.dist_attr(),
                            this_dist_attr));
    }

    auto dist_out = SetKernelDistOutput(this, meta_dist_input_x.dist_attr());
    auto dense_out = dist_out->unsafe_mutable_value();
    if (!rank_is_in_current_mesh) {
      *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr,
            0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    phi::MetaTensor meta_dist_out(dist_out);
    phi::UnchangedInferMeta(MakeMetaTensor(*(src.impl_)), &meta_dist_out);

    if (rank_is_in_current_mesh) {
      auto dist_input_x = static_cast<phi::distributed::DistTensor*>(
                          src.impl().get());;

      auto input_x = &dist_input_x->value();

      phi::MetaTensor meta_dense_out(dense_out);
      phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_dense_out);

      phi::Copy(*dev_ctx, *input_x, target_place, blocking, dense_out);
    }
    return;
  }
#endif
    SetKernelOutput(this);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::DenseTensor>(src.impl_))),
        &meta_out);
    phi::Copy(*dev_ctx,
              (*(std::static_pointer_cast<phi::DenseTensor>(src.impl_))),
              target_place,
              blocking,
              static_cast<phi::DenseTensor *>(impl_.get()));
  } else if (kernel_type == KernelType::SELECTED_ROWS_KERNEL) {
    SetSelectedRowsKernelOutput(this);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::SelectedRows>(src.impl_))),
        &meta_out);
    phi::Copy(*dev_ctx,
              (*(std::static_pointer_cast<phi::SelectedRows>(src.impl_))),
              target_place,
              blocking,
              static_cast<phi::SelectedRows *>(impl_.get()));
  } else if (kernel_type == KernelType::SPARSE_COO_KERNEL) {
    SetSparseKernelOutput(this, TensorType::SPARSE_COO);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::SparseCooTensor>(src.impl_))),
        &meta_out);
    phi::Copy(*dev_ctx,
              (*(std::static_pointer_cast<phi::SparseCooTensor>(src.impl_))),
              target_place,
              blocking,
              static_cast<phi::SparseCooTensor *>(impl_.get()));
  } else if (kernel_type == KernelType::SPARSE_CSR_KERNEL) {
    SetSparseKernelOutput(this, TensorType::SPARSE_CSR);
    phi::MetaTensor meta_out(impl_.get());
    phi::UnchangedInferMeta(
        MakeMetaTensor(
            *(std::static_pointer_cast<phi::SparseCsrTensor>(src.impl_))),
        &meta_out);
    phi::Copy(*dev_ctx,
              (*(std::static_pointer_cast<phi::SparseCsrTensor>(src.impl_))),
              target_place,
              blocking,
              static_cast<phi::SparseCsrTensor *>(impl_.get()));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "We currently only support dense tensor copy for now and if u need to "
        "copy selected rows please raise a issue."));
  }
}

Tensor Tensor::to_sparse_coo(const int64_t sparse_dim) const {
  return experimental::sparse::to_sparse_coo(*this, sparse_dim);
}

Tensor Tensor::to_sparse_csr() const {
  return experimental::sparse::to_sparse_csr(*this);
}

Tensor Tensor::to_dense() const {
  return experimental::sparse::to_dense(*this);
}

}  // namespace paddle
