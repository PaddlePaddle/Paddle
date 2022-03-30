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

#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/api/lib/ext_compat_utils.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
/**
 * [ Why still include the fluid headers? ]
 *
 * We hope to organize the basic implementation of Tensor and the logic related
 * to Tensor computation into an independent library, which we call
 * [Tensor Operation Library, phi], so we extract or rewrite the original
 * Kernels.
 *
 * In the future, the training library, inference library and custom operators
 * will link to this Tensor Operation library.
 *
 * However, if we directly split the link relation, we need to make too many
 * changes, which will affect the stability of the framework, so here we still
 * rely on the implementation of the framework, which is a intermediate state.
 *
 * In the future, the necessary components will be moved to the this library,
 * or the corresponding components will be re-implemented.
 */

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

/////// Tensor Methods ////////

/* Part 1: Construction and destruction methods */

Tensor::Tensor(std::shared_ptr<phi::TensorBase> tensor_impl)
    : impl_(std::move(tensor_impl)) {
  PADDLE_ENFORCE_NOT_NULL(
      impl_,
      phi::errors::InvalidArgument("TensorImpl with nullptr is not supported"));
}

Tensor::Tensor(const PlaceType &place)
    : impl_(std::move(std::make_shared<phi::DenseTensor>(
          std::move(phi::make_intrusive<SharedStorage>(
              ConvertExtPlaceToInnerPlace(place))),
          std::move(phi::DenseTensorMeta(phi::DataType::UNDEFINED,
                                         phi::make_ddim({}),
                                         phi::DataLayout::NCHW))))),
      place_{place} {}

Tensor::Tensor(const PlaceType &place, const std::vector<int64_t> &shape)
    : impl_(std::move(std::make_shared<phi::DenseTensor>(
          std::move(phi::make_intrusive<SharedStorage>(
              ConvertExtPlaceToInnerPlace(place))),
          std::move(phi::DenseTensorMeta(phi::DataType::UNDEFINED,
                                         phi::make_ddim(shape),
                                         phi::DataLayout::NCHW))))),
      place_{place} {}

Tensor::Tensor(std::shared_ptr<phi::TensorBase> tensor_impl,
               const std::string &name)
    : impl_(std::move(tensor_impl)), name_(std::move(name)) {}
/* Part 2: Dimension, DataType and DataLayout methods */

int64_t Tensor::numel() const { return impl_->numel(); }

int64_t Tensor::size() const { return impl_->numel(); }

phi::DDim Tensor::dims() const { return impl_->dims(); }

std::vector<int64_t> Tensor::shape() const {
  return phi::vectorize<int64_t>(impl_->dims());
}

void Tensor::reshape(const std::vector<int64_t> &shape) {
  LOG(WARNING) << "The function of resetting the shape of the uninitialized "
                  "Tensor of the `reshape` method is deprecated since version "
                  "2.3, and will be removed in version 2.4, please use "
                  "`paddle::experimental::full` method to create a new Tensor "
                  "instead. "
                  "reason: `reshape` means changing the tensor shape without "
                  "touching underlying data, this requires the total size of "
                  "the tensor to remain constant.";
  if (is_dense_tensor()) {
    std::dynamic_pointer_cast<phi::DenseTensor>(impl_)->Resize(
        phi::make_ddim(shape));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support reshape operation on DenseTensor now."));
  }
}

DataType Tensor::dtype() const { return impl_->dtype(); }

DataType Tensor::type() const { return impl_->dtype(); }

DataLayout Tensor::layout() const { return impl_->layout(); }

bool Tensor::is_dense_tensor() const {
  return phi::DenseTensor::classof(impl_.get());
}
bool Tensor::is_selected_rows() const {
  return phi::SelectedRows::classof(impl_.get());
}
bool Tensor::is_sparse_coo_tensor() const {
  return phi::SparseCooTensor::classof(impl_.get());
}
bool Tensor::is_sparse_csr_tensor() const {
  return phi::SparseCsrTensor::classof(impl_.get());
}
/* Part 3: Device and Backend methods */

PlaceType Tensor::place() const {
  if (!impl_->initialized()) {
    return place_;
  } else {
    return ConvertInnerPlaceToExtPlace(impl_->place());
  }
}

paddle::platform::Place Tensor::inner_place() const {
  PADDLE_ENFORCE_NOT_NULL(
      impl_,
      phi::errors::PermissionDenied(
          "Null pointer error, the impl_ of Tensor should not be "
          "Null when calling Tensor::inner_place()."));
  return impl_->place();
}

bool Tensor::is_cpu() const {
  return paddle::platform::is_cpu_place(inner_place());
}

bool Tensor::is_gpu() const {
  return paddle::platform::is_gpu_place(inner_place());
}

bool Tensor::is_gpu_pinned() const {
  return paddle::platform::is_cuda_pinned_place(inner_place());
}

/* Part 4: Data Access methods */

template <typename T>
T *Tensor::mutable_data() {
  if (is_dense_tensor()) {
    return std::dynamic_pointer_cast<phi::DenseTensor>(impl_)->mutable_data<T>(
        ConvertExtPlaceToInnerPlace(place()));
  }
  return nullptr;
}

template PADDLE_API float *Tensor::mutable_data<float>();
template PADDLE_API double *Tensor::mutable_data<double>();
template PADDLE_API int64_t *Tensor::mutable_data<int64_t>();
template PADDLE_API int32_t *Tensor::mutable_data<int32_t>();
template PADDLE_API uint8_t *Tensor::mutable_data<uint8_t>();
template PADDLE_API int8_t *Tensor::mutable_data<int8_t>();
template PADDLE_API int16_t *Tensor::mutable_data<int16_t>();
template PADDLE_API bool *Tensor::mutable_data<bool>();
template PADDLE_API phi::dtype::complex<float>
    *Tensor::mutable_data<phi::dtype::complex<float>>();
template PADDLE_API phi::dtype::complex<double>
    *Tensor::mutable_data<phi::dtype::complex<double>>();
template PADDLE_API phi::dtype::float16 *
Tensor::mutable_data<phi::dtype::float16>();

template <typename T>
T *Tensor::mutable_data(const PlaceType &place) {
  auto inner_place = ConvertExtPlaceToInnerPlace(place);
  if (impl_->initialized()) {
    PADDLE_ENFORCE_EQ(
        platform::is_same_place(inner_place, impl_->place()),
        true,
        phi::errors::Unimplemented("Modification of tensor place through "
                                   "mutable_data is not supported now"));
  }
  if (is_dense_tensor()) {
    return std::dynamic_pointer_cast<phi::DenseTensor>(impl_)->mutable_data<T>(
        inner_place);
  }
  return nullptr;
}

template PADDLE_API float *Tensor::mutable_data<float>(const PlaceType &place);
template PADDLE_API double *Tensor::mutable_data<double>(
    const PlaceType &place);
template PADDLE_API int64_t *Tensor::mutable_data<int64_t>(
    const PlaceType &place);
template PADDLE_API int32_t *Tensor::mutable_data<int32_t>(
    const PlaceType &place);
template PADDLE_API uint8_t *Tensor::mutable_data<uint8_t>(
    const PlaceType &place);
template PADDLE_API int8_t *Tensor::mutable_data<int8_t>(
    const PlaceType &place);
template PADDLE_API int16_t *Tensor::mutable_data<int16_t>(
    const PlaceType &place);
template PADDLE_API bool *Tensor::mutable_data<bool>(const PlaceType &place);
template PADDLE_API phi::dtype::complex<float>
    *Tensor::mutable_data<phi::dtype::complex<float>>(const PlaceType &place);
template PADDLE_API phi::dtype::complex<double>
    *Tensor::mutable_data<phi::dtype::complex<double>>(const PlaceType &place);
template PADDLE_API phi::dtype::float16 *
Tensor::mutable_data<phi::dtype::float16>(const PlaceType &place);

template <typename T>
const T *Tensor::data() const {
  if (is_dense_tensor()) {
    return std::dynamic_pointer_cast<phi::DenseTensor>(impl_)->data<T>();
  } else if (phi::SelectedRows::classof(impl_.get())) {
    return std::dynamic_pointer_cast<phi::SelectedRows>(impl_)
        ->value()
        .data<T>();
  }
  return nullptr;
}

template PADDLE_API const float *Tensor::data<float>() const;
template PADDLE_API const double *Tensor::data<double>() const;
template PADDLE_API const int64_t *Tensor::data<int64_t>() const;
template PADDLE_API const int32_t *Tensor::data<int32_t>() const;
template PADDLE_API const uint8_t *Tensor::data<uint8_t>() const;
template PADDLE_API const int8_t *Tensor::data<int8_t>() const;
template PADDLE_API const int16_t *Tensor::data<int16_t>() const;
template PADDLE_API const bool *Tensor::data<bool>() const;
template PADDLE_API const phi::dtype::complex<float>
    *Tensor::data<phi::dtype::complex<float>>() const;
template PADDLE_API const phi::dtype::complex<double>
    *Tensor::data<phi::dtype::complex<double>>() const;
template PADDLE_API const phi::dtype::float16 *
Tensor::data<phi::dtype::float16>() const;
template PADDLE_API const phi::dtype::bfloat16 *
Tensor::data<phi::dtype::bfloat16>() const;

template <typename T>
T *Tensor::data() {
  if (is_dense_tensor()) {
    return std::dynamic_pointer_cast<phi::DenseTensor>(impl_)->data<T>();
  } else if (phi::SelectedRows::classof(impl_.get())) {
    return std::dynamic_pointer_cast<phi::SelectedRows>(impl_)
        ->mutable_value()
        ->data<T>();
  }
  return nullptr;
}

template PADDLE_API float *Tensor::data<float>();
template PADDLE_API double *Tensor::data<double>();
template PADDLE_API int64_t *Tensor::data<int64_t>();
template PADDLE_API int32_t *Tensor::data<int32_t>();
template PADDLE_API uint8_t *Tensor::data<uint8_t>();
template PADDLE_API int8_t *Tensor::data<int8_t>();
template PADDLE_API int16_t *Tensor::data<int16_t>();
template PADDLE_API bool *Tensor::data<bool>();
template PADDLE_API phi::dtype::complex<float>
    *Tensor::data<phi::dtype::complex<float>>();
template PADDLE_API phi::dtype::complex<double>
    *Tensor::data<phi::dtype::complex<double>>();
template PADDLE_API phi::dtype::float16 *Tensor::data<phi::dtype::float16>();

// TODO(chenweihang): replace slice impl by API
Tensor Tensor::slice(int64_t begin_idx, int64_t end_idx) const {
  if (is_dense_tensor()) {
    return Tensor(std::make_shared<phi::DenseTensor>(
        std::move(phi::DenseTensorUtils::Slice(
            *(std::dynamic_pointer_cast<phi::DenseTensor>(impl_).get()),
            begin_idx,
            end_idx))));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support slice operation on DenseTensor now."));
  }
}

const std::shared_ptr<phi::TensorBase> &Tensor::impl() const { return impl_; }

void Tensor::set_impl(const std::shared_ptr<phi::TensorBase> &impl) {
  impl_ = impl;
}

void Tensor::set_impl(std::shared_ptr<phi::TensorBase> &&impl) {
  impl_ = std::move(impl);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
gpuStream_t Tensor::stream() const {
  return platform::stream::get_current_stream(-1)->raw_stream();
}
#endif

/* Part 5: Status utils methods */

bool Tensor::defined() const { return impl_ != nullptr; }

bool Tensor::initialized() const { return defined() && impl_->initialized(); }

bool Tensor::is_initialized() const {
  return defined() && impl_->initialized();
}

void Tensor::reset() { impl_.reset(); }

/* Part 6: Operator overloading */

Tensor &Tensor::operator=(const Tensor &x) & {
  impl_ = x.impl_;
  autograd_meta_ = x.autograd_meta_;
  name_ = x.name_;
  place_ = x.place_;
  return *this;
}

Tensor &Tensor::operator=(Tensor &&x) & {
  impl_ = std::move(x.impl_);
  autograd_meta_ = std::move(x.autograd_meta_);
  name_ = std::move(x.name_);
  place_ = std::move(x.place_);
  return *this;
}

AbstractAutogradMeta *Tensor::get_autograd_meta() const {
  return autograd_meta_.get();
}

void Tensor::set_autograd_meta(
    std::shared_ptr<AbstractAutogradMeta> autograd_meta) {
  autograd_meta_ = std::move(autograd_meta);
}

void Tensor::bump_inplace_version() {
  if (is_dense_tensor()) {
    auto &inplace_version_counter =
        std::dynamic_pointer_cast<phi::DenseTensor>(impl_)
            ->InplaceVersionCounter();
    inplace_version_counter.Bump();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "bump_inplace_version is only supported on DenseTensor now."));
  }
}

uint32_t Tensor::current_inplace_version() {
  if (is_dense_tensor()) {
    auto &inplace_version_counter =
        std::dynamic_pointer_cast<phi::DenseTensor>(impl_)
            ->InplaceVersionCounter();
    return inplace_version_counter.CurrentVersion();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "current_inplace_version is only supported on DenseTensor now."));
  }
  return 0;
}

void Tensor::reset_inplace_version(bool set_to_zero) {
  if (set_to_zero) {
    if (is_dense_tensor()) {
      auto &inplace_version_counter =
          std::dynamic_pointer_cast<phi::DenseTensor>(impl_)
              ->InplaceVersionCounter();
      inplace_version_counter.SetInplaceVersionToZero();
    }
  }
}

}  // namespace experimental
}  // namespace paddle
