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

#include "paddle/pten/api/include/tensor.h"

#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/pten/api/lib/ext_compat_utils.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

/**
 * [ Why still include the fluid headers? ]
 *
 * We hope to organize the basic implementation of Tensor and the logic related
 * to Tensor computation into an independent library, which we call
 * [Tensor Operation Library, pten], so we extract or rewrite the original
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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

namespace paddle {
namespace experimental {

/////// Tensor Methods ////////

/* Part 1: Construction and destruction methods */

Tensor::Tensor(std::shared_ptr<pten::TensorBase> tensor_impl)
    : impl_(std::move(tensor_impl)) {
  PADDLE_ENFORCE_NOT_NULL(impl_,
                          platform::errors::InvalidArgument(
                              "TensorImpl with nullptr is not supported"));
}

Tensor::Tensor(const PlaceType &place)
    : impl_(std::move(std::make_shared<pten::DenseTensor>(
          std::move(pten::make_intrusive<SharedStorage>(
              ConvertExtPlaceToInnerPlace(place))),
          std::move(pten::DenseTensorMeta(pten::DataType::UNDEFINED,
                                          framework::make_ddim({}),
                                          pten::DataLayout::NCHW))))) {}

Tensor::Tensor(const PlaceType &place, const std::vector<int64_t> &shape)
    : impl_(std::move(std::make_shared<pten::DenseTensor>(
          std::move(pten::make_intrusive<SharedStorage>(
              ConvertExtPlaceToInnerPlace(place))),
          std::move(pten::DenseTensorMeta(pten::DataType::UNDEFINED,
                                          framework::make_ddim(shape),
                                          pten::DataLayout::NCHW))))) {}

/* Part 2: Dimension, DataType and DataLayout methods */

int64_t Tensor::numel() const { return impl_->numel(); }

int64_t Tensor::size() const { return impl_->numel(); }

paddle::framework::DDim Tensor::dims() const { return impl_->dims(); }

std::vector<int64_t> Tensor::shape() const {
  return paddle::framework::vectorize<int64_t>(impl_->dims());
}

void Tensor::reshape(const std::vector<int64_t> &shape) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The reshape operation is not supported now, "
      "and it will be implemented by calling the reshape kernel later."));
}

DataType Tensor::dtype() const { return impl_->data_type(); }

DataType Tensor::type() const { return impl_->data_type(); }

DataLayout Tensor::layout() const { return impl_->layout(); }

/* Part 3: Device and Backend methods */

PlaceType Tensor::place() const {
  return ConvertInnerPlaceToExtPlace(impl_->place());
}

paddle::platform::Place Tensor::inner_place() const { return impl_->place(); }

bool Tensor::is_cpu() const {
  return paddle::platform::is_cpu_place(impl_->place());
}

bool Tensor::is_cuda() const {
  return paddle::platform::is_gpu_place(impl_->place());
}

/* Part 4: Data Access methods */

template <typename T>
T *Tensor::mutable_data() {
  if (impl_->type_info().name() == "DenseTensor") {
    return std::dynamic_pointer_cast<pten::DenseTensor>(impl_)
        ->mutable_data<T>();
  }
  return nullptr;
}

template PD_DLL_DECL float *Tensor::mutable_data<float>();
template PD_DLL_DECL double *Tensor::mutable_data<double>();
template PD_DLL_DECL int64_t *Tensor::mutable_data<int64_t>();
template PD_DLL_DECL int32_t *Tensor::mutable_data<int32_t>();
template PD_DLL_DECL uint8_t *Tensor::mutable_data<uint8_t>();
template PD_DLL_DECL int8_t *Tensor::mutable_data<int8_t>();
template PD_DLL_DECL int16_t *Tensor::mutable_data<int16_t>();
template PD_DLL_DECL bool *Tensor::mutable_data<bool>();
template PD_DLL_DECL paddle::platform::complex<float>
    *Tensor::mutable_data<paddle::platform::complex<float>>();
template PD_DLL_DECL paddle::platform::complex<double>
    *Tensor::mutable_data<paddle::platform::complex<double>>();
template PD_DLL_DECL paddle::platform::float16 *
Tensor::mutable_data<paddle::platform::float16>();

template <typename T>
T *Tensor::mutable_data(const PlaceType &place) {
  auto inner_place = ConvertExtPlaceToInnerPlace(place);
  PADDLE_ENFORCE_EQ(
      platform::is_same_place(inner_place, impl_->place()),
      true,
      platform::errors::Unimplemented("Modification of tensor place through "
                                      "mutable_data is not supported now"));
  return mutable_data<T>();
}

template PD_DLL_DECL float *Tensor::mutable_data<float>(const PlaceType &place);
template PD_DLL_DECL double *Tensor::mutable_data<double>(
    const PlaceType &place);
template PD_DLL_DECL int64_t *Tensor::mutable_data<int64_t>(
    const PlaceType &place);
template PD_DLL_DECL int32_t *Tensor::mutable_data<int32_t>(
    const PlaceType &place);
template PD_DLL_DECL uint8_t *Tensor::mutable_data<uint8_t>(
    const PlaceType &place);
template PD_DLL_DECL int8_t *Tensor::mutable_data<int8_t>(
    const PlaceType &place);
template PD_DLL_DECL int16_t *Tensor::mutable_data<int16_t>(
    const PlaceType &place);
template PD_DLL_DECL bool *Tensor::mutable_data<bool>(const PlaceType &place);
template PD_DLL_DECL paddle::platform::complex<float> *
Tensor::mutable_data<paddle::platform::complex<float>>(const PlaceType &place);
template PD_DLL_DECL paddle::platform::complex<double> *
Tensor::mutable_data<paddle::platform::complex<double>>(const PlaceType &place);
template PD_DLL_DECL paddle::platform::float16 *
Tensor::mutable_data<paddle::platform::float16>(const PlaceType &place);

template <typename T>
const T *Tensor::data() const {
  if (impl_->type_info().name() == "DenseTensor") {
    return std::dynamic_pointer_cast<pten::DenseTensor>(impl_)->data<T>();
  }
  return nullptr;
}

template PD_DLL_DECL const float *Tensor::data<float>() const;
template PD_DLL_DECL const double *Tensor::data<double>() const;
template PD_DLL_DECL const int64_t *Tensor::data<int64_t>() const;
template PD_DLL_DECL const int32_t *Tensor::data<int32_t>() const;
template PD_DLL_DECL const uint8_t *Tensor::data<uint8_t>() const;
template PD_DLL_DECL const int8_t *Tensor::data<int8_t>() const;
template PD_DLL_DECL const int16_t *Tensor::data<int16_t>() const;
template PD_DLL_DECL const bool *Tensor::data<bool>() const;
template PD_DLL_DECL const paddle::platform::complex<float>
    *Tensor::data<paddle::platform::complex<float>>() const;
template PD_DLL_DECL const paddle::platform::complex<double>
    *Tensor::data<paddle::platform::complex<double>>() const;
template PD_DLL_DECL const paddle::platform::float16 *
Tensor::data<paddle::platform::float16>() const;

template <typename T>
T *Tensor::data() {
  PADDLE_THROW(platform::errors::Unimplemented(
      "It is not currently supported to directly obtain the modifiable data "
      "address through the tensor::data<T>() method, please use the "
      "tensor::mutable_data<T>() method."));
  return nullptr;
}

template PD_DLL_DECL float *Tensor::data<float>();
template PD_DLL_DECL double *Tensor::data<double>();
template PD_DLL_DECL int64_t *Tensor::data<int64_t>();
template PD_DLL_DECL int32_t *Tensor::data<int32_t>();
template PD_DLL_DECL uint8_t *Tensor::data<uint8_t>();
template PD_DLL_DECL int8_t *Tensor::data<int8_t>();
template PD_DLL_DECL int16_t *Tensor::data<int16_t>();
template PD_DLL_DECL bool *Tensor::data<bool>();
template PD_DLL_DECL paddle::platform::complex<float>
    *Tensor::data<paddle::platform::complex<float>>();
template PD_DLL_DECL paddle::platform::complex<double>
    *Tensor::data<paddle::platform::complex<double>>();
template PD_DLL_DECL paddle::platform::float16 *
Tensor::data<paddle::platform::float16>();

Tensor Tensor::slice(const int64_t begin_idx, const int64_t end_idx) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The slice operation is not supported now, "
      "and it will be implemented by calling the slice kernel later."));
  return Tensor();
}

std::shared_ptr<pten::TensorBase> Tensor::impl() const { return impl_; }

void Tensor::set_impl(const std::shared_ptr<pten::TensorBase> &impl) {
  impl_ = impl;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
gpuStream_t Tensor::stream() const {
  return platform::stream::get_current_stream(-1)->raw_stream();
}
#endif

/* Part 5: Data Transform methods */

template <typename T>
Tensor Tensor::copy_to(const PlaceType &target_place) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The copy_to operation is not supported now, "
      "and it will be implemented by calling the copy kernel later."));
  return Tensor();
}

template PD_DLL_DECL Tensor
Tensor::copy_to<float>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<double>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<int64_t>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<int32_t>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<uint8_t>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<int8_t>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<int16_t>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<bool>(const PlaceType &target_place) const;
template PD_DLL_DECL Tensor Tensor::copy_to<paddle::platform::complex<float>>(
    const PlaceType &target_place) const;
template PD_DLL_DECL Tensor Tensor::copy_to<paddle::platform::complex<double>>(
    const PlaceType &target_place) const;
template PD_DLL_DECL Tensor
Tensor::copy_to<paddle::platform::float16>(const PlaceType &target_place) const;

Tensor Tensor::to(const PlaceType &target_place) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The to operation is not supported now, "
      "and it will be implemented by calling the copy kernel later."));
  return Tensor();
}

Tensor Tensor::cast(const DataType &target_type) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The cast operation is not supported now, "
      "and it will be implemented by calling the cast kernel later."));
  return Tensor();
}

/* Part 6: Status utils methods */

bool Tensor::defined() const { return impl_ != nullptr; }

bool Tensor::initialized() const {
  return impl_ != nullptr && impl_->initialized();
}

bool Tensor::is_initialized() const {
  return impl_ != nullptr && impl_->initialized();
}

void Tensor::reset() { impl_.reset(); }

/* Part 7: Operator overloading */

Tensor &Tensor::operator=(const Tensor &x) & {
  impl_ = x.impl_;
  autograd_meta_ = x.autograd_meta_;
  return *this;
}

Tensor &Tensor::operator=(Tensor &&x) & {
  impl_ = std::move(x.impl_);
  autograd_meta_ = std::move(x.autograd_meta_);
  return *this;
}

AbstractAutogradMeta *Tensor::get_autograd_meta() const {
  return autograd_meta_.get();
}

void Tensor::set_autograd_meta(
    std::shared_ptr<AbstractAutogradMeta> autograd_meta) {
  autograd_meta_ = std::move(autograd_meta);
}

}  // namespace experimental
}  // namespace paddle
