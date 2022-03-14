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

#include "paddle/phi/api/lib/ext_compat_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_base.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
namespace paddle {
namespace experimental {

static void copy__impl(const phi::KernelKey &kernel_key,
                       const phi::DenseTensor &src,
                       phi::DenseTensor *dst,
                       bool blocking) {
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "copy", kernel_key);

  VLOG(6) << "copy API kernel key: " << kernel_key;
  VLOG(6) << "copy API kernel: " << kernel;

  auto *dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  using kernel_signature = void (*)(const platform::DeviceContext &,
                                    const phi::DenseTensor &,
                                    phi::Place,
                                    bool,
                                    phi::DenseTensor *);

  auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  if (!dst->initialized()) {
    (*kernel_fn)(*dev_ctx,
                 src,
                 phi::TransToPhiPlace(kernel_key.backend()),
                 blocking,
                 dst);
  } else {
    (*kernel_fn)(*dev_ctx, src, dst->place(), blocking, dst);
  }
}

// declare cast api
Tensor cast(const Tensor &x, DataType out_dtype);
Tensor copy_to(const Tensor &x, Backend backend, bool blocking);

Tensor Tensor::cast(DataType target_type) const {
  return experimental::cast(*this, target_type);
}

Tensor Tensor::copy_to(Backend backend, bool blocking) const {
  return experimental::copy_to(*this, backend, blocking);
}

template <typename T>
Tensor Tensor::copy_to(const PlaceType &target_place) const {
  LOG(WARNING) << "The Tensor's `copy_to` method is deprecated since version "
                  "2.3, and will be removed in version 2.4, please use "
                  "`copy_to` method without template argument instead. "
                  "reason: copying a Tensor to another device does not need "
                  "to specify the data type template argument.";
  return copy_to(ConvertExtPlaceToBackend(target_place), /*blocking=*/false);
}

template PADDLE_API Tensor
Tensor::copy_to<float>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<double>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int64_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int32_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<uint8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int16_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<bool>(const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<float>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<double>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::float16>(const PlaceType &target_place) const;

void Tensor::copy_(const Tensor &src,
                   bool blocking,
                   const phi::Place &target_place) {
  if (!src.is_initialized()) {
    VLOG(8) << "Src is empty, skip copy";
    return;
  }
  // Prepare copy kernel key and outputs
  auto kernel_key_set = ParseKernelKeyByInputArgs(src);
  VLOG(3) << "Deep copy Tensor from " << src.name() << " to " << name();
  if (is_initialized()) {
    PADDLE_ENFORCE_EQ(dtype(),
                      src.dtype(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s, "
                          "Tensor Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(target_place,
                      inner_place(),
                      platform::errors::PreconditionNotMet(
                          "Place is different of dst tensor and args %s, which "
                          "current tensor holds %s "
                          "Copy cannot be performed!",
                          target_place.DebugString(),
                          inner_place().DebugString()));
    kernel_key_set.backend_set =
        kernel_key_set.backend_set |
        BackendSet(phi::TransToPhiBackend(inner_place()));
  } else {
    if (src.is_dense_tensor()) {
      set_impl(std::make_shared<phi::DenseTensor>(
          phi::make_intrusive<SharedStorage>(target_place),
          phi::DenseTensorMeta()));
    } else if (src.is_selected_rows()) {
      set_impl(std::make_shared<phi::SelectedRows>());
    }
    phi::MetaTensor meta_out(impl_.get());
    meta_out.share_meta(phi::MetaTensor(src.impl_.get()));
    // Deep Copy AutoGrad info from src to self.
    *autograd_meta_ = *(src.autograd_meta_);
  }

  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  if (src.is_dense_tensor()) {
    PADDLE_ENFORCE_EQ(
        is_dense_tensor(),
        true,
        paddle::platform::errors::InvalidArgument(
            "We can only copy same tensor, but we got phi::DenseTensor from "
            "src and %s from dst impl, please check this first",
            impl_->type_info().name()));
    paddle::experimental::copy__impl(
        kernel_key,
        *(std::static_pointer_cast<phi::DenseTensor>(src.impl_)),
        static_cast<phi::DenseTensor *>(impl_.get()),
        blocking);
  } else if (src.is_selected_rows()) {
    PADDLE_ENFORCE_EQ(
        is_selected_rows(),
        true,
        paddle::platform::errors::InvalidArgument(
            "We can only copy same tensor, but we got phi::SelectedRows from "
            "src and %s from dst impl, please check this first",
            impl_->type_info().name()));
    auto src_selected_rows =
        std::static_pointer_cast<phi::SelectedRows>(src.impl_);
    auto dst_selected_rows = std::static_pointer_cast<phi::SelectedRows>(impl_);
    auto &src_tensor = src_selected_rows->value();
    auto *dst_tensor = dst_selected_rows->mutable_value();
    if (dst_tensor && dst_tensor->IsInitialized()) {
      PADDLE_ENFORCE_EQ(dst_tensor->dims(),
                        src_tensor.dims(),
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different dims with Tensor %s, "
                            "Tensor Copy cannot be performed!",
                            name(),
                            src.name()));
    } else {
      dst_tensor->Resize(src_tensor.dims());
    }
    paddle::experimental::copy__impl(
        kernel_key, src_tensor, dst_tensor, blocking);
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "We currently only support dense tensor copy for now and if u need to "
        "copy selected rows please raise a issue."));
  }
}

}  // namespace experimental
}  // namespace paddle
