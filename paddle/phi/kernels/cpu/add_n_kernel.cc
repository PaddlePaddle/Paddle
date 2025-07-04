// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/impl/add_n_kernel_impl.h"

#include "glog/logging.h"

namespace phi {

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const TensorBase*>& x,
                DenseTensor* out) {
  size_t in_num = x.size();
  dev_ctx.template Alloc<T>(out);

  bool in_place = false;
  if (!x.empty() && x[0]->initialized() && DenseTensor::classof(x[0])) {
    if ((static_cast<const DenseTensor*>(x[0]))->Holder() == out->Holder()) {
      in_place = true;
      if (in_num == 1) {
        return;
      }
    }
  }

  // using MPType to keep precision
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();

  if constexpr (std::is_same_v<MPType, T>) {
    // compute in out
    auto result = EigenVector<T>::Flatten(*out);

    if (!in_place) {
      phi::funcs::SetConstant<Context, T> constant_functor;
      constant_functor(dev_ctx, out, static_cast<T>(0));
    }

    phi::funcs::SelectedRowsAddToTensor<Context, T> functor;
    size_t start = in_place ? 1 : 0;
    for (size_t i = start; i < in_num; i++) {
      if (DenseTensor::classof(x[i])) {
        auto& in_t = *(static_cast<const DenseTensor*>(x[i]));
        if (!in_t.initialized() || in_t.numel() == 0) {
          continue;
        }
        auto in = EigenVector<T>::Flatten(in_t);
        result.device(place) = result + in;
      } else if (SelectedRows::classof(x[i])) {
        auto& in_t = *(static_cast<const SelectedRows*>(x[i]));
        functor(dev_ctx, in_t, out);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Expected type of Input(X) of %d-th must be Tensor, "
            "SelectedRows. But got "
            "unsupported type: %s.",
            x[i]->type_info().name()));
      }
    }
  } else {
    // compute in temp_out by using MPType
    DenseTensor temp_out;
    temp_out.Resize(out->dims());
    dev_ctx.template Alloc<MPType>(&temp_out);

    auto result_mp = EigenVector<MPType>::Flatten(temp_out);

    // set temp_out
    phi::funcs::SetConstant<Context, MPType> constant_functor;
    if (in_place && DenseTensor::classof(x[0]) && x[0]->initialized()) {
      auto& in_0 = *(static_cast<const DenseTensor*>(x[0]));
      if (in_0.numel()) {
        auto in_0_e = EigenVector<T>::Flatten(in_0).template cast<MPType>();
        result_mp.device(place) = in_0_e;
      } else {
        constant_functor(dev_ctx, &temp_out, static_cast<MPType>(0));
      }
    } else {
      constant_functor(dev_ctx, &temp_out, static_cast<MPType>(0));
    }

    phi::funcs::SelectedRowsAddToTensor<Context, MPType> functor;
    size_t start = in_place ? 1 : 0;
    for (size_t i = start; i < in_num; i++) {
      if (DenseTensor::classof(x[i])) {
        auto& in_t = *(static_cast<const DenseTensor*>(x[i]));
        if (!in_t.initialized() || in_t.numel() == 0) {
          continue;
        }
        auto in = EigenVector<T>::Flatten(in_t).template cast<MPType>();
        result_mp.device(place) = result_mp + in;
      } else if (SelectedRows::classof(x[i])) {
        auto& in_t = *(static_cast<const SelectedRows*>(x[i]));
        functor(dev_ctx, in_t, &temp_out);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Expected type of Input(X) of %d-th must be Tensor, "
            "SelectedRows. But got "
            "unsupported type: %s.",
            x[i]->type_info().name()));
      }
    }

    // cast back to T, and copy to out
    auto result = EigenVector<T>::Flatten(*out);
    result.device(place) = result_mp.template cast<T>();
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(add_n,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddNKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_n_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddNArrayKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
