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
    }
  }

  // Using MPType to leverage precision
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  DenseTensor temp_out;
  temp_out.Resize(out->dims());
  dev_ctx.template Alloc<MPType>(&temp_out);
  phi::funcs::SetConstant<Context, MPType> constant_functor;
  constant_functor(dev_ctx, &temp_out, static_cast<MPType>(0));

  auto result_mp = EigenVector<MPType>::Flatten(temp_out);
  auto& place = *dev_ctx.eigen_device();
  int start = in_place ? 1 : 0;
  if (!in_place) {
    if ((in_num >= 2) && DenseTensor::classof(x[0]) &&
        DenseTensor::classof(x[1]) && x[0]->initialized() &&
        x[1]->initialized()) {
      auto& in_0 = *(static_cast<const DenseTensor*>(x[0]));
      auto& in_1 = *(static_cast<const DenseTensor*>(x[1]));
      if (in_0.numel() && in_1.numel()) {
        auto in_0_e = EigenVector<T>::Flatten(in_0).template cast<MPType>();
        auto in_1_e = EigenVector<T>::Flatten(in_1).template cast<MPType>();
        result_mp.device(place) = in_0_e + in_1_e;
        start = 2;
      }
    }
  }

  phi::funcs::SelectedRowsAddToTensor<Context, MPType> functor;
  // If in_place, just skip the first tensor
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
  auto result = EigenVector<T>::Flatten(*out);
  result.device(place) = result_mp.template cast<T>();
  VLOG(10) << "end add_n kernel";
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
