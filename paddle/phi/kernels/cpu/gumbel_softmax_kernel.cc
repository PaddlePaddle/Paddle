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

#include "paddle/phi/kernels/gumbel_softmax_kernel.h"
#include "paddle/phi/kernels/impl/gumbel_softmax_kernel_impl.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct GumbleNoiseGenerator<CPUContext, T> {
  static void Transform(const CPUContext& ctx,
                        const T* input_data,
                        T* output_data,
                        int size_to_axis,
                        int size_from_axis,
                        const float temperature) {
    // generate uniform random number
    const int size = size_to_axis * size_from_axis;
    std::uniform_real_distribution<T> dist(0.00001, 1);
    auto engine = ctx.GetGenerator()->GetCPUEngine();
    DenseTensor random_tensor;
    random_tensor.Resize(make_ddim({size}));
    auto* random_data = ctx.template Alloc<T>(&random_tensor);
    for (int64_t i = 0; i < size; ++i) {
      random_data[i] = dist(*engine);
    }

    // generate gumbel noise
    DDim dim_2d{size_to_axis, size_from_axis};
    auto gumbel_noise_eigen = EigenMatrix<T>::From(random_tensor, dim_2d);
    gumbel_noise_eigen = -(((-(gumbel_noise_eigen.log())).log()));

    // add noise
    for (int64_t i = 0; i < size_to_axis * size_from_axis; i++) {
      output_data[i] = (input_data[i] + random_data[i]) / temperature;
    }
  }
};

template <typename T>
struct OneHotGenerator<CPUContext, T> {
  static void Transform(const CPUContext& ctx,
                        const DenseTensor& x,
                        DenseTensor* out,
                        int axis) {
    DenseTensor index;
    std::vector<int> index_dim;
    const auto rank = x.dims().size();
    const int size_to_axis = funcs::SizeToAxis(axis, x.dims());
    const int size_from_axis = funcs::SizeFromAxis(axis, x.dims());
    const int size_out_axis = funcs::SizeOutAxis(axis, x.dims());

    for (int i = 0; i < x.dims().size(); i++) {
      if (i != axis) index_dim.push_back(x.dims().Get()[i]);
    }
    DDim index_ddim(index_dim.data(), rank - 1);
    index.Resize(index_ddim);
    auto* index_data = ctx.template Alloc<int>(&index);

#define CALL_ARG_MINMAX_FUNCTOR(rank)               \
  ArgMaxFunctor<CPUContext, T, rank> functor##rank; \
  functor##rank(ctx, *out, &index, axis);
    switch (out->dims().size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_ENFORCE_LE(
            out->dims().size(),
            6,
            errors::InvalidArgument("gumbel_softmax operator doesn't supports "
                                    "tensors whose ranks are greater "
                                    "than 6 in CPU mode."));
        break;
#undef CALL_ARG_MINMAX_FUNCTOR
    }

    funcs::set_constant(ctx, out, 0.0);
    for (int i = 0; i < size_to_axis; i++) {
      for (int j = 0; j < size_out_axis; j++) {
        *(out->data<T>() + i * size_from_axis + j +
          index_data[i * size_out_axis + j] * size_out_axis) = 1.0;
      }
    }
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(
    gumbel_softmax, CPU, ALL_LAYOUT, phi::GumbelSoftmaxKernel, float, double) {}
