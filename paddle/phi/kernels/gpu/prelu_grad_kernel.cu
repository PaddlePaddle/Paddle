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

#include "paddle/phi/kernels/prelu_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/prelu_funcs.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

namespace phi {

enum PRELU_MODE { Element, ChannelFirst, ChannelLast, PRELU_Scalar };

template <typename T>
__global__ void PReluOpGradKernel(const T* x_ptr,
                                  const T* alpha_ptr,
                                  const T* out_grad_ptr,
                                  T* x_grad_ptr,
                                  T* alpha_grad_ptr,
                                  size_t channel_num,
                                  size_t plane_size,
                                  size_t spatial_size,
                                  size_t numel,
                                  PRELU_MODE mode) {
  CUDA_KERNEL_LOOP(index, numel) {
    T scale;
    if (mode == Element) {
      size_t element_index = index % spatial_size;
      scale = alpha_ptr[element_index];
    } else if (mode == ChannelFirst) {
      size_t temp = index / plane_size;
      size_t channel_index = temp % channel_num;
      scale = alpha_ptr[channel_index];
    } else if (mode == ChannelLast) {
      size_t channel_index = index % channel_num;
      scale = alpha_ptr[channel_index];
    } else {
      scale = alpha_ptr[0];
    }
    T x = x_ptr[index];
    T out_grad = out_grad_ptr[index];
    T zero = static_cast<T>(0);
    if (x_grad_ptr != nullptr)
      x_grad_ptr[index] = (x > zero) ? out_grad : scale * out_grad;
    if (alpha_grad_ptr != nullptr)
      alpha_grad_ptr[index] = (x > zero) ? zero : x * out_grad;
  }
}

template <typename T>
class PreluOpGradFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T* x,
                  const T* alpha,
                  const T* out_grad,
                  T* x_grad,
                  T* alpha_grad,
                  const DDim& input_dims,
                  PRELU_MODE mode) {
    size_t numel = 1;
    for (size_t i = 0; i < input_dims.size(); ++i) {
      numel *= input_dims[i];
    }
    size_t plane_size = numel / input_dims[0] / input_dims[1];
    size_t spatial_size = numel / input_dims[0];
    size_t channel =
        mode == ChannelLast ? input_dims[input_dims.size() - 1] : input_dims[1];

    PReluOpGradKernel<
        T><<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>(
        x,
        alpha,
        out_grad,
        x_grad,
        alpha_grad,
        channel,
        plane_size,
        spatial_size,
        numel,
        mode);
  }
};

template <typename T, typename Context>
void PReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& alpha,
                     const DenseTensor& out_grad,
                     const std::string& mode,
                     const std::string& data_format,
                     DenseTensor* x_grad,
                     DenseTensor* alpha_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const T* x_ptr = x.data<T>();
  const T* alpha_ptr = alpha.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  T* x_grad_ptr = x_grad ? dev_ctx.template Alloc<T>(x_grad) : nullptr;
  T* alpha_grad_ptr =
      alpha_grad ? dev_ctx.template Alloc<T>(alpha_grad) : nullptr;

  if (!x_grad && !alpha_grad) return;

  int numel = x.numel();
  auto dim = x.dims();
  auto x_rank = dim.size();
  std::vector<int> input_shape = phi::vectorize<int>(dim);
  auto stream = dev_ctx.stream();

  T* alpha_grad_tmp_ptr;
  DenseTensor alpha_grad_tmp;
  if (alpha_grad_ptr == nullptr) {
    alpha_grad_tmp_ptr = alpha_grad_ptr;
  } else {
    DenseTensorMeta alpha_grad_meta(
        alpha_grad->dtype(), dim, alpha_grad->layout());
    alpha_grad_tmp = phi::Empty(dev_ctx, std::move(alpha_grad_meta));
    alpha_grad_tmp_ptr = alpha_grad_tmp.data<T>();
  }

  PRELU_MODE m;
  bool channel_last = false;
  if (mode == "element") {
    m = Element;
  } else if (mode == "channel") {
    channel_last = data_format == "NHWC";
    m = channel_last ? ChannelLast : ChannelFirst;
  } else {
    m = PRELU_Scalar;
  }
  PreluOpGradFunctor<T> prelu_grad;
  prelu_grad(stream,
             x_ptr,
             alpha_ptr,
             out_grad_ptr,
             x_grad_ptr,
             alpha_grad_tmp_ptr,
             dim,
             m);

  if (alpha_grad_tmp_ptr == nullptr) return;

  std::vector<int> reduce_dims;
  for (size_t i = 0; i < dim.size(); i++) {
    if (mode == "channel" && !channel_last && i == 1) continue;
    if (mode == "channel" && channel_last && i == dim.size() - 1) continue;
    if (mode == "element" && i != 0) continue;
    reduce_dims.push_back(i);
  }

  phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      static_cast<const phi::GPUContext&>(dev_ctx),
      alpha_grad_tmp,
      alpha_grad,
      kps::IdentityFunctor<T>(),
      reduce_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(prelu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PReluGradKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
