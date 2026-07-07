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
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/impl/gumbel_softmax_kernel_impl.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/common/enforce.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/cub.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;

template <typename T>
struct UniformCUDAGenerator {
  T min_, max_;
  uint32_t seed_;
  uint64_t offset_ = 0;
  HOSTDEVICE UniformCUDAGenerator(T min, T max, uint32_t seed)
      : min_(min), max_(max), seed_(seed) {}
  HOSTDEVICE UniformCUDAGenerator(T min, T max, uint32_t seed, uint64_t offset)
      : min_(min), max_(max), seed_(seed), offset_(offset) {}

  HOSTDEVICE T operator()(const int64_t n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(static_cast<uint64_t>(n) + offset_);
    return dist(rng);
  }
};

template <typename T, size_t BlockDim>
__global__ void OneHotCUDAKernel(const int64_t height,
                                 const int64_t width,
                                 const int64_t size_out_axis,
                                 const T init,
                                 const T* in,
                                 T* out) {
  typedef cub::BlockReduce<KeyValuePair<int64_t, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int64_t idx = blockIdx.x; idx < height; idx += gridDim.x) {
    KeyValuePair<int64_t, T> kv_pair = {-1, init};
    int h = idx / size_out_axis;
    int w = idx % size_out_axis;
    cub::ArgMax reducer;
    for (int64_t k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair = reducer(
          {k, in[h * width * size_out_axis + k * size_out_axis + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      int index = static_cast<int>(kv_pair.key);
      out[h * width * size_out_axis + index * size_out_axis + w] = 1;
    }
    __syncthreads();
  }
}

template <typename T>
struct OneHotGenerator<GPUContext, T> {
  static void Transform(const GPUContext& dev_ctx,
                        const DenseTensor& X,
                        DenseTensor* out,
                        int axis) {
    const int64_t size_to_axis = funcs::SizeToAxis(axis, X.dims());
    const int64_t size_from_axis = funcs::SizeFromAxis(axis, X.dims());
    const int64_t size_out_axis = funcs::SizeOutAxis(axis, X.dims());
    constexpr int thread_size = 512;
    int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int64_t height = static_cast<int64_t>(size_to_axis) * size_out_axis;
    int64_t block_size_64 = height < max_grid_dimx ? height : max_grid_dimx;
    PADDLE_ENFORCE_LE_UINT32_MAX(block_size_64,
                                 "gumbel_softmax one hot grid.x");
    const uint32_t block_size = static_cast<uint32_t>(block_size_64);

    DenseTensor input_tensor;
    input_tensor.Resize(out->dims());
    dev_ctx.template Alloc<T>(&input_tensor);
    Copy(dev_ctx, *out, dev_ctx.GetPlace(), false, &input_tensor);
    funcs::set_constant(dev_ctx, out, static_cast<T>(0.0));
    OneHotCUDAKernel<T, thread_size>
        <<<block_size,
           static_cast<uint32_t>(thread_size),
           0,
           dev_ctx.stream()>>>(height,
                               size_from_axis / size_out_axis,
                               size_out_axis,
                               std::numeric_limits<T>::lowest(),
                               input_tensor.data<T>(),
                               out->data<T>());
  }
};

template <typename T, typename MT>
__global__ void AddGumbelNoiseCUDAKernel(const T* input_data,
                                         T* output_data,
                                         MT* noise,
                                         const float temperature,
                                         int64_t n) {
  int64_t index =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  int64_t step =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (int64_t i = index; i < n; i += step) {
    MT gumbel_noise = -log(-log(noise[i]));
    output_data[i] = static_cast<T>(
        (gumbel_noise + static_cast<MT>(input_data[i])) / temperature);
  }
}

template <typename T>
struct GumbleNoiseGenerator<GPUContext, T> {
  static void Transform(const GPUContext& dev_ctx,
                        const T* input_data,
                        T* output_data,
                        int64_t size_to_axis,
                        int64_t size_from_axis,
                        const float temperature) {
    DenseTensor random_tensor;
    int64_t size = static_cast<int64_t>(size_to_axis) * size_from_axis;
    random_tensor.Resize({size});
    using MT = typename MPTypeTrait<T>::Type;
    MT* random_data = dev_ctx.template Alloc<MT>(&random_tensor);

    // generate gumbel noise
    int device_id = dev_ctx.GetPlace().GetDeviceId();
    auto gen_cuda = dev_ctx.GetGenerator();

    auto seed_offset = gen_cuda->IncrementOffset(1);
    uint64_t seed = seed_offset.first;
    uint64_t offset = seed_offset.second;
    const uint64_t offset_64 = size * offset;
    const uint32_t seed_u32 = static_cast<uint32_t>(seed);

    thrust::counting_iterator<int64_t> index_sequence_begin(0);
    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + size,
        thrust::device_ptr<MT>(random_data),
        UniformCUDAGenerator<MT>(0.00001, 1, seed_u32, offset_64));

    // add gumbel noise to X
    const int thread_size = 512;
    int64_t block_size_64 = size / thread_size + (size % thread_size != 0);
    int64_t max_grid_dim = dev_ctx.GetCUDAMaxGridDimSize()[0];
    block_size_64 = block_size_64 < max_grid_dim ? block_size_64 : max_grid_dim;
    PADDLE_ENFORCE_LE_UINT32_MAX(block_size_64, "gumbel_softmax noise grid.x");
    const uint32_t block_size = static_cast<uint32_t>(block_size_64);
    constexpr uint32_t thread_size_u32 = static_cast<uint32_t>(thread_size);
    AddGumbelNoiseCUDAKernel<T>
        <<<block_size, thread_size_u32, 0, dev_ctx.stream()>>>(
            input_data, output_data, random_data, temperature, size);
  }
};

}  // namespace phi
#endif

PD_REGISTER_KERNEL(gumbel_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::GumbelSoftmaxKernel,
                   phi::float16,
                   float,
                   double) {}
