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

#include "paddle/fluid/operators/gather_scatter_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class TensorAssign {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceMul {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    gpuAtomicMul(*self_data, *src_data);
  }
};
// static ReduceMul reduce_mul;

class ReduceAdd {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    gpuAtomicAdd(*self_data, *src_data);
  }
};
// static ReduceAdd reduce_add;

// // essentialy rewritten related to legacy::launch_kernel parts
// template <int nt, int vt, typename func_t>
// __global__ void _scatter_gather_elementwise_kernel(int N, func_t f) {
//   constexpr int nv = nt * vt;
//   int idx = nv * blockIdx.x + threadIdx.x;

//   #pragma unroll
//   for (int i = 0; i < vt; ++i) {
//     if (idx < N) {
//       f(idx);
//       idx += nt;
//     }
//   }
// }

// template <int nt, int vt, typename func_t>
// static void _launch_scatter_gather_kernel(int64_t N, const func_t& f) {
//   TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
//   if (N == 0) {
//     return;
//   }

//   const dim3 block(nt);
//   const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
//   const auto stream = at::cuda::getCurrentCUDAStream();
//   _scatter_gather_elementwise_kernel<nt, vt, func_t><<<grid, block, 0,
//   stream>>>(N, f);
// }

// template <bool is_scatter_like, typename scalar_t>
// struct _cuda_gather_scatter_kernel {
//   template <typename func_t>
//   void operator() (
//     TensorIterator& iter,
//     scalar_t src_val,
//     int64_t index_size,
//     int64_t index_stride,
//     const func_t& f
//   ) {

//     loop = [=](int i){
//       offsets = getoffset();

//       f(
//         (scalar_t*)self_data + idx_dim * index_stride,
//         (scalar_t*)&src_val
//       );
//     };
//   }

// }; // _cuda_gather_scatter_kernel

template <typename tensor_t, typename index_t = int64_t,
          bool is_scatter_like = true>
struct gpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(Tensor self, int dim, const Tensor& index, const Tensor& src,
                  const std::string& method_name, const func_t& kernel_func) {
    if (index.numel() == 0) {
      return;
    }
    printf("GPU 111111 start>>>>>\n");
    auto* self_data = self.data<tensor_t>();
    // VLOG(3) << "self_data:" << *self_data;
    printf("GPU 22222 %d\n ", sizeof(*self_data));
    auto* index_data = index.data<index_t>();
    printf("GPU 33333 %d\n", sizeof(*index_data));
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    // auto self_dims = self.dims();
    auto index_dims = index.dims();
    // auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) return;
    int select_dim_size = index_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }
    printf("GPU 111111 done\n");

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }
    printf("GPU select_dim_size: %d \n", select_dim_size);
    printf("GPU outer_dim_size: %d \n", outer_dim_size);
    printf("GPU index_size: %d \n", index_size);

    printf("GPU inner_dim_size: %d \n", inner_dim_size);

    int64_t index_idx = 0;
    int64_t self_idx, src_idx;

    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          printf("GPU 00000 index idx : %d \n", index_idx);
          printf("GPU 55555 index_ptr %d :\n", index_data);
          int64_t index = index_data[index_idx];
          printf("GPU 55555 index_ptr:\n", index_data);
          /*
            gather computation formula:

            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

            scatter computation formula:

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

          */
          int64_t replace_index =
              k + index * outer_dim_size + i * outer_dim_size * select_dim_size;
          self_idx = is_scatter_like ? replace_index : index_idx;
          src_idx = is_scatter_like ? index_idx : replace_index;

          kernel_func((tensor_t*)(self_data + self_idx),
                      (tensor_t*)(src_data + src_idx));
          index_idx++;
          printf("GPU 66666 index_idx %d\n  ", index_idx);
        }
      }
    }
  }
};  // struct gpu_gather_scatter_functor

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(const Tensor& input, int dim, const Tensor& index,
                       Tensor result) {
  gpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, input, "gather_out_gpu", tensor_assign);
  return;
}
// gpu_gather_scatter_functor<tensor_t, /*index_t=*/ int64_t,
// /*is_scatter_like=*/false>()(result, dim, index, input, "gather_out_cpu",
// tensor_assign);

namespace plat = paddle::platform;
Instantiate_Template_Funtion(gpu_gather_kernel)
// Instantiate_Template_Funtion(gpu_sca_kernel)

}  // namespace operators
}  // namespace paddle
