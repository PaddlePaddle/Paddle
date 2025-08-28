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

#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

class TensorAssign {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* __restrict__ self_data,
                            const tensor_t* __restrict__ src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  __device__ void operator()(tensor_t* __restrict__ self_data,
                             const tensor_t* __restrict__ src_data) const {
    phi::CudaAtomicAdd(self_data, *src_data);
  }
};
static ReduceAdd reduce_add;

class ReduceMul {
 public:
  template <typename tensor_t>
  __device__ void operator()(tensor_t* self_data,
                             const tensor_t* src_data) const {
    phi::CudaAtomicMul(self_data, *src_data);
  }
};
static ReduceMul reduce_mul;

class ReduceMax {
 public:
  template <typename tensor_t>
  __device__ void operator()(tensor_t* __restrict__ self_data,
                             const tensor_t* __restrict__ src_data) const {
    phi::CudaAtomicMax(self_data, *src_data);
  }
};
static ReduceMax reduce_max;

class ReduceMin {
 public:
  template <typename tensor_t>
  __device__ void operator()(tensor_t* __restrict__ self_data,
                             const tensor_t* __restrict__ src_data) const {
    phi::CudaAtomicMin(self_data, *src_data);
  }
};
static ReduceMin reduce_min;

__global__ void CudaMemsetAsync(int* dest, int value, size_t size) {
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid * sizeof(int) >= size) return;
  dest[tid] = value;
}

struct DivMod {
  template <typename T>
  static __device__ __forceinline__ void divmod(T dividend,
                                                T divisor,
                                                T* __restrict__ quotient,
                                                T* __restrict__ remainder) {
    *quotient = dividend / divisor;
    *remainder = dividend % divisor;
  }
};

// compute two offsets for self tensor and src tensor
// if compute_self is true, other wise only src_offset is useful
// TODO(heqianyue): remove force inline?
// TODO(heqianyue): maybe use int32 to optimize?
template <bool compute_self>
__device__ __forceinline__ void ComputeOffset(
    const int64_t* __restrict__ index_shape,
    const int64_t* __restrict__ src_stride,
    const int64_t* __restrict__ input_stride,
    int64_t* __restrict__ src_offset,
    int64_t* __restrict__ input_offset,
    int64_t tid,
    const int ndim,
    const int dim_to_put,
    const int64_t idx_on_dim = 0) {
  // TODO(heqianyue): maybe smaller tensors can use int32
  // TODO(heqianyue): use fast divmod to optimize the speed of div and mod
  int64_t _input_offset = 0, _src_offset = 0;
  for (int d = ndim - 1; d > dim_to_put; --d) {
    // before the put dim
    int64_t index = 0;
    DivMod::divmod(tid, index_shape[d], &tid, &index);
    _src_offset += index * src_stride[d];
    if constexpr (compute_self) _input_offset += index * input_stride[d];
  }
  if constexpr (compute_self) {  // scatter like
    _src_offset += (tid % index_shape[dim_to_put]) * src_stride[dim_to_put];
    _input_offset += idx_on_dim * input_stride[dim_to_put];
  } else {
    _src_offset += idx_on_dim * src_stride[dim_to_put];
  }
  tid /= index_shape[dim_to_put];
  for (int d = dim_to_put - 1; d >= 0; --d) {
    // after the put dim
    int64_t index = 0;
    DivMod::divmod(tid, index_shape[d], &tid, &index);
    _src_offset += index * src_stride[d];
    if constexpr (compute_self) _input_offset += index * input_stride[d];
  }
  *src_offset = _src_offset;
  if constexpr (compute_self) *input_offset = _input_offset;
}

/**
 * The assign / add / mul / min / max kernels can actually be unified
 *
 * @param index_shape A reused field, the first `ndim` elements are the shape of
 * index tensor and the second `ndim` elements are the strides of src tensor the
 * third `ndim` elements are the strides of input self tensor, these
 * shape/stride info are necessary to perform correct offset mapping between
 * different tensors
 *
 * We need a ComputeOffset as offset remapper, since both the shape of src
 * tensor and input self tensor can be bigger than the shape of index tensor
 *
 * @note these kernels are all marked with __restrict__, since inherently
 * there will be no pointer aliases for normal uses. Therefore, please
 * avoid using the following kernels for INPLACE ops
 */
template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true,
          bool include_self = false>
__global__ void GatherScatterGPUKernel(
    tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    const tensor_t* __restrict__ src_data,
    int64_t self_select_dim_size,
    int64_t src_select_dim_size,
    int64_t numel,
    int dim,
    int ndim,
    const func_t& reduce_op,
    int* __restrict__ aux_buffer = nullptr) {
  extern __shared__ int64_t
      smem_shape_strides[];  // no more than 27 int64_t, won't affect occupancy

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  // we need threads to complete memory write to smem, even if current thread is
  // out of bound
  if (tid >= numel) return;
  index_t index = index_data[tid];

  const int64_t* src_strides = smem_shape_strides + ndim;
  const int64_t* input_strides = nullptr;

  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self = 0, replace_index_src = 0;
  if constexpr (is_scatter_like) {
    input_strides = smem_shape_strides +
                    ndim * 2;  // gather pass actually does not need this
    // scatter
    PADDLE_ENFORCE(
        index >= -self_select_dim_size && index < self_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%ld]",
        -self_select_dim_size,
        self_select_dim_size,
        (int64_t)index);
    if (index < 0) {
      index += self_select_dim_size;
    }
  } else {
    // gather
    PADDLE_ENFORCE(
        index >= -src_select_dim_size && index < src_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%d]",
        -src_select_dim_size,
        src_select_dim_size,
        (int32_t)index);
    if (index < 0) {
      index += src_select_dim_size;
    }
    replace_index_self = tid;
  }
  ComputeOffset<is_scatter_like>(smem_shape_strides,
                                 src_strides,
                                 input_strides,
                                 &replace_index_src,
                                 &replace_index_self,
                                 tid,
                                 ndim,
                                 dim,
                                 index);
  if constexpr (include_self) {
    // unordered-writes branch has the same behavior as torch's. Strangely,
    // the old impl performs ordered access for assign (maybe it is because
    // there was no atomic primitives for assign), and for other ops,
    // unordered atomic access is used
    reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
              static_cast<const tensor_t*>(src_data + replace_index_src));
  } else {
    bool is_op_done = false;
    phi::CudaAtomicMin(aux_buffer + replace_index_self, tid);
    __syncthreads();
    if (tid == aux_buffer[replace_index_self]) {
      self_data[replace_index_self] = src_data[replace_index_src];
      is_op_done = true;
    }
    __syncthreads();
    if (!is_op_done)
      reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
                static_cast<const tensor_t*>(src_data + replace_index_src));
  }
}

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void ScatterMeanGPUKernel(
    tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    const tensor_t* __restrict__ src_data,
    int64_t self_select_dim_size,
    int64_t src_select_dim_size,
    int64_t numel,
    int dim,
    int ndim,
    const func_t& reduce_op,
    bool include_self = true,
    int* __restrict__ aux_buffer = nullptr,
    int* __restrict__ atomic_cnt_buffer = nullptr) {
  extern __shared__ int64_t
      smem_shape_strides[];  // no more than 27 int64_t, won't affect occupancy

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  // we need threads to complete memory write to smem, even if current thread is
  // out of bound
  if (tid >= numel) return;
  index_t index = index_data[tid];

  const int64_t* src_strides = smem_shape_strides + ndim;
  const int64_t* input_strides = nullptr;

  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self = 0, replace_index_src = 0;
  if constexpr (is_scatter_like) {
    input_strides = smem_shape_strides +
                    ndim * 2;  // gather pass actually does not need this
    // scatter
    PADDLE_ENFORCE(
        index >= -self_select_dim_size && index < self_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%ld]",
        -self_select_dim_size,
        self_select_dim_size,
        (int64_t)index);
    if (index < 0) {
      index += self_select_dim_size;
    }
  } else {
    // gather
    PADDLE_ENFORCE(
        index >= -src_select_dim_size && index < src_select_dim_size,
        "The index is out of bounds, "
        "please check whether the index and "
        "input's shape meet the requirements. It should "
        "be greater or equal to [%d] and less than [%d], but received [%d]",
        -src_select_dim_size,
        src_select_dim_size,
        (int32_t)index);
    if (index < 0) {
      index += src_select_dim_size;
    }
    replace_index_self = tid;
  }
  ComputeOffset<is_scatter_like>(smem_shape_strides,
                                 src_strides,
                                 input_strides,
                                 &replace_index_src,
                                 &replace_index_self,
                                 tid,
                                 ndim,
                                 dim,
                                 index);
  if (!include_self) {
    self_data[replace_index_self] = 0;
    __syncthreads();
  }

  reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
            static_cast<const tensor_t*>(src_data + replace_index_src));

  // So this is the culprit
  phi::CudaAtomicMax(aux_buffer + replace_index_self, tid);
  phi::CudaAtomicAdd(atomic_cnt_buffer + replace_index_self, 1);
  __syncthreads();

  if (tid == aux_buffer[replace_index_self]) {
    self_data[replace_index_self] =
        self_data[replace_index_self] /
        static_cast<tensor_t>(atomic_cnt_buffer[replace_index_self]);
  }
}

template <typename index_t>
__global__ void PickWinnersScatterKernel(
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int* __restrict__ winners,
    int64_t self_select_dim_size,
    int64_t numel,
    int dim,
    int ndim) {
  extern __shared__ int64_t
      smem_shape_strides[];  // no more than 27 int64_t, won't affect occupancy

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  // we need threads to complete memory write to smem, even if current thread is
  // out of bound
  if (tid >= numel) return;
  index_t index = index_data[tid];
  if (index < 0) index += static_cast<index_t>(self_select_dim_size);

  const int64_t* input_strides = smem_shape_strides + 2 * ndim;

  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self = 0;
  ComputeOffset<false>(smem_shape_strides,
                       input_strides,
                       nullptr,
                       &replace_index_self,
                       nullptr,
                       tid,
                       ndim,
                       dim,
                       index);

  atomicMax(&winners[replace_index_self], static_cast<int>(tid));
}

template <typename tensor_t, typename index_t, typename func_t>
__global__ void ScatterWriteByWinnersKernel(
    tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ src_data,
    const int64_t* __restrict__ shape_strides,
    const int* __restrict__ winners,
    int64_t self_select_dim_size,
    int64_t numel,
    int dim,
    int ndim) {
  extern __shared__ int64_t
      smem_shape_strides[];  // no more than 27 int64_t, won't affect occupancy

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  // we need threads to complete memory write to smem, even if current thread is
  // out of bound
  if (tid >= numel) return;
  index_t index = index_data[tid];
  if (index < 0) index += static_cast<index_t>(self_select_dim_size);

  const int64_t* src_strides = smem_shape_strides + ndim;
  const int64_t* input_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_src = 0;
  ComputeOffset<true>(smem_shape_strides,
                      src_strides,
                      input_strides,
                      &replace_index_src,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);
  if (static_cast<int>(tid) == winners[replace_index_self]) {
    *(self_data + replace_index_self) = *(src_data + replace_index_src);
  }
}

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct gpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index,
                  phi::DenseTensor src,
                  const std::string& method_name,
                  const func_t& reduce_op,
                  bool include_self,
                  const phi::DeviceContext& dev_ctx) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();
    const auto* index_data = index.data<index_t>();
    const auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    auto index_dims = index.dims();
    auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) return;
    int64_t select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int64_t self_select_dim_size = self_dims[dim];
    int64_t src_select_dim_size = src_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }
    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }

    constexpr int block = 512;
    int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
    int64_t grid = (n + block - 1) / block;
    auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    DenseTensor shared_mem_tensor;
    if (method_name == "scatter_assign_gpu") {
      shared_mem_tensor.Resize({self_size});
      auto* winners = dev_ctx.Alloc<int>(&shared_mem_tensor);
      phi::funcs::set_constant(dev_ctx, &shared_mem_tensor, 0);
    }

    int64_t ndim = index.dims().size();

    DenseTensor shape_stride_dev;
    shape_stride_dev.Resize({3 * ndim});
    dev_ctx.Alloc<int64_t>(&shape_stride_dev);
    {  // deallocate host once the copy is done
      DenseTensor shape_stride_host;
      shape_stride_host.Resize({3 * ndim});
      dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
      int64_t* host_data = shape_stride_host.data<int64_t>();
      for (int64_t i = 0; i < ndim; i++) {
        host_data[i] = index_dims[i];
        host_data[i + ndim] = src.strides()[i];
        host_data[i + (ndim << 1)] = self.strides()[i];
      }
      phi::Copy(dev_ctx,
                shape_stride_host,
                dev_ctx.GetPlace(),
                false,
                &shape_stride_dev);
    }
    const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
    const size_t shared_mem_bytes = sizeof(int64_t) * shape_stride_dev.numel();

    DenseTensor aux_tensor;
    if (method_name == "scatter_assign_gpu") {
      aux_tensor.Resize({self_size});
      dev_ctx.Alloc<int>(&aux_tensor);
      phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);

      int* winners = aux_tensor.data<int>();
      // Stage 1: Get the last index to be assigned the same dst.
      PickWinnersScatterKernel<index_t>
          <<<grid, block, shared_mem_bytes, stream>>>(index_data,
                                                      shape_strides,
                                                      winners,
                                                      self_select_dim_size,
                                                      index_size,
                                                      dim,
                                                      ndim);
      // Stage 2: Only the max tid in stage 1 can write src to dst.
      ScatterWriteByWinnersKernel<tensor_t, index_t, func_t>
          <<<grid, block, shared_mem_bytes, stream>>>(self_data,
                                                      index_data,
                                                      src_data,
                                                      shape_strides,
                                                      winners,
                                                      self_select_dim_size,
                                                      index_size,
                                                      dim,
                                                      ndim);
    } else if (method_name == "scatter_mean_gpu") {
      // TODO(heqianyue): the original impl is too wasteful, this can be
      // optimized
      DenseTensor atomic_cnt_tensor;
      aux_tensor.Resize({self_size});
      atomic_cnt_tensor.Resize({self_size});
      dev_ctx.Alloc<int>(&aux_tensor);
      dev_ctx.Alloc<int>(&atomic_cnt_tensor);

      // threadidx must start with 0, otherwise atomicMax will be faulty
      phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);
      phi::funcs::set_constant(
          dev_ctx, &atomic_cnt_tensor, include_self ? 1 : 0);

      int* aux_buffer = aux_tensor.data<int>();
      int* atomic_cnt_buffer = atomic_cnt_tensor.data<int>();
      ScatterMeanGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, shared_mem_bytes, stream>>>(self_data,
                                                      index_data,
                                                      shape_strides,
                                                      src_data,
                                                      self_select_dim_size,
                                                      src_select_dim_size,
                                                      index_size,
                                                      dim,
                                                      ndim,
                                                      reduce_op,
                                                      include_self,
                                                      aux_buffer,
                                                      atomic_cnt_buffer);
    } else {
      if (include_self) {
        GatherScatterGPUKernel<tensor_t, index_t, func_t, is_scatter_like, true>
            <<<grid, block, shared_mem_bytes, stream>>>(self_data,
                                                        index_data,
                                                        shape_strides,
                                                        src_data,
                                                        self_select_dim_size,
                                                        src_select_dim_size,
                                                        index_size,
                                                        dim,
                                                        ndim,
                                                        reduce_op,
                                                        nullptr);
      } else {
        aux_tensor.Resize({self_size});
        dev_ctx.Alloc<int>(&aux_tensor);
        phi::funcs::set_constant(dev_ctx, &aux_tensor, index_size + 1);

        int* aux_buffer = aux_tensor.data<int>();
        GatherScatterGPUKernel<tensor_t,
                               index_t,
                               func_t,
                               is_scatter_like,
                               false>
            <<<grid, block, shared_mem_bytes, stream>>>(self_data,
                                                        index_data,
                                                        shape_strides,
                                                        src_data,
                                                        self_select_dim_size,
                                                        src_select_dim_size,
                                                        index_size,
                                                        dim,
                                                        ndim,
                                                        reduce_op,
                                                        aux_buffer);
      }
    }
  }
};  // struct gpu_gather_scatter_functor

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       bool include_self,
                       const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(result,
                                                          dim,
                                                          index,
                                                          self,
                                                          "gather_out_gpu",
                                                          tensor_assign,
                                                          include_self,
                                                          dev_ctx);
  return;
}

template <typename tensor_t, typename index_t>
void gpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_assign_gpu",
                                                         tensor_assign,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_add_gpu",
                                                         reduce_add,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_mul_gpu",
                                                         reduce_mul,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_mean_gpu",
                                                         reduce_add,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_max_gpu",
                                                         reduce_max,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_min_gpu",
                                                         reduce_min,
                                                         include_self,
                                                         dev_ctx);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterInputGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel) {
  // no more than 18 int64_t, different from forward kernels
  // the backward kernel does not require src, so src_strides are not needed
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (2 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  int64_t replace_index = 0;
  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;

  ComputeOffset<false>(smem_shape_strides,
                       grad_strides,
                       nullptr,
                       &replace_index,
                       nullptr,
                       tid,
                       ndim,
                       dim,
                       index);
  grad_data[replace_index] = 0;
}

template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& dev_ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  int64_t index_size = index.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int select_dim_size = index_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  constexpr int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

  int64_t ndim = index_dims.size();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({2 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({2 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      host_data[i + ndim] = grad.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  const size_t shared_mem_bytes = sizeof(int64_t) * shape_stride_dev.numel();

  ScatterInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                  index_data,
                                                  shape_strides,
                                                  dim,
                                                  index_dims.size(),
                                                  index_size);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMulInputGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ out_data,
    const tensor_t* __restrict__ x_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (2 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  int64_t replace_index = 0;
  index_t index = index_data[tid];
  // the second `ndim` elements are not used in this kernel
  const int64_t* grad_strides = smem_shape_strides + ndim;

  ComputeOffset<false>(smem_shape_strides,
                       grad_strides,
                       nullptr,
                       &replace_index,
                       nullptr,
                       tid,
                       ndim,
                       dim,
                       index);
  atomicMax(aux_buffer + replace_index, tid);
  __syncthreads();
  if (tid == aux_buffer[replace_index]) {
    grad_data[replace_index] = grad_data[replace_index] *
                               out_data[replace_index] / x_data[replace_index];
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMinMaxInputGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ out_data,
    const tensor_t* __restrict__ x_data,
    const tensor_t* __restrict__ value_data,
    const tensor_t* __restrict__ self_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* src_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index = 0, replace_index_value = 0;
  // the ordering of src_strides and grad_strides in the following function
  // param is correct
  ComputeOffset<true>(smem_shape_strides,
                      src_strides,
                      grad_strides,
                      &replace_index_value,
                      &replace_index,
                      tid,
                      ndim,
                      dim,
                      index);

  if (value_data[replace_index_value] == out_data[replace_index])
    phi::CudaAtomicAdd(aux_buffer + replace_index, 1);
  __syncthreads();
  if (out_data[replace_index] != x_data[replace_index]) {
    grad_data[replace_index] = 0;
  } else {
    grad_data[replace_index] = self_data[replace_index] /
                               static_cast<tensor_t>(aux_buffer[replace_index]);
  }
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_input_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out,
    const phi::DenseTensor& x,
    const phi::DenseTensor& value,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self UNUSED,
    const phi::DeviceContext& dev_ctx) {
  auto* grad_data = grad.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();
  const auto* self_data = self.data<tensor_t>();

  auto index_dims = index.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t select_dim_size = index_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }
  constexpr int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
  DenseTensor aux_tensor;
  aux_tensor.Resize({grad.numel()});
  dev_ctx.Alloc<int>(&aux_tensor);
  int* aux_buffer = aux_tensor.data<int>();

  int64_t ndim = index_dims.size();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({3 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      // notice that the ordering is different from forward, since
      // value.strides() is not used for mul
      host_data[i + ndim] = grad.strides()[i];
      host_data[i + (ndim << 1)] = value.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  size_t shared_mem_bytes = sizeof(int64_t) * ndim;

  if (reduce == "mul" || reduce == "multiply") {
    phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);
    shared_mem_bytes *= 2;  // 1 stride, 1 shape
    ScatterMulInputGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    x_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    aux_buffer);
  } else if (reduce == "amin" || reduce == "amax") {
    phi::funcs::set_constant(dev_ctx, &aux_tensor, 1);
    shared_mem_bytes *= 3;  // two strides, 1 shape
    ScatterMinMaxInputGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    x_data,
                                                    value_data,
                                                    self_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    aux_buffer);
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMeanInputGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int64_t grad_numel,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (2 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;

  int64_t replace_index = 0;
  ComputeOffset<false>(smem_shape_strides,
                       grad_strides,
                       nullptr,
                       &replace_index,
                       nullptr,
                       tid,
                       ndim,
                       dim,
                       index);

  atomicMax(aux_buffer + replace_index, tid);
  phi::CudaAtomicAdd(aux_buffer + grad_numel + replace_index, 1);
  __syncthreads();
  if (tid == aux_buffer[replace_index]) {
    grad_data[replace_index] =
        grad_data[replace_index] /
        static_cast<tensor_t>(aux_buffer[grad_numel + replace_index]);
  }
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_input_grad_kernel(phi::DenseTensor self,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self UNUSED,
                                        const phi::DeviceContext& dev_ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  int64_t grad_size = grad.numel();
  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t select_dim_size = index_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }
  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  DenseTensor aux_tensor;
  aux_tensor.Resize({grad_size * 2});
  dev_ctx.Alloc<int>(&aux_tensor);
  phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);
  int* aux_buffer = aux_tensor.data<int>();

  constexpr int block = 512;
  int64_t grid_memset = (grad_size + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
  // TODO(heqianyue): This kernel can be fused
  CudaMemsetAsync<<<grid_memset, block, 0, stream>>>(
      aux_buffer + grad_size, 1, sizeof(int) * grad_size);

  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;

  int64_t ndim = index_dims.size();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({2 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({2 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      host_data[i + ndim] = grad.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  size_t shared_mem_bytes = sizeof(int64_t) * ndim * 2;

  ScatterMeanInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                  index_data,
                                                  shape_strides,
                                                  dim,
                                                  ndim,
                                                  index.numel(),
                                                  grad_size,
                                                  aux_buffer);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterValueGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* self_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_grad = 0;
  ComputeOffset<true>(smem_shape_strides,
                      grad_strides,
                      self_strides,
                      &replace_index_grad,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);

  atomicMax(aux_buffer + replace_index_self, tid);
  __syncthreads();

  if (tid == aux_buffer[replace_index_self]) {
    grad_data[replace_index_grad] = self_data[replace_index_self];
  }
}

template <typename tensor_t, typename index_t>
void gpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& dev_ctx) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int select_dim_size = index_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }
  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }
  DenseTensor aux_tensor;
  aux_tensor.Resize({self.numel()});
  dev_ctx.Alloc<int>(&aux_tensor);
  phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);
  int* aux_buffer = aux_tensor.data<int>();

  constexpr int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

  int64_t ndim = index_dims.size();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({3 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      host_data[i + ndim] = grad.strides()[i];
      host_data[i + (ndim << 1)] = self.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  size_t shared_mem_bytes = sizeof(int64_t) * ndim * 3;

  ScatterValueGradGPUKernel<tensor_t, index_t>
      <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                  self_data,
                                                  index_data,
                                                  shape_strides,
                                                  dim,
                                                  ndim,
                                                  index.numel(),
                                                  aux_buffer);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMeanValueGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* self_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_grad = 0;
  ComputeOffset<true>(smem_shape_strides,
                      grad_strides,
                      self_strides,
                      &replace_index_grad,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);

  phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
  __syncthreads();

  grad_data[replace_index_grad] =
      self_data[replace_index_self] /
      static_cast<tensor_t>(aux_buffer[replace_index_self]);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterAddValueGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const tensor_t* __restrict__ self_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* self_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_grad = 0;
  ComputeOffset<true>(smem_shape_strides,
                      grad_strides,
                      self_strides,
                      &replace_index_grad,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);
  grad_data[replace_index_grad] = self_data[replace_index_self];
}

template <typename tensor_t, typename index_t>
void gpu_scatter_add_mean_value_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out UNUSED,
    const phi::DenseTensor& x UNUSED,
    const phi::DenseTensor& value UNUSED,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self,
    const phi::DeviceContext& dev_ctx UNUSED) {
  const auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t select_dim_size = index_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }
  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  constexpr int block = 512;
  int64_t ndim = index_dims.size();
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({3 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      host_data[i + ndim] = grad.strides()[i];
      host_data[i + (ndim << 1)] = self.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  size_t shared_mem_bytes = sizeof(int64_t) * ndim * 3;

  if (reduce == "mean") {
    DenseTensor aux_tensor;
    aux_tensor.Resize({self.numel()});
    dev_ctx.Alloc<int>(&aux_tensor);
    phi::funcs::set_constant(dev_ctx, &aux_tensor, include_self ? 1 : 0);
    int* aux_buffer = aux_tensor.data<int>();
    ScatterMeanValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    self_data,
                                                    index_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    aux_buffer);
  } else if (reduce == "add") {
    ScatterAddValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    self_data,
                                                    index_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel());
  }
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMulValueGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ self_data,
    const tensor_t* __restrict__ value_data,
    const tensor_t* __restrict__ out_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* self_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_grad = 0;
  ComputeOffset<true>(smem_shape_strides,
                      grad_strides,
                      self_strides,
                      &replace_index_grad,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);
  grad_data[replace_index_grad] =
      self_data[replace_index_self] *
      (out_data[replace_index_self] / value_data[replace_index_grad]);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterMinMaxValueGradGPUKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ self_data,
    const tensor_t* __restrict__ value_data,
    const tensor_t* __restrict__ out_data,
    const tensor_t* __restrict__ x_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    bool include_self,
    int* __restrict__ aux_buffer) {
  extern __shared__ int64_t smem_shape_strides[];
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;

  index_t index = index_data[tid];
  const int64_t* grad_strides = smem_shape_strides + ndim;
  const int64_t* self_strides = smem_shape_strides + 2 * ndim;

  int64_t replace_index_self = 0, replace_index_grad = 0;
  ComputeOffset<true>(smem_shape_strides,
                      grad_strides,
                      self_strides,
                      &replace_index_grad,
                      &replace_index_self,
                      tid,
                      ndim,
                      dim,
                      index);

  if (include_self &&
      x_data[replace_index_self] == out_data[replace_index_self])
    phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
  __syncthreads();
  grad_data[replace_index_grad] = 0;
  if (value_data[replace_index_grad] == out_data[replace_index_self])
    phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
  __syncthreads();
  if (value_data[replace_index_grad] == out_data[replace_index_self])
    grad_data[replace_index_grad] =
        self_data[replace_index_self] /
        static_cast<tensor_t>(aux_buffer[replace_index_self]);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_value_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out,
    const phi::DenseTensor& x,
    const phi::DenseTensor& value,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self,
    const phi::DeviceContext& dev_ctx) {
  const auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  auto index_dims = index.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t select_dim_size = index_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }
  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  constexpr int block = 512;
  int64_t ndim = index_dims.size();
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

  DenseTensor shape_stride_dev;
  shape_stride_dev.Resize({3 * ndim});
  dev_ctx.Alloc<int64_t>(&shape_stride_dev);
  {  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    for (int64_t i = 0; i < ndim; i++) {
      host_data[i] = index_dims[i];
      host_data[i + ndim] = grad.strides()[i];
      host_data[i + (ndim << 1)] = self.strides()[i];
    }
    phi::Copy(dev_ctx,
              shape_stride_host,
              dev_ctx.GetPlace(),
              false,
              &shape_stride_dev);
  }
  const int64_t* shape_strides = shape_stride_dev.data<int64_t>();
  size_t shared_mem_bytes = sizeof(int64_t) * ndim * 3;

  if (reduce == "mul" || reduce == "multiply") {
    ScatterMulValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    self_data,
                                                    value_data,
                                                    out_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel());
  } else if (reduce == "amin" || reduce == "amax") {
    DenseTensor aux_tensor;
    aux_tensor.Resize({self.numel()});
    dev_ctx.Alloc<int>(&aux_tensor);
    phi::funcs::set_constant(dev_ctx, &aux_tensor, 0);

    int* aux_buffer = aux_tensor.data<int>();
    ScatterMinMaxValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    self_data,
                                                    value_data,
                                                    out_data,
                                                    x_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    include_self,
                                                    aux_buffer);
  }
}

Instantiate_Template_Function(gpu_gather_kernel)                  // NOLINT
    Instantiate_Template_Function(gpu_scatter_assign_kernel)      // NOLINT
    Instantiate_Template_Function(gpu_scatter_add_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_mul_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_min_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_max_kernel)         // NOLINT
    Instantiate_Template_Function(gpu_scatter_mean_kernel)        // NOLINT
    Instantiate_Template_Function(gpu_scatter_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function(gpu_scatter_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_mul_min_max_input_grad_kernel)                     // NOLINT
    Instantiate_Template_Function(gpu_scatter_mean_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_add_mean_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        gpu_scatter_mul_min_max_value_grad_kernel)  // NOLINT
}  // namespace funcs
}  // namespace phi
