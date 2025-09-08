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
#include <type_traits>
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

template <typename SrcT, typename DstT>
__global__ void CastMemcpy(const SrcT* __restrict__ src,
                           DstT* __restrict__ dst,
                           int64_t size) {
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;
  dst[tid] = static_cast<DstT>(src[tid]);
}

template <typename T>
static T ExcludeSelfInitialValue(const std::string& reduce_op) {
  if (reduce_op == "add") {
    return static_cast<T>(0);
  } else if (reduce_op == "mul") {
    return static_cast<T>(1);
  } else if (reduce_op == "max") {
    return std::numeric_limits<T>::lowest();
  } else if (reduce_op == "min") {
    return std::numeric_limits<T>::max();
  } else if (reduce_op == "mean") {
    return static_cast<T>(0);
  } else {
    PADDLE_ENFORCE_EQ(
        0,
        1,
        common::errors::InvalidArgument(
            "Unsupported or unnecessary (assign) reduce op: '%s'", reduce_op));
  }
}

template <typename T>
__device__ __forceinline__ T IntFloorDiv(T a, T b) {
  if ((a < 0) != (b < 0)) {
    // compute div and mod at the same time can be optimized by compilers
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
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

#define COMPUTE_OFFSET_SINGLE_OUTPUT(                                     \
    var_name, smem_offset, id_var_name, copy_size)                        \
  extern __shared__ int64_t smem_shape_strides[];                         \
  int64_t id_var_name = threadIdx.x + blockIdx.x * blockDim.x;            \
  if (threadIdx.x < (copy_size * ndim)) {                                 \
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x); \
  }                                                                       \
  __syncthreads();                                                        \
  if (id_var_name >= numel) return;                                       \
  int64_t var_name = 0;                                                   \
  index_t index = index_data[id_var_name];                                \
  const int64_t* stride_info = smem_shape_strides + smem_offset * ndim;   \
  ComputeOffset<false>(smem_shape_strides,                                \
                       stride_info,                                       \
                       nullptr,                                           \
                       &var_name,                                         \
                       nullptr,                                           \
                       id_var_name,                                       \
                       ndim,                                              \
                       dim,                                               \
                       index);

#define COMPUTE_OFFSET_DOUBLE_OUTPUT(                                     \
    var_name1, var_name2, id_var_name, offset1, offset2)                  \
  extern __shared__ int64_t smem_shape_strides[];                         \
  int64_t id_var_name = threadIdx.x + blockIdx.x * blockDim.x;            \
  if (threadIdx.x < (3 * ndim)) {                                         \
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x); \
  }                                                                       \
  __syncthreads();                                                        \
  if (id_var_name >= numel) return;                                       \
  index_t index = index_data[id_var_name];                                \
  const int64_t* grad_strides = smem_shape_strides + offset1 * ndim;      \
  const int64_t* self_strides = smem_shape_strides + offset2 * ndim;      \
  int64_t var_name1 = 0, var_name2 = 0;                                   \
  ComputeOffset<true>(smem_shape_strides,                                 \
                      grad_strides,                                       \
                      self_strides,                                       \
                      &var_name1,                                         \
                      &var_name2,                                         \
                      id_var_name,                                        \
                      ndim,                                               \
                      dim,                                                \
                      index);

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
          bool is_scatter_like = true>
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

  reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
            static_cast<const tensor_t*>(src_data + replace_index_src));
  if (atomic_cnt_buffer) {
    phi::CudaAtomicAdd(atomic_cnt_buffer + replace_index_self, 1);
  }
}

// TODO(heqianyue): to fully match the behavior of PyTorch, we should implement
// a integer div (floor) in this kernel, instead of default trunc (to zero) div
template <typename tensor_t>
__global__ void CastDivKernel(tensor_t* __restrict__ self_data,
                              int* __restrict__ atomic_cnt_buffer,
                              int64_t numel) {
  // mean kernel has only one purpose after refactoring: div by count
  // to fuse the kernel into other kernels (like scatter add), we might need
  // semaphores to notify when all blocks are done adding. By now, we choose
  // this simpler implementation

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (tid >= numel) return;
  if constexpr (std::is_integral_v<std::decay_t<tensor_t>>) {
    self_data[tid] = IntFloorDiv(self_data[tid],
                                 static_cast<tensor_t>(atomic_cnt_buffer[tid]));
  } else {
    self_data[tid] /= static_cast<tensor_t>(atomic_cnt_buffer[tid]);
  }
}

/**
 * Faster pass for scattering a scalar value.
 *
 * For future optimization:
 * TODO(heqianyue): if, for example, the `values` for put_along_axis (and other
 * APIs that use scatter kernels) is a scalar, for broadcast=True mode, the
 * scalar will be made a tensor and broadcast to specific shape, which is
 * wasteful, if actual memory allocation does happen below the hood. We can
 * create a special fast pass based on this kernel, to scatter a single scalar
 * faster, with less memory consumption, since the current kernel eliminates the
 * need for `broadcast_to` and aux_tensor, which might cut the overhead of the
 * kernel by more than half.
 *
 * To upgrade the scalar scatter, one needs to add func_t and reduce_op in the
 * kernel, but be aware that, to be backward-compatible with the behaviors in
 * the old versions, extra atomic primitives might be needed to make sure the
 * correct ordering of stores.
 */
template <typename tensor_t, typename index_t>
__global__ void ScatterAssignScalarValue(
    tensor_t* __restrict__ input_data,
    const index_t* __restrict__ index_data,
    const int64_t* __restrict__ shape_strides,
    int64_t self_select_dim_size,
    tensor_t value_to_scatter,
    int64_t numel,
    int dim,
    int ndim,
    int* aux_buffer = nullptr) {
  extern __shared__ int64_t
      smem_shape_strides[];  // no more than 27 int64_t, won't affect occupancy

  int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (threadIdx.x < (3 * ndim)) {
    *(smem_shape_strides + threadIdx.x) = *(shape_strides + threadIdx.x);
  }
  __syncthreads();
  if (tid >= numel) return;
  index_t index = index_data[tid];
  if (index < 0) index += static_cast<index_t>(self_select_dim_size);

  // some kernels might store input_strides differently! Be careful when dealing
  // with this.
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

  input_data[replace_index_self] = value_to_scatter;
  if (aux_buffer) {
    // fused: used in mean pass, aux_buffer has the same shape as input
    aux_buffer[replace_index_self] = 0;
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

namespace {
template <typename T, typename U>
constexpr bool is_same_type = std::is_same_v<std::decay_t<T>, std::decay_t<U>>;
}  // anonymous namespace

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
    // index matrix might have different shape with self matrix or src matrix.
    int64_t self_select_dim_size = self_dims[dim];
    int64_t src_select_dim_size = src_dims[dim];

    constexpr int block = 512;
    int64_t grid = (index_size + block - 1) / block;
    auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

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
    if (method_name == "assign") {
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
      return;
    }

    // completely eliminate the need for aux_buffer! For most cases we can have
    // up to 50% memory reduction!
    DenseTensor atomic_cnt_tensor;
    int* atomic_cnt_buffer = nullptr;
    if (method_name == "mean") {
      atomic_cnt_tensor.Resize({self_size});
      dev_ctx.Alloc<int>(&atomic_cnt_tensor);
      phi::funcs::set_constant(dev_ctx, &atomic_cnt_tensor, 1);
      atomic_cnt_buffer = atomic_cnt_tensor.data<int>();
    }
    if (!include_self) {
      tensor_t init_val = ExcludeSelfInitialValue<tensor_t>(method_name);
      // exclude self requires us to overwrite the positions that will have
      // values scattered, we cannot fuse the kernels all in one in a simple
      // way, since when shape is large, atomic primitives will only be synced
      // intra-block-ly, resulting in incorrect results, should inter-block
      // atomic reduce occur.
      ScatterAssignScalarValue<<<grid, block, shared_mem_bytes, stream>>>(
          self_data,
          index_data,
          shape_strides,
          self_select_dim_size,
          init_val,
          index_size,
          dim,
          ndim,
          atomic_cnt_buffer);
    }

    GatherScatterGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
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
                                                    atomic_cnt_buffer);
    if (method_name == "mean") {
      constexpr int _block = 512;
      int64_t grid = (self_size + _block - 1) / _block;
      CastDivKernel<<<grid, _block, 0, stream>>>(
          self_data, atomic_cnt_buffer, self_size);
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
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather", tensor_assign, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "assign", tensor_assign, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "add", reduce_add, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "mul", reduce_mul, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "mean", reduce_add, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "max", reduce_max, include_self, dev_ctx);
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
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "min", reduce_min, include_self, dev_ctx);
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

namespace {
enum GradDispatchTag {
  MulInputGrad = 0x0,
  MinMaxInputGrad,
  MeanInputGrad,
  ValueGrad,
  MeanValueGrad,
  MinMaxValueGrad,
};
}  // anonymous namespace

template <typename tensor_t, typename index_t, GradDispatchTag dispatch>
__global__ void ScatterGradPrePassKernel(
    tensor_t* __restrict__ grad_data,
    const index_t* __restrict__ index_data,
    const tensor_t* __restrict__ out_data,
    const tensor_t* __restrict__ value_data,
    const tensor_t* __restrict__ x_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int64_t grad_numel,
    int* __restrict__ aux_buffer,
    bool include_self = true) {
  if constexpr (dispatch == GradDispatchTag::MulInputGrad) {
    COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index, 1, tid, 2)
    atomicMax(aux_buffer + replace_index, tid);
  } else if constexpr (dispatch == GradDispatchTag::MinMaxInputGrad) {
    // This is a special case, src is stored in shape_strides + 2 * dim but used
    // as the 2nd param for compute offset
    COMPUTE_OFFSET_DOUBLE_OUTPUT(replace_index_value, replace_index, tid, 2, 1)
    if (value_data[replace_index_value] == out_data[replace_index])
      phi::CudaAtomicAdd(aux_buffer + replace_index, 1);
  } else if constexpr (dispatch == GradDispatchTag::MeanInputGrad) {
    COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index, 1, tid, 2)
    atomicMax(aux_buffer + replace_index, tid);
    phi::CudaAtomicAdd(aux_buffer + grad_numel + replace_index, 1);
  } else if constexpr (dispatch == GradDispatchTag::ValueGrad) {
    COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index_self, 2, tid, 3)
    atomicMax(aux_buffer + replace_index_self, tid);
  } else if constexpr (dispatch == GradDispatchTag::MeanValueGrad) {
    COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index_self, 2, tid, 3)
    phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
  } else if constexpr (dispatch == GradDispatchTag::MinMaxValueGrad) {
    COMPUTE_OFFSET_DOUBLE_OUTPUT(
        replace_index_grad, replace_index_self, tid, 1, 2)
    grad_data[replace_index_grad] = 0;
    if (include_self &&
        x_data[replace_index_self] == out_data[replace_index_self])
      phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
    if (value_data[replace_index_grad] == out_data[replace_index_self])
      phi::CudaAtomicAdd(aux_buffer + replace_index_self, 1);
  }
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
  COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index, 1, tid, 2)
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
    const tensor_t* __restrict__ self_data,
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    int* __restrict__ aux_buffer) {
  COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index, 1, tid, 2)
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

    ScatterGradPrePassKernel<tensor_t, index_t, MulInputGrad>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    value_data,
                                                    x_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    grad.numel(),
                                                    aux_buffer);
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
    ScatterGradPrePassKernel<tensor_t, index_t, MinMaxInputGrad>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    value_data,
                                                    x_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    grad.numel(),
                                                    aux_buffer);
    ScatterMinMaxInputGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    x_data,
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
  COMPUTE_OFFSET_SINGLE_OUTPUT(replace_index, 1, tid, 2)
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

  ScatterGradPrePassKernel<tensor_t, index_t, MeanInputGrad>
      <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                  index_data,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  shape_strides,
                                                  dim,
                                                  ndim,
                                                  index.numel(),
                                                  grad_size,
                                                  aux_buffer);
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
  COMPUTE_OFFSET_DOUBLE_OUTPUT(
      replace_index_grad, replace_index_self, tid, 1, 2)
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

  ScatterGradPrePassKernel<tensor_t, index_t, ValueGrad>
      <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                  index_data,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  shape_strides,
                                                  dim,
                                                  ndim,
                                                  index.numel(),
                                                  grad.numel(),
                                                  aux_buffer);
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
  COMPUTE_OFFSET_DOUBLE_OUTPUT(
      replace_index_grad, replace_index_self, tid, 1, 2)
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
  COMPUTE_OFFSET_DOUBLE_OUTPUT(
      replace_index_grad, replace_index_self, tid, 1, 2)
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
    ScatterGradPrePassKernel<tensor_t, index_t, MeanValueGrad>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    grad.numel(),
                                                    aux_buffer);
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
  COMPUTE_OFFSET_DOUBLE_OUTPUT(
      replace_index_grad, replace_index_self, tid, 1, 2)
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
    const int64_t* __restrict__ shape_strides,
    int dim,
    int ndim,
    int64_t numel,
    bool include_self,
    int* __restrict__ aux_buffer) {
  COMPUTE_OFFSET_DOUBLE_OUTPUT(
      replace_index_grad, replace_index_self, tid, 1, 2)
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
    ScatterGradPrePassKernel<tensor_t, index_t, MinMaxValueGrad>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    out_data,
                                                    value_data,
                                                    x_data,
                                                    shape_strides,
                                                    dim,
                                                    ndim,
                                                    index.numel(),
                                                    grad.numel(),
                                                    aux_buffer,
                                                    include_self);
    ScatterMinMaxValueGradGPUKernel<tensor_t, index_t>
        <<<grid, block, shared_mem_bytes, stream>>>(grad_data,
                                                    index_data,
                                                    self_data,
                                                    value_data,
                                                    out_data,
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
