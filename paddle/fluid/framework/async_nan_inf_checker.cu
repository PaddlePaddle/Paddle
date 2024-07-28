// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/async_nan_inf_checker.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace paddle {
namespace framework {

template <typename T>
struct AsyncCheckNaNInfFunctor {
  HOSTDEVICE void operator()(T x) const {
    if (isnan(x) || isinf(x)) {
      asm("trap;");
    }
  }
};

template <typename T, int VecSize>
__global__ void AsyncCheckNaNInfCUDAKernel(const T *x, int64_t numel) {
  int64_t idx =
      (static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x) * VecSize;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VecSize;
  int64_t limit = numel - VecSize;

  phi::AlignedVector<T, VecSize> vec;

  AsyncCheckNaNInfFunctor<T> functor;
  while (idx <= limit) {
    phi::Load(x + idx, &vec);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      functor(vec[i]);
    }
    idx += stride;
  }

  for (; idx < numel; ++idx) {
    functor(x[idx]);
  }
}

AsyncNaNInfChecker::AsyncNaNInfChecker(phi::GPUPlace place) : place_(place) {}

AsyncNaNInfChecker::~AsyncNaNInfChecker() {
  ctx_.reset();
  if (event_ != nullptr) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event_));
  }
}

template <typename T>
bool AsyncNaNInfChecker::CheckImpl(const phi::DenseTensor &x) {
  int64_t numel = x.numel();
  if (numel <= 0) {
    return false;
  }

  PADDLE_ENFORCE_EQ(
      x.meta().is_contiguous(),
      true,
      phi::errors::InvalidArgument(
          "Only support to check NaN-Inf when tensor is contiguous."));

  auto place = x.place();
  PADDLE_ENFORCE_EQ(
      place,
      phi::Place(place_),
      phi::errors::InvalidArgument("Place not matched when checking NaN-Inf."));

  const auto &default_ctx = *static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));

  auto default_stream = default_ctx.stream();

  if (ctx_ == nullptr) {
    ctx_.reset(new phi::GPUContext(place_, true, 0));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  auto stream = ctx_->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_, default_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream, event_));

  const T *x_ptr = x.template data<T>();
  int vec_size = phi::GetVectorizedSize(x_ptr);
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(*ctx_, numel, vec_size);

#define LAUNCH_ASYNC_CHECK_NAN_INF_KERNEL(__vec_size)                    \
  case __vec_size: {                                                     \
    AsyncCheckNaNInfCUDAKernel<T, __vec_size>                            \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            x_ptr, numel);                                               \
    break;                                                               \
  }

  switch (vec_size) {
    LAUNCH_ASYNC_CHECK_NAN_INF_KERNEL(VecSizeL);
    LAUNCH_ASYNC_CHECK_NAN_INF_KERNEL(VecSizeM);
    LAUNCH_ASYNC_CHECK_NAN_INF_KERNEL(VecSizeS);
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
  }

#undef LAUNCH_ASYNC_CHECK_NAN_INF_KERNEL

  ctx_->AddStreamCallback([x] {
    VLOG(10) << "Release Tensor with Ptr[0x" << x.data() << "] , Shape["
             << x.dims() << "] , Dtype[" << x.dtype() << "]";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaPeekAtLastError());
    const_cast<phi::DenseTensor &>(x).clear();
  });

  return true;
}

bool AsyncNaNInfChecker::Check(const phi::DenseTensor &x) {
  if (!x.initialized()) {
    return false;
  }

  auto dtype = x.dtype();

#define CALL_NAN_INF_CHECKER_IMPL(__dtype, __cpp_dtype) \
  do {                                                  \
    if (dtype == phi::DataType::__dtype) {              \
      return CheckImpl<__cpp_dtype>(x);                 \
    }                                                   \
  } while (0)

  CALL_NAN_INF_CHECKER_IMPL(BFLOAT16, phi::dtype::bfloat16);
  CALL_NAN_INF_CHECKER_IMPL(FLOAT16, phi::dtype::float16);
  CALL_NAN_INF_CHECKER_IMPL(FLOAT32, float);
  CALL_NAN_INF_CHECKER_IMPL(FLOAT64, double);

#undef CALL_NAN_INF_CHECKER_IMPL

  return false;
}

bool AsyncNaNInfChecker::Check(const paddle::Tensor &x) {
  if (!x.defined()) {
    return false;
  }

  if (x.is_dist_tensor()) {
    auto *dt = dynamic_cast<phi::distributed::DistTensor *>(x.impl().get());
    return dt ? Check(dt->value()) : false;
  } else {
    auto *dt = dynamic_cast<phi::DenseTensor *>(x.impl().get());
    return dt ? Check(*dt) : false;
  }
}

void AsyncNaNInfChecker::Wait() {
  if (ctx_) {
    ctx_->Wait();
  }
}

}  // namespace framework
}  // namespace paddle

#endif
