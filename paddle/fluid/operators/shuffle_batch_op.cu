// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifndef _MSC_VER
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#endif

#include "paddle/fluid/operators/shuffle_batch_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

struct CacheAllocator {
  typedef char value_type;
  explicit CacheAllocator(platform::Place place) {
    VLOG(2) << "construct allocator";
    place_ = place;
  }

  ~CacheAllocator() { VLOG(2) << "destory allocator"; }

  char *allocate(std::ptrdiff_t num_bytes) {
    VLOG(2) << "allocate " << num_bytes << " bytes";
    auto storage = memory::AllocShared(place_, num_bytes);
    char *ptr = reinterpret_cast<char *>(storage->ptr());
    busy_allocation_.emplace(std::make_pair(ptr, storage));
    return ptr;
  }

  void deallocate(char *ptr, size_t) {
    VLOG(2) << "deallocate ";
    allocation_map_type::iterator iter = busy_allocation_.find(ptr);
    CHECK(iter != busy_allocation_.end());
    busy_allocation_.erase(iter);
  }

 private:
  typedef std::unordered_map<char *, std::shared_ptr<phi::Allocation>>
      allocation_map_type;
  allocation_map_type busy_allocation_;
  platform::Place place_;
};

template <typename T, bool kIsForward>
struct ReorderFunctor {
  ReorderFunctor(const T *x, const int64_t *shuffle_idx, T *y, int64_t stride)
      : x_(x), shuffle_idx_(shuffle_idx), y_(y), stride_(stride) {}

  HOSTDEVICE void operator()(int64_t idx) {
    auto reorder_idx = shuffle_idx_[idx / stride_] * stride_ + idx % stride_;
    if (kIsForward) {
      y_[idx] = x_[reorder_idx];
    } else {
      y_[reorder_idx] = x_[idx];
    }
  }

 private:
  const T *x_;
  const int64_t *shuffle_idx_;
  T *y_;
  int64_t stride_;
};

template <typename T, typename DeviceContext>
class ShuffleBatchGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef _MSC_VER
    PADDLE_THROW(platform::errors::Unimplemented(
        "GPU shuffle_batch_grad is not supported on Windows yet"));
#else
    const auto *out_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    const auto *shuffleidx = ctx.Input<phi::DenseTensor>("ShuffleIdx");
    auto *x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    const auto *out_grad_data = out_grad->data<T>();
    const auto *shuffleidx_data = shuffleidx->data<int64_t>();
    auto *x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());
    auto x_embed_size = x_grad->dims()[x_grad->dims().size() - 1];
    ReorderFunctor<T, false> functor(
        out_grad_data, shuffleidx_data, x_grad_data, x_embed_size);
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    // TODO(zengjinle): for small data, direct cudaMemcpy may be better
    platform::ForRange<phi::GPUContext> for_range(dev_ctx, x_grad->numel());
    for_range(functor);
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
PD_REGISTER_STRUCT_KERNEL(shuffle_batch_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::ShuffleBatchGradCUDAKernel,
                          float,
                          double,
                          int32_t,
                          int64_t) {}
#endif
