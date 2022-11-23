/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lookup_table_v2_op.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename IdT, bool PaddingFlag>
__global__ void LookupTableV2(T *output,
                              const T *table,
                              const IdT *ids,
                              const int64_t N,
                              const int64_t K,
                              const int64_t D,
                              const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    T *out = output + idy * D;
    const T *tab = table + id * D;
    for (int i = idx; i < D; i += blockDim.x) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<T>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += blockDim.y * gridDim.x;
  }
}

template <typename T, typename IdT>
__global__ void LookupTableV2Grad(T *table,
                                  const T *output,
                                  const IdT *ids,
                                  const int64_t N,
                                  const int64_t K,
                                  const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    const T *out = output + idy * D;
    T *tab = table + id * D;
#ifdef PADDLE_WITH_CUDA
    paddle::platform::VectorizedAtomicAddPerBlock(D, idx, blockDim.x, out, tab);
#else
    for (int i = idx; i < D; i += blockDim.x) {
      paddle::platform::CudaAtomicAdd(&tab[i], out[i]);
    }
#endif
    idy += blockDim.y * gridDim.x;
  }
}

template <typename T>
struct LookupTableV2CUDAFunctor {
  LookupTableV2CUDAFunctor(const framework::ExecutionContext &context,
                           const phi::DenseTensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto *table_t = context_.Input<phi::DenseTensor>("W");
    auto *output_t = context_.Output<phi::DenseTensor>("Out");
    int64_t padding_idx = context_.Attr<int64_t>("padding_idx");

    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    size_t K = ids_t_->numel();

    const int gridx = 2 * context_.cuda_device_context().GetSMCount();
    dim3 threads(256, 4);
    dim3 grids(gridx, 1);

    const auto *table = table_t->template data<T>();
    const auto *ids = ids_t_->template data<IdT>();
    auto *output = output_t->template mutable_data<T>(context_.GetPlace());
    auto stream = context_.cuda_device_context().stream();

    if (padding_idx == -1) {
      LookupTableV2<T, IdT, false><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx);
    } else {
      LookupTableV2<T, IdT, true><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx);
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const phi::DenseTensor *ids_t_;
};

template <typename T>
class LookupTableV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids_t = context.Input<phi::DenseTensor>("Ids");
    LookupTableV2CUDAFunctor<T> functor(context, ids_t);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids_t->dtype()),
                                functor);
  }
};

template <typename InT, typename OutT>
__global__ void InputTypeConvert(const InT *in_ids,
                                 const int64_t K,
                                 OutT *out_ids) {
  for (int i = 0; i < K; i++) {
    out_ids[i] = static_cast<OutT>(in_ids[i]);
  }
}

template <typename T>
struct LookupTableV2GradCUDAFunctor {
  LookupTableV2GradCUDAFunctor(const framework::ExecutionContext &context,
                               const phi::DenseTensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto &dev_ctx = context_.template device_context<phi::GPUContext>();
    bool is_sparse = context_.Attr<bool>("is_sparse");

    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *table = context_.Input<phi::DenseTensor>("W");
      auto *d_output =
          context_.Input<phi::DenseTensor>(framework::GradVarName("Out"));
      auto *d_table =
          context_.Output<phi::SelectedRows>(framework::GradVarName("W"));

      const auto *ids_data = ids_t_->template data<IdT>();
      int64_t ids_num = ids_t_->numel();
      dim3 threads(128, 8);
      dim3 grids(8, 1);
      auto stream = dev_ctx.stream();
      framework::Vector<int64_t> new_rows;
      new_rows.resize(ids_num);
      auto gpu_place = context_.GetPlace();

      paddle::framework::MixVector<int64_t> mixv_new_rows(&new_rows);
      if (!std::is_same<IdT, int64_t>::value) {
        InputTypeConvert<<<grids, threads, 0, stream>>>(
            ids_data, ids_num, mixv_new_rows.MutableData(gpu_place));
      } else {
        memory::Copy(gpu_place,
                     mixv_new_rows.CUDAMutableData(gpu_place),
                     gpu_place,
                     ids_data,
                     ids_num * sizeof(int64_t),
                     stream);
      }

      mixv_new_rows.CopyToCPU();
      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table->dims()[1]});
      d_table_value->template mutable_data<T>(gpu_place);

      auto *d_table_data = d_table_value->template data<T>();
      auto *d_output_data = d_output->template data<T>();
      auto d_output_dims = d_output->dims();
      auto d_output_dims_2d =
          phi::flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
      PADDLE_ENFORCE_EQ(d_table_value->dims(),
                        d_output_dims_2d,
                        platform::errors::InvalidArgument(
                            "ShapeError: The shape of lookup_table@Grad and "
                            "output@Grad should be same. "
                            "But received lookup_table@Grad's shape = [%s], "
                            "output@Grad's shape = [%s].",
                            d_table_value->dims(),
                            d_output_dims_2d));
      memory::Copy(gpu_place,
                   d_table_data,
                   gpu_place,
                   d_output_data,
                   d_output->numel() * sizeof(T),
                   stream);

    } else {
      auto d_output_t =
          context_.Input<phi::DenseTensor>(framework::GradVarName("Out"));
      auto d_table_t =
          context_.Output<phi::DenseTensor>(framework::GradVarName("W"));

      int N = d_table_t->dims()[0];
      int D = d_table_t->dims()[1];
      int K = ids_t_->numel();

      const T *d_output = d_output_t->template data<T>();
      const auto *ids = ids_t_->template data<IdT>();
      T *d_table = d_table_t->mutable_data<T>(context_.GetPlace());

#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(d_table, 0, N * D * sizeof(T), dev_ctx.stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(d_table, 0, N * D * sizeof(T), dev_ctx.stream()));
#endif

      const int gridx = 2 * dev_ctx.GetSMCount();
      dim3 threads(128, 8);
      dim3 grids(gridx, 1);
      LookupTableV2Grad<T, IdT><<<grids, threads, 0, dev_ctx.stream()>>>(
          d_table, d_output, ids, N, K, D);
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const phi::DenseTensor *ids_t_;
};

template <typename T>
class LookupTableV2GradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids_t = context.Input<phi::DenseTensor>("Ids");
    LookupTableV2GradCUDAFunctor<T> functor(context, ids_t);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids_t->dtype()),
                                functor);
  }
};

}  // namespace operators
}  // namespace paddle
