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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/lookup_table_v2_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename IdT, int BlockDimX, int BlockDimY, int GridDimX,
          bool PaddingFlag>
__global__ void LookupTableV2(T *output, const T *table, const IdT *ids,
                              const int64_t N, const int64_t K, const int64_t D,
                              const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    T *out = output + idy * D;
    const T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<T>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T, typename IdT, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableV2Grad(T *table, const T *output, const IdT *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    auto id = static_cast<int64_t>(ids[idy]);
    const T *out = output + idy * D;
    T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      paddle::platform::CudaAtomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T>
struct LookupTableV2CUDAFunctor {
  LookupTableV2CUDAFunctor(const framework::ExecutionContext &context,
                           const framework::Tensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto *table_t = context_.Input<framework::Tensor>("W");
    auto *output_t = context_.Output<framework::Tensor>("Out");
    int64_t padding_idx = context_.Attr<int64_t>("padding_idx");

    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    size_t K = ids_t_->numel();

    dim3 threads(256, 4);
    dim3 grids(80, 1);

    const auto *table = table_t->template data<T>();
    const auto *ids = ids_t_->template data<IdT>();
    auto *output = output_t->template mutable_data<T>(context_.GetPlace());
    auto stream = context_.cuda_device_context().stream();

    if (padding_idx == -1) {
      LookupTableV2<T, IdT, 256, 4, 80, false><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx);
    } else {
      LookupTableV2<T, IdT, 256, 4, 80, true><<<grids, threads, 0, stream>>>(
          output, table, ids, N, K, D, padding_idx);
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const framework::Tensor *ids_t_;
};

template <typename T>
class LookupTableV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids_t = context.Input<framework::Tensor>("Ids");
    LookupTableV2CUDAFunctor<T> functor(context, ids_t);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids_t->dtype()),
                                functor);
  }
};

template <typename InT, typename OutT>
__global__ void InputTypeConvert(const InT *in_ids, const int64_t K,
                                 OutT *out_ids) {
  for (int i = 0; i < K; i++) {
    out_ids[i] = static_cast<OutT>(in_ids[i]);
  }
}

template <typename T>
struct LookupTableV2GradCUDAFunctor {
  LookupTableV2GradCUDAFunctor(const framework::ExecutionContext &context,
                               const framework::Tensor *ids_t)
      : context_(context), ids_t_(ids_t) {}

  template <typename IdT>
  void apply() {
    auto &dev_ctx =
        context_.template device_context<platform::CUDADeviceContext>();
    bool is_sparse = context_.Attr<bool>("is_sparse");

    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *table = context_.Input<framework::Tensor>("W");
      auto *d_output =
          context_.Input<framework::Tensor>(framework::GradVarName("Out"));
      auto *d_table =
          context_.Output<pten::SelectedRows>(framework::GradVarName("W"));

      const auto *ids_data = ids_t_->template data<IdT>();
      int64_t ids_num = ids_t_->numel();
      dim3 threads(128, 8);
      dim3 grids(8, 1);
      auto stream = dev_ctx.stream();
      framework::Vector<int64_t> new_rows;
      new_rows.resize(ids_num);
      auto gpu_place = context_.GetPlace();

      if (!std::is_same<IdT, int64_t>::value) {
        InputTypeConvert<<<grids, threads, 0, stream>>>(
            ids_data, ids_num, new_rows.MutableData(gpu_place));
      } else {
        memory::Copy(gpu_place, new_rows.CUDAMutableData(gpu_place), gpu_place,
                     ids_data, ids_num * sizeof(int64_t), stream);
      }

      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table->dims()[1]});
      d_table_value->template mutable_data<T>(gpu_place);

      auto *d_table_data = d_table_value->template data<T>();
      auto *d_output_data = d_output->template data<T>();
      auto d_output_dims = d_output->dims();
      auto d_output_dims_2d =
          framework::flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
      PADDLE_ENFORCE_EQ(d_table_value->dims(), d_output_dims_2d,
                        platform::errors::InvalidArgument(
                            "ShapeError: The shape of lookup_table@Grad and "
                            "output@Grad should be same. "
                            "But received lookup_table@Grad's shape = [%s], "
                            "output@Grad's shape = [%s].",
                            d_table_value->dims(), d_output_dims_2d));
      memory::Copy(gpu_place, d_table_data, gpu_place, d_output_data,
                   d_output->numel() * sizeof(T), stream);

    } else {
      auto d_output_t =
          context_.Input<framework::Tensor>(framework::GradVarName("Out"));
      auto d_table_t =
          context_.Output<framework::Tensor>(framework::GradVarName("W"));

      int N = d_table_t->dims()[0];
      int D = d_table_t->dims()[1];
      int K = ids_t_->numel();

      dim3 threads(128, 8);
      dim3 grids(8, 1);
      const T *d_output = d_output_t->template data<T>();
      const auto *ids = ids_t_->template data<IdT>();
      T *d_table = d_table_t->mutable_data<T>(context_.GetPlace());

      auto t = framework::EigenVector<T>::Flatten(*d_table_t);
      t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

      LookupTableV2Grad<T, IdT, 128, 8,
                        8><<<grids, threads, 0, dev_ctx.stream()>>>(
          d_table, d_output, ids, N, K, D);
    }
  }

 private:
  const framework::ExecutionContext &context_;
  const framework::Tensor *ids_t_;
};

template <typename T>
class LookupTableV2GradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *ids_t = context.Input<framework::Tensor>("Ids");
    LookupTableV2GradCUDAFunctor<T> functor(context, ids_t);
    framework::VisitIntDataType(framework::TransToProtoVarType(ids_t->dtype()),
                                functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(lookup_table_v2, ops::LookupTableV2CUDAKernel<float>,
                        ops::LookupTableV2CUDAKernel<double>,
                        ops::LookupTableV2CUDAKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(lookup_table_v2_grad,
                        ops::LookupTableV2GradCUDAKernel<float>,
                        ops::LookupTableV2GradCUDAKernel<double>,
                        ops::LookupTableV2GradCUDAKernel<plat::float16>);
