/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fusion_transpose_flatten_concat_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class TransposeFlattenConcatFusionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    auto odims = out->dims();

    std::vector<int> trans_axis = ctx.Attr<std::vector<int>>("trans_axis");
    int flatten_axis = ctx.Attr<int>("flatten_axis");
    int concat_axis = ctx.Attr<int>("concat_axis");

    int rank = ins[0]->dims().size();
    // use at least 4D in cudnnTransformTensor
    int max_dim = rank < 4 ? 4 : rank;
    std::vector<int> stride_x(max_dim, 0);
    std::vector<int> stride_y(max_dim, 0);
    std::vector<int> dims_y(max_dim, 0);

    cudnnTensorDescriptor_t in_desc;
    cudnnTensorDescriptor_t out_desc;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&in_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&out_desc));
    cudnnDataType_t cudnn_dtype = CudnnDataType<T>::type;

    auto handle = dev_ctx.cudnn_handle();

    T* odata = out->data<T>();
    for (size_t k = 0; k < ins.size(); ++k) {
      auto perm_shape = GetPermuteShape(trans_axis, ins[k]->dims());
      int osize = 1;
      auto idims = ins[k]->dims();
      for (int i = 0; i < rank; i++) {
        stride_x[i] = 1;
        for (int j = trans_axis[i] + 1; j < rank; j++) {
          stride_x[i] *= idims[j];
        }
        dims_y[i] = perm_shape[i];
        osize *= perm_shape[i];
      }
      stride_y[rank - 1] = 1;
      for (int i = rank - 2; i >= 0; i--) {
        if (((i + 1) == flatten_axis) && (concat_axis == 1)) {
          stride_y[i] = odims[1];
        } else {
          stride_y[i] = stride_y[i + 1] * perm_shape[i + 1];
        }
      }

      // Since concat is after flatten, the output is 2D tensor.
      // If concat_axis is 0, each input's permutated tensor is continuous.
      // If concat_axis is 1, the stride of 0-th dim of each input's
      // permutated tensor is odims()[1].

      for (int i = rank; i < max_dim; i++) {
        stride_x[i] = 1;
        stride_y[i] = 1;
        dims_y[i] = 1;
      }

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          in_desc, cudnn_dtype, max_dim, dims_y.data(), stride_x.data()));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          out_desc, cudnn_dtype, max_dim, dims_y.data(), stride_y.data()));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnTransformTensor(
          handle,
          CudnnDataType<T>::kOne(),
          in_desc,
          static_cast<const void*>(ins[k]->data<T>()),
          CudnnDataType<T>::kZero(),
          out_desc,
          static_cast<void*>(odata)));
      if (concat_axis == 0) {
        odata += osize;
      } else {
        auto flat_shape = GetFlattenShape(flatten_axis, perm_shape);
        odata += flat_shape[1];
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(in_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(out_desc));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fusion_transpose_flatten_concat,
                        ops::TransposeFlattenConcatFusionKernel<float>,
                        ops::TransposeFlattenConcatFusionKernel<double>);
