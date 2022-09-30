/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace paddle {
namespace operators {

namespace {

cnnlPoolingMode_t ToCnnlPoolingMode(const std::string &pooling_type,
                                    bool exclusive,
                                    bool adaptive) {
  cnnlPoolingMode_t pooling_mode;
  if (pooling_type == "max") {
    pooling_mode = CNNL_POOLING_MAX;
  } else if (pooling_type == "avg") {
    if (exclusive && !adaptive) {
      pooling_mode = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
      pooling_mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("Unknown pooling_type: %s",
                                                   pooling_type));
  }
  return pooling_mode;
}
}  // namespace

template <typename T>
class MLUPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    const Tensor *in_x = ctx.Input<phi::DenseTensor>("X");
    Tensor *out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");

    bool global_pooling = ctx.Attr<bool>("global_pooling");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    PADDLE_ENFORCE_EQ(in_x->dims().size(),
                      4,
                      platform::errors::InvalidArgument(
                          "Only support 4-dims for mlu pool2d kernel."));

    const bool channel_last = data_format == "NHWC";
    // default
    cnnlTensorLayout_t cnnl_layout = CNNL_LAYOUT_NCHW;
    auto out_dims = out->dims();
    int64_t out_h = out_dims[2];
    int64_t out_w = out_dims[3];
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());

    if (channel_last) {
      cnnl_layout = CNNL_LAYOUT_NHWC;
      out_h = out_dims[1];
      out_w = out_dims[2];
      data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    }

    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);
    if (global_pooling) {
      phi::funcs::UpdateKernelSize(&ksize, data_dims);
    }

    MLUCnnlTensorDesc in_x_desc(*in_x, cnnl_layout, ToCnnlDataType<T>());
    MLUCnnlTensorDesc out_desc(*out, cnnl_layout, ToCnnlDataType<T>());

    cnnlPoolingMode_t pool_mode =
        ToCnnlPoolingMode(pooling_type, exclusive, adaptive);

    // transpose NCHW to NHWC since cnnl pool2d has worse performance in that
    // layout.
    phi::DenseTensor trans_in_x;
    phi::DenseTensor trans_out;
    if (channel_last) {
      trans_in_x = *in_x;
      trans_out = *out;
    } else {
      std::vector<int> perm{0, 2, 3, 1};
      TransposeFromMLUTensor<T>(
          ctx, perm, in_x, &trans_in_x, true /*need_reshape_or_alloc*/);
      trans_out = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          {out_dims[0], out_dims[2], out_dims[3], out_dims[1]}, dev_ctx);
    }
    MLUCnnlTensorDesc trans_in_x_desc(
        trans_in_x, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc trans_out_desc(
        trans_out, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());

    if (!adaptive) {
      MLUCnnlPoolingDesc pool_desc(pool_mode,
                                   CNNL_NOT_PROPAGATE_NAN,
                                   ksize[0],
                                   ksize[1],
                                   paddings[0],
                                   paddings[1],
                                   paddings[2],
                                   paddings[3],
                                   strides[0],
                                   strides[1],
                                   1 /*row_dilation*/,
                                   1 /*col_dilation*/,
                                   ceil_mode);

      size_t extra_input_size = 0;
      cnnlHandle_t handle =
          ctx.template device_context<MLUDeviceContext>().cnnl_handle();
      cnnlGetPoolingExtraInputSize(
          handle, pool_mode, out_w, out_h, &extra_input_size);

      if (extra_input_size > 0) {
        phi::DenseTensor extra_host_tensor;
        extra_host_tensor.mutable_data<int8_t>(
            {static_cast<int64_t>(extra_input_size)}, platform::CPUPlace());
        cnnlInitPoolingExtraInput(handle,
                                  pool_desc.get(),
                                  trans_in_x_desc.get(),
                                  trans_out_desc.get(),
                                  GetBasePtr(&extra_host_tensor));
        phi::DenseTensor extra_device_tensor =
            ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
                {static_cast<int64_t>(extra_input_size)}, dev_ctx);
        framework::TensorCopy(
            extra_host_tensor, ctx.GetPlace(), &extra_device_tensor);
        // Increase extra_host_tensor holder_ reference count until copy
        // complete.
        auto increase_ref_count = [extra_host_tensor]() {
          VLOG(4) << "Finished copying extra_host_tensor["
                  << GetBasePtr(&extra_host_tensor)
                  << "] in mlu pooling kernel.";
        };
        dev_ctx.AddStreamCallback(increase_ref_count);
        MLUCnnl::PoolingForward(
            ctx,
            pool_mode,
            out_h,
            out_w,
            pool_desc.get(),
            nullptr /*alpha*/,
            trans_in_x_desc.get(),
            GetBasePtr(&trans_in_x),
            nullptr /*beta*/,
            GetBasePtr(&extra_device_tensor) /*params_shape_ptr*/,
            trans_out_desc.get(),
            GetBasePtr(&trans_out));
      } else {
        MLUCnnl::PoolingForward(ctx,
                                pool_mode,
                                out_h,
                                out_w,
                                pool_desc.get(),
                                nullptr /*alpha*/,
                                trans_in_x_desc.get(),
                                GetBasePtr(&trans_in_x),
                                nullptr /*beta*/,
                                nullptr /*params_shape_ptr*/,
                                trans_out_desc.get(),
                                GetBasePtr(&trans_out));
      }
    } else {
      MLUCnnl::AdaptivePoolingForward(ctx,
                                      pool_mode,
                                      trans_in_x_desc.get(),
                                      GetBasePtr(&trans_in_x),
                                      trans_out_desc.get(),
                                      GetBasePtr(&trans_out),
                                      nullptr,
                                      nullptr);
    }
    if (!channel_last) {
      std::vector<int> perm{0, 3, 1, 2};
      TransposeFromMLUTensor<T>(
          ctx, perm, &trans_out, out, false /*need_reshape_or_alloc*/);
    }
  }
};

template <typename T, typename IDX_T>
class MLUPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    const Tensor *in_x = ctx.Input<phi::DenseTensor>("X");
    const Tensor *out = ctx.Input<phi::DenseTensor>("Out");
    const Tensor *out_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    Tensor *in_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    in_x_grad->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    auto in_x_dims = in_x->dims();
    framework::DDim data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    if (channel_last) {
      data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    }

    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);
    if (global_pooling) {
      phi::funcs::UpdateKernelSize(&ksize, data_dims);
    }

    // inputs need with NHWC layout
    phi::DenseTensor trans_in_x;
    phi::DenseTensor trans_out;
    phi::DenseTensor trans_out_grad;
    phi::DenseTensor trans_in_x_grad;
    if (channel_last) {
      trans_in_x = *in_x;
      trans_out = *out;
      trans_out_grad = *out_grad;
      trans_in_x_grad = *in_x_grad;
    } else {
      std::vector<int> perm{0, 2, 3, 1};
      TransposeFromMLUTensor<T>(
          ctx, perm, in_x, &trans_in_x, true /*need_reshape_or_alloc*/);
      TransposeFromMLUTensor<T>(
          ctx, perm, out, &trans_out, true /*need_reshape_or_alloc*/);
      TransposeFromMLUTensor<T>(
          ctx, perm, out_grad, &trans_out_grad, true /*need_reshape_or_alloc*/);
      auto in_x_grad_dims = in_x_grad->dims();
      trans_in_x_grad =
          ctx.AllocateTmpTensor<T, MLUDeviceContext>({in_x_grad_dims[0],
                                                      in_x_grad_dims[2],
                                                      in_x_grad_dims[3],
                                                      in_x_grad_dims[1]},
                                                     dev_ctx);
    }
    MLUCnnlTensorDesc trans_in_x_desc(
        trans_in_x, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc trans_out_desc(
        trans_out, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc trans_out_grad_desc(
        trans_out_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
    MLUCnnlTensorDesc trans_in_x_grad_desc(
        trans_in_x_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());

    cnnlPoolingMode_t pool_mode =
        ToCnnlPoolingMode(pooling_type, exclusive, adaptive);
    MLUCnnlPoolingDesc pool_desc(pool_mode,
                                 CNNL_NOT_PROPAGATE_NAN,
                                 ksize[0],
                                 ksize[1],
                                 paddings[0],
                                 paddings[1],
                                 paddings[2],
                                 paddings[3],
                                 strides[0],
                                 strides[1],
                                 1 /*row_dilation*/,
                                 1 /*col_dilation*/,
                                 ceil_mode);

    if (pooling_type == "max") {
      phi::DenseTensor index_tensor =
          ctx.AllocateTmpTensor<IDX_T, MLUDeviceContext>(trans_out_grad.dims(),
                                                         dev_ctx);
      MLUCnnlTensorDesc index_tensor_desc(
          index_tensor, CNNL_LAYOUT_NHWC, ToCnnlDataType<IDX_T>());
      MLUCnnl::PoolingIndex(ctx,
                            pool_desc.get(),
                            trans_in_x_desc.get(),
                            GetBasePtr(&trans_in_x),
                            index_tensor_desc.get(),
                            GetBasePtr(&index_tensor));
      if (adaptive) {
        MLUCnnl::AdaptivePoolingBackward(ctx,
                                         pool_mode,
                                         trans_out_grad_desc.get(),
                                         GetBasePtr(&trans_out_grad),
                                         index_tensor_desc.get(),
                                         GetBasePtr(&index_tensor),
                                         trans_in_x_grad_desc.get(),
                                         GetBasePtr(&trans_in_x_grad));
      } else {
        MLUCnnl::PoolingBackward(ctx,
                                 pool_desc.get(),
                                 nullptr /*alpha*/,
                                 index_tensor_desc.get(),
                                 GetBasePtr(&index_tensor),
                                 trans_out_grad_desc.get(),
                                 GetBasePtr(&trans_out_grad),
                                 trans_in_x_desc.get(),
                                 GetBasePtr(&trans_in_x),
                                 nullptr /*beta*/,
                                 trans_in_x_grad_desc.get(),
                                 GetBasePtr(&trans_in_x_grad));
      }
    } else {
      if (adaptive) {
        MLUCnnl::AdaptivePoolingBackward(ctx,
                                         pool_mode,
                                         trans_out_grad_desc.get(),
                                         GetBasePtr(&trans_out_grad),
                                         nullptr /*index_tensor_desc.get()*/,
                                         nullptr /*GetBasePtr(&index_tensor)*/,
                                         trans_in_x_grad_desc.get(),
                                         GetBasePtr(&trans_in_x_grad));
      } else {
        MLUCnnl::PoolingBackward(ctx,
                                 pool_desc.get(),
                                 nullptr /*alpha*/,
                                 nullptr,
                                 nullptr,
                                 trans_out_grad_desc.get(),
                                 GetBasePtr(&trans_out_grad),
                                 nullptr,
                                 nullptr,
                                 nullptr /*beta*/,
                                 trans_in_x_grad_desc.get(),
                                 GetBasePtr(&trans_in_x_grad));
      }
    }
    if (!channel_last) {
      std::vector<int> perm{0, 3, 1, 2};
      TransposeFromMLUTensor<T>(ctx,
                                perm,
                                &trans_in_x_grad,
                                in_x_grad,
                                false /*need_reshape_or_alloc*/);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(pool2d,
                       ops::MLUPoolOpKernel<float>,
                       ops::MLUPoolOpKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(pool2d_grad,
                       ops::MLUPoolGradOpKernel<float, int>,
                       ops::MLUPoolGradOpKernel<plat::float16, int16_t>);
