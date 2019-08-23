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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(cudnn_deterministic, false,
            "Whether allow using an autotuning algorithm for convolution "
            "operator. The autotuning algorithm may be non-deterministic. If "
            "true, the algorithm is deterministic.");
DEFINE_uint64(conv_workspace_size_limit,
              paddle::platform::kDefaultConvWorkspaceSizeLimitMB,
              "cuDNN convolution workspace limit in MB unit.");
DEFINE_bool(cudnn_exhaustive_search, false,
            "Whether enable exhaustive search for cuDNN convolution or "
            "not, default is False.");

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;
using framework::AlgorithmsCache;

static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout, int* N, int* C, int* D,
                            int* H, int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

template <typename T>
class CUDNNConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");

    if (exhaustive_search && FLAGS_cudnn_deterministic) {
      PADDLE_THROW(
          "Cann't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    // ------------------- cudnn descriptors ---------------------
    ConvArgs args{input, filter, output, strides, paddings, dilations};
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    auto dtype = platform::CudnnDataType<T>::type;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }
    auto layout_format = GetCudnnTensorFormat(layout);

    args.handle = handle;
    args.cdesc.set(dtype, paddings, strides, dilations);
#if CUDNN_VERSION_MIN(7, 0, 1)
    // cudnn 7 can support groups, no need to do it manually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionGroupCount(
        args.cdesc.desc(), groups));
    groups = 1;
#endif
    args.idesc.set(*input, groups);
    args.wdesc.set(*filter, layout_format, groups);
    args.odesc.set(*output, groups);
    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(input->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(output->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size = 0;  // final workspace to allocate.
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo{};

    using search = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
    algo = search::Find<T>(args, exhaustive_search, false, 0, ctx);
    workspace_size = search::GetWorkspaceSize(args, algo);

    // ------------------- cudnn conv forward ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                handle, &alpha, args.idesc.desc(),
                input_data + i * group_offset_in, args.wdesc.desc(),
                filter_data + i * group_offset_filter, args.cdesc.desc(), algo,
                workspace_ptr, workspace_size, &beta, args.odesc.desc(),
                output_data + i * group_offset_out));
          },
          workspace_size);
    }
  }
};

template <typename T>
class CUDNNConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    if (exhaustive_search && deterministic) {
      PADDLE_THROW(
          "Can't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }

    T* filter_grad_data = nullptr;
    T* input_grad_data = nullptr;
    ConvArgs args1{input_grad, filter,   output_grad,
                   strides,    paddings, dilations};
    ConvArgs args2{input,   filter_grad, output_grad,
                   strides, paddings,    dilations};
    // conv_cudnn_helper.h
    auto handle = dev_ctx.cudnn_handle();
    auto dtype = platform::CudnnDataType<T>::type;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }
    auto layout_tensor = GetCudnnTensorFormat(layout);
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(input->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(output_grad->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h,
             &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
    size_t workspace_size = 0;
    int iwo_groups, c_groups;

#if CUDNN_VERSION_MIN(7, 0, 1)
    iwo_groups = 1;
    c_groups = groups;
    groups = 1;
#endif

    if (input_grad) {
      // ------------------- cudnn descriptors ---------------------
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      args1.handle = handle;
      args1.idesc.set(*input_grad, iwo_groups);
      args1.wdesc.set(*filter, layout_tensor, iwo_groups);
      args1.odesc.set(*output_grad, iwo_groups);
      args1.cdesc.set(dtype, paddings, strides, dilations, c_groups);

      using search1 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search1::Find<T>(args1, exhaustive_search, deterministic, 0, ctx);
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1, data_algo));
    }

    if (filter_grad) {
      // ------------------- cudnn descriptors ---------------------
      filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      args2.handle = handle;
      args2.idesc.set(*input, iwo_groups);
      args2.wdesc.set(*filter_grad, layout_tensor, iwo_groups);
      args2.odesc.set(*output_grad, iwo_groups);
      args2.cdesc.set(dtype, paddings, strides, dilations, c_groups);

      using search2 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search2::Find<T>(args2, exhaustive_search, deterministic, 1, ctx);
      workspace_size = std::max(workspace_size,
                                search2::GetWorkspaceSize(args2, filter_algo));
    }

    // ------------------- cudnn conv backward data ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, args1.wdesc.desc(),
                  filter_data + i * group_offset_filter, args1.odesc.desc(),
                  output_grad_data + i * group_offset_out, args1.cdesc.desc(),
                  data_algo, cudnn_workspace_ptr, workspace_size, &beta,
                  args1.idesc.desc(), input_grad_data + i * group_offset_in));
            },
            workspace_size);
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      // Because beta is zero, it is unnecessary to reset filter_grad.
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args2.idesc.desc(),
                  input_data + i * group_offset_in, args2.odesc.desc(),
                  output_grad_data + i * group_offset_out, args2.cdesc.desc(),
                  filter_algo, cudnn_workspace_ptr, workspace_size, &beta,
                  args2.wdesc.desc(),
                  filter_grad_data + i * group_offset_filter));
            },
            workspace_size);
      }
    }
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 * ddo = conv(ddI, W) + conv(I, ddW)
 * dW = conv_bp_filter(ddI, dO)
 * dI = conv_bp_data(ddW, dO)
 */
template <typename T>
class CUDNNConvDoubleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto X = ctx.Input<Tensor>("Input");
    auto W = ctx.Input<Tensor>("Filter");
    auto dO = ctx.Input<Tensor>("DOutput");
    auto ddX = ctx.Input<Tensor>("DDInput");
    auto ddW = ctx.Input<Tensor>("DDFilter");

    auto ddO = ctx.Output<Tensor>("DDOutput");
    auto dW = ctx.Output<Tensor>("DFilter");
    auto dX = ctx.Output<Tensor>("DInput");

    const T* x = X->data<T>();
    const T* dy = dO->data<T>();
    const T* w = W->data<T>();

    const T* ddx = nullptr;
    const T* ddw = nullptr;
    T *dw, *dx, *ddy;
    dw = dx = ddy = nullptr;

    const std::vector<int>& strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int>& paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int>& dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    if (exhaustive_search && deterministic) {
      PADDLE_THROW(
          "Can't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }

    int iwo_group = groups;
    int c_group = 1;
#if CUDNN_VERSION_MIN(7, 0, 1)
    iwo_group = 1;
    c_group = groups;
#endif
    auto dtype = platform::CudnnDataType<T>::type;

    auto handle = dev_ctx.cudnn_handle();

    ConvArgs args1{ddX, W, ddO, strides, paddings, dilations};
    ConvArgs args2{X, ddW, ddO, strides, paddings, dilations};
    ConvArgs args3{ddX, dW, dO, strides, paddings, dilations};
    ConvArgs args4{dX, ddW, dO, strides, paddings, dilations};

    cudnnConvolutionFwdAlgo_t fwd_algo1 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionFwdAlgo_t fwd_algo2 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);

    auto layout = GetCudnnTensorFormat(DataLayout::kNCHW);

    // ddo = conv(ddI, W) + conv(I, ddW)
    size_t workspace_size = 0;
    if (ddO) {
      ddy = ddO->mutable_data<T>(ctx.GetPlace());
      args1.handle = handle;
      args1.idesc.set(*ddX, iwo_group);
      args1.wdesc.set(*W, layout, iwo_group);
      args1.odesc.set(*ddO, iwo_group);
      args1.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search1 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      fwd_algo1 = search1::Find<T>(args1, exhaustive_search, false, 0, ctx);
      workspace_size = search1::GetWorkspaceSize(args1, fwd_algo1);

      if (ddW) {
        ddw = ddW->data<T>();
        args2.handle = handle;
        args2.idesc.set(*X, iwo_group);
        args2.wdesc.set(*ddW, layout, iwo_group);
        args2.odesc.set(*ddO, iwo_group);
        args2.cdesc.set(dtype, paddings, strides, dilations, c_group);

        using search2 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo2 = search2::Find<T>(args2, exhaustive_search, false, 0, ctx);
        workspace_size = std::max(workspace_size,
                                  search2::GetWorkspaceSize(args2, fwd_algo2));
      }
    }

    if (dW) {
      dw = dW->mutable_data<T>(ctx.GetPlace());
      args3.handle = handle;
      args3.idesc.set(*ddX, iwo_group);
      args3.wdesc.set(*dW, layout, iwo_group);
      args3.odesc.set(*dO, iwo_group);
      args3.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search3 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search3::Find<T>(args3, exhaustive_search, deterministic, 1, ctx);
      workspace_size = std::max(workspace_size,
                                search3::GetWorkspaceSize(args3, filter_algo));
    }

    if (ddW && dX) {
      dx = dX->mutable_data<T>(ctx.GetPlace());
      args4.handle = handle;
      args4.idesc.set(*dX, iwo_group);
      args4.wdesc.set(*ddW, layout, iwo_group);
      args4.odesc.set(*dO, iwo_group);
      args4.cdesc.set(dtype, paddings, strides, dilations, c_group);

      using search4 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search4::Find<T>(args4, exhaustive_search, deterministic, 2, ctx);
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4, data_algo));
    }

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(X->dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(dO->dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d, &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = W->numel() / groups;

    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto wkspace_handle = dev_ctx.cudnn_workspace_handle();

    if (ddO) {
      ddx = ddX->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                  handle, &alpha, args1.idesc.desc(), ddx + i * group_offset_in,
                  args1.wdesc.desc(), w + i * group_offset_filter,
                  args1.cdesc.desc(), fwd_algo1, workspace_ptr, workspace_size,
                  &beta, args1.odesc.desc(), ddy + i * group_offset_out));
            },
            workspace_size);
      }
      if (ddW) {
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                    handle, &alpha, args2.idesc.desc(), x + i * group_offset_in,
                    args2.wdesc.desc(), ddw + i * group_offset_filter,
                    args2.cdesc.desc(), fwd_algo2, workspace_ptr,
                    workspace_size, &alpha, args2.odesc.desc(),
                    ddy + i * group_offset_out));
              },
              workspace_size);
        }
      }
    }

    if (dW) {
      ddx = ddX->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args3.idesc.desc(), ddx + i * group_offset_in,
                  args3.odesc.desc(), dy + i * group_offset_out,
                  args3.cdesc.desc(), filter_algo, workspace_ptr,
                  workspace_size, &beta, args3.wdesc.desc(),
                  dw + i * group_offset_filter));
            },
            workspace_size);
      }
    }

    if (dX && ddW) {
      ddw = ddW->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, args4.wdesc.desc(),
                  ddw + i * group_offset_filter, args4.odesc.desc(),
                  dy + i * group_offset_out, args4.cdesc.desc(), data_algo,
                  workspace_ptr, workspace_size, &beta, args4.idesc.desc(),
                  dx + i * group_offset_in));
            },
            workspace_size);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>);
