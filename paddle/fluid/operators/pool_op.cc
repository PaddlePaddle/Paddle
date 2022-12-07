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

#include "paddle/fluid/operators/pool_op.h"

#include <unordered_map>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

bool CanMKLDNNSupportPool(const framework::ExecutionContext& ctx) {
  if (ctx.Attr<bool>("adaptive") == false) return true;
  // (jczaja): oneDNN is supporting only unchangable in size pool window
  auto src_tz = phi::vectorize(ctx.Input<phi::DenseTensor>("X")->dims());
  if (!ctx.HasAttr("ksize")) {
    return false;
  }
  std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
  // Fast but not exhustive check
  return ((src_tz[src_tz.size() - 1] % ksize[1] == 0) &&
          (src_tz[src_tz.size() - 2] % ksize[0] == 0));
}

framework::OpKernelType PoolOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_MKLDNN
  this->SetDnnFallback(!CanMKLDNNSupportPool(ctx));
  // NOTE(jiahongyu) END: Above codes originally enclosed by PADDLE_WITH_MKLDNN

  return framework::OpKernelType(data_type, ctx.GetPlace());
}

framework::OpKernelType PoolOp::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  if ((expected_kernel_type.data_layout_ == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = phi::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != phi::DataLayout::kAnyLayout) {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(
      expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}

framework::OpKernelType PoolOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_MKLDNN
  this->SetDnnFallback(!CanMKLDNNSupportPool(ctx));
  // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_MKLDNN

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

framework::OpKernelType PoolOpGrad::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  if ((expected_kernel_type.data_layout_ == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(),
                                   phi::StringToDataLayout(data_format));
  }
#endif
  return framework::OpKernelType(
      expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}

void Pool2dOpMaker::Make() {
  AddInput(
      "X",
      "(phi::DenseTensor) The input tensor of pooling operator. "
      "The format of input tensor is NCHW, where N is batch size, C is the "
      "number of channels, H is the height of the feature, "
      "and W is the width of the feature.");
  AddOutput("Out",
            "(phi::DenseTensor) The output tensor of pooling operator. "
            "The format of output tensor is also NCHW, "
            "where N is batch size, C is the number of channels, "
            "H is the height of the feature, "
            "and W is the width of the feature.");

  AddAttr<std::string>("pooling_type",
                       "(string), pooling type, can be \"max\" for max-pooling "
                       "and \"avg\" for average-pooling.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>("ksize",
                            "(vector<int>) The pooling window "
                            "size(height, width) of the pooling operator. "
                            "If global_pooling = true, ksize and paddings will "
                            "be ignored.")
      .SupportTensor();
  AddAttr<bool>(
      "global_pooling",
      "(bool) Whether to use the global pooling. "
      "If global_pooling = true, kernel size and paddings will be ignored. "
      "Default False.")
      .SetDefault(false);
  AddAttr<std::vector<int>>("strides",
                            "(vector<int>, default {1, 1}), strides(height, "
                            "width) of pooling operator.")
      .SetDefault({1, 1});
  // TODO(Chengduo): Add checker. (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int>, default {0,0}), paddings(height_top, height_bottom, "
      "width_left, wifth_right) of pooling operator."
      "If global_pooling = true, paddings and kernel size will be ignored.")
      .SetDefault({0, 0});
  AddAttr<bool>(
      "exclusive",
      "(bool) When true, will exclude the zero-padding in the "
      "averaging calculating, otherwise, include the zero-padding. Note, it "
      "is only used when pooling_type is avg. The default is True. "
      "Default True.")
      .SetDefault(true);
  AddAttr<bool>(
      "adaptive",
      "(bool) When true, will perform adaptive pooling instead, "
      "output shape in H and W dimensions will be same as ksize, input data "
      "will be divided into grids specify by ksize averagely and perform "
      "pooling in each grid area to get output pooling value. "
      "Default False.")
      .SetDefault(false);
  AddAttr<bool>(
      "ceil_mode",
      "(bool) Whether to use the ceil function to calculate "
      "output height and width. False is the default. If it is set to False, "
      "the floor function will be used. Default False")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCHW");
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  // TODO(dzhwinter): need to registered layout transform function
  AddAttr<bool>(
      "use_cudnn",
      "(bool) Only used in cudnn kernel, need install cudnn. Default False")
      .SetDefault(false)
      .AsExtra();
  AddComment(R"DOC(
This operation calculates the pooling output based on
the input, pooling_type and pool_size, pool_stride, pool_padding parameters.
Input(X) and Output(Out) are in NCHW or NHWC format, where N is batch size, C is the
number of channels, H is the height of the feature, and W is the width of the feature.
Parameters(pool_size, pool_stride, pool_padding) hold two integer elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:

  Input:

       X shape: $(N, C, H_{in}, W_{in})$

  Output:

       Out shape: $(N, C, H_{out}, W_{out})$

  For pool_padding = "SAME":
       $$
       H_{out} = \\frac{(H_{in} + strides[0] - 1)}{strides[0]}
       $$
       $$
       W_{out} = \\frac{(W_{in} + strides[1] - 1)}{strides[1]}
       $$

  For pool_padding = "VALID":
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + strides[0])}{strides[0]}
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + strides[1])}{strides[1]}
       $$

  For ceil_mode = false:
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + pad_height_top + pad_height_bottom}{strides[0]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + pad_width_left + pad_width_right}{strides[1]} + 1
       $$

  For ceil_mode = true:
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + pad_height_top + pad_height_bottom + strides[0] - 1)}{strides[0]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + pad_width_left + pad_width_right + strides[1] - 1)}{strides[1]} + 1
       $$

  For exclusive = false:
       $$
       hstart = i * strides[0] - pad_height_top
       $$
       $$
       hend = hstart + ksize[0]
       $$
       $$
       wstart = j * strides[1] - pad_width_left
       $$
       $$
       wend = wstart + ksize[1]
       $$
       $$
       Output(i ,j) = \\frac{sum(Input[hstart:hend, wstart:wend])}{ksize[0] * ksize[1]}
       $$

  For exclusive = true:
       $$
       hstart = max(0, i * strides[0] - pad_height_top)
       $$
       $$
       hend = min(H, hstart + ksize[0])
       $$
       $$
       wstart = max(0, j * strides[1] - pad_width_left)
       $$
       $$
       wend = min(W, wstart + ksize[1])
       $$
       $$
       Output(i ,j) = \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}
       $$

)DOC");
}

template <typename T>
class Pool2dOpGradGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("pool2d_double_grad");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class PoolOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

void Pool3dOpMaker::Make() {
  AddInput("X",
           "(phi::DenseTensor) The input tensor of pooling operator. "
           "The format of input tensor is NCDHW or NDHWC, where N is batch "
           "size, C is "
           "the number of channels, and D, H and W is the depth, height and "
           "width of "
           "the feature, respectively.");
  AddOutput("Out",
            "(phi::DenseTensor) The output tensor of pooling operator."
            "The format of output tensor is also NCDHW or NDHWC, "
            "where N is batch size, C is "
            "the number of channels, and D, H and W is the depth, height and "
            "width of the feature, respectively.");

  AddAttr<std::string>("pooling_type",
                       "(string) Pooling type, can be \"max\" for max-pooling "
                       "and \"avg\" for average-pooling.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>(
      "ksize",
      "(vector<int>) The pooling window size(depth, height, "
      "width) of pooling operator. "
      "If global_pooling = true, ksize and paddings will "
      "be ignored.");
  AddAttr<bool>(
      "global_pooling",
      "(bool) Whether to use the global pooling. "
      "If global_pooling = true, kernel size and paddings will be ignored. "
      "Default False")
      .SetDefault(false);
  AddAttr<std::vector<int>>(
      "strides",
      "(vector<int>, default {1,1,1}) Strides(depth, height, "
      "width) of the pooling operator.")
      .SetDefault({1, 1, 1});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int>, default {0,0,0}), paddings(pad_depth_front, "
      "pad_depth_back, "
      "pad_height_top, pad_height_bottom, pad_width_left, pad_width_right"
      ") of pooling operator. "
      "If global_pooling = true, ksize and paddings will be ignored.")
      .SetDefault({0, 0, 0});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "exclusive",
      "(bool) When true, will exclude the zero-padding in the "
      "averaging calculating, otherwise, include the zero-padding. Note, it "
      "is only used when pooling_type is avg. The default is True. "
      "Default True")
      .SetDefault(true);
  AddAttr<bool>(
      "adaptive",
      "(bool) When true, will perform adaptive pooling instead, "
      "output shape in H and W dimensions will be same as ksize, input data "
      "will be divided into grids specify by ksize averagely and perform "
      "pooling in each grid area to get output pooling value. "
      "Default False")
      .SetDefault(false);
  AddAttr<bool>(
      "ceil_mode",
      "(bool) Whether to use the ceil function to calculate "
      "output height and width. False is the default. If it is set to False, "
      "the floor function will be used. Default False")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCDHW) Only used in "
      "An optional string from: \"NDHWC\", \"NCDHW\". "
      "Defaults to \"NDHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCDHW");
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddAttr<bool>(
      "use_cudnn",
      "(bool) Only used in cudnn kernel, need install cudnn. Default False")
      .SetDefault(false)
      .AsExtra();
  AddComment(R"DOC(
This operation calculates the output based on
the input, pooling_type, pool_size, pool_stride, and pool_padding parameters.
Input(X) and output(Out) are in NCDHW or NDHWC format, where N is batch
size, C is the number of channels, and D, H and W are the depth, height and
width of the feature, respectively. Parameters(pool_size, pool_stride, pool_padding)
hold three integer elements. These three elements represent depth, height and
width, respectively. The input(X) size and output(Out) size may be different.

Example:
  Input:
       X shape: $(N, C, D_{in}, H_{in}, W_{in})$
  Output:
       Out shape: $(N, C, D_{out}, H_{out}, W_{out})$

  For pool_padding = "SAME":
       $$
       D_{out} = \\frac{(D_{in} + strides[0] - 1)}{strides[0]}
       $$
       $$
       H_{out} = \\frac{(H_{in} + strides[1] - 1)}{strides[1]}
       $$
       $$
       W_{out} = \\frac{(W_{in} + strides[2] - 1)}{strides[2]}
       $$

  For pool_padding = "VALID":
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + strides[0])}{strides[0]}
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + strides[1])}{strides[1]}
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + strides[2])}{strides[2]}
       $$

  For ceil_mode = false:
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + pad_depth_front + pad_depth_back)}{strides[0]} + 1
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + pad_height_top + pad_height_bottom)}{strides[1]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + pad_width_left + pad_width_right)}{strides[2]} + 1
       $$
  For ceil_mode = true:
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + pad_depth_front + pad_depth_back + strides[0] -1)}{strides[0]} + 1
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + pad_height_top + pad_height_bottom + strides[1] -1)}{strides[1]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + pad_width_left + pad_width_right + strides[2] -1)}{strides[2]} + 1
       $$

  For exclusive = false:
       $$
       dstart = i * strides[0] - pad_depth_front
       $$
       $$
       dend = dstart + ksize[0]
       $$
       $$
       hstart = j * strides[1] - pad_height_top
       $$
       $$
       hend = hstart + ksize[1]
       $$
       $$
       wstart = k * strides[2] -  pad_width_left
       $$
       $$
       wend = wstart + ksize[2]
       $$
       $$
       Output(i ,j, k) = \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{ksize[0] * ksize[1] * ksize[2]}
       $$

  For exclusive = true:
       $$
       dstart = max(0, i * strides[0] - pad_depth_front)
       $$
       $$
       dend = min(D, dstart + ksize[0])
       $$
       $$
       hstart = max(0, j * strides[1] - pad_height_top)
       $$
       $$
       hend = min(H, hstart + ksize[1])
       $$
       $$
       wstart = max(0, k * strides[2] - pad_width_left)
       $$
       $$
       wend = min(W, wstart + ksize[2])
       $$
       $$
       Output(i ,j, k) = \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}
       $$

)DOC");
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(pool2d,
                            Pool2dInferShapeFunctor,
                            PD_INFER_META(phi::Pool2DInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(pool2d_grad,
                            Pool2dGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(pool2d_double_grad,
                            Pool2dDoubleGradInferShapeFunctor,
                            PD_INFER_META(phi::Pool2DInferMeta));

REGISTER_OPERATOR(
    pool2d,
    ops::PoolOp,
    ops::Pool2dOpMaker,
    ops::PoolOpInferVarType,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    Pool2dInferShapeFunctor);
REGISTER_OPERATOR(pool2d_grad,
                  ops::PoolOpGrad,
                  ops::Pool2dOpGradGradMaker<paddle::framework::OpDesc>,
                  ops::Pool2dOpGradGradMaker<paddle::imperative::OpBase>,
                  Pool2dGradInferShapeFunctor);
REGISTER_OPERATOR(pool2d_double_grad,
                  ops::PoolOp,
                  Pool2dDoubleGradInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(pool3d,
                            Pool3dInferShapeFunctor,
                            PD_INFER_META(phi::PoolInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(pool3d_grad,
                            Pool3dGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(
    pool3d,
    ops::PoolOp,
    ops::Pool3dOpMaker,
    ops::PoolOpInferVarType,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    Pool3dInferShapeFunctor);
REGISTER_OPERATOR(pool3d_grad, ops::PoolOpGrad, Pool3dGradInferShapeFunctor);
