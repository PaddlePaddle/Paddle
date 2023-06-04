// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_factory.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
 
class ExprWrapper {
  public:
  ExprWrapper(){}
  ExprWrapper(const nvinfer1::IDimensionExpr* expr, nvinfer1::IExprBuilder* expr_builder) {
    this->expr = expr;
    this->expr_builder = expr_builder;
  }

  const nvinfer1::IDimensionExpr* extract_expr() const {
    return expr;
  }

  public:

  friend ExprWrapper BinaryOp(const ExprWrapper& a, const ExprWrapper& b, nvinfer1::DimensionOperation op) {
    ExprWrapper result;
    if(a.expr_builder) {
      result.expr_builder = a.expr_builder;
    }
    if(b.expr_builder) {
      result.expr_builder = b.expr_builder;
    }
    assert(result.expr);
    result.expr = result.expr_builder->operation(op, *a.expr, *b.expr);
    return result;
  }

  friend ExprWrapper BinaryOp(const ExprWrapper& a, int b_value, nvinfer1::DimensionOperation op) {
    assert(a.expr_builder);
    ExprWrapper b;
    b.expr_builder = a.expr_builder;
    b.expr = b.expr_builder->constant(b_value);
    return BinaryOp(a, b, op);
  }

  friend ExprWrapper operator+(const ExprWrapper& a, const ExprWrapper& b) {
    return BinaryOp(a, b, nvinfer1::DimensionOperation::kSUM);
  }

  friend ExprWrapper operator+(const ExprWrapper& a, int b_value) {
    return BinaryOp(a, b_value, nvinfer1::DimensionOperation::kSUM);
  }

  friend ExprWrapper operator-(const ExprWrapper& a, const ExprWrapper& b) {
    return BinaryOp(a, b, nvinfer1::DimensionOperation::kSUB);
  }

  friend ExprWrapper operator-(const ExprWrapper& a, int b_value) {
    return BinaryOp(a, b_value, nvinfer1::DimensionOperation::kSUB);
  }

  friend ExprWrapper operator*(const ExprWrapper& a, const ExprWrapper& b) {
    return BinaryOp(a, b, nvinfer1::DimensionOperation::kPROD);
  }

  friend ExprWrapper operator*(const ExprWrapper& a, int b_value) {
    return BinaryOp(a, b_value, nvinfer1::DimensionOperation::kPROD);
  }

  friend ExprWrapper operator/(const ExprWrapper& a, const ExprWrapper& b) {
    return BinaryOp(a, b, nvinfer1::DimensionOperation::kFLOOR_DIV);
  }

  friend ExprWrapper operator/(const ExprWrapper& a, int b_value) {
    return BinaryOp(a, b_value, nvinfer1::DimensionOperation::kFLOOR_DIV);
  }

  public:
  const nvinfer1::IDimensionExpr* expr;
  nvinfer1::IExprBuilder* expr_builder;
};

static std::vector<ExprWrapper> DimsExprs2VecExprWrapper(const nvinfer1::DimsExprs& x_dims, nvinfer1::IExprBuilder& expr_builder) {
  std::vector<ExprWrapper> x_dims_wrap;
  for (int i = 0; i < x_dims.nbDims; i++) {
    x_dims_wrap.push_back(ExprWrapper(x_dims.d[i], &expr_builder));
  }
  return x_dims_wrap;
}

static nvinfer1::DimsExprs VecExprWrapper2DimsExprs(const std::vector<ExprWrapper>& output_dims_wrapper) {
  nvinfer1::DimsExprs output_dims;
  output_dims.nbDims = output_dims_wrapper.size();
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_dims.d[i] = output_dims_wrapper[i].extract_expr();
  }
  return output_dims;
}

nvinfer1::DimsExprs GatherNdInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dims = inputs[0];
  const int x_dims_size = inputs[0].nbDims;
  const nvinfer1::DimsExprs index_dims = inputs[1];
  const int index_dims_size = inputs[1].nbDims;

  std::vector<const nvinfer1::IDimensionExpr*> result_dims;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    result_dims.emplace_back(index_dims.d[i]);
  }

  if (index_dims.d[index_dims_size - 1]->isConstant()) {
    for (int i = index_dims.d[index_dims_size - 1]->getConstantValue();
         i < x_dims_size;
         ++i) {
      result_dims.emplace_back(x_dims.d[i]);
    }
  }

  nvinfer1::DimsExprs output;
  output.nbDims = result_dims.size();
  for (int i = 0; i < output.nbDims; i++) {
    output.d[i] = result_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs YoloBoxInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      2,
      phi::errors::InvalidArgument("inputs of yolo_box should be equal to 2, "
                                   "But received (%s)",
                                   nb_inputs));

  const nvinfer1::DimsExprs dim_x = inputs[0];

  auto anchors = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("anchors"));
  int anchor_num = anchors.size() / 2;

  // box_num = dim_x[2] * dim_x[3] * anchor_num;
  const nvinfer1::IDimensionExpr* box_num = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kPROD, *dim_x.d[2], *dim_x.d[3]),
      *expr_builder.constant(anchor_num));

  nvinfer1::DimsExprs output;
  output.nbDims = 3;
  if (output_index == 0) {
    output.d[0] = dim_x.d[0];
    output.d[1] = box_num;
    output.d[2] = expr_builder.constant(4);
  } else {
    auto class_num = PADDLE_GET_CONST(int, op_desc.GetAttr("class_num"));
    output.d[0] = dim_x.d[0];
    output.d[1] = box_num;
    output.d[2] = expr_builder.constant(class_num);
  }
  return output;
}

nvinfer1::DimsExprs InstanceNormInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  nvinfer1::DimsExprs x_dims = inputs[0];
  return x_dims;
}

inline const nvinfer1::IDimensionExpr* CalcOutputSize(
    const nvinfer1::IDimensionExpr* input_size,
    const nvinfer1::IDimensionExpr* filter_size,
    const nvinfer1::IDimensionExpr* dilation,
    const nvinfer1::IDimensionExpr* padding1,
    const nvinfer1::IDimensionExpr* padding2,
    const nvinfer1::IDimensionExpr* stride,
    nvinfer1::IExprBuilder& expr_builder  // NOLINT
) {
  // dkernel = dilation * (filter_size - 1) + 1;
  const nvinfer1::IDimensionExpr* dkernel = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUM,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kPROD,
          *dilation,
          *expr_builder.operation(nvinfer1::DimensionOperation::kSUB,
                                  *filter_size,
                                  *expr_builder.constant(1))),
      *expr_builder.constant(1));

  // output_size = (input_size + padding1 + padding2 - dkernel) / stride + 1;
  const nvinfer1::IDimensionExpr* tmp = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUB,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kSUM, *input_size, *padding1),
          *padding2),
      *dkernel);

  const nvinfer1::IDimensionExpr* output_size = expr_builder.operation(
      nvinfer1::DimensionOperation::kSUM,
      *expr_builder.operation(
          nvinfer1::DimensionOperation::kFLOOR_DIV, *tmp, *stride),
      *expr_builder.constant(1));
  return output_size;
}

nvinfer1::DimsExprs UnflodInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      phi::errors::InvalidArgument("inputs of unfold should be equal to 1, "
                                   "But received (%s)",
                                   nb_inputs));

  const nvinfer1::DimsExprs in_dims = inputs[0];
  std::vector<const nvinfer1::IDimensionExpr*> out_dims;
  out_dims.push_back(in_dims.d[0]);

  auto kernel_sizes =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("kernel_sizes"));
  auto dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));

  // output_channels = in_dims[1] * kernel_sizes[0] * kernel_sizes[1];
  const nvinfer1::IDimensionExpr* output_channels = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD,
      *in_dims.d[1],
      *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                              *expr_builder.constant(kernel_sizes[0]),
                              *expr_builder.constant(kernel_sizes[1])));
  out_dims.push_back(output_channels);

  const nvinfer1::IDimensionExpr* output_height =
      CalcOutputSize(in_dims.d[2],
                     expr_builder.constant(kernel_sizes[0]),
                     expr_builder.constant(dilations[0]),
                     expr_builder.constant(paddings[0]),
                     expr_builder.constant(paddings[2]),
                     expr_builder.constant(strides[0]),
                     expr_builder);
  const nvinfer1::IDimensionExpr* output_width =
      CalcOutputSize(in_dims.d[3],
                     expr_builder.constant(kernel_sizes[1]),
                     expr_builder.constant(dilations[1]),
                     expr_builder.constant(paddings[1]),
                     expr_builder.constant(paddings[3]),
                     expr_builder.constant(strides[1]),
                     expr_builder);

  const nvinfer1::IDimensionExpr* output_col_length = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD, *output_height, *output_width);

  out_dims.push_back(output_col_length);
  nvinfer1::DimsExprs output;
  output.nbDims = out_dims.size();
  for (size_t i = 0; i < out_dims.size(); i++) output.d[i] = out_dims[i];
  return output;
}

nvinfer1::DimsExprs ScatterNdAddInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    3,
                    phi::errors::InvalidArgument(
                        "inputs of scatter_nd_add should be equal to 3, "
                        "But received (%s)",
                        nb_inputs));
  const nvinfer1::DimsExprs ref_dims = inputs[0];
  return ref_dims;
}

nvinfer1::DimsExprs UnchangedInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    phi::errors::InvalidArgument(
                        "inputs of UnchangedInferMeta should be equal to 1, "
                        "But received (%s)",
                        nb_inputs));
  return inputs[0];
}

nvinfer1::DimsExprs MoeInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  return inputs[0];
}

nvinfer1::DimsExprs Pad3dInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dim = inputs[0];

  nvinfer1::DimsExprs out_dims;
  out_dims.nbDims = x_dim.nbDims;

  out_dims.d[0] = x_dim.d[0];

  auto paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  auto data_format =
      PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));

  if (data_format == "NCDHW") {
    out_dims.d[1] = x_dim.d[1];
  } else {
    out_dims.d[4] = x_dim.d[4];
  }

  if (data_format == "NCDHW") {
    // depth
    out_dims.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[2],
                                *expr_builder.constant(paddings[4])),
        *expr_builder.constant(paddings[5]));
    // height
    out_dims.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[3],
                                *expr_builder.constant(paddings[2])),
        *expr_builder.constant(paddings[3]));
    // width
    out_dims.d[4] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[4],
                                *expr_builder.constant(paddings[0])),
        *expr_builder.constant(paddings[1]));
  } else {  // NDHWC
    // depth
    out_dims.d[1] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[1],
                                *expr_builder.constant(paddings[4])),
        *expr_builder.constant(paddings[5]));
    // height
    out_dims.d[2] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[2],
                                *expr_builder.constant(paddings[2])),
        *expr_builder.constant(paddings[3]));
    // width
    out_dims.d[3] = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUM,
        *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                *x_dim.d[3],
                                *expr_builder.constant(paddings[0])),
        *expr_builder.constant(paddings[1]));
  }
  return out_dims;
}

nvinfer1::DimsExprs PNormInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  bool asvector = PADDLE_GET_CONST(bool, op_desc.GetAttr("asvector"));
  bool keepdim = PADDLE_GET_CONST(bool, op_desc.GetAttr("keepdim"));
  int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));

  auto x_dim = inputs[0];
  auto x_rank = x_dim.nbDims;

  PADDLE_ENFORCE_GE(axis,
                    -x_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], R is "
                        "the rank of Input(X). But received axis: %d, R: %d. "
                        "Current Input(X)'s shape is=[%s].",
                        axis,
                        x_rank,
                        x_dim.d));
  PADDLE_ENFORCE_LT(axis,
                    x_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], R is "
                        "the rank of Input(X). But received axis: %d, R: %d. "
                        "Current Input(X)'s shape is=[%s].",
                        axis,
                        x_rank,
                        x_dim.d));

  // TODO(liuyuanle): support asvector = True
  PADDLE_ENFORCE_EQ(
      asvector,
      false,
      phi::errors::InvalidArgument(
          "p_norm only support asvector=false, but received asvector: %d.",
          asvector));

  std::vector<const nvinfer1::IDimensionExpr*> reduce_dims;

  if (asvector) {
    reduce_dims.emplace_back(expr_builder.constant(1));
    if (keepdim) {
      for (int i = 1; i < x_dim.nbDims; ++i) {
        reduce_dims.emplace_back(expr_builder.constant(1));
      }
      x_dim.nbDims = reduce_dims.size();
      for (size_t i = 0; i < reduce_dims.size(); i++) {
        x_dim.d[i] = reduce_dims[i];
      }
    }
  } else {
    if (axis < 0) axis = x_dim.nbDims + axis;
    for (int i = 0; i < x_dim.nbDims; ++i) {
      if (i != axis) reduce_dims.emplace_back(x_dim.d[i]);
    }
    if (reduce_dims.size() == 0) {
      reduce_dims.emplace_back(expr_builder.constant(1));
    }
  }
  x_dim.d[axis] = expr_builder.constant(1);

  nvinfer1::DimsExprs output;
  if (keepdim) {
    output = x_dim;
  } else {
    output.nbDims = reduce_dims.size();
    for (int i = 0; i < output.nbDims; i++) output.d[i] = reduce_dims[i];
  }
  return output;
}

nvinfer1::DimsExprs GridSamplerInferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const nvinfer1::DimsExprs x_dims = inputs[0];
  const nvinfer1::DimsExprs grid_dims = inputs[1];

  nvinfer1::DimsExprs output;
  if (grid_dims.nbDims == 4) {
    output.nbDims = 4;
    output.d[0] = x_dims.d[0];
    output.d[1] = x_dims.d[1];
    output.d[2] = grid_dims.d[1];
    output.d[3] = grid_dims.d[2];
  } else {
    output.nbDims = 5;
    output.d[0] = x_dims.d[0];
    output.d[1] = x_dims.d[1];
    output.d[2] = grid_dims.d[1];
    output.d[3] = grid_dims.d[2];
    output.d[4] = grid_dims.d[3];
  }
  return output;
}

inline const void UpdatePaddingAndDilation(
     std::vector<const nvinfer1::IDimensionExpr*>* paddings,
     std::vector<const nvinfer1::IDimensionExpr*>* dilation,
     const std::string padding_algorithm,
     const std::vector<const nvinfer1::IDimensionExpr*>& data_dims,
     const std::vector<const nvinfer1::IDimensionExpr*>& strides,
     const std::vector<const nvinfer1::IDimensionExpr*>& ksize,
     nvinfer1::IExprBuilder& expr_builder  // NOLINT
 ) {
   if (paddings->size() == data_dims.size()) {
     for (size_t i = 0; i < data_dims.size(); ++i) {
       const nvinfer1::IDimensionExpr* copy_pad = *(paddings->begin() + 2 * i);
       paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
     }
   }

   // when padding_algorithm is "VALID" or "SAME"
   if (padding_algorithm == "SAME") {
     for (size_t i = 0; i < data_dims.size(); ++i) {
       const nvinfer1::IDimensionExpr* out_size = expr_builder.operation(
           nvinfer1::DimensionOperation::kFLOOR_DIV,
           *expr_builder.operation(
               nvinfer1::DimensionOperation::kSUB,
               *expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                       *data_dims[i],
                                       *strides[i]),
               *expr_builder.constant(1)),
           *strides[i]);
       const nvinfer1::IDimensionExpr* pad_sum = expr_builder.operation(
           nvinfer1::DimensionOperation::kMAX,
           *expr_builder.operation(
               nvinfer1::DimensionOperation::kSUM,
               *expr_builder.operation(
                   nvinfer1::DimensionOperation::kPROD,
                   *expr_builder.operation(nvinfer1::DimensionOperation::kSUB,
                                           *out_size,
                                           *expr_builder.constant(1)),
                   *strides[i]),
               *expr_builder.operation(nvinfer1::DimensionOperation::kSUB,
                                       *ksize[i],
                                       *data_dims[i])),
           *expr_builder.constant(0));
       const nvinfer1::IDimensionExpr* pad_0 =
           expr_builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                  *pad_sum,
                                  *expr_builder.constant(2));
       const nvinfer1::IDimensionExpr* pad_1 = expr_builder.operation(
           nvinfer1::DimensionOperation::kSUB, *pad_sum, *pad_0);

       *(paddings->begin() + i * 2) = pad_0;
       *(paddings->begin() + i * 2 + 1) = pad_1;

       // dilation
       *(dilation->begin() + i) = expr_builder.constant(1);
     }

   } else if (padding_algorithm == "VALID") {
     for (auto it = paddings->begin(); it != paddings->end(); it++) {
       *it = expr_builder.constant(0);
     }
   }
 }

 inline const nvinfer1::IDimensionExpr* ConvOutputSize(
     const nvinfer1::IDimensionExpr* input_size,
     const nvinfer1::IDimensionExpr* filter_size,
     const nvinfer1::IDimensionExpr* dilation,
     const nvinfer1::IDimensionExpr* padding_1,
     const nvinfer1::IDimensionExpr* padding_2,
     const nvinfer1::IDimensionExpr* stride,
     nvinfer1::IExprBuilder& expr_builder  // NOLINT
 ) {
   
   // Here are all examples of using h(height), ok for weight too.
   ExprWrapper ih(input_size, &expr_builder);
   ExprWrapper kh(filter_size, &expr_builder);
   ExprWrapper dilation_h(dilation, &expr_builder);
   ExprWrapper pad_h1(padding_1, &expr_builder);
   ExprWrapper pad_h2(padding_2, &expr_builder);
   ExprWrapper stride_h(stride, &expr_builder);

  ExprWrapper oh = (ih + pad_h1 + pad_h2 - dilation_h * (kh - 1) - 1) / stride_h + 1;
  return oh.extract_expr();

 }

 nvinfer1::DimsExprs Conv2dFusionInferMeta(
     int output_index,
     const nvinfer1::DimsExprs* inputs,
     int nb_inputs,
     nvinfer1::IExprBuilder& expr_builder,  // NOLINT
     const framework::OpDesc& op_desc) {
   const std::vector<int> dilations =
       PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
   const std::vector<int> strides =
       PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
   std::vector<int> paddings =
       PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));

   std::string padding_algorithm = "EXPLICIT";
   if (op_desc.HasAttr("padding_algorithm"))
     padding_algorithm =
         PADDLE_GET_CONST(std::string, op_desc.GetAttr("padding_algorithm"));
   if (padding_algorithm == "VALID") {
     for (size_t i = 0; i < paddings.size(); i++) {
       paddings[i] = 0;
     }
   }

   // TODO(zhangjun): nhwc support
   bool channel_last = false;
   // conv_fusion: input, filter, bias
   const nvinfer1::DimsExprs input_dims = inputs[0];
   const nvinfer1::DimsExprs filter_dims = inputs[1];
   std::vector<const nvinfer1::IDimensionExpr*> in_data_dims;  // d, h, w
   if (channel_last) {
     for (int i = 1; i < input_dims.nbDims - 1; ++i) {
       in_data_dims.emplace_back(input_dims.d[i]);
     }
   } else {
     for (int i = 2; i < input_dims.nbDims; ++i) {
       in_data_dims.emplace_back(input_dims.d[i]);
     }
   }

   std::vector<const nvinfer1::IDimensionExpr*>
       filter_data_dims;  // filter_h, filter_w
   if (channel_last) {
     for (int i = 1; i < filter_dims.nbDims - 1; ++i) {
       filter_data_dims.emplace_back(filter_dims.d[i]);
     }
   } else {
     for (int i = 2; i < filter_dims.nbDims; ++i) {
       filter_data_dims.emplace_back(filter_dims.d[i]);
     }
   }

   std::vector<const nvinfer1::IDimensionExpr*> paddings_exprs;
   for (size_t i = 0; i < paddings.size(); ++i) {
     paddings_exprs.emplace_back(expr_builder.constant(paddings[i]));
   }

   std::vector<const nvinfer1::IDimensionExpr*> dilations_exprs;
   for (size_t i = 0; i < dilations.size(); ++i) {
     dilations_exprs.emplace_back(expr_builder.constant(dilations[i]));
   }

   std::vector<const nvinfer1::IDimensionExpr*> strides_exprs;
   for (size_t i = 0; i < strides.size(); ++i) {
     strides_exprs.emplace_back(expr_builder.constant(strides[i]));
   }

   UpdatePaddingAndDilation(&paddings_exprs,
                            &dilations_exprs,
                            padding_algorithm,
                            in_data_dims,
                            strides_exprs,
                            filter_data_dims,
                            expr_builder);
   nvinfer1::DimsExprs output;
   output.nbDims = 2 + in_data_dims.size();
   int out_idx = 0;
   output.d[out_idx++] = input_dims.d[0];
   if (!channel_last) {
     output.d[out_idx++] = filter_dims.d[0];
   }
   for (size_t i = 0; i < in_data_dims.size(); ++i) {
     output.d[out_idx++] = ConvOutputSize(in_data_dims[i],
                                          filter_data_dims[i],
                                          dilations_exprs[i],
                                          paddings_exprs[2 * i],
                                          paddings_exprs[2 * i + 1],
                                          expr_builder.constant(strides[i]),
                                          expr_builder);
   }
   if (channel_last) {
     output.d[out_idx++] = filter_dims.d[0];
   }
   return output;
 }


nvinfer1::DimsExprs LookupTableV2InferMeta(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder,  // NOLINT
    const framework::OpDesc& op_desc) {
  const auto x_dims = inputs[0];
  const auto weight_dims = inputs[1];

  nvinfer1::DimsExprs output;
  output.nbDims = x_dims.nbDims + 1;
  for (int i = 0; i < x_dims.nbDims; ++i) {
    output.d[i] = x_dims.d[i];
  }
  output.d[x_dims.nbDims] = weight_dims.d[1];
  return output;
}


nvinfer1::DimsExprs Conv2dTransposeInferMeta(int output_index,
                              const nvinfer1::DimsExprs* inputs,
                              int nb_inputs,
                              nvinfer1::IExprBuilder& expr_builder,  // NOLINT
                              const framework::OpDesc& op_desc) {
  auto x_dims = inputs[0];
  auto filter_dims = inputs[1];
  std::vector<ExprWrapper> x_dims_wrap = DimsExprs2VecExprWrapper(x_dims, expr_builder);
  std::vector<ExprWrapper> filter_dims_wrap = DimsExprs2VecExprWrapper(filter_dims, expr_builder);

  const std::vector<int> dilations =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));
  const std::vector<int> strides =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
  std::vector<int> paddings =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
  std::vector<int> output_size =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("output_size"));
  std::vector<int> output_padding =
      PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("output_padding"));
  auto data_format = PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));
  int groups = PADDLE_GET_CONST(int, op_desc.GetAttr("groups"));

   std::string padding_algorithm = "EXPLICIT";
   if (op_desc.HasAttr("padding_algorithm")) {
     padding_algorithm =
         PADDLE_GET_CONST(std::string, op_desc.GetAttr("padding_algorithm"));
   }
  CHECK_EQ(padding_algorithm == "EXPLICIT", true);
  CHECK_EQ(data_format == "NCHW", true);
  CHECK_EQ(output_size.size() == 0, true);
  CHECK_EQ(paddings.size() == 2, true);
  CHECK_EQ(x_dims.nbDims == 4, true);
  CHECK_EQ(x_dims.nbDims == filter_dims.nbDims, true);

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    CHECK_EQ(strides[i] > 0, true);
  }

  int in_sub_stride_size = x_dims.nbDims - stride_size;
  CHECK_EQ(in_sub_stride_size == 2, true);

  if (output_size.size()) {
    CHECK_EQ(output_size.size() == strides.size(), true);
  }

  if (output_padding.size()) {
    CHECK_EQ(strides.size() == output_padding.size(), true);
  }
  
  std::vector<ExprWrapper> output_dims_wrap(x_dims.nbDims);
  output_dims_wrap[0] = x_dims_wrap[0];
  output_dims_wrap[1] = filter_dims_wrap[1] * groups;

  auto ih = x_dims_wrap[2];
  auto iw = x_dims_wrap[3];
  auto kh = filter_dims_wrap[2];
  auto kw = filter_dims_wrap[3];

  int pad_h0 = paddings[0];
  int pad_h1 = paddings[0];
  int pad_w0 = paddings[1];
  int pad_w1 = paddings[1];

  output_dims_wrap[2] = (ih - 1) * strides[0] - pad_h0 - pad_h1  + (kh - 1) * dilations[0] + 1;
  output_dims_wrap[3] = (iw - 1) * strides[1] - pad_w0 - pad_w1  + (kw - 1) * dilations[1] + 1;

  return VecExprWrapper2DimsExprs(output_dims_wrap);
}

PD_REGISTER_DYNAMIC_INFER_META_FN(gather_nd, GatherNdInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(yolo_box, YoloBoxInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(instance_norm, InstanceNormInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(unfold, UnflodInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(scatter_nd_add, ScatterNdAddInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(inverse, UnchangedInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(moe, MoeInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(pad3d, Pad3dInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(grid_sampler, GridSamplerInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(conv2d_fusion, Conv2dFusionInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(conv2d, Conv2dFusionInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(conv2d_transpose, Conv2dTransposeInferMeta);
PD_REGISTER_DYNAMIC_INFER_META_FN(p_norm, PNormInferMeta);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
