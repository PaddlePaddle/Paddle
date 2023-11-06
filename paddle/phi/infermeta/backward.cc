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

#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

void AffineGridGradInferMeta(const MetaTensor& output_grad,
                             const IntArray& outputShape,
                             bool align_corners,
                             MetaTensor* input_grad) {
  if (input_grad) {
    auto output_dims = output_grad.dims();
    if (output_dims.size() == 4) {
      input_grad->set_dims(phi::make_ddim({output_dims[0], 2, 3}));
    } else {
      input_grad->set_dims(phi::make_ddim({output_dims[0], 3, 4}));
    }
  }
}

void AngleGradInferMeta(const MetaTensor& x,
                        const MetaTensor& out_grad,
                        MetaTensor* x_grad) {
  UnchangedInferMeta(x, x_grad);
}

void BilinearGradInferMeta(const MetaTensor& x,
                           const MetaTensor& y,
                           const MetaTensor& weight,
                           const MetaTensor& dout,
                           MetaTensor* dx,
                           MetaTensor* dy,
                           MetaTensor* dweight,
                           MetaTensor* dbias) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto weight_dims = weight.dims();
  auto out_dims = dout.dims();

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      2UL,
      errors::InvalidArgument("The input(Out@GRAD) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      out_dims[0],
      errors::InvalidArgument(
          "The first dimension(batch_size) of input(Out@GRAD) must be "
          "equal to the first dimension of the Input(X)."));
  PADDLE_ENFORCE_EQ(
      weight_dims[0],
      out_dims[1],
      errors::InvalidArgument(
          "The second dimension of input(Out@GRAD) must be equal to "
          "the third dimension of the Input(Weight)."));

  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(x.dtype());
  }
  if (dy) {
    dy->set_dims(y_dims);
    dy->set_dtype(y.dtype());
  }
  if (dweight) {
    dweight->set_dims(weight_dims);
    dweight->set_dtype(weight.dtype());
  }
  if (dbias) {
    dbias->set_dims({1, out_dims[1]});
    dbias->set_dtype(dout.dtype());
  }
}

void BmmGradInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      const MetaTensor& out_grad,
                      MetaTensor* x_grad,
                      MetaTensor* y_grad) {
  if (x_grad) {
    x_grad->set_dims(x.dims());
    x_grad->set_dtype(x.dtype());
  }
  if (y_grad) {
    y_grad->set_dims(y.dims());
    y_grad->set_dtype(y.dtype());
  }
}

void ChannelShuffleGradInferMeta(const MetaTensor& out_grad,
                                 int groups,
                                 const std::string& data_format,
                                 MetaTensor* x_grad) {
  auto do_dims = out_grad.dims();
  PADDLE_ENFORCE_EQ(do_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        do_dims.size()));
  auto dx_dims = do_dims;
  x_grad->set_dims(dx_dims);
  x_grad->set_dtype(out_grad.dtype());
}

void ComplexGradInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          const MetaTensor& dout,
                          MetaTensor* dx,
                          MetaTensor* dy) {
  auto x_dims = x.dims();
  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(x.dtype());
  }
  auto y_dims = y.dims();
  if (dy) {
    dy->set_dims(y_dims);
    dy->set_dtype(y.dtype());
  }
}

void ConvTransposeGradInferMeta(const MetaTensor& x,
                                const MetaTensor& filter,
                                const MetaTensor& dout,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& output_padding,
                                const std::vector<int>& output_size,
                                const std::string& padding_algorithm,
                                int groups,
                                const std::vector<int>& dilations,
                                const std::string& data_format,
                                MetaTensor* dx,
                                MetaTensor* dfilter) {
  GeneralBinaryGradInferMeta(x, filter, dx, dfilter);
}

void Conv2dTransposeGradInferMeta(const MetaTensor& x,
                                  const MetaTensor& filter,
                                  const MetaTensor& dout,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::vector<int>& output_padding,
                                  const IntArray& output_size,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  MetaTensor* dx,
                                  MetaTensor* dfilter) {
  GeneralBinaryGradInferMeta(x, filter, dx, dfilter);
}

void Conv2dTransposeDoubleGradInferMeta(const MetaTensor& x,
                                        const MetaTensor& filter,
                                        const MetaTensor& dout,
                                        const MetaTensor& ddx,
                                        const MetaTensor& ddfilter,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& paddings,
                                        const std::vector<int>& output_padding,
                                        const IntArray& output_size,
                                        const std::string& padding_algorithm,
                                        int groups,
                                        const std::vector<int>& dilations,
                                        const std::string& data_format,
                                        MetaTensor* dx,
                                        MetaTensor* dfilter,
                                        MetaTensor* ddout) {
  GeneralBinaryGradInferMeta(x, filter, dx, dfilter);

  if (ddout) {
    ddout->share_meta(dout);
  }
}

void CropGradInferMeta(const MetaTensor& out_grad,
                       const MetaTensor& x,
                       const IntArray& offsets,
                       MetaTensor* x_grad) {
  auto x_dims = x.dims();

  if (x_grad != nullptr) {
    x_grad->set_dims(x_dims);
    x_grad->set_dtype(x.dtype());
  }
}

void FlashAttnGradInferMeta(const MetaTensor& q,
                            const MetaTensor& k,
                            const MetaTensor& v,
                            MetaTensor* dq,
                            MetaTensor* dk,
                            MetaTensor* dv) {
  if (dq) {
    dq->share_meta(q);
  }
  if (dk && k) {
    dk->share_meta(k);
  }
  if (dv && v) {
    dv->share_meta(v);
  }
}

void FusedDropoutAddGradInferMeta(const MetaTensor& seed_offset,
                                  const MetaTensor& out_grad,
                                  MetaTensor* x_grad,
                                  MetaTensor* y_grad) {
  if (x_grad != nullptr) {
    x_grad->share_meta(out_grad);
  }

  if (y_grad != nullptr) {
    y_grad->share_meta(out_grad);
  }
}

void CrossEntropyWithSoftmaxGradInferMeta(const MetaTensor& label,
                                          const MetaTensor& softmax,
                                          const MetaTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          MetaTensor* logits_grad,
                                          MetaConfig config) {
  auto softmax_dims = softmax.dims();
  auto labels_dims = label.dims();
  auto softmax_rank = softmax_dims.size();
  PADDLE_ENFORCE_GE(axis,
                    -softmax_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(Logits)."));
  PADDLE_ENFORCE_LT(axis,
                    softmax_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(Logits)."));

  axis = phi::funcs::CanonicalAxis(axis, softmax_rank);
  for (int i = 0; i < softmax_rank; i++) {
    if (i != axis) {
      if (config.is_runtime || (softmax_dims[i] > 0 && labels_dims[i] > 0)) {
        PADDLE_ENFORCE_EQ(
            softmax_dims[i],
            labels_dims[i],
            phi::errors::InvalidArgument(
                "Input(Logits) and Input(Label) should in same shape in "
                "dimensions except axis."));
      }
    }
  }

  if (soft_label) {
    if (config.is_runtime ||
        (softmax_dims[axis] > 0 && labels_dims[axis] > 0)) {
      PADDLE_ENFORCE_EQ(softmax_dims[axis],
                        labels_dims[axis],
                        phi::errors::InvalidArgument(
                            "If Attr(soft_label) == true, "
                            "the axis dimension of "
                            "Input(X) and Input(Label) should be equal."));
    }
  } else {
    if (config.is_runtime || labels_dims[axis] > 0) {
      PADDLE_ENFORCE_EQ(
          labels_dims[axis],
          1UL,
          phi::errors::InvalidArgument("If Attr(soft_label) == false, "
                                       "the axis dimension of "
                                       "Input(Label) should be 1."));
    }
  }

  logits_grad->set_dims(softmax.dims());
  logits_grad->set_dtype(softmax.dtype());
}

void DeformableConvGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& offset,
                                 const MetaTensor& filter,
                                 const MetaTensor& mask,
                                 const MetaTensor& out_grad,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& dilations,
                                 int deformable_groups,
                                 int groups,
                                 int im2col_step,
                                 MetaTensor* dx,
                                 MetaTensor* offset_grad,
                                 MetaTensor* filter_grad,
                                 MetaTensor* mask_grad) {
  GeneralTernaryGradInferMeta(x, offset, filter, dx, offset_grad, filter_grad);
  if (mask) {
    UnchangedInferMeta(mask, mask_grad);
  }
}

void EigGradInferMeta(const MetaTensor& out_w,
                      const MetaTensor& out_v,
                      const MetaTensor& dout_w,
                      const MetaTensor& dout_v,
                      MetaTensor* dx) {
  auto dims = out_v.dims();

  if (dx) {
    dx->set_dims(dims);
  }
}

void EigvalshGradInferMeta(const MetaTensor& out_v,
                           const MetaTensor& out_w_grad,
                           const std::string& uplo,
                           bool is_test,
                           MetaTensor* x_grad) {
  auto dims = out_v.dims();
  if (x_grad != nullptr) {
    x_grad->set_dims(dims);
    x_grad->set_dtype(out_v.dtype());
  }
}

void EmbeddingGradInferMeta(const MetaTensor& x,
                            const MetaTensor& weight,
                            MetaTensor* out) {
  (void)x;
  if (weight) {
    out->share_dims(weight);
  }
}

void FFTC2RGradInferMeta(const MetaTensor& x,
                         const std::vector<int64_t>& axes,
                         const std::string& normalization,
                         bool forward,
                         int64_t last_dim_size,
                         MetaTensor* out,
                         MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "Output of fft_c2r _grad should not be null."));
  const phi::DDim x_dim = x.dims();

  // only ensure that fft axes' size greater than zero at runtime
  // they might be -1 to indicate unknown size ar compile time
  if (config.is_runtime) {
    for (auto axis : axes) {
      PADDLE_ENFORCE_GT(x_dim[axis],
                        0,
                        phi::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axis]));
    }
  }

  out->set_layout(x.layout());
  out->set_dtype(ToComplexType(x.dtype()));

  phi::DDim out_dim = x.dims();
  const int last_fft_axis = static_cast<int>(axes.back());
  if (last_dim_size > 0) {
    out_dim.at(last_fft_axis) = last_dim_size / 2 + 1;
  } else if (config.is_runtime) {
    const int64_t last_fft_dim_size = x_dim[last_fft_axis];
    out_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
  } else {
    const int64_t last_fft_dim_size = x_dim[last_fft_axis];
    out_dim.at(last_fft_axis) =
        last_fft_dim_size == -1 ? -1 : last_fft_dim_size / 2 + 1;
  }
  out->set_dims(out_dim);
}

void FillDiagonalGradInferMeta(const MetaTensor& dout,
                               float value,
                               int offset,
                               bool wrap,
                               MetaTensor* dx) {
  auto x_dims = dout.dims();
  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(dout.dtype());
  }
}

void FillDiagonalTensorGradInferMeta(const MetaTensor& out_grad,
                                     int64_t offset,
                                     int dim1,
                                     int dim2,
                                     MetaTensor* x_grad) {
  if (x_grad != nullptr) {
    x_grad->set_dims(out_grad.dims());
    x_grad->set_dtype(out_grad.dtype());
  }
}

void GatherNdGradInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& out_grad,
                           MetaTensor* x_grad) {
  const auto& dtype = out_grad.dtype();
  x_grad->set_dims(x.dims());
  x_grad->share_lod(x);
  x_grad->set_dtype(dtype);
}

void GeneralBinaryGradInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                MetaTensor* dx,
                                MetaTensor* dy) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
}

void GeneralTernaryGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 const MetaTensor& z,
                                 MetaTensor* dx,
                                 MetaTensor* dy,
                                 MetaTensor* dz) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
  if (dz) {
    dz->share_meta(z);
  }
}
void GeneralQuaternaryGradInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& z,
                                    const MetaTensor& k,
                                    MetaTensor* dx,
                                    MetaTensor* dy,
                                    MetaTensor* dz,
                                    MetaTensor* dk) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
  if (dz) {
    dz->share_meta(z);
  }
  if (dk) {
    dk->share_meta(k);
  }
}

void GeneralQuinaryGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 const MetaTensor& z,
                                 const MetaTensor& k,
                                 const MetaTensor& l,
                                 MetaTensor* dx,
                                 MetaTensor* dy,
                                 MetaTensor* dz,
                                 MetaTensor* dk,
                                 MetaTensor* dl) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
  if (dz) {
    dz->share_meta(z);
  }
  if (dk) {
    dk->share_meta(k);
  }
  if (dl) {
    dl->share_meta(l);
  }
}

void GeneralUnaryGradInferMeta(const MetaTensor& x, MetaTensor* dx) {
  if (dx) {
    dx->share_meta(x);
  }
}

void GumbelSoftmaxGradInferMeta(const MetaTensor& out,
                                const MetaTensor& dout,
                                int axis,
                                MetaTensor* dx) {
  PADDLE_ENFORCE_EQ(
      out.dims(),
      dout.dims(),
      errors::InvalidArgument(
          "Input(Out) and its gradients should have the same shape."));

  dx->share_meta(dout);
}

void InstanceNormGradInferMeta(const MetaTensor& x,
                               const MetaTensor& scale,
                               const MetaTensor& saved_mean,
                               const MetaTensor& saved_variance,
                               const MetaTensor& y_grad,
                               float epsilon,
                               MetaTensor* x_grad,
                               MetaTensor* scale_grad,
                               MetaTensor* bias_grad) {
  PADDLE_ENFORCE_NE(
      x_grad,
      nullptr,
      phi::errors::InvalidArgument(
          "The X@GRAD in InstanceNormGradInferMeta can't be nullptr."));
  const auto x_dims = x.dims();
  const int C = static_cast<int>(x_dims[1]);
  x_grad->set_dims(x_dims);
  x_grad->set_dtype(x.dtype());
  x_grad->set_layout(x.layout());
  if (scale_grad) {
    scale_grad->set_dims({C});
  }
  if (bias_grad) {
    bias_grad->set_dims({C});
  }
}
void InstanceNormDoubleGradInferMeta(const MetaTensor& x,
                                     const MetaTensor& scale,
                                     const MetaTensor& saved_mean,
                                     const MetaTensor& saved_variance,
                                     const MetaTensor& dy,
                                     const MetaTensor& ddx,
                                     const MetaTensor& ddscale,
                                     const MetaTensor& ddbias,
                                     float epsilon,
                                     MetaTensor* dx,
                                     MetaTensor* dscale,
                                     MetaTensor* ddy) {
  PADDLE_ENFORCE_NE(
      dx,
      nullptr,
      phi::errors::InvalidArgument(
          "The DX in InstanceNormDoubleGradInferMeta can't be nullptr."));
  const auto x_dims = x.dims();
  const int C = static_cast<int>(x_dims[1]);
  dx->set_dims(x_dims);
  dx->set_dtype(x.dtype());
  dx->set_layout(x.layout());
  if (dscale) {
    dscale->set_dims({C});
  }
  if (ddy) {
    ddy->share_dims(x);
  }
}

void InverseGradInferMeta(const MetaTensor& out,
                          const MetaTensor& dout,
                          MetaTensor* dx) {
  if (dx) {
    dx->set_dims(dout.dims());
    dx->set_dtype(out.dtype());
  }
}

void KernelWithXShapeInferMeta(const MetaTensor& xshape, MetaTensor* dx) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());
  dx->set_dims(x_dims);
  dx->share_lod(xshape);
}

void LUGradInferMeta(const MetaTensor& x,
                     const MetaTensor& out,
                     const MetaTensor& pivots,
                     const MetaTensor& out_grad,
                     bool pivot,
                     MetaTensor* x_grad) {
  auto x_dims = x.dims();

  if (x_grad) {
    x_grad->set_dims(x_dims);
    x_grad->set_dtype(x.dtype());
  }
}

void LUUnpackGradInferMeta(const MetaTensor& x,
                           const MetaTensor& pivots,
                           const MetaTensor& l,
                           const MetaTensor& u,
                           const MetaTensor& pmat,
                           const MetaTensor& l_grad,
                           const MetaTensor& u_grad,
                           bool unpack_ludata,
                           bool unpack_pivots,
                           MetaTensor* x_grad) {
  auto x_dims = x.dims();

  if (x_grad) {
    x_grad->set_dims(x_dims);
    x_grad->set_dtype(x.dtype());
  }
}

void MarginCrossEntropyGradInferMeta(const MetaTensor& logits,
                                     const MetaTensor& label,
                                     const MetaTensor& softmax,
                                     const MetaTensor& loss_grad,
                                     bool return_softmax,
                                     int ring_id,
                                     int rank,
                                     int nranks,
                                     float margin1,
                                     float margin2,
                                     float margin3,
                                     float scale,
                                     MetaTensor* logits_grad) {
  PADDLE_ENFORCE_NE(
      logits_grad,
      nullptr,
      phi::errors::InvalidArgument(
          "The Logits@GRAD in MarginCrossEntropy can't be nullptr."));
  auto softmax_dims = softmax.dims();

  logits_grad->set_dims(softmax_dims);
  logits_grad->set_dtype(softmax.dtype());
}

void MaxPoolWithIndexGradInferMeta(const MetaTensor& x,
                                   const MetaTensor& mask,
                                   const MetaTensor& dout,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   bool global_pooling,
                                   bool adaptive,
                                   MetaTensor* dx) {
  dx->share_meta(x);
}

void MemoryEfficientAttentionGradInferMeta(const MetaTensor& query,
                                           const MetaTensor& key,
                                           const MetaTensor& value,
                                           const MetaTensor& bias,
                                           const MetaTensor& cu_seqlens_q,
                                           const MetaTensor& cu_seqlens_k,
                                           const MetaTensor& output,
                                           const MetaTensor& logsumexp,
                                           const MetaTensor& seed_and_offset,
                                           const MetaTensor& output_grad,
                                           const Scalar& max_seqlen_q,
                                           const Scalar& max_seqlen_k,
                                           const bool causal,
                                           const double dropout_p,
                                           const float scale,
                                           MetaTensor* query_grad,
                                           MetaTensor* key_grad,
                                           MetaTensor* value_grad,
                                           MetaTensor* bias_grad) {
  PADDLE_ENFORCE_EQ(
      output_grad.dims().size(),
      4,
      phi::errors::InvalidArgument("Key should be a 4-D tensor"
                                   "But received Key dimension(%s)",
                                   output_grad.dims().size()));
  PADDLE_ENFORCE_EQ(
      output.dims().size(),
      4,
      phi::errors::InvalidArgument("Key should be a 4-D tensor"
                                   "But received Key dimension(%s)",
                                   output_grad.dims().size()));

  const int64_t query_batch_size = query.dims()[0];
  const int64_t query_seq_length = query.dims()[1];
  const int64_t query_num_head = query.dims()[2];
  const int64_t query_head_size = query.dims()[3];

  const int64_t key_batch_size = key.dims()[0];
  const int64_t key_seq_length = key.dims()[1];
  const int64_t key_num_head = key.dims()[2];
  const int64_t key_head_size = key.dims()[3];

  const int64_t value_batch_size = value.dims()[0];
  const int64_t value_seq_length = value.dims()[1];
  const int64_t value_num_head = value.dims()[2];
  const int64_t value_head_size = value.dims()[3];

  std::vector<int64_t> query_grad_dims(
      {query_batch_size, query_seq_length, query_num_head, query_head_size});
  std::vector<int64_t> key_grad_dims(
      {key_batch_size, key_seq_length, key_num_head, key_head_size});
  std::vector<int64_t> value_grad_dims(
      {value_batch_size, value_seq_length, value_num_head, value_head_size});

  query_grad->set_dims(phi::make_ddim(query_grad_dims));
  query_grad->share_lod(query);
  query_grad->set_dtype(query.dtype());
  query_grad->set_layout(query.layout());

  key_grad->set_dims(phi::make_ddim(key_grad_dims));
  key_grad->share_lod(key);
  key_grad->set_dtype(key.dtype());
  key_grad->set_layout(key.layout());

  value_grad->set_dims(phi::make_ddim(value_grad_dims));
  value_grad->share_lod(value);
  value_grad->set_dtype(value.dtype());
  value_grad->set_layout(value.layout());

  if (bias && bias_grad) {
    const int64_t bias_batch_size = bias.dims()[0];
    const int64_t bias_seq_length = bias.dims()[1];
    const int64_t bias_num_head = bias.dims()[2];
    const int64_t bias_head_size = bias.dims()[3];

    std::vector<int64_t> bias_grad_dims(
        {bias_batch_size, bias_seq_length, bias_num_head, bias_head_size});

    bias_grad->set_dims(phi::make_ddim(bias_grad_dims));
    bias_grad->share_lod(bias);
    bias_grad->set_dtype(bias.dtype());
    bias_grad->set_layout(bias.layout());
  }
}

void MeshgridGradInferMeta(const std::vector<const MetaTensor*>& inputs,
                           const std::vector<const MetaTensor*>& outputs_grad,
                           std::vector<MetaTensor*> inputs_grad) {
  PADDLE_ENFORCE_GT(outputs_grad.size(),
                    1,
                    errors::InvalidArgument(
                        "Number of Inputs(Out@Grad) should be larger than 1."
                        "But received Inputs(Out@Grad)' size = %d .",
                        outputs_grad.size()));
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_grad[i]->share_meta(*inputs[i]);
  }
}

void MultiDotGradInferMeta(const std::vector<const MetaTensor*>& x,
                           const MetaTensor& out_grad,
                           std::vector<MetaTensor*> x_grad) {
  PADDLE_ENFORCE_EQ(
      x.size(),
      x_grad.size(),
      errors::InvalidArgument(
          "Number of Inputs(X) should be equal with Outputs(X@Grad)."
          "But received Inputs(X)' size = %d , Outputs(X@Grad)' size = %d.",
          x.size(),
          x_grad.size()));
  for (size_t i = 0; i < x.size(); i++) {
    if (x_grad[i] != nullptr) {
      x_grad[i]->set_dims(x[i]->dims());
      x_grad[i]->share_lod(*x[i]);
    }
  }
}

void MultiplexGradInferMeta(const MetaTensor& ids,
                            const MetaTensor& out_grad,
                            std::vector<MetaTensor*> ins_grad) {
  PADDLE_ENFORCE_NE(
      ins_grad.empty(),
      true,
      errors::InvalidArgument("Output(X@Grad) should not be null."));
  auto dout_dim = out_grad.dims();
  for (auto in_grad : ins_grad) {
    if (in_grad != nullptr) {
      in_grad->set_dims(dout_dim);
    }
  }
}

void NanmedianGradInferMeta(const MetaTensor& x,
                            const MetaTensor& median_index,
                            const MetaTensor& out_grad,
                            const IntArray& axes,
                            bool keep_dim,
                            MetaTensor* x_grad) {
  auto x_dims = x.dims();
  x_grad->set_dims(x_dims);
  x_grad->set_dtype(x.dtype());
}

void NllLossGradInferMeta(const MetaTensor& x,
                          const MetaTensor& label,
                          const MetaTensor& weight,
                          const MetaTensor& total_weight,
                          const MetaTensor& out_grad,
                          int64_t ignore_index,
                          const std::string& reduction,
                          MetaTensor* dx,
                          MetaConfig config) {
  const auto& x_dims = x.dims();
  const auto& label_dims = label.dims();
  const auto& dout_dims = out_grad.dims();
  bool contain_unknown_dim =
      phi::contain_unknown_dim(x_dims) || phi::contain_unknown_dim(dout_dims);
  bool check = config.is_runtime || !contain_unknown_dim;

  if (check) {
    auto batch_size = x_dims[0];
    if (x_dims.size() == 2) {
      if (reduction == "none") {
        PADDLE_ENFORCE_EQ(dout_dims.size(),
                          1,
                          phi::errors::InvalidArgument(
                              "The dimensions of Input(Out@Grad) must be 1"));
        PADDLE_ENFORCE_EQ(
            dout_dims[0],
            batch_size,
            phi::errors::InvalidArgument(
                "The unreduced size ofInput(Out@Grad) must be the "
                "same as batch_size."));
      } else {
        PADDLE_ENFORCE_EQ(dout_dims.size(),
                          0,
                          phi::errors::InvalidArgument(
                              "The dimensions of Input(Out@Grad) must be 0"));
      }
    } else if (x_dims.size() == 4) {
      if (reduction == "none") {
        PADDLE_ENFORCE_EQ(
            dout_dims.size(),
            3,
            phi::errors::InvalidArgument(
                "The dimensions of Input(Out@Grad) must be 3,But got [%s].",
                dout_dims.size()));
        PADDLE_ENFORCE_EQ(dout_dims[0] == label_dims[0] &&
                              dout_dims[1] == label_dims[1] &&
                              dout_dims[2] == label_dims[2],
                          true,
                          phi::errors::InvalidArgument(
                              "The dimensions of Input(Out@Grad) must be match "
                              "to Input(Label) dimensions."));
      } else {
        PADDLE_ENFORCE_EQ(dout_dims.size(),
                          0,
                          phi::errors::InvalidArgument(
                              "The dimensions of Input(Out@Grad) must be 0"));
      }
    }
  }

  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(x.dtype());
  }
}

void OverlapAddGradInferMeta(const MetaTensor& x,
                             const MetaTensor& out_grad,
                             int hop_length,
                             int axis,
                             MetaTensor* x_grad) {
  const auto x_dims = x.dims();
  if (x_grad != nullptr) {
    x_grad->set_dims(x_dims);
    x_grad->set_dtype(x.dtype());
  }
}

void PixelUnshuffleGradInferMeta(const MetaTensor& out_grad,
                                 int downscale_factor,
                                 const std::string& data_format,
                                 MetaTensor* x_grad) {
  auto do_dims = out_grad.dims();
  PADDLE_ENFORCE_EQ(do_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        do_dims.size()));

  const bool channel_last = (data_format == "NHWC");

  auto dx_dims = do_dims;
  dx_dims[0] = do_dims[0];

  if (!channel_last) {
    dx_dims[1] = do_dims[1] / (downscale_factor * downscale_factor);
    dx_dims[2] = do_dims[2] * downscale_factor;
    dx_dims[3] = do_dims[3] * downscale_factor;
  } else {
    dx_dims[1] = do_dims[1] * downscale_factor;
    dx_dims[2] = do_dims[2] * downscale_factor;
    dx_dims[3] = do_dims[3] / (downscale_factor * downscale_factor);
  }
  x_grad->set_dims(dx_dims);
  x_grad->set_dtype(out_grad.dtype());
}

void PreluGradInferMeta(const MetaTensor& x,
                        const MetaTensor& y,
                        MetaTensor* dx,
                        MetaTensor* dy) {
  if (dx) {
    dx->share_dims(x);
  }
  if (dy) {
    dy->share_dims(y);
  }
}

void PsroiPoolGradInferMeta(const MetaTensor& x,
                            const MetaTensor& rois,
                            const MetaTensor& rois_num,
                            const MetaTensor& dout,
                            int pooled_height,
                            int pooled_width,
                            int output_channels,
                            float spatial_scale,
                            MetaTensor* dx) {
  dx->share_meta(x);
}

void RealAndImagGradInferMeta(const MetaTensor& out_grad, MetaTensor* dx) {
  dx->set_dims(out_grad.dims());
  dx->set_dtype(dtype::ToComplex(out_grad.dtype()));
  dx->set_layout(out_grad.layout());
}

void ReshapeDoubleGradInferMeta(const MetaTensor& out_grad,
                                const MetaTensor& x_grad_grad,
                                MetaTensor* out_grad_grad) {
  if (out_grad_grad != nullptr) {
    out_grad_grad->share_dims(out_grad);
  }
}

void RnnGradInferMeta(const MetaTensor& x,
                      const std::vector<const MetaTensor*>& pre_state,
                      const std::vector<const MetaTensor*>& weight_list,
                      MetaTensor* x_grad,
                      std::vector<MetaTensor*> pre_state_grad,
                      std::vector<MetaTensor*> weight_grad_list) {
  PADDLE_ENFORCE_GT(
      pre_state.size(),
      0UL,
      phi::errors::InvalidArgument(
          "The input pre_state in RnnGradInferMeta can't be empty."));
  PADDLE_ENFORCE_GT(
      weight_grad_list.size(),
      0UL,
      phi::errors::InvalidArgument(
          "The input weight_grad_list in RnnGradInferMeta can't be empty."));
  if (x_grad) {
    UnchangedInferMeta(x, x_grad);
  }
  if (!pre_state_grad.empty()) {
    UnchangedMultiInferMeta(pre_state, pre_state_grad);
  }
  if (!weight_grad_list.empty()) {
    UnchangedMultiInferMeta(weight_list, weight_grad_list);
  }
}

void ScatterGradInferMeta(const MetaTensor& index,
                          const MetaTensor& updates,
                          const MetaTensor& out_grad,
                          bool overwrite,
                          MetaTensor* x_grad,
                          MetaTensor* updates_grad) {
  const auto& dtype = out_grad.dtype();
  if (updates_grad) {
    updates_grad->set_dims(updates.dims());
    updates_grad->set_dtype(dtype);
  }

  if (x_grad) {
    x_grad->set_dims(out_grad.dims());
    x_grad->set_dtype(dtype);
  }
}

void ScatterNdAddGradInferMeta(const MetaTensor& index,
                               const MetaTensor& updates,
                               const MetaTensor& out_grad,
                               MetaTensor* x_grad,
                               MetaTensor* updates_grad) {
  const auto& dtype = out_grad.dtype();
  if (updates_grad) {
    updates_grad->set_dims(updates.dims());
    updates_grad->set_dtype(dtype);
  }

  if (x_grad) {
    x_grad->set_dims(out_grad.dims());
    x_grad->set_dtype(dtype);
  }
}

void SpectralNormGradInferMeta(const MetaTensor& weight,
                               const MetaTensor& u,
                               const MetaTensor& v,
                               const MetaTensor& out_grad,
                               int dim,
                               int power_iters,
                               float eps,
                               MetaTensor* weight_grad) {
  auto dim_x = weight.dims();
  if (weight_grad) {
    weight_grad->set_dims(dim_x);
    weight_grad->set_dtype(out_grad.dtype());
  }
}

void StackGradInferMeta(const MetaTensor& out_grad,
                        int axis,
                        std::vector<MetaTensor*> x_grad) {
  auto dy_dim = out_grad.dims();
  int rank = dy_dim.size();
  PADDLE_ENFORCE_GE(
      axis,
      -rank,
      phi::errors::InvalidArgument(
          "Attr(axis) must be inside [-rank, rank), where rank = %d, "
          "but received axis is:%d.",
          rank,
          axis));
  PADDLE_ENFORCE_LT(
      axis,
      rank,
      phi::errors::InvalidArgument(
          "Attr(axis) must be inside [-rank, rank), where rank = %d, "
          "but received axis is:%d.",
          rank,
          axis));

  if (axis < 0) axis += rank;
  PADDLE_ENFORCE_LE(
      x_grad.size(),
      static_cast<size_t>(dy_dim[axis]),
      phi::errors::InvalidArgument(
          "Number of Outputs(X@Grad) should be less than or equal to dy dim "
          "at axis, but received outputs size is:%d, dy dims is:%d.",
          x_grad.size(),
          static_cast<size_t>(dy_dim[axis])));

  auto vec = phi::vectorize<int>(dy_dim);
  vec.erase(vec.begin() + axis);

  for (auto& grad : x_grad) {
    if (grad) {
      grad->set_dims(phi::make_ddim(vec));
      grad->set_dtype(out_grad.dtype());
    }
  }
}

void TransposeGradInferMeta(const MetaTensor& x,
                            const std::vector<int>& axis,
                            MetaTensor* out) {
  size_t x_rank = x.dims().size();
  std::vector<int> formated_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formated_axis[i] = static_cast<int>(axis[i] + x_rank);
    }
  }

  std::vector<int> reversed_axis(axis);
  for (int i = 0; i < static_cast<int>(formated_axis.size()); i++) {
    reversed_axis[formated_axis[i]] = i;
  }

  TransposeInferMeta(x, reversed_axis, out);
}

void TransLayoutGradInferMeta(const MetaTensor& x,
                              const MetaTensor& out_grad,
                              const std::vector<int>& axis,
                              MetaTensor* x_grad) {
  TransposeGradInferMeta(out_grad, axis, x_grad);
  x_grad->set_layout(static_cast<DataLayout>(x.layout()));
}

void UniformRandomInplaceGradInferMeta(const MetaTensor& out_grad,
                                       float min,
                                       float max,
                                       int seed,
                                       int diag_num,
                                       int diag_step,
                                       float diag_val,
                                       MetaTensor* x_grad) {
  PADDLE_ENFORCE_NE(
      x_grad,
      nullptr,
      phi::errors::InvalidArgument(
          "The X@GRAD in UniformRandomInplaceGradInferMeta can't be nullptr."));
  auto dims = out_grad.dims();
  x_grad->set_dims(dims);
  x_grad->set_dtype(out_grad.dtype());
}

void UnStackGradInferMeta(const std::vector<const MetaTensor*>& out_grad,
                          int axis,
                          MetaTensor* x_grad) {
  std::vector<phi::DDim> input_dims(out_grad.size());
  for (size_t i = 0; i < out_grad.size(); ++i) {
    input_dims[i] = out_grad[i]->dims();
  }
  for (size_t i = 1; i < input_dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        input_dims[i],
        input_dims[0],
        phi::errors::InvalidArgument(
            "The dimensions of all Inputs(Y@Grad) must be the same,"
            "but received Inputs(Y@Grad)'s %d-th dimension is %d, "
            "Inputs(Y@Grad)'s 0-th to %d-th dimension is %d.",
            i,
            input_dims[i],
            i - 1,
            input_dims[0]));
  }

  int rank = input_dims[0].size();
  PADDLE_ENFORCE_GE(axis,
                    -(rank + 1),
                    phi::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-(rank+1), rank+1), where rank = %d",
                        rank));
  PADDLE_ENFORCE_LT(axis,
                    rank + 1,
                    phi::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-(rank+1), rank+1), where rank = %d",
                        rank));
  if (axis < 0) axis += (rank + 1);

  auto vec = phi::vectorize<int>(input_dims[0]);
  vec.insert(vec.begin() + axis, static_cast<int>(input_dims.size()));
  x_grad->set_dims(phi::make_ddim(vec));
  x_grad->set_dtype(out_grad[0]->dtype());
}

void WeightOnlyLinearGradInferMeta(const MetaTensor& x,
                                   const MetaTensor& weight,
                                   const MetaTensor& bias,
                                   const MetaTensor& weight_scale,
                                   const MetaTensor& out_grad,
                                   const std::string& weight_dtype,
                                   MetaTensor* x_grad) {
  x_grad->set_dims(x.dims());
  x_grad->set_dtype(x.dtype());
}

void YoloLossGradInferMeta(const MetaTensor& x,
                           const MetaTensor& gt_box,
                           const MetaTensor& gt_label,
                           const MetaTensor& gt_score,
                           const MetaTensor& objectness_mask,
                           const MetaTensor& gt_match_mask,
                           const MetaTensor& loss_grad,
                           const std::vector<int>& anchors,
                           const std::vector<int>& anchor_mask,
                           int class_num,
                           float ignore_thresh,
                           int downsample_ratio,
                           bool use_label_smooth,
                           float scale_x_y,
                           MetaTensor* x_grad,
                           MetaTensor* gt_box_grad,
                           MetaTensor* gt_label_grad,
                           MetaTensor* gt_score_grad) {
  if (x_grad) {
    x_grad->set_dims(x.dims());
    x_grad->set_dtype(x.dtype());
  }
}

void IndexAddGradInferMeta(const MetaTensor& index,
                           const MetaTensor& add_value,
                           const MetaTensor& out_grad,
                           int axis,
                           MetaTensor* x_grad,
                           MetaTensor* add_value_grad) {
  auto do_dims = out_grad.dims();
  auto add_value_dims = add_value.dims();
  if (x_grad) {
    x_grad->set_dims(do_dims);
    x_grad->set_dtype(out_grad.dtype());
    x_grad->set_layout(out_grad.layout());
    x_grad->share_lod(out_grad);
  }
  if (add_value_grad) {
    add_value_grad->set_dims(add_value_dims);
    add_value_grad->set_dtype(add_value.dtype());
    add_value_grad->set_layout(add_value.layout());
    add_value_grad->share_lod(add_value);
  }
}

void IndexPutGradInferMeta(const MetaTensor& x,
                           const std::vector<const MetaTensor*>& indices,
                           const MetaTensor& value,
                           const MetaTensor& out_grad,
                           bool accumulate,
                           MetaTensor* x_grad,
                           MetaTensor* value_grad) {
  if (x_grad) {
    x_grad->share_meta(x);
  }
  if (value_grad) {
    value_grad->share_meta(value);
  }
}

void FusedRopeGradInferMeta(const MetaTensor& sin,
                            const MetaTensor& cos,
                            const MetaTensor& position_ids,
                            const MetaTensor& dout_q,
                            const MetaTensor& dout_k,
                            const MetaTensor& dout_v,
                            bool use_neox_rotary_style,
                            MetaTensor* dq,
                            MetaTensor* dk,
                            MetaTensor* dv) {
  auto input_dims = dout_q.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      phi::errors::InvalidArgument("Input should be a 4-D tensor of format "
                                   "[batch_size, seq_len, num_heads, head_dim],"
                                   "but got %u.",
                                   input_dims.size()));
  if (dout_q) {
    dq->set_dims(dout_q.dims());
    dq->set_dtype(dout_q.dtype());
  }
  if (dout_k) {
    dk->set_dims(dout_k.dims());
    dk->set_dtype(dout_k.dtype());
  }
  if (dout_v) {
    dv->set_dims(dout_v.dims());
    dv->set_dtype(dout_v.dtype());
  }
}

void SetValueGradInferMeta(const MetaTensor& out_grad,
                           const MetaTensor& values,
                           MetaTensor* x_grad,
                           MetaTensor* value_grad) {
  if (x_grad) {
    x_grad->set_dims(out_grad.dims());
    x_grad->set_dtype(out_grad.dtype());
    x_grad->share_lod(out_grad);
  }
  if (value_grad) {
    value_grad->set_dims(values.dims());
    value_grad->set_dtype(values.dtype());
    value_grad->share_lod(values);
  }
}

}  // namespace phi
