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

namespace phi {

void BilinearTensorProductGradInferMeta(const MetaTensor& x,
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

void Conv2dTransposeDoubleGradInferMeta(const MetaTensor& x,
                                        const MetaTensor& filter,
                                        const MetaTensor& dout,
                                        const MetaTensor& ddx,
                                        const MetaTensor& ddfilter,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& paddings,
                                        const std::vector<int>& output_padding,
                                        const std::vector<int>& output_size,
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

void KernelWithXShapeInferMeta(const MetaTensor& xshape, MetaTensor* dx) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());
  dx->set_dims(x_dims);
  dx->share_lod(xshape);
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

void NllLossGradInferMeta(const MetaTensor& x,
                          const MetaTensor& label,
                          paddle::optional<const MetaTensor&> weight,
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
      PADDLE_ENFORCE_EQ(dout_dims.size(),
                        1,
                        phi::errors::InvalidArgument(
                            "The dimensions of Input(Out@Grad) must be 1"));
      if (reduction == "none") {
        PADDLE_ENFORCE_EQ(
            dout_dims[0],
            batch_size,
            phi::errors::InvalidArgument(
                "The unreduced size ofInput(Out@Grad) must be the "
                "same as batch_size."));
      } else {
        PADDLE_ENFORCE_EQ(dout_dims[0],
                          1,
                          phi::errors::InvalidArgument(
                              "The reduced size of Input(Out@Grad) must be 1"));
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
        PADDLE_ENFORCE_EQ(dout_dims[0],
                          1,
                          phi::errors::InvalidArgument(
                              "The reduced size of Input(Out@Grad) must be 1"));
      }
    }
  }

  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(x.dtype());
  }
}

void PoolGradInferMeta(const MetaTensor& x,
                       const MetaTensor& out,
                       const MetaTensor& dout,
                       const std::vector<int>& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool ceil_mode,
                       bool exclusive,
                       const std::string& data_format,
                       const std::string& pooling_type,
                       bool global_pooling,
                       bool adaptive,
                       const std::string& padding_algorithm,
                       MetaTensor* dx) {
  dx->share_meta(x);
}

void PsroiPoolGradInferMeta(const MetaTensor& x,
                            const MetaTensor& rois,
                            paddle::optional<const MetaTensor&> rois_num,
                            const MetaTensor& dout,
                            int pooled_height,
                            int pooled_width,
                            int output_channels,
                            float spatial_scale,
                            MetaTensor* dx) {
  dx->share_meta(x);
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

}  // namespace phi
