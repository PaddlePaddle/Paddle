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

#pragma once

#include <tuple>
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

// Common InferMeta Functions for backward operators.
//
// NOTE: The InferMeta Functions in this file are arranged in alphabetic order.

void AngleGradInferMeta(const MetaTensor& x,
                        const MetaTensor& out_grad,
                        MetaTensor* x_grad);

void BilinearTensorProductGradInferMeta(const MetaTensor& x,
                                        const MetaTensor& y,
                                        const MetaTensor& weight,
                                        const MetaTensor& dout,
                                        MetaTensor* dx,
                                        MetaTensor* dy,
                                        MetaTensor* dweight,
                                        MetaTensor* dbias);

void BmmGradInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      const MetaTensor& out_grad,
                      MetaTensor* x_grad,
                      MetaTensor* y_grad);

void ChannelShuffleGradInferMeta(const MetaTensor& out_grad,
                                 int groups,
                                 const std::string& data_format,
                                 MetaTensor* x_grad);

void ComplexGradInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          const MetaTensor& dout,
                          MetaTensor* dx,
                          MetaTensor* dy);

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
                                MetaTensor* dfilter);

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
                                        MetaTensor* ddout);

void CropTensorGradInferMeta(const MetaTensor& out_grad,
                             const MetaTensor& x,
                             const IntArray& offsets,
                             MetaTensor* x_grad);

void CrossEntropyWithSoftmaxGradInferMeta(const MetaTensor& label,
                                          const MetaTensor& softmax,
                                          const MetaTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          MetaTensor* logits_grad,
                                          MetaConfig config = MetaConfig());

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
                                 MetaTensor* mask_grad);

void EigGradInferMeta(const MetaTensor& out_w,
                      const MetaTensor& out_v,
                      const MetaTensor& dout_w,
                      const MetaTensor& dout_v,
                      MetaTensor* dx);

void GatherNdGradInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& out_grad,
                           MetaTensor* x_grad);

void GeneralUnaryGradInferMeta(const MetaTensor& x, MetaTensor* dx);

void GeneralBinaryGradInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                MetaTensor* dx,
                                MetaTensor* dy);

void GeneralTernaryGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 const MetaTensor& z,
                                 MetaTensor* dx,
                                 MetaTensor* dy,
                                 MetaTensor* dz);

void GeneralQuaternaryGradInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& z,
                                    const MetaTensor& k,
                                    MetaTensor* dx,
                                    MetaTensor* dy,
                                    MetaTensor* dz,
                                    MetaTensor* dk);

void GeneralQuinaryGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 const MetaTensor& z,
                                 const MetaTensor& k,
                                 const MetaTensor& l,
                                 MetaTensor* dx,
                                 MetaTensor* dy,
                                 MetaTensor* dz,
                                 MetaTensor* dk,
                                 MetaTensor* dl);

void GumbelSoftmaxGradInferMeta(const MetaTensor& out,
                                const MetaTensor& dout,
                                int axis,
                                MetaTensor* dx);

void InstanceNormGradInferMeta(const MetaTensor& x,
                               const MetaTensor& scale,
                               const MetaTensor& saved_mean,
                               const MetaTensor& saved_variance,
                               const MetaTensor& y_grad,
                               float epsilon,
                               MetaTensor* x_grad,
                               MetaTensor* scale_grad,
                               MetaTensor* bias_grad);

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
                                     MetaTensor* ddy);

void InverseGradInferMeta(const MetaTensor& out,
                          const MetaTensor& dout,
                          MetaTensor* dx);

void KernelWithXShapeInferMeta(const MetaTensor& xshape, MetaTensor* dx);

void LUGradInferMeta(const MetaTensor& x,
                     const MetaTensor& out,
                     const MetaTensor& pivots,
                     const MetaTensor& out_grad,
                     bool pivot,
                     MetaTensor* x_grad);

void LUUnpackGradInferMeta(const MetaTensor& x,
                           const MetaTensor& pivots,
                           const MetaTensor& l,
                           const MetaTensor& u,
                           const MetaTensor& pmat,
                           const MetaTensor& l_grad,
                           const MetaTensor& u_grad,
                           bool unpack_ludata,
                           bool unpack_pivots,
                           MetaTensor* x_grad);

void MaxPoolWithIndexGradInferMeta(const MetaTensor& x,
                                   const MetaTensor& mask,
                                   const MetaTensor& dout,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   bool global_pooling,
                                   bool adaptive,
                                   MetaTensor* dx);

void MeshgridGradInferMeta(const std::vector<const MetaTensor*>& inputs,
                           const std::vector<const MetaTensor*>& outputs_grad,
                           std::vector<MetaTensor*> inputs_grad);

void MultiDotGradInferMeta(const std::vector<const MetaTensor*>& x,
                           const MetaTensor& out_grad,
                           std::vector<MetaTensor*> x_grad);

void MultiplexGradInferMeta(const MetaTensor& ids,
                            const MetaTensor& out_grad,
                            std::vector<MetaTensor*> ins_grad);

void NanmedianGradInferMeta(const MetaTensor& x,
                            const MetaTensor& median_index,
                            const MetaTensor& out_grad,
                            const IntArray& axes,
                            bool keep_dim,
                            MetaTensor* x_grad);

void NllLossGradInferMeta(const MetaTensor& input,
                          const MetaTensor& label,
                          const MetaTensor& weight,
                          const MetaTensor& total_weight,
                          const MetaTensor& out_grad,
                          int64_t ignore_index,
                          const std::string& reduction,
                          MetaTensor* intput_grad,
                          MetaConfig config = MetaConfig());

void PixelUnshuffleGradInferMeta(const MetaTensor& out_grad,
                                 int downscale_factor,
                                 const std::string& data_format,
                                 MetaTensor* x_grad);

void OverlapAddGradInferMeta(const MetaTensor& x,
                             const MetaTensor& out_grad,
                             int hop_length,
                             int axis,
                             MetaTensor* x_grad);

void PsroiPoolGradInferMeta(const MetaTensor& x,
                            const MetaTensor& rois,
                            const MetaTensor& rois_num,
                            const MetaTensor& dout,
                            int pooled_height,
                            int pooled_width,
                            int output_channels,
                            float spatial_scale,
                            MetaTensor* dx);

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
                       MetaTensor* dx);

void RealAndImagGradInferMeta(const MetaTensor& out_grad, MetaTensor* dx);

void ReshapeDoubleGradInferMeta(const MetaTensor& out_grad,
                                const MetaTensor& x_grad_grad,
                                MetaTensor* out_grad_grad);

void ScatterGradInferMeta(const MetaTensor& index,
                          const MetaTensor& updates,
                          const MetaTensor& out_grad,
                          bool overwrite,
                          MetaTensor* x_grad,
                          MetaTensor* updates_grad);

void ScatterNdAddGradInferMeta(const MetaTensor& index,
                               const MetaTensor& updates,
                               const MetaTensor& out_grad,
                               MetaTensor* x_grad,
                               MetaTensor* updates_grad);

void SpectralNormGradInferMeta(const MetaTensor& weight,
                               const MetaTensor& u,
                               const MetaTensor& v,
                               const MetaTensor& out_grad,
                               int dim,
                               int power_iters,
                               float eps,
                               MetaTensor* weight_grad);

void StackGradInferMeta(const MetaTensor& out_grad,
                        int axis,
                        std::vector<MetaTensor*> x_grad);

void UniformRandomInplaceGradInferMeta(const MetaTensor& out_grad,
                                       float min,
                                       float max,
                                       int seed,
                                       int diag_num,
                                       int diag_step,
                                       float diag_val,
                                       MetaTensor* x_grad);

void UnStackGradInferMeta(const std::vector<const MetaTensor*>& out_grad,
                          int axis,
                          MetaTensor* x_grad);

void Yolov3LossGradInferMeta(const MetaTensor& x,
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
                             MetaTensor* gt_score_grad);

}  // namespace phi
