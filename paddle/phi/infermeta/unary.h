/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

struct MetaConfig;

// Common InferMeta Functions for unary operators, The format like:
//
//   PADDLE_API void [FunctionDesc|OpName]InferMeta(const MetaTensor& x, ...,
//   MetaTensor* out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
// Because functions in this file not only can infer shape, but also need
// infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

PADDLE_API void AddPositionEncodingInferMeta(const MetaTensor& x,
                                             float alpha,
                                             float beta,
                                             MetaTensor* out);

PADDLE_API void AffineGridInferMeta(const MetaTensor& input,
                                    const IntArray& outputShape,
                                    bool align_corners,
                                    MetaTensor* output);

PADDLE_API void AllGatherInferMeta(const MetaTensor& x,
                                   int nranks,
                                   MetaTensor* out);

PADDLE_API void AllReduceInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void AllToAllInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void AnchorGeneratorInferMeta(
    const MetaTensor& input,
    const std::vector<float>& anchor_sizes,
    const std::vector<float>& aspect_ratios,
    const std::vector<float>& variances,
    const std::vector<float>& stride,
    float offset,
    MetaTensor* anchors,
    MetaTensor* variances_out);

PADDLE_API void ArgMinMaxInferMeta(const MetaTensor& x,
                                   const Scalar& axis,
                                   bool keepdims,
                                   bool flatten,
                                   DataType dtype,
                                   MetaTensor* out,
                                   MetaConfig config = MetaConfig());

PADDLE_API void MinMaxWithIndexInferMeta(const MetaTensor& x,
                                         const Scalar& axis,
                                         bool keepdims,
                                         bool flatten,
                                         MetaTensor* val_out,
                                         MetaTensor* ind_out,
                                         MetaConfig config = MetaConfig());

PADDLE_API void ArgsortInferMeta(const MetaTensor& input,
                                 int axis,
                                 bool descending,
                                 bool stable,
                                 MetaTensor* output,
                                 MetaTensor* indices);

PADDLE_API void ArrayLengthInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void ArrayToTensorInferMeta(const MetaTensor& x,
                                       int axis,
                                       bool use_stack,
                                       MetaTensor* out,
                                       MetaTensor* out_index,
                                       MetaConfig config = MetaConfig());

PADDLE_API void BipartiteMatchInferMeta(const MetaTensor& dist_mat,
                                        const std::string& match_type,
                                        float dist_threshold,
                                        MetaTensor* col_to_row_match_indices,
                                        MetaTensor* col_to_row_match_dist);

PADDLE_API void TensorToArrayInferMeta(const MetaTensor& x,
                                       const MetaTensor& out_grad,
                                       int axis,
                                       bool use_stack,
                                       MetaTensor* x_grad);

PADDLE_API void AsRealInferMeta(const MetaTensor& input, MetaTensor* output);

PADDLE_API void AsComplexInferMeta(const MetaTensor& input, MetaTensor* output);

PADDLE_API void BarrierInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void BatchSizeLikeInferMeta(const MetaTensor& x,
                                       const std::vector<int>& shape,
                                       int x_batch_size_dim,
                                       int out_batch_size_dim,
                                       MetaTensor* out);

PADDLE_API void CastInferMeta(const MetaTensor& x,
                              DataType out_dtype,
                              MetaTensor* out);

PADDLE_API void CConcatInferMeta(const MetaTensor& x,
                                 int nranks,
                                 MetaTensor* out);

PADDLE_API void ChannelShuffleInferMeta(const MetaTensor& x,
                                        int groups,
                                        const std::string& data_format,
                                        MetaTensor* out);

PADDLE_API void CheckNumericsInferMeta(const MetaTensor& tensor,
                                       const std::string& op_type,
                                       const std::string& var_name,
                                       const int check_nan_inf_level,
                                       const int stack_height_limit,
                                       const std::string& output_dir,
                                       MetaTensor* stats,
                                       MetaTensor* values);

PADDLE_API void CholeskyInferMeta(const MetaTensor& x,
                                  bool upper,
                                  MetaTensor* out);

PADDLE_API void CINNBroadcastInferMeta(const MetaTensor& x,
                                       const std::vector<int64_t>& axes,
                                       const std::vector<int64_t>& out_shape,
                                       MetaTensor* output);

PADDLE_API void ClassCenterSampleInferMeta(
    const MetaTensor& label,
    int num_classes,
    int num_samples,
    int ring_id,
    int rank,
    int nranks,
    bool fix_seed,
    int seed,
    MetaTensor* remapped_label,
    MetaTensor* sampled_local_class_center);

PADDLE_API void ClipByNormInferMeta(const MetaTensor& x,
                                    float max_norm,
                                    MetaTensor* out);

PADDLE_API void CIdentityInferMeta(const MetaTensor& x,
                                   int ring_id,
                                   bool use_calc_stream,
                                   bool use_model_parallel,
                                   MetaTensor* out);

PADDLE_API void CreateLikeInferMeta(const MetaTensor& x,
                                    DataType dtype,
                                    MetaTensor* out);

PADDLE_API void CreateArrayLikeInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void CropInferMeta(const MetaTensor& x,
                              const IntArray& shape,
                              const IntArray& offsets,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

PADDLE_API void CScatterInferMeta(
    const MetaTensor& x, int ring_id, int root, int nranks, MetaTensor* out);

PADDLE_API void CSplitInferMeta(const MetaTensor& x,
                                int nranks,
                                MetaTensor* out);

PADDLE_API void CumInferMeta(const MetaTensor& x,
                             int axis,
                             bool flatten,
                             bool exclusive,
                             bool reverse,
                             MetaTensor* out);

PADDLE_API void CumScalarAxisInferMeta(const MetaTensor& x,
                                       const Scalar& axis,
                                       bool flatten,
                                       bool exclusive,
                                       bool reverse,
                                       MetaTensor* out);

PADDLE_API void CumWithIndicesInferMeta(const MetaTensor& x,
                                        int axis,
                                        DataType dtype,
                                        MetaTensor* out,
                                        MetaTensor* indices);

PADDLE_API void DecodeJpegInferMeta(const MetaTensor& x,
                                    const std::string& mode,
                                    MetaTensor* out);

PADDLE_API void DeQuantizeXPUInferMeta(const MetaTensor& x,
                                       DataType out_dtype,
                                       float scale,
                                       MetaTensor* y);

PADDLE_API void DiagEmbedInferMeta(
    const MetaTensor& x, int offset, int dim1, int dim2, MetaTensor* out);

PADDLE_API void DiagInferMeta(const MetaTensor& x,
                              int offset,
                              float padding_value,
                              MetaTensor* out);

PADDLE_API void DiagonalInferMeta(
    const MetaTensor& input, int offset, int axis1, int axis2, MetaTensor* out);

PADDLE_API void DirichletInferMeta(const MetaTensor& alpha, MetaTensor* out);

PADDLE_API void DistBroadcastInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void DistConcatInferMeta(const MetaTensor& x,
                                    int nranks,
                                    MetaTensor* out);

PADDLE_API void DistReduceInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void EmbeddingGradSparseInferMeta(const MetaTensor& x,
                                             const MetaTensor& weight,
                                             MetaTensor* out);

PADDLE_API void EigInferMeta(const MetaTensor& x,
                             MetaTensor* out_w,
                             MetaTensor* out_v);

PADDLE_API void EighInferMeta(const MetaTensor& x,
                              const std::string& uplo,
                              MetaTensor* out_w,
                              MetaTensor* out_v);

PADDLE_API void EigvalsInferMeta(const MetaTensor& x,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());

PADDLE_API void EigvalshInferMeta(const MetaTensor& x,
                                  const std::string& uplo,
                                  bool is_test,
                                  MetaTensor* out_w,
                                  MetaTensor* out_v);

PADDLE_API void EinsumInferMeta(const std::vector<const MetaTensor*>& inputs,
                                const std::string& equation,
                                MetaTensor* out);

PADDLE_API void EinsumRawInferMeta(const std::vector<const MetaTensor*>& inputs,
                                   const std::string& equation,
                                   MetaTensor* out,
                                   std::vector<MetaTensor*> inner_cache,
                                   std::vector<MetaTensor*> xshape);

PADDLE_API void ExpandInferMeta(const MetaTensor& x,
                                const IntArray& shape,
                                MetaTensor* out);

PADDLE_API void ExpandModalityExpertIdInferMeta(const MetaTensor& expert_id,
                                                int64_t num_expert_per_modality,
                                                int64_t group_size,
                                                int64_t modality_offset,
                                                bool is_group_expert,
                                                MetaTensor* expert_id_out);

PADDLE_API void FakeChannelWiseQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                                       int bit_length,
                                                       int round_type,
                                                       int quant_axis,
                                                       bool is_test,
                                                       MetaTensor* out,
                                                       MetaTensor* out_scale);

PADDLE_API void FakeChannelWiseQuantizeDequantizeAbsMaxInferMeta(
    const MetaTensor& x,
    int bit_length,
    int round_type,
    int quant_axis,
    MetaTensor* out,
    MetaTensor* out_scale);

PADDLE_API void FakeQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                            int bit_length,
                                            int round_type,
                                            MetaTensor* out,
                                            MetaTensor* out_scale);

PADDLE_API void FetchBarrierInferMeta(const std::vector<const MetaTensor*>& x,
                                      int trainer_id,
                                      const std::vector<std::string>& endpoints,
                                      std::vector<MetaTensor*> out);

PADDLE_API void FillAnyLikeInferMeta(const MetaTensor& x,
                                     const Scalar& value,
                                     DataType dtype,
                                     MetaTensor* out);

PADDLE_API void FillDiagonalInferMeta(
    const MetaTensor& x, float value, int offset, bool wrap, MetaTensor* out);

PADDLE_API void FFTC2CInferMeta(const MetaTensor& x,
                                const std::vector<int64_t>& axes,
                                const std::string& normalization,
                                bool forward,
                                MetaTensor* out,
                                MetaConfig = MetaConfig());

PADDLE_API void FFTC2RInferMeta(const MetaTensor& x,
                                const std::vector<int64_t>& axes,
                                const std::string& normalization,
                                bool forward,
                                int64_t last_dim_size,
                                MetaTensor* out,
                                MetaConfig = MetaConfig());

PADDLE_API void FFTR2CInferMeta(const MetaTensor& x,
                                const std::vector<int64_t>& axes,
                                const std::string& normalization,
                                bool forward,
                                bool onesided,
                                MetaTensor* out,
                                MetaConfig = MetaConfig());

PADDLE_API void FlattenInferMeta(const MetaTensor& x,
                                 int start_axis,
                                 int stop_axis,
                                 MetaTensor* out);

PADDLE_API void Flatten2InferMeta(const MetaTensor& x,
                                  int axis,
                                  MetaTensor* out,
                                  MetaTensor* x_shape);

PADDLE_API void FlattenWithXShapeInferMeta(const MetaTensor& x,
                                           int start_axis,
                                           int stop_axis,
                                           MetaTensor* out,
                                           MetaTensor* xshape);

PADDLE_API void FlipInferMeta(const MetaTensor& x,
                              const std::vector<int>& axis,
                              MetaTensor* out);

PADDLE_API void FoldInferMeta(const MetaTensor& x,
                              const std::vector<int>& output_sizes,
                              const std::vector<int>& kernel_sizes,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              MetaTensor* out);

PADDLE_API void FractionalMaxPoolInferMeta(const MetaTensor& x,
                                           const std::vector<int>& output_size,
                                           const std::vector<int>& kernel_size,
                                           float random_u,
                                           bool return_mask,
                                           MetaTensor* out,
                                           MetaTensor* mask,
                                           MetaConfig config = MetaConfig());

PADDLE_API void FrameInferMeta(const MetaTensor& x,
                               int frame_length,
                               int hop_length,
                               int axis,
                               MetaTensor* out,
                               MetaConfig = MetaConfig());

PADDLE_API void Fp8QuantBlockwiseInferMeta(const MetaTensor& X,
                                           float epsilon,
                                           bool using_1x128_vec_quant,
                                           bool input_transpose,
                                           bool output_scale_transpose,
                                           bool return_transpose_only,
                                           bool using_e5m2,
                                           bool using_pow2_scale,
                                           MetaTensor* out,
                                           MetaTensor* scale,
                                           MetaTensor* out_transposed,
                                           MetaTensor* scale_transposed);

PADDLE_API void FullBatchSizeLikeInferMeta(const MetaTensor& x,
                                           const std::vector<int>& shape,
                                           const Scalar& val,
                                           DataType dtype,
                                           int x_batch_size_dim,
                                           int out_batch_size_dim,
                                           MetaTensor* out);

PADDLE_API void GumbelSoftmaxInferMeta(const MetaTensor& x,
                                       float temperature,
                                       bool hard,
                                       int axis,
                                       MetaTensor* out);

PADDLE_API void HashInferMeta(const MetaTensor& x,
                              int num_hash,
                              int64_t mod_by,
                              MetaTensor* out);

PADDLE_API void IdentityLossInferMeta(const MetaTensor& x,
                                      int reduction,
                                      MetaTensor* out);

PADDLE_API void IncrementInferMeta(const MetaTensor& x,
                                   float value,
                                   MetaTensor* out);

PADDLE_API void InferMetaFromVecValue(const MetaTensor& x,
                                      const std::vector<int64_t>& shape,
                                      MetaTensor* out);

PADDLE_API void InverseInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void IsEmptyInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void IsfiniteInferMeta(const MetaTensor& input, MetaTensor* out);

PADDLE_API void KthvalueInferMeta(const MetaTensor& x,
                                  int64_t k,
                                  int axis,
                                  bool keepdim,
                                  MetaTensor* out,
                                  MetaTensor* indices,
                                  MetaConfig = MetaConfig());

PADDLE_API void LogicalNotInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void LogsumexpInferMeta(const MetaTensor& input,
                                   const std::vector<int>& axis,
                                   bool keepdim,
                                   bool reduce_all,
                                   MetaTensor* out);

PADDLE_API void LUInferMeta(const MetaTensor& x,
                            bool pivot,
                            MetaTensor* out,
                            MetaTensor* pivots,
                            MetaTensor* infos);

PADDLE_API void MatrixPowerInferMeta(const MetaTensor& x,
                                     int n,
                                     MetaTensor* out);

PADDLE_API void MatrixRankInferMeta(const MetaTensor& x,
                                    bool use_default_tol,
                                    bool hermitian,
                                    MetaTensor* out);

PADDLE_API void MaxOutInferMeta(const MetaTensor& x,
                                int groups,
                                int axis,
                                MetaTensor* out);

PADDLE_API void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                                          const std::vector<int>& kernel_size,
                                          const std::vector<int>& strides,
                                          const std::vector<int>& paddings,
                                          bool global_pooling,
                                          bool adaptive,
                                          bool ceil_mode,
                                          MetaTensor* out,
                                          MetaTensor* mask,
                                          MetaConfig config = MetaConfig());

PADDLE_API void MaxPoolV2InferMeta(const MetaTensor& x,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   const std::string& data_format,
                                   bool global_pooling,
                                   bool adaptive,
                                   MetaTensor* out,
                                   MetaTensor* saved_idx,
                                   MetaConfig config = MetaConfig());

PADDLE_API void MeanAllInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void MedianInferMeta(const MetaTensor& x,
                                const IntArray& axes,
                                bool keep_dim,
                                const std::string& mode,
                                MetaTensor* out,
                                MetaTensor* median_index);

PADDLE_API void ModeInferMeta(const MetaTensor& x,
                              int axis,
                              bool keepdim,
                              MetaTensor* out,
                              MetaTensor* indices);

PADDLE_API void MultinomialInferMeta(const MetaTensor& x,
                                     const Scalar& num_samples,
                                     bool replacement,
                                     MetaTensor* out,
                                     MetaConfig config = MetaConfig());

PADDLE_API void NanmedianInferMeta(const MetaTensor& x,
                                   const IntArray& axes,
                                   bool keep_dim,
                                   const std::string& mode,
                                   MetaTensor* out,
                                   MetaTensor* median_index);

PADDLE_API void NonZeroInferMeta(const MetaTensor& condition, MetaTensor* out);

PADDLE_API void NMSInferMeta(const MetaTensor& x,
                             float threshold,
                             MetaTensor* out);

PADDLE_API void NormInferMeta(const MetaTensor& x,
                              int axis,
                              float epsilon,
                              bool is_test,
                              MetaTensor* out,
                              MetaTensor* norm);

PADDLE_API void OneHotRawInferMeta(const MetaTensor& x,
                                   const Scalar& depth,
                                   DataType dtype,
                                   bool allow_out_of_range,
                                   MetaTensor* out);

PADDLE_API void OneHotInferMeta(const MetaTensor& x,
                                const Scalar& depth,
                                MetaTensor* out);

PADDLE_API void OverlapAddInferMeta(const MetaTensor& x,
                                    int hop_length,
                                    int axis,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

PADDLE_API void PadInferMeta(const MetaTensor& input,
                             const std::vector<int>& paddings,
                             const Scalar& padding_value,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

PADDLE_API void Pad3dInferMeta(const MetaTensor& x,
                               const IntArray& paddings,
                               const std::string& mode,
                               float value,
                               const std::string& data_format,
                               MetaTensor* out,
                               MetaConfig config = MetaConfig());

PADDLE_API void PartialAllgatherInferMeta(const MetaTensor& x,
                                          int nranks,
                                          int rank,
                                          MetaTensor* out);

PADDLE_API void PartialSendInferMeta(const MetaTensor& x,
                                     int peer,
                                     int num,
                                     int id);

PADDLE_API void PixelShuffleInferMeta(const MetaTensor& x,
                                      int upscale_factor,
                                      const std::string& data_format,
                                      MetaTensor* out);

PADDLE_API void PixelShuffleGradInferMeta(const MetaTensor& out_grad,
                                          int upscale_factor,
                                          const std::string& data_format,
                                          MetaTensor* x_grad);

PADDLE_API void PixelUnshuffleInferMeta(const MetaTensor& x,
                                        int downscale_factor,
                                        const std::string& data_format,
                                        MetaTensor* out);

PADDLE_API void PNormInferMeta(const MetaTensor& x,
                               float porder,
                               int axis,
                               float epsilon,
                               bool keepdim,
                               bool asvector,
                               MetaTensor* out);

PADDLE_API void PoolInferMeta(const MetaTensor& x,
                              const std::vector<int64_t>& kernel_size,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& paddings,
                              bool ceil_mode,
                              bool exclusive,
                              const std::string& data_format,
                              const std::string& pooling_type,
                              bool global_pooling,
                              bool adaptive,
                              const std::string& padding_algorithm,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

PADDLE_API void Pool2DInferMeta(const MetaTensor& x,
                                const IntArray& kernel_size,
                                const std::vector<int64_t>& strides,
                                const std::vector<int64_t>& paddings,
                                bool ceil_mode,
                                bool exclusive,
                                const std::string& data_format,
                                const std::string& pooling_type,
                                bool global_pooling,
                                bool adaptive,
                                const std::string& padding_algorithm,
                                MetaTensor* out,
                                MetaConfig config = MetaConfig());

PADDLE_API void PSendInferMeta(const MetaTensor& x, int peer);

PADDLE_API void PSendArrayInferMeta(const MetaTensor& x, int peer);

PADDLE_API void PushDenseInferMeta(const std::vector<const MetaTensor*>& ids,
                                   int table_id,
                                   float scale_data_norm,
                                   const std::vector<std::string>& input_names);

PADDLE_API void SendV2InferMeta(const int peer, const int ring_id);

PADDLE_API void QrInferMeta(const MetaTensor& x,
                            const std::string& mode,
                            MetaTensor* q,
                            MetaTensor* r);

PADDLE_API void QuantizeXPUInferMeta(const MetaTensor& x,
                                     DataType out_dtype,
                                     float scale,
                                     MetaTensor* y);

PADDLE_API void WeightQuantizeInferMeta(const MetaTensor& x,
                                        const std::string& algo,
                                        const int32_t arch,
                                        const int32_t group_size,
                                        MetaTensor* out,
                                        MetaTensor* scale);

PADDLE_API void RealAndImagInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void ReduceSumInferMeta(const MetaTensor& x,
                                   const std::vector<int64_t>& axis,
                                   bool keep_dim,
                                   DataType dtype,
                                   MetaTensor* out);

PADDLE_API void ReduceInferMeta(const MetaTensor& x,
                                const std::vector<int64_t>& axis,
                                bool keep_dim,
                                MetaTensor* out);

PADDLE_API void ReduceInferMetaBase(const MetaTensor& x,
                                    const std::vector<int64_t>& axis,
                                    bool keep_dim,
                                    bool reduce_all,
                                    MetaTensor* out);

PADDLE_API void ReduceIntArrayAxisInferMetaBase(
    const MetaTensor& x,
    const IntArray& axis,
    bool keep_dim,
    bool reduce_all,
    MetaTensor* out,
    MetaConfig config = MetaConfig());

PADDLE_API void ReduceIntArrayAxisInferMeta(const MetaTensor& x,
                                            const IntArray& axis,
                                            bool keep_dim,
                                            MetaTensor* out,
                                            MetaConfig config = MetaConfig());

PADDLE_API void StrictReduceIntArrayAxisInferMetaBase(
    const MetaTensor& x,
    const IntArray& axis,
    bool keep_dim,
    bool reduce_all,
    MetaTensor* out,
    MetaConfig config = MetaConfig());

PADDLE_API void StrictReduceIntArrayAxisInferMeta(
    const MetaTensor& x,
    const IntArray& axis,
    bool keep_dim,
    MetaTensor* out,
    MetaConfig config = MetaConfig());

PADDLE_API void ReduceScatterInferMeta(const MetaTensor& x,
                                       int nranks,
                                       MetaTensor* out);

PADDLE_API void RepeatInterleaveInferMeta(const MetaTensor& x,
                                          int repeats,
                                          int dim,
                                          int64_t output_size,
                                          MetaTensor* out);

PADDLE_API void ReshapeInferMeta(const MetaTensor& x,
                                 const IntArray& shape,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());
PADDLE_API void ViewShapeInferMeta(const MetaTensor& input,
                                   const std::vector<int64_t>& shape,
                                   MetaTensor* out);

PADDLE_API void ReshapeWithXShapeInferMeta(const MetaTensor& x,
                                           const IntArray& shape,
                                           MetaTensor* out,
                                           MetaTensor* xshape,
                                           MetaConfig config = MetaConfig());

PADDLE_API void ReverseInferMeta(const MetaTensor& x,
                                 const IntArray& axis,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());

PADDLE_API void ReverseArrayInferMeta(
    const std::vector<const phi::MetaTensor*>& x,
    const IntArray& axis,
    std::vector<phi::MetaTensor*> out,
    MetaConfig config = MetaConfig());

PADDLE_API void RollInferMeta(const MetaTensor& x,
                              const IntArray& shifts,
                              const std::vector<int64_t>& axis,
                              MetaTensor* out);

PADDLE_API void RReluInferMeta(const MetaTensor& x,
                               float lower,
                               float upper,
                               bool is_test,
                               MetaTensor* out,
                               MetaTensor* noise);

PADDLE_API void RReluGradInferMeta(const MetaTensor& out_grad,
                                   const MetaTensor& noise,
                                   MetaTensor* x_grad);

PADDLE_API void RestrictNonZeroInferMeta(const MetaTensor& condition,
                                         int64_t total_true_num,
                                         MetaTensor* out);

PADDLE_API void SequenceMaskScalarInferMeta(const MetaTensor& x,
                                            const Scalar& max_len,
                                            DataType out_dtype,
                                            MetaTensor* y);

PADDLE_API void SequencePoolInferMeta(const MetaTensor& x,
                                      bool is_test,
                                      const std::string& pooltype,
                                      float pad_value,
                                      MetaTensor* out,
                                      MetaTensor* max_index,
                                      MetaConfig config = MetaConfig());

PADDLE_API void SetValueInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void ShareDataInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void ShapeInferMeta(const MetaTensor& input, MetaTensor* out);

PADDLE_API void Shape64InferMeta(const MetaTensor& input,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());

PADDLE_API void ShardIndexInferMeta(const MetaTensor& in,
                                    int index_num,
                                    int nshards,
                                    int shard_id,
                                    int ignore_value,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

PADDLE_API void NumelInferMeta(const MetaTensor& input, MetaTensor* out);

PADDLE_API void ShuffleChannelInferMeta(const MetaTensor& x,
                                        int group,
                                        MetaTensor* out);

PADDLE_API void SliceArrayInferMeta(const MetaTensor& input,
                                    const IntArray& starts,
                                    const IntArray& ends,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

PADDLE_API void SliceArrayDenseInferMeta(const MetaTensor& input,
                                         const IntArray& starts,
                                         MetaTensor* out,
                                         MetaConfig config = MetaConfig());

PADDLE_API void SliceRawInferMeta(const MetaTensor& input,
                                  const std::vector<int64_t>& axes,
                                  const IntArray& starts,
                                  const IntArray& ends,
                                  const std::vector<int64_t>& infer_flags,
                                  const std::vector<int64_t>& decrease_axis,
                                  MetaTensor* out,
                                  MetaConfig config = MetaConfig());

PADDLE_API void ViewSliceInferMeta(const MetaTensor& input,
                                   int64_t begin_idx,
                                   int64_t end_idx,
                                   MetaTensor* out);

PADDLE_API void SoftmaxInferMeta(const MetaTensor& x,
                                 int axis,
                                 MetaTensor* out);

int GetSplitAxisValue(const MetaTensor& x,
                      const Scalar& axis,
                      MetaConfig config);

PADDLE_API void FillSplitOutDims(const MetaTensor& x,
                                 const int axis_value,
                                 const std::vector<int64_t>& sections_vec,
                                 std::vector<MetaTensor*>* out);

PADDLE_API void SetInferMeta(const MetaTensor& x,
                             const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& stride,
                             MetaTensor* out);

PADDLE_API void SequenceSoftmaxInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void SplitInferMeta(const MetaTensor& x_meta,
                               const IntArray& sections,
                               const Scalar& axis,
                               std::vector<MetaTensor*> out,
                               MetaConfig config = MetaConfig());

PADDLE_API void SplitWithNumInferMeta(const MetaTensor& x_meta,
                                      int num,
                                      const Scalar& axis,
                                      std::vector<MetaTensor*> out,
                                      MetaConfig config = MetaConfig());

PADDLE_API void SquaredL2NormInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void L1NormInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void SqueezeInferMeta(const MetaTensor& x,
                                 const IntArray& axes,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());

PADDLE_API void SqueezeWithXShapeInferMeta(const MetaTensor& x,
                                           const IntArray& axes,
                                           MetaTensor* out,
                                           MetaTensor* xshape,
                                           MetaConfig config = MetaConfig());

PADDLE_API void StridedSliceRawInferMeta(const MetaTensor& x,
                                         const std::vector<int>& axes,
                                         const IntArray& starts,
                                         const IntArray& ends,
                                         const IntArray& strides,
                                         const std::vector<int>& infer_flags,
                                         const std::vector<int>& decrease_axis,
                                         MetaTensor* out,
                                         MetaConfig config = MetaConfig());

PADDLE_API void StridedSliceInferMeta(const MetaTensor& x,
                                      const std::vector<int>& axes,
                                      const IntArray& starts,
                                      const IntArray& ends,
                                      const IntArray& strides,
                                      MetaTensor* out,
                                      MetaConfig config = MetaConfig());

PADDLE_API void SumInferMeta(const MetaTensor& x,
                             const IntArray& axis,
                             DataType dtype,
                             bool keep_dim,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

PADDLE_API void DetInferMeta(const MetaTensor& x,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

PADDLE_API void SumRawInferMeta(const MetaTensor& x,
                                const IntArray& axis,
                                bool keep_dim,
                                bool reduce_all,
                                DataType dtype,
                                MetaTensor* out,
                                MetaConfig config = MetaConfig());

PADDLE_API void PartialConcatInferMeta(const std::vector<const MetaTensor*>& xs,
                                       int start_index,
                                       int length,
                                       MetaTensor* out,
                                       MetaConfig config = MetaConfig());

PADDLE_API void PartialSumInferMeta(const std::vector<const MetaTensor*>& xs,
                                    int start_index,
                                    int length,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

PADDLE_API void SvdvalsInferMeta(const MetaTensor& x, MetaTensor* s);

PADDLE_API void SvdInferMeta(const MetaTensor& x,
                             bool full_matrices,
                             MetaTensor* u,
                             MetaTensor* s,
                             MetaTensor* vh);

PADDLE_API void TemporalShiftInferMeta(const MetaTensor& x,
                                       int seg_num,
                                       float shift_ratio,
                                       const std::string& data_format,
                                       MetaTensor* out,
                                       MetaConfig config = MetaConfig());

PADDLE_API void TileInferMeta(const MetaTensor& x,
                              const IntArray& repeat_times,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

PADDLE_API void TopKInferMeta(const MetaTensor& x,
                              const Scalar& k_scalar,
                              int axis,
                              bool largest,
                              bool sorted,
                              MetaTensor* out,
                              MetaTensor* indices,
                              MetaConfig config = MetaConfig());

PADDLE_API void TopkV1InferMeta(const MetaTensor& x,
                                const Scalar& k_scalar,
                                MetaTensor* out,
                                MetaTensor* indices,
                                MetaConfig config = MetaConfig());

PADDLE_API void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out);

PADDLE_API void TransferLayoutInferMeta(const MetaTensor& x,
                                        int src_layout,
                                        int dst_layout,
                                        MetaTensor* out);

PADDLE_API void TransposeInferMeta(const MetaTensor& x,
                                   const std::vector<int>& axis,
                                   MetaTensor* out);

PADDLE_API void TransposeGradInferMeta(const MetaTensor& x,
                                       const std::vector<int>& axis,
                                       MetaTensor* out);

PADDLE_API void TrilInferMeta(const MetaTensor& x,
                              int diagonal,
                              MetaTensor* out);

PADDLE_API void TriuInferMeta(const MetaTensor& x,
                              int diagonal,
                              MetaTensor* out);

PADDLE_API void TrilTriuInferMeta(const MetaTensor& x,
                                  int diagonal,
                                  bool lower,
                                  MetaTensor* out);

PADDLE_API void UnbindInferMeta(const MetaTensor& x,
                                int axis,
                                std::vector<MetaTensor*> outs);

PADDLE_API void UnchangedExceptLayoutInferMeta(const MetaTensor& x,
                                               MetaTensor* out);
PADDLE_API void UnchangedExceptDtypeInferMeta(const MetaTensor& x,
                                              MetaTensor* out);
PADDLE_API void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out);
PADDLE_API void UnchangedArrayInferMeta(const MetaTensor& x, MetaTensor* out);
PADDLE_API void UnchangedInferMetaIncludingTensorArray(const MetaTensor& x,
                                                       MetaTensor* out);
PADDLE_API void UnchangedVectorInferMeta(
    const std::vector<const MetaTensor*>& xs, std::vector<MetaTensor*> outs);

// meta x -> out without change, check if axis in range [-Rank(x), Rank(x)-1]
PADDLE_API void UnchangedInferMetaCheckAxis(const MetaTensor& x,
                                            int axis,
                                            MetaTensor* out);

PADDLE_API void UnfoldInferMeta(const MetaTensor& x,
                                const std::vector<int>& kernel_sizes,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& dilations,
                                MetaTensor* out,
                                MetaConfig config = MetaConfig());

PADDLE_API void UniformRandomInplaceInferMeta(const MetaTensor& x,
                                              float min,
                                              float max,
                                              int seed,
                                              int diag_num,
                                              int diag_step,
                                              float diag_val,
                                              MetaTensor* out);

PADDLE_API void UniformRandomBatchSizeLikeInferMeta(
    const MetaTensor& input,
    const std::vector<int>& shape,
    int input_dim_idx,
    int output_dim_idx,
    float min,
    float max,
    int seed,
    int diag_num,
    int diag_step,
    float diag_val,
    DataType dtype,
    MetaTensor* out,
    MetaConfig config = MetaConfig());

PADDLE_API void UniqueConsecutiveInferMeta(const MetaTensor& x,
                                           bool return_inverse,
                                           bool return_counts,
                                           const std::vector<int>& axis,
                                           DataType dtype,
                                           MetaTensor* out,
                                           MetaTensor* index,
                                           MetaTensor* counts);

PADDLE_API void UniqueInferMeta(const MetaTensor& x,
                                bool return_index,
                                bool return_inverse,
                                bool return_counts,
                                const std::vector<int>& axis,
                                DataType dtype,
                                MetaTensor* out,
                                MetaTensor* indices,
                                MetaTensor* index,
                                MetaTensor* counts);

PADDLE_API void UniqueRawInferMeta(const MetaTensor& x,
                                   bool return_index,
                                   bool return_inverse,
                                   bool return_counts,
                                   const std::vector<int>& axis,
                                   DataType dtype,
                                   bool is_sorted,
                                   MetaTensor* out,
                                   MetaTensor* indices,
                                   MetaTensor* index,
                                   MetaTensor* counts);

PADDLE_API void UnsqueezeInferMeta(const MetaTensor& x,
                                   const IntArray& axes,
                                   MetaTensor* out,
                                   MetaConfig config = MetaConfig());

PADDLE_API void UnsqueezeWithXShapeInferMeta(const MetaTensor& x,
                                             const IntArray& axes,
                                             MetaTensor* out,
                                             MetaTensor* xshape,
                                             MetaConfig config = MetaConfig());

PADDLE_API void UnStackInferMeta(const MetaTensor& x,
                                 int axis,
                                 int num,
                                 std::vector<MetaTensor*> outs);

PADDLE_API void NumberCountInferMeta(const MetaTensor& x,
                                     int upper_range,
                                     MetaTensor* out);

PADDLE_API void StridedUnChangedInferMeta(const MetaTensor& x, MetaTensor* out);

PADDLE_API void StraightThroughEstimatorInferMeta(const MetaTensor& out_grad,
                                                  MetaTensor* x_grad);

PADDLE_API void LrnInferMeta(const MetaTensor& x,
                             int n,
                             MetaTensor* out,
                             MetaTensor* mid_out);

PADDLE_API void ArrayPopInferMeta(const MetaTensor& array,
                                  int index,
                                  MetaTensor* array_out,
                                  MetaTensor* out,
                                  MetaConfig config = MetaConfig());

PADDLE_API void BuildSrcRankAndLocalExpertIdInferMeta(
    const MetaTensor& expert_num_global_tensor,
    const std::vector<int64_t>& expert_num_global,
    int64_t num_local_experts,
    MetaTensor* src_rank,
    MetaTensor* local_expert_id);

PADDLE_API void IntBincountInferMeta(const MetaTensor& x,
                                     int64_t low,
                                     int64_t high,
                                     int64_t dtype,
                                     MetaTensor* out);

}  // namespace phi
