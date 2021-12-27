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
#include <cn_api.h>
#include <cnnl.h>
#include <concurrentqueue.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using ExecutionContext = framework::ExecutionContext;
using DeviceContextPool = platform::DeviceContextPool;
using MLUDeviceContext = platform::MLUDeviceContext;

enum MLULogicMethod {
  CNNL_LOGIC_OP_EQ = 0,
  CNNL_LOGIC_OP_NE = 1,
  CNNL_LOGIC_OP_GT = 2,
  CNNL_LOGIC_OP_GE = 3,
  CNNL_LOGIC_OP_LT = 4,
  CNNL_LOGIC_OP_LE = 5,
  CNNL_LOGIC_OP_AND = 6,
  CNNL_LOGIC_OP_OR = 7,
};

template <typename T>
inline cnnlDataType_t ToCnnlDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCnnlDataType(type);
}

template <>
inline cnnlDataType_t ToCnnlDataType(const framework::proto::VarType::Type& t) {
  cnnlDataType_t type = CNNL_DTYPE_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = CNNL_DTYPE_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = CNNL_DTYPE_FLOAT;
      break;
    case framework::proto::VarType::INT8:
      type = CNNL_DTYPE_INT8;
      break;
    case framework::proto::VarType::INT32:
      type = CNNL_DTYPE_INT32;
      break;
    case framework::proto::VarType::INT64:
      type = CNNL_DTYPE_INT64;
      break;
    case framework::proto::VarType::BOOL:
      type = CNNL_DTYPE_BOOL;
      break;
    default:
      break;
  }
  return type;
}

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

static cnnlHandle_t GetHandleFromCTX(const ExecutionContext& ctx) {
  return ctx.template device_context<MLUDeviceContext>().cnnl_handle();
}

static const MLUDeviceContext& GetDevCtxFromCTX(const ExecutionContext& ctx) {
  return ctx.template device_context<MLUDeviceContext>();
}

cnnlDeviceType_t GetCnnlDev(int dev_ordinal);

using CnnlTensorDesc = cnnlTensorDescriptor_t;

class MLUCnnlTensorDesc {
 public:
  MLUCnnlTensorDesc() {}

  // SE_DISALLOW_COPY_AND_ASSIGN
  MLUCnnlTensorDesc(const MLUCnnlTensorDesc& desc) = delete;
  MLUCnnlTensorDesc& operator=(const MLUCnnlTensorDesc&) = delete;

  MLUCnnlTensorDesc(MLUCnnlTensorDesc&& rhs)
      : raw_tensor_desc(rhs.raw_tensor_desc) {
    rhs.raw_tensor_desc = nullptr;
  }

  MLUCnnlTensorDesc& operator=(MLUCnnlTensorDesc&& rhs);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const Tensor& tensor, const cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const Tensor& tensor, cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype, int position);

  MLUCnnlTensorDesc(const Tensor& tensor, cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype, int position,
                    float scale);

  ~MLUCnnlTensorDesc();

  const cnnlTensorDescriptor_t get() const { return raw_tensor_desc; }

 private:
  cnnlTensorDescriptor_t raw_tensor_desc = nullptr;
};

class MLUCnnlActivationDesc {
 public:
  MLUCnnlActivationDesc(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc& operator=(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc(const cnnlActivationMode_t act_mode, const float ceof);

  const cnnlActivationDescriptor_t get() const;
  ~MLUCnnlActivationDesc();

 private:
  cnnlActivationDescriptor_t active_desc_ = nullptr;
};

class MLUCnnlPoolingDesc {
 public:
  MLUCnnlPoolingDesc(const MLUCnnlPoolingDesc& desc) = delete;
  MLUCnnlPoolingDesc& operator=(const MLUCnnlPoolingDesc& desc) = delete;

  MLUCnnlPoolingDesc(const cnnlPoolingMode_t mode,
                     const cnnlNanPropagation_t maxpooling_nan_opt,
                     int window_rows, int window_cols, int64_t pad_up,
                     int64_t pad_down, int64_t pad_left, int64_t pad_right,
                     int row_stride, int col_stride);

  MLUCnnlPoolingDesc(const cnnlPoolingMode_t mode,
                     const cnnlNanPropagation_t maxpooling_nan_opt,
                     const int tensor_rank, const std::vector<int>& window,
                     const std::vector<int>& padding,
                     const std::vector<int>& stride);

  const cnnlPoolingDescriptor_t get() const;

  ~MLUCnnlPoolingDesc();

 private:
  cnnlPoolingDescriptor_t pooling_desc_ = nullptr;
};

class MLUCnnlRandomGeneratorDesc {
 public:
  MLUCnnlRandomGeneratorDesc(const bool is_mlu200, const int seed);
  const cnnlRandGenerator_t get() const;
  ~MLUCnnlRandomGeneratorDesc();

 private:
  cnnlRandGenerator_t mlu_generator = nullptr;
};

class MLUCnnlReduceDesc {
 public:
  MLUCnnlReduceDesc(const MLUCnnlReduceDesc& desc) = delete;
  MLUCnnlReduceDesc& operator=(const MLUCnnlReduceDesc& desc) = delete;

  MLUCnnlReduceDesc(const std::vector<int>& axis_vec,
                    const cnnlReduceOp_t reduce_op,
                    const cnnlDataType_t data_type,
                    const cnnlNanPropagation_t nan_propagation,
                    const cnnlReduceIndices_t reduce_indices,
                    const cnnlIndicesType_t indices_type);

  const cnnlReduceDescriptor_t get() const;

  ~MLUCnnlReduceDesc();

 private:
  cnnlReduceDescriptor_t reduction_desc_ = nullptr;
};

class MLUCnnlOpTensorDesc {
 public:
  MLUCnnlOpTensorDesc(const MLUCnnlOpTensorDesc& desc) = delete;
  void operator=(const MLUCnnlOpTensorDesc&) = delete;

  MLUCnnlOpTensorDesc(cnnlOpTensorDesc_t op_tensor_op,
                      cnnlDataType_t op_tensor_comp_type,
                      cnnlNanPropagation_t op_tensor_nan_opt);

  const cnnlOpTensorDescriptor_t get() const;

  ~MLUCnnlOpTensorDesc();

 private:
  cnnlOpTensorDescriptor_t op_tensor_desc_ = nullptr;
};

class MLUCnnlNMSDesc {
 public:
  MLUCnnlNMSDesc(const MLUCnnlNMSDesc& desc) = delete;
  MLUCnnlNMSDesc& operator=(const MLUCnnlNMSDesc& desc) = delete;

  MLUCnnlNMSDesc(const cnnlNmsOutputMode_t mode, const float iou_threshold,
                 const int max_output_size, const float confidence_threshold,
                 const int input_layout);

  const cnnlNmsDescriptor_t get() const;

  ~MLUCnnlNMSDesc();

 private:
  cnnlNmsDescriptor_t nms_desc_ = nullptr;
};

class MLUCnnlConvolutionDesc {
 public:
  MLUCnnlConvolutionDesc(const int dims, const int pad[], const int stride[],
                         const int dilation[], const int group_count,
                         const cnnlDataType_t tensor_dtype);

  MLUCnnlConvolutionDesc(const int dims, const int64_t pad[],
                         const int64_t stride[], const int64_t dilation[],
                         const int group_count,
                         const cnnlDataType_t tensor_dtype);

  MLUCnnlConvolutionDesc(const MLUCnnlConvolutionDesc& desc) = delete;

  MLUCnnlConvolutionDesc& operator=(const MLUCnnlConvolutionDesc& desc) =
      delete;

  const cnnlConvolutionDescriptor_t get() const;

  ~MLUCnnlConvolutionDesc();

 private:
  cnnlConvolutionDescriptor_t conv_desc_ = nullptr;
};

class MLUCnnlBatchSpaceDesc {
 public:
  MLUCnnlBatchSpaceDesc(uint32_t block_shape[], uint32_t paddings[],
                        const uint32_t block_shape_size,
                        const uint32_t paddings_size);

  void getBatch2spaceNdextraInputSize(const ExecutionContext& ctx,
                                      const cnnlTensorDescriptor_t input_desc);

  void getSpace2batchNdextraInputSize(const ExecutionContext& ctx,
                                      const cnnlTensorDescriptor_t input_desc);

  void initSpace2batchNdExtraInput(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t input_desc,
                                   void* extra_host_input);

  void initBatch2spaceNdExtraInput(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t input_desc,
                                   void* extra_host_input);

  const cnnlSpaceBatchNdDescriptor_t get() const;

  size_t getExtraInputSize() const;

  ~MLUCnnlBatchSpaceDesc();

 private:
  cnnlSpaceBatchNdDescriptor_t op_desc_ = nullptr;
  size_t extra_input_size_;
};

class MLUCnnlTrigonDesc {
 public:
  explicit MLUCnnlTrigonDesc(
      const cnnlTrigonFunctionMode_t trigon_function_mode);

  const cnnlTrigonDescriptor_t get() const;

  ~MLUCnnlTrigonDesc();

 private:
  cnnlTrigonDescriptor_t trigon_desc_ = nullptr;
};

class MLUCnnl {
 public:
  static void Active(const ExecutionContext& ctx,
                     cnnlActivationDescriptor_t active_desc,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void ActiveGrad(
      const ExecutionContext& ctx, cnnlActivationDescriptor_t active_desc,
      const void* alpha, const void* beta, const cnnlTensorDescriptor_t y_desc,
      const void* y, const cnnlTensorDescriptor_t diff_y_desc,
      const void* diff_y, const cnnlTensorDescriptor_t x_desc, const void* x,
      const cnnlTensorDescriptor_t diff_x_desc, void* diff_x);

  static void Concat(const ExecutionContext& ctx, const int pack_num,
                     const int axis, const cnnlTensorDescriptor_t inputs_desc[],
                     const void* const inputs[],
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void Cast(const ExecutionContext& ctx, cnnlCastDataType_t cast_type,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void Div(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t in0_desc, const void* in0,
                  const cnnlTensorDescriptor_t in1_desc, const void* in1,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Fill(const ExecutionContext& ctx, float value,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void LRN(const ExecutionContext& ctx, const int local_size,
                  const double alpha, const double beta, const double k,
                  const cnnlTensorDescriptor_t input_quant_desc,
                  const void* input_quant,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void QuantifyOffline(const ExecutionContext& context,
                              cnnlQuantizeMode_t mode,
                              const cnnlTensorDescriptor_t input_desc,
                              const void* input,
                              const cnnlTensorDescriptor_t ouput_desc,
                              void* output);

  static void QuantifyOnline(const ExecutionContext& context,
                             const int bitwidth,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input, const bool compute_scale,
                             void* position, void* scale,
                             const cnnlTensorDescriptor_t ouput_desc,
                             void* output);

  static void SGD(const ExecutionContext& context,
                  const cnnlTensorDescriptor_t grad_desc, const void* grad,
                  const void* lr, const cnnlTensorDescriptor_t var_desc,
                  void* var);

  static void ApplyAdaGrad(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t grad_desc,
                           const void* grad,
                           const cnnlTensorDescriptor_t accum_desc, void* accum,
                           const cnnlTensorDescriptor_t var_desc, void* var,
                           const void* lr, const bool update_slots);

  static void ApplyRMSProp(const ExecutionContext& context,
                           const cnnlTensorDescriptor_t grad_desc,
                           const void* grad, const void* lr, const void* rho,
                           const void* momentum, const void* epsilon,
                           const cnnlTensorDescriptor_t var_desc, void* var,
                           const cnnlTensorDescriptor_t ms_desc, void* ms,
                           const cnnlTensorDescriptor_t mom_desc, void* mom);

  static void ApplyCenterRMSProp(
      const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
      const void* grad, const void* lr, const void* rho, const void* momentum,
      const void* epsilon, const cnnlTensorDescriptor_t var_desc, void* var,
      const cnnlTensorDescriptor_t mg_desc, void* mg,
      const cnnlTensorDescriptor_t ms_desc, void* ms,
      const cnnlTensorDescriptor_t mom_desc, void* mom);

  static void ApplyAdam(const ExecutionContext& ctx,
                        const cnnlTensorDescriptor_t grad_desc,
                        const void* grad, const void* lr, const void* beta1,
                        const void* beta2, const void* beta1_power,
                        const void* beta2_power, const void* epsilon,
                        const bool use_nesterov,
                        const cnnlTensorDescriptor_t var_desc, void* var,
                        const cnnlTensorDescriptor_t m_desc, void* m,
                        const cnnlTensorDescriptor_t v_desc, void* v);

  static void ApplyAdaMax(const ExecutionContext& ctx,
                          const cnnlTensorDescriptor_t grad_desc,
                          const cnnlTensorDescriptor_t var_desc, void* var,
                          const cnnlTensorDescriptor_t m_desc, void* m,
                          const cnnlTensorDescriptor_t v_desc, void* v,
                          const void* diff, const void* lr, const void* beta1,
                          const void* beta2, const void* beta1_power,
                          const void* epsilon);

  static void ApplyMomentum(const ExecutionContext& ctx,
                            const cnnlTensorDescriptor_t grad_desc,
                            const void* grad, const bool use_nesterov,
                            const void* lr, const void* momentum, void* var,
                            void* accum);

  static void ApplyKerasMomentum(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t grad_desc,
                                 const void* grad, const bool use_nesterov,
                                 const void* lr, const void* momentum,
                                 void* var, void* accum);

  static void ApplyAdadelta(const ExecutionContext& ctx,
                            const cnnlTensorDescriptor_t grad_desc,
                            const void* diff, const void* lr, const void* rho,
                            const void* epsilon, void* var, void* accum,
                            void* accum_update);

  static void SparseSoftmaxXentWithLogits(
      const ExecutionContext& ctx, cnnlSoftmaxMode_t mode,
      const cnnlTensorDescriptor_t x_desc, const void* input,
      const cnnlTensorDescriptor_t label_desc, const void* label,
      const cnnlTensorDescriptor_t y_desc, void* output,
      const cnnlTensorDescriptor_t diff_y_desc, void* back_out);

  static void RandomUniform(const ExecutionContext& ctx, const int num,
                            const cnnlDataType_t data_type,
                            const cnnlRandGenerator_t mlu_generator,
                            void* output);

  static void Cumsum(const ExecutionContext& ctx, const int axis,
                     const bool exclusive, const bool reverse,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t ouput_desc, void* output);

  static void BroadcastTo(const ExecutionContext& ctx,
                          const cnnlTensorDescriptor_t input_desc,
                          const void* input,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void GatherFunctor(
      const ExecutionContext& ctx, const int axis, const int batch_dims,
      const cnnlTensorDescriptor_t params_desc, const void* params,
      const cnnlTensorDescriptor_t indices_desc, const void* indices,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void ScatterFunctor(
      const ExecutionContext& ctx, const cnnlTensorDescriptor_t params_desc,
      const void* params, const cnnlTensorDescriptor_t updates_desc,
      const void* updates, const cnnlTensorDescriptor_t indices_desc,
      const void* indices, const cnnlScatterRefMode_t mode);

  static void Range(const ExecutionContext& ctx, const void* start,
                    const void* end, const void* step,
                    const cnnlDataType_t output_dtype, void* output);

  static void Round(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void TopK(const ExecutionContext& ctx, const int k, const int dim,
                   const bool largest, const bool sorted,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t values_output_desc,
                   void* values_out,
                   const cnnlTensorDescriptor_t indices_output_desc,
                   void* indices_out);

  static void StridedSlice(const ExecutionContext& ctx, const int begin[],
                           const int end[], const int strides[],
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output);

  static void Split(const ExecutionContext& ctx, int split_num, int axis,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input_ptr,
                    const cnnlTensorDescriptor_t output_descs[],
                    void* output_ptrs[]);

  static void Scale(const ExecutionContext& ctx, const int axis,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t alpha_desc, const void* alpha,
                    const cnnlTensorDescriptor_t beta_desc, const void* beta,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void AddN(const ExecutionContext& ctx, uint32_t input_num,
                   const cnnlTensorDescriptor_t inputs_desc[],
                   const void* inputs[],
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void Log(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void StridedSliceGrad(const ExecutionContext& ctx, const int begin[],
                               const int end[], const int strides[],
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output);

  static void Logic(const ExecutionContext& ctx,
                    const MLULogicMethod log_method,
                    const cnnlTensorDescriptor_t input1_desc,
                    const void* input1,
                    const cnnlTensorDescriptor_t input2_desc,
                    const void* input2, const cnnlTensorDescriptor_t ouput_desc,
                    void* output);

  static void Select(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t then_desc, const void* p_then,
                     const cnnlTensorDescriptor_t else_desc, const void* p_else,
                     const cnnlTensorDescriptor_t output_desc, void* output,
                     const bool* condition, const int condition_size);

  static void AssignAdd(const ExecutionContext& ctx, const void* alpha,
                        const void* beta,
                        const cnnlTensorDescriptor_t update_desc,
                        const void* update,
                        const cnnlTensorDescriptor_t param_desc, void* param);

  static void AssignSub(const ExecutionContext& ctx, const void* alpha,
                        const void* beta,
                        const cnnlTensorDescriptor_t update_desc,
                        const void* update,
                        const cnnlTensorDescriptor_t param_desc, void* param);

  static void Assign(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t update_desc,
                     const void* update,
                     const cnnlTensorDescriptor_t param_desc, void* param);

  static void GatherNd(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t params_desc,
                       const void* params,
                       const cnnlTensorDescriptor_t indices_desc,
                       const void* indices,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void BatchToSpace(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output, const cnnlSpaceBatchParam_t param);

  static void BatchToSpaceNd(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             cnnlSpaceBatchNdDescriptor_t param,
                             void* extra_device_input, size_t extra_input_size,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void PoolingForward(
      const ExecutionContext& ctx, cnnlPoolingMode_t pool_mode,
      const std::vector<int64_t>& output_shape,
      cnnlPoolingDescriptor_t pooling_desc, const void* alpha,
      const cnnlTensorDescriptor_t input_desc, const void* input,
      const void* beta, const void* extra_input_ptr,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void Pool3D(const ExecutionContext& ctx, cnnlPoolingMode_t pool_mode,
                     const std::vector<int64_t>& output_shape,
                     cnnlPoolingDescriptor_t pooling_desc, const void* alpha,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const void* beta, const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void Pad(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const void* paddings, const void* padding_value,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Matmul(const ExecutionContext& ctx, const bool transpose_a,
                     const bool transpose_b,
                     const cnnlTensorDescriptor_t in0_desc, const void* in0,
                     const cnnlTensorDescriptor_t in1_desc, const void* in1,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void BatchMatmul(
      const ExecutionContext& ctx, const bool transpose_a,
      const bool transpose_b, const cnnlTensorDescriptor_t in0_desc,
      const void* in0, const cnnlTensorDescriptor_t in1_desc, const void* in1,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void OpTensor(const ExecutionContext& ctx,
                       const cnnlOpTensorDescriptor_t op_tensor_desc,
                       const cnnlTensorDescriptor_t a_desc, const void* a,
                       const cnnlTensorDescriptor_t b_desc, const void* b,
                       const cnnlTensorDescriptor_t output_desc, void* output,
                       const cnnlDataType_t dtype);

  static void BiasAddGrad(const ExecutionContext& ctx, const int axis,
                          const cnnlTensorDescriptor_t out_backprop_desc,
                          const void* out_backprop,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void OneHot(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t desc_indices,
                     const void* indices, const int depth, const void* on_value,
                     const void* off_value, const int axis,
                     cnnlDataType_t output_data_type, void* output);

  static void NonMaxSuppression(const ExecutionContext& ctx,
                                const cnnlNmsDescriptor_t nms_desc,
                                const cnnlTensorDescriptor_t boxes_desc,
                                const void* boxes,
                                const cnnlTensorDescriptor_t confidence_desc,
                                const void* confidence,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output, void* output_size);

  static void SoftmaxCrossEntropyWithLogits(
      const ExecutionContext& ctx, cnnlSoftmaxMode_t mode,
      cnnlComputationPreference_t prefer,
      const cnnlTensorDescriptor_t input_desc, const void* logits_in,
      const cnnlTensorDescriptor_t label_desc, const void* labels_in,
      const cnnlTensorDescriptor_t loss_out_desc, void* loss_out,
      const cnnlTensorDescriptor_t back_out_desc, void* back_out);

  static void SoftmaxForward(const ExecutionContext& ctx,
                             cnnlSoftmaxAlgorithm_t algorithm,
                             cnnlSoftmaxMode_t mode, const void* alpha,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input, const void* beta,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void Softplus(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t features_desc,
                       const void* features,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void SoftplusGrad(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t gradients_desc,
                           const void* gradients,
                           const cnnlTensorDescriptor_t features_desc,
                           const void* features,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output);

  static void RsqrtGrad(const ExecutionContext& ctx,
                        const cnnlTensorDescriptor_t data_desc, const void* y,
                        const void* diff_y, void* output);

  static void SqrtGrad(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t data_desc, const void* y,
                       const void* diff_y, void* output);

  static void ConvolutionForward(
      const ExecutionContext& ctx, cnnlConvolutionDescriptor_t conv_desc_,
      const void* alpha, const void* beta,
      const cnnlTensorDescriptor_t bias_desc, const void* bias_ptr,
      const cnnlTensorDescriptor_t input_desc, const void* input,
      const cnnlTensorDescriptor_t filtet_desc, const void* filter,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void FusedConvBNQuantify(
      const ExecutionContext& ctx, cnnlConvolutionDescriptor_t conv_desc,
      const void* epsilon_ptr, const int fused_ops_number,
      const cnnlDataType_t tensor_dtype, const int input_position,
      const float input_scale, const int filter_position,
      const float filter_scale, const cnnlTensorDescriptor_t scale_desc,
      const void* scale_ptr, const cnnlTensorDescriptor_t offset_desc,
      const void* offset_ptr, const cnnlTensorDescriptor_t mean_desc,
      const void* mean_ptr, const cnnlTensorDescriptor_t variance_desc,
      const void* variance_ptr, const cnnlTensorDescriptor_t input_desc,
      const void* input, const cnnlTensorDescriptor_t filtet_desc,
      const void* filter, const cnnlTensorDescriptor_t output_desc,
      void* output);

  static void Tile(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void UnsortedSegmentSum(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t data_desc,
                                 const void* data,
                                 const cnnlTensorDescriptor_t ids_desc,
                                 const int* segment_ids,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output);

  static void Reduce(const ExecutionContext& ctx, const bool need_workspace,
                     const cnnlReduceDescriptor_t reduction_desc,
                     const void* alpha, const cnnlTensorDescriptor_t input_desc,
                     const void* input, const size_t indices_size,
                     void* indices, const void* beta,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void FloorDiv(const ExecutionContext& ctx,
                       cnnlComputationPreference_t prefer,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void FloorMod(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void Maximum(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc, void* output);

  static void Minimum(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc, void* output);

  static void PowR(const ExecutionContext& ctx,
                   cnnlComputationPreference_t prefer,
                   const cnnlTensorDescriptor_t input1_desc, const void* input1,
                   const cnnlTensorDescriptor_t input2_desc, const void* input2,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void DivNoNan(const ExecutionContext& ctx,
                       cnnlComputationPreference_t prefer,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void SquaredDifference(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t input1_desc,
                                const void* input1,
                                const cnnlTensorDescriptor_t input2_desc,
                                const void* input2,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output);

  static void L2Loss(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     void* output);

  static void Abs(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Neg(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Floor(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void Ceil(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void IsNan(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void Square(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void Sqrt(const ExecutionContext& ctx,
                   cnnlComputationPreference_t prefer,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void Rsqrt(const ExecutionContext& ctx,
                    cnnlComputationPreference_t prefer,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void Cos(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Sin(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void TrigonForward(const ExecutionContext& ctx,
                            const cnnlTrigonDescriptor_t trigon_desc,
                            const cnnlTensorDescriptor_t input_desc,
                            const void* input,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void Exp(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Sign(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc, const void* input,
                   const cnnlTensorDescriptor_t output_desc, void* output);

  static void IsFinite(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input,
                       const cnnlTensorDescriptor_t output_desc, void* output);

  static void IsNanInf(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input, void* output);

  static void Erf(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc, const void* input,
                  const cnnlTensorDescriptor_t output_desc, void* output);

  static void Log1p(const ExecutionContext& ctx,
                    cnnlComputationPreference_t prefer,
                    const cnnlTensorDescriptor_t input_desc, const void* input,
                    const cnnlTensorDescriptor_t output_desc, void* output);

  static void LogicalNot(const ExecutionContext& ctx,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);

  static void DynamicStitch(
      const ExecutionContext& ctx, const cnnlTensorDescriptor_t* indices_desc,
      const int** indices, const cnnlTensorDescriptor_t* data_desc,
      const void** data, const int size, int* indices_dims,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void CropAndResize(
      const ExecutionContext& ctx, const std::string method_name,
      const float extrapolation_value, const cnnlTensorDescriptor_t image_desc,
      const void* image, const cnnlTensorDescriptor_t boxes_desc,
      const void* boxes, const cnnlTensorDescriptor_t box_index_desc,
      const void* box_index, const cnnlTensorDescriptor_t output_desc,
      void* output);

  static void CropAndResizeBackwardImage(
      const ExecutionContext& ctx, const std::string method_name,
      const cnnlTensorDescriptor_t image_desc, const void* image,
      const cnnlTensorDescriptor_t boxes_desc, const void* boxes,
      const cnnlTensorDescriptor_t box_idx_desc, const void* box_idx,
      const cnnlTensorDescriptor_t grads_image_desc, void* grads_image);

  static void CropAndResizeBackwardBoxes(
      const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
      const void* input, const cnnlTensorDescriptor_t image_desc,
      const void* image, const cnnlTensorDescriptor_t boxes_desc,
      const void* boxes, const cnnlTensorDescriptor_t box_idx_desc,
      const void* box_idx, const cnnlTensorDescriptor_t output_desc,
      void* output);

  static void PoolingBackward(
      const ExecutionContext& ctx, const cnnlPoolingDescriptor_t pooling_desc,
      const void* alpha, const cnnlTensorDescriptor_t y_desc, const void* y,
      const cnnlTensorDescriptor_t diff_y_desc, const void* diff_y,
      const cnnlTensorDescriptor_t x_desc, const void* x, const void* beta,
      const cnnlTensorDescriptor_t diff_x_desc, void* diff_x);

  static void PoolingIndex(const ExecutionContext& ctx,
                           const cnnlPoolingDescriptor_t pooling_desc,
                           const cnnlTensorDescriptor_t x_desc, const void* x,
                           const cnnlTensorDescriptor_t y_desc, void* y);

  static void SpaceToBatch(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output, const int64_t block_shape[]);

  static void SpaceToBatchNd(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             cnnlSpaceBatchNdDescriptor_t param,
                             void* extra_device_input, size_t extra_input_size,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void Interp(const ExecutionContext& ctx, const cnnlInterpMode_t mode,
                     const bool align_corners, const bool half_pixel_centers,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void InterpBackward(
      const ExecutionContext& ctx, const cnnlInterpBackwardMode_t mode,
      const bool align_corners, const bool half_pixel_centers,
      const cnnlTensorDescriptor_t input_desc, const void* input,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void QuantizeParam(const ExecutionContext& ctx,
                            const cnnlQuantizeMode_t mode, const int bitwidth,
                            const cnnlTensorDescriptor_t input_desc,
                            const void* input, void* position, void* scale,
                            void* offset);

  static void QuantizeMatMul(
      const ExecutionContext& ctx, const bool transpose_a,
      const bool transpose_b, const cnnlTensorDescriptor_t a_desc,
      const void* a, const void* a_position, const void* a_scale,
      const void* a_offset, const cnnlTensorDescriptor_t b_desc, const void* b,
      const void* b_position, const void* b_scale, const void* b_offset,
      const cnnlDataType_t quant_type, const cnnlDataType_t data_type,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void QuantizeBatchMatMul(
      const ExecutionContext& ctx, const bool adj_x, const bool adj_y,
      const cnnlTensorDescriptor_t a_desc, const void* a,
      const void* a_position, const void* a_scale, const void* a_offset,
      const cnnlTensorDescriptor_t b_desc, const void* b,
      const void* b_position, const void* b_scale, const void* b_offset,
      const cnnlDataType_t quant_type, const cnnlDataType_t data_type,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void QuantizeBatchMatMulBCast(
      const ExecutionContext& ctx, const bool adj_x, const bool adj_y,
      const cnnlTensorDescriptor_t a_desc, const void* a,
      const void* a_position, const void* a_scale, const void* a_offset,
      const cnnlTensorDescriptor_t b_desc, const void* b,
      const void* b_position, const void* b_scale, const void* b_offset,
      const cnnlDataType_t quant_type, const cnnlDataType_t data_type,
      const cnnlTensorDescriptor_t output_desc, void* output);

  static void FusedBatchNorm(
      const ExecutionContext& ctx, const bool is_training,
      const cnnlTensorDescriptor_t x_desc, const void* x,
      const cnnlTensorDescriptor_t scale_desc, const void* scale,
      const void* offset, const void* estimated_mean,
      const void* estimated_variance, float epsilon, float momentum,
      const cnnlTensorDescriptor_t output_desc, void* output, void* batch_mean,
      void* batch_var, void* saved_mean, void* saved_var);

  static void FusedBatchNormGrad(
      const ExecutionContext& ctx, const bool is_training,
      const cnnlTensorDescriptor_t y_backprop_desc, const void* y_backprop,
      const cnnlTensorDescriptor_t x_desc, const void* x,
      const cnnlTensorDescriptor_t scale_desc, const void* scale,
      const void* saved_mean, const void* saved_var, float epsilon,
      const cnnlTensorDescriptor_t x_backprop_desc, void* x_backprop,
      void* scale_backprop, void* offset_backprop);

  static void Transpose(const ExecutionContext& ctx,
                        const std::vector<int> perm, const int input_dim,
                        const cnnlTensorDescriptor_t input_desc,
                        const void* input,
                        const cnnlTensorDescriptor_t output_desc, void* output);

  static void MatrixBandPart(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t data_desc,
                             const void* input, const int num_lower,
                             const int num_upper, void* output);

  static void NumTrue(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t x_desc, const void* x,
                      Tensor index, uint32_t* num_true);

  static void Where(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t x_desc, const void* x,
                    const uint32_t* strides, const uint32_t* index,
                    const cnnlTensorDescriptor_t y_desc, int* y,
                    const bool as_tuple);

  static void Conv2D(const ExecutionContext& ctx,
                     const cnnlConvolutionDescriptor_t conv_desc,
                     const cnnlDataType_t tensor_dtype,
                     const cnnlDataType_t dt_onchip, const void* input_position,
                     const void* input_scale, const void* input_offset,
                     const void* filter_position, const void* filter_scale,
                     const void* filter_offset,
                     const cnnlTensorDescriptor_t input_desc, const void* input,
                     const cnnlTensorDescriptor_t filter_desc,
                     const void* filter, const cnnlTensorDescriptor_t bias_desc,
                     const void* bias, const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void ConvBackpropInput(
      const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlTensorDescriptor_t input_desc, const void* filter,
      const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
      const cnnlTensorDescriptor_t in_backprop_desc, void* in_backprop);

  static void QuantizeConvBackpropInput(
      const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlDataType_t tensor_dtype, const cnnlDataType_t dt_onchip,
      const void* filter_position, const void* filter_scale,
      const void* filter_offset, const void* out_backprop_position,
      const void* out_backprop_scale, const void* out_backprop_offset,
      const cnnlTensorDescriptor_t input_desc, const void* filter,
      const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
      const cnnlTensorDescriptor_t in_backprop_desc, void* in_backprop);

  static void ConvBackpropFilter(
      const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlTensorDescriptor_t input_desc, const void* input,
      const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
      const cnnlTensorDescriptor_t filter_backprop_desc, void* filter_backprop);

  static void QuantizeConvBackpropFilter(
      const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlDataType_t tensor_dtype, const cnnlDataType_t dt_onchip,
      const void* input_position, const void* input_scale,
      const void* input_offset, const void* out_backprop_position,
      const void* out_backprop_scale, const void* out_backprop_offset,
      const cnnlTensorDescriptor_t input_desc, const void* input,
      const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
      const cnnlTensorDescriptor_t filter_backprop_desc, void* filter_backprop);

  static void InTopK(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t predictions_desc,
                     const void* predictions,
                     const cnnlTensorDescriptor_t targets_desc,
                     const void* targets, const cnnlTensorDescriptor_t k_desc,
                     const void* k, const int k_int,
                     const cnnlTensorDescriptor_t output_desc, void* output);

  static void ScatterNd(const ExecutionContext& ctx,
                        const cnnlTensorDescriptor_t indices_desc,
                        const void* indices,
                        const cnnlTensorDescriptor_t updates_desc,
                        const void* updates,
                        const cnnlTensorDescriptor_t output_desc, void* output);

  static void BitWise(const ExecutionContext& ctx,
                      const cnnlBitComputeOp_t optype,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc, void* output);

  static void QR(const ExecutionContext& ctx,
                 const cnnlTensorDescriptor_t a_desc, const void* a,
                 const cnnlTensorDescriptor_t q_desc, void* q,
                 const cnnlTensorDescriptor_t r_desc, void* r, const bool some);

  static void Reciprocal(const ExecutionContext& ctx,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);
};

}  // namespace operators
}  // namespace paddle
