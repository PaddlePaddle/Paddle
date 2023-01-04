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
#include <mlu_op.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;
using ExecutionContext = framework::ExecutionContext;
using DeviceContextPool = platform::DeviceContextPool;
using MLUDeviceContext = platform::MLUDeviceContext;

const std::map<std::string, cnnlReduceOp_t> MLUReduceOpMap = {
    {"reduce_all", CNNL_REDUCE_AND},
    {"reduce_any", CNNL_REDUCE_OR},
    {"reduce_max", CNNL_REDUCE_MAX},
    {"reduce_mean", CNNL_REDUCE_AVG},
    {"reduce_min", CNNL_REDUCE_MIN},
    {"reduce_sum", CNNL_REDUCE_ADD},
    {"reduce_prod", CNNL_REDUCE_MUL},
};

const std::map<std::string, cnnlInterpMode_t> MLUInterpModeMap = {
    {"bilinear", CNNL_INTERP_BILINEAR},
    {"nearest", CNNL_INTERP_NEAREST},
    {"linear", CNNL_INTERP_LINEAR},
    {"trilinear", CNNL_INTERP_TRILINEAR},
    {"bicubic", CNNL_INTERP_BICUBIC}};

const std::map<std::string, cnnlInterpBackwardMode_t> MLUInterpBackwardModeMap =
    {{"bilinear", CNNL_INTERP_BACKWARD_BILINEAR},
     {"nearest", CNNL_INTERP_BACKWARD_NEAREST},
     {"linear", CNNL_INTERP_BACKWARD_LINEAR},
     {"trilinear", CNNL_INTERP_BACKWARD_TRILINEAR},
     {"bicubic", CNNL_INTERP_BACKWARD_BICUBIC}};

inline cnnlReduceOp_t GetMLUCnnlReduceOp(const std::string reduce_name) {
  auto iter = MLUReduceOpMap.find(reduce_name);
  if (iter != MLUReduceOpMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not support reduce op type of MLU Device: %s", reduce_name));
}

inline cnnlInterpMode_t GetMLUCnnlInterpMode(const std::string interp_mode) {
  auto iter = MLUInterpModeMap.find(interp_mode);
  if (iter != MLUInterpModeMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not support interp mode of MLU Device: %s", interp_mode));
}

inline cnnlInterpBackwardMode_t GetMLUCnnlInterpBackwardMode(
    const std::string interp_mode) {
  auto iter = MLUInterpBackwardModeMap.find(interp_mode);
  if (iter != MLUInterpBackwardModeMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not support interp mode of MLU Device: %s", interp_mode));
}

inline const void* GetBasePtr(const phi::DenseTensor* t) { return t->data(); }

inline void* GetBasePtr(phi::DenseTensor* t) { return t->data(); }

inline cnnlDataType_t ToCnnlDataType(
    const paddle::experimental::DataType& dtype) {
  cnnlDataType_t type = CNNL_DTYPE_FLOAT;
  switch (dtype) {
    case DataType::FLOAT16:
      type = CNNL_DTYPE_HALF;
      break;
    case DataType::FLOAT32:
      type = CNNL_DTYPE_FLOAT;
      break;
    case DataType::FLOAT64:
      type = CNNL_DTYPE_DOUBLE;
      break;
    case DataType::INT8:
      type = CNNL_DTYPE_INT8;
      break;
    case DataType::INT16:
      type = CNNL_DTYPE_INT16;
      break;
    case DataType::INT32:
      type = CNNL_DTYPE_INT32;
      break;
    case DataType::INT64:
      type = CNNL_DTYPE_INT64;
      break;
    case DataType::BOOL:
      type = CNNL_DTYPE_BOOL;
      break;
    case DataType::UINT8:
      type = CNNL_DTYPE_UINT8;
      break;
    default:
      break;
  }
  return type;
}

inline cnnlDataType_t ToCnnlDataType(
    const paddle::framework::proto::VarType::Type& type) {
  return ToCnnlDataType(framework::TransToPhiDataType(type));
}

template <typename T>
inline cnnlDataType_t ToCnnlDataType() {
  auto type = framework::ToDataType(std::type_index(typeid(T)));
  return ToCnnlDataType(type);
}

inline mluOpDataType_t ToMluOpDataType(
    const paddle::experimental::DataType& dtype) {
  mluOpDataType_t type = MLUOP_DTYPE_FLOAT;
  switch (dtype) {
    case DataType::FLOAT16:
      type = MLUOP_DTYPE_HALF;
      break;
    case DataType::FLOAT32:
      type = MLUOP_DTYPE_FLOAT;
      break;
    case DataType::FLOAT64:
      type = MLUOP_DTYPE_DOUBLE;
      break;
    case DataType::INT8:
      type = MLUOP_DTYPE_INT8;
      break;
    case DataType::INT16:
      type = MLUOP_DTYPE_INT16;
      break;
    case DataType::INT32:
      type = MLUOP_DTYPE_INT32;
      break;
    case DataType::INT64:
      type = MLUOP_DTYPE_INT64;
      break;
    case DataType::BOOL:
      type = MLUOP_DTYPE_BOOL;
      break;
    case DataType::UINT8:
      type = MLUOP_DTYPE_UINT8;
      break;
    default:
      break;
  }
  return type;
}

inline mluOpDataType_t ToMluOpDataType(
    const paddle::framework::proto::VarType::Type& type) {
  return ToMluOpDataType(framework::TransToPhiDataType(type));
}

template <typename T>
inline mluOpDataType_t ToMluOpDataType() {
  auto type = framework::ToDataType(std::type_index(typeid(T)));
  return ToMluOpDataType(type);
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

inline static cnnlHandle_t GetHandleFromCTX(const ExecutionContext& ctx) {
  return ctx.template device_context<MLUDeviceContext>().cnnl_handle();
}

inline static mluOpHandle_t GetMLUOpHandleFromCTX(const ExecutionContext& ctx) {
  return ctx.template device_context<MLUDeviceContext>().mluOp_handle();
}

inline static const MLUDeviceContext& GetDevCtxFromCTX(
    const ExecutionContext& ctx) {
  return ctx.template device_context<MLUDeviceContext>();
}

using VT = framework::proto::VarType;
const std::map<std::pair<VT::Type, VT::Type>, cnnlCastDataType_t>
    MLU_SUPPORTED_CAST_TYPE = {
        {{VT::FP32, /*cast to*/ VT::FP16}, CNNL_CAST_FLOAT_TO_HALF},
        {{VT::FP32, /*cast to*/ VT::INT32}, CNNL_CAST_FLOAT_TO_INT32},
        {{VT::FP32, /*cast to*/ VT::INT16}, CNNL_CAST_FLOAT_TO_INT16},
        {{VT::FP32, /*cast to*/ VT::INT8}, CNNL_CAST_FLOAT_TO_INT8},
        {{VT::FP32, /*cast to*/ VT::UINT8}, CNNL_CAST_FLOAT_TO_UINT8},
        {{VT::FP32, /*cast to*/ VT::BOOL}, CNNL_CAST_FLOAT_TO_BOOL},
        {{VT::FP16, /*cast to*/ VT::FP32}, CNNL_CAST_HALF_TO_FLOAT},
        {{VT::FP16, /*cast to*/ VT::INT32}, CNNL_CAST_HALF_TO_INT32},
        {{VT::FP16, /*cast to*/ VT::INT16}, CNNL_CAST_HALF_TO_INT16},
        {{VT::FP16, /*cast to*/ VT::INT8}, CNNL_CAST_HALF_TO_INT8},
        {{VT::FP16, /*cast to*/ VT::UINT8}, CNNL_CAST_HALF_TO_UINT8},
        {{VT::FP16, /*cast to*/ VT::BOOL}, CNNL_CAST_HALF_TO_BOOL},
        {{VT::INT32, /*cast to*/ VT::FP32}, CNNL_CAST_INT32_TO_FLOAT},
        {{VT::INT32, /*cast to*/ VT::FP16}, CNNL_CAST_INT32_TO_HALF},
        {{VT::INT32, /*cast to*/ VT::INT8}, CNNL_CAST_INT32_TO_INT8},
        {{VT::INT32, /*cast to*/ VT::INT16}, CNNL_CAST_INT32_TO_INT16},
        {{VT::INT16, /*cast to*/ VT::FP32}, CNNL_CAST_INT16_TO_FLOAT},
        {{VT::INT16, /*cast to*/ VT::FP16}, CNNL_CAST_INT16_TO_HALF},
        {{VT::INT16, /*cast to*/ VT::INT32}, CNNL_CAST_INT16_TO_INT32},
        {{VT::INT8, /*cast to*/ VT::FP32}, CNNL_CAST_INT8_TO_FLOAT},
        {{VT::INT8, /*cast to*/ VT::FP16}, CNNL_CAST_INT8_TO_HALF},
        {{VT::INT8, /*cast to*/ VT::INT32}, CNNL_CAST_INT8_TO_INT32},
        {{VT::UINT8, /*cast to*/ VT::FP32}, CNNL_CAST_UINT8_TO_FLOAT},
        {{VT::UINT8, /*cast to*/ VT::FP16}, CNNL_CAST_UINT8_TO_HALF},
        {{VT::BOOL, /*cast to*/ VT::FP32}, CNNL_CAST_BOOL_TO_FLOAT},
        {{VT::BOOL, /*cast to*/ VT::FP16}, CNNL_CAST_BOOL_TO_HALF},
        {{VT::BOOL, /*cast to*/ VT::INT32}, CNNL_CAST_BOOL_TO_INT32},
        {{VT::UINT8, /*cast to*/ VT::INT32}, CNNL_CAST_UINT8_TO_INT32},
        {{VT::INT32, /*cast to*/ VT::INT64}, CNNL_CAST_INT32_TO_INT64},
        {{VT::INT64, /*cast to*/ VT::INT32}, CNNL_CAST_INT64_TO_INT32},
        {{VT::INT32, /*cast to*/ VT::BOOL}, CNNL_CAST_INT32_TO_BOOL},
        {{VT::UINT8, /*cast to*/ VT::INT64}, CNNL_CAST_UINT8_TO_INT64},
        {{VT::INT8, /*cast to*/ VT::INT16}, CNNL_CAST_INT8_TO_INT16},
        {{VT::FP32, /*cast to*/ VT::FP64}, CNNL_CAST_FLOAT_TO_DOUBLE},
        {{VT::FP64, /*cast to*/ VT::FP32}, CNNL_CAST_DOUBLE_TO_FLOAT},
        {{VT::INT64, /*cast to*/ VT::FP32}, CNNL_CAST_INT64_TO_FLOAT},
        {{VT::INT64, /*cast to*/ VT::FP16}, CNNL_CAST_INT64_TO_HALF},
        {{VT::FP32, /*cast to*/ VT::INT64}, CNNL_CAST_FLOAT_TO_INT64},
        {{VT::FP16, /*cast to*/ VT::INT64}, CNNL_CAST_HALF_TO_INT64},
};

cnnlCastDataType_t GetCastDataType(const VT::Type& src_type,
                                   const VT::Type& dst_type);

cnnlCastDataType_t GetCastDataType(const DataType& src_type,
                                   const DataType& dst_type);

bool MLUSupportsCast(const VT::Type& src_type, const VT::Type& dst_type);

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

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    int position);

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim,
                    const int64_t dim_sizes[],
                    const cnnlDataType_t tensor_dtype,
                    int position);

  MLUCnnlTensorDesc(const phi::DenseTensor& tensor,
                    const cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype);

  explicit MLUCnnlTensorDesc(const phi::DenseTensor& tensor);

  MLUCnnlTensorDesc(const phi::DenseTensor& tensor,
                    cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype,
                    int position);

  MLUCnnlTensorDesc(const phi::DenseTensor& tensor,
                    cnnlTensorLayout_t layout,
                    const cnnlDataType_t tensor_dtype,
                    int position,
                    float scale);

  ~MLUCnnlTensorDesc();

  const cnnlTensorDescriptor_t get() const { return raw_tensor_desc; }

 private:
  cnnlTensorDescriptor_t raw_tensor_desc = nullptr;
};

class MLUOpTensorDesc {
 public:
  MLUOpTensorDesc() {}

  // SE_DISALLOW_COPY_AND_ASSIGN
  MLUOpTensorDesc(const MLUOpTensorDesc& desc) = delete;
  MLUOpTensorDesc& operator=(const MLUOpTensorDesc&) = delete;

  MLUOpTensorDesc(MLUOpTensorDesc&& rhs)
      : raw_tensor_desc(rhs.raw_tensor_desc) {
    rhs.raw_tensor_desc = nullptr;
  }

  MLUOpTensorDesc& operator=(MLUOpTensorDesc&& rhs);

  MLUOpTensorDesc(const int tensor_dim,
                  const int dim_sizes[],
                  const mluOpDataType_t tensor_dtype);

  MLUOpTensorDesc(const int tensor_dim,
                  const int dim_sizes[],
                  const mluOpDataType_t tensor_dtype,
                  const mluOpTensorLayout_t layout);

  MLUOpTensorDesc(const int tensor_dim,
                  const int dim_sizes[],
                  const mluOpDataType_t tensor_dtype,
                  int position);

  MLUOpTensorDesc(const int tensor_dim,
                  const int64_t dim_sizes[],
                  const mluOpDataType_t tensor_dtype);

  MLUOpTensorDesc(const int tensor_dim,
                  const int64_t dim_sizes[],
                  const mluOpDataType_t tensor_dtype,
                  const mluOpTensorLayout_t layout);

  MLUOpTensorDesc(const int tensor_dim,
                  const int64_t dim_sizes[],
                  const mluOpDataType_t tensor_dtype,
                  int position);

  MLUOpTensorDesc(const phi::DenseTensor& tensor,
                  const mluOpTensorLayout_t layout,
                  const mluOpDataType_t tensor_dtype);

  explicit MLUOpTensorDesc(const phi::DenseTensor& tensor);

  MLUOpTensorDesc(const phi::DenseTensor& tensor,
                  mluOpTensorLayout_t layout,
                  const mluOpDataType_t tensor_dtype,
                  int position);

  MLUOpTensorDesc(const phi::DenseTensor& tensor,
                  mluOpTensorLayout_t layout,
                  const mluOpDataType_t tensor_dtype,
                  int position,
                  float scale);

  ~MLUOpTensorDesc();

  const mluOpTensorDescriptor_t get() const { return raw_tensor_desc; }

 private:
  mluOpTensorDescriptor_t raw_tensor_desc = nullptr;
};

class MLUCnnlActivationDesc {
 public:
  MLUCnnlActivationDesc(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc& operator=(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc(const cnnlActivationMode_t act_mode, const float ceof);
  MLUCnnlActivationDesc(const cnnlActivationMode_t act_mode,
                        const float ceof,
                        const float sliced_dim,
                        const float selu_alpha,
                        const float selu_lambda);

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
                     int window_rows,
                     int window_cols,
                     int64_t pad_up,
                     int64_t pad_down,
                     int64_t pad_left,
                     int64_t pad_right,
                     int row_stride,
                     int col_stride,
                     int row_dilation,
                     int col_dilation,
                     bool ceil_mode);

  MLUCnnlPoolingDesc(const cnnlPoolingMode_t mode,
                     const cnnlNanPropagation_t maxpooling_nan_opt,
                     const int tensor_rank,
                     const std::vector<int>& window,
                     const std::vector<int>& padding,
                     const std::vector<int>& stride);

  const cnnlPoolingDescriptor_t get() const;

  ~MLUCnnlPoolingDesc();

 private:
  cnnlPoolingDescriptor_t pooling_desc_ = nullptr;
};

class MLUCnnlRandomGeneratorDesc {
 public:
  MLUCnnlRandomGeneratorDesc(const ExecutionContext& ctx, const int seed);
  const cnnlRandGenerator_t get() const;
  phi::DenseTensor& get_state();
  ~MLUCnnlRandomGeneratorDesc();

 private:
  phi::DenseTensor mlu_state;
  cnnlRandGenerator_t mlu_generator = nullptr;
};

const std::shared_ptr<MLUCnnlRandomGeneratorDesc>& GetMLURandomGenerator(
    const ExecutionContext& ctx, const int64_t device_id, const int seed);

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

  MLUCnnlNMSDesc(const cnnlNmsOutputMode_t mode,
                 const float iou_threshold,
                 const int max_output_size,
                 const float confidence_threshold,
                 const int input_layout);

  const cnnlNmsDescriptor_t get() const;

  ~MLUCnnlNMSDesc();

 private:
  cnnlNmsDescriptor_t nms_desc_ = nullptr;
};

class MLUCnnlConvolutionDesc {
 public:
  MLUCnnlConvolutionDesc(const int dims,
                         const int pad[],
                         const int stride[],
                         const int dilation[],
                         const int group_count,
                         const cnnlDataType_t tensor_dtype);

  MLUCnnlConvolutionDesc(const int dims,
                         const int64_t pad[],
                         const int64_t stride[],
                         const int64_t dilation[],
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
  MLUCnnlBatchSpaceDesc(uint32_t block_shape[],
                        uint32_t paddings[],
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

class MLUCnnlDCNDesc {
 public:
  MLUCnnlDCNDesc(int dimNb,
                 const int* pad,
                 const int* stride,
                 const int* dilation,
                 int deformable_group,
                 int conv_group,
                 int im2col_step);
  const cnnlDCNDescriptor_t get() const;

  ~MLUCnnlDCNDesc();

 private:
  cnnlDCNDescriptor_t dcn_desc_ = nullptr;
};

class MLUCnnlGridSampleDesc {
 public:
  MLUCnnlGridSampleDesc(const std::string& interp_mode_str,
                        const std::string& padding_mode_str,
                        bool align_corners);

  const cnnlGridSampleDescriptor_t get() const;

  ~MLUCnnlGridSampleDesc();

 private:
  cnnlGridSampleDescriptor_t grid_sample_desc_ = nullptr;
};

class MLUSeqDataDesc {
 public:
  MLUSeqDataDesc(const MLUSeqDataDesc& desc) = delete;
  MLUSeqDataDesc& operator=(const MLUSeqDataDesc& desc) = delete;

  MLUSeqDataDesc(cnnlSeqDataLayout_t layout,
                 cnnlDataType_t dtype,
                 int dimNb,
                 const int dimSize[],
                 int seqLengthArraySize,
                 const int seqLengthArray[],
                 void* paddingFill);

  const cnnlSeqDataDescriptor_t get() const;

  ~MLUSeqDataDesc();

 private:
  cnnlSeqDataDescriptor_t seq_data_desc_ = nullptr;
};

class MLURNNDesc {
 public:
  MLURNNDesc(const MLURNNDesc& desc) = delete;
  MLURNNDesc& operator=(const MLURNNDesc& desc) = delete;

  MLURNNDesc(const int hidden_size,
             const int num_layers,
             const cnnlRNNInputMode_t input_mode,
             const cnnlDirectionMode_t direction,
             const cnnlRNNMode_t rnn_mode);

  MLURNNDesc(cnnlRNNMode_t cell_mode,
             cnnlRNNBiasMode_t bias_mode,
             cnnlDirectionMode_t direction,
             cnnlRNNInputMode_t input_mode,
             cnnlDataType_t data_type,
             cnnlDataType_t math_prec,
             int input_size,
             int hidden_size,
             int proj_size,
             int layer_num,
             void* dropout_desc,
             cnnlRNNPaddingMode_t padding_mode);

  void SetRNNProjectionLayers(const int rec_proj_size,
                              const int out_proj_size) {
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlSetRNNProjectionLayers(rnn_desc_, rec_proj_size, out_proj_size));
  }

  void SetPeepholeMode(const cnnlRNNPeepholeMode_t peephole_mode) {
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlSetRNNPeepholeMode(rnn_desc_, peephole_mode));
  }

  void SetRNNBiasMode(const cnnlRNNBiasMode_t bias_mode) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetRNNBiasMode(rnn_desc_, bias_mode));
  }

  void SetRNNMaskMode(const cnnlRNNMaskMode_t mask_mode) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetRNNMaskMode(rnn_desc_, mask_mode));
  }

  void SetRNNClip(const cnnlRNNClipMode_t clip_mode,
                  const cnnlNanPropagation_t clip_nan_opt,
                  const double left_clip,
                  const double right_clip) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetRNNClip(
        rnn_desc_, clip_mode, clip_nan_opt, left_clip, right_clip));
  }

  void SetRNNPaddingMode(const cnnlRNNPaddingMode_t padding_mode) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetRNNPaddingMode(rnn_desc_, padding_mode));
  }

  const cnnlRNNDescriptor_t get() const;

  ~MLURNNDesc();

 private:
  cnnlRNNDescriptor_t rnn_desc_ = nullptr;
};

class MLUCnnl {
 public:
  static void Active(const ExecutionContext& ctx,
                     cnnlActivationDescriptor_t active_desc,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void ActiveGrad(const ExecutionContext& ctx,
                         cnnlActivationDescriptor_t active_desc,
                         const void* alpha,
                         const void* beta,
                         const cnnlTensorDescriptor_t y_desc,
                         const void* y,
                         const cnnlTensorDescriptor_t diff_y_desc,
                         const void* diff_y,
                         const cnnlTensorDescriptor_t x_desc,
                         const void* x,
                         const cnnlTensorDescriptor_t diff_x_desc,
                         void* diff_x);

  static void Concat(const ExecutionContext& ctx,
                     const int pack_num,
                     const int axis,
                     const cnnlTensorDescriptor_t inputs_desc[],
                     const void* const inputs[],
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void Concat(const MLUDeviceContext& dev_ctx,
                     const int pack_num,
                     const int axis,
                     const cnnlTensorDescriptor_t inputs_desc[],
                     const void* const inputs[],
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void Cast(const ExecutionContext& ctx,
                   cnnlCastDataType_t cast_type,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void Clip(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const void* min,
                   const void* max,
                   void* y);

  static void HardtanhBackward(const ExecutionContext& ctx,
                               const cnnlTensorDescriptor_t x_desc,
                               const void* x,
                               const cnnlTensorDescriptor_t diff_y_desc,
                               const void* diff_y,
                               const float max_val,
                               const float min_val,
                               const cnnlTensorDescriptor_t diff_x_desc,
                               void* diff_x);

  static void Div(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t in0_desc,
                  const void* in0,
                  const cnnlTensorDescriptor_t in1_desc,
                  const void* in1,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Fill(const ExecutionContext& ctx,
                   const cnnlPointerMode_t pointer_mode,
                   const void* value_ptr,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void LRN(const ExecutionContext& ctx,
                  const int local_size,
                  const double alpha,
                  const double beta,
                  const double k,
                  const cnnlTensorDescriptor_t input_quant_desc,
                  const void* input_quant,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void QuantifyOffline(const ExecutionContext& context,
                              cnnlQuantizeMode_t mode,
                              const cnnlTensorDescriptor_t input_desc,
                              const void* input,
                              const cnnlTensorDescriptor_t ouput_desc,
                              void* output);

  static void QuantifyOnline(const ExecutionContext& context,
                             const int bitwidth,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             const bool compute_scale,
                             void* position,
                             void* scale,
                             const cnnlTensorDescriptor_t ouput_desc,
                             void* output);

  static void SGD(const ExecutionContext& context,
                  const cnnlTensorDescriptor_t grad_desc,
                  const void* grad,
                  const void* lr,
                  const cnnlTensorDescriptor_t var_desc,
                  void* var);

  static void ApplyAdaGrad(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t grad_desc,
                           const void* grad,
                           const cnnlTensorDescriptor_t accum_desc,
                           void* accum,
                           const cnnlTensorDescriptor_t var_desc,
                           void* var,
                           const void* lr,
                           const bool update_slots);

  static void ApplyRMSProp(const ExecutionContext& context,
                           const cnnlTensorDescriptor_t grad_desc,
                           const void* grad,
                           const void* lr,
                           const void* rho,
                           const void* momentum,
                           const void* epsilon,
                           const cnnlTensorDescriptor_t var_desc,
                           void* var,
                           const cnnlTensorDescriptor_t ms_desc,
                           void* ms,
                           const cnnlTensorDescriptor_t mom_desc,
                           void* mom);

  static void ApplyCenterRMSProp(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t grad_desc,
                                 const void* grad,
                                 const void* lr,
                                 const void* rho,
                                 const void* momentum,
                                 const void* epsilon,
                                 const cnnlTensorDescriptor_t var_desc,
                                 void* var,
                                 const cnnlTensorDescriptor_t mg_desc,
                                 void* mg,
                                 const cnnlTensorDescriptor_t ms_desc,
                                 void* ms,
                                 const cnnlTensorDescriptor_t mom_desc,
                                 void* mom);

  static void ApplyAdam(const ExecutionContext& ctx,
                        const cnnlTensorDescriptor_t var_desc,
                        void* var,
                        const cnnlTensorDescriptor_t m_desc,
                        void* m,
                        const cnnlTensorDescriptor_t v_desc,
                        void* v,
                        const cnnlTensorDescriptor_t grad_desc,
                        const void* grad,
                        const void* lr,
                        const void* beta1,
                        const void* beta2,
                        const void* beta1_power,
                        const void* beta2_power,
                        const void* epsilon,
                        const bool use_nesterov);

  static void ApplyAdaMax(const ExecutionContext& ctx,
                          const cnnlTensorDescriptor_t grad_desc,
                          const cnnlTensorDescriptor_t var_desc,
                          void* var,
                          const cnnlTensorDescriptor_t m_desc,
                          void* m,
                          const cnnlTensorDescriptor_t v_desc,
                          void* v,
                          const void* diff,
                          const void* lr,
                          const void* beta1,
                          const void* beta2,
                          const void* beta1_power,
                          const void* epsilon);

  static void ApplyMomentum(const ExecutionContext& ctx,
                            const cnnlTensorDescriptor_t grad_desc,
                            const void* grad,
                            const bool use_nesterov,
                            const void* lr,
                            const void* momentum,
                            void* var,
                            void* accum);

  static void ApplyKerasMomentum(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t grad_desc,
                                 const void* grad,
                                 const bool use_nesterov,
                                 const void* lr,
                                 const void* momentum,
                                 void* var,
                                 void* accum);

  static void ApplyAdadelta(const ExecutionContext& ctx,
                            const cnnlTensorDescriptor_t grad_desc,
                            const void* diff,
                            const void* lr,
                            const void* rho,
                            const void* epsilon,
                            void* var,
                            void* accum,
                            void* accum_update);

  static void SparseSoftmaxXentWithLogits(
      const ExecutionContext& ctx,
      cnnlSoftmaxMode_t mode,
      const cnnlTensorDescriptor_t x_desc,
      const void* input,
      const cnnlTensorDescriptor_t label_desc,
      const void* label,
      const cnnlTensorDescriptor_t y_desc,
      void* output,
      const cnnlTensorDescriptor_t diff_y_desc,
      void* back_out);

  static void RandomUniform(const ExecutionContext& ctx,
                            const int num,
                            const cnnlDataType_t data_type,
                            const cnnlRandGenerator_t mlu_generator,
                            void* mlu_state,
                            void* output);

  static void FusedDropout(const ExecutionContext& ctx,
                           const cnnlRandGenerator_t generator,
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const float p,
                           void* state,
                           const cnnlTensorDescriptor_t mask_desc,
                           const void* mask,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output);

  static void Cumsum(const ExecutionContext& ctx,
                     const int axis,
                     const bool exclusive,
                     const bool reverse,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t ouput_desc,
                     void* output);

  static void BroadcastTo(const ExecutionContext& ctx,
                          const cnnlTensorDescriptor_t input_desc,
                          const void* input,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void GatherFunctor(const ExecutionContext& ctx,
                            const int axis,
                            const int batch_dims,
                            const cnnlTensorDescriptor_t params_desc,
                            const void* params,
                            const cnnlTensorDescriptor_t indices_desc,
                            const void* indices,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void ScatterRefFunctor(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t params_desc,
                                const void* params,
                                const cnnlTensorDescriptor_t updates_desc,
                                const void* updates,
                                const cnnlTensorDescriptor_t indices_desc,
                                const void* indices,
                                const cnnlScatterRefMode_t mode);

  static void ScatterFunctor(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t params_desc,
                             void* params,
                             const cnnlTensorDescriptor_t updates_desc,
                             const void* updates,
                             const cnnlTensorDescriptor_t indices_desc,
                             const void* indices,
                             const int dim,
                             const cnnlScatterMode_t mode = CNNL_SCATTER);

  static void Range(const ExecutionContext& ctx,
                    const void* start,
                    const void* end,
                    const void* step,
                    const cnnlDataType_t output_dtype,
                    void* output);

  static void Round(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void TopK(const ExecutionContext& ctx,
                   const int k,
                   const int dim,
                   const bool largest,
                   const bool sorted,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t values_output_desc,
                   void* values_out,
                   const cnnlTensorDescriptor_t indices_output_desc,
                   void* indices_out);

  static void StridedSlice(const ExecutionContext& ctx,
                           const int begin[],
                           const int end[],
                           const int strides[],
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output);

  static void Split(const ExecutionContext& ctx,
                    int split_num,
                    int axis,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input_ptr,
                    const cnnlTensorDescriptor_t output_descs[],
                    void* output_ptrs[]);

  static void Split(const MLUDeviceContext& dev_ctx,
                    int split_num,
                    int axis,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input_ptr,
                    const cnnlTensorDescriptor_t output_descs[],
                    void* output_ptrs[]);

  static void Scale(const ExecutionContext& ctx,
                    const int axis,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t alpha_desc,
                    const void* alpha,
                    const cnnlTensorDescriptor_t beta_desc,
                    const void* beta,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void AddN(const ExecutionContext& ctx,
                   uint32_t input_num,
                   const cnnlTensorDescriptor_t inputs_desc[],
                   const void* inputs[],
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void Log(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  cnnlLogBase_t log_base,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void StridedSliceGrad(const ExecutionContext& ctx,
                               const int begin[],
                               const int end[],
                               const int strides[],
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output);

  static void Logic(const ExecutionContext& ctx,
                    const cnnlLogicOp_t log_method,
                    const cnnlTensorDescriptor_t input1_desc,
                    const void* input1,
                    const cnnlTensorDescriptor_t input2_desc,
                    const void* input2,
                    const cnnlTensorDescriptor_t ouput_desc,
                    void* output);

  static void Select(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t condition_desc,
                     const void* condition_ptr,
                     const cnnlTensorDescriptor_t then_desc,
                     const void* then_ptr,
                     const cnnlTensorDescriptor_t else_desc,
                     const void* else_ptr,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output_ptr);

  static void AssignAdd(const ExecutionContext& ctx,
                        const void* alpha,
                        const void* beta,
                        const cnnlTensorDescriptor_t update_desc,
                        const void* update,
                        const cnnlTensorDescriptor_t param_desc,
                        void* param);

  static void AssignSub(const ExecutionContext& ctx,
                        const void* alpha,
                        const void* beta,
                        const cnnlTensorDescriptor_t update_desc,
                        const void* update,
                        const cnnlTensorDescriptor_t param_desc,
                        void* param);

  static void Assign(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t update_desc,
                     const void* update,
                     const cnnlTensorDescriptor_t param_desc,
                     void* param);

  static void GatherNd(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t params_desc,
                       const void* params,
                       const cnnlTensorDescriptor_t indices_desc,
                       const void* indices,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void BatchToSpace(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output,
                           const cnnlSpaceBatchParam_t param);

  static void BatchToSpaceNd(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             cnnlSpaceBatchNdDescriptor_t param,
                             void* extra_device_input,
                             size_t extra_input_size,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void PoolingForward(const ExecutionContext& ctx,
                             cnnlPoolingMode_t pool_mode,
                             int64_t output_h,
                             int64_t output_w,
                             cnnlPoolingDescriptor_t pooling_desc,
                             const void* alpha,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             const void* beta,
                             const void* extra_input_ptr,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void AdaptivePoolingForward(const ExecutionContext& ctx,
                                     cnnlPoolingMode_t pool_mode,
                                     const cnnlTensorDescriptor_t input_desc,
                                     const void* input,
                                     const cnnlTensorDescriptor_t output_desc,
                                     void* output,
                                     const cnnlTensorDescriptor_t index_desc,
                                     void* index);

  static void Pool3D(const ExecutionContext& ctx,
                     cnnlPoolingMode_t pool_mode,
                     const std::vector<int64_t>& output_shape,
                     cnnlPoolingDescriptor_t pooling_desc,
                     const void* alpha,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const void* beta,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void Pad(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const void* paddings,
                  const void* padding_value,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Matmul(const ExecutionContext& ctx,
                     const bool transpose_a,
                     const bool transpose_b,
                     const cnnlTensorDescriptor_t in0_desc,
                     const void* in0,
                     const cnnlTensorDescriptor_t in1_desc,
                     const void* in1,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void BatchMatmul(const ExecutionContext& ctx,
                          const bool transpose_a,
                          const bool transpose_b,
                          const cnnlTensorDescriptor_t in0_desc,
                          const void* in0,
                          const cnnlTensorDescriptor_t in1_desc,
                          const void* in1,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void MulAx(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t alpha_desc,
                    const void* alpha,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void OpTensor(const ExecutionContext& ctx,
                       const cnnlOpTensorDescriptor_t op_tensor_desc,
                       const cnnlTensorDescriptor_t a_desc,
                       const void* a,
                       const cnnlTensorDescriptor_t b_desc,
                       const void* b,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output,
                       const cnnlDataType_t dtype,
                       const float alpha1_float = 1.f,
                       const float alpha2_float = 1.f,
                       const float beta_float = 0.f);

  static void BiasAddGrad(const ExecutionContext& ctx,
                          const int axis,
                          const cnnlTensorDescriptor_t out_backprop_desc,
                          const void* out_backprop,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void OneHot(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t desc_indices,
                     const void* indices,
                     const int depth,
                     const void* on_value,
                     const void* off_value,
                     const int axis,
                     cnnlDataType_t output_data_type,
                     void* output);

  static void NonMaxSuppression(const ExecutionContext& ctx,
                                const cnnlNmsDescriptor_t nms_desc,
                                const cnnlTensorDescriptor_t boxes_desc,
                                const void* boxes,
                                const cnnlTensorDescriptor_t confidence_desc,
                                const void* confidence,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output,
                                void* output_size);

  static void SoftmaxCrossEntropyWithLogits(
      const ExecutionContext& ctx,
      cnnlSoftmaxMode_t mode,
      cnnlComputationPreference_t prefer,
      const cnnlTensorDescriptor_t input_desc,
      const void* logits_in,
      const cnnlTensorDescriptor_t label_desc,
      const void* labels_in,
      const cnnlTensorDescriptor_t loss_out_desc,
      void* loss_out,
      const cnnlTensorDescriptor_t back_out_desc,
      void* back_out);

  static void SoftmaxForward(const ExecutionContext& ctx,
                             cnnlSoftmaxAlgorithm_t algorithm,
                             cnnlSoftmaxMode_t mode,
                             const void* alpha,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             const void* beta,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void SoftmaxBackward(const ExecutionContext& ctx,
                              cnnlSoftmaxAlgorithm_t algorithm,
                              cnnlSoftmaxMode_t mode,
                              const cnnlTensorDescriptor_t y_desc,
                              const void* y,
                              const cnnlTensorDescriptor_t diff_y_desc,
                              const void* diff_y,
                              const cnnlTensorDescriptor_t diff_x_desc,
                              void* diff_x);

  static void Softplus(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t features_desc,
                       const void* features,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void SoftplusGrad(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t gradients_desc,
                           const void* gradients,
                           const cnnlTensorDescriptor_t features_desc,
                           const void* features,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output);

  static void RsqrtGrad(const ExecutionContext& ctx,
                        const cnnlTensorDescriptor_t data_desc,
                        const void* y,
                        const void* diff_y,
                        void* output);

  static void SqrtGrad(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t data_desc,
                       const void* y,
                       const void* diff_y,
                       void* output);

  static void ConvolutionForward(const ExecutionContext& ctx,
                                 cnnlConvolutionDescriptor_t conv_desc_,
                                 const void* alpha,
                                 const void* beta,
                                 const cnnlTensorDescriptor_t bias_desc,
                                 const void* bias_ptr,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t filtet_desc,
                                 const void* filter,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output);

  static void FusedConvBNQuantify(const ExecutionContext& ctx,
                                  cnnlConvolutionDescriptor_t conv_desc,
                                  const void* epsilon_ptr,
                                  const int fused_ops_number,
                                  const cnnlDataType_t tensor_dtype,
                                  const int input_position,
                                  const float input_scale,
                                  const int filter_position,
                                  const float filter_scale,
                                  const cnnlTensorDescriptor_t scale_desc,
                                  const void* scale_ptr,
                                  const cnnlTensorDescriptor_t offset_desc,
                                  const void* offset_ptr,
                                  const cnnlTensorDescriptor_t mean_desc,
                                  const void* mean_ptr,
                                  const cnnlTensorDescriptor_t variance_desc,
                                  const void* variance_ptr,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void* input,
                                  const cnnlTensorDescriptor_t filtet_desc,
                                  const void* filter,
                                  const cnnlTensorDescriptor_t output_desc,
                                  void* output);

  static void Tile(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void UnsortedSegmentSum(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t data_desc,
                                 const void* data,
                                 const cnnlTensorDescriptor_t ids_desc,
                                 const int* segment_ids,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output);

  static void Reduce(const ExecutionContext& ctx,
                     const bool need_workspace,
                     const cnnlReduceDescriptor_t reduction_desc,
                     const void* alpha,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const size_t indices_size,
                     void* indices,
                     const void* beta,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void FloorDiv(const ExecutionContext& ctx,
                       cnnlComputationPreference_t prefer,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void FloorMod(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void Maximum(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc,
                      void* output);

  static void Minimum(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc,
                      void* output);

  static void Pow(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input1_desc,
                  const void* input1,
                  const cnnlTensorDescriptor_t input2_desc,
                  const void* input2,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void PowR(const ExecutionContext& ctx,
                   cnnlComputationPreference_t prefer,
                   const cnnlTensorDescriptor_t input1_desc,
                   const void* input1,
                   const cnnlTensorDescriptor_t input2_desc,
                   const void* input2,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void DivNoNan(const ExecutionContext& ctx,
                       cnnlComputationPreference_t prefer,
                       const cnnlTensorDescriptor_t input1_desc,
                       const void* input1,
                       const cnnlTensorDescriptor_t input2_desc,
                       const void* input2,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void SquaredDifference(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t input1_desc,
                                const void* input1,
                                const cnnlTensorDescriptor_t input2_desc,
                                const void* input2,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output);

  static void L2Loss(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     void* output);

  static void Abs(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Neg(const ExecutionContext& ctx,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Floor(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void Ceil(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void IsNan(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void Square(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void Sqrt(const ExecutionContext& ctx,
                   cnnlComputationPreference_t prefer,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void Rsqrt(const ExecutionContext& ctx,
                    cnnlComputationPreference_t prefer,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void Cos(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Sin(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void TrigonForward(const ExecutionContext& ctx,
                            const cnnlTrigonDescriptor_t trigon_desc,
                            const cnnlTensorDescriptor_t input_desc,
                            const void* input,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void Exp(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Sign(const ExecutionContext& ctx,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output);

  static void IndexSelect(const ExecutionContext& ctx,
                          const int dim,
                          cnnlTensorDescriptor_t input_desc,
                          const void* input,
                          const cnnlTensorDescriptor_t index_desc,
                          const void* index,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output);

  static void IsFinite(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void IsNanInf(const ExecutionContext& ctx,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input,
                       void* output);

  static void Erf(const ExecutionContext& ctx,
                  cnnlComputationPreference_t prefer,
                  const cnnlTensorDescriptor_t input_desc,
                  const void* input,
                  const cnnlTensorDescriptor_t output_desc,
                  void* output);

  static void Log1p(const ExecutionContext& ctx,
                    cnnlComputationPreference_t prefer,
                    const cnnlTensorDescriptor_t input_desc,
                    const void* input,
                    const cnnlTensorDescriptor_t output_desc,
                    void* output);

  static void LogicalNot(const ExecutionContext& ctx,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);

  static void DynamicStitch(const ExecutionContext& ctx,
                            const cnnlTensorDescriptor_t* indices_desc,
                            const int** indices,
                            const cnnlTensorDescriptor_t* data_desc,
                            const void** data,
                            const int size,
                            int* indices_dims,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void CropAndResize(const ExecutionContext& ctx,
                            const std::string method_name,
                            const float extrapolation_value,
                            const cnnlTensorDescriptor_t image_desc,
                            const void* image,
                            const cnnlTensorDescriptor_t boxes_desc,
                            const void* boxes,
                            const cnnlTensorDescriptor_t box_index_desc,
                            const void* box_index,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void CropAndResizeBackwardImage(
      const ExecutionContext& ctx,
      const std::string method_name,
      const cnnlTensorDescriptor_t image_desc,
      const void* image,
      const cnnlTensorDescriptor_t boxes_desc,
      const void* boxes,
      const cnnlTensorDescriptor_t box_idx_desc,
      const void* box_idx,
      const cnnlTensorDescriptor_t grads_image_desc,
      void* grads_image);

  static void CropAndResizeBackwardBoxes(
      const ExecutionContext& ctx,
      const cnnlTensorDescriptor_t input_desc,
      const void* input,
      const cnnlTensorDescriptor_t image_desc,
      const void* image,
      const cnnlTensorDescriptor_t boxes_desc,
      const void* boxes,
      const cnnlTensorDescriptor_t box_idx_desc,
      const void* box_idx,
      const cnnlTensorDescriptor_t output_desc,
      void* output);

  static void PoolingBackward(const ExecutionContext& ctx,
                              const cnnlPoolingDescriptor_t pooling_desc,
                              const void* alpha,
                              const cnnlTensorDescriptor_t y_desc,
                              const void* y,
                              const cnnlTensorDescriptor_t diff_y_desc,
                              const void* diff_y,
                              const cnnlTensorDescriptor_t x_desc,
                              const void* x,
                              const void* beta,
                              const cnnlTensorDescriptor_t diff_x_desc,
                              void* diff_x);

  static void AdaptivePoolingBackward(const ExecutionContext& ctx,
                                      const cnnlPoolingMode_t pool_mode,
                                      const cnnlTensorDescriptor_t y_desc,
                                      const void* y,
                                      const cnnlTensorDescriptor_t index_desc,
                                      const void* index,
                                      const cnnlTensorDescriptor_t diff_x_desc,
                                      void* diff_x);

  static void PoolingIndex(const ExecutionContext& ctx,
                           const cnnlPoolingDescriptor_t pooling_desc,
                           const cnnlTensorDescriptor_t x_desc,
                           const void* x,
                           const cnnlTensorDescriptor_t y_desc,
                           void* y);

  static void SpaceToBatch(const ExecutionContext& ctx,
                           const cnnlTensorDescriptor_t input_desc,
                           const void* input,
                           const cnnlTensorDescriptor_t output_desc,
                           void* output,
                           const int64_t block_shape[]);

  static void SpaceToBatchNd(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             cnnlSpaceBatchNdDescriptor_t param,
                             void* extra_device_input,
                             size_t extra_input_size,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void Interp(const ExecutionContext& ctx,
                     const cnnlInterpMode_t mode,
                     const bool align_corners,
                     const bool half_pixel_centers,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void InterpBackward(const ExecutionContext& ctx,
                             const cnnlInterpBackwardMode_t mode,
                             const bool align_corners,
                             const bool half_pixel_centers,
                             const cnnlTensorDescriptor_t input_desc,
                             const void* input,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void QuantizeParam(const ExecutionContext& ctx,
                            const cnnlQuantizeMode_t mode,
                            const int bitwidth,
                            const cnnlTensorDescriptor_t input_desc,
                            const void* input,
                            void* position,
                            void* scale,
                            void* offset);

  static void QuantizeMatMul(const ExecutionContext& ctx,
                             const bool transpose_a,
                             const bool transpose_b,
                             const cnnlTensorDescriptor_t a_desc,
                             const void* a,
                             const void* a_position,
                             const void* a_scale,
                             const void* a_offset,
                             const cnnlTensorDescriptor_t b_desc,
                             const void* b,
                             const void* b_position,
                             const void* b_scale,
                             const void* b_offset,
                             const cnnlDataType_t quant_type,
                             const cnnlDataType_t data_type,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output);

  static void QuantizeBatchMatMul(const ExecutionContext& ctx,
                                  const bool adj_x,
                                  const bool adj_y,
                                  const cnnlTensorDescriptor_t a_desc,
                                  const void* a,
                                  const void* a_position,
                                  const void* a_scale,
                                  const void* a_offset,
                                  const cnnlTensorDescriptor_t b_desc,
                                  const void* b,
                                  const void* b_position,
                                  const void* b_scale,
                                  const void* b_offset,
                                  const cnnlDataType_t quant_type,
                                  const cnnlDataType_t data_type,
                                  const cnnlTensorDescriptor_t output_desc,
                                  void* output);

  static void QuantizeBatchMatMulBCast(const ExecutionContext& ctx,
                                       const bool adj_x,
                                       const bool adj_y,
                                       const cnnlTensorDescriptor_t a_desc,
                                       const void* a,
                                       const void* a_position,
                                       const void* a_scale,
                                       const void* a_offset,
                                       const cnnlTensorDescriptor_t b_desc,
                                       const void* b,
                                       const void* b_position,
                                       const void* b_scale,
                                       const void* b_offset,
                                       const cnnlDataType_t quant_type,
                                       const cnnlDataType_t data_type,
                                       const cnnlTensorDescriptor_t output_desc,
                                       void* output);

  static void FusedBatchNorm(const ExecutionContext& ctx,
                             const bool is_training,
                             const cnnlTensorDescriptor_t x_desc,
                             const void* x,
                             const cnnlTensorDescriptor_t scale_desc,
                             const void* scale,
                             const void* offset,
                             const void* estimated_mean,
                             const void* estimated_variance,
                             float epsilon,
                             float momentum,
                             const cnnlTensorDescriptor_t output_desc,
                             void* output,
                             void* batch_mean,
                             void* batch_var,
                             void* saved_mean,
                             void* saved_var);

  static void FusedBatchNormGrad(const ExecutionContext& ctx,
                                 const bool is_training,
                                 const cnnlTensorDescriptor_t y_backprop_desc,
                                 const void* y_backprop,
                                 const cnnlTensorDescriptor_t x_desc,
                                 const void* x,
                                 const cnnlTensorDescriptor_t scale_desc,
                                 const void* scale,
                                 const void* saved_mean,
                                 const void* saved_var,
                                 float epsilon,
                                 const cnnlTensorDescriptor_t x_backprop_desc,
                                 void* x_backprop,
                                 void* scale_backprop,
                                 void* offset_backprop);

  static void LayerNormForward(const ExecutionContext& ctx,
                               int axis,
                               const cnnlTensorDescriptor_t x_desc,
                               const void* x,
                               const cnnlTensorDescriptor_t weight_bias_desc,
                               const void* weight,
                               const void* bias,
                               float eps,
                               const cnnlTensorDescriptor_t y_desc,
                               void* y,
                               const cnnlTensorDescriptor_t mean_rstd_desc,
                               void* saved_mean,
                               void* saved_rstd);

  static void LayerNormBackward(const ExecutionContext& ctx,
                                int axis,
                                const cnnlTensorDescriptor_t x_desc,
                                const void* x,
                                const cnnlTensorDescriptor_t diff_z_desc,
                                const void* diff_z,
                                const cnnlTensorDescriptor_t weight_bias_desc,
                                const void* weight,
                                const cnnlTensorDescriptor_t mean_rstd_desc,
                                const void* saved_mean,
                                const void* saved_rstd,
                                const cnnlTensorDescriptor_t diff_x_desc,
                                void* diff_x,
                                void* diff_weight,
                                void* diff_bias);

  static void Transpose(const ExecutionContext& ctx,
                        const std::vector<int> perm,
                        const int input_dim,
                        const cnnlTensorDescriptor_t input_desc,
                        const void* input,
                        const cnnlTensorDescriptor_t output_desc,
                        void* output);

  static void TrilTriu(const ExecutionContext& ctx,
                       const int diagonal_k,
                       const bool tri_up_mode,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void MatrixBandPart(const ExecutionContext& ctx,
                             const cnnlTensorDescriptor_t data_desc,
                             const void* input,
                             const int num_lower,
                             const int num_upper,
                             void* output);

  static void NumTrue(const ExecutionContext& ctx,
                      const cnnlTensorDescriptor_t x_desc,
                      const void* x,
                      const cnnlTensorDescriptor_t num_true_desc,
                      void* num_true);

  static void Where(const ExecutionContext& ctx,
                    const cnnlTensorDescriptor_t x_desc,
                    const void* x,
                    const cnnlTensorDescriptor_t num_true_desc,
                    const void* num_true,
                    const bool as_tuple,
                    const cnnlTensorDescriptor_t y_desc,
                    void* y);
  static void Conv2D(const ExecutionContext& ctx,
                     const cnnlConvolutionDescriptor_t conv_desc,
                     const cnnlDataType_t tensor_dtype,
                     const cnnlDataType_t dt_onchip,
                     const void* input_position,
                     const void* input_scale,
                     const void* input_offset,
                     const void* filter_position,
                     const void* filter_scale,
                     const void* filter_offset,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t filter_desc,
                     const void* filter,
                     const cnnlTensorDescriptor_t bias_desc,
                     const void* bias,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void ConvBackpropInput(const ExecutionContext& ctx,
                                const cnnlConvolutionDescriptor_t conv_desc,
                                const cnnlTensorDescriptor_t filter_desc,
                                const void* filter,
                                const cnnlTensorDescriptor_t out_backprop_desc,
                                const void* out_backprop,
                                const cnnlTensorDescriptor_t in_backprop_desc,
                                void* in_backprop);

  static void QuantizeConvBackpropInput(
      const ExecutionContext& ctx,
      const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlDataType_t tensor_dtype,
      const cnnlDataType_t dt_onchip,
      const void* filter_position,
      const void* filter_scale,
      const void* filter_offset,
      const void* out_backprop_position,
      const void* out_backprop_scale,
      const void* out_backprop_offset,
      const cnnlTensorDescriptor_t input_desc,
      const void* filter,
      const cnnlTensorDescriptor_t out_backprop_desc,
      const void* out_backprop,
      const cnnlTensorDescriptor_t in_backprop_desc,
      void* in_backprop);

  static void ConvBackpropFilter(
      const ExecutionContext& ctx,
      const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlTensorDescriptor_t input_desc,
      const void* input,
      const cnnlTensorDescriptor_t out_backprop_desc,
      const void* out_backprop,
      const cnnlTensorDescriptor_t filter_backprop_desc,
      void* filter_backprop);

  static void QuantizeConvBackpropFilter(
      const ExecutionContext& ctx,
      const cnnlConvolutionDescriptor_t conv_desc,
      const cnnlDataType_t tensor_dtype,
      const cnnlDataType_t dt_onchip,
      const void* input_position,
      const void* input_scale,
      const void* input_offset,
      const void* out_backprop_position,
      const void* out_backprop_scale,
      const void* out_backprop_offset,
      const cnnlTensorDescriptor_t input_desc,
      const void* input,
      const cnnlTensorDescriptor_t out_backprop_desc,
      const void* out_backprop,
      const cnnlTensorDescriptor_t filter_backprop_desc,
      void* filter_backprop);

  static void DCNForward(const ExecutionContext& ctx,
                         const cnnlDCNDescriptor_t dcn_desc,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t offset_desc,
                         const void* offset,
                         const cnnlTensorDescriptor_t mask_desc,
                         const void* mask,
                         const cnnlTensorDescriptor_t weight_desc,
                         const void* weight,
                         const cnnlTensorDescriptor_t bias_desc,
                         const void* bias,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);

  static void DCNBackwardData(const ExecutionContext& ctx,
                              const cnnlDCNDescriptor_t dcn_desc,
                              const cnnlTensorDescriptor_t input_desc,
                              const void* input,
                              const cnnlTensorDescriptor_t offset_desc,
                              const void* offset,
                              const cnnlTensorDescriptor_t mask_desc,
                              const void* mask,
                              const cnnlTensorDescriptor_t weight_desc,
                              const void* weight,
                              const cnnlTensorDescriptor_t grad_output_desc,
                              const void* grad_output,
                              const cnnlTensorDescriptor_t grad_input_desc,
                              void* grad_input,
                              const cnnlTensorDescriptor_t grad_offset_desc,
                              void* grad_offset,
                              const cnnlTensorDescriptor_t grad_mask_desc,
                              void* grad_mask);

  static void DCNBackwardWeight(const ExecutionContext& ctx,
                                const cnnlDCNDescriptor_t dcn_desc,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t offset_desc,
                                const void* offset,
                                const cnnlTensorDescriptor_t mask_desc,
                                const void* mask,
                                const cnnlTensorDescriptor_t grad_output_desc,
                                const void* grad_output,
                                const cnnlTensorDescriptor_t grad_weight_desc,
                                void* grad_weight,
                                const cnnlTensorDescriptor_t grad_bias_desc,
                                void* grad_bias);

  static void InTopK(const ExecutionContext& ctx,
                     const cnnlTensorDescriptor_t predictions_desc,
                     const void* predictions,
                     const cnnlTensorDescriptor_t targets_desc,
                     const void* targets,
                     const cnnlTensorDescriptor_t k_desc,
                     const void* k,
                     const int k_int,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);

  static void ScatterNd(const ExecutionContext& ctx,
                        cnnlScatterNdMode_t mode,
                        const cnnlTensorDescriptor_t indices_desc,
                        const void* indices,
                        const cnnlTensorDescriptor_t updates_desc,
                        const void* updates,
                        const cnnlTensorDescriptor_t input_desc,
                        const void* input,
                        const cnnlTensorDescriptor_t output_desc,
                        void* output);

  static void BitWise(const ExecutionContext& ctx,
                      const cnnlBitComputeOp_t optype,
                      const cnnlTensorDescriptor_t input1_desc,
                      const void* input1,
                      const cnnlTensorDescriptor_t input2_desc,
                      const void* input2,
                      const cnnlTensorDescriptor_t output_desc,
                      void* output);

  static void QR(const ExecutionContext& ctx,
                 const cnnlTensorDescriptor_t a_desc,
                 const void* a,
                 const cnnlTensorDescriptor_t q_desc,
                 void* q,
                 const cnnlTensorDescriptor_t r_desc,
                 void* r,
                 const bool some);

  static void Reciprocal(const ExecutionContext& ctx,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);

  static void BceLoss(const ExecutionContext& ctx,
                      const cnnlBceLossReduction_t reduction,
                      const cnnlTensorDescriptor_t input_desc,
                      const void* input,
                      const cnnlTensorDescriptor_t target_desc,
                      const void* target,
                      const cnnlTensorDescriptor_t weight_desc,
                      const void* weight,
                      const cnnlTensorDescriptor_t output_desc,
                      void* output);

  static void BceLossBackward(const ExecutionContext& ctx,
                              const cnnlBceLossReduction_t reduction,
                              const cnnlTensorDescriptor_t grad_desc,
                              const void* grad,
                              const cnnlTensorDescriptor_t input_desc,
                              const void* input,
                              const cnnlTensorDescriptor_t target_desc,
                              const void* target,
                              const cnnlTensorDescriptor_t weight_desc,
                              const void* weight,
                              const cnnlTensorDescriptor_t output_desc,
                              void* output);

  static void SmoothL1LossForward(const ExecutionContext& ctx,
                                  const cnnlTensorDescriptor_t x_desc,
                                  const void* x,
                                  const cnnlTensorDescriptor_t t_desc,
                                  const void* target,
                                  const float beta,
                                  const cnnlSmoothL1LossAlgorithm_t algorithm,
                                  const cnnlTensorDescriptor_t y_desc,
                                  void* y);

  static void SmoothL1LossBackward(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t x_desc,
                                   const void* x,
                                   const cnnlTensorDescriptor_t target_desc,
                                   const void* target,
                                   const cnnlTensorDescriptor_t dy_desc,
                                   const void* dy,
                                   const float beta,
                                   const cnnlSmoothL1LossAlgorithm_t algorithm,
                                   const cnnlTensorDescriptor_t dx_desc,
                                   void* dx);

  static void EmbeddingForward(const ExecutionContext& ctx,
                               const int padding_idx,
                               const cnnlTensorDescriptor_t weight_desc,
                               const void* weight,
                               const cnnlTensorDescriptor_t indices_desc,
                               const int* indices,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output);

  static void RNNForward(const ExecutionContext& ctx,
                         const cnnlRNNDescriptor_t rnn_desc,
                         const int dev_seq_lengths[],
                         const void* weight_param_ptr,
                         size_t weightspace_size,
                         const cnnlSeqDataDescriptor_t x_desc,
                         const void* x,
                         const cnnlSeqDataDescriptor_t y_desc,
                         void* y,
                         const cnnlTensorDescriptor_t h_desc,
                         const void* hx,
                         void* hy,
                         const cnnlTensorDescriptor_t c_desc,
                         const void* cx,
                         void* cy,
                         void* reservespace_ptr);

  static void RNNBackward(const ExecutionContext& ctx,
                          const cnnlRNNDescriptor_t rnn_desc,
                          cnnlWgradMode_t add_grad,
                          const int dev_seq_lengths[],
                          const void* weight_param_ptr,
                          void* dweight_param_ptr,
                          size_t weightspace_size,
                          const cnnlSeqDataDescriptor_t x_desc,
                          const void* x,
                          void* dx,
                          const cnnlSeqDataDescriptor_t y_desc,
                          const void* y,
                          const void* dy,
                          const cnnlTensorDescriptor_t hx_desc,
                          const void* hx,
                          const void* dhy,
                          void* dhx,
                          const cnnlTensorDescriptor_t cx_desc,
                          const void* cx,
                          const void* dcy,
                          void* dcx,
                          void* reservespace_ptr,
                          size_t reservespace_size);

  static void Mask(const ExecutionContext& ctx,
                   cnnlMaskedOp_t masked_mode,
                   const cnnlTensorDescriptor_t input_desc,
                   const void* input,
                   const cnnlTensorDescriptor_t masked_desc,
                   const void* masked,
                   const cnnlTensorDescriptor_t value_desc,
                   const void* value,
                   const cnnlTensorDescriptor_t output_desc,
                   void* output,
                   uint32_t* number);

  static void Transform(const ExecutionContext& ctx,
                        const void* alpha,
                        const void* beta,
                        const cnnlTensorDescriptor_t input_desc,
                        const void* input,
                        const cnnlTensorDescriptor_t output_desc,
                        void* output);

  static void EmbeddingBackward(const ExecutionContext& ctx,
                                int padding_idx,
                                bool scale_grad_by_freq,
                                const cnnlTensorDescriptor_t indices_desc,
                                const void* indices,
                                const cnnlTensorDescriptor_t diff_desc,
                                const void* diff,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output);

  static void BceWithLogits(const ExecutionContext& ctx,
                            cnnlBceWithLogitsReduction_t reduction,
                            const cnnlTensorDescriptor_t input_desc,
                            const void* input,
                            const cnnlTensorDescriptor_t target_desc,
                            const void* target,
                            const cnnlTensorDescriptor_t weight_desc,
                            const void* weight,
                            const cnnlTensorDescriptor_t pos_weight_desc,
                            const void* pos_weight,
                            const cnnlTensorDescriptor_t output_desc,
                            void* output);

  static void BceWithLogitsBackward(
      const ExecutionContext& ctx,
      cnnlBceWithLogitsReduction_t reduction,
      const cnnlTensorDescriptor_t grad_desc,
      const void* grad,
      const cnnlTensorDescriptor_t input_desc,
      const void* input,
      const cnnlTensorDescriptor_t target_desc,
      const void* target,
      const cnnlTensorDescriptor_t weight_desc,
      const void* weight,
      const cnnlTensorDescriptor_t pos_weight_desc,
      const void* pos_weight,
      const cnnlTensorDescriptor_t diff_input_desc,
      void* diff_input);

  static void RoiAlign(const ExecutionContext& ctx,
                       const int pooled_height,
                       const int pooled_width,
                       const int sampling_ratio,
                       const float spatial_scale,
                       const bool aligned,
                       const cnnlTensorDescriptor_t input_desc,
                       const void* input,
                       const cnnlTensorDescriptor_t boxes_desc,
                       const void* boxes,
                       const cnnlTensorDescriptor_t output_desc,
                       void* output);

  static void RoiAlignBackward(const ExecutionContext& ctx,
                               const int sampling_ratio,
                               const float spatial_scale,
                               const bool aligned,
                               const cnnlTensorDescriptor_t grads_desc,
                               const void* grads,
                               const cnnlTensorDescriptor_t boxes_desc,
                               const void* boxes,
                               const cnnlTensorDescriptor_t grads_image_desc,
                               void* grads_image);

  static void GridSample(const ExecutionContext& ctx,
                         const cnnlGridSampleDescriptor_t grid_sample_desc,
                         const cnnlTensorDescriptor_t input_desc,
                         const void* input,
                         const cnnlTensorDescriptor_t grid_desc,
                         const void* grid,
                         const cnnlTensorDescriptor_t output_desc,
                         void* output);

  static void SyncBatchNormStats(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t x_desc,
                                 const void* x,
                                 const float eps,
                                 const cnnlTensorDescriptor_t mean_desc,
                                 void* mean,
                                 const cnnlTensorDescriptor_t invstd_desc,
                                 void* invstd);

  static void SyncBatchNormGatherStatsWithCounts(
      const ExecutionContext& ctx,
      float momentum,
      float eps,
      const cnnlTensorDescriptor_t mean_all_desc,
      const void* mean_all,
      const cnnlTensorDescriptor_t invstd_all_desc,
      const void* invstd_all,
      const cnnlTensorDescriptor_t moving_mean_desc,
      void* moving_mean,
      const cnnlTensorDescriptor_t moving_var_desc,
      void* moving_var,
      const cnnlTensorDescriptor_t count_all_desc,
      const void* count_all,
      const cnnlTensorDescriptor_t mean_desc,
      void* mean,
      const cnnlTensorDescriptor_t invstd_desc,
      void* invstd);

  static void SyncBatchNormElemt(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t x_desc,
                                 const void* x,
                                 const cnnlTensorDescriptor_t mean_desc,
                                 const void* mean,
                                 const cnnlTensorDescriptor_t invstd_desc,
                                 const void* invstd,
                                 const cnnlTensorDescriptor_t weight_desc,
                                 const void* weight,
                                 const cnnlTensorDescriptor_t bias_desc,
                                 const void* bias,
                                 const cnnlTensorDescriptor_t y_desc,
                                 void* y);

  static void SyncBatchnormBackwardReduce(
      const ExecutionContext& ctx,
      const cnnlTensorDescriptor_t desc_dz,
      const void* dz,
      const cnnlTensorDescriptor_t desc_x,
      const void* x,
      const cnnlTensorDescriptor_t desc_mean,
      const void* mean,
      const cnnlTensorDescriptor_t desc_invstd,
      const void* invstd,
      const cnnlTensorDescriptor_t desc_dweight,
      void* dweight,
      const cnnlTensorDescriptor_t desc_dbias,
      void* dbias,
      const cnnlTensorDescriptor_t desc_sum_dy,
      void* sum_dy,
      const cnnlTensorDescriptor_t desc_sum_dy_xmu,
      void* sum_dy_xmu,
      const bool needs_input_grad0,
      const bool needs_input_grad1,
      const bool needs_input_grad2);

  static void SyncBatchNormBackwardElemt(
      const ExecutionContext& ctx,
      const cnnlTensorDescriptor_t diff_y_desc,
      const void* diff_y,
      const cnnlTensorDescriptor_t x_desc,
      const void* x,
      const cnnlTensorDescriptor_t mean_desc,
      const void* mean,
      const cnnlTensorDescriptor_t invstd_desc,
      const void* invstd,
      const cnnlTensorDescriptor_t weight_desc,
      const void* weight,
      const cnnlTensorDescriptor_t sum_dy_desc,
      const void* sum_dy,
      const cnnlTensorDescriptor_t sum_dy_xmu_desc,
      const void* sum_dy_xmu,
      const cnnlTensorDescriptor_t count_desc,
      const void* count,
      const cnnlTensorDescriptor_t diff_x_desc,
      void* diff_x);
};

class MLUOP {
 public:
  static void OpYoloBox(const ExecutionContext& ctx,
                        const mluOpTensorDescriptor_t x_desc,
                        const void* x,
                        const mluOpTensorDescriptor_t img_size_desc,
                        const void* img_size,
                        const mluOpTensorDescriptor_t anchors_desc,
                        const void* anchors,
                        const int class_num,
                        const float conf_thresh,
                        const int downsample_ratio,
                        const bool clip_bbox,
                        const float scale,
                        const bool iou_aware,
                        const float iou_aware_factor,
                        const mluOpTensorDescriptor_t boxes_desc,
                        void* boxes,
                        const mluOpTensorDescriptor_t scores_desc,
                        void* scores);

  static void OpPriorBox(const ExecutionContext& ctx,
                         const mluOpTensorDescriptor_t min_sizes_desc,
                         const void* min_sizes,
                         const mluOpTensorDescriptor_t aspect_ratios_desc,
                         const void* aspect_ratios,
                         const mluOpTensorDescriptor_t variances_desc,
                         const void* variances,
                         const mluOpTensorDescriptor_t max_sizes_desc,
                         const void* max_sizes,
                         const int height,
                         const int width,
                         const int im_height,
                         const int im_width,
                         const float step_h,
                         const float step_w,
                         const float offset,
                         const bool clip,
                         const bool min_max_aspect_ratios_order,
                         const mluOpTensorDescriptor_t output_desc,
                         void* output,
                         const mluOpTensorDescriptor_t var_desc,
                         void* var);
};
const std::map<const std::string, std::pair<std::vector<int>, std::vector<int>>>
    TransPermMap = {
        // trans_mode, (forward_perm, backward_perm)
        {"3D_NCHW2NHWC", {{0, 2, 1}, {0, 2, 1}}},
        {"4D_NCHW2NHWC", {{0, 2, 3, 1}, {0, 3, 1, 2}}},
        {"5D_NCHWD2NDHWC", {{0, 4, 2, 3, 1}, {0, 4, 2, 3, 1}}},
        {"5D_NHWDC2NDHWC", {{0, 3, 1, 2, 4}, {0, 2, 3, 4, 1}}}};

inline void SetMLUTransposePerm(const framework::DDim& dims,
                                const DataLayout& data_layout,
                                std::vector<int>* forward_perm,
                                std::vector<int>* backward_perm,
                                std::vector<int>* out_shape) {
  const int dim_size = dims.size();
  PADDLE_ENFORCE_EQ((dim_size >= 3) && (dim_size <= 5),
                    true,
                    platform::errors::InvalidArgument(
                        "MLUTransposePerm func only support (dim_size >= 3) && "
                        "(dim_size <= 5), but now dim_size is %d.",
                        dim_size));

  PADDLE_ENFORCE_EQ(
      (data_layout == DataLayout::kNCHW) || (data_layout == DataLayout::kNHWC),
      true,
      platform::errors::InvalidArgument(
          "MLUTransposePerm func only support DataLayout: kNCHW or kNHWC, but "
          "now data_layout is %s.",
          data_layout));

  // case 1: NCHW of Paddle != NHWC of MLU when dims==3,4
  // case 2： NHWDC and NCHWD of Paddle != NDHWC of MLU when dims==5
  std::string map_key = "";
  if (data_layout == DataLayout::kNCHW) {
    switch (dim_size) {
      case 3:
        map_key = "3D_NCHW2NHWC";
        break;
      case 4:
        map_key = "4D_NCHW2NHWC";
        break;
      case 5:
        map_key = "5D_NCHWD2NDHWC";
        break;
    }
  } else if (data_layout == DataLayout::kNHWC && dim_size == 5) {
    map_key = "5D_NHWDC2NDHWC";
  }
  assert(map_key != "");
  forward_perm->assign(TransPermMap.at(map_key).first.begin(),
                       TransPermMap.at(map_key).first.end());
  backward_perm->assign(TransPermMap.at(map_key).second.begin(),
                        TransPermMap.at(map_key).second.end());

  auto in_dims = phi::vectorize(dims);
  for (size_t i = 0; i < in_dims.size(); i++) {
    out_shape->push_back(in_dims[forward_perm->at(i)]);
  }
}

template <typename T>
inline void TransposeFromMLUTensor(const ExecutionContext& ctx,
                                   const std::vector<int> perm,
                                   const phi::DenseTensor* transformed_input,
                                   phi::DenseTensor* transformed_output,
                                   bool need_reshape_or_alloc) {
  const int dim_size = perm.size();
  if (need_reshape_or_alloc) {
    std::vector<int> output_shape;
    auto input_dims = transformed_input->dims();
    for (int i = 0; i < dim_size; ++i) {
      output_shape.push_back(input_dims[perm[i]]);
    }
    transformed_output->mutable_data<T>(
        framework::DDim(output_shape.data(), dim_size), ctx.GetPlace());
  }
  MLUCnnlTensorDesc trans_in_desc(
      *transformed_input, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc trans_out_desc(
      *transformed_output, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::Transpose(ctx,
                     perm,
                     dim_size,
                     trans_in_desc.get(),
                     GetBasePtr(transformed_input),
                     trans_out_desc.get(),
                     GetBasePtr(transformed_output));
}

template <typename T>
inline void FillMLUTensorWithHostValue(const ExecutionContext& ctx,
                                       T value,
                                       phi::DenseTensor* out) {
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Fill(
      ctx, CNNL_POINTER_MODE_HOST, &value, out_desc.get(), GetBasePtr(out));
}

}  // namespace operators
}  // namespace paddle
