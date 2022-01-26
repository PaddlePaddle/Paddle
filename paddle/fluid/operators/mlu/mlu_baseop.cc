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

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

cnnlCastDataType_t GetCastDataType(const VT::Type& src_type,
                                   const VT::Type& dst_type) {
  cnnlCastDataType_t cast_type = CNNL_CAST_FLOAT_TO_HALF;
  for (auto it = MLU_SUPPORTED_CAST_TYPE.begin();
       it != MLU_SUPPORTED_CAST_TYPE.end(); ++it) {
    if (it->first.first == src_type && it->first.second == dst_type) {
      cast_type = it->second;
      break;
    }
  }
  return cast_type;
}

bool MLUSupportsCast(const VT::Type& src_type, const VT::Type& dst_type) {
  for (auto it = MLU_SUPPORTED_CAST_TYPE.begin();
       it != MLU_SUPPORTED_CAST_TYPE.end(); ++it) {
    if (it->first.first == src_type && it->first.second == dst_type) {
      return true;
    }
  }
  return false;
}

class MLUCnnlTensorDescPool {
 public:
  cnnlTensorDescriptor_t Pop() {
    cnnlTensorDescriptor_t raw_desc;
    if (q_.try_dequeue(raw_desc)) {
      return raw_desc;
    } else {
      cnnlCreateTensorDescriptor(&raw_desc);
      return raw_desc;
    }
  }

  void Recycle(cnnlTensorDescriptor_t desc) {
    cnnlResetTensorDescriptor(desc);
    q_.enqueue(desc);
  }

  ~MLUCnnlTensorDescPool() {
    auto size = q_.size_approx();
    if (size > 0) {
      std::vector<cnnlTensorDescriptor_t> vec(size);
      q_.try_dequeue_bulk(vec.data(), size);
      for (auto desc : vec) {
        cnnlDestroyTensorDescriptor(desc);
      }
    }
  }

 private:
  moodycamel::ConcurrentQueue<cnnlTensorDescriptor_t> q_;
};

static MLUCnnlTensorDescPool g_cnnl_tensor_desc_pool;

MLUCnnlTensorDesc& MLUCnnlTensorDesc::operator=(MLUCnnlTensorDesc&& rhs) {
  if (raw_tensor_desc) {
    g_cnnl_tensor_desc_pool.Recycle(raw_tensor_desc);
  }
  raw_tensor_desc = rhs.raw_tensor_desc;
  rhs.raw_tensor_desc = nullptr;
  return *this;
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype) {
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
      raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype, tensor_dim, dim_sizes));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     const cnnlTensorLayout_t layout) {
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
      raw_tensor_desc, layout, tensor_dtype, tensor_dim, dim_sizes));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     int position)
    : MLUCnnlTensorDesc(tensor_dim, dim_sizes, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptor(raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype,
                              tensor_dim, dim_sizes_int32.data()));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     const cnnlTensorLayout_t layout) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(raw_tensor_desc, layout,
                                                     tensor_dtype, tensor_dim,
                                                     dim_sizes_int32.data()));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     int position) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptor(raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype,
                              tensor_dim, dim_sizes_int32.data()));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor& tensor,
                                     const cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype) {
  auto dims = framework::vectorize<int>(tensor.dims());
  int tensor_dim = dims.size();
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  if (tensor_dim == 0) {
    int scalar_dims[1] = {1};
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
        raw_tensor_desc, layout, tensor_dtype, 1, scalar_dims));
  } else {
    std::vector<int> tensor_dim_sizes_int(dims.begin(), dims.end());
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlSetTensorDescriptor(raw_tensor_desc, layout, tensor_dtype,
                                tensor_dim, tensor_dim_sizes_int.data()));
  }
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor& tensor)
    : MLUCnnlTensorDesc(tensor, CNNL_LAYOUT_ARRAY,
                        ToCnnlDataType(tensor.type())) {}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor& tensor,
                                     cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype,
                                     int position)
    : MLUCnnlTensorDesc(tensor, layout, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor& tensor,
                                     cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype,
                                     int position, float scale)
    : MLUCnnlTensorDesc(tensor, layout, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptorPositionAndScale(
      raw_tensor_desc, position, scale));
}

MLUCnnlTensorDesc::~MLUCnnlTensorDesc() {
  if (raw_tensor_desc) {
    g_cnnl_tensor_desc_pool.Recycle(raw_tensor_desc);
  }
}

MLUCnnlActivationDesc::MLUCnnlActivationDesc(
    const cnnlActivationMode_t act_mode, const float ceof) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateActivationDescriptor(&active_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetActivationDescriptor(
      active_desc_, act_mode, CNNL_NOT_PROPAGATE_NAN, ceof));
}

const cnnlActivationDescriptor_t MLUCnnlActivationDesc::get() const {
  return active_desc_;
}

MLUCnnlActivationDesc::~MLUCnnlActivationDesc() {
  if (active_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyActivationDescriptor(active_desc_));
  }
}

MLUCnnlPoolingDesc::MLUCnnlPoolingDesc(
    const cnnlPoolingMode_t mode, const cnnlNanPropagation_t maxpooling_nan_opt,
    int window_rows, int window_cols, int64_t pad_up, int64_t pad_down,
    int64_t pad_left, int64_t pad_right, int row_stride, int col_stride) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreatePoolingDescriptor(&pooling_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetPooling2dDescriptor(
      pooling_desc_, mode, maxpooling_nan_opt, window_rows, window_cols, pad_up,
      pad_down, pad_left, pad_right, row_stride, col_stride));
}

MLUCnnlPoolingDesc::MLUCnnlPoolingDesc(
    const cnnlPoolingMode_t mode, const cnnlNanPropagation_t maxpooling_nan_opt,
    const int tensor_rank, const std::vector<int>& window,
    const std::vector<int>& padding, const std::vector<int>& stride) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreatePoolingDescriptor(&pooling_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetPoolingNdDescriptor(
      pooling_desc_, mode, maxpooling_nan_opt, tensor_rank, window.data(),
      padding.data(), stride.data()));
}

const cnnlPoolingDescriptor_t MLUCnnlPoolingDesc::get() const {
  return pooling_desc_;
}

MLUCnnlPoolingDesc::~MLUCnnlPoolingDesc() {
  if (pooling_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyPoolingDescriptor(pooling_desc_));
  }
}

MLUCnnlRandomGeneratorDesc::MLUCnnlRandomGeneratorDesc(const bool is_mlu200,
                                                       const int seed) {
  if (is_mlu200) {
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlRandCreateGenerator(&mlu_generator, CNNL_RAND_RNG_FAST));
  } else {
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlRandCreateGenerator(&mlu_generator, CNNL_RAND_RNG_MTGP32));
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlRandSetPseudoRandomGeneratorSeed(mlu_generator, seed));
  }
}

const cnnlRandGenerator_t MLUCnnlRandomGeneratorDesc::get() const {
  return mlu_generator;
}

MLUCnnlRandomGeneratorDesc::~MLUCnnlRandomGeneratorDesc() {
  if (mlu_generator) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlRandDestroyGenerator(mlu_generator));
  }
}

MLUCnnlNMSDesc::MLUCnnlNMSDesc(const cnnlNmsOutputMode_t mode,
                               const float iou_threshold,
                               const int max_output_size,
                               const float confidence_threshold,
                               const int input_layout) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateNmsDescriptor(&nms_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetNmsDescriptor_v2(nms_desc_, mode, iou_threshold, max_output_size,
                              confidence_threshold, input_layout));
}

const cnnlNmsDescriptor_t MLUCnnlNMSDesc::get() const { return nms_desc_; }

MLUCnnlNMSDesc::~MLUCnnlNMSDesc() {
  if (nms_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyNmsDescriptor(nms_desc_));
  }
}

MLUCnnlReduceDesc::MLUCnnlReduceDesc(const std::vector<int>& axis_vec,
                                     const cnnlReduceOp_t reduce_op,
                                     const cnnlDataType_t data_type,
                                     const cnnlNanPropagation_t nan_propagation,
                                     const cnnlReduceIndices_t reduce_indices,
                                     const cnnlIndicesType_t indices_type) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateReduceDescriptor(&reduction_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetReduceDescriptor(
      reduction_desc_, const_cast<int*>(axis_vec.data()), axis_vec.size(),
      reduce_op, data_type, nan_propagation, reduce_indices, indices_type));
}

const cnnlReduceDescriptor_t MLUCnnlReduceDesc::get() const {
  return reduction_desc_;
}

MLUCnnlReduceDesc::~MLUCnnlReduceDesc() {
  if (reduction_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyReduceDescriptor(reduction_desc_));
  }
}

MLUCnnlOpTensorDesc::MLUCnnlOpTensorDesc(
    cnnlOpTensorDesc_t op_tensor_op, cnnlDataType_t op_tensor_comp_type,
    cnnlNanPropagation_t op_tensor_nan_opt) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateOpTensorDescriptor(&op_tensor_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetOpTensorDescriptor(
      op_tensor_desc_, op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt));
}

const cnnlOpTensorDescriptor_t MLUCnnlOpTensorDesc::get() const {
  return op_tensor_desc_;
}

MLUCnnlOpTensorDesc::~MLUCnnlOpTensorDesc() {
  if (op_tensor_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyOpTensorDescriptor(op_tensor_desc_));
  }
}

MLUCnnlConvolutionDesc::MLUCnnlConvolutionDesc(
    const int dims, const int pad[], const int stride[], const int dilation[],
    const int group_count, const cnnlDataType_t tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateConvolutionDescriptor(&conv_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetConvolutionDescriptor(
      conv_desc_, dims, pad, stride, dilation, group_count, tensor_dtype));
}

MLUCnnlConvolutionDesc::MLUCnnlConvolutionDesc(
    const int dims, const int64_t pad[], const int64_t stride[],
    const int64_t dilation[], const int group_count,
    const cnnlDataType_t tensor_dtype) {
  const int spatial_dims = dims - 2;
  const int pad_dims = spatial_dims * 2;
  std::vector<int> pad_int32(pad_dims);
  std::vector<int> stride_int32(spatial_dims);
  std::vector<int> dilation_int32(spatial_dims);
  std::vector<int64_t>::const_iterator int64_pad_cbegin(pad);
  std::vector<int64_t>::const_iterator int64_pad_cend(pad + pad_dims);
  std::vector<int64_t>::const_iterator int64_stride_cbegin(stride);
  std::vector<int64_t>::const_iterator int64_stride_cend(stride + spatial_dims);
  std::vector<int64_t>::const_iterator int64_dilation_cbegin(dilation);
  std::vector<int64_t>::const_iterator int64_dilation_cend(dilation +
                                                           spatial_dims);
  std::transform(int64_pad_cbegin, int64_pad_cend, pad_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  std::transform(int64_stride_cbegin, int64_stride_cend, stride_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  std::transform(int64_dilation_cbegin, int64_dilation_cend,
                 dilation_int32.begin(), &CheckedNarrowing<int64_t, int>);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateConvolutionDescriptor(&conv_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetConvolutionDescriptor(
      conv_desc_, dims, pad_int32.data(), stride_int32.data(),
      dilation_int32.data(), group_count, tensor_dtype));
}

const cnnlConvolutionDescriptor_t MLUCnnlConvolutionDesc::get() const {
  return conv_desc_;
}

MLUCnnlConvolutionDesc::~MLUCnnlConvolutionDesc() {
  if (conv_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyConvolutionDescriptor(conv_desc_));
  }
}

MLUCnnlBatchSpaceDesc::MLUCnnlBatchSpaceDesc(uint32_t block_shape[],
                                             uint32_t paddings[],
                                             const uint32_t block_shape_size,
                                             const uint32_t paddings_size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateSpaceBatchNdDescriptor(&op_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetSpaceBatchNdDescriptor(
      op_desc_, block_shape, block_shape_size, paddings, paddings_size));
}

void MLUCnnlBatchSpaceDesc::getSpace2batchNdextraInputSize(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetSpace2batchNdExtraInputSize(
      handle, input_desc, op_desc_, &extra_input_size_));
}

void MLUCnnlBatchSpaceDesc::getBatch2spaceNdextraInputSize(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetBatch2spaceNdExtraInputSize(
      handle, input_desc, op_desc_, &extra_input_size_));
}

void MLUCnnlBatchSpaceDesc::initSpace2batchNdExtraInput(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    void* extra_host_input) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlInitSpace2batchNdExtraInput(
      handle, input_desc, op_desc_, extra_host_input));
}

void MLUCnnlBatchSpaceDesc::initBatch2spaceNdExtraInput(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    void* extra_host_input) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlInitBatch2spaceNdExtraInput(
      handle, input_desc, op_desc_, extra_host_input));
}

const cnnlSpaceBatchNdDescriptor_t MLUCnnlBatchSpaceDesc::get() const {
  return op_desc_;
}

size_t MLUCnnlBatchSpaceDesc::getExtraInputSize() const {
  return extra_input_size_;
}

MLUCnnlBatchSpaceDesc::~MLUCnnlBatchSpaceDesc() {
  if (op_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroySpaceBatchNdDescriptor(op_desc_));
  }
}

MLUCnnlTrigonDesc::MLUCnnlTrigonDesc(
    const cnnlTrigonFunctionMode_t trigon_function_mode) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateTrigonDescriptor(&trigon_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTrigonDescriptor(trigon_desc_, trigon_function_mode));
}

const cnnlTrigonDescriptor_t MLUCnnlTrigonDesc::get() const {
  return trigon_desc_;
}

MLUCnnlTrigonDesc::~MLUCnnlTrigonDesc() {
  if (trigon_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyTrigonDescriptor(trigon_desc_));
  }
}

/* static */ void MLUCnnl::Active(const ExecutionContext& ctx,
                                  cnnlActivationDescriptor_t active_desc,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void* input,
                                  const cnnlTensorDescriptor_t output_desc,
                                  void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlActivationForward(
      handle, active_desc, NULL, input_desc, input, NULL, output_desc, output));
}

/* static */ void MLUCnnl::ActiveGrad(
    const ExecutionContext& ctx, cnnlActivationDescriptor_t active_desc,
    const void* alpha, const void* beta, const cnnlTensorDescriptor_t y_desc,
    const void* y, const cnnlTensorDescriptor_t diff_y_desc, const void* diff_y,
    const cnnlTensorDescriptor_t x_desc, const void* x,
    const cnnlTensorDescriptor_t diff_x_desc, void* diff_x) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlActivationBackward(handle, active_desc, alpha, y_desc, y, diff_y_desc,
                             diff_y, x_desc, x, beta, diff_x_desc, diff_x));
}

/* static */ void MLUCnnl::Concat(const ExecutionContext& ctx,
                                  const int pack_num, const int axis,
                                  const cnnlTensorDescriptor_t inputs_desc[],
                                  const void* const inputs[],
                                  const cnnlTensorDescriptor_t output_desc,
                                  void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetConcatWorkspaceSize(handle, pack_num, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlConcat(handle, pack_num, axis, inputs_desc,
                                        inputs, workspace_ptr, workspace_size,
                                        output_desc, output));
}

/* static */ void MLUCnnl::Div(
    const ExecutionContext& ctx, cnnlComputationPreference_t prefer,
    const cnnlTensorDescriptor_t in0_desc, const void* in0,
    const cnnlTensorDescriptor_t in1_desc, const void* in1,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetDivWorkspaceSize(
      handle, in0_desc, in1_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlDiv_v2(handle, prefer, in0_desc, in0, in1_desc,
                                        in1, workspace_ptr, workspace_size,
                                        output_desc, output));
}

/* static */ void MLUCnnl::Fill(const ExecutionContext& ctx, float value,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlFill(handle, value, output_desc, output));
}

/* static */ void MLUCnnl::QuantifyOffline(
    const ExecutionContext& ctx, cnnlQuantizeMode_t mode,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlQuantizeV1(handle, mode, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::LRN(const ExecutionContext& ctx,
                               const int local_size, const double alpha,
                               const double beta, const double k,
                               const cnnlTensorDescriptor_t input_quant_desc,
                               const void* input_quant,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetLrnWorkspaceSize(
      handle, input_quant_desc, output_desc, local_size, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  const cnnlLrnMode_t mode = CNNL_LRN_CROSS_CHANNEL;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlLrn(
      handle, mode, local_size, alpha, beta, k, workspace_ptr, workspace_size,
      input_quant_desc, const_cast<void*>(input_quant), output_desc, output));
}

/* static */ void MLUCnnl::QuantifyOnline(
    const ExecutionContext& ctx, const int bitwidth,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const bool compute_scale, void* position, void* scale,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetQuantizeParamWorkspaceSize(handle, input_desc, &workspace_size));

  // use ctx allocate interface for profiling purpose
  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  const cnnlQuantizeMode_t mode =
      compute_scale ? CNNL_QUANTIZE_POSITION_SCALE : CNNL_QUANTIZE_POSITION;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeParam(
      handle, mode, input_desc, input, bitwidth, workspace_ptr, workspace_size,
      position, scale, nullptr));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeV2(handle, mode, input_desc, input,
                                            position, scale, nullptr,
                                            output_desc, output));
}

/* static */ void MLUCnnl::Range(const ExecutionContext& ctx, const void* start,
                                 const void* end, const void* step,
                                 const cnnlDataType_t output_dtype,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlArange(handle, start, end, step, output_dtype, output));
}

/* static */ void MLUCnnl::Round(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlRound(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::SparseSoftmaxXentWithLogits(
    const ExecutionContext& ctx, cnnlSoftmaxMode_t mode,
    const cnnlTensorDescriptor_t x_desc, const void* input,
    const cnnlTensorDescriptor_t label_desc, const void* label,
    const cnnlTensorDescriptor_t y_desc, void* output,
    const cnnlTensorDescriptor_t diff_y_desc, void* back_out) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSparseSoftmaxCrossEntropyWithLogits(
      handle, mode, x_desc, input, label_desc, label, y_desc, output,
      diff_y_desc, back_out));
}

/* static */ void MLUCnnl::Cumsum(const ExecutionContext& ctx, const int axis,
                                  const bool exclusive, const bool reverse,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void* input,
                                  const cnnlTensorDescriptor_t ouput_desc,
                                  void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // NAN propagation mode: Only support CNNL_NOT_PROPAGATE_NAN now.
  cnnlNanPropagation_t mode = CNNL_NOT_PROPAGATE_NAN;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCumsum(handle, input_desc, input, axis,
                                        exclusive, reverse, mode, ouput_desc,
                                        output));
}

/* static */ void MLUCnnl::BroadcastTo(const ExecutionContext& ctx,
                                       const cnnlTensorDescriptor_t input_desc,
                                       const void* input,
                                       const cnnlTensorDescriptor_t output_desc,
                                       void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlExpand(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::AssignAdd(const ExecutionContext& ctx,
                                     const void* alpha, const void* beta,
                                     const cnnlTensorDescriptor_t update_desc,
                                     const void* update,
                                     const cnnlTensorDescriptor_t param_desc,
                                     void* param) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlAssignAdd(
      handle, alpha, update_desc, update, nullptr, 0, beta, param_desc, param));
}

/* static */ void MLUCnnl::AssignSub(const ExecutionContext& ctx,
                                     const void* alpha, const void* beta,
                                     const cnnlTensorDescriptor_t update_desc,
                                     const void* update,
                                     const cnnlTensorDescriptor_t param_desc,
                                     void* param) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlAssignSub(
      handle, alpha, update_desc, update, nullptr, 0, beta, param_desc, param));
}

/* static */ void MLUCnnl::Assign(const ExecutionContext& ctx,
                                  const cnnlTensorDescriptor_t update_desc,
                                  const void* update,
                                  const cnnlTensorDescriptor_t param_desc,
                                  void* param) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlCopy(handle, update_desc, update, param_desc, param));
}

/* static */ void MLUCnnl::SGD(const ExecutionContext& ctx,
                               const cnnlTensorDescriptor_t grad_desc,
                               const void* grad, const void* lr,
                               const cnnlTensorDescriptor_t var_desc,
                               void* var) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGradientDescent(handle, grad_desc, grad, lr, var_desc, var));
}

/* static */ void MLUCnnl::ApplyAdaGrad(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const void* grad, const cnnlTensorDescriptor_t accum_desc, void* accum,
    const cnnlTensorDescriptor_t var_desc, void* var, const void* lr,
    const bool update_slots) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlApplyAdaGrad(handle, grad_desc, grad,
                                              accum_desc, accum, var_desc, var,
                                              lr, update_slots));
}

/* static */ void MLUCnnl::ApplyRMSProp(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const void* grad, const void* lr, const void* rho, const void* momentum,
    const void* epsilon, const cnnlTensorDescriptor_t var_desc, void* var,
    const cnnlTensorDescriptor_t ms_desc, void* ms,
    const cnnlTensorDescriptor_t mom_desc, void* mom) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlRMSProp(handle, lr, rho, epsilon, momentum,
                                         grad_desc, grad, var_desc, var,
                                         ms_desc, ms, mom_desc, mom));
}

/* static */ void MLUCnnl::ApplyCenterRMSProp(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const void* grad, const void* lr, const void* rho, const void* momentum,
    const void* epsilon, const cnnlTensorDescriptor_t var_desc, void* var,
    const cnnlTensorDescriptor_t mg_desc, void* mg,
    const cnnlTensorDescriptor_t ms_desc, void* ms,
    const cnnlTensorDescriptor_t mom_desc, void* mom) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlApplyCenterRMSProp(
      handle, var_desc, var, mg_desc, mg, ms_desc, ms, mom_desc, mom, grad_desc,
      grad, lr, rho, momentum, epsilon));
}

/* static */ void MLUCnnl::ApplyAdam(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const void* grad, const void* lr, const void* beta1, const void* beta2,
    const void* beta1_power, const void* beta2_power, const void* epsilon,
    const bool use_nesterov, const cnnlTensorDescriptor_t var_desc, void* var,
    const cnnlTensorDescriptor_t m_desc, void* m,
    const cnnlTensorDescriptor_t v_desc, void* v) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlApplyAdam(
      handle, grad_desc, var, grad_desc, m, grad_desc, v, grad_desc, grad, lr,
      beta1, beta2, beta1_power, beta2_power, epsilon, use_nesterov));
}

/* static */ void MLUCnnl::ApplyAdaMax(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const cnnlTensorDescriptor_t var_desc, void* var,
    const cnnlTensorDescriptor_t m_desc, void* m,
    const cnnlTensorDescriptor_t v_desc, void* v, const void* diff,
    const void* lr, const void* beta1, const void* beta2,
    const void* beta1_power, const void* epsilon) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlApplyAdaMax(handle, var_desc, var, m_desc, m, v_desc, v, grad_desc,
                      diff, lr, beta1, beta2, beta1_power, epsilon));
}

/* static */ void MLUCnnl::ApplyMomentum(const ExecutionContext& ctx,
                                         const cnnlTensorDescriptor_t grad_desc,
                                         const void* grad,
                                         const bool use_nesterov,
                                         const void* lr, const void* momentum,
                                         void* var, void* accum) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMomentum(handle, grad_desc, var, grad_desc,
                                          accum, grad_desc, grad, lr, momentum,
                                          use_nesterov));
}

/* static */ void MLUCnnl::ApplyKerasMomentum(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t grad_desc,
    const void* grad, const bool use_nesterov, const void* lr,
    const void* momentum, void* var, void* accum) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlKerasMomentum(handle, grad_desc, var, grad_desc, accum, grad_desc,
                        grad, lr, momentum, use_nesterov));
}

/* static */ void MLUCnnl::ApplyAdadelta(const ExecutionContext& ctx,
                                         const cnnlTensorDescriptor_t grad_desc,
                                         const void* diff, const void* lr,
                                         const void* rho, const void* epsilon,
                                         void* var, void* accum,
                                         void* accum_update) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlApplyAdadelta(handle, grad_desc, var, grad_desc, accum, grad_desc,
                        accum_update, grad_desc, diff, lr, rho, epsilon));
}

/* static */ void MLUCnnl::Scale(
    const ExecutionContext& ctx, const int axis,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t alpha_desc, const void* alpha,
    const cnnlTensorDescriptor_t beta_desc, const void* beta,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlScale(handle, axis, input_desc, input,
                                       alpha_desc, alpha, beta_desc, beta,
                                       output_desc, output));
}

/* static */ void MLUCnnl::AddN(const ExecutionContext& ctx, uint32_t input_num,
                                const cnnlTensorDescriptor_t inputs_desc[],
                                const void* inputs[],
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlAddN(handle, inputs_desc, inputs, input_num, output_desc, output));
}

/* static */ void MLUCnnl::Log(const ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  cnnlLogBase_t log_base = CNNL_LOG_E;

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlLog_v2(handle, prefer, log_base, input_desc,
                                        input, output_desc, output));
}

/* static */ void MLUCnnl::Matmul(
    const ExecutionContext& ctx, const bool transpose_a, const bool transpose_b,
    const cnnlTensorDescriptor_t in0_desc, const void* in0,
    const cnnlTensorDescriptor_t in1_desc, const void* in1,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  float alpha = 1.0f;
  float beta = 0.0f;

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlMatMul(handle, transpose_a, transpose_b,
                 reinterpret_cast<void*>(&alpha), in0_desc, in0, in1_desc, in1,
                 reinterpret_cast<void*>(&beta), output_desc, output));
}

/* static */ void MLUCnnl::BatchMatmul(
    const ExecutionContext& ctx, const bool transpose_a, const bool transpose_b,
    const cnnlTensorDescriptor_t in0_desc, const void* in0,
    const cnnlTensorDescriptor_t in1_desc, const void* in1,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetBatchMatMulBCastWorkspaceSize(
      handle, in0_desc, in1_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulBCast(
      handle, transpose_a, transpose_b, in0_desc, in0, in1_desc, in1,
      workspace_ptr, workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::OpTensor(
    const ExecutionContext& ctx, const cnnlOpTensorDescriptor_t op_tensor_desc,
    const cnnlTensorDescriptor_t a_desc, const void* a,
    const cnnlTensorDescriptor_t b_desc, const void* b,
    const cnnlTensorDescriptor_t output_desc, void* output,
    const cnnlDataType_t dtype) {
  static const int alpha1_int = 1, alpha2_int = 1, beta_int = 0;
  static const float alpha1_float = 1.f, alpha2_float = 1.f, beta_float = 0.f;

  const void* alpha1_ptr = static_cast<const void*>(&alpha1_float);
  const void* alpha2_ptr = static_cast<const void*>(&alpha2_float);
  const void* beta_ptr = static_cast<const void*>(&beta_float);

  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;

  bool is_dt_float = (dtype == CNNL_DTYPE_FLOAT || dtype == CNNL_DTYPE_HALF);

  //  if datatype is not float, we set alpha and beta to be int
  if (!is_dt_float) {
    alpha1_ptr = static_cast<const void*>(&alpha1_int);
    alpha2_ptr = static_cast<const void*>(&alpha2_int);
    beta_ptr = static_cast<const void*>(&beta_int);
  }

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetOpTensorWorkspaceSize_v2(
      handle, op_tensor_desc, alpha1_ptr, a_desc, a, alpha2_ptr, b_desc, b,
      beta_ptr, output_desc, output, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlOpTensor(
      handle, op_tensor_desc, alpha1_ptr, a_desc, a, alpha2_ptr, b_desc, b,
      workspace_ptr, workspace_size, beta_ptr, output_desc, output));
}

/* static */ void MLUCnnl::BiasAddGrad(
    const ExecutionContext& ctx, const int axis,
    const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBiasAddBackward(
      handle, out_backprop_desc, out_backprop, axis, output_desc, output));
}

/* static */ void MLUCnnl::RandomUniform(
    const ExecutionContext& ctx, const int num, const cnnlDataType_t data_type,
    const cnnlRandGenerator_t mlu_generator, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlRandGenerateUniform(
      handle, mlu_generator, data_type, nullptr, num, 0, 1, output));
}

/* static */ void MLUCnnl::TopK(
    const ExecutionContext& ctx, const int k, const int dim, const bool largest,
    const bool sorted, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t values_output_desc,
    void* values_out, const cnnlTensorDescriptor_t indices_output_desc,
    void* indices_out) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlTopKTensor(
      handle, input_desc, input, k, dim, largest, sorted, values_output_desc,
      values_out, indices_output_desc, indices_out));
}

/* static */ void MLUCnnl::StridedSlice(
    const ExecutionContext& ctx, const int begin[], const int end[],
    const int strides[], const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlStridedSlice(
      handle, input_desc, input, begin, end, strides, output_desc, output));
}

/* static */ void MLUCnnl::Split(const ExecutionContext& ctx, int split_num,
                                 int axis,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input_ptr,
                                 const cnnlTensorDescriptor_t output_descs[],
                                 void* output_ptrs[]) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetSplitWorkspaceSize(handle, split_num, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSplit(handle, split_num, axis, input_desc,
                                       input_ptr, workspace_ptr, workspace_size,
                                       output_descs, output_ptrs));
}

/* static */ void MLUCnnl::GatherFunctor(
    const ExecutionContext& ctx, const int axis, const int batch_dims,
    const cnnlTensorDescriptor_t params_desc, const void* params,
    const cnnlTensorDescriptor_t indices_desc, const void* indices,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlBatchGatherV2(handle, axis, batch_dims, params_desc, params,
                        indices_desc, indices, output_desc, output));
}

/* static */ void MLUCnnl::ScatterFunctor(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t params_desc,
    const void* params, const cnnlTensorDescriptor_t updates_desc,
    const void* updates, const cnnlTensorDescriptor_t indices_desc,
    const void* indices, const cnnlScatterRefMode_t mode) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlScatterRef(handle, params_desc, params,
                                            indices_desc, indices, updates_desc,
                                            updates, 0, mode));
}

/* static */ void MLUCnnl::StridedSliceGrad(
    const ExecutionContext& ctx, const int begin[], const int end[],
    const int strides[], const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlStridedSliceBackward(
      handle, begin, end, strides, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Logic(
    const ExecutionContext& ctx, const MLULogicMethod log_method,
    const cnnlTensorDescriptor_t input1_desc, const void* input1,
    const cnnlTensorDescriptor_t input2_desc, const void* input2,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetLogicOpWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlLogicOp(
      handle, cnnlLogicOp_t(log_method), input1_desc, input1, input2_desc,
      input2, workspace_ptr, workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::Select(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t then_desc,
    const void* p_then, const cnnlTensorDescriptor_t else_desc,
    const void* p_else, const cnnlTensorDescriptor_t output_desc, void* output,
    const bool* condition, const int condition_size) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSelect(handle, then_desc, p_then, else_desc,
                                        p_else, output_desc, output, condition,
                                        condition_size));
}

/*static */ void MLUCnnl::GatherNd(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t params_desc,
                                   const void* params,
                                   const cnnlTensorDescriptor_t indices_desc,
                                   const void* indices,
                                   const cnnlTensorDescriptor_t output_desc,
                                   void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGatherNd(
      handle, params_desc, params, indices_desc, indices, output_desc, output));
}

/* static */ void MLUCnnl::BatchToSpace(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t output_desc, void* output,
    const cnnlSpaceBatchParam_t param) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetBatch2spaceWorkspaceSize(
      handle, input_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatch2space(handle, input_desc, input,
                                             output_desc, output, param,
                                             workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::BatchToSpaceNd(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    const void* input, cnnlSpaceBatchNdDescriptor_t param,
    void* extra_device_input, size_t extra_input_size,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlBatch2spaceNd_v2(handle, input_desc, input, output_desc, output,
                           param, extra_device_input, extra_input_size));
}

/* static */ void MLUCnnl::SoftmaxForward(
    const ExecutionContext& ctx, cnnlSoftmaxAlgorithm_t algorithm,
    cnnlSoftmaxMode_t mode, const void* alpha,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const void* beta, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSoftmaxForward(handle, algorithm, mode, alpha,
                                                input_desc, input, beta,
                                                output_desc, output));
}

/* static */ void MLUCnnl::Softplus(const ExecutionContext& ctx,
                                    const cnnlTensorDescriptor_t features_desc,
                                    const void* features,
                                    const cnnlTensorDescriptor_t output_desc,
                                    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  const int beta = 1;
  const int threshold = 20;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSoftplusForward(
      handle, features_desc, features, output_desc, output, beta, threshold));
}

/* static */ void MLUCnnl::SoftplusGrad(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t gradients_desc,
    const void* gradients, const cnnlTensorDescriptor_t features_desc,
    const void* features, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  int beta = 1;
  int threshold = 20;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSoftplusBackward(handle, features_desc, features, gradients_desc,
                           gradients, output_desc, output, beta, threshold));
}

/* static */ void MLUCnnl::PoolingForward(
    const ExecutionContext& ctx, cnnlPoolingMode_t pool_mode,
    const std::vector<int64_t>& output_shape,
    const cnnlPoolingDescriptor_t pooling_desc, const void* alpha,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const void* beta, const void* extra_input_ptr,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetPoolingWorkspaceSize(
      handle, pool_mode, output_shape[2], output_shape[1], &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlPoolingForward_v2(
      handle, pooling_desc, alpha, input_desc, input, beta, extra_input_ptr,
      output_desc, output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::Pool3D(
    const ExecutionContext& ctx, cnnlPoolingMode_t pool_mode,
    const std::vector<int64_t>& output_shape,
    const cnnlPoolingDescriptor_t pooling_desc, const void* alpha,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const void* beta, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetPoolingWorkspaceSize(
      handle, pool_mode, output_shape[2], output_shape[1], &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlPoolingForward(handle, pooling_desc, alpha, input_desc, input, beta,
                         output_desc, output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::RsqrtGrad(const ExecutionContext& ctx,
                                     const cnnlTensorDescriptor_t data_desc,
                                     const void* y, const void* diff_y,
                                     void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlRsqrtBackward(handle, data_desc, y, diff_y, output));
}

/* static */ void MLUCnnl::SqrtGrad(const ExecutionContext& ctx,
                                    const cnnlTensorDescriptor_t data_desc,
                                    const void* y, const void* diff_y,
                                    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSqrtBackward(handle, data_desc, y, diff_y, output));
}

/* static */ void MLUCnnl::UnsortedSegmentSum(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t data_desc,
    const void* data, const cnnlTensorDescriptor_t ids_desc,
    const int* segment_ids, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetUnsortedSegmentSumWorkspaceSize(
      handle, data_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlUnsortedSegmentSum(
      handle, data_desc, data, ids_desc, segment_ids, workspace_ptr,
      workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::Pad(const ExecutionContext& ctx,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input, const void* paddings,
                               const void* padding_value,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlPad(handle, input_desc, input, paddings,
                                     padding_value, output_desc, output));
}

/* static */ void MLUCnnl::OneHot(const ExecutionContext& ctx,
                                  const cnnlTensorDescriptor_t desc_indices,
                                  const void* indices, const int depth,
                                  const void* on_value, const void* off_value,
                                  const int axis,
                                  cnnlDataType_t output_data_type,
                                  void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlOneHot(handle, desc_indices, indices, depth,
                                        on_value, off_value, axis,
                                        output_data_type, output));
}

/* static */ void MLUCnnl::ConvolutionForward(
    const ExecutionContext& ctx, cnnlConvolutionDescriptor_t conv_desc,
    const void* alpha, const void* beta, const cnnlTensorDescriptor_t bias_desc,
    const void* bias_ptr, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t filtet_desc,
    const void* filter, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // cnnl: select best algorithm for convolution compution.
  cnnlConvolutionForwardAlgo_t algo;
  cnnlConvolutionFwdPreference_t preference = CNNL_CONVOLUTION_FWD_FASTEST;
  cnnlGetConvolutionForwardAlgorithm(handle, conv_desc, input_desc, filtet_desc,
                                     output_desc, preference, &algo);

  // get workspace size
  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionForwardWorkspaceSize(
      handle, input_desc, filtet_desc, output_desc, bias_desc, conv_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlConvolutionForward(
      handle, conv_desc, algo, alpha, input_desc, input, filtet_desc, filter,
      bias_desc, bias_ptr, workspace_ptr, workspace_size, beta, output_desc,
      output));
}

/* static */ void MLUCnnl::Tile(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlTile(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::SoftmaxCrossEntropyWithLogits(
    const ExecutionContext& ctx, cnnlSoftmaxMode_t mode,
    cnnlComputationPreference_t prefer, const cnnlTensorDescriptor_t input_desc,
    const void* logits_in, const cnnlTensorDescriptor_t label_desc,
    const void* labels_in, const cnnlTensorDescriptor_t loss_out_desc,
    void* loss_out, const cnnlTensorDescriptor_t back_out_desc,
    void* back_out) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSoftmaxCrossEntropyWithLogits_v2(
      handle, mode, prefer, input_desc, logits_in, label_desc, labels_in,
      loss_out_desc, loss_out, back_out_desc, back_out));
}

/* static */ void MLUCnnl::Reduce(
    const ExecutionContext& ctx, const bool need_workspace,
    const cnnlReduceDescriptor_t reduction_desc, const void* alpha,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const size_t indices_size, void* indices, const void* beta,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  void* workspace_ptr = nullptr;
  Tensor workspace;
  if (need_workspace) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetReduceOpWorkspaceSize(
        handle, input_desc, output_desc, reduction_desc, &workspace_size));

    auto& dev_ctx = GetDevCtxFromCTX(ctx);
    workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
        {static_cast<int64_t>(workspace_size)}, dev_ctx);

    workspace_ptr = workspace.mutable_data(ctx.GetPlace());
  }

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlReduce(
      handle, reduction_desc, workspace_ptr, workspace_size, alpha, input_desc,
      input, indices_size, indices, beta, output_desc, output));
}

/* static */ void MLUCnnl::FloorDiv(
    const ExecutionContext& ctx, cnnlComputationPreference_t prefer,
    const cnnlTensorDescriptor_t input1_desc, const void* input1,
    const cnnlTensorDescriptor_t input2_desc, const void* input2,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetFloorDivWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlFloorDiv_v2(handle, prefer, input1_desc, input1, input2_desc, input2,
                      output_desc, output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::FloorMod(const ExecutionContext& ctx,
                                    const cnnlTensorDescriptor_t input1_desc,
                                    const void* input1,
                                    const cnnlTensorDescriptor_t input2_desc,
                                    const void* input2,
                                    const cnnlTensorDescriptor_t output_desc,
                                    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetFloorModWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlFloorMod(handle, input1_desc, input1, input2_desc, input2,
                   output_desc, output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::Maximum(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t input1_desc,
                                   const void* input1,
                                   const cnnlTensorDescriptor_t input2_desc,
                                   const void* input2,
                                   const cnnlTensorDescriptor_t output_desc,
                                   void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetMaximumWorkspaceSize(handle, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlMaximum(handle, input1_desc, input1, input2_desc, input2, output_desc,
                  output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::Minimum(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t input1_desc,
                                   const void* input1,
                                   const cnnlTensorDescriptor_t input2_desc,
                                   const void* input2,
                                   const cnnlTensorDescriptor_t output_desc,
                                   void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetMinimumWorkspaceSize(handle, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlMinimum(handle, input1_desc, input1, input2_desc, input2, output_desc,
                  output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::PowR(
    const ExecutionContext& ctx, cnnlComputationPreference_t prefer,
    const cnnlTensorDescriptor_t input1_desc, const void* input1,
    const cnnlTensorDescriptor_t input2_desc, const void* input2,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetPowRWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlPowR_v2(handle, prefer, input1_desc, input1,
                                         input2_desc, input2, workspace_ptr,
                                         workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::DivNoNan(
    const ExecutionContext& ctx, cnnlComputationPreference_t prefer,
    const cnnlTensorDescriptor_t input1_desc, const void* input1,
    const cnnlTensorDescriptor_t input2_desc, const void* input2,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetDivNoNanWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlDivNoNan_v2(handle, prefer, input1_desc, input1, input2_desc, input2,
                      workspace_ptr, workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::SquaredDifference(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input1_desc,
    const void* input1, const cnnlTensorDescriptor_t input2_desc,
    const void* input2, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetSquaredDifferenceWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSquaredDifference(
      handle, input1_desc, input1, input2_desc, input2, output_desc, output,
      workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::L2Loss(const ExecutionContext& ctx,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void* input, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlL2Loss(handle, input_desc, input, output));
}

/* static */ void MLUCnnl::Abs(const ExecutionContext& ctx,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlAbs(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Neg(const ExecutionContext& ctx,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlNegTensor(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Floor(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlFloor(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Ceil(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlCeil(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::IsNan(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlIsNan(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Square(const ExecutionContext& ctx,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void* input,
                                  const cnnlTensorDescriptor_t output_desc,
                                  void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSquare(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Sqrt(const ExecutionContext& ctx,
                                cnnlComputationPreference_t prefer,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSqrt_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Rsqrt(const ExecutionContext& ctx,
                                 cnnlComputationPreference_t prefer,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlRsqrt_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Cos(const ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlCos_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Sin(const ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSin_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::TrigonForward(
    const ExecutionContext& ctx, const cnnlTrigonDescriptor_t trigon_desc,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlTrigonForward(handle, trigon_desc, input_desc,
                                               input, output_desc, output));
}

/* static */ void MLUCnnl::Exp(const ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlExp_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Sign(const ExecutionContext& ctx,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSign(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::IsFinite(const ExecutionContext& ctx,
                                    const cnnlTensorDescriptor_t input_desc,
                                    const void* input,
                                    const cnnlTensorDescriptor_t output_desc,
                                    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlIsFinite(handle, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::IsNanInf(const ExecutionContext& ctx,
                                    const cnnlTensorDescriptor_t input_desc,
                                    const void* input, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // TODO(CTR-3849): output type should be void*, but now bool*.
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlNanInf(handle, input_desc, input, reinterpret_cast<bool*>(output)));
}

/* static */ void MLUCnnl::Erf(const ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t input_desc,
                               const void* input,
                               const cnnlTensorDescriptor_t output_desc,
                               void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlErf_v2(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Log1p(const ExecutionContext& ctx,
                                 cnnlComputationPreference_t prefer,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlLog1p(handle, prefer, input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::LogicalNot(const ExecutionContext& ctx,
                                      const cnnlTensorDescriptor_t input_desc,
                                      const void* input,
                                      const cnnlTensorDescriptor_t output_desc,
                                      void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlLogicOp(handle, CNNL_LOGIC_OP_NOT, input_desc,
                                         input, input_desc, input, nullptr, 0,
                                         output_desc, output));
}

/* static */ void MLUCnnl::DynamicStitch(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t* indices_desc,
    const int** indices, const cnnlTensorDescriptor_t* data_desc,
    const void** data, const int size, int* indices_dims,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetDynamicStitchWorkspaceSize(handle, size, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlDynamicStitch(
      handle, indices_desc, indices, data_desc, data, size, indices_dims,
      workspace_ptr, workspace_size, output_desc, output));
}

/* static */ void MLUCnnl::CropAndResize(
    const ExecutionContext& ctx, const std::string method_name,
    const float extrapolation_value, const cnnlTensorDescriptor_t image_desc,
    const void* image, const cnnlTensorDescriptor_t boxes_desc,
    const void* boxes, const cnnlTensorDescriptor_t box_index_desc,
    const void* box_index, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  cnnlCropAndResizeMode_t mode = CNNL_CROP_AND_RESIZE_BILINEAR;
  if (method_name == "nearest") {
    mode = CNNL_CROP_AND_RESIZE_NEAREST;
  }

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCropAndResize(
      handle, image_desc, image, boxes_desc, boxes, box_index_desc, box_index,
      mode, extrapolation_value, output_desc, output));
}

/* static */ void MLUCnnl::CropAndResizeBackwardImage(
    const ExecutionContext& ctx, const std::string method_name,
    const cnnlTensorDescriptor_t grads_desc, const void* grads,
    const cnnlTensorDescriptor_t boxes_desc, const void* boxes,
    const cnnlTensorDescriptor_t box_idx_desc, const void* box_idx,
    const cnnlTensorDescriptor_t grads_image_desc, void* grads_image) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  cnnlCropAndResizeMode_t mode = CNNL_CROP_AND_RESIZE_BILINEAR;
  if (method_name == "nearest") {
    mode = CNNL_CROP_AND_RESIZE_NEAREST;
  }

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCropAndResizeBackwardImage(
      handle, grads_desc, grads, boxes_desc, boxes, box_idx_desc, box_idx, mode,
      grads_image_desc, grads_image));
}

/* static */ void MLUCnnl::CropAndResizeBackwardBoxes(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t image_desc,
    const void* image, const cnnlTensorDescriptor_t boxes_desc,
    const void* boxes, const cnnlTensorDescriptor_t box_idx_desc,
    const void* box_idx, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  cnnlCropAndResizeMode_t mode = CNNL_CROP_AND_RESIZE_BILINEAR;

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCropAndResizeBackwardBoxes(
      handle, input_desc, input, image_desc, image, boxes_desc, boxes,
      box_idx_desc, box_idx, output_desc, output, mode));
}

/* static */ void MLUCnnl::Interp(
    const ExecutionContext& ctx, const cnnlInterpMode_t mode,
    const bool align_corners, const bool half_pixel_centers,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlInterp_v2(handle, align_corners, half_pixel_centers, mode, NULL, true,
                    input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::InterpBackward(
    const ExecutionContext& ctx, const cnnlInterpBackwardMode_t mode,
    const bool align_corners, const bool half_pixel_centers,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlInterpBackward(handle, align_corners, half_pixel_centers, mode,
                         input_desc, input, output_desc, output));
}

/* static */ void MLUCnnl::Cast(const ExecutionContext& ctx,
                                cnnlCastDataType_t cast_type,
                                const cnnlTensorDescriptor_t input_desc,
                                const void* input,
                                const cnnlTensorDescriptor_t output_desc,
                                void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCastDataType(handle, input_desc, input,
                                              cast_type, output_desc, output));
}

/* static */ void MLUCnnl::PoolingBackward(
    const ExecutionContext& ctx, const cnnlPoolingDescriptor_t pooling_desc,
    const void* alpha, const cnnlTensorDescriptor_t y_desc, const void* y,
    const cnnlTensorDescriptor_t diff_y_desc, const void* diff_y,
    const cnnlTensorDescriptor_t x_desc, const void* x, const void* beta,
    const cnnlTensorDescriptor_t diff_x_desc, void* diff_x) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlPoolingBackward(
      handle, const_cast<cnnlPoolingDescriptor_t>(pooling_desc), alpha, y_desc,
      y, diff_y_desc, diff_y, x_desc, x, beta, diff_x_desc, diff_x));
}

/* static */ void MLUCnnl::NonMaxSuppression(
    const ExecutionContext& ctx, const cnnlNmsDescriptor_t nms_desc,
    const cnnlTensorDescriptor_t boxes_desc, const void* boxes,
    const cnnlTensorDescriptor_t confidence_desc, const void* confidence,
    const cnnlTensorDescriptor_t output_desc, void* output, void* output_size) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetNmsWorkspaceSize_v2(handle, confidence_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlNms_v2(
      handle, nms_desc, boxes_desc, boxes, confidence_desc, confidence,
      workspace_ptr, workspace_size, output_desc, output, output_size));
}

/* static */ void MLUCnnl::PoolingIndex(
    const ExecutionContext& ctx, const cnnlPoolingDescriptor_t pooling_desc,
    const cnnlTensorDescriptor_t x_desc, const void* x,
    const cnnlTensorDescriptor_t y_desc, void* y) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlPoolingIndex(
      handle, const_cast<cnnlPoolingDescriptor_t>(pooling_desc), x_desc, x,
      y_desc, y));
}

/* static */ void MLUCnnl::SpaceToBatch(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t output_desc, void* output,
    const int64_t block_shape[]) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetSpace2batchWorkspaceSize(
      handle, input_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  cnnlSpaceBatchParam_t param = {static_cast<uint32_t>(block_shape[0]),
                                 static_cast<uint32_t>(block_shape[1])};
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSpace2batch(handle, input_desc, input,
                                             output_desc, output, param,
                                             workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::SpaceToBatchNd(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t input_desc,
    const void* input, cnnlSpaceBatchNdDescriptor_t param,
    void* extra_device_input, size_t extra_host_input,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSpace2batchNd_v2(handle, input_desc, input, output_desc, output,
                           param, extra_device_input, extra_host_input));
}

/* static */ void MLUCnnl::FusedBatchNorm(
    const ExecutionContext& ctx, const bool is_training,
    const cnnlTensorDescriptor_t x_desc, const void* x,
    const cnnlTensorDescriptor_t scale_desc, const void* scale,
    const void* offset, const void* running_mean_input,
    const void* running_variance_input, float epsilon, float momentum,
    const cnnlTensorDescriptor_t output_desc, void* output,
    void* running_mean_output, void* running_var_output,
    void* saved_batch_mean_output, void* saved_batch_var_output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  if (is_training) {
    /*
     *  In Paddle, running_mean_output = momentum * runnning_mean_input +
     *  (1 - momentum) * batch_mean. However, In CNNL,
     *  running_mean_output = (1 - momentum) * running_mean_input +
     *  momentum * batch_mean. So we pass (1.0 - momentum) to momentum param.
     */
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchNormForwardTraining(
        handle, NULL, NULL, x_desc, x, scale_desc, scale, offset,
        running_mean_output, running_var_output, epsilon, 1.0 - momentum,
        output_desc, output, saved_batch_mean_output, saved_batch_var_output));
  } else {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchNormForwardInference(
        handle, NULL, NULL, x_desc, x, scale_desc, scale, offset,
        running_mean_input, running_variance_input, epsilon, output_desc,
        output));
  }
}

/* static */ void MLUCnnl::FusedBatchNormGrad(
    const ExecutionContext& ctx, const bool is_training,
    const cnnlTensorDescriptor_t y_backprop_desc, const void* y_backprop,
    const cnnlTensorDescriptor_t x_desc, const void* x,
    const cnnlTensorDescriptor_t scale_desc, const void* scale,
    const void* saved_mean, const void* saved_var, float epsilon,
    const cnnlTensorDescriptor_t x_backprop_desc, void* x_backprop,
    void* scale_backprop, void* offset_backprop) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  if (is_training) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchNormBackward(
        handle, NULL, NULL, NULL, NULL, x_desc, x, y_backprop_desc, y_backprop,
        scale_desc, scale, saved_mean, saved_var, epsilon, x_backprop_desc,
        x_backprop, scale_backprop, offset_backprop));
  } else {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlFrozenBatchNormBackward(
        handle, x_desc, x, y_backprop_desc, y_backprop, scale_desc, scale,
        saved_mean, saved_var, epsilon, x_backprop_desc, x_backprop,
        scale_backprop, offset_backprop));
  }
}

/* static */ void MLUCnnl::QuantizeParam(
    const ExecutionContext& ctx, const cnnlQuantizeMode_t mode,
    const int bitwidth, const cnnlTensorDescriptor_t input_desc,
    const void* input, void* position, void* scale, void* offset) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetQuantizeParamWorkspaceSize(handle, input_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeParam(
      handle, mode, input_desc, input, bitwidth, workspace_ptr, workspace_size,
      position, scale, offset));
}

/* static */ void MLUCnnl::Conv2D(
    const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
    const cnnlDataType_t tensor_dtype, const cnnlDataType_t dt_onchip,
    const void* input_position, const void* input_scale,
    const void* input_offset, const void* filter_position,
    const void* filter_scale, const void* filter_offset,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t filter_desc, const void* filter,
    const cnnlTensorDescriptor_t bias_desc, const void* bias,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(input_desc, dt_onchip));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(filter_desc, dt_onchip));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(output_desc, tensor_dtype));

  cnnlConvolutionForwardAlgo_t algo;
  const cnnlConvolutionFwdPreference_t preference =
      CNNL_CONVOLUTION_FWD_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionForwardAlgorithm(
      handle, conv_desc, input_desc, filter_desc, output_desc, preference,
      &algo));

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionForwardWorkspaceSize(
      handle, input_desc, filter_desc, output_desc, bias_desc, conv_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeConvolutionForward(
      handle, conv_desc, algo, nullptr /*alpha*/, input_desc, input,
      input_position, input_scale, input_offset, filter_desc, filter,
      filter_position, filter_scale, filter_offset, bias_desc, bias,
      workspace_ptr, workspace_size, nullptr /*beta*/, output_desc, output));
}

/* static */ void MLUCnnl::FusedConvBNQuantify(
    const ExecutionContext& ctx, cnnlConvolutionDescriptor_t conv_desc,
    const void* epsilon_ptr, const int fused_ops_number,
    const cnnlDataType_t tensor_dtype, const int input_position,
    const float input_scale, const int filter_position,
    const float filter_scale, const cnnlTensorDescriptor_t scale_desc,
    const void* scale_ptr, const cnnlTensorDescriptor_t offset_desc,
    const void* offset_ptr, const cnnlTensorDescriptor_t mean_desc,
    const void* mean_ptr, const cnnlTensorDescriptor_t variance_desc,
    const void* variance_ptr, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t filter_desc,
    const void* filter, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(input_desc, CNNL_DTYPE_INT16));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(filter_desc, CNNL_DTYPE_INT16));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(output_desc, tensor_dtype));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptorPositionAndScale(
      input_desc, input_position, input_scale));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptorPositionAndScale(
      filter_desc, filter_position, filter_scale));

  cnnlFusedOpsPlan_t fusion_plan = nullptr;
  cnnlActivationDescriptor_t active_desc = nullptr;
  cnnlFusedOpsConstParamPack_t cparam_pack = nullptr;
  cnnlFusedOpsVariantParamPack_t vparam_pack = nullptr;
  cnnlConvolutionForwardAlgo_t algo;
  cnnlFusedOps_t fusion_type = CNNL_CONV_SCALE_BN_ACTIVATION;
  cnnlConvolutionCastMode_t cast_mode = CNNL_OFFLINE_SYMMETRIC_QUANTIZE;
  cnnlConvolutionFwdPreference_t preference = CNNL_CONVOLUTION_FWD_FASTEST;

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionForwardAlgorithm(
      handle, conv_desc, input_desc, filter_desc, output_desc, preference,
      &algo));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateFusedOpsPlan(&fusion_plan, fusion_type));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlCreateFusedOpsConstParamPack(&cparam_pack, fusion_type));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlCreateFusedOpsVariantParamPack(&vparam_pack, fusion_type));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_XDESC, input_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_X, input));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_WDESC, filter_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_W, filter));

  if (fused_ops_number > 1) {
    cnnlCreateActivationDescriptor(&active_desc);
    cnnlNanPropagation_t nan_opt = CNNL_NOT_PROPAGATE_NAN;
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetActivationDescriptor(
        active_desc, CNNL_ACTIVATION_RELU, nan_opt, 0.0));
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
        cparam_pack, CNNL_ACTIVATION_DESC, active_desc));
  }
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_BN_WEIGHT_BIAS_MEAN_VAR_DESC, scale_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_BN_WEIGHT, scale_ptr));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_BN_WEIGHT_BIAS_MEAN_VAR_DESC, offset_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_BN_BIAS, offset_ptr));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_BN_WEIGHT_BIAS_MEAN_VAR_DESC, mean_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_BN_MEAN, mean_ptr));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_BN_WEIGHT_BIAS_MEAN_VAR_DESC, variance_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_BN_VAR, variance_ptr));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_CONV_DESC, conv_desc));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_SCALAR_CONV_FWD_ALGO, &algo));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_SCALAR_CONV_FWD_CAST_MODE, &cast_mode));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_SCALAR_BN_EPSILON, epsilon_ptr));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsConstParamPackAttribute(
      cparam_pack, CNNL_YDESC, output_desc));

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
      vparam_pack, CNNL_PTR_Y, output));

  // get workspace size
  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlMakeFusedOpsPlan(handle, fusion_plan, cparam_pack, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  if (workspace_size > 0) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
        vparam_pack, CNNL_PTR_WORKSPACE, workspace_ptr));
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetFusedOpsVariantParamPackAttribute(
        vparam_pack, CNNL_SCALAR_WORKSPACE_SIZE, &workspace_size));
  }
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlFusedOpsExecute(handle, fusion_plan, vparam_pack));

  if (active_desc) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyActivationDescriptor(active_desc));
  }

  if (cparam_pack) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyFusedOpsConstParamPack(cparam_pack));
  }

  if (vparam_pack) {
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlDestroyFusedOpsVariantParamPack(vparam_pack));
  }

  if (fusion_plan) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyFusedOpsPlan(fusion_plan));
  }
}

/* static */ void MLUCnnl::ConvBackpropInput(
    const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
    const cnnlTensorDescriptor_t filter_desc, const void* filter,
    const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
    const cnnlTensorDescriptor_t in_backprop_desc, void* in_backprop) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  cnnlConvolutionBwdDataAlgo_t algo;
  const cnnlConvolutionBwdDataPreference_t preference =
      CNNL_CONVOLUTION_BWD_DATA_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardDataAlgorithm(
      handle, filter_desc, out_backprop_desc, conv_desc, in_backprop_desc,
      preference, &algo));

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle, filter_desc, out_backprop_desc, conv_desc, in_backprop_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlConvolutionBackwardData(
      handle, nullptr /*alpha*/, filter_desc, filter, out_backprop_desc,
      out_backprop, conv_desc, algo, workspace_ptr, workspace_size,
      nullptr /*beta*/, in_backprop_desc, in_backprop));
}

/* static */ void MLUCnnl::QuantizeConvBackpropInput(
    const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
    const cnnlDataType_t tensor_dtype, const cnnlDataType_t dt_onchip,
    const void* filter_position, const void* filter_scale,
    const void* filter_offset, const void* out_backprop_position,
    const void* out_backprop_scale, const void* out_backprop_offset,
    const cnnlTensorDescriptor_t filter_desc, const void* filter,
    const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
    const cnnlTensorDescriptor_t in_backprop_desc, void* in_backprop) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(filter_desc, dt_onchip));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(out_backprop_desc, dt_onchip));

  cnnlConvolutionBwdDataAlgo_t algo;
  const cnnlConvolutionBwdDataPreference_t preference =
      CNNL_CONVOLUTION_BWD_DATA_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardDataAlgorithm(
      handle, filter_desc, out_backprop_desc, conv_desc, in_backprop_desc,
      preference, &algo));

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle, filter_desc, out_backprop_desc, conv_desc, in_backprop_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeConvolutionBackwardData(
      handle, nullptr /*alpha*/, filter_desc, filter, filter_position,
      filter_scale, filter_offset, out_backprop_desc, out_backprop,
      out_backprop_position, out_backprop_scale, out_backprop_offset, conv_desc,
      algo, workspace_ptr, workspace_size, nullptr /*beta*/, in_backprop_desc,
      in_backprop));
}

/* static */ void MLUCnnl::ConvBackpropFilter(
    const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
    const cnnlTensorDescriptor_t filter_backprop_desc, void* filter_backprop) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  cnnlConvolutionBwdFilterAlgo_t algo;
  const cnnlConvolutionBwdFilterPreference_t preference =
      CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle, conv_desc, input_desc, out_backprop_desc, filter_backprop_desc,
      preference, &algo));

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc, out_backprop_desc, filter_backprop_desc, conv_desc,
      algo, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlConvolutionBackwardFilter(
      handle, nullptr /*alpha*/, input_desc, input, out_backprop_desc,
      out_backprop, conv_desc, algo, workspace_ptr, workspace_size,
      nullptr /*beta*/, filter_backprop_desc, filter_backprop));
}

/* static */ void MLUCnnl::QuantizeConvBackpropFilter(
    const ExecutionContext& ctx, const cnnlConvolutionDescriptor_t conv_desc,
    const cnnlDataType_t tensor_dtype, const cnnlDataType_t dt_onchip,
    const void* input_position, const void* input_scale,
    const void* input_offset, const void* out_backprop_position,
    const void* out_backprop_scale, const void* out_backprop_offset,
    const cnnlTensorDescriptor_t input_desc, const void* input,
    const cnnlTensorDescriptor_t out_backprop_desc, const void* out_backprop,
    const cnnlTensorDescriptor_t filter_backprop_desc, void* filter_backprop) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(input_desc, dt_onchip));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorOnchipDataType(out_backprop_desc, dt_onchip));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptorOnchipDataType(
      filter_backprop_desc, tensor_dtype));

  cnnlConvolutionBwdFilterAlgo_t algo;
  const cnnlConvolutionBwdFilterPreference_t preference =
      CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle, conv_desc, input_desc, out_backprop_desc, filter_backprop_desc,
      preference, &algo));

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc, out_backprop_desc, filter_backprop_desc, conv_desc,
      algo, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeConvolutionBackwardFilter(
      handle, nullptr /*alpha*/, input_desc, input, input_position, input_scale,
      input_offset, out_backprop_desc, out_backprop, out_backprop_position,
      out_backprop_scale, out_backprop_offset, conv_desc, algo, workspace_ptr,
      workspace_size, nullptr /*beta*/, filter_backprop_desc, filter_backprop));
}

/* static */ void MLUCnnl::QuantizeMatMul(
    const ExecutionContext& ctx, const bool transpose_a, const bool transpose_b,
    const cnnlTensorDescriptor_t a_desc, const void* a, const void* a_position,
    const void* a_scale, const void* a_offset,
    const cnnlTensorDescriptor_t b_desc, const void* b, const void* b_position,
    const void* b_scale, const void* b_offset, const cnnlDataType_t quant_type,
    const cnnlDataType_t data_type, const cnnlTensorDescriptor_t output_desc,
    void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // Set onchip data type
  cnnlSetTensorDescriptorOnchipDataType(a_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(b_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(output_desc, data_type);

  // Create and set matmul descriptor
  cnnlMatMulDescriptor_t matmul_desc;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMatMulDescCreate(&matmul_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetMatMulDescAttr(
      matmul_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE, &data_type, sizeof(int)));
  int transpose_a_int = static_cast<int>(transpose_a);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetMatMulDescAttr(
      matmul_desc, CNNL_MATMUL_DESC_TRANSA, &(transpose_a_int), sizeof(int)));
  int transpose_b_int = static_cast<int>(transpose_b);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetMatMulDescAttr(
      matmul_desc, CNNL_MATMUL_DESC_TRANSB, &(transpose_b_int), sizeof(int)));

  // Create and get matmul algorithim
  cnnlMatMulAlgo_t algo;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMatMulAlgoCreate(&algo));
  const cnnlMatMulPreference_t preference = CNNL_MATMUL_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeMatMulAlgorithm(
      handle, matmul_desc, a_desc, b_desc, output_desc, preference, &algo));

  // Get workspace
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeMatMulWorkspaceSize(
      handle, matmul_desc, a_desc, b_desc, output_desc, algo, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  // Compute
  float alpha = 1.0;
  float beta = 0.0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeMatMul(
      handle, matmul_desc, reinterpret_cast<void*>(&alpha), a_desc, a,
      a_position, a_scale, a_offset, b_desc, b, b_position, b_scale, b_offset,
      reinterpret_cast<void*>(&beta), output_desc, output, algo, workspace_ptr,
      workspace_size));

  // Destroy matmul descriptor and algorithim
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMatMulDescDestroy(matmul_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMatMulAlgoDestroy(algo));
}

/* static */ void MLUCnnl::QuantizeBatchMatMul(
    const ExecutionContext& ctx, const bool adj_x, const bool adj_y,
    const cnnlTensorDescriptor_t in0_desc, const void* in0,
    const void* in0_position, const void* in0_scale, const void* in0_offset,
    const cnnlTensorDescriptor_t in1_desc, const void* in1,
    const void* in1_position, const void* in1_scale, const void* in1_offset,
    const cnnlDataType_t quant_type, const cnnlDataType_t data_type,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // Set onchip data type
  cnnlSetTensorDescriptorOnchipDataType(in0_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(in1_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(output_desc, data_type);

  // Create and set batch matmul descriptor
  cnnlBatchMatMulDescriptor_t bmm_desc;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulDescCreate(&bmm_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulDescAttr(
      bmm_desc, CNNL_BMM_DESC_COMPUTE_TYPE, &data_type, sizeof(int)));
  int transpose_a_int = static_cast<int>(adj_x);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulDescAttr(
      bmm_desc, CNNL_BMM_DESC_TRANSA, &(transpose_a_int), sizeof(int)));
  int transpose_b_int = static_cast<int>(adj_y);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulDescAttr(
      bmm_desc, CNNL_BMM_DESC_TRANSB, &(transpose_b_int), sizeof(int)));

  // Create and get batch matmul algorithim
  cnnlBatchMatMulAlgo_t algo;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulAlgoCreate(&algo));
  const cnnlBatchMatMulPreference_t preference = CNNL_BMM_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeBatchMatMulAlgorithm(
      handle, bmm_desc, in0_desc, in1_desc, output_desc, preference, &algo));

  // Get workspace
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeBatchMatMulWorkspaceSize(
      handle, bmm_desc, in0_desc, in1_desc, output_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  // Compute
  float alpha = 1.0;
  float beta = 0.0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeBatchMatMul(
      handle, bmm_desc, reinterpret_cast<void*>(&alpha), in0_desc, in0,
      in0_position, in0_scale, in0_offset, in1_desc, in1, in1_position,
      in1_scale, in1_offset, reinterpret_cast<void*>(&beta), output_desc,
      output, algo, workspace_ptr, workspace_size));

  // Destroy matmul descriptor and algorithim
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulDescDestroy(bmm_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulAlgoDestroy(algo));
}

/* static */ void MLUCnnl::QuantizeBatchMatMulBCast(
    const ExecutionContext& ctx, const bool adj_x, const bool adj_y,
    const cnnlTensorDescriptor_t in0_desc, const void* in0,
    const void* in0_position, const void* in0_scale, const void* in0_offset,
    const cnnlTensorDescriptor_t in1_desc, const void* in1,
    const void* in1_position, const void* in1_scale, const void* in1_offset,
    const cnnlDataType_t quant_type, const cnnlDataType_t data_type,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  // Set onchip data type
  cnnlSetTensorDescriptorOnchipDataType(in0_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(in1_desc, quant_type);
  cnnlSetTensorDescriptorOnchipDataType(output_desc, data_type);

  // Create and set batch matmul descriptor
  cnnlBatchMatMulBCastDescriptor_t bmm_bcast_desc;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulBCastDescCreate(&bmm_bcast_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulBCastDescAttr(
      bmm_bcast_desc, CNNL_BMM_BCAST_DESC_COMPUTE_TYPE, &data_type,
      sizeof(int)));
  int transpose_a_int = static_cast<int>(adj_x);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulBCastDescAttr(
      bmm_bcast_desc, CNNL_BMM_BCAST_DESC_TRANSA, &(transpose_a_int),
      sizeof(int)));
  int transpose_b_int = static_cast<int>(adj_y);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetBatchMatMulBCastDescAttr(
      bmm_bcast_desc, CNNL_BMM_BCAST_DESC_TRANSB, &(transpose_b_int),
      sizeof(int)));

  // Create and get batch matmul algorithim
  cnnlBatchMatMulBCastAlgo_t algo;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulBCastAlgoCreate(&algo));
  const cnnlBatchMatMulBCastPreference_t preference = CNNL_BMM_BCAST_FASTEST;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeBatchMatMulBCastAlgorithm(
      handle, bmm_bcast_desc, in0_desc, in1_desc, output_desc, preference,
      &algo));

  // Get workspace
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetQuantizeBatchMatMulBCastWorkspaceSize(
      handle, bmm_bcast_desc, in0_desc, in1_desc, output_desc, algo,
      &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  // Compute
  float alpha = 1.0;
  float beta = 0.0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQuantizeBatchMatMulBCast(
      handle, bmm_bcast_desc, reinterpret_cast<void*>(&alpha), in0_desc, in0,
      in0_position, in0_scale, in0_offset, in1_desc, in1, in1_position,
      in1_scale, in1_offset, reinterpret_cast<void*>(&beta), output_desc,
      output, algo, workspace_ptr, workspace_size));

  // Destroy matmul descriptor and algorithim
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulBCastDescDestroy(bmm_bcast_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBatchMatMulBCastAlgoDestroy(algo));
}

/* static */ void MLUCnnl::Transpose(
    const ExecutionContext& ctx, const std::vector<int> perm,
    const int input_dim, const cnnlTensorDescriptor_t input_desc,
    const void* input, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  cnnlTransposeDescriptor_t perm_desc;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateTransposeDescriptor(&perm_desc));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTransposeDescriptor(perm_desc, input_dim, perm.data()));

  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetTransposeWorkspaceSize(
      handle, input_desc, perm_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlTranspose_v2(handle, perm_desc, input_desc,
                                              input, output_desc, output,
                                              workspace_ptr, workspace_size));
  if (perm_desc) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyTransposeDescriptor(perm_desc));
  }
}

/* static */ void MLUCnnl::MatrixBandPart(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t data_desc,
    const void* input, const int num_lower, const int num_upper, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlMatrixBandPart(handle, data_desc, input,
                                                num_lower, num_upper, output));
}

/* static */ void MLUCnnl::NumTrue(const ExecutionContext& ctx,
                                   const cnnlTensorDescriptor_t x_desc,
                                   const void* x, Tensor index,
                                   uint32_t* num_true) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  size_t workspace_size = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetNumTrueWorkspaceSize(handle, x_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  index = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* index_ptr = index.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlNumTrue(
      handle, x_desc, x, static_cast<uint32_t*>(index_ptr), num_true));
}

/* static */ void MLUCnnl::Where(const ExecutionContext& ctx,
                                 const cnnlTensorDescriptor_t x_desc,
                                 const void* x, const uint32_t* strides,
                                 const uint32_t* index,
                                 const cnnlTensorDescriptor_t y_desc, int* y,
                                 const bool as_tuple) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlWhere(handle, x_desc, x, strides, index, y_desc, y, as_tuple));
}

/* static */ void MLUCnnl::InTopK(
    const ExecutionContext& ctx, const cnnlTensorDescriptor_t predictions_desc,
    const void* predictions, const cnnlTensorDescriptor_t targets_desc,
    const void* targets, const cnnlTensorDescriptor_t k_desc, const void* k,
    const int k_int, const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlInTopK(handle, predictions_desc, predictions,
                                        targets_desc, targets, k_desc, k, k_int,
                                        output_desc, output));
}

/* static */ void MLUCnnl::ScatterNd(const ExecutionContext& ctx,
                                     const cnnlTensorDescriptor_t indices_desc,
                                     const void* indices,
                                     const cnnlTensorDescriptor_t updates_desc,
                                     const void* updates,
                                     const cnnlTensorDescriptor_t output_desc,
                                     void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlScatterNd(handle, indices_desc, indices,
                                           updates_desc, updates, output_desc,
                                           output));
}

/* static */ void MLUCnnl::BitWise(
    const ExecutionContext& ctx, const cnnlBitComputeOp_t optype,
    const cnnlTensorDescriptor_t input1_desc, const void* input1,
    const cnnlTensorDescriptor_t input2_desc, const void* input2,
    const cnnlTensorDescriptor_t output_desc, void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetBitComputeWorkspaceSize(
      handle, input1_desc, input2_desc, output_desc, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlBitCompute_v2(
      handle, optype, input1_desc, input1, input2_desc, input2, output_desc,
      output, workspace_ptr, workspace_size));
}

/* static */ void MLUCnnl::QR(const ExecutionContext& ctx,
                              const cnnlTensorDescriptor_t a_desc,
                              const void* a,
                              const cnnlTensorDescriptor_t q_desc, void* q,
                              const cnnlTensorDescriptor_t r_desc, void* r,
                              const bool some) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);
  size_t workspace_size;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlGetQRWorkspaceSize(handle, a_desc, some, &workspace_size));

  auto& dev_ctx = GetDevCtxFromCTX(ctx);
  Tensor workspace = ctx.AllocateTmpTensor<int8_t, MLUDeviceContext>(
      {static_cast<int64_t>(workspace_size)}, dev_ctx);
  void* workspace_ptr = workspace.mutable_data(ctx.GetPlace());

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlQR(handle, a_desc, a, q_desc, q, r_desc, r,
                                    workspace_ptr, workspace_size, some));
}

/* static */ void MLUCnnl::Reciprocal(const ExecutionContext& ctx,
                                      const cnnlTensorDescriptor_t input_desc,
                                      const void* input,
                                      const cnnlTensorDescriptor_t output_desc,
                                      void* output) {
  cnnlHandle_t handle = GetHandleFromCTX(ctx);

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlReciprocal(handle, input_desc, input, output_desc, output));
}

}  // namespace operators
}  // namespace paddle
