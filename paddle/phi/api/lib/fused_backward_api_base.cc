
#include "paddle/phi/api/backward/fused_backward_api_base.h"
#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/api/include/fused_api.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/fusion.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

PD_DECLARE_bool(conv2d_disable_cudnn);
COMMON_DECLARE_int32(low_precision_op_list);
COMMON_DECLARE_bool(benchmark);

namespace paddle {
namespace experimental {


PADDLE_API void fused_bias_dropout_residual_layer_norm_grad(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, const Tensor& ln_mean, const Tensor& ln_variance, const Tensor& bias_dropout_residual_out, const Tensor& dropout_mask_out, const Tensor& y_grad, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon, Tensor* x_grad, Tensor* residual_grad, Tensor* bias_grad, Tensor* ln_scale_grad, Tensor* ln_bias_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(y_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, residual, bias, ln_scale, ln_bias, ln_mean, ln_variance, bias_dropout_residual_out, dropout_mask_out, y_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fused_bias_dropout_residual_layer_norm_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_bias_dropout_residual_layer_norm_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_bias_dropout_residual_layer_norm_grad", kernel_data_type);
  }
  VLOG(6) << "fused_bias_dropout_residual_layer_norm_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_residual = PrepareData(residual, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_scale = PrepareData(ln_scale, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_bias = PrepareData(ln_bias, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_mean = PrepareData(ln_mean, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_variance = PrepareData(ln_variance, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias_dropout_residual_out = PrepareData(bias_dropout_residual_out, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dropout_mask_out = PrepareData(dropout_mask_out, GetKernelInputArgDef(kernel.InputAt(8), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_y_grad = PrepareData(y_grad, GetKernelInputArgDef(kernel.InputAt(9), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<phi::DDim> ln_scale_record_shapes;
     if(input_ln_scale){
       ln_scale_record_shapes.push_back((*input_ln_scale).dims());
     }
     std::vector<phi::DDim> ln_bias_record_shapes;
     if(input_ln_bias){
       ln_bias_record_shapes.push_back((*input_ln_bias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"residual", {
     (*input_residual).dims()}},
     {"bias", bias_record_shapes},
     {"ln_scale", ln_scale_record_shapes},
     {"ln_bias", ln_bias_record_shapes},
     {"ln_mean", {
     (*input_ln_mean).dims()}},
     {"ln_variance", {
     (*input_ln_variance).dims()}},
     {"bias_dropout_residual_out", {
     (*input_bias_dropout_residual_out).dims()}},
     {"dropout_mask_out", {
     (*input_dropout_mask_out).dims()}},
     {"y_grad", {
     (*input_y_grad).dims()}}};
     phi::AttributeMap attrs;
     attrs["dropout_rate"] = dropout_rate;
     attrs["is_test"] = is_test;
     attrs["dropout_fix_seed"] = dropout_fix_seed;
     attrs["dropout_seed"] = dropout_seed;
     attrs["dropout_implementation"] = dropout_implementation;
     attrs["ln_epsilon"] = ln_epsilon;
     phi::RecordOpInfoSupplement("fused_bias_dropout_residual_layer_norm_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(x_grad);
  auto kernel_out_1 = SetKernelOutput(residual_grad);
  auto kernel_out_2 = SetKernelOutput(bias_grad);
  auto kernel_out_3 = SetKernelOutput(ln_scale_grad);
  auto kernel_out_4 = SetKernelOutput(ln_bias_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);

  phi::FusedBiasDropoutResidualLnGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), MakeMetaTensor(*input_ln_mean), MakeMetaTensor(*input_ln_variance), MakeMetaTensor(*input_bias_dropout_residual_out), MakeMetaTensor(*input_dropout_mask_out), MakeMetaTensor(*input_y_grad), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, *input_ln_mean, *input_ln_variance, *input_bias_dropout_residual_out, *input_dropout_mask_out, *input_y_grad, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "fused_bias_dropout_residual_layer_norm_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}

PADDLE_API void fused_dot_product_attention_grad(const Tensor& q, const Tensor& k, const Tensor& v, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlen_q, const paddle::optional<Tensor>& cu_seqlen_kv, const Tensor& out, const Tensor& softmax_out, const Tensor& rng_state, const Tensor& out_grad, float scaling_factor, float dropout_probability, const std::string& mask_type_str, const std::string& bias_type_str, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad, Tensor* bias_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(q);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(q, k, v, bias, cu_seqlen_q, cu_seqlen_kv, out, softmax_out, rng_state, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fused_dot_product_attention_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_dot_product_attention_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_dot_product_attention_grad", kernel_data_type);
  }
  VLOG(6) << "fused_dot_product_attention_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_q = PrepareData(q, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_k = PrepareData(k, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_v = PrepareData(v, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlen_q = PrepareData(cu_seqlen_q, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlen_kv = PrepareData(cu_seqlen_kv, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out = PrepareData(out, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_softmax_out = PrepareData(softmax_out, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_rng_state = PrepareData(rng_state, GetKernelInputArgDef(kernel.InputAt(8), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_grad = PrepareData(out_grad, GetKernelInputArgDef(kernel.InputAt(9), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<phi::DDim> cu_seqlen_q_record_shapes;
     if(input_cu_seqlen_q){
       cu_seqlen_q_record_shapes.push_back((*input_cu_seqlen_q).dims());
     }
     std::vector<phi::DDim> cu_seqlen_kv_record_shapes;
     if(input_cu_seqlen_kv){
       cu_seqlen_kv_record_shapes.push_back((*input_cu_seqlen_kv).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"q", {
     (*input_q).dims()}},
     {"k", {
     (*input_k).dims()}},
     {"v", {
     (*input_v).dims()}},
     {"bias", bias_record_shapes},
     {"cu_seqlen_q", cu_seqlen_q_record_shapes},
     {"cu_seqlen_kv", cu_seqlen_kv_record_shapes},
     {"out", {
     (*input_out).dims()}},
     {"softmax_out", {
     (*input_softmax_out).dims()}},
     {"rng_state", {
     (*input_rng_state).dims()}},
     {"out_grad", {
     (*input_out_grad).dims()}}};
     phi::AttributeMap attrs;
     attrs["scaling_factor"] = scaling_factor;
     attrs["dropout_probability"] = dropout_probability;
     attrs["mask_type_str"] = mask_type_str;
     attrs["bias_type_str"] = bias_type_str;
     phi::RecordOpInfoSupplement("fused_dot_product_attention_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(q_grad);
  auto kernel_out_1 = SetKernelOutput(k_grad);
  auto kernel_out_2 = SetKernelOutput(v_grad);
  auto kernel_out_3 = SetKernelOutput(bias_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_dot_product_attention_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);

  phi::FusedDotProductAttentionGradInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), MakeMetaTensor(input_bias), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, const std::string&, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_dot_product_attention_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, input_bias, input_cu_seqlen_q, input_cu_seqlen_kv, *input_out, *input_softmax_out, *input_rng_state, *input_out_grad, scaling_factor, dropout_probability, mask_type_str, bias_type_str, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "fused_dot_product_attention_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}

PADDLE_API void fused_dropout_add_grad(const Tensor& seed_offset, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, bool fix_seed, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(seed_offset, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fused_dropout_add_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_dropout_add_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_dropout_add_grad", kernel_data_type);
  }
  VLOG(6) << "fused_dropout_add_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_seed_offset = PrepareData(seed_offset, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_grad = PrepareData(out_grad, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"seed_offset", {
     (*input_seed_offset).dims()}},
     {"out_grad", {
     (*input_out_grad).dims()}}};
     phi::AttributeMap attrs;
    switch (p.dtype()) {
      case DataType::FLOAT32:
          attrs["p"] = static_cast<float>(p.to<float>());
          break;
      case DataType::FLOAT64:
          attrs["p"] = static_cast<double>(p.to<double>());
          break;
      case DataType::FLOAT16:
          attrs["p"] = static_cast<float>(p.to<float16>());
          break;
      case DataType::BFLOAT16:
          attrs["p"] = static_cast<float>(p.to<bfloat16>());
          break;
      case DataType::INT32:
          attrs["p"] = static_cast<int32_t>(p.to<int32_t>());
          break;
      case DataType::INT64:
          attrs["p"] = static_cast<int64_t>(p.to<int64_t>());
          break;
      case DataType::INT16:
          attrs["p"] = static_cast<int16_t>(p.to<int16_t>());
          break;
      case DataType::INT8:
          attrs["p"] = static_cast<int8_t>(p.to<int8_t>());
          break;
      case DataType::UINT16:
          attrs["p"] = static_cast<uint16_t>(p.to<uint16_t>());
          break;
      case DataType::UINT8:
          attrs["p"] = static_cast<uint8_t>(p.to<uint8_t>());
          break;
      case DataType::BOOL:
          attrs["p"] = static_cast<bool>(p.to<bool>());
          break;
      case DataType::COMPLEX64:
          attrs["p"] = static_cast<float>(p.to<complex64>());
          break;
      case DataType::COMPLEX128:
          attrs["p"] = static_cast<double>(p.to<complex128>());
          break;
      default:
          attrs["p"] = "";
          break;
    }
     attrs["is_test"] = is_test;
     attrs["mode"] = mode;
     attrs["fix_seed"] = fix_seed;
     phi::RecordOpInfoSupplement("fused_dropout_add_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(x_grad);
  auto kernel_out_1 = SetKernelOutput(y_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_dropout_add_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FusedDropoutAddGradInferMeta(MakeMetaTensor(*input_seed_offset), MakeMetaTensor(*input_out_grad), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, bool, const std::string&, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_dropout_add_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_seed_offset, *input_out_grad, phi::Scalar(p), is_test, mode, fix_seed, kernel_out_0, kernel_out_1);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "fused_dropout_add_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}

PADDLE_API void fused_rotary_position_embedding_grad(const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, const Tensor& out_q_grad, const paddle::optional<Tensor>& out_k_grad, const paddle::optional<Tensor>& out_v_grad, bool use_neox_rotary_style, bool time_major, float rotary_emb_base, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_q_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(sin, cos, position_ids, out_q_grad, out_k_grad, out_v_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fused_rotary_position_embedding_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_rotary_position_embedding_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_rotary_position_embedding_grad", kernel_data_type);
  }
  VLOG(6) << "fused_rotary_position_embedding_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_sin = PrepareData(sin, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cos = PrepareData(cos, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_position_ids = PrepareData(position_ids, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_q_grad = PrepareData(out_q_grad, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_k_grad = PrepareData(out_k_grad, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_v_grad = PrepareData(out_v_grad, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> sin_record_shapes;
     if(input_sin){
       sin_record_shapes.push_back((*input_sin).dims());
     }
     std::vector<phi::DDim> cos_record_shapes;
     if(input_cos){
       cos_record_shapes.push_back((*input_cos).dims());
     }
     std::vector<phi::DDim> position_ids_record_shapes;
     if(input_position_ids){
       position_ids_record_shapes.push_back((*input_position_ids).dims());
     }
     std::vector<phi::DDim> out_k_grad_record_shapes;
     if(input_out_k_grad){
       out_k_grad_record_shapes.push_back((*input_out_k_grad).dims());
     }
     std::vector<phi::DDim> out_v_grad_record_shapes;
     if(input_out_v_grad){
       out_v_grad_record_shapes.push_back((*input_out_v_grad).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"sin", sin_record_shapes},
     {"cos", cos_record_shapes},
     {"position_ids", position_ids_record_shapes},
     {"out_q_grad", {
     (*input_out_q_grad).dims()}},
     {"out_k_grad", out_k_grad_record_shapes},
     {"out_v_grad",
     out_v_grad_record_shapes}};
     phi::AttributeMap attrs;
     attrs["use_neox_rotary_style"] = use_neox_rotary_style;
     attrs["time_major"] = time_major;
     attrs["rotary_emb_base"] = rotary_emb_base;
     phi::RecordOpInfoSupplement("fused_rotary_position_embedding_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(q_grad);
  auto kernel_out_1 = SetKernelOutput(k_grad);
  auto kernel_out_2 = SetKernelOutput(v_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_rotary_position_embedding_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::FusedRopeGradInferMeta(MakeMetaTensor(input_sin), MakeMetaTensor(input_cos), MakeMetaTensor(input_position_ids), MakeMetaTensor(*input_out_q_grad), MakeMetaTensor(input_out_k_grad), MakeMetaTensor(input_out_v_grad), use_neox_rotary_style, time_major, rotary_emb_base, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_rotary_position_embedding_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, input_sin, input_cos, input_position_ids, *input_out_q_grad, input_out_k_grad, input_out_v_grad, use_neox_rotary_style, time_major, rotary_emb_base, kernel_out_0, kernel_out_1, kernel_out_2);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "fused_rotary_position_embedding_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}

PADDLE_API void resnet_basic_block_grad(const Tensor& x, const Tensor& filter1, const Tensor& conv1, const Tensor& scale1, const Tensor& bias1, const Tensor& saved_mean1, const Tensor& saved_invstd1, const Tensor& filter2, const Tensor& conv2, const Tensor& conv2_input, const Tensor& scale2, const Tensor& bias2, const Tensor& saved_mean2, const Tensor& saved_invstd2, const paddle::optional<Tensor>& filter3, const paddle::optional<Tensor>& conv3, const paddle::optional<Tensor>& scale3, const paddle::optional<Tensor>& bias3, const paddle::optional<Tensor>& saved_mean3, const paddle::optional<Tensor>& saved_invstd3, const Tensor& max_input1, const Tensor& max_filter1, const Tensor& max_input2, const Tensor& max_filter2, const Tensor& max_input3, const Tensor& max_filter3, const Tensor& out, const Tensor& out_grad, int stride1, int stride2, int stride3, int padding1, int padding2, int padding3, int dilation1, int dilation2, int dilation3, int group, float momentum, float epsilon, const std::string& data_format, bool has_shortcut, bool use_global_stats, bool is_test, bool trainable_statistics, const std::string& act_type, bool find_conv_input_max, Tensor* x_grad, Tensor* filter1_grad, Tensor* scale1_grad, Tensor* bias1_grad, Tensor* filter2_grad, Tensor* scale2_grad, Tensor* bias2_grad, Tensor* filter3_grad, Tensor* scale3_grad, Tensor* bias3_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter1, conv1, scale1, bias1, saved_mean1, saved_invstd1, filter2, conv2, conv2_input, scale2, bias2, saved_mean2, saved_invstd2, filter3, conv3, scale3, bias3, saved_mean3, saved_invstd3, max_input1, max_filter1, max_input2, max_filter2, max_input3, max_filter3, out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "resnet_basic_block_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "resnet_basic_block_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("resnet_basic_block_grad", kernel_data_type);
  }
  VLOG(6) << "resnet_basic_block_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_filter1 = PrepareData(filter1, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv1 = PrepareData(conv1, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale1 = PrepareData(scale1, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias1 = PrepareData(bias1, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_mean1 = PrepareData(saved_mean1, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_invstd1 = PrepareData(saved_invstd1, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_filter2 = PrepareData(filter2, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv2 = PrepareData(conv2, GetKernelInputArgDef(kernel.InputAt(8), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv2_input = PrepareData(conv2_input, GetKernelInputArgDef(kernel.InputAt(9), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale2 = PrepareData(scale2, GetKernelInputArgDef(kernel.InputAt(10), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias2 = PrepareData(bias2, GetKernelInputArgDef(kernel.InputAt(11), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_mean2 = PrepareData(saved_mean2, GetKernelInputArgDef(kernel.InputAt(12), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_invstd2 = PrepareData(saved_invstd2, GetKernelInputArgDef(kernel.InputAt(13), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_filter3 = PrepareData(filter3, GetKernelInputArgDef(kernel.InputAt(14), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv3 = PrepareData(conv3, GetKernelInputArgDef(kernel.InputAt(15), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale3 = PrepareData(scale3, GetKernelInputArgDef(kernel.InputAt(16), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias3 = PrepareData(bias3, GetKernelInputArgDef(kernel.InputAt(17), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_mean3 = PrepareData(saved_mean3, GetKernelInputArgDef(kernel.InputAt(18), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_invstd3 = PrepareData(saved_invstd3, GetKernelInputArgDef(kernel.InputAt(19), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_input1 = PrepareData(max_input1, GetKernelInputArgDef(kernel.InputAt(20), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_filter1 = PrepareData(max_filter1, GetKernelInputArgDef(kernel.InputAt(21), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_input2 = PrepareData(max_input2, GetKernelInputArgDef(kernel.InputAt(22), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_filter2 = PrepareData(max_filter2, GetKernelInputArgDef(kernel.InputAt(23), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_input3 = PrepareData(max_input3, GetKernelInputArgDef(kernel.InputAt(24), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_max_filter3 = PrepareData(max_filter3, GetKernelInputArgDef(kernel.InputAt(25), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out = PrepareData(out, GetKernelInputArgDef(kernel.InputAt(26), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_grad = PrepareData(out_grad, GetKernelInputArgDef(kernel.InputAt(27), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> filter3_record_shapes;
     if(input_filter3){
       filter3_record_shapes.push_back((*input_filter3).dims());
     }
     std::vector<phi::DDim> conv3_record_shapes;
     if(input_conv3){
       conv3_record_shapes.push_back((*input_conv3).dims());
     }
     std::vector<phi::DDim> scale3_record_shapes;
     if(input_scale3){
       scale3_record_shapes.push_back((*input_scale3).dims());
     }
     std::vector<phi::DDim> bias3_record_shapes;
     if(input_bias3){
       bias3_record_shapes.push_back((*input_bias3).dims());
     }
     std::vector<phi::DDim> saved_mean3_record_shapes;
     if(input_saved_mean3){
       saved_mean3_record_shapes.push_back((*input_saved_mean3).dims());
     }
     std::vector<phi::DDim> saved_invstd3_record_shapes;
     if(input_saved_invstd3){
       saved_invstd3_record_shapes.push_back((*input_saved_invstd3).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"filter1", {
     (*input_filter1).dims()}},
     {"conv1", {
     (*input_conv1).dims()}},
     {"scale1", {
     (*input_scale1).dims()}},
     {"bias1", {
     (*input_bias1).dims()}},
     {"saved_mean1", {
     (*input_saved_mean1).dims()}},
     {"saved_invstd1", {
     (*input_saved_invstd1).dims()}},
     {"filter2", {
     (*input_filter2).dims()}},
     {"conv2", {
     (*input_conv2).dims()}},
     {"conv2_input", {
     (*input_conv2_input).dims()}},
     {"scale2", {
     (*input_scale2).dims()}},
     {"bias2", {
     (*input_bias2).dims()}},
     {"saved_mean2", {
     (*input_saved_mean2).dims()}},
     {"saved_invstd2", {
     (*input_saved_invstd2).dims()}},
     {"filter3", filter3_record_shapes},
     {"conv3", conv3_record_shapes},
     {"scale3", scale3_record_shapes},
     {"bias3", bias3_record_shapes},
     {"saved_mean3", saved_mean3_record_shapes},
     {"saved_invstd3", saved_invstd3_record_shapes},
     {"max_input1", {
     (*input_max_input1).dims()}},
     {"max_filter1", {
     (*input_max_filter1).dims()}},
     {"max_input2", {
     (*input_max_input2).dims()}},
     {"max_filter2", {
     (*input_max_filter2).dims()}},
     {"max_input3", {
     (*input_max_input3).dims()}},
     {"max_filter3", {
     (*input_max_filter3).dims()}},
     {"out", {
     (*input_out).dims()}},
     {"out_grad", {
     (*input_out_grad).dims()}}};
     phi::AttributeMap attrs;
     attrs["stride1"] = stride1;
     attrs["stride2"] = stride2;
     attrs["stride3"] = stride3;
     attrs["padding1"] = padding1;
     attrs["padding2"] = padding2;
     attrs["padding3"] = padding3;
     attrs["dilation1"] = dilation1;
     attrs["dilation2"] = dilation2;
     attrs["dilation3"] = dilation3;
     attrs["group"] = group;
     attrs["momentum"] = momentum;
     attrs["epsilon"] = epsilon;
     attrs["data_format"] = data_format;
     attrs["has_shortcut"] = has_shortcut;
     attrs["use_global_stats"] = use_global_stats;
     attrs["is_test"] = is_test;
     attrs["trainable_statistics"] = trainable_statistics;
     attrs["act_type"] = act_type;
     attrs["find_conv_input_max"] = find_conv_input_max;
     phi::RecordOpInfoSupplement("resnet_basic_block_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(x_grad);
  auto kernel_out_1 = SetKernelOutput(filter1_grad);
  auto kernel_out_2 = SetKernelOutput(scale1_grad);
  auto kernel_out_3 = SetKernelOutput(bias1_grad);
  auto kernel_out_4 = SetKernelOutput(filter2_grad);
  auto kernel_out_5 = SetKernelOutput(scale2_grad);
  auto kernel_out_6 = SetKernelOutput(bias2_grad);
  auto kernel_out_7 = SetKernelOutput(filter3_grad);
  auto kernel_out_8 = SetKernelOutput(scale3_grad);
  auto kernel_out_9 = SetKernelOutput(bias3_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("resnet_basic_block_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_5(kernel_out_5, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_6(kernel_out_6, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_7(kernel_out_7, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_8(kernel_out_8, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_9(kernel_out_9, kernel_result.is_stride_kernel);

  phi::ResnetBasicBlockGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter1), MakeMetaTensor(*input_conv1), MakeMetaTensor(*input_scale1), MakeMetaTensor(*input_bias1), MakeMetaTensor(*input_saved_mean1), MakeMetaTensor(*input_saved_invstd1), MakeMetaTensor(*input_filter2), MakeMetaTensor(*input_conv2), MakeMetaTensor(*input_conv2_input), MakeMetaTensor(*input_scale2), MakeMetaTensor(*input_bias2), MakeMetaTensor(*input_saved_mean2), MakeMetaTensor(*input_saved_invstd2), MakeMetaTensor(input_filter3), MakeMetaTensor(input_conv3), MakeMetaTensor(input_scale3), MakeMetaTensor(input_bias3), MakeMetaTensor(input_saved_mean3), MakeMetaTensor(input_saved_invstd3), MakeMetaTensor(*input_max_input1), MakeMetaTensor(*input_max_filter1), MakeMetaTensor(*input_max_input2), MakeMetaTensor(*input_max_filter2), MakeMetaTensor(*input_max_input3), MakeMetaTensor(*input_max_filter3), MakeMetaTensor(*input_out), MakeMetaTensor(*input_out_grad), stride1, stride2, stride3, padding1, padding2, padding3, dilation1, dilation2, dilation3, group, momentum, epsilon, data_format, has_shortcut, use_global_stats, is_test, trainable_statistics, act_type, find_conv_input_max, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr, kernel_out_6 ? &meta_out_6 : nullptr, kernel_out_7 ? &meta_out_7 : nullptr, kernel_out_8 ? &meta_out_8 : nullptr, kernel_out_9 ? &meta_out_9 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, int, int, int, int, int, int, int, int, int, float, float, const std::string&, bool, bool, bool, bool, const std::string&, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("resnet_basic_block_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter1, *input_conv1, *input_scale1, *input_bias1, *input_saved_mean1, *input_saved_invstd1, *input_filter2, *input_conv2, *input_conv2_input, *input_scale2, *input_bias2, *input_saved_mean2, *input_saved_invstd2, input_filter3, input_conv3, input_scale3, input_bias3, input_saved_mean3, input_saved_invstd3, *input_max_input1, *input_max_filter1, *input_max_input2, *input_max_filter2, *input_max_input3, *input_max_filter3, *input_out, *input_out_grad, stride1, stride2, stride3, padding1, padding2, padding3, dilation1, dilation2, dilation3, group, momentum, epsilon, data_format, has_shortcut, use_global_stats, is_test, trainable_statistics, act_type, find_conv_input_max, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5, kernel_out_6, kernel_out_7, kernel_out_8, kernel_out_9);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "resnet_basic_block_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
    TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
    TransDataBackend(kernel_out_6, kernel_backend, kernel_out_6);
    TransDataBackend(kernel_out_7, kernel_backend, kernel_out_7);
    TransDataBackend(kernel_out_8, kernel_backend, kernel_out_8);
    TransDataBackend(kernel_out_9, kernel_backend, kernel_out_9);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}

PADDLE_API void resnet_unit_grad(const Tensor& x, const Tensor& filter_x, const Tensor& conv_x, const Tensor& scale_x, const Tensor& bias_x, const Tensor& saved_mean_x, const Tensor& saved_invstd_x, const paddle::optional<Tensor>& z, const paddle::optional<Tensor>& filter_z, const paddle::optional<Tensor>& conv_z, const paddle::optional<Tensor>& scale_z, const paddle::optional<Tensor>& bias_z, const paddle::optional<Tensor>& saved_mean_z, const paddle::optional<Tensor>& saved_invstd_z, const Tensor& out, const Tensor& bit_mask, const Tensor& out_grad, int stride, int stride_z, int padding, int dilation, int group, float momentum, float epsilon, const std::string& data_format, bool fuse_add, bool has_shortcut, bool use_global_stats, bool is_test, bool use_addto, const std::string& act_type, Tensor* x_grad, Tensor* filter_x_grad, Tensor* scale_x_grad, Tensor* bias_x_grad, Tensor* z_grad, Tensor* filter_z_grad, Tensor* scale_z_grad, Tensor* bias_z_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter_x, conv_x, scale_x, bias_x, saved_mean_x, saved_invstd_x, z, filter_z, conv_z, scale_z, bias_z, saved_mean_z, saved_invstd_z, out, bit_mask, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "resnet_unit_grad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "resnet_unit_grad", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("resnet_unit_grad", kernel_data_type);
  }
  VLOG(6) << "resnet_unit_grad kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_filter_x = PrepareData(filter_x, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv_x = PrepareData(conv_x, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale_x = PrepareData(scale_x, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias_x = PrepareData(bias_x, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_mean_x = PrepareData(saved_mean_x, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_invstd_x = PrepareData(saved_invstd_x, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_z = PrepareData(z, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_filter_z = PrepareData(filter_z, GetKernelInputArgDef(kernel.InputAt(8), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_conv_z = PrepareData(conv_z, GetKernelInputArgDef(kernel.InputAt(9), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale_z = PrepareData(scale_z, GetKernelInputArgDef(kernel.InputAt(10), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias_z = PrepareData(bias_z, GetKernelInputArgDef(kernel.InputAt(11), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_mean_z = PrepareData(saved_mean_z, GetKernelInputArgDef(kernel.InputAt(12), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_saved_invstd_z = PrepareData(saved_invstd_z, GetKernelInputArgDef(kernel.InputAt(13), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out = PrepareData(out, GetKernelInputArgDef(kernel.InputAt(14), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bit_mask = PrepareData(bit_mask, GetKernelInputArgDef(kernel.InputAt(15), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_grad = PrepareData(out_grad, GetKernelInputArgDef(kernel.InputAt(16), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> z_record_shapes;
     if(input_z){
       z_record_shapes.push_back((*input_z).dims());
     }
     std::vector<phi::DDim> filter_z_record_shapes;
     if(input_filter_z){
       filter_z_record_shapes.push_back((*input_filter_z).dims());
     }
     std::vector<phi::DDim> conv_z_record_shapes;
     if(input_conv_z){
       conv_z_record_shapes.push_back((*input_conv_z).dims());
     }
     std::vector<phi::DDim> scale_z_record_shapes;
     if(input_scale_z){
       scale_z_record_shapes.push_back((*input_scale_z).dims());
     }
     std::vector<phi::DDim> bias_z_record_shapes;
     if(input_bias_z){
       bias_z_record_shapes.push_back((*input_bias_z).dims());
     }
     std::vector<phi::DDim> saved_mean_z_record_shapes;
     if(input_saved_mean_z){
       saved_mean_z_record_shapes.push_back((*input_saved_mean_z).dims());
     }
     std::vector<phi::DDim> saved_invstd_z_record_shapes;
     if(input_saved_invstd_z){
       saved_invstd_z_record_shapes.push_back((*input_saved_invstd_z).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"filter_x", {
     (*input_filter_x).dims()}},
     {"conv_x", {
     (*input_conv_x).dims()}},
     {"scale_x", {
     (*input_scale_x).dims()}},
     {"bias_x", {
     (*input_bias_x).dims()}},
     {"saved_mean_x", {
     (*input_saved_mean_x).dims()}},
     {"saved_invstd_x", {
     (*input_saved_invstd_x).dims()}},
     {"z", z_record_shapes},
     {"filter_z", filter_z_record_shapes},
     {"conv_z", conv_z_record_shapes},
     {"scale_z", scale_z_record_shapes},
     {"bias_z", bias_z_record_shapes},
     {"saved_mean_z", saved_mean_z_record_shapes},
     {"saved_invstd_z", saved_invstd_z_record_shapes},
     {"out", {
     (*input_out).dims()}},
     {"bit_mask", {
     (*input_bit_mask).dims()}},
     {"out_grad", {
     (*input_out_grad).dims()}}};
     phi::AttributeMap attrs;
     attrs["stride"] = stride;
     attrs["stride_z"] = stride_z;
     attrs["padding"] = padding;
     attrs["dilation"] = dilation;
     attrs["group"] = group;
     attrs["momentum"] = momentum;
     attrs["epsilon"] = epsilon;
     attrs["data_format"] = data_format;
     attrs["fuse_add"] = fuse_add;
     attrs["has_shortcut"] = has_shortcut;
     attrs["use_global_stats"] = use_global_stats;
     attrs["is_test"] = is_test;
     attrs["use_addto"] = use_addto;
     attrs["act_type"] = act_type;
     phi::RecordOpInfoSupplement("resnet_unit_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(x_grad);
  auto kernel_out_1 = SetKernelOutput(filter_x_grad);
  auto kernel_out_2 = SetKernelOutput(scale_x_grad);
  auto kernel_out_3 = SetKernelOutput(bias_x_grad);
  auto kernel_out_4 = SetKernelOutput(z_grad);
  auto kernel_out_5 = SetKernelOutput(filter_z_grad);
  auto kernel_out_6 = SetKernelOutput(scale_z_grad);
  auto kernel_out_7 = SetKernelOutput(bias_z_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("resnet_unit_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_5(kernel_out_5, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_6(kernel_out_6, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_7(kernel_out_7, kernel_result.is_stride_kernel);

  phi::ResnetUnitGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter_x), MakeMetaTensor(*input_conv_x), MakeMetaTensor(*input_scale_x), MakeMetaTensor(*input_bias_x), MakeMetaTensor(*input_saved_mean_x), MakeMetaTensor(*input_saved_invstd_x), MakeMetaTensor(input_z), MakeMetaTensor(input_filter_z), MakeMetaTensor(input_conv_z), MakeMetaTensor(input_scale_z), MakeMetaTensor(input_bias_z), MakeMetaTensor(input_saved_mean_z), MakeMetaTensor(input_saved_invstd_z), MakeMetaTensor(*input_out), MakeMetaTensor(*input_bit_mask), MakeMetaTensor(*input_out_grad), stride, stride_z, padding, dilation, group, momentum, epsilon, data_format, fuse_add, has_shortcut, use_global_stats, is_test, use_addto, act_type, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr, kernel_out_6 ? &meta_out_6 : nullptr, kernel_out_7 ? &meta_out_7 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, int, int, int, int, float, float, const std::string&, bool, bool, bool, bool, bool, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("resnet_unit_grad kernel launch", phi::TracerEventType::DygraphKernelLaunch, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter_x, *input_conv_x, *input_scale_x, *input_bias_x, *input_saved_mean_x, *input_saved_invstd_x, input_z, input_filter_z, input_conv_z, input_scale_z, input_bias_z, input_saved_mean_z, input_saved_invstd_z, *input_out, *input_bit_mask, *input_out_grad, stride, stride_z, padding, dilation, group, momentum, epsilon, data_format, fuse_add, has_shortcut, use_global_stats, is_test, use_addto, act_type, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5, kernel_out_6, kernel_out_7);
  if (FLAGS_benchmark) {
      dev_ctx->Wait();
      std::cout << "resnet_unit_grad kernel run finish." << std::endl;
  }
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
    TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
    TransDataBackend(kernel_out_6, kernel_backend, kernel_out_6);
    TransDataBackend(kernel_out_7, kernel_backend, kernel_out_7);

  }
  dev_ctx = GetDeviceContextByBackend(kernel_backend);

  
}


}  // namespace experimental
}  // namespace paddle
