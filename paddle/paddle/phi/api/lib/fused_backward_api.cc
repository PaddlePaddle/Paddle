
#include "paddle/phi/api/backward/fused_backward_api.h"
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

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#endif

PD_DECLARE_bool(conv2d_disable_cudnn);
COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle {
namespace experimental {


PADDLE_API void fused_bias_dropout_residual_layer_norm_grad(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, const Tensor& ln_mean, const Tensor& ln_variance, const Tensor& bias_dropout_residual_out, const Tensor& dropout_mask_out, const Tensor& y_grad, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon, Tensor* x_grad, Tensor* residual_grad, Tensor* bias_grad, Tensor* ln_scale_grad, Tensor* ln_bias_grad) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, residual, bias, ln_scale, ln_bias, ln_mean, ln_variance, bias_dropout_residual_out, dropout_mask_out, y_grad);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(y_grad.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
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
  }

  // Kernel Dispatch Body
  // Auto Parallel condition
  if (run_auto_parallel) {
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs)
    auto meta_dist_input_x = MakeDistMetaTensor(*x.impl());
    auto meta_dist_input_residual = MakeDistMetaTensor(*residual.impl());
    auto meta_dist_input_bias = bias ? MakeDistMetaTensor(*(*bias).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_ln_scale = ln_scale ? MakeDistMetaTensor(*(*ln_scale).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_ln_bias = ln_bias ? MakeDistMetaTensor(*(*ln_bias).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_ln_mean = MakeDistMetaTensor(*ln_mean.impl());
    auto meta_dist_input_ln_variance = MakeDistMetaTensor(*ln_variance.impl());
    auto meta_dist_input_bias_dropout_residual_out = MakeDistMetaTensor(*bias_dropout_residual_out.impl());
    auto meta_dist_input_dropout_mask_out = MakeDistMetaTensor(*dropout_mask_out.impl());
    auto meta_dist_input_y_grad = MakeDistMetaTensor(*y_grad.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_residual, meta_dist_input_bias, meta_dist_input_ln_scale, meta_dist_input_ln_bias, meta_dist_input_ln_mean, meta_dist_input_ln_variance, meta_dist_input_bias_dropout_residual_out, meta_dist_input_dropout_mask_out, meta_dist_input_y_grad);
    DebugInfoForInferSpmd("fused_bias_dropout_residual_layer_norm_grad", spmd_info);

    // 2. Create Temporary Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_0 =
        CreateKernelDistOutput(x_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_0 = shared_dist_out_0.get();
    phi::DenseTensor* dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (dense_out_0 && !rank_is_in_current_mesh && !dist_out_0->defined()) {
      *dense_out_0 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_1 =
        CreateKernelDistOutput(residual_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_1 = shared_dist_out_1.get();
    phi::DenseTensor* dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (dense_out_1 && !rank_is_in_current_mesh && !dist_out_1->defined()) {
      *dense_out_1 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_2 =
        CreateKernelDistOutput(bias_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_2 = shared_dist_out_2.get();
    phi::DenseTensor* dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (dense_out_2 && !rank_is_in_current_mesh && !dist_out_2->defined()) {
      *dense_out_2 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_3 =
        CreateKernelDistOutput(ln_scale_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_3 = shared_dist_out_3.get();
    phi::DenseTensor* dense_out_3 = dist_out_3 ? dist_out_3->unsafe_mutable_value() : nullptr;
    if (dense_out_3 && !rank_is_in_current_mesh && !dist_out_3->defined()) {
      *dense_out_3 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_4 =
        CreateKernelDistOutput(ln_bias_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_4 = shared_dist_out_4.get();
    phi::DenseTensor* dense_out_4 = dist_out_4 ? dist_out_4->unsafe_mutable_value() : nullptr;
    if (dense_out_4 && !rank_is_in_current_mesh && !dist_out_4->defined()) {
      *dense_out_4 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_out_3(dist_out_3);
    phi::MetaTensor meta_dist_out_4(dist_out_4);
    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_ln_scale = ln_scale ? MakeMetaTensor(*(*ln_scale).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_ln_bias = ln_bias ? MakeMetaTensor(*(*ln_bias).impl()) : phi::MetaTensor();

    phi::FusedBiasDropoutResidualLnGradInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*residual.impl()), meta_dist_bias, meta_dist_ln_scale, meta_dist_ln_bias, MakeMetaTensor(*ln_mean.impl()), MakeMetaTensor(*ln_variance.impl()), MakeMetaTensor(*bias_dropout_residual_out.impl()), MakeMetaTensor(*dropout_mask_out.impl()), MakeMetaTensor(*y_grad.impl()), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr, dist_out_4 ? &meta_dist_out_4 : nullptr);


    // 4. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_3, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_4, current_process_mesh);

    if (rank_is_in_current_mesh) {
      // 5. Select Kernel
      VLOG(6) << "fused_bias_dropout_residual_layer_norm_grad API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_bias_dropout_residual_layer_norm_grad", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_bias_dropout_residual_layer_norm_grad kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 6. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_residual = ReshardApiInputToKernelInput(dev_ctx, residual, spmd_info.first[1], "residual");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");
      auto dist_input_ln_scale = ReshardApiInputToKernelInput(dev_ctx, ln_scale, spmd_info.first[3], "ln_scale");
      auto dist_input_ln_bias = ReshardApiInputToKernelInput(dev_ctx, ln_bias, spmd_info.first[4], "ln_bias");
      auto dist_input_ln_mean = ReshardApiInputToKernelInput(dev_ctx, ln_mean, spmd_info.first[5], "ln_mean");
      auto dist_input_ln_variance = ReshardApiInputToKernelInput(dev_ctx, ln_variance, spmd_info.first[6], "ln_variance");
      auto dist_input_bias_dropout_residual_out = ReshardApiInputToKernelInput(dev_ctx, bias_dropout_residual_out, spmd_info.first[7], "bias_dropout_residual_out");
      auto dist_input_dropout_mask_out = ReshardApiInputToKernelInput(dev_ctx, dropout_mask_out, spmd_info.first[8], "dropout_mask_out");
      auto dist_input_y_grad = ReshardApiInputToKernelInput(dev_ctx, y_grad, spmd_info.first[9], "y_grad");

      // 7. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_residual = PrepareDataForDistTensor(dist_input_residual, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_residual = &dist_input_residual->value();

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      dist_input_ln_scale = PrepareDataForDistTensor(dist_input_ln_scale, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_ln_scale = dist_input_ln_scale ? paddle::make_optional<phi::DenseTensor>((*dist_input_ln_scale)->value()) : paddle::none;

      dist_input_ln_bias = PrepareDataForDistTensor(dist_input_ln_bias, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_ln_bias = dist_input_ln_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_ln_bias)->value()) : paddle::none;

      dist_input_ln_mean = PrepareDataForDistTensor(dist_input_ln_mean, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_ln_mean = &dist_input_ln_mean->value();

      dist_input_ln_variance = PrepareDataForDistTensor(dist_input_ln_variance, GetKernelInputArgDef(kernel.InputAt(6), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_ln_variance = &dist_input_ln_variance->value();

      dist_input_bias_dropout_residual_out = PrepareDataForDistTensor(dist_input_bias_dropout_residual_out, GetKernelInputArgDef(kernel.InputAt(7), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_bias_dropout_residual_out = &dist_input_bias_dropout_residual_out->value();

      dist_input_dropout_mask_out = PrepareDataForDistTensor(dist_input_dropout_mask_out, GetKernelInputArgDef(kernel.InputAt(8), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_dropout_mask_out = &dist_input_dropout_mask_out->value();

      dist_input_y_grad = PrepareDataForDistTensor(dist_input_y_grad, GetKernelInputArgDef(kernel.InputAt(9), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_y_grad = &dist_input_y_grad->value();

      // 8. RecordOpInfoSupplement
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
      // 9. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::MetaTensor meta_dense_out_4(dense_out_4);
      phi::FusedBiasDropoutResidualLnGradInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), MakeMetaTensor(*input_ln_mean), MakeMetaTensor(*input_ln_variance), MakeMetaTensor(*input_bias_dropout_residual_out), MakeMetaTensor(*input_dropout_mask_out), MakeMetaTensor(*input_y_grad), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr, dense_out_4 ? &meta_dense_out_4 : nullptr);

      // 10. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm_grad dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, *input_ln_mean, *input_ln_variance, *input_bias_dropout_residual_out, *input_dropout_mask_out, *input_y_grad, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0, dense_out_1, dense_out_2, dense_out_3, dense_out_4);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 11. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
        TransDataBackend(dense_out_3, kernel_backend, dense_out_3);
        TransDataBackend(dense_out_4, kernel_backend, dense_out_4);
      }
    }
    // 12. Reshard Kernel Output to API output
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_0, x_grad, "x_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_1, residual_grad, "residual_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_2, bias_grad, "bias_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_3, ln_scale_grad, "ln_scale_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_4, ln_bias_grad, "ln_bias_grad");

    // 13. Return
    return;
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
    kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm_grad compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, *input_ln_mean, *input_ln_variance, *input_bias_dropout_residual_out, *input_dropout_mask_out, *input_y_grad, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4);
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
  
}

PADDLE_API void fused_dot_product_attention_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_out, const Tensor& rng_state, const Tensor& mask, const Tensor& out_grad, float scaling_factor, float dropout_probability, bool is_causal_masking, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(q, k, v, out, softmax_out, rng_state, mask, out_grad);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(out_grad.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(q);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(q, k, v, out, softmax_out, rng_state, mask, out_grad);
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
  }

  // Kernel Dispatch Body
  // Auto Parallel condition
  if (run_auto_parallel) {
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs)
    auto meta_dist_input_q = MakeDistMetaTensor(*q.impl());
    auto meta_dist_input_k = MakeDistMetaTensor(*k.impl());
    auto meta_dist_input_v = MakeDistMetaTensor(*v.impl());
    auto meta_dist_input_out = MakeDistMetaTensor(*out.impl());
    auto meta_dist_input_softmax_out = MakeDistMetaTensor(*softmax_out.impl());
    auto meta_dist_input_rng_state = MakeDistMetaTensor(*rng_state.impl());
    auto meta_dist_input_mask = MakeDistMetaTensor(*mask.impl());
    auto meta_dist_input_out_grad = MakeDistMetaTensor(*out_grad.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_q, meta_dist_input_k, meta_dist_input_v, meta_dist_input_out, meta_dist_input_softmax_out, meta_dist_input_rng_state, meta_dist_input_mask, meta_dist_input_out_grad);
    DebugInfoForInferSpmd("fused_dot_product_attention_grad", spmd_info);

    // 2. Create Temporary Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_0 =
        CreateKernelDistOutput(q_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_0 = shared_dist_out_0.get();
    phi::DenseTensor* dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (dense_out_0 && !rank_is_in_current_mesh && !dist_out_0->defined()) {
      *dense_out_0 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_1 =
        CreateKernelDistOutput(k_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_1 = shared_dist_out_1.get();
    phi::DenseTensor* dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (dense_out_1 && !rank_is_in_current_mesh && !dist_out_1->defined()) {
      *dense_out_1 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_2 =
        CreateKernelDistOutput(v_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_2 = shared_dist_out_2.get();
    phi::DenseTensor* dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (dense_out_2 && !rank_is_in_current_mesh && !dist_out_2->defined()) {
      *dense_out_2 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::FusedDotProductAttentionGradInferMeta(MakeMetaTensor(*q.impl()), MakeMetaTensor(*k.impl()), MakeMetaTensor(*v.impl()), dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    // 4. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);

    if (rank_is_in_current_mesh) {
      // 5. Select Kernel
      VLOG(6) << "fused_dot_product_attention_grad API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_dot_product_attention_grad", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_dot_product_attention_grad kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 6. Reshard Input
      auto dist_input_q = ReshardApiInputToKernelInput(dev_ctx, q, spmd_info.first[0], "q");
      auto dist_input_k = ReshardApiInputToKernelInput(dev_ctx, k, spmd_info.first[1], "k");
      auto dist_input_v = ReshardApiInputToKernelInput(dev_ctx, v, spmd_info.first[2], "v");
      auto dist_input_out = ReshardApiInputToKernelInput(dev_ctx, out, spmd_info.first[3], "out");
      auto dist_input_softmax_out = ReshardApiInputToKernelInput(dev_ctx, softmax_out, spmd_info.first[4], "softmax_out");
      auto dist_input_rng_state = ReshardApiInputToKernelInput(dev_ctx, rng_state, spmd_info.first[5], "rng_state");
      auto dist_input_mask = ReshardApiInputToKernelInput(dev_ctx, mask, spmd_info.first[6], "mask");
      auto dist_input_out_grad = ReshardApiInputToKernelInput(dev_ctx, out_grad, spmd_info.first[7], "out_grad");

      // 7. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_q = PrepareDataForDistTensor(dist_input_q, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_q = &dist_input_q->value();

      dist_input_k = PrepareDataForDistTensor(dist_input_k, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_k = &dist_input_k->value();

      dist_input_v = PrepareDataForDistTensor(dist_input_v, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_v = &dist_input_v->value();

      dist_input_out = PrepareDataForDistTensor(dist_input_out, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_out = &dist_input_out->value();

      dist_input_softmax_out = PrepareDataForDistTensor(dist_input_softmax_out, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_softmax_out = &dist_input_softmax_out->value();

      dist_input_rng_state = PrepareDataForDistTensor(dist_input_rng_state, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_rng_state = &dist_input_rng_state->value();

      dist_input_mask = PrepareDataForDistTensor(dist_input_mask, GetKernelInputArgDef(kernel.InputAt(6), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_mask = &dist_input_mask->value();

      dist_input_out_grad = PrepareDataForDistTensor(dist_input_out_grad, GetKernelInputArgDef(kernel.InputAt(7), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_out_grad = &dist_input_out_grad->value();

      // 8. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"q", {
         (*input_q).dims()}},
         {"k", {
         (*input_k).dims()}},
         {"v", {
         (*input_v).dims()}},
         {"out", {
         (*input_out).dims()}},
         {"softmax_out", {
         (*input_softmax_out).dims()}},
         {"rng_state", {
         (*input_rng_state).dims()}},
         {"mask", {
         (*input_mask).dims()}},
         {"out_grad", {
         (*input_out_grad).dims()}}};
         phi::AttributeMap attrs;
         attrs["scaling_factor"] = scaling_factor;
         attrs["dropout_probability"] = dropout_probability;
         attrs["is_causal_masking"] = is_causal_masking;
         phi::RecordOpInfoSupplement("fused_dot_product_attention_grad", input_shapes, attrs);
      }
      // 9. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::FusedDotProductAttentionGradInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 10. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_dot_product_attention_grad dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_out, *input_softmax_out, *input_rng_state, *input_mask, *input_out_grad, scaling_factor, dropout_probability, is_causal_masking, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 11. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }
    // 12. Reshard Kernel Output to API output
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_0, q_grad, "q_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_1, k_grad, "k_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_2, v_grad, "v_grad");

    // 13. Return
    return;
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
  auto input_out = PrepareData(out, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_softmax_out = PrepareData(softmax_out, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_rng_state = PrepareData(rng_state, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_mask = PrepareData(mask, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_grad = PrepareData(out_grad, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"q", {
     (*input_q).dims()}},
     {"k", {
     (*input_k).dims()}},
     {"v", {
     (*input_v).dims()}},
     {"out", {
     (*input_out).dims()}},
     {"softmax_out", {
     (*input_softmax_out).dims()}},
     {"rng_state", {
     (*input_rng_state).dims()}},
     {"mask", {
     (*input_mask).dims()}},
     {"out_grad", {
     (*input_out_grad).dims()}}};
     phi::AttributeMap attrs;
     attrs["scaling_factor"] = scaling_factor;
     attrs["dropout_probability"] = dropout_probability;
     attrs["is_causal_masking"] = is_causal_masking;
     phi::RecordOpInfoSupplement("fused_dot_product_attention_grad", input_shapes, attrs);
  }

  auto kernel_out_0 = SetKernelOutput(q_grad);
  auto kernel_out_1 = SetKernelOutput(k_grad);
  auto kernel_out_2 = SetKernelOutput(v_grad);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_dot_product_attention_grad infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::FusedDotProductAttentionGradInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_dot_product_attention_grad compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_out, *input_softmax_out, *input_rng_state, *input_mask, *input_out_grad, scaling_factor, dropout_probability, is_causal_masking, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  
}

PADDLE_API void fused_dropout_add_grad(const Tensor& seed_offset, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, bool fix_seed, Tensor* x_grad, Tensor* y_grad) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(seed_offset, out_grad);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(out_grad.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
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
  }

  // Kernel Dispatch Body
  // Auto Parallel condition
  if (run_auto_parallel) {
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs)
    auto meta_dist_input_seed_offset = MakeDistMetaTensor(*seed_offset.impl());
    auto meta_dist_input_out_grad = MakeDistMetaTensor(*out_grad.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_seed_offset, meta_dist_input_out_grad);
    DebugInfoForInferSpmd("fused_dropout_add_grad", spmd_info);

    // 2. Create Temporary Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_0 =
        CreateKernelDistOutput(x_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_0 = shared_dist_out_0.get();
    phi::DenseTensor* dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (dense_out_0 && !rank_is_in_current_mesh && !dist_out_0->defined()) {
      *dense_out_0 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_1 =
        CreateKernelDistOutput(y_grad, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_1 = shared_dist_out_1.get();
    phi::DenseTensor* dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (dense_out_1 && !rank_is_in_current_mesh && !dist_out_1->defined()) {
      *dense_out_1 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::FusedDropoutAddGradInferMeta(MakeMetaTensor(*seed_offset.impl()), MakeMetaTensor(*out_grad.impl()), dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    // 4. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    if (rank_is_in_current_mesh) {
      // 5. Select Kernel
      VLOG(6) << "fused_dropout_add_grad API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_dropout_add_grad", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_dropout_add_grad kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 6. Reshard Input
      auto dist_input_seed_offset = ReshardApiInputToKernelInput(dev_ctx, seed_offset, spmd_info.first[0], "seed_offset");
      auto dist_input_out_grad = ReshardApiInputToKernelInput(dev_ctx, out_grad, spmd_info.first[1], "out_grad");

      // 7. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_seed_offset = PrepareDataForDistTensor(dist_input_seed_offset, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_seed_offset = &dist_input_seed_offset->value();

      dist_input_out_grad = PrepareDataForDistTensor(dist_input_out_grad, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_out_grad = &dist_input_out_grad->value();

      // 8. RecordOpInfoSupplement
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
      // 9. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::FusedDropoutAddGradInferMeta(MakeMetaTensor(*input_seed_offset), MakeMetaTensor(*input_out_grad), dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 10. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_dropout_add_grad dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, bool, const std::string&, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_seed_offset, *input_out_grad, phi::Scalar(p), is_test, mode, fix_seed, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 11. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }
    // 12. Reshard Kernel Output to API output
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_0, x_grad, "x_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_1, y_grad, "y_grad");

    // 13. Return
    return;
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
    kernel_record_event = new phi::RecordEvent("fused_dropout_add_grad compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_seed_offset, *input_out_grad, phi::Scalar(p), is_test, mode, fix_seed, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  
}

PADDLE_API void fused_rotary_position_embedding_grad(const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, const Tensor& out_q_grad, const paddle::optional<Tensor>& out_k_grad, const paddle::optional<Tensor>& out_v_grad, bool use_neox_rotary_style, bool time_major, float rotary_emb_base, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(sin, cos, position_ids, out_q_grad, out_k_grad, out_v_grad);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(out_q_grad.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
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
  }

  // Kernel Dispatch Body
  // Auto Parallel condition
  if (run_auto_parallel) {
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs)
    auto meta_dist_input_sin = sin ? MakeDistMetaTensor(*(*sin).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cos = cos ? MakeDistMetaTensor(*(*cos).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_position_ids = position_ids ? MakeDistMetaTensor(*(*position_ids).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_out_q_grad = MakeDistMetaTensor(*out_q_grad.impl());
    auto meta_dist_input_out_k_grad = out_k_grad ? MakeDistMetaTensor(*(*out_k_grad).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_out_v_grad = out_v_grad ? MakeDistMetaTensor(*(*out_v_grad).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::FusedRopeGradInferSpmd(meta_dist_input_sin, meta_dist_input_cos, meta_dist_input_position_ids, meta_dist_input_out_q_grad, meta_dist_input_out_k_grad, meta_dist_input_out_v_grad, use_neox_rotary_style, time_major, rotary_emb_base);
    DebugInfoForInferSpmd("fused_rotary_position_embedding_grad", spmd_info);

    // 2. Create Temporary Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_0 =
        CreateKernelDistOutput(q_grad, !rank_is_in_current_mesh, spmd_info.second[0]);
    phi::distributed::DistTensor* dist_out_0 = shared_dist_out_0.get();
    phi::DenseTensor* dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (dense_out_0 && !rank_is_in_current_mesh && !dist_out_0->defined()) {
      *dense_out_0 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_1 =
        CreateKernelDistOutput(k_grad, !rank_is_in_current_mesh, spmd_info.second[1]);
    phi::distributed::DistTensor* dist_out_1 = shared_dist_out_1.get();
    phi::DenseTensor* dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (dense_out_1 && !rank_is_in_current_mesh && !dist_out_1->defined()) {
      *dense_out_1 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_2 =
        CreateKernelDistOutput(v_grad, !rank_is_in_current_mesh, spmd_info.second[2]);
    phi::distributed::DistTensor* dist_out_2 = shared_dist_out_2.get();
    phi::DenseTensor* dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (dense_out_2 && !rank_is_in_current_mesh && !dist_out_2->defined()) {
      *dense_out_2 = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_sin = sin ? MakeMetaTensor(*(*sin).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cos = cos ? MakeMetaTensor(*(*cos).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_position_ids = position_ids ? MakeMetaTensor(*(*position_ids).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_out_k_grad = out_k_grad ? MakeMetaTensor(*(*out_k_grad).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_out_v_grad = out_v_grad ? MakeMetaTensor(*(*out_v_grad).impl()) : phi::MetaTensor();

    phi::FusedRopeGradInferMeta(meta_dist_sin, meta_dist_cos, meta_dist_position_ids, MakeMetaTensor(*out_q_grad.impl()), meta_dist_out_k_grad, meta_dist_out_v_grad, use_neox_rotary_style, time_major, rotary_emb_base, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    // 4. Set Output Dist Attr For Default Impl
    // API `fused_rotary_position_embedding_grad` does not need to set DistAttr for output.

    if (rank_is_in_current_mesh) {
      // 5. Select Kernel
      VLOG(6) << "fused_rotary_position_embedding_grad API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_rotary_position_embedding_grad", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_rotary_position_embedding_grad kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 6. Reshard Input
      auto dist_input_sin = ReshardApiInputToKernelInput(dev_ctx, sin, spmd_info.first[0], "sin");
      auto dist_input_cos = ReshardApiInputToKernelInput(dev_ctx, cos, spmd_info.first[1], "cos");
      auto dist_input_position_ids = ReshardApiInputToKernelInput(dev_ctx, position_ids, spmd_info.first[2], "position_ids");
      auto dist_input_out_q_grad = ReshardApiInputToKernelInput(dev_ctx, out_q_grad, spmd_info.first[3], "out_q_grad");
      auto dist_input_out_k_grad = ReshardApiInputToKernelInput(dev_ctx, out_k_grad, spmd_info.first[4], "out_k_grad");
      auto dist_input_out_v_grad = ReshardApiInputToKernelInput(dev_ctx, out_v_grad, spmd_info.first[5], "out_v_grad");

      // 7. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_sin = PrepareDataForDistTensor(dist_input_sin, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_sin = dist_input_sin ? paddle::make_optional<phi::DenseTensor>((*dist_input_sin)->value()) : paddle::none;

      dist_input_cos = PrepareDataForDistTensor(dist_input_cos, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cos = dist_input_cos ? paddle::make_optional<phi::DenseTensor>((*dist_input_cos)->value()) : paddle::none;

      dist_input_position_ids = PrepareDataForDistTensor(dist_input_position_ids, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_position_ids = dist_input_position_ids ? paddle::make_optional<phi::DenseTensor>((*dist_input_position_ids)->value()) : paddle::none;

      dist_input_out_q_grad = PrepareDataForDistTensor(dist_input_out_q_grad, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_out_q_grad = &dist_input_out_q_grad->value();

      dist_input_out_k_grad = PrepareDataForDistTensor(dist_input_out_k_grad, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_out_k_grad = dist_input_out_k_grad ? paddle::make_optional<phi::DenseTensor>((*dist_input_out_k_grad)->value()) : paddle::none;

      dist_input_out_v_grad = PrepareDataForDistTensor(dist_input_out_v_grad, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_out_v_grad = dist_input_out_v_grad ? paddle::make_optional<phi::DenseTensor>((*dist_input_out_v_grad)->value()) : paddle::none;

      // 8. RecordOpInfoSupplement
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
      // 9. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::FusedRopeGradInferMeta(MakeMetaTensor(input_sin), MakeMetaTensor(input_cos), MakeMetaTensor(input_position_ids), MakeMetaTensor(*input_out_q_grad), MakeMetaTensor(input_out_k_grad), MakeMetaTensor(input_out_v_grad), use_neox_rotary_style, time_major, rotary_emb_base, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 10. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_rotary_position_embedding_grad dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, input_sin, input_cos, input_position_ids, *input_out_q_grad, input_out_k_grad, input_out_v_grad, use_neox_rotary_style, time_major, rotary_emb_base, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 11. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }
    // 12. Reshard Kernel Output to API output
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_0, q_grad, "q_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_1, k_grad, "k_grad");
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_2, v_grad, "v_grad");

    // 13. Return
    return;
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
    kernel_record_event = new phi::RecordEvent("fused_rotary_position_embedding_grad compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, input_sin, input_cos, input_position_ids, *input_out_q_grad, input_out_k_grad, input_out_v_grad, use_neox_rotary_style, time_major, rotary_emb_base, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  
}


}  // namespace experimental
}  // namespace paddle
