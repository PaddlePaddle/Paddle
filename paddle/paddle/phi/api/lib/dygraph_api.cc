#include "paddle/phi/api/lib/dygraph_api.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/multiary.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#endif

COMMON_DECLARE_int32(low_precision_op_list);

namespace paddle {
namespace experimental {


PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> flash_attn_unpadded_intermediate(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal, bool return_softmax, bool is_test, const std::string& rng_name) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(q, k, v, cu_seqlens_q, cu_seqlens_k, fixed_seed_offset, attn_mask);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(cu_seqlens_k.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(q);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(q, k, v, cu_seqlens_q, cu_seqlens_k, fixed_seed_offset, attn_mask);
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
    auto meta_dist_input_cu_seqlens_q = MakeDistMetaTensor(*cu_seqlens_q.impl());
    auto meta_dist_input_cu_seqlens_k = MakeDistMetaTensor(*cu_seqlens_k.impl());
    auto meta_dist_input_fixed_seed_offset = fixed_seed_offset ? MakeDistMetaTensor(*(*fixed_seed_offset).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_attn_mask = attn_mask ? MakeDistMetaTensor(*(*attn_mask).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_q, meta_dist_input_k, meta_dist_input_v, meta_dist_input_cu_seqlens_q, meta_dist_input_cu_seqlens_k, meta_dist_input_fixed_seed_offset, meta_dist_input_attn_mask);
    DebugInfoForInferSpmd("flash_attn_unpadded", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_3 = SetKernelDistOutput(&std::get<3>(api_output));
    auto dense_out_3 = dist_out_3 ? dist_out_3->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_3 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_out_3(dist_out_3);
    phi::FlashAttnInferMeta(MakeMetaTensor(*q.impl()), MakeMetaTensor(*k.impl()), MakeMetaTensor(*v.impl()), dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "flash_attn_unpadded API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "flash_attn_unpadded", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "flash_attn_unpadded kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_q = ReshardApiInputToKernelInput(dev_ctx, q, spmd_info.first[0], "q");
      auto dist_input_k = ReshardApiInputToKernelInput(dev_ctx, k, spmd_info.first[1], "k");
      auto dist_input_v = ReshardApiInputToKernelInput(dev_ctx, v, spmd_info.first[2], "v");
      auto dist_input_cu_seqlens_q = ReshardApiInputToKernelInput(dev_ctx, cu_seqlens_q, spmd_info.first[3], "cu_seqlens_q");
      auto dist_input_cu_seqlens_k = ReshardApiInputToKernelInput(dev_ctx, cu_seqlens_k, spmd_info.first[4], "cu_seqlens_k");
      auto dist_input_fixed_seed_offset = ReshardApiInputToKernelInput(dev_ctx, fixed_seed_offset, spmd_info.first[5], "fixed_seed_offset");
      auto dist_input_attn_mask = ReshardApiInputToKernelInput(dev_ctx, attn_mask, spmd_info.first[6], "attn_mask");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_q = PrepareDataForDistTensor(dist_input_q, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_q = &dist_input_q->value();

      dist_input_k = PrepareDataForDistTensor(dist_input_k, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_k = &dist_input_k->value();

      dist_input_v = PrepareDataForDistTensor(dist_input_v, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_v = &dist_input_v->value();

      dist_input_cu_seqlens_q = PrepareDataForDistTensor(dist_input_cu_seqlens_q, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_cu_seqlens_q = &dist_input_cu_seqlens_q->value();

      dist_input_cu_seqlens_k = PrepareDataForDistTensor(dist_input_cu_seqlens_k, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_cu_seqlens_k = &dist_input_cu_seqlens_k->value();

      dist_input_fixed_seed_offset = PrepareDataForDistTensor(dist_input_fixed_seed_offset, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_fixed_seed_offset = dist_input_fixed_seed_offset ? paddle::make_optional<phi::DenseTensor>((*dist_input_fixed_seed_offset)->value()) : paddle::none;

      dist_input_attn_mask = PrepareDataForDistTensor(dist_input_attn_mask, GetKernelInputArgDef(kernel.InputAt(6), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_attn_mask = dist_input_attn_mask ? paddle::make_optional<phi::DenseTensor>((*dist_input_attn_mask)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> fixed_seed_offset_record_shapes;
         if(input_fixed_seed_offset){
           fixed_seed_offset_record_shapes.push_back((*input_fixed_seed_offset).dims());
         }
         std::vector<phi::DDim> attn_mask_record_shapes;
         if(input_attn_mask){
           attn_mask_record_shapes.push_back((*input_attn_mask).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"q", {
         (*input_q).dims()}},
         {"k", {
         (*input_k).dims()}},
         {"v", {
         (*input_v).dims()}},
         {"cu_seqlens_q", {
         (*input_cu_seqlens_q).dims()}},
         {"cu_seqlens_k", {
         (*input_cu_seqlens_k).dims()}},
         {"fixed_seed_offset", fixed_seed_offset_record_shapes},
         {"attn_mask",
         attn_mask_record_shapes}};
         phi::AttributeMap attrs;
         attrs["max_seqlen_q"] = max_seqlen_q;
         attrs["max_seqlen_k"] = max_seqlen_k;
         attrs["scale"] = scale;
         attrs["dropout"] = dropout;
         attrs["causal"] = causal;
         attrs["return_softmax"] = return_softmax;
         attrs["is_test"] = is_test;
         attrs["rng_name"] = rng_name;
         phi::RecordOpInfoSupplement("flash_attn_unpadded", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::FlashAttnInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("flash_attn_unpadded dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int64_t, int64_t, float, float, bool, bool, bool, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_cu_seqlens_q, *input_cu_seqlens_k, input_fixed_seed_offset, input_attn_mask, max_seqlen_q, max_seqlen_k, scale, dropout, causal, return_softmax, is_test, rng_name, dense_out_0, dense_out_1, dense_out_2, dense_out_3);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
        TransDataBackend(dense_out_3, kernel_backend, dense_out_3);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_3, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "flash_attn_unpadded API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flash_attn_unpadded", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("flash_attn_unpadded", kernel_data_type);
  }
  VLOG(6) << "flash_attn_unpadded kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_q = PrepareData(q, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_k = PrepareData(k, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_v = PrepareData(v, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlens_q = PrepareData(cu_seqlens_q, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlens_k = PrepareData(cu_seqlens_k, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_fixed_seed_offset = PrepareData(fixed_seed_offset, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_attn_mask = PrepareData(attn_mask, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> fixed_seed_offset_record_shapes;
     if(input_fixed_seed_offset){
       fixed_seed_offset_record_shapes.push_back((*input_fixed_seed_offset).dims());
     }
     std::vector<phi::DDim> attn_mask_record_shapes;
     if(input_attn_mask){
       attn_mask_record_shapes.push_back((*input_attn_mask).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"q", {
     (*input_q).dims()}},
     {"k", {
     (*input_k).dims()}},
     {"v", {
     (*input_v).dims()}},
     {"cu_seqlens_q", {
     (*input_cu_seqlens_q).dims()}},
     {"cu_seqlens_k", {
     (*input_cu_seqlens_k).dims()}},
     {"fixed_seed_offset", fixed_seed_offset_record_shapes},
     {"attn_mask",
     attn_mask_record_shapes}};
     phi::AttributeMap attrs;
     attrs["max_seqlen_q"] = max_seqlen_q;
     attrs["max_seqlen_k"] = max_seqlen_k;
     attrs["scale"] = scale;
     attrs["dropout"] = dropout;
     attrs["causal"] = causal;
     attrs["return_softmax"] = return_softmax;
     attrs["is_test"] = is_test;
     attrs["rng_name"] = rng_name;
     phi::RecordOpInfoSupplement("flash_attn_unpadded", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("flash_attn_unpadded infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);

  phi::FlashAttnInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int64_t, int64_t, float, float, bool, bool, bool, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("flash_attn_unpadded compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_cu_seqlens_q, *input_cu_seqlens_k, input_fixed_seed_offset, input_attn_mask, max_seqlen_q, max_seqlen_k, scale, dropout, causal, return_softmax, is_test, rng_name, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> flatten_intermediate(const Tensor& x, int start_axis, int stop_axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x);
    DebugInfoForInferSpmd("flatten", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_attr_0 = static_cast<phi::distributed::DistTensor*>((std::get<0>(api_output)).impl().get())->dist_attr();

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*x.impl()), start_axis, stop_axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "flatten API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "flatten", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "flatten kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // dense_out_0 is view output, it shares memory with input.
      // If input is resharded, dense_out_0 may hold
      // different memory with origin input.
      dense_out_0->ShareBufferWith(*input_x);
      dense_out_0->ShareInplaceVersionCounterWith(*input_x);

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["start_axis"] = start_axis;
         attrs["stop_axis"] = stop_axis;
         phi::RecordOpInfoSupplement("flatten", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("flatten dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "flatten API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flatten", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("flatten", kernel_data_type);
  }
  VLOG(6) << "flatten kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["start_axis"] = start_axis;
     attrs["stop_axis"] = stop_axis;
     phi::RecordOpInfoSupplement("flatten", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
      kernel_out_0->ShareBufferWith(*input_x);
      kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
      VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("flatten infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("flatten compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

    phi::DenseTensor * x_remap = static_cast<phi::DenseTensor*>(x.impl().get());
    x_remap->ShareBufferWith(*kernel_out_0);
    kernel_out_0->ShareInplaceVersionCounterWith(*x_remap);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> flatten_intermediate_(Tensor& x, int start_axis, int stop_axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x);
    DebugInfoForInferSpmd("flatten", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor&, Tensor> api_output{x, Tensor()};

    auto dist_out_attr_0 = static_cast<phi::distributed::DistTensor*>((std::get<0>(api_output)).impl().get())->dist_attr();

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*x.impl()), start_axis, stop_axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "flatten API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "flatten", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "flatten kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["start_axis"] = start_axis;
         attrs["stop_axis"] = stop_axis;
         phi::RecordOpInfoSupplement("flatten", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("flatten dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "flatten API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flatten", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("flatten", kernel_data_type);
  }
  VLOG(6) << "flatten kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["start_axis"] = start_axis;
     attrs["stop_axis"] = stop_axis;
     phi::RecordOpInfoSupplement("flatten", input_shapes, attrs);
  }

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("flatten infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("flatten compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> group_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_format) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, scale, bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
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
    auto meta_dist_input_scale = scale ? MakeDistMetaTensor(*(*scale).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_bias = bias ? MakeDistMetaTensor(*(*bias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_scale, meta_dist_input_bias);
    DebugInfoForInferSpmd("group_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_scale = scale ? MakeMetaTensor(*(*scale).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::GroupNormInferMeta(MakeMetaTensor(*x.impl()), meta_dist_scale, meta_dist_bias, epsilon, groups, data_format, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "group_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "group_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "group_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_scale = ReshardApiInputToKernelInput(dev_ctx, scale, spmd_info.first[1], "scale");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_scale = PrepareDataForDistTensor(dist_input_scale, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_scale = dist_input_scale ? paddle::make_optional<phi::DenseTensor>((*dist_input_scale)->value()) : paddle::none;

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> scale_record_shapes;
         if(input_scale){
           scale_record_shapes.push_back((*input_scale).dims());
         }
         std::vector<phi::DDim> bias_record_shapes;
         if(input_bias){
           bias_record_shapes.push_back((*input_bias).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"scale", scale_record_shapes},
         {"bias",
         bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["epsilon"] = epsilon;
         attrs["groups"] = groups;
         attrs["data_format"] = data_format;
         phi::RecordOpInfoSupplement("group_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::GroupNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, groups, data_format, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("group_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, groups, data_format, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "group_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "group_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("group_norm", kernel_data_type);
  }
  VLOG(6) << "group_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale = PrepareData(scale, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> scale_record_shapes;
     if(input_scale){
       scale_record_shapes.push_back((*input_scale).dims());
     }
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"scale", scale_record_shapes},
     {"bias",
     bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["epsilon"] = epsilon;
     attrs["groups"] = groups;
     attrs["data_format"] = data_format;
     phi::RecordOpInfoSupplement("group_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("group_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::GroupNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, groups, data_format, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("group_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, groups, data_format, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> huber_loss_intermediate(const Tensor& input, const Tensor& label, float delta) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(input, label);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(label.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(input, label);
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
    auto meta_dist_input_input = MakeDistMetaTensor(*input.impl());
    auto meta_dist_input_label = MakeDistMetaTensor(*label.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_input, meta_dist_input_label);
    DebugInfoForInferSpmd("huber_loss", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::HuberLossInferMeta(MakeMetaTensor(*input.impl()), MakeMetaTensor(*label.impl()), delta, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "huber_loss API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "huber_loss", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "huber_loss kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_input = ReshardApiInputToKernelInput(dev_ctx, input, spmd_info.first[0], "input");
      auto dist_input_label = ReshardApiInputToKernelInput(dev_ctx, label, spmd_info.first[1], "label");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_input = PrepareDataForDistTensor(dist_input_input, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_input = &dist_input_input->value();

      dist_input_label = PrepareDataForDistTensor(dist_input_label, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_label = &dist_input_label->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"input", {
         (*input_input).dims()}},
         {"label", {
         (*input_label).dims()}}};
         phi::AttributeMap attrs;
         attrs["delta"] = delta;
         phi::RecordOpInfoSupplement("huber_loss", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::HuberLossInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), delta, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("huber_loss dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_input, *input_label, delta, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "huber_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "huber_loss", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("huber_loss", kernel_data_type);
  }
  VLOG(6) << "huber_loss kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_input = PrepareData(input, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_label = PrepareData(label, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"input", {
     (*input_input).dims()}},
     {"label", {
     (*input_label).dims()}}};
     phi::AttributeMap attrs;
     attrs["delta"] = delta;
     phi::RecordOpInfoSupplement("huber_loss", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("huber_loss infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::HuberLossInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), delta, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("huber_loss compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, delta, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> instance_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, scale, bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
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
    auto meta_dist_input_scale = scale ? MakeDistMetaTensor(*(*scale).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_bias = bias ? MakeDistMetaTensor(*(*bias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_scale, meta_dist_input_bias);
    DebugInfoForInferSpmd("instance_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_scale = scale ? MakeMetaTensor(*(*scale).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::InstanceNormInferMeta(MakeMetaTensor(*x.impl()), meta_dist_scale, meta_dist_bias, epsilon, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "instance_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "instance_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "instance_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_scale = ReshardApiInputToKernelInput(dev_ctx, scale, spmd_info.first[1], "scale");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_scale = PrepareDataForDistTensor(dist_input_scale, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_scale = dist_input_scale ? paddle::make_optional<phi::DenseTensor>((*dist_input_scale)->value()) : paddle::none;

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> scale_record_shapes;
         if(input_scale){
           scale_record_shapes.push_back((*input_scale).dims());
         }
         std::vector<phi::DDim> bias_record_shapes;
         if(input_bias){
           bias_record_shapes.push_back((*input_bias).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"scale", scale_record_shapes},
         {"bias",
         bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["epsilon"] = epsilon;
         phi::RecordOpInfoSupplement("instance_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::InstanceNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("instance_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "instance_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "instance_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("instance_norm", kernel_data_type);
  }
  VLOG(6) << "instance_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale = PrepareData(scale, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> scale_record_shapes;
     if(input_scale){
       scale_record_shapes.push_back((*input_scale).dims());
     }
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"scale", scale_record_shapes},
     {"bias",
     bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["epsilon"] = epsilon;
     phi::RecordOpInfoSupplement("instance_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("instance_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::InstanceNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("instance_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> layer_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int begin_norm_axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, scale, bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
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
    auto meta_dist_input_scale = scale ? MakeDistMetaTensor(*(*scale).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_bias = bias ? MakeDistMetaTensor(*(*bias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::LayerNormInferSpmd(meta_dist_input_x, meta_dist_input_scale, meta_dist_input_bias, epsilon, begin_norm_axis);
    DebugInfoForInferSpmd("layer_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output), spmd_info.second[2]);
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_scale = scale ? MakeMetaTensor(*(*scale).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::LayerNormInferMeta(MakeMetaTensor(*x.impl()), meta_dist_scale, meta_dist_bias, epsilon, begin_norm_axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "layer_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "layer_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "layer_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_scale = ReshardApiInputToKernelInput(dev_ctx, scale, spmd_info.first[1], "scale");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_scale = PrepareDataForDistTensor(dist_input_scale, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_scale = dist_input_scale ? paddle::make_optional<phi::DenseTensor>((*dist_input_scale)->value()) : paddle::none;

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> scale_record_shapes;
         if(input_scale){
           scale_record_shapes.push_back((*input_scale).dims());
         }
         std::vector<phi::DDim> bias_record_shapes;
         if(input_bias){
           bias_record_shapes.push_back((*input_bias).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"scale", scale_record_shapes},
         {"bias",
         bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["epsilon"] = epsilon;
         attrs["begin_norm_axis"] = begin_norm_axis;
         phi::RecordOpInfoSupplement("layer_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::LayerNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, begin_norm_axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("layer_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, begin_norm_axis, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `layer_norm` does not need to set DistAttr for output.

    // 12. Return
    return api_output;
  }

  VLOG(6) << "layer_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "layer_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("layer_norm", kernel_data_type);
  }
  VLOG(6) << "layer_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_scale = PrepareData(scale, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> scale_record_shapes;
     if(input_scale){
       scale_record_shapes.push_back((*input_scale).dims());
     }
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"scale", scale_record_shapes},
     {"bias",
     bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["epsilon"] = epsilon;
     attrs["begin_norm_axis"] = begin_norm_axis;
     phi::RecordOpInfoSupplement("layer_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("layer_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::LayerNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, begin_norm_axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("layer_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, begin_norm_axis, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> rms_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, bias, residual, norm_weight, norm_bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(norm_weight.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, bias, residual, norm_weight, norm_bias);
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
    auto meta_dist_input_bias = bias ? MakeDistMetaTensor(*(*bias).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_residual = residual ? MakeDistMetaTensor(*(*residual).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_norm_weight = MakeDistMetaTensor(*norm_weight.impl());
    auto meta_dist_input_norm_bias = norm_bias ? MakeDistMetaTensor(*(*norm_bias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_bias, meta_dist_input_residual, meta_dist_input_norm_weight, meta_dist_input_norm_bias);
    DebugInfoForInferSpmd("rms_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_residual = residual ? MakeMetaTensor(*(*residual).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_norm_bias = norm_bias ? MakeMetaTensor(*(*norm_bias).impl()) : phi::MetaTensor();

    phi::RmsNormInferMeta(MakeMetaTensor(*x.impl()), meta_dist_bias, meta_dist_residual, MakeMetaTensor(*norm_weight.impl()), meta_dist_norm_bias, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "rms_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "rms_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "rms_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[1], "bias");
      auto dist_input_residual = ReshardApiInputToKernelInput(dev_ctx, residual, spmd_info.first[2], "residual");
      auto dist_input_norm_weight = ReshardApiInputToKernelInput(dev_ctx, norm_weight, spmd_info.first[3], "norm_weight");
      auto dist_input_norm_bias = ReshardApiInputToKernelInput(dev_ctx, norm_bias, spmd_info.first[4], "norm_bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      dist_input_residual = PrepareDataForDistTensor(dist_input_residual, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_residual = dist_input_residual ? paddle::make_optional<phi::DenseTensor>((*dist_input_residual)->value()) : paddle::none;

      dist_input_norm_weight = PrepareDataForDistTensor(dist_input_norm_weight, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_norm_weight = &dist_input_norm_weight->value();

      dist_input_norm_bias = PrepareDataForDistTensor(dist_input_norm_bias, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_norm_bias = dist_input_norm_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_norm_bias)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> bias_record_shapes;
         if(input_bias){
           bias_record_shapes.push_back((*input_bias).dims());
         }
         std::vector<phi::DDim> residual_record_shapes;
         if(input_residual){
           residual_record_shapes.push_back((*input_residual).dims());
         }
         std::vector<phi::DDim> norm_bias_record_shapes;
         if(input_norm_bias){
           norm_bias_record_shapes.push_back((*input_norm_bias).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"bias", bias_record_shapes},
         {"residual", residual_record_shapes},
         {"norm_weight", {
         (*input_norm_weight).dims()}},
         {"norm_bias",
         norm_bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["epsilon"] = epsilon;
         attrs["begin_norm_axis"] = begin_norm_axis;
         attrs["quant_scale"] = quant_scale;
         attrs["quant_round_type"] = quant_round_type;
         attrs["quant_max_bound"] = quant_max_bound;
         attrs["quant_min_bound"] = quant_min_bound;
         phi::RecordOpInfoSupplement("rms_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::RmsNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_residual), MakeMetaTensor(*input_norm_weight), MakeMetaTensor(input_norm_bias), epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("rms_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, int, float, int, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_residual, *input_norm_weight, input_norm_bias, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "rms_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "rms_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("rms_norm", kernel_data_type);
  }
  VLOG(6) << "rms_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_residual = PrepareData(residual, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_norm_weight = PrepareData(norm_weight, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_norm_bias = PrepareData(norm_bias, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<phi::DDim> residual_record_shapes;
     if(input_residual){
       residual_record_shapes.push_back((*input_residual).dims());
     }
     std::vector<phi::DDim> norm_bias_record_shapes;
     if(input_norm_bias){
       norm_bias_record_shapes.push_back((*input_norm_bias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"bias", bias_record_shapes},
     {"residual", residual_record_shapes},
     {"norm_weight", {
     (*input_norm_weight).dims()}},
     {"norm_bias",
     norm_bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["epsilon"] = epsilon;
     attrs["begin_norm_axis"] = begin_norm_axis;
     attrs["quant_scale"] = quant_scale;
     attrs["quant_round_type"] = quant_round_type;
     attrs["quant_max_bound"] = quant_max_bound;
     attrs["quant_min_bound"] = quant_min_bound;
     phi::RecordOpInfoSupplement("rms_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("rms_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::RmsNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_residual), MakeMetaTensor(*input_norm_weight), MakeMetaTensor(input_norm_bias), epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, int, float, int, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("rms_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_residual, *input_norm_weight, input_norm_bias, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> roi_pool_intermediate(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, boxes, boxes_num);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(boxes.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, boxes, boxes_num);
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
    auto meta_dist_input_boxes = MakeDistMetaTensor(*boxes.impl());
    auto meta_dist_input_boxes_num = boxes_num ? MakeDistMetaTensor(*(*boxes_num).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_boxes, meta_dist_input_boxes_num);
    DebugInfoForInferSpmd("roi_pool", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_boxes_num = boxes_num ? MakeMetaTensor(*(*boxes_num).impl()) : phi::MetaTensor();

    phi::RoiPoolInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*boxes.impl()), meta_dist_boxes_num, pooled_height, pooled_width, spatial_scale, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "roi_pool API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "roi_pool", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "roi_pool kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_boxes = ReshardApiInputToKernelInput(dev_ctx, boxes, spmd_info.first[1], "boxes");
      auto dist_input_boxes_num = ReshardApiInputToKernelInput(dev_ctx, boxes_num, spmd_info.first[2], "boxes_num");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_boxes = PrepareDataForDistTensor(dist_input_boxes, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_boxes = &dist_input_boxes->value();

      dist_input_boxes_num = PrepareDataForDistTensor(dist_input_boxes_num, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_boxes_num = dist_input_boxes_num ? paddle::make_optional<phi::DenseTensor>((*dist_input_boxes_num)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> boxes_num_record_shapes;
         if(input_boxes_num){
           boxes_num_record_shapes.push_back((*input_boxes_num).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"boxes", {
         (*input_boxes).dims()}},
         {"boxes_num",
         boxes_num_record_shapes}};
         phi::AttributeMap attrs;
         attrs["pooled_height"] = pooled_height;
         attrs["pooled_width"] = pooled_width;
         attrs["spatial_scale"] = spatial_scale;
         phi::RecordOpInfoSupplement("roi_pool", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::RoiPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_boxes), MakeMetaTensor(input_boxes_num), pooled_height, pooled_width, spatial_scale, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("roi_pool dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int, int, float, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_boxes, input_boxes_num, pooled_height, pooled_width, spatial_scale, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "roi_pool API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "roi_pool", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("roi_pool", kernel_data_type);
  }
  VLOG(6) << "roi_pool kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_boxes = PrepareData(boxes, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_boxes_num = PrepareData(boxes_num, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> boxes_num_record_shapes;
     if(input_boxes_num){
       boxes_num_record_shapes.push_back((*input_boxes_num).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"boxes", {
     (*input_boxes).dims()}},
     {"boxes_num",
     boxes_num_record_shapes}};
     phi::AttributeMap attrs;
     attrs["pooled_height"] = pooled_height;
     attrs["pooled_width"] = pooled_width;
     attrs["spatial_scale"] = spatial_scale;
     phi::RecordOpInfoSupplement("roi_pool", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("roi_pool infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::RoiPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_boxes), MakeMetaTensor(input_boxes_num), pooled_height, pooled_width, spatial_scale, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int, int, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("roi_pool compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_boxes, input_boxes_num, pooled_height, pooled_width, spatial_scale, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> segment_pool_intermediate(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, segment_ids);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(segment_ids.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, segment_ids);
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
    auto meta_dist_input_segment_ids = MakeDistMetaTensor(*segment_ids.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_segment_ids);
    DebugInfoForInferSpmd("segment_pool", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::SegmentPoolInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*segment_ids.impl()), pooltype, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "segment_pool API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "segment_pool", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "segment_pool kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_segment_ids = ReshardApiInputToKernelInput(dev_ctx, segment_ids, spmd_info.first[1], "segment_ids");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_segment_ids = PrepareDataForDistTensor(dist_input_segment_ids, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_segment_ids = &dist_input_segment_ids->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"segment_ids", {
         (*input_segment_ids).dims()}}};
         phi::AttributeMap attrs;
         attrs["pooltype"] = pooltype;
         phi::RecordOpInfoSupplement("segment_pool", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::SegmentPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_segment_ids), pooltype, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("segment_pool dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_segment_ids, pooltype, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "segment_pool API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "segment_pool", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("segment_pool", kernel_data_type);
  }
  VLOG(6) << "segment_pool kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_segment_ids = PrepareData(segment_ids, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"segment_ids", {
     (*input_segment_ids).dims()}}};
     phi::AttributeMap attrs;
     attrs["pooltype"] = pooltype;
     phi::RecordOpInfoSupplement("segment_pool", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("segment_pool infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::SegmentPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_segment_ids), pooltype, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("segment_pool compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_segment_ids, pooltype, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> send_u_recv_intermediate(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op, const IntArray& out_size) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, src_index, dst_index);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(dst_index.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, src_index, dst_index);
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
    auto meta_dist_input_src_index = MakeDistMetaTensor(*src_index.impl());
    auto meta_dist_input_dst_index = MakeDistMetaTensor(*dst_index.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_src_index, meta_dist_input_dst_index);
    DebugInfoForInferSpmd("send_u_recv", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::SendURecvInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*src_index.impl()), MakeMetaTensor(*dst_index.impl()), reduce_op, out_size, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "send_u_recv API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "send_u_recv", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "send_u_recv kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_src_index = ReshardApiInputToKernelInput(dev_ctx, src_index, spmd_info.first[1], "src_index");
      auto dist_input_dst_index = ReshardApiInputToKernelInput(dev_ctx, dst_index, spmd_info.first[2], "dst_index");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_src_index = PrepareDataForDistTensor(dist_input_src_index, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_src_index = &dist_input_src_index->value();

      dist_input_dst_index = PrepareDataForDistTensor(dist_input_dst_index, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_dst_index = &dist_input_dst_index->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"src_index", {
         (*input_src_index).dims()}},
         {"dst_index", {
         (*input_dst_index).dims()}}};
         phi::AttributeMap attrs;
         attrs["reduce_op"] = reduce_op;
         attrs["out_size"] = out_size.GetData();
         phi::RecordOpInfoSupplement("send_u_recv", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::SendURecvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_src_index), MakeMetaTensor(*input_dst_index), reduce_op, out_size, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("send_u_recv dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_src_index, *input_dst_index, reduce_op, phi::IntArray(out_size), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "send_u_recv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "send_u_recv", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("send_u_recv", kernel_data_type);
  }
  VLOG(6) << "send_u_recv kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_src_index = PrepareData(src_index, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dst_index = PrepareData(dst_index, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"src_index", {
     (*input_src_index).dims()}},
     {"dst_index", {
     (*input_dst_index).dims()}}};
     phi::AttributeMap attrs;
     attrs["reduce_op"] = reduce_op;
     attrs["out_size"] = out_size.GetData();
     phi::RecordOpInfoSupplement("send_u_recv", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("send_u_recv infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::SendURecvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_src_index), MakeMetaTensor(*input_dst_index), reduce_op, out_size, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("send_u_recv compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_src_index, *input_dst_index, reduce_op, phi::IntArray(out_size), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> send_ue_recv_intermediate(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op, const std::string& reduce_op, const IntArray& out_size) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, y, src_index, dst_index);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(dst_index.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, src_index, dst_index);
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
    auto meta_dist_input_y = MakeDistMetaTensor(*y.impl());
    auto meta_dist_input_src_index = MakeDistMetaTensor(*src_index.impl());
    auto meta_dist_input_dst_index = MakeDistMetaTensor(*dst_index.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_y, meta_dist_input_src_index, meta_dist_input_dst_index);
    DebugInfoForInferSpmd("send_ue_recv", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::SendUERecvInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*y.impl()), MakeMetaTensor(*src_index.impl()), MakeMetaTensor(*dst_index.impl()), message_op, reduce_op, out_size, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "send_ue_recv API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "send_ue_recv", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "send_ue_recv kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_y = ReshardApiInputToKernelInput(dev_ctx, y, spmd_info.first[1], "y");
      auto dist_input_src_index = ReshardApiInputToKernelInput(dev_ctx, src_index, spmd_info.first[2], "src_index");
      auto dist_input_dst_index = ReshardApiInputToKernelInput(dev_ctx, dst_index, spmd_info.first[3], "dst_index");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_y = PrepareDataForDistTensor(dist_input_y, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_y = &dist_input_y->value();

      dist_input_src_index = PrepareDataForDistTensor(dist_input_src_index, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_src_index = &dist_input_src_index->value();

      dist_input_dst_index = PrepareDataForDistTensor(dist_input_dst_index, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_dst_index = &dist_input_dst_index->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"y", {
         (*input_y).dims()}},
         {"src_index", {
         (*input_src_index).dims()}},
         {"dst_index", {
         (*input_dst_index).dims()}}};
         phi::AttributeMap attrs;
         attrs["message_op"] = message_op;
         attrs["reduce_op"] = reduce_op;
         attrs["out_size"] = out_size.GetData();
         phi::RecordOpInfoSupplement("send_ue_recv", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::SendUERecvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), MakeMetaTensor(*input_src_index), MakeMetaTensor(*input_dst_index), message_op, reduce_op, out_size, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("send_ue_recv dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const std::string&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_y, *input_src_index, *input_dst_index, message_op, reduce_op, phi::IntArray(out_size), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "send_ue_recv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "send_ue_recv", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("send_ue_recv", kernel_data_type);
  }
  VLOG(6) << "send_ue_recv kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_y = PrepareData(y, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_src_index = PrepareData(src_index, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dst_index = PrepareData(dst_index, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"y", {
     (*input_y).dims()}},
     {"src_index", {
     (*input_src_index).dims()}},
     {"dst_index", {
     (*input_dst_index).dims()}}};
     phi::AttributeMap attrs;
     attrs["message_op"] = message_op;
     attrs["reduce_op"] = reduce_op;
     attrs["out_size"] = out_size.GetData();
     phi::RecordOpInfoSupplement("send_ue_recv", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("send_ue_recv infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::SendUERecvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), MakeMetaTensor(*input_src_index), MakeMetaTensor(*input_dst_index), message_op, reduce_op, out_size, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const std::string&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("send_ue_recv compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, *input_src_index, *input_dst_index, message_op, reduce_op, phi::IntArray(out_size), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> squeeze_intermediate(const Tensor& x, const IntArray& axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::SqueezeInferSpmd(meta_dist_input_x, axis.GetData());
    DebugInfoForInferSpmd("squeeze", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "squeeze API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "squeeze", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "squeeze kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // dense_out_0 is view output, it shares memory with input.
      // If input is resharded, dense_out_0 may hold
      // different memory with origin input.
      dense_out_0->ShareBufferWith(*input_x);
      dense_out_0->ShareInplaceVersionCounterWith(*input_x);

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["axis"] = axis.GetData();
         phi::RecordOpInfoSupplement("squeeze", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("squeeze dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `squeeze` does not need to set DistAttr for output.

    // 12. Return
    return api_output;
  }

  VLOG(6) << "squeeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "squeeze", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("squeeze", kernel_data_type);
  }
  VLOG(6) << "squeeze kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["axis"] = axis.GetData();
     phi::RecordOpInfoSupplement("squeeze", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
      kernel_out_0->ShareBufferWith(*input_x);
      kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
      VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("squeeze infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("squeeze compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

    phi::DenseTensor * x_remap = static_cast<phi::DenseTensor*>(x.impl().get());
    x_remap->ShareBufferWith(*kernel_out_0);
    kernel_out_0->ShareInplaceVersionCounterWith(*x_remap);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> squeeze_intermediate_(Tensor& x, const IntArray& axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::SqueezeInferSpmd(meta_dist_input_x, axis.GetData());
    DebugInfoForInferSpmd("squeeze", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor&, Tensor> api_output{x, Tensor()};

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "squeeze API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "squeeze", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "squeeze kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["axis"] = axis.GetData();
         phi::RecordOpInfoSupplement("squeeze", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("squeeze dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `squeeze` does not need to set DistAttr for output.
    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_0 = std::get<0>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_0, spmd_info.second[0], false);


    // 12. Return
    return api_output;
  }

  VLOG(6) << "squeeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "squeeze", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("squeeze", kernel_data_type);
  }
  VLOG(6) << "squeeze kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["axis"] = axis.GetData();
     phi::RecordOpInfoSupplement("squeeze", input_shapes, attrs);
  }

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("squeeze infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("squeeze compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> unsqueeze_intermediate(const Tensor& x, const IntArray& axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::UnsqueezeInferSpmd(meta_dist_input_x, axis.GetData());
    DebugInfoForInferSpmd("unsqueeze", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "unsqueeze API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "unsqueeze", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "unsqueeze kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // dense_out_0 is view output, it shares memory with input.
      // If input is resharded, dense_out_0 may hold
      // different memory with origin input.
      dense_out_0->ShareBufferWith(*input_x);
      dense_out_0->ShareInplaceVersionCounterWith(*input_x);

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["axis"] = axis.GetData();
         phi::RecordOpInfoSupplement("unsqueeze", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("unsqueeze dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `unsqueeze` does not need to set DistAttr for output.

    // 12. Return
    return api_output;
  }

  VLOG(6) << "unsqueeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unsqueeze", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("unsqueeze", kernel_data_type);
  }
  VLOG(6) << "unsqueeze kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["axis"] = axis.GetData();
     phi::RecordOpInfoSupplement("unsqueeze", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
      kernel_out_0->ShareBufferWith(*input_x);
      kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
      VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("unsqueeze infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("unsqueeze compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

    phi::DenseTensor * x_remap = static_cast<phi::DenseTensor*>(x.impl().get());
    x_remap->ShareBufferWith(*kernel_out_0);
    kernel_out_0->ShareInplaceVersionCounterWith(*x_remap);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> unsqueeze_intermediate_(Tensor& x, const IntArray& axis) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::UnsqueezeInferSpmd(meta_dist_input_x, axis.GetData());
    DebugInfoForInferSpmd("unsqueeze", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor&, Tensor> api_output{x, Tensor()};

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), axis, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "unsqueeze API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "unsqueeze", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "unsqueeze kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["axis"] = axis.GetData();
         phi::RecordOpInfoSupplement("unsqueeze", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("unsqueeze dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `unsqueeze` does not need to set DistAttr for output.
    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_0 = std::get<0>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_0, spmd_info.second[0], false);


    // 12. Return
    return api_output;
  }

  VLOG(6) << "unsqueeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unsqueeze", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("unsqueeze", kernel_data_type);
  }
  VLOG(6) << "unsqueeze kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["axis"] = axis.GetData();
     phi::RecordOpInfoSupplement("unsqueeze", input_shapes, attrs);
  }

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("unsqueeze infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("unsqueeze compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> warpctc_intermediate(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(logits, label, logits_length, labels_length);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(label.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(logits);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(logits, label, logits_length, labels_length);
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
    auto meta_dist_input_logits = MakeDistMetaTensor(*logits.impl());
    auto meta_dist_input_label = MakeDistMetaTensor(*label.impl());
    auto meta_dist_input_logits_length = logits_length ? MakeDistMetaTensor(*(*logits_length).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_labels_length = labels_length ? MakeDistMetaTensor(*(*labels_length).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_logits, meta_dist_input_label, meta_dist_input_logits_length, meta_dist_input_labels_length);
    DebugInfoForInferSpmd("warpctc", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_logits_length = logits_length ? MakeMetaTensor(*(*logits_length).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_labels_length = labels_length ? MakeMetaTensor(*(*labels_length).impl()) : phi::MetaTensor();

    phi::WarpctcInferMeta(MakeMetaTensor(*logits.impl()), MakeMetaTensor(*label.impl()), meta_dist_logits_length, meta_dist_labels_length, blank, norm_by_times, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "warpctc API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "warpctc", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "warpctc kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_logits = ReshardApiInputToKernelInput(dev_ctx, logits, spmd_info.first[0], "logits");
      auto dist_input_label = ReshardApiInputToKernelInput(dev_ctx, label, spmd_info.first[1], "label");
      auto dist_input_logits_length = ReshardApiInputToKernelInput(dev_ctx, logits_length, spmd_info.first[2], "logits_length");
      auto dist_input_labels_length = ReshardApiInputToKernelInput(dev_ctx, labels_length, spmd_info.first[3], "labels_length");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_logits = PrepareDataForDistTensor(dist_input_logits, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_logits = &dist_input_logits->value();

      dist_input_label = PrepareDataForDistTensor(dist_input_label, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_label = &dist_input_label->value();

      dist_input_logits_length = PrepareDataForDistTensor(dist_input_logits_length, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_logits_length = dist_input_logits_length ? paddle::make_optional<phi::DenseTensor>((*dist_input_logits_length)->value()) : paddle::none;

      dist_input_labels_length = PrepareDataForDistTensor(dist_input_labels_length, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_labels_length = dist_input_labels_length ? paddle::make_optional<phi::DenseTensor>((*dist_input_labels_length)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> logits_length_record_shapes;
         if(input_logits_length){
           logits_length_record_shapes.push_back((*input_logits_length).dims());
         }
         std::vector<phi::DDim> labels_length_record_shapes;
         if(input_labels_length){
           labels_length_record_shapes.push_back((*input_labels_length).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"logits", {
         (*input_logits).dims()}},
         {"label", {
         (*input_label).dims()}},
         {"logits_length", logits_length_record_shapes},
         {"labels_length",
         labels_length_record_shapes}};
         phi::AttributeMap attrs;
         attrs["blank"] = blank;
         attrs["norm_by_times"] = norm_by_times;
         phi::RecordOpInfoSupplement("warpctc", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::WarpctcInferMeta(MakeMetaTensor(*input_logits), MakeMetaTensor(*input_label), MakeMetaTensor(input_logits_length), MakeMetaTensor(input_labels_length), blank, norm_by_times, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("warpctc dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_logits, *input_label, input_logits_length, input_labels_length, blank, norm_by_times, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "warpctc API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "warpctc", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("warpctc", kernel_data_type);
  }
  VLOG(6) << "warpctc kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_logits = PrepareData(logits, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_label = PrepareData(label, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_logits_length = PrepareData(logits_length, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_labels_length = PrepareData(labels_length, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> logits_length_record_shapes;
     if(input_logits_length){
       logits_length_record_shapes.push_back((*input_logits_length).dims());
     }
     std::vector<phi::DDim> labels_length_record_shapes;
     if(input_labels_length){
       labels_length_record_shapes.push_back((*input_labels_length).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"logits", {
     (*input_logits).dims()}},
     {"label", {
     (*input_label).dims()}},
     {"logits_length", logits_length_record_shapes},
     {"labels_length",
     labels_length_record_shapes}};
     phi::AttributeMap attrs;
     attrs["blank"] = blank;
     attrs["norm_by_times"] = norm_by_times;
     phi::RecordOpInfoSupplement("warpctc", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("warpctc infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::WarpctcInferMeta(MakeMetaTensor(*input_logits), MakeMetaTensor(*input_label), MakeMetaTensor(input_logits_length), MakeMetaTensor(input_labels_length), blank, norm_by_times, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("warpctc compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_logits, *input_label, input_logits_length, input_labels_length, blank, norm_by_times, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> warprnnt_intermediate(const Tensor& input, const Tensor& label, const Tensor& input_lengths, const Tensor& label_lengths, int blank, float fastemit_lambda) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(input, label, input_lengths, label_lengths);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(label_lengths.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(input);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(input, label, input_lengths, label_lengths);
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
    auto meta_dist_input_input = MakeDistMetaTensor(*input.impl());
    auto meta_dist_input_label = MakeDistMetaTensor(*label.impl());
    auto meta_dist_input_input_lengths = MakeDistMetaTensor(*input_lengths.impl());
    auto meta_dist_input_label_lengths = MakeDistMetaTensor(*label_lengths.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_input, meta_dist_input_label, meta_dist_input_input_lengths, meta_dist_input_label_lengths);
    DebugInfoForInferSpmd("warprnnt", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::WarprnntInferMeta(MakeMetaTensor(*input.impl()), MakeMetaTensor(*label.impl()), MakeMetaTensor(*input_lengths.impl()), MakeMetaTensor(*label_lengths.impl()), blank, fastemit_lambda, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "warprnnt API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "warprnnt", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "warprnnt kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_input = ReshardApiInputToKernelInput(dev_ctx, input, spmd_info.first[0], "input");
      auto dist_input_label = ReshardApiInputToKernelInput(dev_ctx, label, spmd_info.first[1], "label");
      auto dist_input_input_lengths = ReshardApiInputToKernelInput(dev_ctx, input_lengths, spmd_info.first[2], "input_lengths");
      auto dist_input_label_lengths = ReshardApiInputToKernelInput(dev_ctx, label_lengths, spmd_info.first[3], "label_lengths");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_input = PrepareDataForDistTensor(dist_input_input, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_input = &dist_input_input->value();

      dist_input_label = PrepareDataForDistTensor(dist_input_label, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_label = &dist_input_label->value();

      dist_input_input_lengths = PrepareDataForDistTensor(dist_input_input_lengths, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_input_lengths = &dist_input_input_lengths->value();

      dist_input_label_lengths = PrepareDataForDistTensor(dist_input_label_lengths, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_label_lengths = &dist_input_label_lengths->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"input", {
         (*input_input).dims()}},
         {"label", {
         (*input_label).dims()}},
         {"input_lengths", {
         (*input_input_lengths).dims()}},
         {"label_lengths", {
         (*input_label_lengths).dims()}}};
         phi::AttributeMap attrs;
         attrs["blank"] = blank;
         attrs["fastemit_lambda"] = fastemit_lambda;
         phi::RecordOpInfoSupplement("warprnnt", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::WarprnntInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), MakeMetaTensor(*input_input_lengths), MakeMetaTensor(*input_label_lengths), blank, fastemit_lambda, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("warprnnt dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, float, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_input, *input_label, *input_input_lengths, *input_label_lengths, blank, fastemit_lambda, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "warprnnt API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "warprnnt", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("warprnnt", kernel_data_type);
  }
  VLOG(6) << "warprnnt kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_input = PrepareData(input, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_label = PrepareData(label, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_input_lengths = PrepareData(input_lengths, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_label_lengths = PrepareData(label_lengths, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"input", {
     (*input_input).dims()}},
     {"label", {
     (*input_label).dims()}},
     {"input_lengths", {
     (*input_input_lengths).dims()}},
     {"label_lengths", {
     (*input_label_lengths).dims()}}};
     phi::AttributeMap attrs;
     attrs["blank"] = blank;
     attrs["fastemit_lambda"] = fastemit_lambda;
     phi::RecordOpInfoSupplement("warprnnt", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("warprnnt infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::WarprnntInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), MakeMetaTensor(*input_input_lengths), MakeMetaTensor(*input_label_lengths), blank, fastemit_lambda, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("warprnnt compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, *input_input_lengths, *input_label_lengths, blank, fastemit_lambda, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> yolo_loss_intermediate(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, gt_box, gt_label, gt_score);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(gt_label.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, gt_box, gt_label, gt_score);
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
    auto meta_dist_input_gt_box = MakeDistMetaTensor(*gt_box.impl());
    auto meta_dist_input_gt_label = MakeDistMetaTensor(*gt_label.impl());
    auto meta_dist_input_gt_score = gt_score ? MakeDistMetaTensor(*(*gt_score).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_gt_box, meta_dist_input_gt_label, meta_dist_input_gt_score);
    DebugInfoForInferSpmd("yolo_loss", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_2 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_gt_score = gt_score ? MakeMetaTensor(*(*gt_score).impl()) : phi::MetaTensor();

    phi::YoloLossInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*gt_box.impl()), MakeMetaTensor(*gt_label.impl()), meta_dist_gt_score, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "yolo_loss API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "yolo_loss", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "yolo_loss kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_gt_box = ReshardApiInputToKernelInput(dev_ctx, gt_box, spmd_info.first[1], "gt_box");
      auto dist_input_gt_label = ReshardApiInputToKernelInput(dev_ctx, gt_label, spmd_info.first[2], "gt_label");
      auto dist_input_gt_score = ReshardApiInputToKernelInput(dev_ctx, gt_score, spmd_info.first[3], "gt_score");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_gt_box = PrepareDataForDistTensor(dist_input_gt_box, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_gt_box = &dist_input_gt_box->value();

      dist_input_gt_label = PrepareDataForDistTensor(dist_input_gt_label, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_gt_label = &dist_input_gt_label->value();

      dist_input_gt_score = PrepareDataForDistTensor(dist_input_gt_score, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_gt_score = dist_input_gt_score ? paddle::make_optional<phi::DenseTensor>((*dist_input_gt_score)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> gt_score_record_shapes;
         if(input_gt_score){
           gt_score_record_shapes.push_back((*input_gt_score).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"gt_box", {
         (*input_gt_box).dims()}},
         {"gt_label", {
         (*input_gt_label).dims()}},
         {"gt_score",
         gt_score_record_shapes}};
         phi::AttributeMap attrs;
         attrs["anchors"] = anchors;
         attrs["anchor_mask"] = anchor_mask;
         attrs["class_num"] = class_num;
         attrs["ignore_thresh"] = ignore_thresh;
         attrs["downsample_ratio"] = downsample_ratio;
         attrs["use_label_smooth"] = use_label_smooth;
         attrs["scale_x_y"] = scale_x_y;
         phi::RecordOpInfoSupplement("yolo_loss", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::YoloLossInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_gt_box), MakeMetaTensor(*input_gt_label), MakeMetaTensor(input_gt_score), anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("yolo_loss dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const std::vector<int>&, const std::vector<int>&, int, float, int, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_gt_box, *input_gt_label, input_gt_score, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y, dense_out_0, dense_out_1, dense_out_2);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "yolo_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "yolo_loss", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("yolo_loss", kernel_data_type);
  }
  VLOG(6) << "yolo_loss kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_gt_box = PrepareData(gt_box, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_gt_label = PrepareData(gt_label, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_gt_score = PrepareData(gt_score, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> gt_score_record_shapes;
     if(input_gt_score){
       gt_score_record_shapes.push_back((*input_gt_score).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"gt_box", {
     (*input_gt_box).dims()}},
     {"gt_label", {
     (*input_gt_label).dims()}},
     {"gt_score",
     gt_score_record_shapes}};
     phi::AttributeMap attrs;
     attrs["anchors"] = anchors;
     attrs["anchor_mask"] = anchor_mask;
     attrs["class_num"] = class_num;
     attrs["ignore_thresh"] = ignore_thresh;
     attrs["downsample_ratio"] = downsample_ratio;
     attrs["use_label_smooth"] = use_label_smooth;
     attrs["scale_x_y"] = scale_x_y;
     phi::RecordOpInfoSupplement("yolo_loss", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("yolo_loss infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::YoloLossInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_gt_box), MakeMetaTensor(*input_gt_label), MakeMetaTensor(input_gt_score), anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const std::vector<int>&, const std::vector<int>&, int, float, int, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("yolo_loss compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_gt_box, *input_gt_label, input_gt_score, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y, kernel_out_0, kernel_out_1, kernel_out_2);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> dropout_intermediate(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, seed_tensor);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, seed_tensor);
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
    auto meta_dist_input_seed_tensor = seed_tensor ? MakeDistMetaTensor(*(*seed_tensor).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_seed_tensor);
    DebugInfoForInferSpmd("dropout", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_seed_tensor = seed_tensor ? MakeMetaTensor(*(*seed_tensor).impl()) : phi::MetaTensor();

    phi::DropoutInferMeta(MakeMetaTensor(*x.impl()), meta_dist_seed_tensor, p, is_test, mode, seed, fix_seed, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "dropout API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "dropout", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "dropout kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_seed_tensor = ReshardApiInputToKernelInput(dev_ctx, seed_tensor, spmd_info.first[1], "seed_tensor");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_seed_tensor = PrepareDataForDistTensor(dist_input_seed_tensor, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_seed_tensor = dist_input_seed_tensor ? paddle::make_optional<phi::DenseTensor>((*dist_input_seed_tensor)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> seed_tensor_record_shapes;
         if(input_seed_tensor){
           seed_tensor_record_shapes.push_back((*input_seed_tensor).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"seed_tensor",
         seed_tensor_record_shapes}};
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
         attrs["seed"] = seed;
         attrs["fix_seed"] = fix_seed;
         phi::RecordOpInfoSupplement("dropout", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::DropoutInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_seed_tensor), p, is_test, mode, seed, fix_seed, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("dropout dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, bool, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_seed_tensor, phi::Scalar(p), is_test, mode, seed, fix_seed, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "dropout API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "dropout", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("dropout", kernel_data_type);
  }
  VLOG(6) << "dropout kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seed_tensor = PrepareData(seed_tensor, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> seed_tensor_record_shapes;
     if(input_seed_tensor){
       seed_tensor_record_shapes.push_back((*input_seed_tensor).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"seed_tensor",
     seed_tensor_record_shapes}};
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
     attrs["seed"] = seed;
     attrs["fix_seed"] = fix_seed;
     phi::RecordOpInfoSupplement("dropout", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("dropout infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::DropoutInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_seed_tensor), p, is_test, mode, seed, fix_seed, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, bool, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("dropout compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_seed_tensor, phi::Scalar(p), is_test, mode, seed, fix_seed, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> reshape_intermediate(const Tensor& x, const IntArray& shape) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::ReshapeInferSpmdDynamic(meta_dist_input_x, shape.GetData());
    DebugInfoForInferSpmd("reshape", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), shape, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "reshape API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "reshape", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "reshape kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // dense_out_0 is view output, it shares memory with input.
      // If input is resharded, dense_out_0 may hold
      // different memory with origin input.
      dense_out_0->ShareBufferWith(*input_x);
      dense_out_0->ShareInplaceVersionCounterWith(*input_x);

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["shape"] = shape.GetData();
         phi::RecordOpInfoSupplement("reshape", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);

      // The dist_input_x is a dist tensor, the dims() func return the global dims.
      auto x_shape = dist_input_x->dims();
      auto x_numel = dist_input_x->numel();
      bool visit_negative = false;
      auto global_shape = shape;
      std::vector<int> local_shape;
      for (size_t i = 0; i < global_shape.size(); i++) {
        auto& out_dist_attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmd_info.second[0]);
        if (out_dist_attr.dims_mapping()[i] >= 0) {
          int shape_i = global_shape[i];
          if (shape_i == 0) {
            shape_i = x_shape[i];
          } else if (shape_i == -1) {
            PADDLE_ENFORCE(not visit_negative,
                           phi::errors::InvalidArgument(
                               "reshape can only have one -1 in the shape."));
            visit_negative = true;
            int64_t non_negative_product = 1;
            for (size_t j = 0; j < global_shape.size(); j++) {
              if (i == j) {
                continue;
              }
              int64_t tmp_j = global_shape[j];
              if (tmp_j == 0) {
                tmp_j = x_shape[j];
              }
              non_negative_product *= tmp_j;
            }
            PADDLE_ENFORCE(x_numel % non_negative_product == 0,
                           phi::errors::InvalidArgument("Cannot infer real shape for -1."));
            shape_i = x_numel / non_negative_product;
          }
          int64_t dim = out_dist_attr.dims_mapping()[i];
          int64_t mesh_dim = out_dist_attr.process_mesh().shape()[dim];
          // TODO: Support aliquant condition.
          PADDLE_ENFORCE(shape_i % mesh_dim == 0,
                phi::errors::InvalidArgument(
                    "reshape only support local shape dim is divisible "
                    "by the mesh dim, however local_shape[%lld] is %lld "
                    "and shard mesh dims is %lld.", i, shape_i, mesh_dim));
          local_shape.push_back(shape_i / mesh_dim);
        } else {
          local_shape.push_back(shape[i]);
        }
      }

      phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), local_shape, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("reshape dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(local_shape), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `reshape` does not need to set DistAttr for output.

    // 12. Return
    return api_output;
  }

  VLOG(6) << "reshape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape", kernel_data_type);
  }
  VLOG(6) << "reshape kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["shape"] = shape.GetData();
     phi::RecordOpInfoSupplement("reshape", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
      kernel_out_0->ShareBufferWith(*input_x);
      kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
      VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("reshape infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), shape, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("reshape compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

    phi::DenseTensor * x_remap = static_cast<phi::DenseTensor*>(x.impl().get());
    x_remap->ShareBufferWith(*kernel_out_0);
    kernel_out_0->ShareInplaceVersionCounterWith(*x_remap);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> reshape_intermediate_(Tensor& x, const IntArray& shape) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::ReshapeInferSpmdDynamic(meta_dist_input_x, shape.GetData());
    DebugInfoForInferSpmd("reshape", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor&, Tensor> api_output{x, Tensor()};

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output), spmd_info.second[0]);
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output), spmd_info.second[1]);
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*x.impl()), shape, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "reshape API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "reshape", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "reshape kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["shape"] = shape.GetData();
         phi::RecordOpInfoSupplement("reshape", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);

      // The dist_input_x is a dist tensor, the dims() func return the global dims.
      auto x_shape = dist_input_x->dims();
      auto x_numel = dist_input_x->numel();
      bool visit_negative = false;
      auto global_shape = shape;
      std::vector<int> local_shape;
      for (size_t i = 0; i < global_shape.size(); i++) {
        auto& out_dist_attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmd_info.second[0]);
        if (out_dist_attr.dims_mapping()[i] >= 0) {
          int shape_i = global_shape[i];
          if (shape_i == 0) {
            shape_i = x_shape[i];
          } else if (shape_i == -1) {
            PADDLE_ENFORCE(not visit_negative,
                           phi::errors::InvalidArgument(
                               "reshape can only have one -1 in the shape."));
            visit_negative = true;
            int64_t non_negative_product = 1;
            for (size_t j = 0; j < global_shape.size(); j++) {
              if (i == j) {
                continue;
              }
              int64_t tmp_j = global_shape[j];
              if (tmp_j == 0) {
                tmp_j = x_shape[j];
              }
              non_negative_product *= tmp_j;
            }
            PADDLE_ENFORCE(x_numel % non_negative_product == 0,
                           phi::errors::InvalidArgument("Cannot infer real shape for -1."));
            shape_i = x_numel / non_negative_product;
          }
          int64_t dim = out_dist_attr.dims_mapping()[i];
          int64_t mesh_dim = out_dist_attr.process_mesh().shape()[dim];
          // TODO: Support aliquant condition.
          PADDLE_ENFORCE(shape_i % mesh_dim == 0,
                phi::errors::InvalidArgument(
                    "reshape only support local shape dim is divisible "
                    "by the mesh dim, however local_shape[%lld] is %lld "
                    "and shard mesh dims is %lld.", i, shape_i, mesh_dim));
          local_shape.push_back(shape_i / mesh_dim);
        } else {
          local_shape.push_back(shape[i]);
        }
      }

      phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), local_shape, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("reshape dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(local_shape), dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    // API `reshape` does not need to set DistAttr for output.
    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_0 = std::get<0>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_0, spmd_info.second[0], false);


    // 12. Return
    return api_output;
  }

  VLOG(6) << "reshape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("reshape", kernel_data_type);
  }
  VLOG(6) << "reshape kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["shape"] = shape.GetData();
     phi::RecordOpInfoSupplement("reshape", input_shapes, attrs);
  }

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("reshape infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), shape, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("reshape compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, std::vector<Tensor>, Tensor> rnn_intermediate(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, pre_state, weight_list, sequence_length, dropout_state_in);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(dropout_state_in.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, pre_state, weight_list, sequence_length, dropout_state_in);
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
    std::vector<phi::distributed::DistMetaTensor> meta_dist_input_pre_state;
    for(auto& e : pre_state) {
        meta_dist_input_pre_state.push_back(MakeDistMetaTensor(*e.impl()));
    }
    std::vector<phi::distributed::DistMetaTensor> meta_dist_input_weight_list;
    for(auto& e : weight_list) {
        meta_dist_input_weight_list.push_back(MakeDistMetaTensor(*e.impl()));
    }
    auto meta_dist_input_sequence_length = sequence_length ? MakeDistMetaTensor(*(*sequence_length).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_pre_state, meta_dist_input_weight_list, meta_dist_input_sequence_length);
    DebugInfoForInferSpmd("rnn", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, std::vector<Tensor>, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_attr_1 = static_cast<phi::distributed::DistTensor*>((std::get<1>(api_output)).impl().get())->dist_attr();

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_2 = SetKernelDistOutput(pre_state.size(), &std::get<2>(api_output));
    std::vector<phi::DenseTensor*> dense_out_2(dist_out_2.size());
    for (size_t i = 0; i < dist_out_2.size(); ++i) {
        dense_out_2[i] = const_cast<phi::DenseTensor*>(&dist_out_2[i]->value());
        if (!rank_is_in_current_mesh) {
          *dense_out_2[i] = phi::DenseTensor(
                  std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                  phi::DenseTensorMeta());
        }
    }

    auto dist_out_3 = SetKernelDistOutput(&std::get<3>(api_output));
    auto dense_out_3 = dist_out_3 ? dist_out_3->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_3 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    std::vector<phi::MetaTensor> dist_out_2_meta_vec;
    for (auto tmp : dist_out_2) {
      dist_out_2_meta_vec.emplace_back(phi::MetaTensor(tmp));
    }
    std::vector<phi::MetaTensor*> dist_out_2_meta_ptr_vec(dist_out_2.size());
    for (size_t i = 0; i < dist_out_2_meta_vec.size(); ++i) {
      dist_out_2_meta_ptr_vec[i] = &dist_out_2_meta_vec[i];
    }

    phi::MetaTensor meta_dist_out_3(dist_out_3);
    std::vector<phi::MetaTensor> pre_state_meta_vec;
    for (auto tmp : pre_state) {
      pre_state_meta_vec.emplace_back(MakeMetaTensor(*tmp.impl()));
    }
    std::vector<const phi::MetaTensor*> pre_state_meta_ptr_vec(pre_state_meta_vec.size());
    for (size_t i=0; i < pre_state_meta_ptr_vec.size(); ++i) {
      pre_state_meta_ptr_vec[i] = &pre_state_meta_vec[i];
    }

    std::vector<phi::MetaTensor> weight_list_meta_vec;
    for (auto tmp : weight_list) {
      weight_list_meta_vec.emplace_back(MakeMetaTensor(*tmp.impl()));
    }
    std::vector<const phi::MetaTensor*> weight_list_meta_ptr_vec(weight_list_meta_vec.size());
    for (size_t i=0; i < weight_list_meta_ptr_vec.size(); ++i) {
      weight_list_meta_ptr_vec[i] = &weight_list_meta_vec[i];
    }

    phi::MetaTensor meta_dist_sequence_length = sequence_length ? MakeMetaTensor(*(*sequence_length).impl()) : phi::MetaTensor();

    phi::RnnInferMeta(MakeMetaTensor(*x.impl()), pre_state_meta_ptr_vec, weight_list_meta_ptr_vec, meta_dist_sequence_length, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2_meta_ptr_vec, dist_out_3 ? &meta_dist_out_3 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "rnn API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "rnn", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "rnn kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_pre_state = ReshardApiInputToKernelInput(dev_ctx, pre_state, spmd_info.first[1], "pre_state");
      auto dist_input_weight_list = ReshardApiInputToKernelInput(dev_ctx, weight_list, spmd_info.first[2], "weight_list");
      auto dist_input_sequence_length = ReshardApiInputToKernelInput(dev_ctx, sequence_length, spmd_info.first[3], "sequence_length");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      auto dist_input_pre_state_vec = PrepareDataForDistTensor(dist_input_pre_state, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      std::vector<const phi::DenseTensor*> dense_input_pre_state_vec;
      for (auto tmp : dist_input_pre_state_vec) {
        dense_input_pre_state_vec.emplace_back(&tmp->value());
      }
      std::vector<phi::MetaTensor> dense_input_pre_state_meta_vec = MakeMetaTensor(dense_input_pre_state_vec);
      std::vector<const phi::MetaTensor*> dense_input_pre_state_meta_ptr_vec(dense_input_pre_state_meta_vec.size());
      for (size_t i = 0; i < dense_input_pre_state_meta_ptr_vec.size(); ++i) {
        dense_input_pre_state_meta_ptr_vec[i] = &dense_input_pre_state_meta_vec[i];
      }

      auto dist_input_weight_list_vec = PrepareDataForDistTensor(dist_input_weight_list, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      std::vector<const phi::DenseTensor*> dense_input_weight_list_vec;
      for (auto tmp : dist_input_weight_list_vec) {
        dense_input_weight_list_vec.emplace_back(&tmp->value());
      }
      std::vector<phi::MetaTensor> dense_input_weight_list_meta_vec = MakeMetaTensor(dense_input_weight_list_vec);
      std::vector<const phi::MetaTensor*> dense_input_weight_list_meta_ptr_vec(dense_input_weight_list_meta_vec.size());
      for (size_t i = 0; i < dense_input_weight_list_meta_ptr_vec.size(); ++i) {
        dense_input_weight_list_meta_ptr_vec[i] = &dense_input_weight_list_meta_vec[i];
      }

      dist_input_sequence_length = PrepareDataForDistTensor(dist_input_sequence_length, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_sequence_length = dist_input_sequence_length ? paddle::make_optional<phi::DenseTensor>((*dist_input_sequence_length)->value()) : paddle::none;

      // dense_out_1 is view output, it shares memory with input.
      // If input is resharded, dense_out_1 may hold
      // different memory with origin input.
      dense_out_1->ShareBufferWith(std::static_pointer_cast<phi::distributed::DistTensor>(dropout_state_in.impl())->value());
      dense_out_1->ShareInplaceVersionCounterWith(std::static_pointer_cast<phi::distributed::DistTensor>(dropout_state_in.impl())->value());

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> sequence_length_record_shapes;
         if(input_sequence_length){
           sequence_length_record_shapes.push_back((*input_sequence_length).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"sequence_length",
         sequence_length_record_shapes}};
         std::vector<phi::DDim> ddims_vec;
         ddims_vec.clear();
         ddims_vec.reserve(dense_input_pre_state_vec.size());
         for (size_t i = 0; i < dense_input_pre_state_vec.size(); ++i) {
           ddims_vec.emplace_back((*dense_input_pre_state_vec[i]).dims());
         }
         input_shapes.emplace_back("pre_state", ddims_vec);
         ddims_vec.clear();
         ddims_vec.reserve(dense_input_weight_list_vec.size());
         for (size_t i = 0; i < dense_input_weight_list_vec.size(); ++i) {
           ddims_vec.emplace_back((*dense_input_weight_list_vec[i]).dims());
         }
         input_shapes.emplace_back("weight_list", ddims_vec);
         phi::AttributeMap attrs;
         attrs["dropout_prob"] = dropout_prob;
         attrs["is_bidirec"] = is_bidirec;
         attrs["input_size"] = input_size;
         attrs["hidden_size"] = hidden_size;
         attrs["num_layers"] = num_layers;
         attrs["mode"] = mode;
         attrs["seed"] = seed;
         attrs["is_test"] = is_test;
         phi::RecordOpInfoSupplement("rnn", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      std::vector<phi::MetaTensor> dense_out_2_meta_vec = MakeMetaTensor(dense_out_2);
      std::vector<phi::MetaTensor*> dense_out_2_meta_ptr_vec(dense_out_2_meta_vec.size());
      for (size_t i = 0; i < dense_out_2_meta_vec.size(); ++i) {
        dense_out_2_meta_ptr_vec[i] = &dense_out_2_meta_vec[i];
      }

      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::RnnInferMeta(MakeMetaTensor(*input_x), dense_input_pre_state_meta_ptr_vec, dense_input_weight_list_meta_ptr_vec, MakeMetaTensor(input_sequence_length), dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2_meta_ptr_vec, dense_out_3 ? &meta_dense_out_3 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("rnn dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const std::vector<const phi::DenseTensor*>&, const std::vector<const phi::DenseTensor*>&, const paddle::optional<phi::DenseTensor>&, float, bool, int, int, int, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*, std::vector<phi::DenseTensor*>, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, dense_input_pre_state_vec, dense_input_weight_list_vec, input_sequence_length, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test, dense_out_0, dense_out_1, dense_out_2, dense_out_3);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
        TransDataBackend(dense_out_3, kernel_backend, dense_out_3);
        phi::DenseTensor* dropout_state_in_remap = static_cast<phi::distributed::DistTensor*>(dropout_state_in.impl().get())->unsafe_mutable_value();
        dropout_state_in_remap->ShareBufferWith(dist_out_1->value());
        dist_out_1->unsafe_mutable_value()->ShareInplaceVersionCounterWith(*dropout_state_in_remap);

      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    for (size_t i = 0; i < dist_out_2.size(); ++i) {
        SetReplicatedDistAttrForOutput(dist_out_2[i], current_process_mesh);
    }

    SetReplicatedDistAttrForOutput(dist_out_3, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "rnn API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "rnn", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("rnn", kernel_data_type);
  }
  VLOG(6) << "rnn kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_pre_state_vec = PrepareData(pre_state, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  std::vector<const phi::DenseTensor*> input_pre_state(input_pre_state_vec->size());
  for (size_t i = 0; i < input_pre_state.size(); ++i) {
    input_pre_state[i] = &input_pre_state_vec->at(i);
  }
  auto input_weight_list_vec = PrepareData(weight_list, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  std::vector<const phi::DenseTensor*> input_weight_list(input_weight_list_vec->size());
  for (size_t i = 0; i < input_weight_list.size(); ++i) {
    input_weight_list[i] = &input_weight_list_vec->at(i);
  }
  auto input_sequence_length = PrepareData(sequence_length, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dropout_state_in = PrepareData(dropout_state_in, kernel.InputAt(0), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> sequence_length_record_shapes;
     if(input_sequence_length){
       sequence_length_record_shapes.push_back((*input_sequence_length).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"sequence_length",
     sequence_length_record_shapes}};
     std::vector<phi::DDim> ddims_vec;
     ddims_vec.clear();
     ddims_vec.reserve(input_pre_state.size());
     for (size_t i = 0; i < input_pre_state.size(); ++i) {
       ddims_vec.emplace_back((*input_pre_state[i]).dims());
     }
     input_shapes.emplace_back("pre_state", ddims_vec);
     ddims_vec.clear();
     ddims_vec.reserve(input_weight_list.size());
     for (size_t i = 0; i < input_weight_list.size(); ++i) {
       ddims_vec.emplace_back((*input_weight_list[i]).dims());
     }
     input_shapes.emplace_back("weight_list", ddims_vec);
     phi::AttributeMap attrs;
     attrs["dropout_prob"] = dropout_prob;
     attrs["is_bidirec"] = is_bidirec;
     attrs["input_size"] = input_size;
     attrs["hidden_size"] = hidden_size;
     attrs["num_layers"] = num_layers;
     attrs["mode"] = mode;
     attrs["seed"] = seed;
     attrs["is_test"] = is_test;
     phi::RecordOpInfoSupplement("rnn", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, std::vector<Tensor>, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
      kernel_out_1->ShareBufferWith(*input_dropout_state_in);
      kernel_out_1->ShareInplaceVersionCounterWith(*input_dropout_state_in);
      VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_2 = SetKernelOutput(pre_state.size(), &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("rnn infer_meta", phi::TracerEventType::OperatorInner, 1);
  }

  auto pre_state_meta_vec = MakeMetaTensor(input_pre_state);
  std::vector<const phi::MetaTensor*> pre_state_metas(pre_state_meta_vec.size());
  for (size_t i = 0; i < pre_state_meta_vec.size(); ++i) {
    pre_state_metas[i] = &pre_state_meta_vec[i];
  }

  auto weight_list_meta_vec = MakeMetaTensor(input_weight_list);
  std::vector<const phi::MetaTensor*> weight_list_metas(weight_list_meta_vec.size());
  for (size_t i = 0; i < weight_list_meta_vec.size(); ++i) {
    weight_list_metas[i] = &weight_list_meta_vec[i];
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  auto kernel_out_2_meta_vec = MakeMetaTensor(kernel_out_2);
  std::vector<phi::MetaTensor*> kernel_out_2_metas(kernel_out_2_meta_vec.size());
  for (size_t i = 0; i < kernel_out_2_meta_vec.size(); ++i) {
    kernel_out_2_metas[i] = kernel_out_2[i] ? &kernel_out_2_meta_vec[i] : nullptr;
  }  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);

  phi::RnnInferMeta(MakeMetaTensor(*input_x), pre_state_metas, weight_list_metas, MakeMetaTensor(input_sequence_length), dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2_metas, kernel_out_3 ? &meta_out_3 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const std::vector<const phi::DenseTensor*>&, const std::vector<const phi::DenseTensor*>&, const paddle::optional<phi::DenseTensor>&, float, bool, int, int, int, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*, std::vector<phi::DenseTensor*>, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("rnn compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_pre_state, input_weight_list, input_sequence_length, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);

    phi::DenseTensor * dropout_state_in_remap = static_cast<phi::DenseTensor*>(dropout_state_in.impl().get());
    dropout_state_in_remap->ShareBufferWith(*kernel_out_1);
    kernel_out_1->ShareInplaceVersionCounterWith(*dropout_state_in_remap);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> rrelu_intermediate(const Tensor& x, float lower, float upper, bool is_test) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(x.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x);
    DebugInfoForInferSpmd("rrelu", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor> api_output;

    auto dist_out_0 = SetKernelDistOutput(&std::get<0>(api_output));
    auto dense_out_0 = dist_out_0 ? dist_out_0->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_0 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    auto dist_out_1 = SetKernelDistOutput(&std::get<1>(api_output));
    auto dense_out_1 = dist_out_1 ? dist_out_1->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
      *dense_out_1 = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::RReluInferMeta(MakeMetaTensor(*x.impl()), lower, upper, is_test, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "rrelu API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "rrelu", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "rrelu kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}}};
         phi::AttributeMap attrs;
         attrs["lower"] = lower;
         attrs["upper"] = upper;
         attrs["is_test"] = is_test;
         phi::RecordOpInfoSupplement("rrelu", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::RReluInferMeta(MakeMetaTensor(*input_x), lower, upper, is_test, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("rrelu dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, float, float, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, lower, upper, is_test, dense_out_0, dense_out_1);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "rrelu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "rrelu", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("rrelu", kernel_data_type);
  }
  VLOG(6) << "rrelu kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}}};
     phi::AttributeMap attrs;
     attrs["lower"] = lower;
     attrs["upper"] = upper;
     attrs["is_test"] = is_test;
     phi::RecordOpInfoSupplement("rrelu", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("rrelu infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::RReluInferMeta(MakeMetaTensor(*input_x), lower, upper, is_test, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, float, float, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("rrelu compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, lower, upper, is_test, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

namespace sparse {

PADDLE_API std::tuple<Tensor, Tensor, Tensor> conv3d_intermediate(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, kernel);
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

  if (x.is_sparse_coo_tensor() && phi::DenseTensor::classof(kernel.impl().get())) {

    VLOG(6) << "conv3d api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "conv3d_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("conv3d", kernel_data_type);
    }
    VLOG(6) << "conv3d api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);
    auto input_kernel = PrepareDataForDenseTensorInSparse(kernel);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::sparse::Conv3dInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_kernel), paddings, dilations, strides, groups, subm, key, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackInput(input_kernel.get());
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackAttr(groups);
    kernel_context.EmplaceBackAttr(subm);
    kernel_context.EmplaceBackAttr(key);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor> fused_attention_intermediate(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(sparse_mask);

  kernel_data_type = ParseDataType(query);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, sparse_mask, key_padding_mask, attn_mask);
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

  if (phi::DenseTensor::classof(query.impl().get()) && phi::DenseTensor::classof(key.impl().get()) && phi::DenseTensor::classof(value.impl().get()) && sparse_mask.is_sparse_csr_tensor() && (!key_padding_mask || phi::DenseTensor::classof(key_padding_mask->impl().get())) && (!attn_mask || phi::DenseTensor::classof(attn_mask->impl().get()))) {

    VLOG(6) << "fused_attention api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "fused_attention_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_attention", kernel_data_type);
    }
    VLOG(6) << "fused_attention api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::SPARSE_CSR);
    auto input_query = PrepareDataForDenseTensorInSparse(query);
    auto input_key = PrepareDataForDenseTensorInSparse(key);
    auto input_value = PrepareDataForDenseTensorInSparse(value);
    auto input_sparse_mask = PrepareDataForSparseCsrTensor(sparse_mask);
    auto input_key_padding_mask = PrepareDataForDenseTensorInSparse(key_padding_mask);
    auto input_attn_mask = PrepareDataForDenseTensorInSparse(attn_mask);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::sparse::FusedAttentionInferMeta(MakeMetaTensor(*input_query), MakeMetaTensor(*input_key), MakeMetaTensor(*input_value), MakeMetaTensor(*input_sparse_mask), MakeMetaTensor(input_key_padding_mask), MakeMetaTensor(input_attn_mask), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    kernel_context.EmplaceBackInput(input_query.get());
    kernel_context.EmplaceBackInput(input_key.get());
    kernel_context.EmplaceBackInput(input_value.get());
    kernel_context.EmplaceBackInput(input_sparse_mask.get());
    kernel_context.EmplaceBackInput(key_padding_mask ? &(*input_key_padding_mask) : nullptr);
    kernel_context.EmplaceBackInput(attn_mask ? &(*input_attn_mask) : nullptr);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> maxpool_intermediate(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
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

  if (x.is_sparse_coo_tensor()) {

    VLOG(6) << "maxpool api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "maxpool_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    if (FLAGS_low_precision_op_list) {
      phi::KernelFactory::Instance().AddToLowPrecisionKernelList("maxpool", kernel_data_type);
    }
    VLOG(6) << "maxpool api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);
    auto input_x = PrepareDataForSparseCooTensor(x);

  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::sparse::Pool3dInferMeta(MakeMetaTensor(*input_x), kernel_sizes, paddings, dilations, strides, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


    kernel_context.EmplaceBackInput(input_x.get());
    kernel_context.EmplaceBackAttr(kernel_sizes);
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool) for input tensors is unimplemented, please check the type of input tensors."));
}

}  // namespace sparse


}  // namespace experimental
}  // namespace paddle
