
#include "paddle/phi/api/include/fused_api.h"
#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"
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


PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor&> block_multihead_attention_(Tensor& qkv, Tensor& key_cache, Tensor& value_cache, const Tensor& seq_lens_encoder, const Tensor& seq_lens_decoder, const Tensor& seq_lens_this_time, const Tensor& padding_offsets, const Tensor& cum_offsets, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& block_tables, const paddle::optional<Tensor>& pre_key_cache, const paddle::optional<Tensor>& pre_value_cache, const paddle::optional<Tensor>& rope_emb, const paddle::optional<Tensor>& mask, const paddle::optional<Tensor>& tgt_mask, const paddle::optional<Tensor>& cache_k_quant_scales, const paddle::optional<Tensor>& cache_v_quant_scales, const paddle::optional<Tensor>& cache_k_dequant_scales, const paddle::optional<Tensor>& cache_v_dequant_scales, const paddle::optional<Tensor>& qkv_out_scale, const paddle::optional<Tensor>& qkv_bias, const paddle::optional<Tensor>& out_shift, const paddle::optional<Tensor>& out_smooth, int max_seq_len, int block_size, bool use_neox_style, bool dynamic_cachekv_quant, int quant_round_type, float quant_max_bound, float quant_min_bound, float out_scale, const std::string& compute_dtype) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(qkv, key_cache, value_cache, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k, block_tables, pre_key_cache, pre_value_cache, rope_emb, mask, tgt_mask, cache_k_quant_scales, cache_v_quant_scales, cache_k_dequant_scales, cache_v_dequant_scales, qkv_out_scale, qkv_bias, out_shift, out_smooth);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(block_tables.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(qkv);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(qkv, key_cache, value_cache, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k, block_tables, pre_key_cache, pre_value_cache, rope_emb, mask, tgt_mask, cache_k_quant_scales, cache_v_quant_scales, cache_k_dequant_scales, cache_v_dequant_scales, qkv_out_scale, qkv_bias, out_shift, out_smooth);
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
    auto meta_dist_input_qkv = MakeDistMetaTensor(*qkv.impl());
    auto meta_dist_input_key_cache = MakeDistMetaTensor(*key_cache.impl());
    auto meta_dist_input_value_cache = MakeDistMetaTensor(*value_cache.impl());
    auto meta_dist_input_seq_lens_encoder = MakeDistMetaTensor(*seq_lens_encoder.impl());
    auto meta_dist_input_seq_lens_decoder = MakeDistMetaTensor(*seq_lens_decoder.impl());
    auto meta_dist_input_seq_lens_this_time = MakeDistMetaTensor(*seq_lens_this_time.impl());
    auto meta_dist_input_padding_offsets = MakeDistMetaTensor(*padding_offsets.impl());
    auto meta_dist_input_cum_offsets = MakeDistMetaTensor(*cum_offsets.impl());
    auto meta_dist_input_cu_seqlens_q = MakeDistMetaTensor(*cu_seqlens_q.impl());
    auto meta_dist_input_cu_seqlens_k = MakeDistMetaTensor(*cu_seqlens_k.impl());
    auto meta_dist_input_block_tables = MakeDistMetaTensor(*block_tables.impl());
    auto meta_dist_input_pre_key_cache = pre_key_cache ? MakeDistMetaTensor(*(*pre_key_cache).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_pre_value_cache = pre_value_cache ? MakeDistMetaTensor(*(*pre_value_cache).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_rope_emb = rope_emb ? MakeDistMetaTensor(*(*rope_emb).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_mask = mask ? MakeDistMetaTensor(*(*mask).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_tgt_mask = tgt_mask ? MakeDistMetaTensor(*(*tgt_mask).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cache_k_quant_scales = cache_k_quant_scales ? MakeDistMetaTensor(*(*cache_k_quant_scales).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cache_v_quant_scales = cache_v_quant_scales ? MakeDistMetaTensor(*(*cache_v_quant_scales).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cache_k_dequant_scales = cache_k_dequant_scales ? MakeDistMetaTensor(*(*cache_k_dequant_scales).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cache_v_dequant_scales = cache_v_dequant_scales ? MakeDistMetaTensor(*(*cache_v_dequant_scales).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_qkv_out_scale = qkv_out_scale ? MakeDistMetaTensor(*(*qkv_out_scale).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_qkv_bias = qkv_bias ? MakeDistMetaTensor(*(*qkv_bias).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_out_shift = out_shift ? MakeDistMetaTensor(*(*out_shift).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_out_smooth = out_smooth ? MakeDistMetaTensor(*(*out_smooth).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_qkv, meta_dist_input_key_cache, meta_dist_input_value_cache, meta_dist_input_seq_lens_encoder, meta_dist_input_seq_lens_decoder, meta_dist_input_seq_lens_this_time, meta_dist_input_padding_offsets, meta_dist_input_cum_offsets, meta_dist_input_cu_seqlens_q, meta_dist_input_cu_seqlens_k, meta_dist_input_block_tables, meta_dist_input_pre_key_cache, meta_dist_input_pre_value_cache, meta_dist_input_rope_emb, meta_dist_input_mask, meta_dist_input_tgt_mask, meta_dist_input_cache_k_quant_scales, meta_dist_input_cache_v_quant_scales, meta_dist_input_cache_k_dequant_scales, meta_dist_input_cache_v_dequant_scales, meta_dist_input_qkv_out_scale, meta_dist_input_qkv_bias, meta_dist_input_out_shift, meta_dist_input_out_smooth);
    DebugInfoForInferSpmd("block_multihead_attention_", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor&, Tensor&, Tensor&> api_output{Tensor(), qkv, key_cache, value_cache};

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

    auto dist_out_attr_2 = static_cast<phi::distributed::DistTensor*>((std::get<2>(api_output)).impl().get())->dist_attr();

    auto dist_out_2 = SetKernelDistOutput(&std::get<2>(api_output));
    auto dense_out_2 = dist_out_2 ? dist_out_2->unsafe_mutable_value() : nullptr;

    auto dist_out_attr_3 = static_cast<phi::distributed::DistTensor*>((std::get<3>(api_output)).impl().get())->dist_attr();

    auto dist_out_3 = SetKernelDistOutput(&std::get<3>(api_output));
    auto dense_out_3 = dist_out_3 ? dist_out_3->unsafe_mutable_value() : nullptr;

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out_0(dist_out_0);
    phi::MetaTensor meta_dist_out_1(dist_out_1);
    phi::MetaTensor meta_dist_out_2(dist_out_2);
    phi::MetaTensor meta_dist_out_3(dist_out_3);
    phi::MetaTensor meta_dist_pre_key_cache = pre_key_cache ? MakeMetaTensor(*(*pre_key_cache).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_pre_value_cache = pre_value_cache ? MakeMetaTensor(*(*pre_value_cache).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_rope_emb = rope_emb ? MakeMetaTensor(*(*rope_emb).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_mask = mask ? MakeMetaTensor(*(*mask).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_tgt_mask = tgt_mask ? MakeMetaTensor(*(*tgt_mask).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cache_k_quant_scales = cache_k_quant_scales ? MakeMetaTensor(*(*cache_k_quant_scales).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cache_v_quant_scales = cache_v_quant_scales ? MakeMetaTensor(*(*cache_v_quant_scales).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cache_k_dequant_scales = cache_k_dequant_scales ? MakeMetaTensor(*(*cache_k_dequant_scales).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cache_v_dequant_scales = cache_v_dequant_scales ? MakeMetaTensor(*(*cache_v_dequant_scales).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_qkv_out_scale = qkv_out_scale ? MakeMetaTensor(*(*qkv_out_scale).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_qkv_bias = qkv_bias ? MakeMetaTensor(*(*qkv_bias).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_out_shift = out_shift ? MakeMetaTensor(*(*out_shift).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_out_smooth = out_smooth ? MakeMetaTensor(*(*out_smooth).impl()) : phi::MetaTensor();

    phi::BlockMultiheadAttentionInferMeta(MakeMetaTensor(*qkv.impl()), MakeMetaTensor(*key_cache.impl()), MakeMetaTensor(*value_cache.impl()), MakeMetaTensor(*seq_lens_encoder.impl()), MakeMetaTensor(*seq_lens_decoder.impl()), MakeMetaTensor(*seq_lens_this_time.impl()), MakeMetaTensor(*padding_offsets.impl()), MakeMetaTensor(*cum_offsets.impl()), MakeMetaTensor(*cu_seqlens_q.impl()), MakeMetaTensor(*cu_seqlens_k.impl()), MakeMetaTensor(*block_tables.impl()), meta_dist_pre_key_cache, meta_dist_pre_value_cache, meta_dist_rope_emb, meta_dist_mask, meta_dist_tgt_mask, meta_dist_cache_k_quant_scales, meta_dist_cache_v_quant_scales, meta_dist_cache_k_dequant_scales, meta_dist_cache_v_dequant_scales, meta_dist_qkv_out_scale, meta_dist_qkv_bias, meta_dist_out_shift, meta_dist_out_smooth, max_seq_len, block_size, use_neox_style, dynamic_cachekv_quant, quant_round_type, quant_max_bound, quant_min_bound, out_scale, compute_dtype, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "block_multihead_attention_ API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "block_multihead_attention", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "block_multihead_attention kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_qkv = ReshardApiInputToKernelInput(dev_ctx, qkv, spmd_info.first[0], "qkv");
      auto dist_input_key_cache = ReshardApiInputToKernelInput(dev_ctx, key_cache, spmd_info.first[1], "key_cache");
      auto dist_input_value_cache = ReshardApiInputToKernelInput(dev_ctx, value_cache, spmd_info.first[2], "value_cache");
      auto dist_input_seq_lens_encoder = ReshardApiInputToKernelInput(dev_ctx, seq_lens_encoder, spmd_info.first[3], "seq_lens_encoder");
      auto dist_input_seq_lens_decoder = ReshardApiInputToKernelInput(dev_ctx, seq_lens_decoder, spmd_info.first[4], "seq_lens_decoder");
      auto dist_input_seq_lens_this_time = ReshardApiInputToKernelInput(dev_ctx, seq_lens_this_time, spmd_info.first[5], "seq_lens_this_time");
      auto dist_input_padding_offsets = ReshardApiInputToKernelInput(dev_ctx, padding_offsets, spmd_info.first[6], "padding_offsets");
      auto dist_input_cum_offsets = ReshardApiInputToKernelInput(dev_ctx, cum_offsets, spmd_info.first[7], "cum_offsets");
      auto dist_input_cu_seqlens_q = ReshardApiInputToKernelInput(dev_ctx, cu_seqlens_q, spmd_info.first[8], "cu_seqlens_q");
      auto dist_input_cu_seqlens_k = ReshardApiInputToKernelInput(dev_ctx, cu_seqlens_k, spmd_info.first[9], "cu_seqlens_k");
      auto dist_input_block_tables = ReshardApiInputToKernelInput(dev_ctx, block_tables, spmd_info.first[10], "block_tables");
      auto dist_input_pre_key_cache = ReshardApiInputToKernelInput(dev_ctx, pre_key_cache, spmd_info.first[11], "pre_key_cache");
      auto dist_input_pre_value_cache = ReshardApiInputToKernelInput(dev_ctx, pre_value_cache, spmd_info.first[12], "pre_value_cache");
      auto dist_input_rope_emb = ReshardApiInputToKernelInput(dev_ctx, rope_emb, spmd_info.first[13], "rope_emb");
      auto dist_input_mask = ReshardApiInputToKernelInput(dev_ctx, mask, spmd_info.first[14], "mask");
      auto dist_input_tgt_mask = ReshardApiInputToKernelInput(dev_ctx, tgt_mask, spmd_info.first[15], "tgt_mask");
      auto dist_input_cache_k_quant_scales = ReshardApiInputToKernelInput(dev_ctx, cache_k_quant_scales, spmd_info.first[16], "cache_k_quant_scales");
      auto dist_input_cache_v_quant_scales = ReshardApiInputToKernelInput(dev_ctx, cache_v_quant_scales, spmd_info.first[17], "cache_v_quant_scales");
      auto dist_input_cache_k_dequant_scales = ReshardApiInputToKernelInput(dev_ctx, cache_k_dequant_scales, spmd_info.first[18], "cache_k_dequant_scales");
      auto dist_input_cache_v_dequant_scales = ReshardApiInputToKernelInput(dev_ctx, cache_v_dequant_scales, spmd_info.first[19], "cache_v_dequant_scales");
      auto dist_input_qkv_out_scale = ReshardApiInputToKernelInput(dev_ctx, qkv_out_scale, spmd_info.first[20], "qkv_out_scale");
      auto dist_input_qkv_bias = ReshardApiInputToKernelInput(dev_ctx, qkv_bias, spmd_info.first[21], "qkv_bias");
      auto dist_input_out_shift = ReshardApiInputToKernelInput(dev_ctx, out_shift, spmd_info.first[22], "out_shift");
      auto dist_input_out_smooth = ReshardApiInputToKernelInput(dev_ctx, out_smooth, spmd_info.first[23], "out_smooth");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_qkv = PrepareDataForDistTensor(dist_input_qkv, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_qkv = &dist_input_qkv->value();

      dist_input_key_cache = PrepareDataForDistTensor(dist_input_key_cache, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_key_cache = &dist_input_key_cache->value();

      dist_input_value_cache = PrepareDataForDistTensor(dist_input_value_cache, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_value_cache = &dist_input_value_cache->value();

      dist_input_seq_lens_encoder = PrepareDataForDistTensor(dist_input_seq_lens_encoder, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_seq_lens_encoder = &dist_input_seq_lens_encoder->value();

      dist_input_seq_lens_decoder = PrepareDataForDistTensor(dist_input_seq_lens_decoder, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_seq_lens_decoder = &dist_input_seq_lens_decoder->value();

      dist_input_seq_lens_this_time = PrepareDataForDistTensor(dist_input_seq_lens_this_time, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_seq_lens_this_time = &dist_input_seq_lens_this_time->value();

      dist_input_padding_offsets = PrepareDataForDistTensor(dist_input_padding_offsets, GetKernelInputArgDef(kernel.InputAt(6), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_padding_offsets = &dist_input_padding_offsets->value();

      dist_input_cum_offsets = PrepareDataForDistTensor(dist_input_cum_offsets, GetKernelInputArgDef(kernel.InputAt(7), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_cum_offsets = &dist_input_cum_offsets->value();

      dist_input_cu_seqlens_q = PrepareDataForDistTensor(dist_input_cu_seqlens_q, GetKernelInputArgDef(kernel.InputAt(8), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_cu_seqlens_q = &dist_input_cu_seqlens_q->value();

      dist_input_cu_seqlens_k = PrepareDataForDistTensor(dist_input_cu_seqlens_k, GetKernelInputArgDef(kernel.InputAt(9), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_cu_seqlens_k = &dist_input_cu_seqlens_k->value();

      dist_input_block_tables = PrepareDataForDistTensor(dist_input_block_tables, GetKernelInputArgDef(kernel.InputAt(10), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_block_tables = &dist_input_block_tables->value();

      dist_input_pre_key_cache = PrepareDataForDistTensor(dist_input_pre_key_cache, GetKernelInputArgDef(kernel.InputAt(11), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_pre_key_cache = dist_input_pre_key_cache ? paddle::make_optional<phi::DenseTensor>((*dist_input_pre_key_cache)->value()) : paddle::none;

      dist_input_pre_value_cache = PrepareDataForDistTensor(dist_input_pre_value_cache, GetKernelInputArgDef(kernel.InputAt(12), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_pre_value_cache = dist_input_pre_value_cache ? paddle::make_optional<phi::DenseTensor>((*dist_input_pre_value_cache)->value()) : paddle::none;

      dist_input_rope_emb = PrepareDataForDistTensor(dist_input_rope_emb, GetKernelInputArgDef(kernel.InputAt(13), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_rope_emb = dist_input_rope_emb ? paddle::make_optional<phi::DenseTensor>((*dist_input_rope_emb)->value()) : paddle::none;

      dist_input_mask = PrepareDataForDistTensor(dist_input_mask, GetKernelInputArgDef(kernel.InputAt(14), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_mask = dist_input_mask ? paddle::make_optional<phi::DenseTensor>((*dist_input_mask)->value()) : paddle::none;

      dist_input_tgt_mask = PrepareDataForDistTensor(dist_input_tgt_mask, GetKernelInputArgDef(kernel.InputAt(15), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_tgt_mask = dist_input_tgt_mask ? paddle::make_optional<phi::DenseTensor>((*dist_input_tgt_mask)->value()) : paddle::none;

      dist_input_cache_k_quant_scales = PrepareDataForDistTensor(dist_input_cache_k_quant_scales, GetKernelInputArgDef(kernel.InputAt(16), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cache_k_quant_scales = dist_input_cache_k_quant_scales ? paddle::make_optional<phi::DenseTensor>((*dist_input_cache_k_quant_scales)->value()) : paddle::none;

      dist_input_cache_v_quant_scales = PrepareDataForDistTensor(dist_input_cache_v_quant_scales, GetKernelInputArgDef(kernel.InputAt(17), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cache_v_quant_scales = dist_input_cache_v_quant_scales ? paddle::make_optional<phi::DenseTensor>((*dist_input_cache_v_quant_scales)->value()) : paddle::none;

      dist_input_cache_k_dequant_scales = PrepareDataForDistTensor(dist_input_cache_k_dequant_scales, GetKernelInputArgDef(kernel.InputAt(18), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cache_k_dequant_scales = dist_input_cache_k_dequant_scales ? paddle::make_optional<phi::DenseTensor>((*dist_input_cache_k_dequant_scales)->value()) : paddle::none;

      dist_input_cache_v_dequant_scales = PrepareDataForDistTensor(dist_input_cache_v_dequant_scales, GetKernelInputArgDef(kernel.InputAt(19), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cache_v_dequant_scales = dist_input_cache_v_dequant_scales ? paddle::make_optional<phi::DenseTensor>((*dist_input_cache_v_dequant_scales)->value()) : paddle::none;

      dist_input_qkv_out_scale = PrepareDataForDistTensor(dist_input_qkv_out_scale, GetKernelInputArgDef(kernel.InputAt(20), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_qkv_out_scale = dist_input_qkv_out_scale ? paddle::make_optional<phi::DenseTensor>((*dist_input_qkv_out_scale)->value()) : paddle::none;

      dist_input_qkv_bias = PrepareDataForDistTensor(dist_input_qkv_bias, GetKernelInputArgDef(kernel.InputAt(21), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_qkv_bias = dist_input_qkv_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_qkv_bias)->value()) : paddle::none;

      dist_input_out_shift = PrepareDataForDistTensor(dist_input_out_shift, GetKernelInputArgDef(kernel.InputAt(22), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_out_shift = dist_input_out_shift ? paddle::make_optional<phi::DenseTensor>((*dist_input_out_shift)->value()) : paddle::none;

      dist_input_out_smooth = PrepareDataForDistTensor(dist_input_out_smooth, GetKernelInputArgDef(kernel.InputAt(23), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_out_smooth = dist_input_out_smooth ? paddle::make_optional<phi::DenseTensor>((*dist_input_out_smooth)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> pre_key_cache_record_shapes;
         if(input_pre_key_cache){
           pre_key_cache_record_shapes.push_back((*input_pre_key_cache).dims());
         }
         std::vector<phi::DDim> pre_value_cache_record_shapes;
         if(input_pre_value_cache){
           pre_value_cache_record_shapes.push_back((*input_pre_value_cache).dims());
         }
         std::vector<phi::DDim> rope_emb_record_shapes;
         if(input_rope_emb){
           rope_emb_record_shapes.push_back((*input_rope_emb).dims());
         }
         std::vector<phi::DDim> mask_record_shapes;
         if(input_mask){
           mask_record_shapes.push_back((*input_mask).dims());
         }
         std::vector<phi::DDim> tgt_mask_record_shapes;
         if(input_tgt_mask){
           tgt_mask_record_shapes.push_back((*input_tgt_mask).dims());
         }
         std::vector<phi::DDim> cache_k_quant_scales_record_shapes;
         if(input_cache_k_quant_scales){
           cache_k_quant_scales_record_shapes.push_back((*input_cache_k_quant_scales).dims());
         }
         std::vector<phi::DDim> cache_v_quant_scales_record_shapes;
         if(input_cache_v_quant_scales){
           cache_v_quant_scales_record_shapes.push_back((*input_cache_v_quant_scales).dims());
         }
         std::vector<phi::DDim> cache_k_dequant_scales_record_shapes;
         if(input_cache_k_dequant_scales){
           cache_k_dequant_scales_record_shapes.push_back((*input_cache_k_dequant_scales).dims());
         }
         std::vector<phi::DDim> cache_v_dequant_scales_record_shapes;
         if(input_cache_v_dequant_scales){
           cache_v_dequant_scales_record_shapes.push_back((*input_cache_v_dequant_scales).dims());
         }
         std::vector<phi::DDim> qkv_out_scale_record_shapes;
         if(input_qkv_out_scale){
           qkv_out_scale_record_shapes.push_back((*input_qkv_out_scale).dims());
         }
         std::vector<phi::DDim> qkv_bias_record_shapes;
         if(input_qkv_bias){
           qkv_bias_record_shapes.push_back((*input_qkv_bias).dims());
         }
         std::vector<phi::DDim> out_shift_record_shapes;
         if(input_out_shift){
           out_shift_record_shapes.push_back((*input_out_shift).dims());
         }
         std::vector<phi::DDim> out_smooth_record_shapes;
         if(input_out_smooth){
           out_smooth_record_shapes.push_back((*input_out_smooth).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"qkv", {
         (*input_qkv).dims()}},
         {"key_cache", {
         (*input_key_cache).dims()}},
         {"value_cache", {
         (*input_value_cache).dims()}},
         {"seq_lens_encoder", {
         (*input_seq_lens_encoder).dims()}},
         {"seq_lens_decoder", {
         (*input_seq_lens_decoder).dims()}},
         {"seq_lens_this_time", {
         (*input_seq_lens_this_time).dims()}},
         {"padding_offsets", {
         (*input_padding_offsets).dims()}},
         {"cum_offsets", {
         (*input_cum_offsets).dims()}},
         {"cu_seqlens_q", {
         (*input_cu_seqlens_q).dims()}},
         {"cu_seqlens_k", {
         (*input_cu_seqlens_k).dims()}},
         {"block_tables", {
         (*input_block_tables).dims()}},
         {"pre_key_cache", pre_key_cache_record_shapes},
         {"pre_value_cache", pre_value_cache_record_shapes},
         {"rope_emb", rope_emb_record_shapes},
         {"mask", mask_record_shapes},
         {"tgt_mask", tgt_mask_record_shapes},
         {"cache_k_quant_scales", cache_k_quant_scales_record_shapes},
         {"cache_v_quant_scales", cache_v_quant_scales_record_shapes},
         {"cache_k_dequant_scales", cache_k_dequant_scales_record_shapes},
         {"cache_v_dequant_scales", cache_v_dequant_scales_record_shapes},
         {"qkv_out_scale", qkv_out_scale_record_shapes},
         {"qkv_bias", qkv_bias_record_shapes},
         {"out_shift", out_shift_record_shapes},
         {"out_smooth",
         out_smooth_record_shapes}};
         phi::AttributeMap attrs;
         attrs["max_seq_len"] = max_seq_len;
         attrs["block_size"] = block_size;
         attrs["use_neox_style"] = use_neox_style;
         attrs["dynamic_cachekv_quant"] = dynamic_cachekv_quant;
         attrs["quant_round_type"] = quant_round_type;
         attrs["quant_max_bound"] = quant_max_bound;
         attrs["quant_min_bound"] = quant_min_bound;
         attrs["out_scale"] = out_scale;
         attrs["compute_dtype"] = compute_dtype;
         phi::RecordOpInfoSupplement("block_multihead_attention_", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::BlockMultiheadAttentionInferMeta(MakeMetaTensor(*input_qkv), MakeMetaTensor(*input_key_cache), MakeMetaTensor(*input_value_cache), MakeMetaTensor(*input_seq_lens_encoder), MakeMetaTensor(*input_seq_lens_decoder), MakeMetaTensor(*input_seq_lens_this_time), MakeMetaTensor(*input_padding_offsets), MakeMetaTensor(*input_cum_offsets), MakeMetaTensor(*input_cu_seqlens_q), MakeMetaTensor(*input_cu_seqlens_k), MakeMetaTensor(*input_block_tables), MakeMetaTensor(input_pre_key_cache), MakeMetaTensor(input_pre_value_cache), MakeMetaTensor(input_rope_emb), MakeMetaTensor(input_mask), MakeMetaTensor(input_tgt_mask), MakeMetaTensor(input_cache_k_quant_scales), MakeMetaTensor(input_cache_v_quant_scales), MakeMetaTensor(input_cache_k_dequant_scales), MakeMetaTensor(input_cache_v_dequant_scales), MakeMetaTensor(input_qkv_out_scale), MakeMetaTensor(input_qkv_bias), MakeMetaTensor(input_out_shift), MakeMetaTensor(input_out_smooth), max_seq_len, block_size, use_neox_style, dynamic_cachekv_quant, quant_round_type, quant_max_bound, quant_min_bound, out_scale, compute_dtype, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("block_multihead_attention_ dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, int, bool, bool, int, float, float, float, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_qkv, *input_key_cache, *input_value_cache, *input_seq_lens_encoder, *input_seq_lens_decoder, *input_seq_lens_this_time, *input_padding_offsets, *input_cum_offsets, *input_cu_seqlens_q, *input_cu_seqlens_k, *input_block_tables, input_pre_key_cache, input_pre_value_cache, input_rope_emb, input_mask, input_tgt_mask, input_cache_k_quant_scales, input_cache_v_quant_scales, input_cache_k_dequant_scales, input_cache_v_dequant_scales, input_qkv_out_scale, input_qkv_bias, input_out_shift, input_out_smooth, max_seq_len, block_size, use_neox_style, dynamic_cachekv_quant, quant_round_type, quant_max_bound, quant_min_bound, out_scale, compute_dtype, dense_out_0, dense_out_1, dense_out_2, dense_out_3);
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
    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_1 = std::get<1>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_1, dist_out_attr_1, true);

    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_2 = std::get<2>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_2, dist_out_attr_2, true);

    // Set correct dist_attr for inplace output:
    // If no_spmd_rules, reshard it to origin dist_attr,
    // Or set correct spmd output dist_attr
    auto& output_3 = std::get<3>(api_output);
    SetInplaceOutputCorrectDistAttr(dev_ctx, output_3, dist_out_attr_3, true);


    // 12. Return
    return api_output;
  }

  VLOG(6) << "block_multihead_attention_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "block_multihead_attention", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("block_multihead_attention_", kernel_data_type);
  }
  VLOG(6) << "block_multihead_attention kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_qkv = PrepareData(qkv, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_key_cache = PrepareData(key_cache, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_value_cache = PrepareData(value_cache, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seq_lens_encoder = PrepareData(seq_lens_encoder, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seq_lens_decoder = PrepareData(seq_lens_decoder, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seq_lens_this_time = PrepareData(seq_lens_this_time, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_padding_offsets = PrepareData(padding_offsets, GetKernelInputArgDef(kernel.InputAt(6), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cum_offsets = PrepareData(cum_offsets, GetKernelInputArgDef(kernel.InputAt(7), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlens_q = PrepareData(cu_seqlens_q, GetKernelInputArgDef(kernel.InputAt(8), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cu_seqlens_k = PrepareData(cu_seqlens_k, GetKernelInputArgDef(kernel.InputAt(9), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_block_tables = PrepareData(block_tables, GetKernelInputArgDef(kernel.InputAt(10), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_pre_key_cache = PrepareData(pre_key_cache, GetKernelInputArgDef(kernel.InputAt(11), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_pre_value_cache = PrepareData(pre_value_cache, GetKernelInputArgDef(kernel.InputAt(12), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_rope_emb = PrepareData(rope_emb, GetKernelInputArgDef(kernel.InputAt(13), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_mask = PrepareData(mask, GetKernelInputArgDef(kernel.InputAt(14), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_tgt_mask = PrepareData(tgt_mask, GetKernelInputArgDef(kernel.InputAt(15), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cache_k_quant_scales = PrepareData(cache_k_quant_scales, GetKernelInputArgDef(kernel.InputAt(16), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cache_v_quant_scales = PrepareData(cache_v_quant_scales, GetKernelInputArgDef(kernel.InputAt(17), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cache_k_dequant_scales = PrepareData(cache_k_dequant_scales, GetKernelInputArgDef(kernel.InputAt(18), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cache_v_dequant_scales = PrepareData(cache_v_dequant_scales, GetKernelInputArgDef(kernel.InputAt(19), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_qkv_out_scale = PrepareData(qkv_out_scale, GetKernelInputArgDef(kernel.InputAt(20), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_qkv_bias = PrepareData(qkv_bias, GetKernelInputArgDef(kernel.InputAt(21), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_shift = PrepareData(out_shift, GetKernelInputArgDef(kernel.InputAt(22), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_out_smooth = PrepareData(out_smooth, GetKernelInputArgDef(kernel.InputAt(23), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> pre_key_cache_record_shapes;
     if(input_pre_key_cache){
       pre_key_cache_record_shapes.push_back((*input_pre_key_cache).dims());
     }
     std::vector<phi::DDim> pre_value_cache_record_shapes;
     if(input_pre_value_cache){
       pre_value_cache_record_shapes.push_back((*input_pre_value_cache).dims());
     }
     std::vector<phi::DDim> rope_emb_record_shapes;
     if(input_rope_emb){
       rope_emb_record_shapes.push_back((*input_rope_emb).dims());
     }
     std::vector<phi::DDim> mask_record_shapes;
     if(input_mask){
       mask_record_shapes.push_back((*input_mask).dims());
     }
     std::vector<phi::DDim> tgt_mask_record_shapes;
     if(input_tgt_mask){
       tgt_mask_record_shapes.push_back((*input_tgt_mask).dims());
     }
     std::vector<phi::DDim> cache_k_quant_scales_record_shapes;
     if(input_cache_k_quant_scales){
       cache_k_quant_scales_record_shapes.push_back((*input_cache_k_quant_scales).dims());
     }
     std::vector<phi::DDim> cache_v_quant_scales_record_shapes;
     if(input_cache_v_quant_scales){
       cache_v_quant_scales_record_shapes.push_back((*input_cache_v_quant_scales).dims());
     }
     std::vector<phi::DDim> cache_k_dequant_scales_record_shapes;
     if(input_cache_k_dequant_scales){
       cache_k_dequant_scales_record_shapes.push_back((*input_cache_k_dequant_scales).dims());
     }
     std::vector<phi::DDim> cache_v_dequant_scales_record_shapes;
     if(input_cache_v_dequant_scales){
       cache_v_dequant_scales_record_shapes.push_back((*input_cache_v_dequant_scales).dims());
     }
     std::vector<phi::DDim> qkv_out_scale_record_shapes;
     if(input_qkv_out_scale){
       qkv_out_scale_record_shapes.push_back((*input_qkv_out_scale).dims());
     }
     std::vector<phi::DDim> qkv_bias_record_shapes;
     if(input_qkv_bias){
       qkv_bias_record_shapes.push_back((*input_qkv_bias).dims());
     }
     std::vector<phi::DDim> out_shift_record_shapes;
     if(input_out_shift){
       out_shift_record_shapes.push_back((*input_out_shift).dims());
     }
     std::vector<phi::DDim> out_smooth_record_shapes;
     if(input_out_smooth){
       out_smooth_record_shapes.push_back((*input_out_smooth).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"qkv", {
     (*input_qkv).dims()}},
     {"key_cache", {
     (*input_key_cache).dims()}},
     {"value_cache", {
     (*input_value_cache).dims()}},
     {"seq_lens_encoder", {
     (*input_seq_lens_encoder).dims()}},
     {"seq_lens_decoder", {
     (*input_seq_lens_decoder).dims()}},
     {"seq_lens_this_time", {
     (*input_seq_lens_this_time).dims()}},
     {"padding_offsets", {
     (*input_padding_offsets).dims()}},
     {"cum_offsets", {
     (*input_cum_offsets).dims()}},
     {"cu_seqlens_q", {
     (*input_cu_seqlens_q).dims()}},
     {"cu_seqlens_k", {
     (*input_cu_seqlens_k).dims()}},
     {"block_tables", {
     (*input_block_tables).dims()}},
     {"pre_key_cache", pre_key_cache_record_shapes},
     {"pre_value_cache", pre_value_cache_record_shapes},
     {"rope_emb", rope_emb_record_shapes},
     {"mask", mask_record_shapes},
     {"tgt_mask", tgt_mask_record_shapes},
     {"cache_k_quant_scales", cache_k_quant_scales_record_shapes},
     {"cache_v_quant_scales", cache_v_quant_scales_record_shapes},
     {"cache_k_dequant_scales", cache_k_dequant_scales_record_shapes},
     {"cache_v_dequant_scales", cache_v_dequant_scales_record_shapes},
     {"qkv_out_scale", qkv_out_scale_record_shapes},
     {"qkv_bias", qkv_bias_record_shapes},
     {"out_shift", out_shift_record_shapes},
     {"out_smooth",
     out_smooth_record_shapes}};
     phi::AttributeMap attrs;
     attrs["max_seq_len"] = max_seq_len;
     attrs["block_size"] = block_size;
     attrs["use_neox_style"] = use_neox_style;
     attrs["dynamic_cachekv_quant"] = dynamic_cachekv_quant;
     attrs["quant_round_type"] = quant_round_type;
     attrs["quant_max_bound"] = quant_max_bound;
     attrs["quant_min_bound"] = quant_min_bound;
     attrs["out_scale"] = out_scale;
     attrs["compute_dtype"] = compute_dtype;
     phi::RecordOpInfoSupplement("block_multihead_attention_", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor&, Tensor&, Tensor&> api_output{Tensor(), qkv, key_cache, value_cache};
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));
  auto backup0 = ProcessStrideBackup(&kernel_out_0);
  auto backup1 = ProcessStrideBackup(&kernel_out_1);
  auto backup2 = ProcessStrideBackup(&kernel_out_2);
  auto backup3 = ProcessStrideBackup(&kernel_out_3);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("block_multihead_attention_ infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);

  phi::BlockMultiheadAttentionInferMeta(MakeMetaTensor(*input_qkv), MakeMetaTensor(*input_key_cache), MakeMetaTensor(*input_value_cache), MakeMetaTensor(*input_seq_lens_encoder), MakeMetaTensor(*input_seq_lens_decoder), MakeMetaTensor(*input_seq_lens_this_time), MakeMetaTensor(*input_padding_offsets), MakeMetaTensor(*input_cum_offsets), MakeMetaTensor(*input_cu_seqlens_q), MakeMetaTensor(*input_cu_seqlens_k), MakeMetaTensor(*input_block_tables), MakeMetaTensor(input_pre_key_cache), MakeMetaTensor(input_pre_value_cache), MakeMetaTensor(input_rope_emb), MakeMetaTensor(input_mask), MakeMetaTensor(input_tgt_mask), MakeMetaTensor(input_cache_k_quant_scales), MakeMetaTensor(input_cache_v_quant_scales), MakeMetaTensor(input_cache_k_dequant_scales), MakeMetaTensor(input_cache_v_dequant_scales), MakeMetaTensor(input_qkv_out_scale), MakeMetaTensor(input_qkv_bias), MakeMetaTensor(input_out_shift), MakeMetaTensor(input_out_smooth), max_seq_len, block_size, use_neox_style, dynamic_cachekv_quant, quant_round_type, quant_max_bound, quant_min_bound, out_scale, compute_dtype, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, int, bool, bool, int, float, float, float, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("block_multihead_attention_ compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_qkv, *input_key_cache, *input_value_cache, *input_seq_lens_encoder, *input_seq_lens_decoder, *input_seq_lens_this_time, *input_padding_offsets, *input_cum_offsets, *input_cu_seqlens_q, *input_cu_seqlens_k, *input_block_tables, input_pre_key_cache, input_pre_value_cache, input_rope_emb, input_mask, input_tgt_mask, input_cache_k_quant_scales, input_cache_v_quant_scales, input_cache_k_dequant_scales, input_cache_v_dequant_scales, input_qkv_out_scale, input_qkv_bias, input_out_shift, input_out_smooth, max_seq_len, block_size, use_neox_style, dynamic_cachekv_quant, quant_round_type, quant_max_bound, quant_min_bound, out_scale, compute_dtype, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }
  TransStride(dev_ctx, kernel_out_0, backup0);
  TransStride(dev_ctx, kernel_out_1, backup1);
  TransStride(dev_ctx, kernel_out_2, backup2);
  TransStride(dev_ctx, kernel_out_3, backup3);

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);

  }
  return api_output;
}

PADDLE_API Tensor fused_bias_act(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& dequant_scales, const paddle::optional<Tensor>& shift, const paddle::optional<Tensor>& smooth, const std::string& act_method, const std::string& compute_dtype, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, bias, dequant_scales, shift, smooth);
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
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, bias, dequant_scales, shift, smooth);
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
    auto meta_dist_input_dequant_scales = dequant_scales ? MakeDistMetaTensor(*(*dequant_scales).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_shift = shift ? MakeDistMetaTensor(*(*shift).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_smooth = smooth ? MakeDistMetaTensor(*(*smooth).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_bias, meta_dist_input_dequant_scales, meta_dist_input_shift, meta_dist_input_smooth);
    DebugInfoForInferSpmd("fused_bias_act", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    Tensor api_output;

    auto dist_out = SetKernelDistOutput(&api_output);
    auto dense_out = dist_out->unsafe_mutable_value();
    if (!rank_is_in_current_mesh) {{
      *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out(dist_out);
    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_dequant_scales = dequant_scales ? MakeMetaTensor(*(*dequant_scales).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_shift = shift ? MakeMetaTensor(*(*shift).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_smooth = smooth ? MakeMetaTensor(*(*smooth).impl()) : phi::MetaTensor();

    phi::FusedBiasActInferMeta(MakeMetaTensor(*x.impl()), meta_dist_bias, meta_dist_dequant_scales, meta_dist_shift, meta_dist_smooth, act_method, compute_dtype, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, &meta_dist_out);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_bias_act API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_bias_act", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_bias_act kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[1], "bias");
      auto dist_input_dequant_scales = ReshardApiInputToKernelInput(dev_ctx, dequant_scales, spmd_info.first[2], "dequant_scales");
      auto dist_input_shift = ReshardApiInputToKernelInput(dev_ctx, shift, spmd_info.first[3], "shift");
      auto dist_input_smooth = ReshardApiInputToKernelInput(dev_ctx, smooth, spmd_info.first[4], "smooth");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_bias = PrepareDataForDistTensor(dist_input_bias, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_bias = dist_input_bias ? paddle::make_optional<phi::DenseTensor>((*dist_input_bias)->value()) : paddle::none;

      dist_input_dequant_scales = PrepareDataForDistTensor(dist_input_dequant_scales, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_dequant_scales = dist_input_dequant_scales ? paddle::make_optional<phi::DenseTensor>((*dist_input_dequant_scales)->value()) : paddle::none;

      dist_input_shift = PrepareDataForDistTensor(dist_input_shift, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_shift = dist_input_shift ? paddle::make_optional<phi::DenseTensor>((*dist_input_shift)->value()) : paddle::none;

      dist_input_smooth = PrepareDataForDistTensor(dist_input_smooth, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_smooth = dist_input_smooth ? paddle::make_optional<phi::DenseTensor>((*dist_input_smooth)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> bias_record_shapes;
         if(input_bias){
           bias_record_shapes.push_back((*input_bias).dims());
         }
         std::vector<phi::DDim> dequant_scales_record_shapes;
         if(input_dequant_scales){
           dequant_scales_record_shapes.push_back((*input_dequant_scales).dims());
         }
         std::vector<phi::DDim> shift_record_shapes;
         if(input_shift){
           shift_record_shapes.push_back((*input_shift).dims());
         }
         std::vector<phi::DDim> smooth_record_shapes;
         if(input_smooth){
           smooth_record_shapes.push_back((*input_smooth).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"bias", bias_record_shapes},
         {"dequant_scales", dequant_scales_record_shapes},
         {"shift", shift_record_shapes},
         {"smooth",
         smooth_record_shapes}};
         phi::AttributeMap attrs;
         attrs["act_method"] = act_method;
         attrs["compute_dtype"] = compute_dtype;
         attrs["quant_scale"] = quant_scale;
         attrs["quant_round_type"] = quant_round_type;
         attrs["quant_max_bound"] = quant_max_bound;
         attrs["quant_min_bound"] = quant_min_bound;
         phi::RecordOpInfoSupplement("fused_bias_act", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out(dense_out);
      phi::FusedBiasActInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_dequant_scales), MakeMetaTensor(input_shift), MakeMetaTensor(input_smooth), act_method, compute_dtype, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, &meta_dense_out);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_bias_act dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const std::string&, const std::string&, float, int, float, float, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_dequant_scales, input_shift, input_smooth, act_method, compute_dtype, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dense_out);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out, kernel_backend, dense_out);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "fused_bias_act API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_bias_act", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_bias_act", kernel_data_type);
  }
  VLOG(6) << "fused_bias_act kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dequant_scales = PrepareData(dequant_scales, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_shift = PrepareData(shift, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_smooth = PrepareData(smooth, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> bias_record_shapes;
     if(input_bias){
       bias_record_shapes.push_back((*input_bias).dims());
     }
     std::vector<phi::DDim> dequant_scales_record_shapes;
     if(input_dequant_scales){
       dequant_scales_record_shapes.push_back((*input_dequant_scales).dims());
     }
     std::vector<phi::DDim> shift_record_shapes;
     if(input_shift){
       shift_record_shapes.push_back((*input_shift).dims());
     }
     std::vector<phi::DDim> smooth_record_shapes;
     if(input_smooth){
       smooth_record_shapes.push_back((*input_smooth).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"bias", bias_record_shapes},
     {"dequant_scales", dequant_scales_record_shapes},
     {"shift", shift_record_shapes},
     {"smooth",
     smooth_record_shapes}};
     phi::AttributeMap attrs;
     attrs["act_method"] = act_method;
     attrs["compute_dtype"] = compute_dtype;
     attrs["quant_scale"] = quant_scale;
     attrs["quant_round_type"] = quant_round_type;
     attrs["quant_max_bound"] = quant_max_bound;
     attrs["quant_min_bound"] = quant_min_bound;
     phi::RecordOpInfoSupplement("fused_bias_act", input_shapes, attrs);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(&api_output);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_bias_act infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::FusedBiasActInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_dequant_scales), MakeMetaTensor(input_shift), MakeMetaTensor(input_smooth), act_method, compute_dtype, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, &meta_out);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const std::string&, const std::string&, float, int, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_bias_act compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_dequant_scales, input_shift, input_smooth, act_method, compute_dtype, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, kernel_out);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);

  }
  return api_output;
}

PADDLE_API Tensor fused_bias_dropout_residual_layer_norm(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, residual, bias, ln_scale, ln_bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(residual.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, residual, bias, ln_scale, ln_bias);
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
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_residual, meta_dist_input_bias, meta_dist_input_ln_scale, meta_dist_input_ln_bias);
    DebugInfoForInferSpmd("fused_bias_dropout_residual_layer_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> api_output;

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

    auto dist_out_4 = SetKernelDistOutput(&std::get<4>(api_output));
    auto dense_out_4 = dist_out_4 ? dist_out_4->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
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

    phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*residual.impl()), meta_dist_bias, meta_dist_ln_scale, meta_dist_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr, dist_out_4 ? &meta_dist_out_4 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_bias_dropout_residual_layer_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_bias_dropout_residual_layer_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_bias_dropout_residual_layer_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_residual = ReshardApiInputToKernelInput(dev_ctx, residual, spmd_info.first[1], "residual");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");
      auto dist_input_ln_scale = ReshardApiInputToKernelInput(dev_ctx, ln_scale, spmd_info.first[3], "ln_scale");
      auto dist_input_ln_bias = ReshardApiInputToKernelInput(dev_ctx, ln_bias, spmd_info.first[4], "ln_bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
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

      // 7. RecordOpInfoSupplement
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
         {"ln_bias",
         ln_bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["dropout_rate"] = dropout_rate;
         attrs["is_test"] = is_test;
         attrs["dropout_fix_seed"] = dropout_fix_seed;
         attrs["dropout_seed"] = dropout_seed;
         attrs["dropout_implementation"] = dropout_implementation;
         attrs["ln_epsilon"] = ln_epsilon;
         phi::RecordOpInfoSupplement("fused_bias_dropout_residual_layer_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::MetaTensor meta_dense_out_4(dense_out_4);
      phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr, dense_out_4 ? &meta_dense_out_4 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0, dense_out_1, dense_out_2, dense_out_3, dense_out_4);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
        TransDataBackend(dense_out_3, kernel_backend, dense_out_3);
        TransDataBackend(dense_out_4, kernel_backend, dense_out_4);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_3, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_4, current_process_mesh);

    // 12. Return
    return std::get<0>(api_output);
  }

  VLOG(6) << "fused_bias_dropout_residual_layer_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_bias_dropout_residual_layer_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_bias_dropout_residual_layer_norm", kernel_data_type);
  }
  VLOG(6) << "fused_bias_dropout_residual_layer_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_residual = PrepareData(residual, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_scale = PrepareData(ln_scale, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_bias = PrepareData(ln_bias, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
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
     {"ln_bias",
     ln_bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["dropout_rate"] = dropout_rate;
     attrs["is_test"] = is_test;
     attrs["dropout_fix_seed"] = dropout_fix_seed;
     attrs["dropout_seed"] = dropout_seed;
     attrs["dropout_implementation"] = dropout_implementation;
     attrs["ln_epsilon"] = ln_epsilon;
     phi::RecordOpInfoSupplement("fused_bias_dropout_residual_layer_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));
  auto kernel_out_4 = SetKernelOutput(&std::get<4>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);

  phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4);
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
  return std::get<0>(api_output);
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fused_bias_dropout_residual_layer_norm_intermediate(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, residual, bias, ln_scale, ln_bias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(residual.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, residual, bias, ln_scale, ln_bias);
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
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_residual, meta_dist_input_bias, meta_dist_input_ln_scale, meta_dist_input_ln_bias);
    DebugInfoForInferSpmd("fused_bias_dropout_residual_layer_norm", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> api_output;

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

    auto dist_out_4 = SetKernelDistOutput(&std::get<4>(api_output));
    auto dense_out_4 = dist_out_4 ? dist_out_4->unsafe_mutable_value() : nullptr;
    if (!rank_is_in_current_mesh) {
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

    phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*residual.impl()), meta_dist_bias, meta_dist_ln_scale, meta_dist_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr, dist_out_4 ? &meta_dist_out_4 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_bias_dropout_residual_layer_norm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_bias_dropout_residual_layer_norm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_bias_dropout_residual_layer_norm kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_residual = ReshardApiInputToKernelInput(dev_ctx, residual, spmd_info.first[1], "residual");
      auto dist_input_bias = ReshardApiInputToKernelInput(dev_ctx, bias, spmd_info.first[2], "bias");
      auto dist_input_ln_scale = ReshardApiInputToKernelInput(dev_ctx, ln_scale, spmd_info.first[3], "ln_scale");
      auto dist_input_ln_bias = ReshardApiInputToKernelInput(dev_ctx, ln_bias, spmd_info.first[4], "ln_bias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
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

      // 7. RecordOpInfoSupplement
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
         {"ln_bias",
         ln_bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["dropout_rate"] = dropout_rate;
         attrs["is_test"] = is_test;
         attrs["dropout_fix_seed"] = dropout_fix_seed;
         attrs["dropout_seed"] = dropout_seed;
         attrs["dropout_implementation"] = dropout_implementation;
         attrs["ln_epsilon"] = ln_epsilon;
         phi::RecordOpInfoSupplement("fused_bias_dropout_residual_layer_norm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::MetaTensor meta_dense_out_4(dense_out_4);
      phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr, dense_out_4 ? &meta_dense_out_4 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, dense_out_0, dense_out_1, dense_out_2, dense_out_3, dense_out_4);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out_0, kernel_backend, dense_out_0);
        TransDataBackend(dense_out_1, kernel_backend, dense_out_1);
        TransDataBackend(dense_out_2, kernel_backend, dense_out_2);
        TransDataBackend(dense_out_3, kernel_backend, dense_out_3);
        TransDataBackend(dense_out_4, kernel_backend, dense_out_4);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out_0, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_1, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_2, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_3, current_process_mesh);
    SetReplicatedDistAttrForOutput(dist_out_4, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "fused_bias_dropout_residual_layer_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_bias_dropout_residual_layer_norm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_bias_dropout_residual_layer_norm", kernel_data_type);
  }
  VLOG(6) << "fused_bias_dropout_residual_layer_norm kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_residual = PrepareData(residual, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_bias = PrepareData(bias, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_scale = PrepareData(ln_scale, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_ln_bias = PrepareData(ln_bias, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
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
     {"ln_bias",
     ln_bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["dropout_rate"] = dropout_rate;
     attrs["is_test"] = is_test;
     attrs["dropout_fix_seed"] = dropout_fix_seed;
     attrs["dropout_seed"] = dropout_seed;
     attrs["dropout_implementation"] = dropout_implementation;
     attrs["ln_epsilon"] = ln_epsilon;
     phi::RecordOpInfoSupplement("fused_bias_dropout_residual_layer_norm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));
  auto kernel_out_4 = SetKernelOutput(&std::get<4>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_4(kernel_out_4, kernel_result.is_stride_kernel);

  phi::FusedBiasDropoutResidualLnInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_residual), MakeMetaTensor(input_bias), MakeMetaTensor(input_ln_scale), MakeMetaTensor(input_ln_bias), dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, bool, bool, int, const std::string&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_bias_dropout_residual_layer_norm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_residual, input_bias, input_ln_scale, input_ln_bias, dropout_rate, is_test, dropout_fix_seed, dropout_seed, dropout_implementation, ln_epsilon, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4);
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
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> fused_bias_residual_layernorm(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const paddle::optional<Tensor>& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, float residual_alpha, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, bias, residual, norm_weight, norm_bias);
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
    auto meta_dist_input_norm_weight = norm_weight ? MakeDistMetaTensor(*(*norm_weight).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_norm_bias = norm_bias ? MakeDistMetaTensor(*(*norm_bias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_bias, meta_dist_input_residual, meta_dist_input_norm_weight, meta_dist_input_norm_bias);
    DebugInfoForInferSpmd("fused_bias_residual_layernorm", spmd_info);

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
    phi::MetaTensor meta_dist_bias = bias ? MakeMetaTensor(*(*bias).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_residual = residual ? MakeMetaTensor(*(*residual).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_norm_weight = norm_weight ? MakeMetaTensor(*(*norm_weight).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_norm_bias = norm_bias ? MakeMetaTensor(*(*norm_bias).impl()) : phi::MetaTensor();

    phi::FusedLayerNormInferMeta(MakeMetaTensor(*x.impl()), meta_dist_bias, meta_dist_residual, meta_dist_norm_weight, meta_dist_norm_bias, epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr, dist_out_3 ? &meta_dist_out_3 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_bias_residual_layernorm API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_bias_residual_layernorm", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_bias_residual_layernorm kernel: " << kernel;
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
      paddle::optional<phi::DenseTensor> input_norm_weight = dist_input_norm_weight ? paddle::make_optional<phi::DenseTensor>((*dist_input_norm_weight)->value()) : paddle::none;

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
         std::vector<phi::DDim> norm_weight_record_shapes;
         if(input_norm_weight){
           norm_weight_record_shapes.push_back((*input_norm_weight).dims());
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
         {"norm_weight", norm_weight_record_shapes},
         {"norm_bias",
         norm_bias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["epsilon"] = epsilon;
         attrs["residual_alpha"] = residual_alpha;
         attrs["begin_norm_axis"] = begin_norm_axis;
         attrs["quant_scale"] = quant_scale;
         attrs["quant_round_type"] = quant_round_type;
         attrs["quant_max_bound"] = quant_max_bound;
         attrs["quant_min_bound"] = quant_min_bound;
         phi::RecordOpInfoSupplement("fused_bias_residual_layernorm", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::MetaTensor meta_dense_out_3(dense_out_3);
      phi::FusedLayerNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_residual), MakeMetaTensor(input_norm_weight), MakeMetaTensor(input_norm_bias), epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr, dense_out_3 ? &meta_dense_out_3 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_bias_residual_layernorm dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, float, int, float, int, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_residual, input_norm_weight, input_norm_bias, epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, dense_out_0, dense_out_1, dense_out_2, dense_out_3);
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

  VLOG(6) << "fused_bias_residual_layernorm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_bias_residual_layernorm", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_bias_residual_layernorm", kernel_data_type);
  }
  VLOG(6) << "fused_bias_residual_layernorm kernel: " << kernel;
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
     std::vector<phi::DDim> norm_weight_record_shapes;
     if(input_norm_weight){
       norm_weight_record_shapes.push_back((*input_norm_weight).dims());
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
     {"norm_weight", norm_weight_record_shapes},
     {"norm_bias",
     norm_bias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["epsilon"] = epsilon;
     attrs["residual_alpha"] = residual_alpha;
     attrs["begin_norm_axis"] = begin_norm_axis;
     attrs["quant_scale"] = quant_scale;
     attrs["quant_round_type"] = quant_round_type;
     attrs["quant_max_bound"] = quant_max_bound;
     attrs["quant_min_bound"] = quant_min_bound;
     phi::RecordOpInfoSupplement("fused_bias_residual_layernorm", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(&std::get<3>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_bias_residual_layernorm infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_3(kernel_out_3, kernel_result.is_stride_kernel);

  phi::FusedLayerNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_bias), MakeMetaTensor(input_residual), MakeMetaTensor(input_norm_weight), MakeMetaTensor(input_norm_bias), epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, float, int, float, int, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_bias_residual_layernorm compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, input_bias, input_residual, input_norm_weight, input_norm_bias, epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
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

PADDLE_API std::tuple<Tensor, Tensor, Tensor> fused_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask, float scaling_factor, float dropout_probability, bool is_training, bool is_causal_masking) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(q, k, v, mask);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(mask.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(q);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(q, k, v, mask);
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
    auto meta_dist_input_mask = MakeDistMetaTensor(*mask.impl());
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_q, meta_dist_input_k, meta_dist_input_v, meta_dist_input_mask);
    DebugInfoForInferSpmd("fused_dot_product_attention", spmd_info);

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
    phi::FusedDotProductAttentionInferMeta(MakeMetaTensor(*q.impl()), MakeMetaTensor(*k.impl()), MakeMetaTensor(*v.impl()), dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_dot_product_attention API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_dot_product_attention", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_dot_product_attention kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_q = ReshardApiInputToKernelInput(dev_ctx, q, spmd_info.first[0], "q");
      auto dist_input_k = ReshardApiInputToKernelInput(dev_ctx, k, spmd_info.first[1], "k");
      auto dist_input_v = ReshardApiInputToKernelInput(dev_ctx, v, spmd_info.first[2], "v");
      auto dist_input_mask = ReshardApiInputToKernelInput(dev_ctx, mask, spmd_info.first[3], "mask");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_q = PrepareDataForDistTensor(dist_input_q, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_q = &dist_input_q->value();

      dist_input_k = PrepareDataForDistTensor(dist_input_k, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_k = &dist_input_k->value();

      dist_input_v = PrepareDataForDistTensor(dist_input_v, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_v = &dist_input_v->value();

      dist_input_mask = PrepareDataForDistTensor(dist_input_mask, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_mask = &dist_input_mask->value();

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"q", {
         (*input_q).dims()}},
         {"k", {
         (*input_k).dims()}},
         {"v", {
         (*input_v).dims()}},
         {"mask", {
         (*input_mask).dims()}}};
         phi::AttributeMap attrs;
         attrs["scaling_factor"] = scaling_factor;
         attrs["dropout_probability"] = dropout_probability;
         attrs["is_training"] = is_training;
         attrs["is_causal_masking"] = is_causal_masking;
         phi::RecordOpInfoSupplement("fused_dot_product_attention", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::FusedDotProductAttentionInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_dot_product_attention dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_mask, scaling_factor, dropout_probability, is_training, is_causal_masking, dense_out_0, dense_out_1, dense_out_2);
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

  VLOG(6) << "fused_dot_product_attention API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_dot_product_attention", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_dot_product_attention", kernel_data_type);
  }
  VLOG(6) << "fused_dot_product_attention kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_q = PrepareData(q, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_k = PrepareData(k, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_v = PrepareData(v, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_mask = PrepareData(mask, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"q", {
     (*input_q).dims()}},
     {"k", {
     (*input_k).dims()}},
     {"v", {
     (*input_v).dims()}},
     {"mask", {
     (*input_mask).dims()}}};
     phi::AttributeMap attrs;
     attrs["scaling_factor"] = scaling_factor;
     attrs["dropout_probability"] = dropout_probability;
     attrs["is_training"] = is_training;
     attrs["is_causal_masking"] = is_causal_masking;
     phi::RecordOpInfoSupplement("fused_dot_product_attention", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_dot_product_attention infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::FusedDotProductAttentionInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(*input_k), MakeMetaTensor(*input_v), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_dot_product_attention compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_q, *input_k, *input_v, *input_mask, scaling_factor, dropout_probability, is_training, is_causal_masking, kernel_out_0, kernel_out_1, kernel_out_2);
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

PADDLE_API std::tuple<Tensor, Tensor> fused_dropout_add(const Tensor& x, const Tensor& y, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, y, seed_tensor);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(y.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(x);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, seed_tensor);
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
    auto meta_dist_input_seed_tensor = seed_tensor ? MakeDistMetaTensor(*(*seed_tensor).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_y, meta_dist_input_seed_tensor);
    DebugInfoForInferSpmd("fused_dropout_add", spmd_info);

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
    phi::FusedDropoutAddInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*y.impl()), dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_dropout_add API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_dropout_add", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_dropout_add kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_y = ReshardApiInputToKernelInput(dev_ctx, y, spmd_info.first[1], "y");
      auto dist_input_seed_tensor = ReshardApiInputToKernelInput(dev_ctx, seed_tensor, spmd_info.first[2], "seed_tensor");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_y = PrepareDataForDistTensor(dist_input_y, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_y = &dist_input_y->value();

      dist_input_seed_tensor = PrepareDataForDistTensor(dist_input_seed_tensor, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
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
         {"y", {
         (*input_y).dims()}},
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
         phi::RecordOpInfoSupplement("fused_dropout_add", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::FusedDropoutAddInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_dropout_add dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, bool, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_y, input_seed_tensor, phi::Scalar(p), is_test, mode, seed, fix_seed, dense_out_0, dense_out_1);
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

  VLOG(6) << "fused_dropout_add API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_dropout_add", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_dropout_add", kernel_data_type);
  }
  VLOG(6) << "fused_dropout_add kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_y = PrepareData(y, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seed_tensor = PrepareData(seed_tensor, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> seed_tensor_record_shapes;
     if(input_seed_tensor){
       seed_tensor_record_shapes.push_back((*input_seed_tensor).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"y", {
     (*input_y).dims()}},
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
     phi::RecordOpInfoSupplement("fused_dropout_add", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_dropout_add infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FusedDropoutAddInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, bool, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_dropout_add compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, input_seed_tensor, phi::Scalar(p), is_test, mode, seed, fix_seed, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> fused_linear_param_grad_add(const Tensor& x, const Tensor& dout, const paddle::optional<Tensor>& dweight, const paddle::optional<Tensor>& dbias, bool multi_precision, bool has_bias) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(x, dout, dweight, dbias);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(dout.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(dout);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(x, dout, dweight, dbias);
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
    auto meta_dist_input_dout = MakeDistMetaTensor(*dout.impl());
    auto meta_dist_input_dweight = dweight ? MakeDistMetaTensor(*(*dweight).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_dbias = dbias ? MakeDistMetaTensor(*(*dbias).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_x, meta_dist_input_dout, meta_dist_input_dweight, meta_dist_input_dbias);
    DebugInfoForInferSpmd("fused_linear_param_grad_add", spmd_info);

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
    phi::MetaTensor meta_dist_dweight = dweight ? MakeMetaTensor(*(*dweight).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_dbias = dbias ? MakeMetaTensor(*(*dbias).impl()) : phi::MetaTensor();

    phi::FusedLinearParamGradAddInferMeta(MakeMetaTensor(*x.impl()), MakeMetaTensor(*dout.impl()), meta_dist_dweight, meta_dist_dbias, multi_precision, has_bias, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_linear_param_grad_add API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_linear_param_grad_add", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_linear_param_grad_add kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_x = ReshardApiInputToKernelInput(dev_ctx, x, spmd_info.first[0], "x");
      auto dist_input_dout = ReshardApiInputToKernelInput(dev_ctx, dout, spmd_info.first[1], "dout");
      auto dist_input_dweight = ReshardApiInputToKernelInput(dev_ctx, dweight, spmd_info.first[2], "dweight");
      auto dist_input_dbias = ReshardApiInputToKernelInput(dev_ctx, dbias, spmd_info.first[3], "dbias");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_x = PrepareDataForDistTensor(dist_input_x, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_x = &dist_input_x->value();

      dist_input_dout = PrepareDataForDistTensor(dist_input_dout, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_dout = &dist_input_dout->value();

      dist_input_dweight = PrepareDataForDistTensor(dist_input_dweight, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_dweight = dist_input_dweight ? paddle::make_optional<phi::DenseTensor>((*dist_input_dweight)->value()) : paddle::none;

      dist_input_dbias = PrepareDataForDistTensor(dist_input_dbias, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_dbias = dist_input_dbias ? paddle::make_optional<phi::DenseTensor>((*dist_input_dbias)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> dweight_record_shapes;
         if(input_dweight){
           dweight_record_shapes.push_back((*input_dweight).dims());
         }
         std::vector<phi::DDim> dbias_record_shapes;
         if(input_dbias){
           dbias_record_shapes.push_back((*input_dbias).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"x", {
         (*input_x).dims()}},
         {"dout", {
         (*input_dout).dims()}},
         {"dweight", dweight_record_shapes},
         {"dbias",
         dbias_record_shapes}};
         phi::AttributeMap attrs;
         attrs["multi_precision"] = multi_precision;
         attrs["has_bias"] = has_bias;
         phi::RecordOpInfoSupplement("fused_linear_param_grad_add", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::FusedLinearParamGradAddInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_dout), MakeMetaTensor(input_dweight), MakeMetaTensor(input_dbias), multi_precision, has_bias, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_linear_param_grad_add dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_x, *input_dout, input_dweight, input_dbias, multi_precision, has_bias, dense_out_0, dense_out_1);
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

  VLOG(6) << "fused_linear_param_grad_add API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_linear_param_grad_add", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_linear_param_grad_add", kernel_data_type);
  }
  VLOG(6) << "fused_linear_param_grad_add kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_x = PrepareData(x, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dout = PrepareData(dout, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dweight = PrepareData(dweight, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_dbias = PrepareData(dbias, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> dweight_record_shapes;
     if(input_dweight){
       dweight_record_shapes.push_back((*input_dweight).dims());
     }
     std::vector<phi::DDim> dbias_record_shapes;
     if(input_dbias){
       dbias_record_shapes.push_back((*input_dbias).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"x", {
     (*input_x).dims()}},
     {"dout", {
     (*input_dout).dims()}},
     {"dweight", dweight_record_shapes},
     {"dbias",
     dbias_record_shapes}};
     phi::AttributeMap attrs;
     attrs["multi_precision"] = multi_precision;
     attrs["has_bias"] = has_bias;
     phi::RecordOpInfoSupplement("fused_linear_param_grad_add", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_linear_param_grad_add infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);

  phi::FusedLinearParamGradAddInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_dout), MakeMetaTensor(input_dweight), MakeMetaTensor(input_dbias), multi_precision, has_bias, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_linear_param_grad_add compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_x, *input_dout, input_dweight, input_dbias, multi_precision, has_bias, kernel_out_0, kernel_out_1);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);

  }
  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> fused_rotary_position_embedding(const Tensor& q, const paddle::optional<Tensor>& k, const paddle::optional<Tensor>& v, const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, bool use_neox_rotary_style, bool time_major, float rotary_emb_base) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(q, k, v, sin, cos, position_ids);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(q.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(q);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(q, k, v, sin, cos, position_ids);
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
    auto meta_dist_input_k = k ? MakeDistMetaTensor(*(*k).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_v = v ? MakeDistMetaTensor(*(*v).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_sin = sin ? MakeDistMetaTensor(*(*sin).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_cos = cos ? MakeDistMetaTensor(*(*cos).impl()) : phi::distributed::DistMetaTensor();
    auto meta_dist_input_position_ids = position_ids ? MakeDistMetaTensor(*(*position_ids).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::FusedRopeInferSpmd(meta_dist_input_q, meta_dist_input_k, meta_dist_input_v, meta_dist_input_sin, meta_dist_input_cos, meta_dist_input_position_ids, use_neox_rotary_style, time_major, rotary_emb_base);
    DebugInfoForInferSpmd("fused_rotary_position_embedding", spmd_info);

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
    phi::MetaTensor meta_dist_k = k ? MakeMetaTensor(*(*k).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_v = v ? MakeMetaTensor(*(*v).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_sin = sin ? MakeMetaTensor(*(*sin).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_cos = cos ? MakeMetaTensor(*(*cos).impl()) : phi::MetaTensor();

    phi::MetaTensor meta_dist_position_ids = position_ids ? MakeMetaTensor(*(*position_ids).impl()) : phi::MetaTensor();

    phi::FusedRopeInferMeta(MakeMetaTensor(*q.impl()), meta_dist_k, meta_dist_v, meta_dist_sin, meta_dist_cos, meta_dist_position_ids, use_neox_rotary_style, time_major, rotary_emb_base, dist_out_0 ? &meta_dist_out_0 : nullptr, dist_out_1 ? &meta_dist_out_1 : nullptr, dist_out_2 ? &meta_dist_out_2 : nullptr);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "fused_rotary_position_embedding API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "fused_rotary_position_embedding", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "fused_rotary_position_embedding kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_q = ReshardApiInputToKernelInput(dev_ctx, q, spmd_info.first[0], "q");
      auto dist_input_k = ReshardApiInputToKernelInput(dev_ctx, k, spmd_info.first[1], "k");
      auto dist_input_v = ReshardApiInputToKernelInput(dev_ctx, v, spmd_info.first[2], "v");
      auto dist_input_sin = ReshardApiInputToKernelInput(dev_ctx, sin, spmd_info.first[3], "sin");
      auto dist_input_cos = ReshardApiInputToKernelInput(dev_ctx, cos, spmd_info.first[4], "cos");
      auto dist_input_position_ids = ReshardApiInputToKernelInput(dev_ctx, position_ids, spmd_info.first[5], "position_ids");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_q = PrepareDataForDistTensor(dist_input_q, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_q = &dist_input_q->value();

      dist_input_k = PrepareDataForDistTensor(dist_input_k, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_k = dist_input_k ? paddle::make_optional<phi::DenseTensor>((*dist_input_k)->value()) : paddle::none;

      dist_input_v = PrepareDataForDistTensor(dist_input_v, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_v = dist_input_v ? paddle::make_optional<phi::DenseTensor>((*dist_input_v)->value()) : paddle::none;

      dist_input_sin = PrepareDataForDistTensor(dist_input_sin, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_sin = dist_input_sin ? paddle::make_optional<phi::DenseTensor>((*dist_input_sin)->value()) : paddle::none;

      dist_input_cos = PrepareDataForDistTensor(dist_input_cos, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_cos = dist_input_cos ? paddle::make_optional<phi::DenseTensor>((*dist_input_cos)->value()) : paddle::none;

      dist_input_position_ids = PrepareDataForDistTensor(dist_input_position_ids, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_position_ids = dist_input_position_ids ? paddle::make_optional<phi::DenseTensor>((*dist_input_position_ids)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> k_record_shapes;
         if(input_k){
           k_record_shapes.push_back((*input_k).dims());
         }
         std::vector<phi::DDim> v_record_shapes;
         if(input_v){
           v_record_shapes.push_back((*input_v).dims());
         }
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
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"q", {
         (*input_q).dims()}},
         {"k", k_record_shapes},
         {"v", v_record_shapes},
         {"sin", sin_record_shapes},
         {"cos", cos_record_shapes},
         {"position_ids",
         position_ids_record_shapes}};
         phi::AttributeMap attrs;
         attrs["use_neox_rotary_style"] = use_neox_rotary_style;
         attrs["time_major"] = time_major;
         attrs["rotary_emb_base"] = rotary_emb_base;
         phi::RecordOpInfoSupplement("fused_rotary_position_embedding", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out_0(dense_out_0);
      phi::MetaTensor meta_dense_out_1(dense_out_1);
      phi::MetaTensor meta_dense_out_2(dense_out_2);
      phi::FusedRopeInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(input_k), MakeMetaTensor(input_v), MakeMetaTensor(input_sin), MakeMetaTensor(input_cos), MakeMetaTensor(input_position_ids), use_neox_rotary_style, time_major, rotary_emb_base, dense_out_0 ? &meta_dense_out_0 : nullptr, dense_out_1 ? &meta_dense_out_1 : nullptr, dense_out_2 ? &meta_dense_out_2 : nullptr);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("fused_rotary_position_embedding dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_q, input_k, input_v, input_sin, input_cos, input_position_ids, use_neox_rotary_style, time_major, rotary_emb_base, dense_out_0, dense_out_1, dense_out_2);
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
    // API `fused_rotary_position_embedding` does not need to set DistAttr for output.

    // 12. Return
    return api_output;
  }

  VLOG(6) << "fused_rotary_position_embedding API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fused_rotary_position_embedding", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("fused_rotary_position_embedding", kernel_data_type);
  }
  VLOG(6) << "fused_rotary_position_embedding kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_q = PrepareData(q, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_k = PrepareData(k, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_v = PrepareData(v, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_sin = PrepareData(sin, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_cos = PrepareData(cos, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_position_ids = PrepareData(position_ids, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> k_record_shapes;
     if(input_k){
       k_record_shapes.push_back((*input_k).dims());
     }
     std::vector<phi::DDim> v_record_shapes;
     if(input_v){
       v_record_shapes.push_back((*input_v).dims());
     }
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
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"q", {
     (*input_q).dims()}},
     {"k", k_record_shapes},
     {"v", v_record_shapes},
     {"sin", sin_record_shapes},
     {"cos", cos_record_shapes},
     {"position_ids",
     position_ids_record_shapes}};
     phi::AttributeMap attrs;
     attrs["use_neox_rotary_style"] = use_neox_rotary_style;
     attrs["time_major"] = time_major;
     attrs["rotary_emb_base"] = rotary_emb_base;
     phi::RecordOpInfoSupplement("fused_rotary_position_embedding", input_shapes, attrs);
  }

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(&std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(&std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(&std::get<2>(api_output));

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("fused_rotary_position_embedding infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out_0(kernel_out_0, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_1(kernel_out_1, kernel_result.is_stride_kernel);
  phi::MetaTensor meta_out_2(kernel_out_2, kernel_result.is_stride_kernel);

  phi::FusedRopeInferMeta(MakeMetaTensor(*input_q), MakeMetaTensor(input_k), MakeMetaTensor(input_v), MakeMetaTensor(input_sin), MakeMetaTensor(input_cos), MakeMetaTensor(input_position_ids), use_neox_rotary_style, time_major, rotary_emb_base, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, bool, bool, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("fused_rotary_position_embedding compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_q, input_k, input_v, input_sin, input_cos, input_position_ids, use_neox_rotary_style, time_major, rotary_emb_base, kernel_out_0, kernel_out_1, kernel_out_2);
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

PADDLE_API Tensor variable_length_memory_efficient_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& seq_lens, const Tensor& kv_seq_lens, const paddle::optional<Tensor>& mask, float scale, bool causal, int pre_cache_length) {
  // Kernel Key Construction
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  bool run_auto_parallel = AllInputsAreDistTensor(query, key, value, seq_lens, kv_seq_lens, mask);
  bool rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh = std::static_pointer_cast<phi::distributed::DistTensor>(kv_seq_lens.impl())->dist_attr().process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);
  }
  if (rank_is_in_current_mesh) {
    kernel_data_type = ParseDataType(query);

    if (kernel_backend == Backend::UNDEFINED
          || kernel_layout == DataLayout::UNDEFINED
          || kernel_data_type == DataType::UNDEFINED ) {
      auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, seq_lens, kv_seq_lens, mask);
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
    auto meta_dist_input_query = MakeDistMetaTensor(*query.impl());
    auto meta_dist_input_key = MakeDistMetaTensor(*key.impl());
    auto meta_dist_input_value = MakeDistMetaTensor(*value.impl());
    auto meta_dist_input_seq_lens = MakeDistMetaTensor(*seq_lens.impl());
    auto meta_dist_input_kv_seq_lens = MakeDistMetaTensor(*kv_seq_lens.impl());
    auto meta_dist_input_mask = mask ? MakeDistMetaTensor(*(*mask).impl()) : phi::distributed::DistMetaTensor();
    auto spmd_info = phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_input_query, meta_dist_input_key, meta_dist_input_value, meta_dist_input_seq_lens, meta_dist_input_kv_seq_lens, meta_dist_input_mask);
    DebugInfoForInferSpmd("variable_length_memory_efficient_attention", spmd_info);

    // 2. Create API Output & Prepare Dist and Dense Output
    phi::DeviceContext* dev_ctx = nullptr;
    Tensor api_output;

    auto dist_out = SetKernelDistOutput(&api_output);
    auto dense_out = dist_out->unsafe_mutable_value();
    if (!rank_is_in_current_mesh) {{
      *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}

    // 3. Infer DistTensor's Global Shape
    phi::MetaTensor meta_dist_out(dist_out);
    phi::MetaTensor meta_dist_mask = mask ? MakeMetaTensor(*(*mask).impl()) : phi::MetaTensor();

    phi::VariableLengthMemoryEfficientAttentionInferMeta(MakeMetaTensor(*query.impl()), MakeMetaTensor(*key.impl()), MakeMetaTensor(*value.impl()), MakeMetaTensor(*seq_lens.impl()), MakeMetaTensor(*kv_seq_lens.impl()), meta_dist_mask, scale, causal, pre_cache_length, &meta_dist_out);


    if (rank_is_in_current_mesh) {
      // 4. Select Kernel
      VLOG(6) << "variable_length_memory_efficient_attention API dist branch: kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
      auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
          "variable_length_memory_efficient_attention", {kernel_backend, kernel_layout, kernel_data_type});
      const auto& kernel = kernel_result.kernel;
      VLOG(6) << "variable_length_memory_efficient_attention kernel: " << kernel;
      dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

      // 5. Reshard Input
      auto dist_input_query = ReshardApiInputToKernelInput(dev_ctx, query, spmd_info.first[0], "query");
      auto dist_input_key = ReshardApiInputToKernelInput(dev_ctx, key, spmd_info.first[1], "key");
      auto dist_input_value = ReshardApiInputToKernelInput(dev_ctx, value, spmd_info.first[2], "value");
      auto dist_input_seq_lens = ReshardApiInputToKernelInput(dev_ctx, seq_lens, spmd_info.first[3], "seq_lens");
      auto dist_input_kv_seq_lens = ReshardApiInputToKernelInput(dev_ctx, kv_seq_lens, spmd_info.first[4], "kv_seq_lens");
      auto dist_input_mask = ReshardApiInputToKernelInput(dev_ctx, mask, spmd_info.first[5], "mask");

      // 6. PrepareData (DataTransform & Prepare Dense Input)
      dist_input_query = PrepareDataForDistTensor(dist_input_query, GetKernelInputArgDef(kernel.InputAt(0), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_query = &dist_input_query->value();

      dist_input_key = PrepareDataForDistTensor(dist_input_key, GetKernelInputArgDef(kernel.InputAt(1), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_key = &dist_input_key->value();

      dist_input_value = PrepareDataForDistTensor(dist_input_value, GetKernelInputArgDef(kernel.InputAt(2), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_value = &dist_input_value->value();

      dist_input_seq_lens = PrepareDataForDistTensor(dist_input_seq_lens, GetKernelInputArgDef(kernel.InputAt(3), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_seq_lens = &dist_input_seq_lens->value();

      dist_input_kv_seq_lens = PrepareDataForDistTensor(dist_input_kv_seq_lens, GetKernelInputArgDef(kernel.InputAt(4), kernel_backend), {}, kernel_result.is_stride_kernel);
      auto input_kv_seq_lens = &dist_input_kv_seq_lens->value();

      dist_input_mask = PrepareDataForDistTensor(dist_input_mask, GetKernelInputArgDef(kernel.InputAt(5), kernel_backend), {}, kernel_result.is_stride_kernel);
      paddle::optional<phi::DenseTensor> input_mask = dist_input_mask ? paddle::make_optional<phi::DenseTensor>((*dist_input_mask)->value()) : paddle::none;

      // 7. RecordOpInfoSupplement
      if(phi::RecordOpInfoSupplement::IsEnabled()){
         std::vector<phi::DDim> mask_record_shapes;
         if(input_mask){
           mask_record_shapes.push_back((*input_mask).dims());
         }
         std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
         {"query", {
         (*input_query).dims()}},
         {"key", {
         (*input_key).dims()}},
         {"value", {
         (*input_value).dims()}},
         {"seq_lens", {
         (*input_seq_lens).dims()}},
         {"kv_seq_lens", {
         (*input_kv_seq_lens).dims()}},
         {"mask",
         mask_record_shapes}};
         phi::AttributeMap attrs;
         attrs["scale"] = scale;
         attrs["causal"] = causal;
         attrs["pre_cache_length"] = pre_cache_length;
         phi::RecordOpInfoSupplement("variable_length_memory_efficient_attention", input_shapes, attrs);
      }
      // 8. Infer Local DenseTensor Meta
      phi::MetaTensor meta_dense_out(dense_out);
      phi::VariableLengthMemoryEfficientAttentionInferMeta(MakeMetaTensor(*input_query), MakeMetaTensor(*input_key), MakeMetaTensor(*input_value), MakeMetaTensor(*input_seq_lens), MakeMetaTensor(*input_kv_seq_lens), MakeMetaTensor(input_mask), scale, causal, pre_cache_length, &meta_dense_out);

      // 9. DenseTensor Kernel Call
      phi::RecordEvent* kernel_record_event = nullptr;
      if(phi::RecordEvent::IsEnabled()){
        kernel_record_event = new phi::RecordEvent("variable_length_memory_efficient_attention dist compute", phi::TracerEventType::OperatorInner, 1);
      }
      using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, bool, int, phi::DenseTensor*);
      auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
      (*kernel_fn)(*dev_ctx, *input_query, *input_key, *input_value, *input_seq_lens, *input_kv_seq_lens, input_mask, scale, causal, pre_cache_length, dense_out);
      if(kernel_record_event != nullptr){
        delete kernel_record_event;
      }

      // 10. Fallback
      if (kernel_result.has_fallback_cpu) {
        TransDataBackend(dense_out, kernel_backend, dense_out);
      }
    }

    // 11. Set Output Dist Attr For Default Impl
    auto current_process_mesh = paddle::holds_alternative<phi::distributed::TensorDistAttr>(spmd_info.first[0]) ?
               paddle::get<0>(spmd_info.first[0]).process_mesh() : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();
    SetReplicatedDistAttrForOutput(dist_out, current_process_mesh);

    // 12. Return
    return api_output;
  }

  VLOG(6) << "variable_length_memory_efficient_attention API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "variable_length_memory_efficient_attention", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  if (FLAGS_low_precision_op_list) {
    phi::KernelFactory::Instance().AddToLowPrecisionKernelList("variable_length_memory_efficient_attention", kernel_data_type);
  }
  VLOG(6) << "variable_length_memory_efficient_attention kernel: " << kernel;
  // add actual_kernel_backend to select actual kernel backend after a potential falling-back to CPU
  Backend actual_kernel_backend = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  auto input_query = PrepareData(query, GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_key = PrepareData(key, GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_value = PrepareData(value, GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_seq_lens = PrepareData(seq_lens, GetKernelInputArgDef(kernel.InputAt(3), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_kv_seq_lens = PrepareData(kv_seq_lens, GetKernelInputArgDef(kernel.InputAt(4), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  auto input_mask = PrepareData(mask, GetKernelInputArgDef(kernel.InputAt(5), actual_kernel_backend), {}, kernel_result.is_stride_kernel);
  if(phi::RecordOpInfoSupplement::IsEnabled()){
     std::vector<phi::DDim> mask_record_shapes;
     if(input_mask){
       mask_record_shapes.push_back((*input_mask).dims());
     }
     std::vector<std::pair<const char*, std::vector<phi::DDim>>> input_shapes{
     {"query", {
     (*input_query).dims()}},
     {"key", {
     (*input_key).dims()}},
     {"value", {
     (*input_value).dims()}},
     {"seq_lens", {
     (*input_seq_lens).dims()}},
     {"kv_seq_lens", {
     (*input_kv_seq_lens).dims()}},
     {"mask",
     mask_record_shapes}};
     phi::AttributeMap attrs;
     attrs["scale"] = scale;
     attrs["causal"] = causal;
     attrs["pre_cache_length"] = pre_cache_length;
     phi::RecordOpInfoSupplement("variable_length_memory_efficient_attention", input_shapes, attrs);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(&api_output);

  phi::RecordEvent *infer_shape_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    infer_shape_record_event = new phi::RecordEvent("variable_length_memory_efficient_attention infer_meta", phi::TracerEventType::OperatorInner, 1);
  }
  phi::MetaTensor meta_out(kernel_out, kernel_result.is_stride_kernel);

  phi::VariableLengthMemoryEfficientAttentionInferMeta(MakeMetaTensor(*input_query), MakeMetaTensor(*input_key), MakeMetaTensor(*input_value), MakeMetaTensor(*input_seq_lens), MakeMetaTensor(*input_kv_seq_lens), MakeMetaTensor(input_mask), scale, causal, pre_cache_length, &meta_out);

  if(infer_shape_record_event != nullptr){
    delete infer_shape_record_event;
  }
  using kernel_signature = void(*)(const phi::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, bool, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  phi::RecordEvent* kernel_record_event = nullptr;
  if(phi::RecordEvent::IsEnabled()){
    kernel_record_event = new phi::RecordEvent("variable_length_memory_efficient_attention compute", phi::TracerEventType::OperatorInner, 1);
  }
    (*kernel_fn)(*dev_ctx, *input_query, *input_key, *input_value, *input_seq_lens, *input_kv_seq_lens, input_mask, scale, causal, pre_cache_length, kernel_out);
  if(kernel_record_event != nullptr){
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);

  }
  return api_output;
}


}  // namespace experimental
}  // namespace paddle

namespace paddle {
PD_DECLARE_API(from_blob);
#ifdef PADDLE_WITH_DISTRIBUTE
PD_DECLARE_API(reshard);
#endif
}  // namespace paddle
