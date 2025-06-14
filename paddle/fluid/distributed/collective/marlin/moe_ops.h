#pragma once
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "paddle/phi/core/enforce.h"
#include "paddle/fluid/distributed/collective/marlin/kernels/kernel.h"
#include "paddle/fluid/distributed/collective/marlin/include/types.h"
#include "paddle/phi/api/include/api.h"

paddle::Tensor moe_wna16_marlin_gemm_api(
    const paddle::Tensor& a,
    const std::optional<paddle::Tensor>& c_or_none,
    const paddle::Tensor& b_q_weight,
    const paddle::Tensor& b_scales,
    const std::optional<paddle::Tensor>& global_scale_or_none,
    const std::optional<paddle::Tensor>& b_zeros_or_none,
    const std::optional<paddle::Tensor>& g_idx_or_none,
    const std::optional<paddle::Tensor>& perm_or_none,
    const paddle::Tensor& workspace,
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& expert_ids,
    const paddle::Tensor& num_tokens_post_padded,
    const paddle::Tensor& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    const std::string& b_q_type_str,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float);