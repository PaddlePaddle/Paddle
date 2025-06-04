// #include "core/registration.h"
// #include "moe_ops.h"

// TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
//   // Apply topk softmax to the gating outputs.

// #ifndef USE_ROCM
//   m.def(
//       "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
//       "Tensor b_scales, Tensor? b_qzeros, "
//       "Tensor? topk_weights, Tensor sorted_token_ids, "
//       "Tensor expert_ids, Tensor num_tokens_post_pad, "
//       "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
//       "int bit) -> Tensor");

//   m.impl("moe_wna16_gemm", torch::kCUDA, &moe_wna16_gemm);

//   m.def(
//       "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
//       "Tensor! b_q_weight, Tensor! b_scales, Tensor? global_scale, Tensor? "
//       "b_zeros_or_none,"
//       "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
//       "Tensor sorted_token_ids,"
//       "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
//       "Tensor! topk_weights, int moe_block_size, int top_k, "
//       "bool mul_topk_weights, bool is_ep, int b_q_type_id,"
//       "int size_m, int size_n, int size_k,"
//       "bool is_full_k, bool use_atomic_add,"
//       "bool use_fp32_reduce, bool is_zp_float) -> Tensor");
//   m.def(
//       "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
//       "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
//       "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
//       "int b_q_type, SymInt size_m, "
//       "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
//       "topk, "
//       "int moe_block_size, bool replicate_input, bool apply_weights)"
//       " -> Tensor");

// #endif
// }

// REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
