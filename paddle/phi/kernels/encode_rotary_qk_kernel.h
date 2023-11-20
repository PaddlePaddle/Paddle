#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void RotaryQK(const Context& dev_ctx,
              const DenseTensor& q, 
              const DenseTensor& kv, 
              const DenseTensor& rotary_emb, 
              const DenseTensor& seq_lens,
              const int32_t rotary_emb_dims, 
              bool use_neox,
              DenseTensor* rotary_q_out,
              DenseTensor* rotary_kv_out);

}  // namespace phi