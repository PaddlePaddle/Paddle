// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

#include <cub/cub.cuh>
#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

#ifdef PADDLE_WITH_CUDNN_FRONTEND

namespace phi {
namespace cudnn_fused_attn {

#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

#define CHECK_CUDNN_FE(func)                                               \
  {                                                                        \
    auto error = func;                                                     \
    if (error.is_bad()) {                                                  \
      throw std::runtime_error(std::string("CUDNN Frontend error at ") +   \
                               __FILE__ + ":" + std::to_string(__LINE__) + \
                               " " + error.err_msg);                       \
    }                                                                      \
  }

void generateMatrixStrides(int64_t b,
                           int64_t h,
                           int64_t s_q,
                           int64_t s_kv,
                           int64_t d,
                           int64_t *strideA,
                           MHA_Layout layout,
                           MHA_Matrix matrix) {
  constexpr int batch_dim_idx = 0;
  constexpr int head_dim_idx = 1;
  constexpr int seqlen_dim_idx = 2;
  constexpr int hidden_dim_idx = 3;

  constexpr int seqlen_transpose_dim_idx = 3;
  constexpr int hidden_transpose_dim_idx = 2;

  constexpr int seqlen_q_dim_idx = 2;
  constexpr int seqlen_kv_dim_idx = 3;

  switch (layout) {
    case MHA_Layout::BS3HD:
      if ((matrix == MHA_Matrix::Q_Matrix) ||
          (matrix == MHA_Matrix::K_Matrix) ||
          (matrix == MHA_Matrix::V_Matrix)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == MHA_Matrix::K_Matrix_Transpose) ||
                 (matrix == MHA_Matrix::V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if (matrix == MHA_Matrix::O_Matrix) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case MHA_Layout::BSHD_BS2HD:
      if ((matrix == MHA_Matrix::K_Matrix) ||
          (matrix == MHA_Matrix::V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = 2 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == MHA_Matrix::K_Matrix_Transpose) ||
                 (matrix == MHA_Matrix::V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == MHA_Matrix::Q_Matrix) ||
                 (matrix == MHA_Matrix::O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case MHA_Layout::BSHD_BSHD_BSHD:
      if ((matrix == MHA_Matrix::Q_Matrix) ||
          (matrix == MHA_Matrix::O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == MHA_Matrix::K_Matrix) ||
                 (matrix == MHA_Matrix::V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == MHA_Matrix::K_Matrix_Transpose) ||
                 (matrix == MHA_Matrix::V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      }
      break;
  }

  if (matrix == MHA_Matrix::S_Matrix) {
    strideA[seqlen_kv_dim_idx] = 1;
    strideA[seqlen_q_dim_idx] = s_kv;
    strideA[head_dim_idx] = s_q * s_kv;
    strideA[batch_dim_idx] = h * s_q * s_kv;
  }
}

static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type,
                                            int64_t id,
                                            int64_t const *dim,
                                            int64_t const *stride,
                                            bool is_virtual,
                                            bool is_value) {
  int nbDims = 4;
  auto tensor_created =
      cudnn_frontend::TensorBuilder()
          .setDim(nbDims, dim)
          .setStride(nbDims, stride)
          .setId(id)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(type)
          .setVirtual(is_virtual)
          .setByValue(is_value)
          .build();
  VLOG(10) << tensor_created.describe();
  return tensor_created;
}

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type,
                                                    cudnnPointwiseMode_t mode) {
  auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(mode)
                             .setComputeType(type)
                             .build();

  VLOG(10) << pw_desc_created.describe();
  return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &bDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &bDesc,
    cudnn_frontend::Tensor const &tDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .settDesc(tDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

}  // namespace cudnn_fused_attn
}  // namespace phi

using namespace phi::cudnn_fused_attn;  // NOLINT

constexpr int BLOCK_SIZE = 512;

__global__ __launch_bounds__(BLOCK_SIZE) void mask_to_actual_seqlens_kernel(
    const int32_t *mask,
    int32_t *q_actual_seqlen,
    int32_t *kv_actual_seqlen,
    int q_seqlen,
    int kv_seqlen,
    bool need_kv) {
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage q_smem;
  __shared__ typename BlockReduce::TempStorage kv_smem;
  unsigned int tid = threadIdx.x;
  unsigned int batch_offset = blockIdx.x * q_seqlen * kv_seqlen;

  // load mask, convert to 1/0, do accumulation
  int q = 0, kv = 0;
  for (unsigned int q_idx = tid * kv_seqlen; q_idx < q_seqlen * kv_seqlen;
       q_idx += BLOCK_SIZE * kv_seqlen) {
    q += (mask[q_idx + batch_offset] ? 1 : 0);
  }

  if (need_kv) {
    for (unsigned int kv_idx = tid; kv_idx < kv_seqlen; kv_idx += BLOCK_SIZE) {
      kv += (mask[kv_idx + batch_offset] ? 1 : 0);
    }
  }
  __syncthreads();

  // compute cub::BlockReduce
  int q_sum, kv_sum;
  q_sum = BlockReduce(q_smem).Sum(q);
  if (need_kv) kv_sum = BlockReduce(kv_smem).Sum(kv);

  // write result for this block to global mem
  if (tid == 0) {
    q_actual_seqlen[blockIdx.x] = q_sum;
    if (need_kv) {
      kv_actual_seqlen[blockIdx.x] = kv_sum;
    }
  }
}

//  convert cu_seqlens to actual_seqlens
__global__ void cu_seqlens_to_actual_seqlens(size_t b,
                                             int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens,
                                             int32_t *q_seqlens,
                                             int32_t *kv_seqlens) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < b) {
    q_seqlens[tid] = q_cu_seqlens[tid + 1] - q_cu_seqlens[tid];
    kv_seqlens[tid] = kv_cu_seqlens[tid + 1] - kv_cu_seqlens[tid];
  }
}

// fill constant
template <typename scalar_t>
__global__ void fill_cu_seqlen_with_constant(scalar_t *cu_seqlens_q,
                                             scalar_t *cu_seqlens_kv,
                                             scalar_t q_seqlen,
                                             scalar_t kv_seqlen,
                                             size_t n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    cu_seqlens_q[tid] = q_seqlen;
    cu_seqlens_kv[tid] = kv_seqlen;
  }
}

void fused_attn_arbitrary_seqlen_fwd_impl(int64_t b,
                                          int64_t h,
                                          int64_t hg,
                                          int64_t s_q,
                                          int64_t s_kv,
                                          int64_t d,
                                          int64_t bias_b,
                                          int64_t bias_h,
                                          bool is_training,
                                          float scaling_factor,
                                          float dropout_probability,
                                          MHA_Layout layout,
                                          MHA_Bias_Type bias_type,
                                          MHA_Mask_Type mask_type,
                                          void *devPtrQ,
                                          void *devPtrK,
                                          void *devPtrV,
                                          void *devPtrBias,
                                          void *devPtrSoftmaxStats,
                                          void *devPtrO,
                                          void *devPtrDropoutSeed,
                                          void *devPtrDropoutOffset,
                                          void *devPtrCuSeqlensQ,
                                          void *devPtrCuSeqlensKV,
                                          cudnn_frontend::DataType_t tensorType,
                                          void *workspace,
                                          size_t *workspace_size,
                                          const phi::GPUContext &dev_ctx) {
  bool is_bias = (bias_type == MHA_Bias_Type::POST_SCALE_BIAS);
  bool is_alibi = false;
  bool is_causal = ((mask_type == MHA_Mask_Type::CAUSAL_MASK) ||
                    (mask_type == MHA_Mask_Type::PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == MHA_Mask_Type::PADDING_MASK) ||
                     (mask_type == MHA_Mask_Type::PADDING_CAUSAL_MASK));
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  auto handle = dev_ctx.cudnn_handle();

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               is_training,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
                               tensorType};

    namespace fe = cudnn_frontend;
    using graph_and_tensors = std::tuple<
        std::shared_ptr<fe::graph::Graph>,
        std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
        std::shared_ptr<fe::graph::Tensor_attributes>,   // K
        std::shared_ptr<fe::graph::Tensor_attributes>,   // V
        std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
        std::shared_ptr<fe::graph::Tensor_attributes>,   // O
        std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
        std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
        std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
        std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
        std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_f16_fprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph =
        [&](CacheType &cache,
            const FADescriptor_v1 &descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed,
          dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      generateMatrixStrides(
          b, h, s_q, s_kv, d, q_stride.data(), layout, MHA_Matrix::Q_Matrix);
      generateMatrixStrides(
          b, hg, s_q, s_kv, d, k_stride.data(), layout, MHA_Matrix::K_Matrix);
      generateMatrixStrides(
          b, hg, s_q, s_kv, d, v_stride.data(), layout, MHA_Matrix::V_Matrix);
      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d})
                                .set_stride(q_stride));
      K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d})
                                .set_stride(k_stride));
      V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d})
                                .set_stride(v_stride));

      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      fe::graph::SDPA_attributes sdpa_options;
      sdpa_options = fe::graph::SDPA_attributes()
                         .set_name("flash_attention")
                         .set_is_inference(!is_training)
                         .set_causal_mask(is_causal)
                         .set_attn_scale(attn_scale);

      sdpa_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({bias_b, bias_h, s_q, s_kv})
                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);
      }

      if (is_padding) {
        seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(is_padding)
            .set_seq_len_q(seq_q)
            .set_seq_len_kv(seq_kv);
      }

      if (is_dropout) {
        dropout_seed =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Seed")
                                  .set_dim({1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT64));
        dropout_offset =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Offset")
                                  .set_dim({1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT64));
        sdpa_options.set_dropout(
            dropout_probability, dropout_seed, dropout_offset);
      }

      auto [O, Stats] = mha_graph->sdpa(Q, K, V, sdpa_options);

      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(
          b, h, s_q, s_kv, d, o_stride.data(), layout, MHA_Matrix::O_Matrix);
      O->set_output(true).set_dim({b, h, s_q, d}).set_stride(o_stride);

      if (is_training) {
        Stats->set_output(true)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({b, h, s_q, 1})
            .set_stride({h * s_q, s_q, 1, 1});
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // O
          key_tensors_tuple = std::make_tuple(Q, K, V, attn_scale, O);
      auto Stats_tuple =
          is_training ? std::make_tuple(Stats) : std::make_tuple(nullptr);
      auto bias_tuple =
          is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
      auto padding_tuple = is_padding ? std::make_tuple(seq_q, seq_kv)
                                      : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout
                               ? std::make_tuple(dropout_seed, dropout_offset)
                               : std::make_tuple(nullptr, nullptr);
      auto return_empty_tuple = std::tuple_cat(std::make_tuple(nullptr),
                                               key_tensors_tuple,
                                               Stats_tuple,
                                               bias_tuple,
                                               padding_tuple,
                                               dropout_tuple);

      CHECK_CUDNN_FE(mha_graph->validate());
      CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      CHECK_CUDNN_FE(mha_graph->check_support(handle));
      CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph),
                                         key_tensors_tuple,
                                         Stats_tuple,
                                         bias_tuple,
                                         padding_tuple,
                                         dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph,
          Q,
          K,
          V,
          attn_scale,
          O,
          Stats,
          bias,
          seq_q,
          seq_kv,
          dropout_seed,
          dropout_offset] = get_graph(sdpa_f16_fprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    if (workspace == nullptr) {
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls
    // with null streams for sizing the cuDNN workspace.
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetStream(handle, dev_ctx.stream()));

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {{Q, devPtrQ},
                        {K, devPtrK},
                        {V, devPtrV},
                        {attn_scale, &scaling_factor},
                        {O, devPtrO}};

    if (is_training) {
      variant_pack[Stats] = devPtrSoftmaxStats;
    }

    if (is_bias) {
      variant_pack[bias] = devPtrBias;
    }

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void *devActualSeqlenQ =
          static_cast<int8_t *>(workspace) + plan_workspace_size;
      void *devActualSeqlenKV =
          static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
      if (devPtrCuSeqlensQ != nullptr && devPtrCuSeqlensKV != nullptr) {
        cu_seqlens_to_actual_seqlens<<<grid,
                                       nthreads_per_block,
                                       0,
                                       dev_ctx.stream()>>>(
            b,
            static_cast<const int32_t *>(devPtrCuSeqlensQ),
            static_cast<const int32_t *>(devPtrCuSeqlensKV),
            static_cast<int32_t *>(devActualSeqlenQ),
            static_cast<int32_t *>(devActualSeqlenKV));
      } else {
        // set all actual seqlens to max seqlen
        fill_cu_seqlen_with_constant<<<grid,
                                       nthreads_per_block,
                                       0,
                                       dev_ctx.stream()>>>(
            static_cast<int32_t *>(devActualSeqlenQ),
            static_cast<int32_t *>(devActualSeqlenKV),
            static_cast<int32_t>(s_q),
            static_cast<int32_t>(s_kv),
            b);
      }
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    PADDLE_THROW(common::errors::Fatal(std::string(e.what())));
  }
}

void fused_attn_arbitrary_seqlen_bwd_impl(int64_t b,
                                          int64_t h,
                                          int64_t hg,
                                          int64_t s_q,
                                          int64_t s_kv,
                                          int64_t d,
                                          int64_t bias_b,
                                          int64_t bias_h,
                                          float scaling_factor,
                                          float dropout_probability,
                                          MHA_Layout layout,
                                          MHA_Bias_Type bias_type,
                                          MHA_Mask_Type mask_type,
                                          bool deterministic,
                                          void *devPtrQ,
                                          void *devPtrKTranspose,
                                          void *devPtrVTranspose,
                                          void *devPtrO,
                                          void *devPtrSoftmaxStats,
                                          void *devPtrBias,
                                          void *devPtrdQ,
                                          void *devPtrdK,
                                          void *devPtrdV,
                                          void *devPtrdO,
                                          void *devPtrdBias,
                                          void *devPtrDropoutSeed,
                                          void *devPtrDropoutOffset,
                                          void *devPtrCuSeqlensQ,
                                          void *devPtrCuSeqlensKV,
                                          cudnn_frontend::DataType_t tensorType,
                                          void *workspace,
                                          size_t *workspace_size,
                                          const phi::GPUContext &dev_ctx) {
  bool is_bias = (bias_type == MHA_Bias_Type::POST_SCALE_BIAS);
  bool need_dbias = (bias_b == 1) && (bias_h == h) && devPtrdBias != nullptr;
  bool is_alibi = false;
  bool is_causal = ((mask_type == MHA_Mask_Type::CAUSAL_MASK) ||
                    (mask_type == MHA_Mask_Type::PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == MHA_Mask_Type::PADDING_MASK) ||
                     (mask_type == MHA_Mask_Type::PADDING_CAUSAL_MASK));
  bool is_dropout = (dropout_probability != 0.0f);
  auto handle = dev_ctx.cudnn_handle();

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               true,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
                               tensorType};

    namespace fe = cudnn_frontend;
    using graph_and_tensors = std::tuple<
        std::shared_ptr<fe::graph::Graph>,
        std::shared_ptr<fe::graph::Tensor_attributes>,   // q
        std::shared_ptr<fe::graph::Tensor_attributes>,   // k
        std::shared_ptr<fe::graph::Tensor_attributes>,   // v
        std::shared_ptr<fe::graph::Tensor_attributes>,   // o
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dO
        std::shared_ptr<fe::graph::Tensor_attributes>,   // stats
        std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dQ
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dK
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dV
        std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dBias
        std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
        std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
        std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
        std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_f16_bprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph =
        [&](CacheType &cache,
            const FADescriptor_v1 &descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dO, stats,
          attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed,
          dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(
          b, h, s_q, s_kv, d, q_stride.data(), layout, MHA_Matrix::Q_Matrix);
      generateMatrixStrides(
          b, hg, s_q, s_kv, d, k_stride.data(), layout, MHA_Matrix::K_Matrix);
      generateMatrixStrides(
          b, hg, s_q, s_kv, d, v_stride.data(), layout, MHA_Matrix::V_Matrix);
      generateMatrixStrides(
          b, h, s_q, s_kv, d, o_stride.data(), layout, MHA_Matrix::O_Matrix);
      q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d})
                                .set_stride(q_stride));
      k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d})
                                .set_stride(k_stride));
      v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d})
                                .set_stride(v_stride));
      o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("O")
                                .set_dim({b, h, s_q, d})
                                .set_stride(o_stride));
      dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("dO")
                                 .set_dim({b, h, s_q, d})
                                 .set_stride(o_stride));
      stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("stats")
                                    .set_dim({b, h, s_q, 1})
                                    .set_stride({h * s_q, s_q, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));

      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      fe::graph::SDPA_backward_attributes sdpa_backward_options;
      sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                  .set_name("flash_attention_backward")
                                  .set_causal_mask(is_causal)
                                  .set_attn_scale(attn_scale);

      sdpa_backward_options.set_deterministic_algorithm(deterministic);

      sdpa_backward_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({bias_b, bias_h, s_q, s_kv})
                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        dBias = mha_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("dBias")
                .set_dim({bias_b, bias_h, s_q, s_kv})
                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_backward_options.set_bias(bias);
        // shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s]
        // are not supported for dbias calculation but they are
        // supported for forward bias calculation
        if (need_dbias) {
          sdpa_backward_options.set_dbias(dBias);
        }
      }

      if (is_padding) {
        seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        sdpa_backward_options.set_padding_mask(is_padding)
            .set_seq_len_q(seq_q)
            .set_seq_len_kv(seq_kv);
      }

      if (is_dropout) {
        dropout_seed =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Seed")
                                  .set_dim({1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT64));
        dropout_offset =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Offset")
                                  .set_dim({1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::INT64));
        sdpa_backward_options.set_dropout(
            dropout_probability, dropout_seed, dropout_offset);
      }

      auto [dQ, dK, dV] = mha_graph->sdpa_backward(
          q, k, v, o, dO, stats, sdpa_backward_options);

      dQ->set_output(true).set_dim({b, h, s_q, d}).set_stride(q_stride);
      dK->set_output(true).set_dim({b, hg, s_kv, d}).set_stride(k_stride);
      dV->set_output(true).set_dim({b, hg, s_kv, d}).set_stride(v_stride);

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // dV
          key_tensors_tuple =
              std::make_tuple(q, k, v, o, dO, stats, attn_scale, dQ, dK, dV);
      auto bias_tuple = is_bias ? std::make_tuple(bias, dBias)
                                : std::make_tuple(nullptr, nullptr);
      auto padding_tuple = is_padding ? std::make_tuple(seq_q, seq_kv)
                                      : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout
                               ? std::make_tuple(dropout_seed, dropout_offset)
                               : std::make_tuple(nullptr, nullptr);
      auto return_empty_tuple = std::tuple_cat(std::make_tuple(nullptr),
                                               key_tensors_tuple,
                                               bias_tuple,
                                               padding_tuple,
                                               dropout_tuple);

      CHECK_CUDNN_FE(mha_graph->validate());
      CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      CHECK_CUDNN_FE(mha_graph->check_support(handle));
      CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph),
                                         key_tensors_tuple,
                                         bias_tuple,
                                         padding_tuple,
                                         dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph,
          q,
          k,
          v,
          o,
          dO,
          stats,
          attn_scale,
          dQ,
          dK,
          dV,
          bias,
          dBias,
          seq_q,
          seq_kv,
          dropout_seed,
          dropout_offset] = get_graph(sdpa_f16_bprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    if (workspace == nullptr) {
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls
    // with null streams for sizing the cuDNN workspace.
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetStream(handle, dev_ctx.stream()));

    // build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {
            {q, devPtrQ},
            {k, devPtrKTranspose},
            {v, devPtrVTranspose},
            {o, devPtrO},
            {dO, devPtrdO},
            {stats, devPtrSoftmaxStats},
            {attn_scale, &scaling_factor},
            {dQ, devPtrdQ},
            {dK, devPtrdK},
            {dV, devPtrdV},
        };

    if (is_bias) {
      variant_pack[bias] = devPtrBias;
      if (need_dbias) {
        variant_pack[dBias] = devPtrdBias;
      } else {
        variant_pack[dBias] = nullptr;
      }
    }

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void *devActualSeqlenQ =
          static_cast<int8_t *>(workspace) + plan_workspace_size;
      void *devActualSeqlenKV =
          static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
      if (devPtrCuSeqlensQ != nullptr && devPtrCuSeqlensKV != nullptr) {
        cu_seqlens_to_actual_seqlens<<<grid,
                                       nthreads_per_block,
                                       0,
                                       dev_ctx.stream()>>>(
            b,
            static_cast<const int32_t *>(devPtrCuSeqlensQ),
            static_cast<const int32_t *>(devPtrCuSeqlensKV),
            static_cast<int32_t *>(devActualSeqlenQ),
            static_cast<int32_t *>(devActualSeqlenKV));
      } else {
        // set all actual seqlens to max seqlen
        fill_cu_seqlen_with_constant<<<grid,
                                       nthreads_per_block,
                                       0,
                                       dev_ctx.stream()>>>(
            static_cast<int32_t *>(devActualSeqlenQ),
            static_cast<int32_t *>(devActualSeqlenKV),
            static_cast<int32_t>(s_q),
            static_cast<int32_t>(s_kv),
            b);
      }
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    PADDLE_THROW(common::errors::Fatal(std::string(e.what())));
  }
}

#endif
