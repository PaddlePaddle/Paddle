// #include <cmath>
// #include <vector>

// #include "cutlass/bfloat16.h"
// #include "cutlass/gemm/gemm.h"
// #include "cutlass/layout/matrix.h"
// #include "cutlass/layout/vector.h"
// #include "cutlass/numeric_types.h"

// // todo move below
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/attention_scaling_coefs_updater.h"
// #include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
// #include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
// #include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
// #include "cutlass/gemm/device/default_gemm_configuration.h"
// #include "cutlass/gemm/kernel/default_gemm.h"
// #include "cutlass/gemm/threadblock/default_mma.h"
// #include "cutlass/gemm/threadblock/default_mma_core_simt.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
// #include "cutlass/gemm/threadblock/threadblock_swizzle.h"
// #include "cutlass/matrix_shape.h"
// #include "cutlass/platform/platform.h"
// #include "cutlass/transform/threadblock/predicated_tile_iterator.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/debug_utils.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/epilogue_pipelined.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/epilogue_rescale_output.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/find_default_mma.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/gemm_kernel_utils.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/mma_from_smem.h"
// #include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/tile_smem_loader.h"

#include <cmath>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"

#include "attention_scaling_coefs_updater.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "debug_utils.h"
#include "epilogue_pipelined.h"
#include "epilogue_rescale_output.h"
#include "find_default_mma.h"
#include "gemm_kernel_utils.h"
#include "mma_from_smem.h"
#include "tile_smem_loader.h"

#include <inttypes.h>

using namespace gemm_kernel_utils;

namespace {
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSm() {
  return (
      Arch::kMinComputeCapability >= 80 &&
              !cutlass::platform::is_same<scalar_t, float>::value
          ? 16
          : 12);
}
} // namespace

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int kQueriesPerBlock,
    int kKeysPerBlock,
    bool kSingleValueIteration // = `value.shape[-1] <= kKeysPerBlock`
    >
struct AttentionKernel {
  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = scalar_t;
  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  using output_accum_t = accum_t;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kPreloadV = ArchTag::kMinComputeCapability >= 80 &&
      cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock % 32 == 0, "");
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int kWarpSize = 32;

  // Launch bounds
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int kMinBlocksPerSm =
      getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, num_heads, head_dim]
    scalar_t* key_ptr; // [num_keys, num_heads, head_dim]
    scalar_t* attn_bias_ptr = nullptr; // [num_heads, num_queries, num_keys]
    scalar_t* value_ptr; // [num_keys, num_heads, head_dim_value]
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    output_t* output_ptr; // [num_queries, num_heads, head_dim_value]
    output_accum_t*
        output_accum_ptr; // [num_queries, num_heads, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_heads, num_queries] - can be null

    // Scale
    accum_t scale;

    // Dimensions/strides
    int32_t head_dim;
    int32_t head_dim_value;
    int32_t num_queries;
    int32_t num_keys;

    bool causal;
    
    int32_t q_strideM;
    int32_t k_strideM;
    int32_t v_strideM;
    int32_t bias_strideM;

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int32_t q_strideH;
    int32_t k_strideH;
    int32_t v_strideH;
    
    int32_t o_strideH;

    int64_t bias_strideH;

    int64_t q_strideB;
    int64_t k_strideB;
    int64_t v_strideB;
    
    int64_t o_strideB;

    int64_t bias_strideB;

    int32_t num_batches;
    int32_t num_heads;

    CUTLASS_HOST_DEVICE int32_t o_strideM() const {
      return head_dim_value;
    }
    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto query_start = blockIdx.x * kQueriesPerBlock;

      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      int64_t q_start, k_start;
      // Advance to current batch - in case of different sequence lengths
      if (cu_seqlens_q_ptr != nullptr) {
        assert(cu_seqlens_k_ptr != nullptr);
        cu_seqlens_q_ptr += batch_id;
        cu_seqlens_k_ptr += batch_id;
        q_start = cu_seqlens_q_ptr[0];
        k_start = cu_seqlens_k_ptr[0];
        int64_t q_next_start = cu_seqlens_q_ptr[1];
        int64_t k_next_start = cu_seqlens_k_ptr[1];
        num_queries = q_next_start - q_start;
        num_keys = k_next_start - k_start;

        if (query_start >= num_queries) {
          return false;
        }
      } else {
        query_ptr += batch_id * q_strideB;
        key_ptr += batch_id * k_strideB;
        value_ptr += batch_id * v_strideB;
        output_ptr += batch_id * o_strideB;
        if (output_accum_ptr != nullptr) {
          output_accum_ptr += batch_id * o_strideB;
        }
        q_start = 0;
        k_start = 0;
      }

      // Advance to the current batch / head / query_start
      query_ptr += (q_start + query_start) * q_strideM + head_id * q_strideH;
      key_ptr += k_start * k_strideM + head_id * k_strideH;
      value_ptr += k_start * v_strideM + head_id * v_strideH;
      output_ptr += int64_t(q_start + query_start) * o_strideM() +
          head_id * o_strideH;

      if (attn_bias_ptr != nullptr) {
        const int64_t bias_offset = (batch_id * bias_strideB) + (head_id * bias_strideH); 
        attn_bias_ptr += bias_offset; 
      }

      if (output_accum_ptr != nullptr) {
        output_accum_ptr += int64_t(q_start + query_start) * o_strideM() +
            head_id * o_strideH;
      } else {
        // Accumulate directly in the destination buffer (eg for f32)
        output_accum_ptr = (accum_t*)output_ptr;
      }
      if (logsumexp_ptr != nullptr) {
        // lse[batch_id, head_id, query_start]
        logsumexp_ptr +=
            batch_id * lse_dim * num_heads + head_id * lse_dim + query_start;
      }

      num_queries -= query_start;
      if (causal) {
        num_keys = cutlass::fast_min(
            int32_t(query_start + kQueriesPerBlock), num_keys);
      }
      num_batches = 0; // no longer used after

      // Make sure the compiler knows these variables are the same on all
      // the threads of the warp.
      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      attn_bias_ptr = warp_uniform(attn_bias_ptr);

      output_ptr = warp_uniform(output_ptr);
      output_accum_ptr = warp_uniform(output_accum_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      head_dim = warp_uniform(head_dim);
      head_dim_value = warp_uniform(head_dim_value);
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          ceil_div(num_queries, (int32_t)kQueriesPerBlock),
          num_heads,
          num_batches);
    }
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        DefaultConfig::kStages, // Should use `DefaultConfig::kStages`, but that
                                // uses too much smem
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using Mma = typename DefaultMma::ThreadblockMma;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Updater;
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    using BiasLoader = TileSmemLoader<
        scalar_t,
        cutlass::MatrixShape<kQueriesPerBlock, kKeysPerBlock>,
        MmaCore::kThreads,
        // input restriction: kv_len has to be a multiple of this value
        128 / cutlass::sizeof_bits<scalar_t>::value>;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA; // from smem
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MM0::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;

    struct SharedStorageMM1 {
      typename Mma::SharedStorage mm;
    };
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      typename MM0::BiasLoader::SmemTile bias;
      typename MM0::AccumulatorSharedStorage si;
      typename MM1::SharedStorageMM1 mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      typename MM0::BiasLoader::SmemTile bias;
      typename MM0::AccumulatorSharedStorage si;
      typename MM1::SharedStorageMM1 mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kSingleValueIteration || kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    XFORMERS_CHECK(
        p.q_strideM % kAlignmentQ == 0, "query is not correctly aligned");
    XFORMERS_CHECK(
        p.k_strideM % kAlignmentK == 0, "key is not correctly aligned");
    XFORMERS_CHECK(
        p.v_strideM % kAlignmentV == 0, "value is not correctly aligned");
    XFORMERS_CHECK(
        p.q_strideH % kAlignmentQ == 0, "query is not correctly aligned");
    XFORMERS_CHECK(
        p.k_strideH % kAlignmentK == 0, "key is not correctly aligned");
    XFORMERS_CHECK(
        p.v_strideH % kAlignmentV == 0, "value is not correctly aligned");
    return true;
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.mi;
    const uint32_t query_start = blockIdx.x * kQueriesPerBlock;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }
    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM()},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries, p.head_dim_value},
          thread_id(),
          {0, col});
    };

    auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{(int32_t)p.o_strideM()},
              p.output_accum_ptr,
              typename OutputTileIteratorAccum::TensorCoord{
                  p.num_queries, p.head_dim_value},
              thread_id(),
              {0, col});
        };

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock) {
      int32_t problem_size_0_m =
          cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
      int32_t problem_size_0_n = cutlass::fast_min(
          int32_t(kKeysPerBlock), p.num_keys - iter_key_start);
      int32_t const& problem_size_0_k = p.head_dim;
      int32_t const& problem_size_1_n = p.head_dim_value;
      int32_t const& problem_size_1_k = problem_size_0_n;

      auto prologueV = [&](int blockN) {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1.mm,
            iterator_V,
            thread_id(),
            problem_size_1_k);
      };

      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[query_start:query_end, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //

      // Compute threadblock location
      cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

      cutlass::MatrixCoord tb_offset_A{
          tb_tile_offset.m() * MM0::Mma::Shape::kM, tb_tile_offset.k()};

      cutlass::MatrixCoord tb_offset_B{
          tb_tile_offset.k(), tb_tile_offset.n() * MM0::Mma::Shape::kN};

      // Construct iterators to A and B operands
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(p.q_strideM)),
          p.query_ptr,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          tb_offset_A);

      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(
              typename MM0::MmaCore::LayoutB(p.k_strideM)),
          p.key_ptr + iter_key_start * p.k_strideM,
          {problem_size_0_k, problem_size_0_n},
          thread_id(),
          tb_offset_B);

      auto my_warp_id = warp_id();
      auto my_lane_id = lane_id();

      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();

      if (kPreloadV) {
        prologueV(0);
      }

      typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {
              (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +
                  (my_warp_id % MM0::Mma::WarpCount::kM),
              (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +
                  (my_warp_id / MM0::Mma::WarpCount::kM)};

      // multiply by scaling factor
      // problem is here!
      // accum = cutlass::multiplies<typename MM0::Mma::FragmentC>()(1.0f / cutlass::fast_sqrt(float(p.head_dim)), accum);
      accum = cutlass::multiplies<typename MM0::Mma::FragmentC>()(p.scale, accum);

      // apply attention bias if applicable
      if (p.attn_bias_ptr != nullptr) {
        typename MM0::BiasLoader::GmemTileIterator bias_iter(
            {cutlass::layout::RowMajor(p.bias_strideM)},
            // attn_bias_pointer points to matrix of size (n_queries, n_keys)
            // for the relevant batch_id and head_id
            p.attn_bias_ptr + query_start * p.bias_strideM + iter_key_start,
            {1, problem_size_0_n},
            thread_id());


        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.after_mm0.bias.data(),
            cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
        typename MM0::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id());
        MM0::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, Pij is in register fragment and Bij is in shared memory
        auto lane_offset = MM0::ScalingCoefsUpdater::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        MM0::ScalingCoefsUpdater::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                accum[idx] += bias_tensor_ref.at({0, accum_n});
              }
            },
            [&](int accum_m) {});
      }

      // Mask out last if causal
      if (p.causal && p.num_keys - iter_key_start <= kKeysPerBlock) {
        auto query_start = blockIdx.x * kQueriesPerBlock;
        auto lane_offset = MM0::ScalingCoefsUpdater::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        int32_t last_col;
        MM0::ScalingCoefsUpdater::iterateRows(
            lane_offset,
            [&](int accum_m) {
              last_col = query_start + accum_m - iter_key_start;
            },
            [&](int accum_m, int accum_n, int idx) {
              if (accum_n > last_col) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }

      DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
                      DISPATCH_BOOL(
                          p.num_keys - iter_key_start >= kKeysPerBlock,
                          kFullColumns,
                          ([&] {
                            // Update `mi` from accum stored in registers
                            // Also updates `accum` with accum[i] <-
                            // exp(accum[i] * scale
                            // - mi)
                            MM0::ScalingCoefsUpdater::update<
                                kQueriesPerBlock,
                                kFullColumns,
                                kIsFirst,
                                kKeepOutputInRF>(
                                accum_o,
                                accum,
                                mi,
                                m_prime,
                                s_prime,
                                lane_id(),
                                thread_id(),
                                warp_id(),
                                p.num_keys - iter_key_start,
                                iteratorC_tile_offset, 
                                // 1.0f / cutlass::fast_sqrt(float(p.head_dim))); 
                                1.0f); 
                          }));
                    }));

      // Output results to shared-memory
      int warp_idx_mn_0 = my_warp_id %
          (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

      __syncthreads();

      //
      // MATMUL: Attn . V
      // Run the matmul `attn @ V` for a block of attn and V.
      // `attn` is read from shared memory (in `shared_storage_si`)
      // `V` is read from global memory (with iterator_B)
      //

      const int64_t nBlockN = kSingleValueIteration
          ? 1
          : ceil_div(
                (int64_t)problem_size_1_n, int64_t(MM1::ThreadblockShape::kN));
      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        int gemm_k_iterations =
            (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add and store it in accum
        // (in registers)
        if (!kPreloadV) {
          __syncthreads(); // we share shmem between mma and epilogue
        }

        typename MM1::Mma::IteratorB iterator_V(
             typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        typename MM1::Mma mma_pv(
            shared_storage.after_mm0.mm1.mm,
            shared_storage.after_mm0.si,
            (int)thread_id(),
            (int)warp_id(),
            (int)lane_id(),
            (int)problem_size_1_k);
        mma_pv.set_prologue_done(kPreloadV);
        if (!kKeepOutputInRF) {
          accum_o.clear();
        }
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
        __syncthreads();

        if (kPreloadV && !kSingleValueIteration && blockN + 1 < nBlockN) {
          prologueV(blockN + 1);
        }

        if (!kKeepOutputInRF) {
          DISPATCH_BOOL(
              iter_key_start == 0, kIsFirst, ([&] {
                DISPATCH_BOOL(
                    (iter_key_start + kKeysPerBlock) >= p.num_keys,
                    kIsLast,
                    ([&] {
                      using DefaultEpilogue = typename MM1::DefaultEpilogue;
                      using DefaultOp =
                          typename MM1::DefaultConfig::EpilogueOutputOp;
                      using ElementCompute = typename DefaultOp::ElementCompute;
                      using EpilogueOutputOp = typename cutlass::epilogue::
                          thread::MemoryEfficientAttentionNormalize<
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  output_t,
                                  output_accum_t>::type,
                              output_accum_t,
                              DefaultOp::kCount,
                              typename DefaultOp::ElementAccumulator,
                              ElementCompute,
                              kIsFirst,
                              kIsLast,
                              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                      using Epilogue = typename cutlass::epilogue::threadblock::
                          EpiloguePipelined<
                              typename DefaultEpilogue::Shape,
                              typename MM1::Mma::Operator,
                              DefaultEpilogue::kPartitionsK,
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  typename MM1::OutputTileIterator,
                                  typename MM1::OutputTileIteratorAccum>::type,
                              typename DefaultEpilogue::
                                  AccumulatorFragmentIterator,
                              typename DefaultEpilogue::WarpTileIterator,
                              typename DefaultEpilogue::SharedLoadIterator,
                              EpilogueOutputOp,
                              typename DefaultEpilogue::Padding,
                              DefaultEpilogue::kFragmentsPerIteration,
                              true, // IterationsUnroll
                              typename MM1::OutputTileIteratorAccum // Read
                                                                    // iterator
                              >;

                      int col = blockN * MM1::Mma::Shape::kN;
                      auto source_iter = createOutputAccumIter(col);
                      auto dest_iter = call_conditional<
                          kIsLast,
                          decltype(createOutputIter),
                          decltype(createOutputAccumIter)>::
                          apply(createOutputIter, createOutputAccumIter, col);
                      EpilogueOutputOp rescale(s_prime, m_prime);
                      Epilogue epilogue(
                          shared_storage.epilogue_shared_storage(),
                          thread_id(),
                          warp_id(),
                          lane_id());
                      epilogue(rescale, dest_iter, accum_o, source_iter);
                    }));
              }));
          if (!kSingleValueIteration) {
            __syncthreads();
          }
        }
      }
      __syncthreads(); // we modify `m_prime` after
    }

    if (kKeepOutputInRF) {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      using ElementCompute = typename DefaultOp::ElementCompute;
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // output
              output_accum_t, // source
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // accum
              output_accum_t, // compute
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // destination
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter(0);
      EpilogueOutputOp rescale(s_prime, m_prime);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      epilogue(rescale, dest_iter, accum_o);
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
      if (thread_id() < p.num_queries) {
        p.logsumexp_ptr[thread_id()] = accum_t(mi[thread_id()]) +
            cutlass::fast_log(accum_t(s_prime[thread_id()]));
      } else if (thread_id() < lse_dim) {
        p.logsumexp_ptr[thread_id()] =
            cutlass::platform::numeric_limits<accum_t>::infinity();
      }
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }
  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }
  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched(typename AK::Params params);

#define _ATTENTION_KERNEL_FORWARD_BEGIN(...)                                  \
  template <>                                                                 \
  __global__ void __launch_bounds__(                                          \
      __VA_ARGS__::kNumThreads, __VA_ARGS__::kMinBlocksPerSm)                 \
      attention_kernel_batched<__VA_ARGS__>(typename __VA_ARGS__::Params p) { \
    using Kernel = __VA_ARGS__;
#define _ATTENTION_KERNEL_FORWARD_END() }

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD(              \
    ARCH,                                                  \
    SCALAR_T,                                              \
    IS_ALIGNED,                                            \
    QUERIES_PER_BLOCK,                                     \
    KEYS_PER_BLOCK,                                        \
    SINGLE_VALUE_ITER)                                     \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<         \
                                  SCALAR_T,                \
                                  cutlass::arch::Sm##ARCH, \
                                  IS_ALIGNED,              \
                                  QUERIES_PER_BLOCK,       \
                                  KEYS_PER_BLOCK,          \
                                  SINGLE_VALUE_ITER>)      \
  if (!p.advance_to_block()) {                             \
    return;                                                \
  }                                                        \
  Kernel::attention_kernel(p);                             \
  _ATTENTION_KERNEL_FORWARD_END();

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(              \
    ARCH,                                                           \
    SCALAR_T,                                                       \
    IS_ALIGNED,                                                     \
    QUERIES_PER_BLOCK,                                              \
    KEYS_PER_BLOCK,                                                 \
    SINGLE_VALUE_ITER)                                              \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<                  \
                                  SCALAR_T,                         \
                                  cutlass::arch::Sm##ARCH,          \
                                  IS_ALIGNED,                       \
                                  QUERIES_PER_BLOCK,                \
                                  KEYS_PER_BLOCK,                   \
                                  SINGLE_VALUE_ITER>)               \
  printf(                                                           \
      "FATAL: this function is for sm%d, but was built for sm%d\n", \
      int(ARCH),                                                    \
      int(__CUDA_ARCH_OR_ZERO__));                                  \
  _ATTENTION_KERNEL_FORWARD_END();

// All kernels are disabled by default
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(50, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(70, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(75, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(80, __VA_ARGS__)

// Enable the right one based on __CUDA_ARCH__
#ifndef __CUDA_ARCH__
#elif __CUDA_ARCH__ < 500
#error "Need cuda arch at least 5.0"
#elif __CUDA_ARCH__ < 700
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(50, __VA_ARGS__)
#elif __CUDA_ARCH__ < 750
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(70, __VA_ARGS__)
#elif __CUDA_ARCH__ < 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(75, __VA_ARGS__)
#elif __CUDA_ARCH__ >= 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(80, __VA_ARGS__)
#endif