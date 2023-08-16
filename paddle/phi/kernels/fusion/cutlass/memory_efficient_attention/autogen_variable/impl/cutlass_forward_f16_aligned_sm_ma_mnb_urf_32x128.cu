// This file is auto-generated. See "generate_variable_forward_kernels.py"
#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/memory_efficient_variable_attention.h"
namespace phi {


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true, false> default_fmha, Params &params, const phi::GPUContext& ctx) {
  using AttentionKernel = typename decltype(default_fmha)::FMHAKernel;
  using FMHA = cutlass::gemm::device::GemmGrouped<AttentionKernel>;
  using scalar_t = typename FMHA::GemmKernel::scalar_t;
  using accum_t = typename FMHA::GemmKernel::accum_t;
  using output_t = typename FMHA::GemmKernel::output_t;
  using output_accum_t = typename FMHA::GemmKernel::output_accum_t;
  using ElementQ = scalar_t;
  using ElementK = scalar_t;
  using ElementP = accum_t;
  using ElementM = scalar_t;
  using ElementAccumulator = accum_t;
  using ElementV = scalar_t;
  using ElementO = output_t;
  using ElementOAccum = output_accum_t;

  int problem_count = params.num_batches * params.num_heads;

  std::vector<GemmCoord> problem_sizes1;
  problem_sizes1.reserve(problem_count);

  phi::Allocator::AllocationPtr problem_sizes_device0{nullptr};
  phi::Allocator::AllocationPtr problem_sizes_device1{nullptr};
  problem_sizes_device0 = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      problem_count * sizeof(GemmCoord),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  problem_sizes_device1 = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      problem_count * sizeof(GemmCoord),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  GemmCoord* problem0_device =
      reinterpret_cast<GemmCoord*>(problem_sizes_device0->ptr());
  GemmCoord* problem1_device =
      reinterpret_cast<GemmCoord*>(problem_sizes_device1->ptr());
  get_problem_sizes<<<params.num_batches, params.num_heads, 0, ctx.stream()>>>(
      params.seq_lens,
      params.kv_seq_lens,
      problem0_device,
      problem1_device,
      params.num_batches,
      params.num_heads,
      params.head_size,
      params.value_head_size);
  phi::memory_utils::Copy(phi::CPUPlace(),
                       problem_sizes1.data(),
                       ctx.GetPlace(),
                       problem1_device,
                       sizeof(GemmCoord) * problem_count,
                       ctx.stream());
  if (AttentionKernel::kNeedsOutputAccumulatorBuffer) {
    const int64_t output_size = params.num_batches * params.num_heads *
                                params.query_seq_len * params.value_head_size;
    phi::Allocator::AllocationPtr tmp_output_accum_buffer_ptr{nullptr};
    tmp_output_accum_buffer_ptr = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        output_size * sizeof(ElementOAccum),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    params.output_accum_ptr = tmp_output_accum_buffer_ptr->ptr();
  }
  int threadblock_count =
      FMHA::sufficient(problem_sizes1.data(), problem_count);
  typename FMHA::Arguments args(
      problem0_device,
      problem1_device,
      problem_count,
      threadblock_count,
      params.num_heads,
      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.query_ptr)),
      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.key_ptr)),
      params.mask_ptr
          ? const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.mask_ptr))
          : nullptr,
      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.value_ptr)),
      reinterpret_cast<scalar_t*>(params.output_ptr),
      AttentionKernel::kNeedsOutputAccumulatorBuffer
          ? reinterpret_cast<output_accum_t*>(params.output_accum_ptr)
          : nullptr,
      params.ldq,
      params.ldk,
      params.ldm,
      params.ldv,
      params.ldo,
      params.ElementQ,
      params.ElementK,
      params.ElementM,
      params.ElementV,
      params.ElementO,
      params.causal,
      params.scale,
      problem_sizes1.data());

  FMHA fmha;
  cutlass::Status status;
  size_t workspace_size = fmha.get_workspace_size(args);
  phi::DenseTensor workspace;
  workspace.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  ctx.template Alloc<uint8_t>(&workspace);
  status = fmha.initialize(args, workspace.data<uint8_t>());
  if (status != cutlass::Status::kSuccess) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Failed to initialize CUTLASS Grouped FMHA kernel."));
  }
  status = fmha.run(ctx.stream());
  if (status != cutlass::Status::kSuccess) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Failed to run CUTLASS Grouped FMHA kernel."));
  }
}
} // namespace phi
#endif // PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
