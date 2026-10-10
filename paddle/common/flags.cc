// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/common/flags.h"

namespace phi {

const ExportedFlagInfoMap &GetExportedFlagInfoMap() {
  return *GetMutableExportedFlagInfoMap();
}

ExportedFlagInfoMap *GetMutableExportedFlagInfoMap() {
  static ExportedFlagInfoMap g_exported_flag_info_map;
  return &g_exported_flag_info_map;
}

}  // namespace phi

PHI_DEFINE_EXPORTED_int32(inner_op_parallelism,
                          0,
                          "number of threads for inner op");

/**
 * Note: Enable CUDNNv8 Frontend API for CUDNN kernels.
 * Disabled by default on WITH_XPU_CADA because xtrans's xpudnn 8.x does not
 * implement the cudnn_frontend graph-based backend operations; the legacy
 * cudnnConvolutionForward / cudnnPoolingForward path works correctly.
 */
#ifdef PADDLE_WITH_XPU_CADA
PHI_DEFINE_EXPORTED_bool(enable_cudnn_frontend, false, "");
#else
PHI_DEFINE_EXPORTED_bool(enable_cudnn_frontend, true, "");
#endif

/**
 * CUDNNv8 related FLAG
 * Name: cudnn_cache_saturation_count
 * Since Version: 2.5.0
 * Value Range: int64_t, default=1
 * Example:
 * Note: Set saturation count for CUDNNv8 cache. A candidate execution
 * plan need to be considered as the fastest plan by exhaustive search
 * N times before it is actually added in the cache. It is useful when
 * the result of exhaustive search is unstable.
 */
PHI_DEFINE_EXPORTED_int32(cudnn_cache_saturation_count, 1, "");
#endif  // PADDLE_WITH_CUDNN_FRONTEND

/**
 * CI related FLAG
 * Name: trt_ibuilder_cache
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: This FLAG is only enabled when CI is running. If True, a persistent
 * IBuilder is added to avoid TensorRT unload/reload kernels.
 */
PHI_DEFINE_EXPORTED_bool(trt_ibuilder_cache,
                         false,
                         "Add a persistent ibuilder.");

/**
 * mmap_allocator related FLAG
 * Name: use_shm_cache
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: . If True, mmap_allocator will cache shm file to decrease munmap
 * operation.
 */
PHI_DEFINE_EXPORTED_bool(use_shm_cache,
                         false,
                         "Use shm cache in mmap_allocator.");

/**
 * mmap_allocator related FLAG
 * Name: dataloader_use_file_descriptor
 * Since Version: 2.6.2
 * Value Range: bool, default=false
 * Example:
 * Note: . If True, mmap_allocator will use file descriptor to open shared
 * memory operation.
 */
PHI_DEFINE_EXPORTED_bool(dataloader_use_file_descriptor,
                         false,
                         "Use file descriptor in mmap_allocator.");

/**
 * Tensor operants related FLAG
 * Name: tensor_operants_mode
 * Since Version: 2.5.0
 * Value Range: string, {eager, phi, static}
 * default=eager
 * Example:
 * Note: For switching tensor operants mode of PaddlePaddle.
 *       - eager mode: tensor operants with dygraph autograd;
 *       - phi mode: tensor operants with only phi forward API;
 *       - static mode: tensor operants within static graph.
 */
PHI_DEFINE_EXPORTED_string(tensor_operants_mode,
                           "eager",
                           "Tensor operants mode");

/**
 * Using PIR in executor  FLAG
 * Name: enable_pir_in_executor
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, executor will use PIR
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_in_executor,
                         false,
                         "Enable PIR in executor");

/**
 * Using PIR API in Python
 * Name: enable_custom_engine
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, CustomDevice can use subgraph engine optimize
 */
PHI_DEFINE_EXPORTED_string(enable_custom_engine,
                           "",
                           "Set CustomDevice subgraph engine translate pass");

/**
 * Using PIR by translating legacy program to pir program
 * for dy2st mode  FLAG
 * Name: enable_pir_in_executor
 * Since Version: 2.6.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, program will be translated to pir program
 * and then run in executor for dy2st mode.
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_with_pt_in_dy2st,
                         true,
                         "Enable PIR in executor");

PHI_DEFINE_EXPORTED_string(logging_pir_py_code_dir,
                           "",
                           "the logging directory to save pir py code");

PHI_DEFINE_EXPORTED_int64(
    logging_pir_py_code_int_tensor_element_limit,
    2048,
    "dump int tensor data if its element count less than this limit.");

PHI_DEFINE_EXPORTED_bool(logging_trunc_pir_py_code,
                         true,
                         "whether truncate the logging files under directory "
                         "FLAGS_logging_pir_py_code_dir");

PHI_DEFINE_EXPORTED_bool(logging_pir_py_code_dump_symbolic_dims,
                         false,
                         "whether dump symbolic dims into pir py code.");

/**
 * Enable Abstract Pass
 * Name: enable_ap
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, abstract pass will be enabled to optimize performance.
 */
PHI_DEFINE_EXPORTED_bool(enable_ap, false, "whether enable abstract pass.");

/**
 * Enable Classic fused_gemm_epilogue when Abstract Pass is enabled.
 * Name: ap_enable_classic_gemm_epilogue
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, classic fused_gemm_epilogue will be enabled.
 */
PHI_DEFINE_EXPORTED_bool(ap_enable_classic_gemm_epilogue,
                         false,
                         "whether enable classic fused_gemm_epilogue when "
                         "abstract pass is enabled.");

PHI_DEFINE_EXPORTED_bool(
    pir_interpreter_record_stream_for_gc_cache,
    false,
    "whether PirInterpreter::RecordStreamForGC use cache strategy.");

/**
 * Using PIR API in Python
 * Name: enable_pir_api
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, PIR API will be used in Python
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_api, true, "Enable PIR API in Python");

/**
 * Using PIR in executor FLAG
 * Name: enable_pir_in_executor_trace_run
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, executor will use PIR and run in beta version by for trace
 * version.
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_in_executor_trace_run,
                         false,
                         "Enable PIR in executor");

/**
 * Apply inplace pass to PIR FLAG
 * Name: pir_apply_inplace_pass
 * Since Version: 2.6.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, will apply inplace pass to PIR.
 */
PHI_DEFINE_EXPORTED_bool(pir_apply_inplace_pass,
                         true,
                         "Whether to apply inplace pass on lowering "
                         "::pir::Program to Kernel Dialect");

PHI_DEFINE_EXPORTED_string(
    ir_inplace_kernel_blacklist,
    "",
    "It controls the ir inplace kernel subset do not use.");

PHI_DEFINE_EXPORTED_bool(enable_record_memory, false, "Enable memory recorder");

PHI_DEFINE_EXPORTED_bool(
    eager_delete_scope,
    true,
    "Delete local scope eagerly. It will reduce GPU memory usage but "
    "slow down the destruction of variables.(around 1% performance harm)");

// Used to filter events, works like glog VLOG(level).
// RecordEvent will works if host_trace_level >= level.
PHI_DEFINE_EXPORTED_int64(host_trace_level,
                          1,
                          "RecordEvent will works "
                          "if host_trace_level >= level.");

PHI_DEFINE_EXPORTED_int32(
    multiple_of_cupti_buffer_size,
    1,
    "Multiple of the CUPTI device buffer size. If the timestamps have "
    "been dropped when you are profiling, try increasing this value.");

PHI_DEFINE_EXPORTED_bool(print_ir, false, "Whether print ir debug str.");

// Whether to enable CINN kernel cache
// When enabled, generated files will be saved under:
// FLAGS_cinn_kernel_cache_save_path/virtual_device_id/HostFuncName__fushionHashKey
// Files:
// - cinn_cuda_kernel.fatbin (CUDA kernels)
// - cinn_cache.so (host modules)
// This cache can accelerate subsequent CINN compilations
PHI_DEFINE_EXPORTED_bool(enable_cinn_kernel_cache,
                         false,
                         "Whether enable cinn kernel cache.");

// Specify the directory path of generated cinn kernel cache
PHI_DEFINE_EXPORTED_string(
    cinn_kernel_cache_save_path,
    "/tmp/cinn/",
    "Specify the directory path of generated cinn kernel cache.");

PHI_DEFINE_EXPORTED_bool(
    comp_skip_default_ops,
    true,
    "Whether to skip decomposing comp op in default list (decomp_trans.cc).");

PHI_DEFINE_EXPORTED_bool(
    prim_skip_dynamic,
    true,
    "Whether to skip decomposing vjp op with dynamic shape.");
PHI_DEFINE_EXPORTED_bool(
    prim_enable_dynamic,
    false,
    "Whether to enable decomposing composite op with dynamic shape.");
PHI_DEFINE_EXPORTED_bool(prim_check_ops,
                         false,
                         "Whether to check the decomposed program, to ensure "
                         "that only the primitive operator is present.");

// PIR and prim related FLAG
// Example: FLAGS_prim_forward_blacklist="pd_op.relu;pd_op.mean" would block
// `relu` and `mean` two ops in decompsition.
PHI_DEFINE_EXPORTED_string(
    prim_forward_blacklist,
    "",
    "It controls the forward blacklist ops not to be decomposed.");
PHI_DEFINE_EXPORTED_bool(prim_forward, false, "enable prim_forward or not");
PHI_DEFINE_EXPORTED_bool(prim_backward, false, "enable prim_backward or not");

/**
 * Remove some redundant information when printing the pir program
 * Name: disable_logging_op_attr_list
 * Since Version: 3.0.0
 * Value Range: string, default=""
 * Example: FLAGS_disable_logging_op_attr_list="op_dist_attr"
 * Note: If "dtype", "dtype:float32" will be deleted in Pir program
 */
PHI_DEFINE_EXPORTED_string(
    disable_logging_op_attr_list,
    "",
    "Remove some redundant information when printing the pir program");

#ifdef _WIN32
PHI_DEFINE_EXPORTED_string(
    flagcx_dir,  // NOLINT
    "",
    "Specify path for loading libflagcx.so. For instance, "
    "For instance, /usr/local/flagcx/lib. If default, "
    "dlopen will search flagcx from LD_LIBRARY_PATH");
#endif

/**
 * ProcessGroupNCCL related FLAG
 * Name: enable_async_trace
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note: enable nccl async trace.
 */

PHI_DEFINE_EXPORTED_bool(enable_async_trace,
                         false,
                         "enable collective async trace");

PHI_DEFINE_EXPORTED_int32(async_trace_count, 5, "collective async trace count");

PHI_DEFINE_EXPORTED_bool(
    use_auto_growth_pinned_allocator,
    false,
    "Whether to use the auto_growth CUDA pinned allocator.");

PHI_DEFINE_EXPORTED_bool(
    sync_after_alloc,
    false,
    "Whether to perform device synchronization after allocation.");
PHI_DEFINE_EXPORTED_int64(alloc_fill_value,
                          -1,
                          "Whether to fill fixed value after allocation. "
                          "This is useful for debugging.");

PHI_DEFINE_EXPORTED_int64(
    pir_broadcast_tree_limit,
    32,
    "Maximum number of broadcast nodes allowed in a tree");

PHI_DEFINE_EXPORTED_string(
    nvidia_package_dir,  // NOLINT
    "",
    "Specify root dir path for nvidia site-package, such as "
    "python3.9/site-packages/nvidia");

PHI_DEFINE_EXPORTED_string(cuda_cccl_dir,  // NOLINT
                           "",
                           "Specify root dir path for nv/target, such as "
                           "python3.9/site-packages/nvidia/cuda_cccl/include/");

PHI_DEFINE_EXPORTED_string(
    cudnn_dir,  // NOLINT
    "",
    "Specify path for loading libcudnn.so. For instance, "
    "/usr/local/cudnn/lib. If empty [default], dlopen "
    "will search cudnn from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(  // NOLINT
    cuda_dir,
    "",
    "Specify path for loading cuda library, such as libcublas, libcublasLt "
    "libcurand, libcusolver. For instance, /usr/local/cuda/lib64. "
    "If default, dlopen will search cuda from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(cublas_dir,  // NOLINT
                           "",
                           "Specify path for loading libcublas.so.");
PHI_DEFINE_EXPORTED_string(
    nccl_dir,  // NOLINT
    "",
    "Specify path for loading nccl library, such as libnccl.so. "
    "For instance, /usr/local/cuda/lib64. If default, "
    "dlopen will search cuda from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(cupti_dir,
                           "",
                           "Specify path for loading cupti.so.");  // NOLINT

PHI_DEFINE_EXPORTED_string(  // NOLINT
    tensorrt_dir,
    "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

PHI_DEFINE_EXPORTED_string(
    mklml_dir,
    "",
    "Specify path for loading libmklml_intel.so.");  // NOLINT

PHI_DEFINE_EXPORTED_string(hml_dir,
                           "",
                           "Specify path for loading libhml_rt.so.");  // NOLINT

PHI_DEFINE_EXPORTED_string(lapack_dir,
                           "",
                           "Specify path for loading liblapack.so.");  // NOLINT

#ifdef PADDLE_WITH_MAGMA
PHI_DEFINE_EXPORTED_string(magma_dir,
                           "",
                           "Specify path for loading libmagma.so.");  // NOLINT
#endif

/**
 * Apply check infer symbolic pass FLAG
 * Name: check_infer_symbolic_pass
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will apply check_infer_symbolic pass.
 */
PHI_DEFINE_EXPORTED_bool(
    check_infer_symbolic,
    false,
    "Whether to use check_infer_symbolic_pass. This pass can check "
    "the symbolic inference accuracy by comparing the the value "
    "shape between dynamic shape and static shape.");

/**
 * Name: manually_trans_conv_filter
 * Since Version: 3.0.0 Beta
 * Value Range: bool, default=false
 */
PHI_DEFINE_EXPORTED_bool(
    manually_trans_conv_filter,
    false,
    "Whether to manually transpose the filter of conv2d. This pass can "
    "accelerate the performance of conv2d since it transpose filter ahead");

/**
 * Apply CSE optimize pass in Dy2St
 * Name: enable_cse_in_dy2st
 * Since Version: 3.0.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, will apply CSE optimize pass in Dy2St.
 */
PHI_DEFINE_EXPORTED_bool(enable_cse_in_dy2st,
                         true,
                         "Apply CSE optimize pass in Dy2St");

/**
 * Run Dy2St in specialized device
 * Name: specialize_device_in_dy2st
 * Since Version: 3.1.0 Beta
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will specialize device for DataOp's place based on input
 * tensor's place before lowering.
 */
PHI_DEFINE_EXPORTED_bool(specialize_device_in_dy2st,
                         false,
                         "Run Dy2St in specialized device");

/**
 * Persist parameters in scope to avoid the overhead of
 * repeated sharing during each execution period.
 * Name: parameters_persistent_mode_in_dy2st
 * Since Version: 3.1.1
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will persist parameters in scope to avoid the overhead of
 * repeated sharing during each execution period.
 */
PHI_DEFINE_EXPORTED_bool(parameters_persistent_mode_in_dy2st,
                         false,
                         "Persist parameters in scope to avoid the overhead of "
                         "repeated sharing during each execution period.");

/**
 * Max count of eliminate redundant computation in CSE, for debug usage
 * Name: cse_max_count
 * Since Version: 3.0.0
 * Value Range: int32, default=-1
 * Example:
 * Note: If -1, will not limit the max count of eliminate redundant computation.
 */
PHI_DEFINE_EXPORTED_int32(
    cse_max_count,
    -1,
    "Max count of eliminate redundant computation in CSE, for debug usage");

/**
 * Apply global search in cublaslt gemm
 * Name: enable_blaslt_global_search
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will apply global search in blaslt.
 */
PHI_DEFINE_EXPORTED_bool(enable_blaslt_global_search,
                         false,
                         "Whether to use global search in cublaslt gemm.");

/**
 * Apply load search configs file generated by offline in cublaslt gemm
 * Name: cublaslt_device_best_config
 * Since Version: 3.0.0
 * Value Range: string, default="", a absolute file path
 * Example:
 * Note: If set this flag, will load search configs file generated by offline.
 */
PHI_DEFINE_EXPORTED_string(cublaslt_device_best_config,
                           "",
                           "Whether to load search configs file generated by "
                           "offline in cublaslt gemm.");

/**
 * Whether to use xqa optim in block_multihead_attention kernel (GQA)
 * Name: use_xqa_optim
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will use xqa optim in block_multihead_attention kernel (GQA).
 */
PHI_DEFINE_EXPORTED_bool(
    use_xqa_optim,
    false,
    "Enable xqa optim in block_multihead_attention kernel (GQA).");

/**
 * Whether to use FP32 for accumulation of QK output in
 * block_multihead_attention kernel(fp16)
 * Name: blha_use_fp32_qk_sum Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If TRUE, FP32 will be used for accumulation of the QK output
 * in block_multihead_attention kernel(fp16) .
 */
PHI_DEFINE_EXPORTED_bool(blha_use_fp32_qk_sum,
                         false,
                         "use FP32 for accumulation of QK output in "
                         "block_multihead_attention kernel(fp16).");

PHI_DEFINE_EXPORTED_bool(cuda_core_int8_gemm,
                         false,
                         "Enable speed up int8 gemm calculations when m<=4");

PHI_DEFINE_EXPORTED_string(
    mkl_dir,  // NOLINT
    "",
    "Specify path for loading libmkl_rt.so. "
    "For instance, /opt/intel/oneapi/mkl/latest/lib/intel64/."
    "If default, "
    "dlopen will search mkl from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(op_dir,  // NOLINT
                           "",
                           "Specify path for loading user-defined op library.");

PHI_DEFINE_EXPORTED_string(cusparselt_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusparseLt.so.");
PHI_DEFINE_EXPORTED_string(curand_dir,  // NOLINT
                           "",
                           "Specify path for loading libcurand.so.10.");
PHI_DEFINE_EXPORTED_string(cusolver_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusolver.so.*.");
PHI_DEFINE_EXPORTED_string(cusparse_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusparse.so.*.");
PHI_DEFINE_EXPORTED_string(
    win_cuda_bin_dir,  // NOLINT
    "",
    "Specify path for loading *.dll about cuda on windows");

/**
 * Collect shapes of value for TensorRTEngine
 * Name: enable_collect_shape
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will collect shapes of value when run executor.
 */
PHI_DEFINE_EXPORTED_bool(enable_collect_shape,
                         false,
                         "Collect shapes of value for TensorRTEngine");
// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_fp32,
                           1e-6,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_fp32,
                           1e-6,
                           "It controls the rtol of accuracy_check op");

// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_fp16,
                           1e-3,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_fp16,
                           1e-3,
                           "It controls the rtol of accuracy_check op");

// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_bf16,
                           1e-3,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_bf16,
                           1e-3,
                           "It controls the rtol of accuracy_check op");

PHI_DEFINE_EXPORTED_bool(
    pinned_memory_as_cpu_backend,
    false,
    "Whether use CPU backend, when tensor is pinned_memory.");

PHI_DEFINE_EXPORTED_int32(
    trt_min_group_size,
    3,
    "when the trt subgraph size is not larger than `trt_min_group_size`, the "
    "group will fallback to original graph.");

/**
 * Enable align mode for auto parallel. If True, the loss results will aligned
 * with dynamic manual-parallel.
 * Name: enable_auto_parallel_align_mode
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Note: Just used for testing. Do not use in model training.
 */
PHI_DEFINE_EXPORTED_bool(enable_auto_parallel_align_mode,
                         false,
                         "Enable align mode for auto parallel");

/**
 * fused_multi_transformer_op related FLAG
 * Name: fused_multi_transformer_op_use_mbfmha
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Enable flash decoding for mmha kernels in fused_multi_transformer_op.
 */
PHI_DEFINE_EXPORTED_bool(fused_multi_transformer_op_use_mbfmha,
                         false,
                         "Enable flash decoding for mmha kernels in "
                         "fused_multi_transformer_op.");

PHI_DEFINE_EXPORTED_int64(multi_block_attention_min_partition_size,
                          1024,
                          "The minimum partition size for flash decoding");

PHI_DEFINE_EXPORTED_bool(save_cf_stack_op,
                         false,
                         "Save cf stack op for higher-order derivatives.");

PHI_DEFINE_EXPORTED_bool(
    enable_auto_growth_allocator_add_lock,
    false,
    "Enable add lock when call AutoGrowthBestFitAllocator::ReleaseImpl");

PHI_DEFINE_EXPORTED_int64(offload_retry_times, -1, "Offload retry times.");

PHI_DEFINE_EXPORTED_bool(offload_inplace_tensor,
                         true,
                         "Whether to allow offload inplace tensor.");

PHI_DEFINE_EXPORTED_bool(print_offload_info,
                         false,
                         "Whether to print the offload information.");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * FlashAttention related FLAG
 * Name: FLAGS_flash_attn_version
 * Value Range: int32, default=2
 * Example:
 * Note: Specify the version of FlashAttention to use, options are 2 or 3.
 *        Version 2 requires Ampere architecture or higher,
 *        while version 3 requires Hopper architecture.
 */
PHI_DEFINE_EXPORTED_int32(
    flash_attn_version,
    2,
    "Specify the version of FlashAttention to use, options are 2 or 3. "
    "Version 2 requires Ampere architecture or higher, "
    "while version 3 requires Hopper architecture.");
#endif

/**
 * Operator related FLAG
 * Name: FLAGS_check_cuda_error
 * Value Range: bool, default=false
 * Example:
 * Note: Used to debug. Checking whether CUDA error occurred or not.
 */
PHI_DEFINE_EXPORTED_bool(check_cuda_error,
                         false,
                         "Checking whether CUDA error occurred or not.");

/**
 * Stream related FLAG
 * Name: FLAGS_use_default_stream
 * Since Version: 3.1.1
 * Value Range: bool, default=false
 * Example:
 * Note: Whether use default stream.
 */
PHI_DEFINE_EXPORTED_bool(use_default_stream,
                         false,
                         "Whether use default stream.");

/**
 * Stride_Compute_Kernel related FLAG
 * Name: FLAGS_use_stride_compute_kernel
 * Since Version: 3.2
 * Value Range: bool, default=false
 * Example:
 * Note: Whether use Stride_Compute_Kernel.
 */
PHI_DEFINE_EXPORTED_bool(use_stride_compute_kernel,
                         true,
                         "Whether use Stride_Compute_Kernel.");

/**
 * Allocator related FLAG
 * Name: FLAGS_deep_ep_comm_prealloc_in_mb
 * Since Version: 3.2
 * Value Range: int64, default=0
 * Example:
 * Note: Whether use prealloc for deepep communication.
 */
PHI_DEFINE_EXPORTED_int64(deep_ep_comm_prealloc_in_mb,
                          0,
                          "Whether use prealloc for deepep communication.");

/**
 * Stride_Compute_Kernel related FLAG
 * Name: FLAGS_force_stride_compute_contig_out
 * Since Version: 3.2.1
 * Value Range: bool, default=false
 * Example:
 * Note: Whether force Stride_Compute_Kernel output contiguous.
 */
PHI_DEFINE_EXPORTED_bool(
    force_stride_compute_contig_out,
    false,
    "Whether force Stride_Compute_Kernel output contiguous.");

/**
 * Torch Compatible related FLAG
 * Name: FLAGS_use_accuracy_compatible_kernel
 * Since Version: 3.2.2
 * Value Range: bool, default=false
 * Example:
 * Note: Whether use torch compatible version kernel.
 */
PHI_DEFINE_EXPORTED_bool(use_accuracy_compatible_kernel,
                         false,
                         "Whether use torch compatible version kernel.");
/**
 * Legacy gemm related FLAG
 * Name: FLAGS_use_legacy_gemm
 * Since Version: 3.2.2
 * Value Range: bool, default=false
 * Example:
 * Note: Whether use legacy gemm kernel.
 */
PHI_DEFINE_EXPORTED_bool(use_legacy_gemm,
                         false,
                         "Whether use legacy gemm dispatch logics.");

/**
 * Legacy gemm related FLAG
 * Name: FLAGS_use_legacy_linear
 * Since Version: 3.3.1
 * Value Range: bool, default=false
 * Example:
 * Note: Whether use legacy linear kernel.
 */
PHI_DEFINE_EXPORTED_bool(use_legacy_linear,
                         false,
                         "Whether use legacy linear dispatch logics.");

/**
 * Allocator Compact related FLAG
 * Name: FLAGS_enable_compact_mem
 * Since Version: 3.3
 * Value Range: bool, default=false
 * Example:
 * Note: whether start compact memory.
 */
PHI_DEFINE_EXPORTED_bool(enable_compact_mem,
                         false,
                         "whether start compact memory or not.");
/**
 * Allocator Compact related FLAG
 * Name: FLAGS_max_reserved_threshold_in_gb
 * Since Version: 3.3
 * Value Range: int64, default=70
 * Example:
 * Note: Threshold (GB) used in compact memory. Only reserved_mem greater than
 * threshold may trigger defragmentation.
 */
PHI_DEFINE_EXPORTED_int64(
    max_reserved_threshold_in_gb,
    70,
    "Threshold (GB) used in compact memory. Only reserved_mem greater than "
    "threshold may trigger defragmentation.");

/**
 * Allocator Compact related FLAG
 * Name: FLAGS_cur_allocated_threshold_in_gb
 * Since Version: 3.3
 * Value Range: int64, default=70
 * Example:
 * Note: Threshold (GB) used in compact memory. Only reserved_mem greater than
 * threshold may trigger defragmentation.
 */
PHI_DEFINE_EXPORTED_int64(
    cur_allocated_threshold_in_gb,
    55,
    "Threshold (GB) used in compact memory. Only reserved_mem greater than "
    "threshold may trigger defragmentation.");

/**
 * Allocator Compact related FLAG
 * Name: FLAGS_try_allocate
 * Since Version: 3.3
 * Value Range: bool, default=false
 * Example:
 * Note: whether start compact memory.
 */
PHI_DEFINE_EXPORTED_bool(try_allocate,
                         false,
                         "whether use try allocate in memory compact.");

/**
 * Allocator Compact related FLAG
 * Name: FLAGS_record_alloc_event
 * Since Version: 3.3
 * Value Range: bool, default=false
 * Example:
 * Note: whether record allocate event.
 */
PHI_DEFINE_EXPORTED_bool(record_alloc_event,
                         false,
                         "whether record allocate event.");
