# Copyright (c) 2023 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from .common import get_cutlass_path
from .gemm_profiler import GemmProfiler

logger = logging.getLogger("cutlass")


def select_gemm_kernel(
    cutlass_profiler,
    op_type,
    MM,
    KK,
    NN,
    in0_dtype,
    in1_dtype,
    out_dtype,
    use_3xtf32,
    batched,
    find_first_valid,
    use_multiprocessing,
):
    """Run CUTLASS profiler to select the best kernel."""
    name, cutlass_op_def, _ = cutlass_profiler.profile(
        op_type,
        MM,
        NN,
        KK,
        in0_dtype,
        in1_dtype,
        out_dtype,
        use_3xtf32,
        batched=batched,
        find_first_valid=find_first_valid,
        use_multiprocessing=use_multiprocessing,
    )
    if not find_first_valid:
        logger.info("The best kernel is %s", name)
    else:
        logger.info("Picked the first kernel found %s", name)

    return name, cutlass_op_def


def handle_batch_matmul(
    cutlass_profiler,
    op_type,
    M,
    N,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    use_3xtf32,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for batch_matmul op workload."""
    MM = M
    KK = K
    NN = N

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        in0_dtype,
        in1_dtype,
        use_3xtf32,
        True,
        find_first_valid,
        use_multiprocessing,
    )

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def handle_matmul(
    cutlass_profiler,
    op_type,
    M,
    N,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    use_3xtf32,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for dense op workload."""
    MM = M
    KK = K
    NN = N

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        in0_dtype,
        in1_dtype,
        out_dtype,
        use_3xtf32,
        False,
        find_first_valid,
        use_multiprocessing,
    )

    assert (
        "tn_align" in name
    ), "Only supports (row_major, col_major) input layout for now."

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def gen_gemm_kernel(
    sm,
    op_type='cutlass_matmul',
    M=128,
    N=128,
    K=128,
    in0_dtype='float16',
    in1_dtype='float16',
    out_dtype='float16',
    find_first_valid=True,
    use_3xtf32=True,
    split_k_slices=[1],
    profile_all_alignments=False,
    use_multiprocessing=False,
    tmp_dir="./tmp",
):
    """(TODO)Given a module partitioned for CUTLASS offloading, profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    use_3xtf32 : bool
        Wheter or not use slower but very accurate (compared to tf32) 3xtf32 mode for
        fp32 inputs on tensorcore.

    split_k_slices : list of int
        Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
        parallel across split-K blocks, and a separate global reduction kernel is launched to
        accumulate partial reductions. The profiler will pick the best split-k factor from the
        given candidate list. Note that the larger split-K factor requires a larger workspace.
        Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
        kinds, split_k_slices is ignored.

    profile_all_alignments : bool
        When True, profile all kernal variants with smaller alignments than the largest possible.

    find_first_valid : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    mod : IRModule
        The updated module annotated with cutlass profiling information.
    """
    # Profiler for dense operators. May cache results between tuned functions.
    gemm_profiler = GemmProfiler(sm, get_cutlass_path(), tmp_dir)

    new_attrs = {"op_type": op_type}

    if "batch_matmul" in op_type:
        new_attrs.update(
            handle_batch_matmul(
                gemm_profiler,
                op_type,
                M,
                N,
                K,
                in0_dtype,
                in1_dtype,
                out_dtype,
                use_3xtf32,
                find_first_valid,
                use_multiprocessing,
            )
        )
    elif "matmul" in op_type:
        new_attrs.update(
            handle_matmul(
                gemm_profiler,
                op_type,
                M,
                N,
                K,
                in0_dtype,
                in1_dtype,
                out_dtype,
                use_3xtf32,
                find_first_valid,
                use_multiprocessing,
            )
        )
    else:
        raise ValueError(f"{op_type} unsupported composite")
    print(new_attrs['cutlass_op_name'])

    return new_attrs['cutlass_op_name'], new_attrs['cutlass_op_def']
