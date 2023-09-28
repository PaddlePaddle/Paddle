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
from .common import DEFAULT_KERNELS, SUPPORTED_GENERATOR_ARCH, ProfilerEngine
from .gemm_ops import (
    create_gemm_operator_with_epilogue,
    enumerate_gemm_operators,
)


class GemmProfiler:
    """(TODO)Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        assert (
            sm in SUPPORTED_GENERATOR_ARCH and sm in DEFAULT_KERNELS
        ), f"sm{sm} not supported yet."
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache = {}

    def select_op(
        self,
        M,
        N,
        K,
        in0_dtype,
        in1_dtype,
        out_dtype,
        use_3xtf32,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
    ):
        """
        Profile and select the best kernel from candidate kernels.
        See the documentation for the profile method below.
        """
        if (M, N, K) in self.cache:
            op = self.cache[(M, N, K)]
            return op

        # TODO(masahi): CUTLASS alignment check on gemm kernels is too restrictive.
        # See https://github.com/NVIDIA/cutlass/issues/362.
        # When the above issue is resolved, we can remove the alignment check on M below.

        ops = SUPPORTED_GENERATOR_ARCH[self.sm](
            in0_dtype,
            in1_dtype,
            out_dtype,
            enumerate_gemm_operators,
            lambda align: all(dim % align == 0 for dim in [M, N, K]),
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            # TODO(masahi): Invesitigate when fp32 accumulation is needed for gemm
            accumlator_dtype=out_dtype,
        )

        if not find_first_valid:
            self.engine.compile_all(ops, use_multiprocessing)

        for op in ops:
            out = self.engine.evaluate(op, [M, N, K])
            op["runtime"] = out
            if out < float("inf") and find_first_valid:
                self.cache[(M, N, K)] = op
                return op

        op = min(ops, key=lambda i: i["runtime"])
        self.cache[(M, N, K)] = op
        return op

    def profile(
        self,
        op_type,
        M,
        N,
        K,
        in0_dtype,
        in1_dtype,
        out_dtype,
        use_3xtf32=True,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
        batched=False,
    ):
        """Profile and select the best kernel from candidate kernels.
        If find_first_valid is True, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        op = self.select_op(
            M,
            N,
            K,
            in0_dtype,
            in1_dtype,
            out_dtype,
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )

        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            batched=batched,
        )

        return name, opdef, op["runtime"]
