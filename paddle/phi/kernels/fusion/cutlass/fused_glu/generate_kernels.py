# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Generates combination of kernels - implementations and registry

# Kernels are ordered (see `sort_index`), and when dispatching,
# we select the first kernel in the list that supports the inputs

import collections
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

# TODO(zhengzekang): Currently we only register FP16 kernel. 
DTYPES = {
    "f16": "cutlass::half_t",
}

ACC_DTYPES = {
    "f32": "float",
    "f16": "cutlass::half_t",
}

ACT_TYPES = {
    "silu": "cutlass::epilogue::thread::SiLu",
    "sigmoid": "cutlass::epilogue::thread::Sigmoid",
    "gelu": "cutlass::epilogue::thread::GELU_taylor"
}

SM = [70, 75, 80]

KERNEL_IMPL_TEMPLATE = """
template<>
__global__ void {NAME}<{CPP_CLASS}>(typename {CPP_CLASS}::Params params) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0 && __CUDA_ARCH__ < {SM_MAX}0
  using Operator = {CPP_CLASS}; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""


@dataclass(order=True)
class DualGemmFwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    dtype: str
    acc_dtype: str
    store_d: bool 
    act_type: str
    sm_range: Tuple[int, int]

    # def __post_init__(self) -> None:
    #     # Set kernel selection priority
    #     # The lowest value that matches inputs
    #     # will be selected
    #     self.sort_index = (
    #         # First select aligned kernel
    #         0 if self.aligned else 1,
    #         # Then keep output in RF
    #         0 if self.single_value_iter else 1,
    #         self.k,
    #         # Prefer kernels without dropout/bias if available
    #         1 if self.add_mask else 0,
    #         1 if self.mask_broadcast_row else 0,
    #     )

    @property
    def _sm_suffix(self) -> str:
        return self.sm_range[0]

    @property
    def name(self) -> str:
        return f"DualKernel"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                ACC_DTYPES[self.acc_dtype],
                "true" if self.store_d else "false",
                ACT_TYPES[self.act_type],
                f"cutlass::arch::Sm{self.sm_range[0]}"
            ]
        )
        return f"cutlass::gemm::kernel::DualGemm<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        # TODO(zhengzekang) For example: cutlass_fmha_forward_f16_aligned_70.cu contains kernel with fp16, Aligned, SM70 implementation. 
        return f"{self.dtype}__{self.acc_dtype}_{self._sm_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> List["DualGemmFwdKernel"]:
        kernels: List[DualGemmFwdKernel] = []
        for dtype, acc_dtype, (sm, sm_max) in itertools.product(
            DTYPES.keys(), ACC_DTYPES.keys(), zip(SM, SM[1:] + [90])
        ):
            for store_d in [True, False]:
                for act_type in ["sigmoid", "silu", "gelu"]: 
                    kernels.append(
                        cls(
                            dtype=dtype,
                            acc_dtype=acc_dtype, 
                            store_d=store_d, 
                            act_type=act_type, 
                            sm_range=(sm, sm_max),
                        )
                    )
        return kernels


T = DualGemmFwdKernel

def write_decl_impl(
    kernels: List[T], family_name: str, impl_file: str, disable_def: str
) -> None:
    cpp_file_header = """// This file is auto-generated. See "generate_kernels.py"
"""

    # kernels.sort()

    implfile_to_kernels: Dict[str, List[T]] = collections.defaultdict(list)
    cat_to_kernels: Dict[Tuple[str, int, int], List[T]] = collections.defaultdict(list)

    dispatch_all = ""
    declarations = cpp_file_header + "#pragma once\n"
    # declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include "../{impl_file}"\n"""

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"\n// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        declarations += "\n\n"

    # declarations += f"#endif // {disable_def}\n"

    autogen_dir = Path(__file__).parent / "test_kernels"

    (autogen_dir / f"{family_name}.h").write_text(declarations)

    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        # impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += f"""#include "../../{impl_file}"\n"""
        for k in f_kernels:
            impl_cu += k.cpp_impl
        # impl_cu += f"#endif // {disable_def}\n"
        (autogen_dir / "impl" / f"{family_name}_{f}.cu").write_text(impl_cu)


write_decl_impl(
    DualGemmFwdKernel.get_all(),
    "cutlass_dual_gemm_fwd",
    impl_file="cutlass_dual_gemm.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD",
)
