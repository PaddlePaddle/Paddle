# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Generates combination of kernels - implementations and registry

# Kernels are ordered (see `sort_index`), and when dispatching,
# we select the first kernel in the list that supports the inputs

from __future__ import annotations

import argparse
import collections
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

DEFAULT_ARCH = [50, 70, 75, 80]
MAX_ARCH = 90
ENABLE_MACRO = "PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION"

assert sorted(DEFAULT_ARCH) == DEFAULT_ARCH


def find_arch_range(min_arch, max_arch):
    assert min_arch >= DEFAULT_ARCH[0] and min_arch <= MAX_ARCH
    assert max_arch >= DEFAULT_ARCH[0] and max_arch <= MAX_ARCH
    assert min_arch <= max_arch
    n = len(DEFAULT_ARCH)

    start_idx = n - 1
    for i in range(n - 1):
        if DEFAULT_ARCH[i] <= min_arch and min_arch < DEFAULT_ARCH[i + 1]:
            start_idx = i
            break

    end_idx = n
    for i in range(n - 1):
        if DEFAULT_ARCH[i] <= max_arch and max_arch < DEFAULT_ARCH[i + 1]:
            end_idx = i + 1

    return DEFAULT_ARCH[start_idx:end_idx]


def find_max_arch(arch):
    arch = sorted(arch)
    idx = DEFAULT_ARCH.index(arch[-1])
    if idx == len(DEFAULT_ARCH) - 1:
        return MAX_ARCH
    else:
        return DEFAULT_ARCH[idx + 1]


def convert_to_arch_list(arch):
    arch = arch.lower().strip()
    if arch == "all":
        return DEFAULT_ARCH

    arch = [int(s.strip()) for s in arch.split(';') if s.strip()]
    arch = list(set(arch))
    arch.sort()
    return find_arch_range(arch[0], arch[-1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="The argument for generating the memory efficient kernels."
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default=str(Path(__file__).parent),
        help="The destination path to save the generated files.",
    )
    parser.add_argument(
        "--cuda_arch",
        type=convert_to_arch_list,
        default=convert_to_arch_list("All"),
        help="The CUDA architecture to be generated.",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="autogen_variable",
        help="The directory to save the generated files.",
    )
    args = parser.parse_args()
    args.max_arch = find_max_arch(args.cuda_arch)
    return args


args = parse_args()

DTYPES = {
    "f32": "float",
    "f16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = args.cuda_arch

KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0
#if __CUDA_ARCH__ < {SM_MAX}0
  if (!p.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""


@dataclass(order=True)
class FwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    aligned: bool
    dtype: str
    sm_range: tuple[int, int]
    q: int
    k: int
    single_value_iter: bool
    supports_dropout: bool = True
    supports_bias: bool = True
    dispatch_cond: str | None = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Then keep output in RF
            0 if self.single_value_iter else 1,
            self.k,
            # Prefer kernels without dropout/bias if available
            1 if self.supports_dropout else 0,
            1 if self.supports_bias else 0,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        acc = "rf" if self.single_value_iter else "gmem"
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{acc}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                "true" if self.single_value_iter else "false",
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> list[FwdKernel]:
        kernels: list[FwdKernel] = []
        for aligned, dtype, (sm, sm_max) in itertools.product(
            [True, False], DTYPES.keys(), zip(SM, SM[1:] + [args.max_arch])
        ):
            # Remove some kernels we don't use
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            for q, k, single_value_iter in [
                (32, 128, True),
                (32, 128, False),
                (64, 64, True),
            ]:
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        q=q,
                        k=k,
                        single_value_iter=single_value_iter,
                    )
                )
        return kernels


@dataclass(order=True)
class BwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    sm_range: tuple[int, int]
    dtype: str
    aligned: bool
    apply_dropout: bool
    preload_mmas: bool
    block_i: int
    block_j: int
    max_k: int
    dispatch_cond: str | None = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Take a kernel without dropout if possible
            1 if self.apply_dropout else 0,
            # Then take the smallest maxK
            self.max_k,
            # .. and the highest block_i
            -self.block_i,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return (
            f"fmha_cutlassB_{self.dtype}_{self._aligned_suffix}"
            f"_{self.block_i}x{self.block_j}_k{self.max_k}{dropout_suffix}_sm{self.sm_range[0]}"
        )

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                f"cutlass::arch::Sm{self.sm_range[0]}",
                DTYPES[self.dtype],
                "true" if self.aligned else "false",
                "true" if self.apply_dropout else "false",
                "true" if self.preload_mmas else "false",
                str(self.block_i),
                str(self.block_j),
                str(self.max_k),
            ]
        )
        return f"AttentionBackwardKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return (
            f"{self.dtype}_{self._aligned_suffix}_k{self.max_k}{dropout_suffix}"
        )

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> list[BwdKernel]:
        kernels: list[BwdKernel] = []
        for (
            aligned,
            dtype,
            (sm, sm_max),
            apply_dropout,
            max_k,
        ) in itertools.product(
            [True, False],
            DTYPES.keys(),
            zip(SM, SM[1:] + [args.max_arch]),
            [True, False],
            [32, 64, 128, 2**16],
        ):
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            is_half = dtype in ["bf16", "f16"]

            bi_values = [64]
            # Some architectures have more shmem and can use 128
            # We still need fallback to 64 for GPUs with less shmem
            # (Sm75, Sm86 ...)
            if sm >= 80 or (sm >= 70 and is_half):
                if max_k > 64:
                    bi_values.append(128)
            for bi in bi_values:
                output_in_rf = is_half and max_k <= bi
                preload_mmas = is_half and sm >= 80 and output_in_rf
                bj = 128 if (preload_mmas and max_k > 64) else 64
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=bi,
                        block_j=bj,
                        max_k=max_k,
                    )
                )
        # Add some specialized kernels for stable diffusion BW (K=80)
        # This is the only kernel that can keep the outputs on RF on
        # Sm86/Sm89, so it's much faster than the 64x64 one
        for dtype in ["f16", "bf16"]:
            if max(args.cuda_arch) < 80:
                continue
            kernels.append(
                cls(
                    aligned=True,
                    dtype=dtype,
                    sm_range=(80, MAX_ARCH),
                    apply_dropout=False,
                    preload_mmas=True,
                    block_i=128,
                    block_j=64,
                    max_k=96,
                    # Sm80 has a faster kernel for this case
                    dispatch_cond="cc == 86 || cc == 89",
                )
            )
        return kernels


T = TypeVar("T", FwdKernel, BwdKernel)


def write_decl_impl(
    kernels: list[T], family_name: str, impl_file: str, enable_def: str
) -> None:
    cpp_file_header = """// This file is auto-generated. See "generate_kernels.py"
"""

    kernels.sort()

    implfile_to_kernels: dict[str, list[T]] = collections.defaultdict(list)
    cat_to_kernels: dict[tuple[str, int, int], list[T]] = (
        collections.defaultdict(list)
    )

    dispatch_all = ""
    declarations = cpp_file_header + "#pragma once\n"
    declarations += f"#ifdef {enable_def}\n"
    declarations += f"""#include "{impl_file}"\n"""
    declarations += "namespace phi {\n"

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        dispatch_category_fn = f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}"
        declarations += f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb, int cc) {{\n"
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
        declarations += "}\n\n"
        dispatch_all += f"""
    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""

    declarations += f"""
template <typename PaddleT, typename T>
void dispatch_{family_name}(const ::phi::GPUContext &ctx, T cb) {{
    auto cc = ctx.GetComputeCapability();
    using DT = typename ::phi::CutlassTrait<PaddleT>::Type;

{dispatch_all}
}}
"""
    declarations += "} // namespace phi\n"
    declarations += f"#endif // {enable_def}\n"

    autogen_dir = Path(args.dst_path) / args.gen_dir
    os.makedirs(autogen_dir, exist_ok=True)
    declaration_path = autogen_dir / f"{family_name}.h"
    declaration_path.write_text(declarations)

    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        impl_cu += f"#ifdef {enable_def}\n"
        impl_cu += f"""#include "{impl_file}"\n"""
        impl_cu += "namespace phi {\n"
        for k in f_kernels:
            impl_cu += k.cpp_impl
        impl_cu += "} // namespace phi\n"
        impl_cu += f"#endif // {enable_def}\n"
        impl_path = autogen_dir / "impl"
        os.makedirs(impl_path, exist_ok=True)
        (impl_path / f"{family_name}_{f}.cu").write_text(impl_cu)


def write_main_header(forward_impl, backward_impl):
    main_header_content = f'''
#pragma once

#ifdef {ENABLE_MACRO}

#include "{forward_impl}"
#include "{backward_impl}"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {{

template <typename T>
struct CutlassTrait {{
  using Type = T;
}};

template <>
struct CutlassTrait<dtype::float16> {{
  using Type = cutlass::half_t;
}};

template <>
struct CutlassTrait<dtype::bfloat16> {{
  using Type = cutlass::bfloat16_t;
}};


template <typename T>
struct ToPhiDTypeTrait {{
 private:
  using NonConstT = typename std::remove_const<T>::type;
  static constexpr bool kIsFP16 = std::is_same<NonConstT, cutlass::half_t>::value;
  static constexpr bool kIsBF16 = std::is_same<NonConstT, cutlass::bfloat16_t>::value;

 public:
  using Type = typename std::conditional<kIsFP16, dtype::float16,
      typename std::conditional<kIsBF16, dtype::bfloat16, NonConstT>::type>::type;
}};


template <typename T>
T *SafeGetTensorPtr(const DenseTensor &t) {{
  using PDT = typename ToPhiDTypeTrait<T>::Type;
  return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(t.template data<PDT>()));
}}

template <typename T>
T *SafeGetTensorPtr(const DenseTensor *t) {{
  return t ? SafeGetTensorPtr<T>(*t) : nullptr;
}}

template <typename T>
T *SafeGetTensorPtr(const paddle::optional<DenseTensor> &t) {{
  return t ? SafeGetTensorPtr<T>(t.get()) : nullptr;
}}

template <typename T, typename Context>
T *SafeAllocTensor(const Context &ctx, DenseTensor *t) {{
  using PDT = typename ToPhiDTypeTrait<T>::Type;
  void *ptr = ctx.template Alloc<PDT>(t);
  return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(ptr));
}}

inline int64_t DimStride(const phi::DDim &dims, int n) {{
  int rank = dims.size();
  if (n < 0) {{
    n += rank;
  }}
  int64_t stride = 1;
  for (int i = n+1; i < rank; ++i) {{
    stride *= dims[i];
  }}
  return stride;
}}

}} // namespace phi

#include "./cutlass_forward.h"
#include "./cutlass_backward.h"

#endif
'''

    path = Path(args.dst_path) / args.gen_dir
    os.makedirs(path, exist_ok=True)
    path = Path(path) / "memory_efficient_attention.h"
    path.write_text(main_header_content)


forward_impl = "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/kernel_forward.h"
backward_impl = "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/kernel_backward.h"

write_main_header(forward_impl, backward_impl)

write_decl_impl(
    FwdKernel.get_all(),
    "cutlass_forward",
    impl_file=forward_impl,
    enable_def=ENABLE_MACRO,
)
write_decl_impl(
    BwdKernel.get_all(),
    "cutlass_backward",
    impl_file=backward_impl,
    enable_def=ENABLE_MACRO,
)
