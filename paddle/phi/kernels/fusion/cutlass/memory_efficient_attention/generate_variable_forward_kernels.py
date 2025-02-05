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

KERNEL_IMPL_TEMPLATE = """

void  {NAME}({CPP_CLASS} default_fmha, Params &params, const phi::GPUContext& ctx) {{
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

  phi::Allocator::AllocationPtr problem_sizes_device0{{nullptr}};
  phi::Allocator::AllocationPtr problem_sizes_device1{{nullptr}};
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
      params.pre_cache_length,
      params.num_heads,
      params.head_size,
      params.value_head_size);
  phi::memory_utils::Copy(phi::CPUPlace(),
                       problem_sizes1.data(),
                       ctx.GetPlace(),
                       problem1_device,
                       sizeof(GemmCoord) * problem_count,
                       ctx.stream());
  if (AttentionKernel::kNeedsOutputAccumulatorBuffer) {{
    const int64_t output_size = params.num_batches * params.num_heads *
                                params.query_seq_len * params.value_head_size;
    phi::Allocator::AllocationPtr tmp_output_accum_buffer_ptr{{nullptr}};
    tmp_output_accum_buffer_ptr = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        output_size * sizeof(ElementOAccum),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    params.output_accum_ptr = tmp_output_accum_buffer_ptr->ptr();
  }}
  int threadblock_count =
      FMHA::sufficient(problem_sizes1.data(), problem_count);
  typename FMHA::Arguments args(
      problem0_device,
      problem1_device,
      problem_count,
      threadblock_count,
      params.num_heads,
      params.kv_num_heads,
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
      params.mask_broadcast_head,
      params.scale,
      problem_sizes1.data());

  FMHA fmha;
  cutlass::Status status;
  size_t workspace_size = fmha.get_workspace_size(args);
  phi::DenseTensor workspace;
  workspace.Resize(common::make_ddim({{static_cast<int64_t>(workspace_size)}}));
  ctx.template Alloc<uint8_t>(&workspace);
  status = fmha.initialize(args, workspace.data<uint8_t>());
  if (status != cutlass::Status::kSuccess) {{
    PADDLE_THROW(common::errors::Unimplemented(
        "Failed to initialize CUTLASS Grouped FMHA kernel."));
  }}
  status = fmha.run(ctx.stream());
  if (status != cutlass::Status::kSuccess) {{
    PADDLE_THROW(common::errors::Unimplemented(
        "Failed to run CUTLASS Grouped FMHA kernel."));
  }}
}}
"""


@dataclass(order=True)
class FwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    aligned: bool
    mask_aligned: bool
    dtype: str
    sm_range: tuple[int, int]
    q: int
    k: int
    single_value_iter: bool
    support_mask: bool = True
    dispatch_cond: str | None = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            0 if self.support_mask else 1,
            # Then keep output in RF
            0 if self.single_value_iter else 1,
            self.q,
            0 if self.mask_aligned else 1,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def _mask_aligned_suffix(self) -> str:
        return "ma" if self.mask_aligned else "mua"

    @property
    def _mask_support_suffix(self) -> str:
        return "sm" if self.support_mask else "usm"

    @property
    def _single_value_suffix(self) -> str:
        return "rf" if self.single_value_iter else "urf"

    @property
    def name(self) -> str:
        return f"fmha_cutlassF_variable_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{self._single_value_suffix}_{self._mask_support_suffix}_{self._mask_aligned_suffix}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                "true" if self.mask_aligned else "false",
                str(self.q),
                str(self.k),
                "true" if self.single_value_iter else "false",
                "cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly",
                "true" if self.support_mask else "false",
            ]
        )
        return f"cutlass::gemm::kernel::DefaultFMHAGrouped<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}_{self._mask_support_suffix}_{self._mask_aligned_suffix}_{self._single_value_suffix}_{self.q}x{self.k}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class, NAME=self.name
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
                for support_mask, mask_aligned in [
                    (False, False),
                    (True, False),
                    (True, True),
                ]:
                    kernels.append(
                        cls(
                            aligned=aligned,
                            dtype=dtype,
                            sm_range=(sm, sm_max),
                            q=q,
                            k=k,
                            single_value_iter=single_value_iter,
                            support_mask=support_mask,
                            mask_aligned=mask_aligned,
                        )
                    )
        return kernels


T = TypeVar("T", bound=FwdKernel)


def write_decl_impl(
    kernels: list[T], family_name: str, impl_file: str, enable_def: str
) -> None:
    cpp_file_header = """// This file is auto-generated. See "generate_variable_forward_kernels.py"
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
        declarations += (
            f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb) {{\n"
        )
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
        declarations += "}\n\n"
        dispatch_all += f"""
    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb);
    }}"""

    declarations += f"""
template <typename PaddleT, typename T>
void dispatch_{family_name}(const ::phi::GPUContext &ctx, T cb) {{
    auto cc = ctx.GetComputeCapability();
    PADDLE_ENFORCE_GE(
        cc,
        70,
        common::errors::InvalidArgument("the Nvidia GPU's Compute Capability must be greater or equal than 70"));

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


def write_main_header():
    main_header_content = f'''
#pragma once

#ifdef {ENABLE_MACRO}

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "cutlass/util/device_memory.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/default_fmha_grouped.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/gemm/gemm_grouped.h"

namespace phi {{

using GemmCoord = cutlass::gemm::GemmCoord;

struct Params {{
  // meta params
  phi::DataType datatype;

  // [bs, nh, seq_len, dh]
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;

  // and it can be broadcasted in axis0, 1, 2.
  const void* mask_ptr = nullptr;

  const int* seq_lens = nullptr;
  const int* kv_seq_lens = nullptr;

  // Output tensors
  void* output_ptr;  // [num_batches, num_heads, query_seq_len, head_size]
  void* output_accum_ptr =
      nullptr;  // [num_batches, num_heads, query_seq_len, head_size]

  // Scale
  float scale;

  // Dimensions/strides
  int32_t num_batches;
  int32_t num_heads;
  int32_t kv_num_heads;
  int32_t query_seq_len;
  int32_t key_value_seq_len;
  int32_t head_size;
  int32_t value_head_size;

  int64_t ldq;
  int64_t ldk;
  int64_t ldm;
  int64_t ldv;
  int64_t ldo;

  int64_t ElementQ;
  int64_t ElementK;
  int64_t ElementM;
  int64_t ElementV;
  int64_t ElementO;

  bool causal;
  bool mask_broadcast_head;
  int pre_cache_length;
}};

__global__ static void get_problem_sizes(const int* seq_lens,
                                         const int* kv_seq_lens,
                                         GemmCoord* problem_sizes0,
                                         GemmCoord* problem_sizes1,
                                         const int bs,
                                         const int pre_cache_length,
                                         const int num_head,
                                         const int head_size,
                                         const int value_head_size) {{
  int bi = blockIdx.x;
  int hi = threadIdx.x;
  if (bi < bs && hi < num_head) {{
    int id = bi * num_head + hi;
    int m = seq_lens[bi];
    int mkv = kv_seq_lens[bi] + (m == 0 ? 0 : pre_cache_length);
    int k0 = head_size;
    int k1 = value_head_size;
    GemmCoord problem0(m, mkv, k0);
    GemmCoord problem1(m, k1, mkv);
    problem_sizes0[id] = problem0;
    problem_sizes1[id] = problem1;
  }}
}}

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

}} // namespace phi

#include "./cutlass_forward.h"

#endif
'''

    path = Path(args.dst_path) / args.gen_dir
    os.makedirs(path, exist_ok=True)
    path = Path(path) / "memory_efficient_variable_attention.h"
    path.write_text(main_header_content)


forward_impl = "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/memory_efficient_variable_attention.h"

write_main_header()

write_decl_impl(
    FwdKernel.get_all(),
    "cutlass_forward",
    impl_file=forward_impl,
    enable_def=ENABLE_MACRO,
)
