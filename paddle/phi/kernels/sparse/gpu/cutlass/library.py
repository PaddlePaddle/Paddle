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

#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import re

# The following block implements enum.auto() for Python 3.5 variants that don't include it such
# as the default 3.5.2 on Ubuntu 16.04.
#
# https://codereview.stackexchange.com/questions/177309/reimplementing-pythons-enum-auto-for-compatibility

try:
    from enum import auto as enum_auto
except ImportError:
    __cutlass_library_auto_enum = 0

    def enum_auto() -> int:
        global __cutlass_library_auto_enum
        i = __cutlass_library_auto_enum
        __cutlass_library_auto_enum += 1
        return i


class GeneratorTarget(enum.Enum):
    Library = enum_auto()


GeneratorTargetNames = {GeneratorTarget.Library: 'library'}


class DataType(enum.Enum):
    b1 = enum_auto()
    u4 = enum_auto()
    u8 = enum_auto()
    u16 = enum_auto()
    u32 = enum_auto()
    u64 = enum_auto()
    s4 = enum_auto()
    s8 = enum_auto()
    s16 = enum_auto()
    s32 = enum_auto()
    s64 = enum_auto()
    e4m3 = enum_auto()
    e5m2 = enum_auto()
    f16 = enum_auto()
    bf16 = enum_auto()
    f32 = enum_auto()
    tf32 = enum_auto()
    f64 = enum_auto()
    invalid = enum_auto()


ShortDataTypeNames = {
    DataType.s32: 'i',
    DataType.e4m3: 'e4m3',
    DataType.e5m2: 'e5m2',
    DataType.f16: 'h',
    DataType.f32: 's',
    DataType.f64: 'd',
}

DataTypeNames = {
    DataType.b1: "b1",
    DataType.u4: "u4",
    DataType.u8: "u8",
    DataType.u16: "u16",
    DataType.u32: "u32",
    DataType.u64: "u64",
    DataType.s4: "s4",
    DataType.s8: "s8",
    DataType.s16: "s16",
    DataType.s32: "s32",
    DataType.s64: "s64",
    DataType.e4m3: 'e4m3',
    DataType.e5m2: 'e5m2',
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.tf32: "tf32",
    DataType.f64: "f64",
}

DataTypeTag = {
    DataType.b1: "cutlass::uint1b_t",
    DataType.u4: "cutlass::uint4b_t",
    DataType.u8: "uint8_t",
    DataType.u16: "uint16_t",
    DataType.u32: "uint32_t",
    DataType.u64: "uint64_t",
    DataType.s4: "cutlass::int4b_t",
    DataType.s8: "int8_t",
    DataType.s16: "int16_t",
    DataType.s32: "int32_t",
    DataType.s64: "int64_t",
    DataType.e4m3: 'cutlass::float_e4m3_t',
    DataType.e5m2: 'cutlass::float_e5m2_t',
    DataType.f16: "cutlass::half_t",
    DataType.bf16: "cutlass::bfloat16_t",
    DataType.f32: "float",
    DataType.tf32: "cutlass::tfloat32_t",
    DataType.f64: "double",
}

DataTypeSize = {
    DataType.b1: 1,
    DataType.u4: 4,
    DataType.u8: 8,
    DataType.u16: 16,
    DataType.u32: 32,
    DataType.u64: 64,
    DataType.s4: 4,
    DataType.s8: 8,
    DataType.s16: 16,
    DataType.s32: 32,
    DataType.s64: 64,
    DataType.e4m3: 8,
    DataType.e5m2: 8,
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.tf32: 32,
    DataType.f64: 64,
}


class ComplexTransform(enum.Enum):
    none = enum_auto()
    conj = enum_auto()


ComplexTransformTag = {
    ComplexTransform.none: 'cutlass::ComplexTransform::kNone',
    ComplexTransform.conj: 'cutlass::ComplexTransform::kConjugate',
}


class MathOperation(enum.Enum):
    multiply_add = enum_auto()
    multiply_add_saturate = enum_auto()
    xor_popc = enum_auto()
    multiply_add_fast_bf16 = enum_auto()
    multiply_add_fast_f16 = enum_auto()
    multiply_add_fast_f32 = enum_auto()


MathOperationTag = {
    MathOperation.multiply_add: 'cutlass::arch::OpMultiplyAdd',
    MathOperation.multiply_add_saturate: 'cutlass::arch::OpMultiplyAddSaturate',
    MathOperation.multiply_add_fast_bf16: 'cutlass::arch::OpMultiplyAddFastBF16',
    MathOperation.multiply_add_fast_f16: 'cutlass::arch::OpMultiplyAddFastF16',
    MathOperation.multiply_add_fast_f32: 'cutlass::arch::OpMultiplyAddFastF32',
}


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()
    ColumnMajorInterleaved2 = enum_auto()
    RowMajorInterleaved2 = enum_auto()
    ColumnMajorInterleaved32 = enum_auto()
    RowMajorInterleaved32 = enum_auto()
    ColumnMajorInterleaved64 = enum_auto()
    RowMajorInterleaved64 = enum_auto()


LayoutTag = {
    LayoutType.ColumnMajor: 'cutlass::layout::ColumnMajor',
    LayoutType.RowMajor: 'cutlass::layout::RowMajor',
    LayoutType.ColumnMajorInterleaved2: 'cutlass::layout::ColumnMajorInterleaved<2>',
    LayoutType.RowMajorInterleaved2: 'cutlass::layout::RowMajorInterleaved<2>',
    LayoutType.ColumnMajorInterleaved32: 'cutlass::layout::ColumnMajorInterleaved<32>',
    LayoutType.RowMajorInterleaved32: 'cutlass::layout::RowMajorInterleaved<32>',
    LayoutType.ColumnMajorInterleaved64: 'cutlass::layout::ColumnMajorInterleaved<64>',
    LayoutType.RowMajorInterleaved64: 'cutlass::layout::RowMajorInterleaved<64>',
}

TransposedLayout = {
    LayoutType.ColumnMajor: LayoutType.RowMajor,
    LayoutType.RowMajor: LayoutType.ColumnMajor,
    LayoutType.ColumnMajorInterleaved2: LayoutType.RowMajorInterleaved2,
    LayoutType.RowMajorInterleaved2: LayoutType.ColumnMajorInterleaved2,
    LayoutType.ColumnMajorInterleaved32: LayoutType.RowMajorInterleaved32,
    LayoutType.RowMajorInterleaved32: LayoutType.ColumnMajorInterleaved32,
    LayoutType.ColumnMajorInterleaved64: LayoutType.RowMajorInterleaved64,
    LayoutType.RowMajorInterleaved64: LayoutType.ColumnMajorInterleaved64,
}

ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: 't',
    LayoutType.ColumnMajorInterleaved2: 't2',
    LayoutType.ColumnMajorInterleaved32: 't32',
    LayoutType.ColumnMajorInterleaved64: 't64',
    LayoutType.RowMajor: 'n',
    LayoutType.RowMajorInterleaved2: 'n2',
    LayoutType.RowMajorInterleaved32: 'n32',
    LayoutType.RowMajorInterleaved64: 'n64',
}

ShortComplexLayoutNames = {
    (LayoutType.ColumnMajor, ComplexTransform.none): 't',
    (LayoutType.ColumnMajor, ComplexTransform.conj): 'c',
    (LayoutType.RowMajor, ComplexTransform.none): 'n',
    (LayoutType.RowMajor, ComplexTransform.conj): 'h',
}


class OpcodeClass(enum.Enum):
    Simt = enum_auto()
    TensorOp = enum_auto()
    WmmaTensorOp = enum_auto()
    SparseTensorOp = enum_auto()


OpcodeClassNames = {
    OpcodeClass.Simt: 'simt',
    OpcodeClass.TensorOp: 'tensorop',
    OpcodeClass.WmmaTensorOp: 'wmma_tensorop',
}

OpcodeClassTag = {
    OpcodeClass.Simt: 'cutlass::arch::OpClassSimt',
    OpcodeClass.TensorOp: 'cutlass::arch::OpClassTensorOp',
    OpcodeClass.WmmaTensorOp: 'cutlass::arch::OpClassWmmaTensorOp',
}


class OperationKind(enum.Enum):
    Gemm = enum_auto()


OperationKindNames = {OperationKind.Gemm: 'gemm'}


class Target(enum.Enum):
    library = enum_auto()


ArchitectureNames = {
    50: 'maxwell',
    60: 'pascal',
    61: 'pascal',
    70: 'volta',
    75: 'turing',
    80: 'ampere',
    89: 'ada',
    90: 'hopper',
}

SharedMemPerCC = {
    70: 96,  # 96KB of SMEM
    72: 96,  # 96KB of SMEM
    75: 64,  # 64KB of SMEM
    80: 163,  # 163KB of SMEM - 1KB reserved for the driver
    86: 99,  # 99KB of SMEM - 1KB reserved for the driver
    87: 163,  # 163KB of SMEM - 1KB reserved for the driver
    89: 99,  # 99KB of SMEM - 1KB reserved for the driver
    90: 227,  # 227KB of SMEM - 1KB reserved for the driver
}


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class GemmKind(enum.Enum):
    Universal = enum_auto()


GemmKindNames = {GemmKind.Universal: "gemm"}


class EpilogueFunctor(enum.Enum):
    LinearCombination = enum_auto()
    LinearCombinationClamp = enum_auto()


EpilogueFunctorTag = {
    EpilogueFunctor.LinearCombination: 'cutlass::epilogue::thread::LinearCombination',
    EpilogueFunctor.LinearCombinationClamp: 'cutlass::epilogue::thread::LinearCombinationClamp',
}


class SwizzlingFunctor(enum.Enum):
    Identity1 = enum_auto()
    Identity2 = enum_auto()
    Identity4 = enum_auto()
    Identity8 = enum_auto()
    Horizontal = enum_auto()
    StridedDgradIdentity1 = enum_auto()
    StridedDgradIdentity4 = enum_auto()
    StridedDgradHorizontal = enum_auto()
    StreamK = enum_auto()


SwizzlingFunctorTag = {
    SwizzlingFunctor.Identity1: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>',
    SwizzlingFunctor.Identity2: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>',
    SwizzlingFunctor.Identity4: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>',
    SwizzlingFunctor.Identity8: 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>',
    SwizzlingFunctor.Horizontal: 'cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle',
    SwizzlingFunctor.StridedDgradIdentity1: 'cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>',
    SwizzlingFunctor.StridedDgradIdentity4: 'cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>',
    SwizzlingFunctor.StridedDgradHorizontal: 'cutlass::conv::threadblock::StridedDgradHorizontalThreadblockSwizzle',
    SwizzlingFunctor.StreamK: 'cutlass::gemm::threadblock::ThreadblockSwizzleStreamK',
}


class MathInstruction:
    def __init__(
        self,
        instruction_shape,
        element_a,
        element_b,
        element_accumulator,
        opcode_class,
        math_operation=MathOperation.multiply_add,
    ):
        self.instruction_shape = instruction_shape
        self.element_a = element_a
        self.element_b = element_b
        self.element_accumulator = element_accumulator
        self.opcode_class = opcode_class
        self.math_operation = math_operation


class TileDescription:
    def __init__(
        self,
        threadblock_shape,
        stages,
        warp_count,
        math_instruction,
        min_compute,
        max_compute,
        cluster_shape=[1, 1, 1],
    ):
        self.threadblock_shape = threadblock_shape
        self.stages = stages
        self.warp_count = warp_count
        self.math_instruction = math_instruction
        self.minimum_compute_capability = min_compute
        self.maximum_compute_capability = max_compute
        self.cluster_shape = cluster_shape

    def procedural_name(self):
        if self.minimum_compute_capability >= 90:
            return "{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{s}".format(
                tbm=self.threadblock_shape[0],
                tbn=self.threadblock_shape[1],
                tbk=self.threadblock_shape[2],
                cm=self.cluster_shape[0],
                cn=self.cluster_shape[1],
                ck=self.cluster_shape[2],
                s=self.stages,
            )
        else:
            return "%dx%d_%dx%d" % (
                self.threadblock_shape[0],
                self.threadblock_shape[1],
                self.threadblock_shape[2],
                self.stages,
            )


class TensorDescription:
    def __init__(
        self,
        element,
        layout,
        alignment=1,
        complex_transform=ComplexTransform.none,
    ):
        self.element = element
        self.layout = layout
        self.alignment = alignment
        self.complex_transform = complex_transform


def CalculateSmemUsage(operation):
    cta_shape = operation.tile_description.threadblock_shape
    stages = operation.tile_description.stages

    if (
        operation.operation_kind == OperationKind.Gemm
        and operation.gemm_kind == GemmKind.Sparse
    ):
        # Elements represented by 8 bits of metadata (based on 4:8, 2:4 or 1:2 sparsity)
        if DataTypeSize[operation.A.element] == 32:
            elements_per_8b_md = 2
        elif DataTypeSize[operation.A.element] == 4:
            elements_per_8b_md = 8
        else:
            elements_per_8b_md = 4

        smem_per_stage = (
            DataTypeSize[operation.A.element]
            * cta_shape[0]
            * (cta_shape[2] // 2)
            // 8
            + DataTypeSize[operation.B.element]
            * cta_shape[1]
            * cta_shape[2]
            // 8
            + cta_shape[0] * (cta_shape[2] // 2) // elements_per_8b_md
        )
    else:
        # Few BLAS3 operations only have A tensor
        smem_per_stage = (
            DataTypeSize[operation.A.element] * cta_shape[0] * cta_shape[2] // 8
            + DataTypeSize[operation.A.element]
            * cta_shape[1]
            * cta_shape[2]
            // 8
        )

    smem_usage = smem_per_stage * stages
    return smem_usage >> 10
