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

import collections
import enum
import os.path

from library import (
    ComplexTransformTag,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    EpilogueFunctor,
    EpilogueFunctorTag,
    GemmKind,
    GemmKindNames,
    LayoutTag,
    LayoutType,
    MathOperation,
    MathOperationTag,
    OpcodeClass,
    OpcodeClassNames,
    OpcodeClassTag,
    OperationKind,
    ShortComplexLayoutNames,
    ShortDataTypeNames,
    ShortLayoutTypeNames,
    SubstituteTemplate,
    SwizzlingFunctor,
    SwizzlingFunctorTag,
)

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################


class GemmOperation:
    #
    def __init__(
        self,
        gemm_kind,
        arch,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue_functor=EpilogueFunctor.LinearCombination,
        swizzling_functor=SwizzlingFunctor.Identity8,
    ):

        self.prefix = ""
        self.operation_kind = OperationKind.Gemm
        self.arch = arch
        self.tile_description = tile_description
        self.gemm_kind = gemm_kind
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor

    #
    def is_complex(self):
        return False

    #
    def is_planar_complex(self):
        return False

    #
    def accumulator_type(self):
        accum = self.tile_description.math_instruction.element_accumulator

        return accum

    #
    def short_math_name(self):
        return ShortDataTypeNames[self.accumulator_type()]

    #
    def core_name(self):
        '''The basic operation kind is prefixed with a letter indicating the accumulation type.'''

        inst_shape = ''
        inst_operation = ''
        intermediate_type = ''

        math_operations_map = {
            MathOperation.xor_popc: 'xor',
        }

        if (
            self.tile_description.math_instruction.opcode_class
            == OpcodeClass.TensorOp
            or self.tile_description.math_instruction.opcode_class
            == OpcodeClass.WmmaTensorOp
        ):

            math_op = self.tile_description.math_instruction.math_operation
            math_op_string = (
                math_operations_map[math_op]
                if math_op in math_operations_map.keys()
                else ''
            )

            inst_shape = "{0}{1}{2}".format(
                *tuple(self.tile_description.math_instruction.instruction_shape)
            )

            inst_shape += math_op_string

            if (
                self.tile_description.math_instruction.element_a
                != self.A.element
                and self.tile_description.math_instruction.element_a
                != self.tile_description.math_instruction.element_accumulator
            ):
                intermediate_type = DataTypeNames[
                    self.tile_description.math_instruction.element_a
                ]

        return "%s%s%s%s" % (
            self.short_math_name(),
            inst_shape,
            intermediate_type,
            GemmKindNames[self.gemm_kind],
        )

    # Generates a string representing the MMA instruction.
    def extended_name(self):
        '''Append data types if they differ from compute type.'''
        if self.is_complex():
            extended_name = "${core_name}"
        else:
            if (
                self.C.element
                != self.tile_description.math_instruction.element_accumulator
                and self.A.element
                != self.tile_description.math_instruction.element_accumulator
            ):
                extended_name = "${element_c}_${core_name}_${element_a}"
            elif (
                self.C.element
                == self.tile_description.math_instruction.element_accumulator
                and self.A.element
                != self.tile_description.math_instruction.element_accumulator
            ):
                extended_name = "${core_name}_${element_a}"
            else:
                extended_name = "${core_name}"

        extended_name = SubstituteTemplate(
            extended_name,
            {
                'element_a': DataTypeNames[self.A.element],
                'element_c': DataTypeNames[self.C.element],
                'core_name': self.core_name(),
            },
        )

        return extended_name

    def extended_name_3x(self):
        '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
        extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}".format(
            element_a=DataTypeNames[self.A.element],
            element_b=DataTypeNames[self.B.element],
            element_acc=DataTypeNames[
                self.tile_description.math_instruction.element_accumulator
            ],
            element_c=DataTypeNames[self.C.element],
            core_name=self.core_name(),
        )
        return extended_name

    # Generates a short string representing the AB layout tags (e.g. nt or tn)
    def layout_name(self):
        if self.is_complex() or self.is_planar_complex():
            return "%s%s" % (
                ShortComplexLayoutNames[
                    (self.A.layout, self.A.complex_transform)
                ],
                ShortComplexLayoutNames[
                    (self.B.layout, self.B.complex_transform)
                ],
            )
        return "%s%s" % (
            ShortLayoutTypeNames[self.A.layout],
            ShortLayoutTypeNames[self.B.layout],
        )

    # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
    def layout_name_3x(self):
        if self.is_complex() or self.is_planar_complex():
            return "{}{}{}".format(
                ShortComplexLayoutNames[
                    (self.A.layout, self.A.complex_transform)
                ],
                ShortComplexLayoutNames[
                    (self.B.layout, self.B.complex_transform)
                ],
                ShortComplexLayoutNames[
                    (self.C.layout, self.C.complex_transform)
                ],
            )
        else:
            return "{}{}{}".format(
                ShortLayoutTypeNames[self.A.layout],
                ShortLayoutTypeNames[self.B.layout],
                ShortLayoutTypeNames[self.C.layout],
            )

    # Generates the full kernel function name
    def procedural_name(self):
        '''The full procedural name indicates architecture, extended name, tile size, and layout.'''
        opcode_class_name = OpcodeClassNames[
            self.tile_description.math_instruction.opcode_class
        ]
        if self.arch >= 90:
            kernel_name_template = "cutlass{p}_sm{ar}_{op}_{ex}_{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{l}_{s}_align{al}"
            return kernel_name_template.format(
                p=self.prefix,
                ar=self.arch,
                op=opcode_class_name,
                ex=self.extended_name_3x(),
                tbm=self.tile_description.threadblock_shape[0],
                tbn=self.tile_description.threadblock_shape[1],
                tbk=self.tile_description.threadblock_shape[2],
                cm=self.tile_description.cluster_shape[0],
                cn=self.tile_description.cluster_shape[1],
                ck=self.tile_description.cluster_shape[2],
                l=self.tile_description.stages,
                s=self.layout_name_3x(),
                al=str(max(self.A.alignment, self.B.alignment)),
            )
        else:
            threadblock = self.tile_description.procedural_name()
            return "cutlass{p}_{op}_{ex}_{tb}_{l}_align{a}".format(
                p=self.prefix,
                op=opcode_class_name,
                ex=self.extended_name(),
                tb=threadblock,
                l=self.layout_name(),
                a=str(self.A.alignment),
            )

    #
    def configuration_name(self):
        '''The full procedural name indicates architecture, extended name, tile size, and layout.'''
        return self.procedural_name()


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmUniversalInstance:
    '''Responsible for emitting a CUTLASS template definition'''

    def __init__(self, operation_suffix=''):
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "cutlass/gemm/device/gemm.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/gemm/kernel/default_gemm_universal.h",
        ]
        self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
        self.gemm_template = """
// Gemm operator ${operation_name}
struct ${operation_name} {
  using Gemm =
    cutlass::gemm::device::GemmUniversal<
      ${element_a},
      ${layout_a}, ${transform_a}, ${align_a},
      ${element_b},
      ${layout_b}, ${transform_b}, ${align_b},
      ${element_c},
      ${layout_c},
      ${element_accumulator},
      ${opcode_class},
      ${arch},
      cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
      cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
      cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
      ${epilogue_functor},
      ${swizzling_functor},
      ${stages},
      ${align_a},
      ${align_b},
      ${math_operation},
      ${transform_a},
      ${transform_b},
      true, // gather a
      false, // gather b
      true // scatter d
    >;
};
"""

    #
    def instance_template(self):
        return ""

    #
    def emit(self, operation):

        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count

        warp_shape = [
            threadblock_shape[idx] // warp_count[idx] for idx in range(3)
        ]

        transpose_layouts = {
            LayoutType.ColumnMajor: LayoutType.ColumnMajor,
            LayoutType.RowMajor: LayoutType.RowMajor,
        }

        if (
            operation.A.layout in transpose_layouts.keys()
            and operation.B.layout in transpose_layouts.keys()
            and operation.C.layout in transpose_layouts.keys()
        ):

            instance_layout_A = transpose_layouts[operation.A.layout]
            instance_layout_B = transpose_layouts[operation.B.layout]
            instance_layout_C = transpose_layouts[operation.C.layout]

            gemm_template = self.gemm_template
        else:
            instance_layout_A, instance_layout_B, instance_layout_C = (
                operation.A.layout,
                operation.B.layout,
                operation.C.layout,
            )

            gemm_template = self.gemm_template_interleaved
        #

        # Support built-in epilogue functors or user-defined functions
        if isinstance(operation.epilogue_functor, enum.Enum):

            epilogue_vector_length = (
                min(
                    operation.C.alignment * DataTypeSize[operation.C.element],
                    128,
                )
                // DataTypeSize[operation.C.element]
            )

            values = {
                'epilogue_vector_length': str(epilogue_vector_length),
                'element_epilogue': str(
                    DataTypeTag[operation.element_epilogue]
                ),
                'epilogue_functor': EpilogueFunctorTag[
                    operation.epilogue_functor
                ],
            }
            epilogue_functor = SubstituteTemplate(
                self.builtin_epilogue_functor_template, values
            )
        else:
            epilogue_functor = self.epilogue_functor.emit_declaration()
        #

        values = {
            'operation_name': operation.procedural_name(),
            'operation_suffix': self.operation_suffix,
            'element_a': DataTypeTag[operation.A.element],
            'layout_a': LayoutTag[instance_layout_A],
            'element_b': DataTypeTag[operation.B.element],
            'layout_b': LayoutTag[instance_layout_B],
            'element_c': DataTypeTag[operation.C.element],
            'layout_c': LayoutTag[instance_layout_C],
            'element_accumulator': DataTypeTag[operation.accumulator_type()],
            'opcode_class': OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            'arch': "cutlass::arch::Sm%d" % operation.arch,
            'threadblock_shape_m': str(
                operation.tile_description.threadblock_shape[0]
            ),
            'threadblock_shape_n': str(
                operation.tile_description.threadblock_shape[1]
            ),
            'threadblock_shape_k': str(
                operation.tile_description.threadblock_shape[2]
            ),
            'warp_shape_m': str(warp_shape[0]),
            'warp_shape_n': str(warp_shape[1]),
            'warp_shape_k': str(warp_shape[2]),
            'instruction_shape_m': str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            'instruction_shape_n': str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            'instruction_shape_k': str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            'epilogue_functor': epilogue_functor,
            'swizzling_functor': SwizzlingFunctorTag[
                operation.swizzling_functor
            ],
            'stages': str(operation.tile_description.stages),
            'align_a': str(operation.A.alignment),
            'align_b': str(operation.B.alignment),
            'transform_a': ComplexTransformTag[operation.A.complex_transform],
            'transform_b': ComplexTransformTag[operation.B.complex_transform],
            'math_operation': MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
        }

        return SubstituteTemplate(gemm_template, values)


###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################


class EmitGemmConfigurationLibrary:
    def __init__(self, operation_path, configuration_name):
        self.configuration_name = configuration_name
        self.configuration_path = os.path.join(
            operation_path, "%s.h" % configuration_name
        ).replace('\\', '/')

        self.instance_emitter = {
            GemmKind.Universal: EmitGemmUniversalInstance,
        }

        self.gemm_kind_wrappers = {
            GemmKind.Universal: 'GemmUniversalOperation',
        }

        self.wmma_guard_start = (
            "#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"
        )

        self.separator = """
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

        self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/
#pragma once
#ifdef PADDLE_WITH_CUTLASS
"""

        self.namespace_template = """
namespace phi {
namespace sparse {
"""
        self.epilogue_template = """
}  // namespace sparse
}  // namespace phi
#endif
"""

    def __enter__(self):
        self.configuration_file = open(self.configuration_path, "w")
        self.configuration_file.write(self.header_template)
        self.configuration_file.write(self.separator)

        self.includes = collections.OrderedDict(
            [
                ("cutlass/cutlass.h", None),
                ("cutlass/library/library.h", None),
                ("cutlass/library/manifest.h", None),
                ("library_internal.h", None),
                ("gemm_operation.h", None),
                ("gemm_operation_3x.hpp", None),
                ("cutlass/arch/wmma.h", None),
                ("cutlass/numeric_types.h", None),
            ]
        )
        self.instance_definitions = []
        self.instance_wrappers = []

        self.operations = []
        return self

    def emit(self, operation):
        emitter = self.instance_emitter[operation.gemm_kind]()

        for incl in emitter.includes:
            self.includes[incl] = None

        self.operations.append(operation)

        self.instance_definitions.append(emitter.emit(operation))

        self.instance_wrappers.append(
            SubstituteTemplate(
                emitter.instance_template(),
                {
                    'configuration_name': self.configuration_name,
                    'operation_name': operation.procedural_name(),
                    'gemm_kind': self.gemm_kind_wrappers[operation.gemm_kind],
                    'compile_guard_start': SubstituteTemplate(
                        self.wmma_guard_start,
                        {'sm_number': str(operation.arch)},
                    )
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                    'compile_guard_end': "#endif"
                    if operation.tile_description.math_instruction.opcode_class
                    == OpcodeClass.WmmaTensorOp
                    else "",
                },
            )
        )

    def __exit__(self, exception_type, exception_value, traceback):

        # Write includes
        for incl, _ in self.includes.items():
            include_statement = "#include \"%s\"\n" % incl
            self.configuration_file.write(include_statement)

        self.configuration_file.write(self.separator)
        self.configuration_file.write(self.namespace_template)

        # Write instance definitions in top-level namespace
        for instance_definition in self.instance_definitions:
            self.configuration_file.write(instance_definition)

        for instance_wrapper in self.instance_wrappers:
            self.configuration_file.write(instance_wrapper)

        self.configuration_file.write(self.epilogue_template)
        self.configuration_file.close()


###################################################################################################
###################################################################################################
