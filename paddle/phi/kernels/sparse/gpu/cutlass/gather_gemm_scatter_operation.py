class EmitGatherGemmScatterInstance(EmitGemmInstance):
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
    def instance_template(self):
        return ""

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

class EmitGatherGemmScatterConfigurationLibrary:
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
