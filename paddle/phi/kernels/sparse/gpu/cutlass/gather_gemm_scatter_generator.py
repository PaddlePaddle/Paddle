import sys
import os
sys.path.append(os.getcwd() + "/cutlass")
from manifest import GeneratorTarget
from gather_gemm_scatter_manifest import GatherGemmScatterManifest
from generator import (CudaToolkitVersionSatisfies,
                       CreateGemmOperator,
                       )
from library import (LayoutType,
                     MathInstruction,
                     DataType,
                     OpcodeClass,
                     MathOperation,
                     TileDescription,
                     )
def GenerateSM70_TensorOp_884(manifest, cuda_version):

    if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
        return

    layouts = [
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    ]

    math_instructions = [
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 70
    max_cc = 75

    alignment_constraints = [8, 4, 2, 1]

    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription(
                [256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [256, 64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
            ),
            TileDescription(
                [64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
            ),
        ]

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
        )

        # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
        if math_inst.element_a != math_inst.element_accumulator:

            data_type_mixed = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_a,
                math_inst.element_accumulator,
            ]

            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type_mixed,
                alignment_constraints,
            )


def GenerateSM70(manifest, cuda_version):
    GenerateSM70_TensorOp_884(manifest, cuda_version)


class KernelCfg:
    def __init__(
        self,
        architectures,
        build_dir,
        cuda_version,
        curr_build_dir,
        disable_full_archs_compilation,
        filter_by_cc,
        generator_target,
        ignore_kernels,
        interface_dir,
        kernel_filter_file,
        kernels,
        operations,
        selected_kernel_list,
    ):
        self.architectures = architectures
        self.build_dir = build_dir
        self.cuda_version = cuda_version
        self.curr_build_dir = curr_build_dir
        self.disable_full_archs_compilation = disable_full_archs_compilation
        self.filter_by_cc = filter_by_cc
        self.generator_target = generator_target
        self.ignore_kernels = ignore_kernels
        self.interface_dir = interface_dir
        self.kernel_filter_file = kernel_filter_file
        self.kernels = kernels
        self.operations = operations
        self.selected_kernel_list = selected_kernel_list


###################################################################################################

if __name__ == "__main__":

    args = KernelCfg(
        architectures='70',
        build_dir='/paddle/Paddle/paddle/phi/kernels/sparse/gpu/cutlass/build',
        cuda_version='11.7.64',
        curr_build_dir='/paddle/Paddle/paddle/phi/kernels/sparse/gpu/cutlass/build',
        disable_full_archs_compilation=False,
        filter_by_cc='True',
        generator_target='library',
        ignore_kernels='',
        interface_dir=None,
        kernel_filter_file=None,
        kernels='',
        operations='all',
        selected_kernel_list=None,
    )
    manifest = GatherGemmScatterManifest(args)

    GenerateSM70(manifest, args.cuda_version)

    manifest.emit(GeneratorTarget.Library)
#
###################################################################################################
