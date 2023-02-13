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

import os.path
import shutil

from gemm_operation import EmitGemmConfigurationLibrary
from library import (
    GeneratorTarget,
    OperationKind,
    OperationKindNames,
    SubstituteTemplate,
)


class EmitOperationKindLibrary:
    def __init__(self, generated_path, kind, args):
        self.generated_path = generated_path
        self.kind = kind
        self.args = args
        self.emitters = {OperationKind.Gemm: EmitGemmConfigurationLibrary}

        self.configurations = []

        self.header_template = '''#pragma once\n#ifdef PADDLE_WITH_CUTLASS\n'''
        self.entry_template = ""
        self.configuration_prototype_template = ""
        self.configuration_template = ""

        self.epilogue_template = '''#endif'''

    #
    def __enter__(self):
        self.operation_path = os.path.join(
            self.generated_path, OperationKindNames[self.kind]
        )
        os.mkdir(self.operation_path)

        self.top_level_path = os.path.join(
            self.operation_path,
            "all_%s_operations.h" % OperationKindNames[self.kind],
        )

        self.top_level_file = open(self.top_level_path, "w")
        self.top_level_file.write(self.header_template)

        self.source_files = [
            self.top_level_path,
        ]

        return self

    #
    def emit(self, configuration_name, operations):

        with self.emitters[self.kind](
            self.operation_path, configuration_name
        ) as configuration_emitter:
            for operation in operations:
                configuration_emitter.emit(operation)

            self.source_files.append(configuration_emitter.configuration_path)

        self.configurations.append(configuration_name)
        self.top_level_file.write(
            '#include "'
            + self.operation_path
            + '/'
            + configuration_name
            + '.h"\n'
        )

    #
    def __exit__(self, exception_type, exception_value, traceback):
        self.top_level_file.write(
            SubstituteTemplate(
                self.entry_template,
                {'operation_name': OperationKindNames[self.kind]},
            )
        )

        for configuration_name in self.configurations:
            self.top_level_file.write(
                SubstituteTemplate(
                    self.configuration_template,
                    {'configuration_name': configuration_name},
                )
            )

        self.top_level_file.write(self.epilogue_template)
        self.top_level_file.close()


class Manifest:

    #
    def __init__(self, args=None):
        self.operations = {}
        self.args = args
        self.operation_count = 0
        self.operations_by_name = {}

        self.kernel_filter = ''
        self.kernel_filter_list = []
        self.kernel_names = []
        self.operations_enabled = []
        self.selected_kernels = []
        self.ignore_kernel_names = []
        self.compute_capabilities = [
            50,
        ]
        self.curr_build_dir = '.'
        self.filter_by_cc = True

        if self.args:
            self.kernel_filter = self.args.kernels
            self.curr_build_dir = args.curr_build_dir

            architectures = (
                args.architectures.split(';')
                if len(args.architectures)
                else [
                    '50',
                ]
            )
            architectures = [x if x != '90a' else '90' for x in architectures]

            self.compute_capabilities = [int(x) for x in architectures]

            if args.filter_by_cc in ['false', 'False', '0']:
                self.filter_by_cc = False

        if args.operations == 'all':
            self.operations_enabled = []
        else:
            operations_list = [
                OperationKind.Gemm,
                OperationKind.Conv2d,
                OperationKind.Conv3d,
                OperationKind.RankK,
                OperationKind.Trmm,
                OperationKind.Symm,
            ]
            self.operations_enabled = [
                x
                for x in operations_list
                if OperationKindNames[x] in args.operations.split(',')
            ]

        if args.kernels == 'all':
            self.kernel_names = []
        else:
            self.kernel_names = [x for x in args.kernels.split(',') if x != '']

        self.ignore_kernel_names = [
            x for x in args.ignore_kernels.split(',') if x != ''
        ]

        if args.kernel_filter_file is None:
            self.kernel_filter_list = []
        else:
            self.kernel_filter_list = self.get_kernel_filters(
                args.kernel_filter_file
            )

        self.operation_count = 0
        self.operations_by_name = {}
        self.disable_full_archs_compilation = (
            args.disable_full_archs_compilation
        )

    def append(self, operation):
        '''
        Inserts the operation.

        operation_kind -> configuration_name -> []
        '''

        self.selected_kernels.append(operation.procedural_name())

        self.operations_by_name[operation.procedural_name()] = operation

        # add the configuration
        configuration_name = operation.configuration_name()

        if operation.operation_kind not in self.operations.keys():
            self.operations[operation.operation_kind] = {}

        if (
            configuration_name
            not in self.operations[operation.operation_kind].keys()
        ):
            self.operations[operation.operation_kind][configuration_name] = []

            self.operations[operation.operation_kind][
                configuration_name
            ].append(operation)
            self.operation_count += 1

    #

    #
    def emit(self, target=GeneratorTarget.Library):

        operation_emitters = {GeneratorTarget.Library: EmitOperationKindLibrary}

        generated_path = os.path.join(self.curr_build_dir, 'generated')

        # create generated/
        if os.path.exists(generated_path):
            shutil.rmtree(generated_path)

        os.mkdir(generated_path)

        source_files = []

        # for each operation kind, emit initializer for all configurations
        for operation_kind, configurations in self.operations.items():
            with operation_emitters[target](
                generated_path, operation_kind, self.args
            ) as operation_kind_emitter:
                for configuration_name, operations in configurations.items():
                    operation_kind_emitter.emit(configuration_name, operations)

                source_files += operation_kind_emitter.source_files
