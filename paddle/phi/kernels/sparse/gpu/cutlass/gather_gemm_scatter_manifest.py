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

import os
import shutil

from gather_gemm_scatter_operation import (
    EmitGatherGemmScatterConfigurationLibrary,
)
from library import OperationKind, OperationKindNames
from manifest import EmitOperationKindLibrary, GeneratorTarget, Manifest


class GatherGemmScatterEmitOperationKindLibrary(EmitOperationKindLibrary):
    def __init__(self, generated_path, kind, args):
        super().__init__(generated_path, kind, args)
        self.emitters = {
            OperationKind.Gemm: EmitGatherGemmScatterConfigurationLibrary
        }
        self.header_template = "#pragma once\n#ifdef PADDLE_WITH_CUTLASS\n"
        self.entry_template = ""
        self.configuration_prototype_template = ""
        self.configuration_template = ""
        self.epilogue_template = "#endif"

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


class GatherGemmScatterManifest(Manifest):
    def emit(self, target=GeneratorTarget.Library):

        operation_emitters = {
            GeneratorTarget.Library: GatherGemmScatterEmitOperationKindLibrary
        }

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
