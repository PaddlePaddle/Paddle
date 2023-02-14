from manifest import (EmitOperationKindLibrary,
                      Manifest,
                      )
class GatherGemmScatterEmitOperationKindLibrary(EmitOperationKindLibrary):
    def __init__(self, generated_path, kind, args):
        self.header_template = "#pragma once\n#ifdef PADDLE_WITH_CUTLASS\n"
        self.configuration_prototype_template = ""
        self.configuration_template = ""
        self.epilogue_template = "#endif"
        
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
