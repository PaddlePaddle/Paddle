import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import paddle.v2.framework.framework as framework
import paddle.v2.framework.executor as executor

import tarfile
import cStringIO
import os


def _dump_all_persistable_vars_(folder_path, program=None):
    if program is None:
        program = framework.g_program
    dump_program = framework.Program()
    dump_block = dump_program.global_block()

    save_op_inputs = []
    for var in program.global_block().vars.itervalues():
        if var.desc.persistable():
            v = dump_block.create_var(
                name=var.name, dtype=var.data_type, persistable=True)
            save_op_inputs.append(var)
    dump_block.append_op(
        type="save",
        inputs={"X": save_op_inputs},
        attrs={"folderPath": folder_path})

    exe = executor.Executor(core.CPUPlace())
    exe.run(dump_program, feed={}, fetch_list=[])


def dump_inference_model(folder_path,
                         target_var,
                         program=None,
                         init_program=None):
    if program is None:
        program = framework.g_program
    if init_program is None:
        init_program = framework.g_init_program

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # Dump network topology
    dump_list = [(program, "__program_desc__"),
                 (init_program, "__init_program_desc__"),
                 (target_var, "__target_op_desc__")]
    model_file_name = folder_path + "/model.tar"
    with open(model_file_name, "w") as f:
        tar = tarfile.TarFile(fileobj=f, mode="w")
        for to_dump in dump_list:
            buf = cStringIO.StringIO()
            buf.write(to_dump[0].desc.serialize_to_string())
            tarinfo = tarfile.TarInfo(name=to_dump[1])
            buf.seek(0)
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo=tarinfo, fileobj=buf)

    # Build another program to hold save_ops
    _dump_all_persistable_vars_(folder_path, program)


def dump_checkpoint(folder_path, program=None):
    if program is None:
        program = framework.g_program
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    _dump_all_persistable_vars_(folder_path, program)
