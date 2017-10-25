import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import paddle.v2.framework.framework as framework
import paddle.v2.framework.executor as executor
import paddle.v2.framework.utility as utility
import pickle

import tarfile
import cStringIO
import os


def _dump_all_persistable_vars_(folder_path, program=None, exempt_list=None):
    if program is None:
        program = framework.g_program
    dump_program = framework.Program()
    dump_block = dump_program.global_block()
    if exempt_list is None:
        exempt_list = {}

    save_op_inputs = []
    for var in program.global_block().vars.itervalues():
        if var.desc.persistable() and not var.name in exempt_list:
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
                         feeded_var_names,
                         target_vars,
                         program=None):
    if program is None:
        program = framework.g_program
    if not isinstance(target_vars, list):
        target_vars = [target_vars]

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    block = program.global_block()
    # add feed/fetch vars and ops to block
    feed_order = utility.add_feed_components(block, feeded_var_names, "feed")
    fetch_order = utility.add_fetch_components(block, target_vars, "fetch")
    # label target op
    for var in target_vars:
        var.op.mark_as_target()

    dump_list = [(program, "__program_desc__", "cpp"),
                 (feed_order, "__feed_order__", "python"),
                 (fetch_order, "__fetch_order__", "python")]
    model_file_name = folder_path + "/model.tar"
    with open(model_file_name, "w") as f:
        tar = tarfile.TarFile(fileobj=f, mode="w")
        for to_dump in dump_list:
            if to_dump[2] == "cpp":
                binary_str = to_dump[0].desc.serialize_to_string()
            elif to_dump[2] == "python":
                binary_str = pickle.dumps(to_dump[0])
            else:
                raise ValueError("Unknown dump type: '%s'", to_dump[2])
            buf = cStringIO.StringIO()
            buf.write(binary_str)
            tarinfo = tarfile.TarInfo(name=to_dump[1])
            buf.seek(0)
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo=tarinfo, fileobj=buf)

    # Build another program to hold save_ops
    _dump_all_persistable_vars_(
        folder_path=folder_path, program=program,
        exempt_list={"feed", "fetch"})


def dump_checkpoint(folder_path, program=None):
    if program is None:
        program = framework.g_program
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    _dump_all_persistable_vars_(folder_path=folder_path, program=program)


def load_inference_model(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("No folder named '%s'.", folder_path)
