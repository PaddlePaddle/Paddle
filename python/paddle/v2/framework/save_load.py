import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import paddle.v2.framework.framework as framework
import paddle.v2.framework.executor as executor
import paddle.v2.framework.utility as utility
import pickle

import tarfile
import cStringIO
import os


def _save_all_persistable_vars_(folder_path, program=None):
    if program is None:
        program = framework.g_program
    save_program = framework.Program()
    save_block = save_program.global_block()

    save_op_inputs = []
    for var in program.global_block().vars.itervalues():
        if var.desc.persistable(
        ) and var.type == core.VarDesc.VarType.LOD_TENSOR:
            v = save_block.create_var(
                name=var.name, dtype=var.data_type, persistable=True)
            save_op_inputs.append(v)
    save_block.append_op(
        type="save",
        inputs={"X": save_op_inputs},
        attrs={"folderPath": folder_path})

    exe = executor.Executor(core.CPUPlace())
    exe.run(save_program, feed={}, fetch_list=[])


def save_inference_model(folder_path,
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

    model_file_name = folder_path + "/__model__"
    with open(model_file_name, "w") as f:
        pickle.dump({
            "program_desc_str": program.desc.serialize_to_string(),
            "feed_order": feed_order,
            "fetch_order": fetch_order
        }, f, -1)

    # Build another program to hold save_ops
    _save_all_persistable_vars_(folder_path=folder_path, program=program)


def save_checkpoint(folder_path, program=None):
    if program is None:
        program = framework.g_program
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    _save_all_persistable_vars_(folder_path=folder_path, program=program)


def load_inference_model(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("No folder named '%s'.", folder_path)
