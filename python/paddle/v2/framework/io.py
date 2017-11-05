import os
import cPickle as pickle

from paddle.v2.framework.framework import Program, Parameter, g_main_program, \
    Variable

__all__ = [
    'save_vars', 'save_params', 'save_persistables', 'load_vars', 'load_params',
    'load_persistables', "save_inference_model", "load_inference_model"
]


def is_parameter(var):
    return isinstance(var, Parameter)


def is_persistable(var):
    return var.persistable


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.data_type,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


def save_vars(executor, dirname, main_program=None, vars=None, predicate=None):
    """
    Save variables to directory by executor.

    :param executor: executor that save variable
    :param dirname: directory path
    :param main_program: program. If vars is None, then filter all variables in this 
    program which fit `predicate`. Default g_program.
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the variables will be saved.
    :param vars: variables need to be saved. If specify vars, program & predicate
    will be ignored
    :return: None
    """
    if vars is None:
        if main_program is None:
            main_program = g_main_program
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        save_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, main_program.list_vars()))
    else:
        save_program = Program()
        save_block = save_program.global_block()
        for each_var in vars:
            new_var = _clone_var_in_block_(save_block, each_var)
            save_block.append_op(
                type='save',
                inputs={'X': [new_var]},
                outputs={},
                attrs={'file_path': os.path.join(dirname, new_var.name)})
        executor.run(save_program)


def save_params(executor, dirname, main_program=None):
    """
    Save all parameters to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_parameter)


def save_persistables(executor, dirname, main_program=None):
    """
    Save all persistables to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_persistable)


def load_vars(executor, dirname, main_program=None, vars=None, predicate=None):
    """
    Load variables from directory by executor.

    :param executor: executor that save variable
    :param dirname: directory path
    :param main_program: program. If vars is None, then filter all variables in this 
    program which fit `predicate`. Default g_program.
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the variables will be loaded.
    :param vars: variables need to be loaded. If specify vars, program & 
    predicate will be ignored
    :return: None
    """
    if vars is None:
        if main_program is None:
            main_program = g_main_program
        if not isinstance(main_program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, main_program.list_vars()))
    else:
        load_prog = Program()
        load_block = load_prog.global_block()
        for each_var in vars:
            assert isinstance(each_var, Variable)
            new_var = _clone_var_in_block_(load_block, each_var)
            load_block.append_op(
                type='load',
                inputs={},
                outputs={"Out": [new_var]},
                attrs={'file_path': os.path.join(dirname, new_var.name)})

        executor.run(load_prog)


def load_params(executor, dirname, main_program=None):
    """
    load all parameters from directory by executor.
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_parameter)


def load_persistables(executor, dirname, main_program=None):
    """
    load all persistables from directory by executor.
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_persistable)


def save_inference_model(dirname,
                         feeded_var_names,
                         target_vars,
                         executor,
                         main_program=None):
    """
    Build a model especially for inference, 
    and save it to directory by the executor.

    :param dirname: directory path
    :param feeded_var_names: Names of variables that need to be feeded data during inference
    :param target_vars: Variables from which we can get inference results.
    :param executor: executor that save inference model
    :param main_program: original program, which will be pruned to build the inference model. 
    Default g_program.

    :return: None
    """
    if main_program is None:
        main_program = g_main_program
    if not isinstance(target_vars, list):
        target_vars = [target_vars]

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    pruned_program = main_program.prune(target_vars)
    fetch_var_names = [v.name for v in target_vars]

    model_file_name = dirname + "/__model__"
    with open(model_file_name, "w") as f:
        pickle.dump({
            "program_desc_str": pruned_program.desc.serialize_to_string(),
            "feed_var_names": feeded_var_names,
            "fetch_var_names": fetch_var_names
        }, f, -1)

    save_params(executor, dirname, main_program)


def load_persistables_if_exist(executor, dirname, main_program=None):
    filenames = next(os.walk(dirname))[2]
    filenames = set(filenames)

    def _is_presistable_and_exist_(var):
        if not is_persistable(var):
            return False
        else:
            return var.name in filenames

    load_vars(
        executor,
        dirname,
        main_program=main_program,
        vars=None,
        predicate=_is_presistable_and_exist_)


def load_inference_model(dirname, executor):
    """
    Load inference model from a directory

    :param dirname: directory path
    :param executor: executor that load inference model

    :return: [program, feed_var_names, fetch_var_names]
             program: program especially for inference.
             feeded_var_names: Names of variables that need to feed data
             fetch_vars: Variables from which we can get inference results.
    """
    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    model_file_name = dirname + "/__model__"
    model = pickle.load(open(model_file_name, "r"))
    program_desc_str = model["program_desc_str"]
    feed_var_names = model["feed_var_names"]
    fetch_var_names = model["fetch_var_names"]
    program = Program.parse_from_string(program_desc_str)
    load_persistables_if_exist(executor, dirname, program)
    fetch_vars = [program.global_block().var(name) for name in fetch_var_names]

    return [program, feed_var_names, fetch_vars]
