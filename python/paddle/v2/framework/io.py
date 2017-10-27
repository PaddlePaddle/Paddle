import os

from paddle.v2.framework.framework import Program, Parameter, g_program, \
    Variable

__all__ = [
    'save_vars', 'save_params', 'save_persistables', 'load_vars', 'load_params',
    'load_persistables'
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


def save_vars(executor, dirname, program=None, vars=None, predicate=None):
    """
    Save variables to directory by executor.
    
    :param executor: executor that save variable
    :param dirname: directory path
    :param program: program. If vars is None, then filter all variables in this 
    program which fit `predicate`. Default g_program.
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the variables will be saved.
    :param vars: variables need to be saved. If specify vars, program & predicate
    will be ignored
    :return: None
    """
    if vars is None:
        if program is None:
            program = g_program
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type or None")

        save_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, program.list_vars()))
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


def save_params(executor, dirname, program=None):
    """
    Save all parameters to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        program=program,
        vars=None,
        predicate=is_parameter)


def save_persistables(executor, dirname, program=None):
    """
    Save all persistables to directory with executor.
    """
    save_vars(
        executor,
        dirname=dirname,
        program=program,
        vars=None,
        predicate=is_persistable)


def load_vars(executor, dirname, program=None, vars=None, predicate=None):
    """
    Load variables from directory by executor.
    
    :param executor: executor that save variable
    :param dirname: directory path
    :param program: program. If vars is None, then filter all variables in this 
    program which fit `predicate`. Default g_program.
    :param predicate: The Predicate describes a callable that returns a variable
    as a bool. If it returns true, the variables will be loaded.
    :param vars: variables need to be loaded. If specify vars, program & 
    predicate will be ignored
    :return: None
    """
    if vars is None:
        if program is None:
            program = g_program
        if not isinstance(program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=dirname,
            vars=filter(predicate, program.list_vars()))
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


def load_params(executor, dirname, program=None):
    """
    load all parameters from directory by executor.
    """
    load_vars(
        executor, dirname=dirname, program=program, predicate=is_parameter)


def load_persistables(executor, dirname, program=None):
    """
    load all persistables from directory by executor.
    """
    load_vars(
        executor, dirname=dirname, program=program, predicate=is_persistable)
