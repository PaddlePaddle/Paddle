"""
Default scope function.

`Paddle` manages Scope as programming language's scope.  It just a 
thread-local stack of Scope. Top of that stack is current scope, the bottom 
of that stack is all scopes' parent. 

Invoking `var/find_var`  can `new/find` variable in current scope. 
Invoking `enter_local_scope/leave_local_scope` can create or destroy local 
scope. 

A `scoped_function` will take a `function` as input. That function will be 
invoked in a new local scope. 
"""

import paddle.v2.fluid.core
import threading

__tl_scope__ = threading.local()

__all__ = [
    'get_cur_scope', 'enter_local_scope', 'leave_local_scope', 'var',
    'find_var', 'scoped_function'
]


def get_cur_scope():
    """
    Get current scope.
    :rtype: paddle.v2.fluid.core.Scope
    """
    cur_scope_stack = getattr(__tl_scope__, 'cur_scope', None)
    if cur_scope_stack is None:
        __tl_scope__.cur_scope = list()
    if len(__tl_scope__.cur_scope) == 0:
        __tl_scope__.cur_scope.append(paddle.v2.fluid.core.Scope())
    return __tl_scope__.cur_scope[-1]


def enter_local_scope():
    """
    Enter a new local scope
    """
    cur_scope = get_cur_scope()
    new_scope = cur_scope.new_scope()
    __tl_scope__.cur_scope.append(new_scope)


def leave_local_scope():
    """
    Leave local scope
    """
    __tl_scope__.cur_scope.pop()
    get_cur_scope().drop_kids()


def var(name):
    """
    create variable in current scope.
    """
    return get_cur_scope().var(name)


def find_var(name):
    """
    get variable in current scope.
    """
    return get_cur_scope().find_var(name)


def scoped_function(func):
    """
    invoke `func` in new scope.
    
    :param func: a callable function that will be run in new scope.
    :type func: callable
    """
    enter_local_scope()
    try:
        func()
    except:
        raise
    finally:
        leave_local_scope()
