cdef extern from 'paddle/c/scope.h':
    ctypedef void* paddle_scope_handle
    ctypedef int paddle_error
    cdef const paddle_error PADDLE_OK = 0
    paddle_scope_handle paddle_new_scope()
    paddle_error paddle_new_scope_with_parent(paddle_scope_handle parent,
                                              paddle_scope_handle* self)
    paddle_error paddle_destroy_scope(paddle_scope_handle)

    ctypedef void* paddle_variable_handle
    paddle_error paddle_destroy_variable(paddle_variable_handle var)
    paddle_error paddle_scope_get_var(paddle_scope_handle scope,
                                  const char* name,
                                  paddle_variable_handle* var)
    paddle_error paddle_scope_create_var(paddle_scope_handle scope,
                                  const char* name,
                                  paddle_variable_handle* var)



class PaddleCAPIError(BaseException):
    def __init__(self, errno):
        self.errno = errno

cdef class Variable(object):
    cdef paddle_variable_handle __handle__

    cdef set_handle(self, paddle_variable_handle hdl):
        self.__handle__ = hdl

    def __cinit__(self):
        self.__handle__ = NULL

    def __dealloc__(self):
        if self.__handle__ is not NULL:
            paddle_destroy_variable(self.__handle__)


cdef class Scope(object):
    cdef paddle_scope_handle __handle__
    def __cinit__(self):
        self.__handle__ = NULL

    def __init__(self, Scope parent=None):
        if parent is None:
            self.__handle__ = paddle_new_scope()
        elif not isinstance(parent, Scope):
            raise TypeError("Parent must be a scope")
        else:
            err = paddle_new_scope_with_parent(parent.__handle__, &self.__handle__)
            if err != PADDLE_OK:
                raise PaddleCAPIError(err)

    def get_var(self, name):
        cdef paddle_variable_handle var_handle = NULL
        err = paddle_scope_get_var(self.__handle__, name, &var_handle)
        if err != PADDLE_OK:
            raise PaddleCAPIError(err)
        if var_handle == NULL:
            return None
        else:
            var = Variable()
            var.set_handle(var_handle)
            return var

    def create_var(self, name):
        cdef paddle_variable_handle var_handle = NULL
        err = paddle_scope_create_var(self.__handle__, name, &var_handle)
        if err != PADDLE_OK:
            raise PaddleCAPIError(err)
        var = Variable()
        var.set_handle(var_handle)
        return var

    def __dealloc__(self):
        if self.__handle__ is not NULL:
            paddle_destroy_scope(self.__handle__)
