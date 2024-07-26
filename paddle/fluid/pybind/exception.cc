/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/exception.h"

#include "paddle/common/exception.h"
#include "paddle/fluid/memory/allocation/allocator.h"
namespace paddle::pybind {

/* Paddle Exception mapping rules:
 *   - InvalidArgumentError -> ValueError
 *   - NotFoundError -> RuntimeError
 *   - OutOfRangeError -> IndexError
 *   - AlreadyExistsError -> RuntimeError
 *   - ResourceExhaustedError -> MemoryError
 *   - PreconditionNotMetError -> RuntimeError
 *   - PermissionDeniedError -> RuntimeError
 *   - ExecutionTimeoutError -> RuntimeError
 *   - UnimplementedError -> NotImplementedError
 *   - UnavailableError -> RuntimeError
 *   - FatalError -> SystemError
 *   - ExternalError -> OSError
 *   - INVALID_TYPE -> PyExc_TypeError
 */

void BindException(pybind11::module* m) {
  static pybind11::exception<platform::EOFException> eof(*m, "EOFException");
  static pybind11::exception<platform::EnforceNotMet> exc(*m, "EnforceNotMet");
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const platform::EOFException& e) {
      pybind11::set_error(eof, e.what());
    } catch (const memory::allocation::BadAlloc& e) {
      PyErr_SetString(PyExc_MemoryError, e.what());
    } catch (const platform::EnforceNotMet& e) {
      switch (e.code()) {
        case phi::ErrorCode::INVALID_ARGUMENT:
          PyErr_SetString(PyExc_ValueError, e.what());
          break;
        case phi::ErrorCode::NOT_FOUND:
        case phi::ErrorCode::ALREADY_EXISTS:
        case phi::ErrorCode::PRECONDITION_NOT_MET:
        case phi::ErrorCode::PERMISSION_DENIED:
        case phi::ErrorCode::EXECUTION_TIMEOUT:
        case phi::ErrorCode::UNAVAILABLE:
          PyErr_SetString(PyExc_RuntimeError, e.what());
          break;
        case phi::ErrorCode::OUT_OF_RANGE:
          PyErr_SetString(PyExc_IndexError, e.what());
          break;
        case phi::ErrorCode::RESOURCE_EXHAUSTED:
          PyErr_SetString(PyExc_MemoryError, e.what());
          break;
        case phi::ErrorCode::UNIMPLEMENTED:
          PyErr_SetString(PyExc_NotImplementedError, e.what());
          break;
        case phi::ErrorCode::FATAL:
          PyErr_SetString(PyExc_SystemError, e.what());
          break;
        case phi::ErrorCode::EXTERNAL:
          PyErr_SetString(PyExc_OSError, e.what());
          break;
        case phi::ErrorCode::INVALID_TYPE:
          PyErr_SetString(PyExc_TypeError, e.what());
          break;
        default:
          pybind11::set_error(exc, e.what());
          break;
      }
    }
  });

  m->def("__unittest_throw_exception__", [] {
    PADDLE_THROW(phi::errors::PermissionDenied("This is a test of exception"));
  });
}

void ThrowExceptionToPython(std::exception_ptr p) {
  static PyObject* EOFExceptionException =
      PyErr_NewException("paddle.EOFException", PyExc_Exception, nullptr);
  static PyObject* EnforceNotMetException =
      PyErr_NewException("paddle.EnforceNotMet", PyExc_Exception, nullptr);
  try {
    if (p) std::rethrow_exception(p);
  } catch (const platform::EOFException& e) {
    PyErr_SetString(EOFExceptionException, e.what());
  } catch (const memory::allocation::BadAlloc& e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
  } catch (const platform::EnforceNotMet& e) {
    switch (e.code()) {
      case phi::ErrorCode::INVALID_ARGUMENT:
        PyErr_SetString(PyExc_ValueError, e.what());
        break;
      case phi::ErrorCode::NOT_FOUND:
      case phi::ErrorCode::ALREADY_EXISTS:
      case phi::ErrorCode::PRECONDITION_NOT_MET:
      case phi::ErrorCode::PERMISSION_DENIED:
      case phi::ErrorCode::EXECUTION_TIMEOUT:
      case phi::ErrorCode::UNAVAILABLE:
        PyErr_SetString(PyExc_RuntimeError, e.what());
        break;
      case phi::ErrorCode::OUT_OF_RANGE:
        PyErr_SetString(PyExc_IndexError, e.what());
        break;
      case phi::ErrorCode::RESOURCE_EXHAUSTED:
        PyErr_SetString(PyExc_MemoryError, e.what());
        break;
      case phi::ErrorCode::UNIMPLEMENTED:
        PyErr_SetString(PyExc_NotImplementedError, e.what());
        break;
      case phi::ErrorCode::FATAL:
        PyErr_SetString(PyExc_SystemError, e.what());
        break;
      case phi::ErrorCode::EXTERNAL:
        PyErr_SetString(PyExc_OSError, e.what());
        break;
      case phi::ErrorCode::INVALID_TYPE:
        PyErr_SetString(PyExc_TypeError, e.what());
        break;
      default:
        PyErr_SetString(EnforceNotMetException, e.what());
        break;
    }
  } catch (const common::PD_Exception& e) {
    PyErr_SetString(PyExc_OSError, e.what());
  }
}
}  // namespace paddle::pybind
