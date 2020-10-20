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

#include <string>
#include <utility>

#include "paddle/fluid/pybind/exception.h"

namespace paddle {
namespace pybind {

void BindException(pybind11::module* m) {
  static pybind11::exception<platform::EOFException> eof(*m, "EOFException");
  static pybind11::exception<platform::EnforceNotMet> ex_base(*m, "Error");
  static pybind11::exception<platform::errors::InvalidArgumentError>
      ex_invalid_argument(*m, "InvalidArgumentError");
  static pybind11::exception<platform::errors::NotFoundError> ex_not_found(
      *m, "NotFoundError");
  static pybind11::exception<platform::errors::OutOfRangeError> ex_out_of_range(
      *m, "OutOfRangeError");
  static pybind11::exception<platform::errors::AlreadyExistsError>
      ex_already_exists(*m, "AlreadyExistsError");
  static pybind11::exception<platform::errors::ResourceExhaustedError>
      ex_resource_exhausted(*m, "ResourceExhaustedError");
  static pybind11::exception<platform::errors::PreconditionNotMetError>
      ex_precondition_not_met(*m, "PreconditionNotMetError");
  static pybind11::exception<platform::errors::PermissionDeniedError>
      ex_premission_denied(*m, "PermissionDeniedError");
  static pybind11::exception<platform::errors::ExecutionTimeoutError>
      ex_execution_timeout(*m, "ExecutionTimeoutError");
  static pybind11::exception<platform::errors::UnimplementedError>
      ex_unimplemented(*m, "UnimplementedError");
  static pybind11::exception<platform::errors::UnavailableError> ex_unavailable(
      *m, "UnavailableError");
  static pybind11::exception<platform::errors::FatalError> ex_fatal(
      *m, "FatalError");
  static pybind11::exception<platform::errors::ExternalError> ex_external(
      *m, "ExternalError");
  pybind11::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const platform::EOFException& e) {
      eof(e.what());
    } catch (const platform::EnforceNotMet& e) {
      switch (e.code()) {
        case paddle::platform::error::INVALID_ARGUMENT:
          ex_invalid_argument(e.what());
          break;
        case paddle::platform::error::NOT_FOUND:
          ex_not_found(e.what());
          break;
        case paddle::platform::error::OUT_OF_RANGE:
          ex_out_of_range(e.what());
          break;
        case paddle::platform::error::ALREADY_EXISTS:
          ex_already_exists(e.what());
          break;
        case paddle::platform::error::RESOURCE_EXHAUSTED:
          ex_resource_exhausted(e.what());
          break;
        case paddle::platform::error::PRECONDITION_NOT_MET:
          ex_precondition_not_met(e.what());
          break;
        case paddle::platform::error::PERMISSION_DENIED:
          ex_premission_denied(e.what());
          break;
        case paddle::platform::error::EXECUTION_TIMEOUT:
          ex_execution_timeout(e.what());
          break;
        case paddle::platform::error::UNIMPLEMENTED:
          ex_unimplemented(e.what());
          break;
        case paddle::platform::error::UNAVAILABLE:
          ex_unavailable(e.what());
          break;
        case paddle::platform::error::FATAL:
          ex_fatal(e.what());
          break;
        case paddle::platform::error::EXTERNAL:
          ex_external(e.what());
          break;
        default:
          ex_base(e.what());
          break;
      }
    }
  });

  m->def("__unittest_throw_exception__", [] {
    PADDLE_THROW(
        platform::errors::PermissionDenied("This is a test of exception"));
  });
}

}  // namespace pybind
}  // namespace paddle
