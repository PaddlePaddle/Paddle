// Copyright (c) 2023 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cinn {
namespace ir {

/**
 *  \brief Indicates the level of printing error message in the current Schedule
 */
enum class ScheduleErrorMessageLevel : int32_t {
  /** \brief No error message*/
  kBlank = 0,
  /** \brief  Print an error message in short mode*/
  kGenearl = 1,
  /** \brief Print an error message in detailed mode*/
  kDetailed = 2,
};

/**
 * This handler is dealing with the errors happen in in the current Scheduling.
 */
class IRScheduleErrorHandler : public std::runtime_error {
 public:
  IRScheduleErrorHandler() : std::runtime_error("") {}
  /**
   * \brief constructor
   * \param s the error message
   */
  explicit IRScheduleErrorHandler(const std::string &s)
      : std::runtime_error(s) {}

  /**
   * \brief Returns a detailed error message corresponding to the kDetailed
   * error level.
   */
  std::string FormatErrorMessage(const std::string &primitive) const;

  /**
   * \brief Returns a short error message corresponding to the kGeneral error
   * level.
   */
  virtual std::string GeneralErrorMessage() const = 0;

  /**
   * \brief Returns a detailed error message corresponding to the kDetailed
   * error level.
   */
  virtual std::string DetailedErrorMessage() const = 0;
};

}  // namespace ir
}  // namespace cinn
