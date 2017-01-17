/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdio.h>
#include <memory>
#include <string>

namespace paddle {

/**
 * Status is Paddle error code. It only contain a std::string as error message.
 * Although Status inherits the std::exception, but do not throw it except you
 * know what you are doing.
 *
 *
 * There are two styles to return status in Paddle.
 *
 * 1. Return Status
 *
 *
 */
class Status final : public std::exception {
public:
  /**
   * Default Status. OK
   */
  Status() noexcept {}

  /**
   * @brief Create Status with error message
   * @param msg
   */
  explicit Status(const std::string& msg) : errMsg_(new std::string(msg)) {}

  /**
   * @brief set a error message for status.
   * @param msg
   */
  inline void set(const std::string& msg) noexcept {
    errMsg_.reset(new std::string(msg));
  }

  /**
   * @brief set a error message for status. Use C style printf
   * @param fmt
   */
  template <typename... ARGS>
  inline void setByPrintf(const char* fmt, ARGS... args) noexcept {
    constexpr size_t kBufferSize = 1024;  // 1KB buffer
    char buffer[kBufferSize];
    snprintf(buffer, kBufferSize, fmt, args...);
    errMsg_.reset(new std::string(buffer));
  }

  /**
   * create a error status by C style printf.
   */
  template <typename... ARGS>
  inline static Status printf(const char* fmt, ARGS... args) noexcept {
    Status s;
    s.setByPrintf(fmt, args...);
    return s;
  }

  /**
   * @brief what will return the error message. If status is OK, return nullptr.
   */
  const char* what() const noexcept override {
    if (errMsg_) {
      return errMsg_->data();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief isOK
   * @return true if OK.
   */
  inline bool isOK() const noexcept { return errMsg_ == nullptr; }

private:
  std::shared_ptr<std::string> errMsg_;
};

}  // namespace paddle
