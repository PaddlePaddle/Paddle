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

#pragma once
#include <memory>
#include <string>
namespace paddle {
typedef std::unique_ptr<std::exception> ErrorPtr;
class Error : public std::exception {
public:
  explicit inline Error(const std::string& what) noexcept : what_(what) {}
  virtual const char* what() const noexcept { return this->what_.c_str(); }

  static void throwError(const std::string& what) throw(ErrorPtr&);

private:
  std::string what_;
};

}  // namespace paddle
