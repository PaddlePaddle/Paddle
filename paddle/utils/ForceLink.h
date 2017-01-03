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

/// Declare a force link file ID. It can be enabled by
/// `PADDLE_ENABLE_FORCE_LINK_FILE`. It is
///
/// Example:
///
/// In some_file.cpp
/// @code{cpp}
/// static paddle::InitFunction init([]{...});
/// PADDLE_REGISTER_FORCE_LINK_FILE(some_file)
/// @endcode{cpp}
///
/// In main.cpp
/// @code{cpp}
/// PADDLE_ENABLE_FORCE_LINK_FILE(some_file);
///
/// int main() {
///   ...
/// }
/// @endcode{cpp}
///
/// Then the InitFunction in some_file.cpp can be invoked.
#define PADDLE_REGISTER_FORCE_LINK_FILE(ID) \
  int __paddle_register_force_link_file_##ID##_method__() { return 0; }

/// Enable a force link file. The file with ID's static variables could
/// be all initialized.
#define PADDLE_ENABLE_FORCE_LINK_FILE(ID)                         \
  extern int __paddle_register_force_link_file_##ID##_method__(); \
  static int __paddle_register_force_link_file_##ID##_handler__ = \
      __paddle_register_force_link_file_##ID##_method__();
