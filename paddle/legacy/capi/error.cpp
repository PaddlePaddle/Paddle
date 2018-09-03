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

#include "error.h"

extern "C" const char* paddle_error_string(paddle_error err) {
  switch (err) {
    case kPD_NULLPTR:
      return "nullptr error";
    case kPD_OUT_OF_RANGE:
      return "out of range error";
    case kPD_PROTOBUF_ERROR:
      return "protobuf error";
    case kPD_NOT_SUPPORTED:
      return "not supported error";
    case kPD_UNDEFINED_ERROR:
      return "undefined error";
    default:
      return "";
  }
}
