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

#ifndef __PADDLE_CAPI_ERROR_H__
#define __PADDLE_CAPI_ERROR_H__

/**
 * Error Type for Paddle API.
 */
typedef enum {
  kPD_NO_ERROR = 0,
  kPD_NULLPTR = 1,
  kPD_OUT_OF_RANGE = 2,
  kPD_PROTOBUF_ERROR = 3,
  kPD_NOT_SUPPORTED = 4,
  kPD_UNDEFINED_ERROR = -1,
} paddle_error;

static const char* paddle_error_string(paddle_error err) {
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

#endif
