/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete;

namespace paddle {
namespace platform {
namespace stream {
namespace internal {

class StreamInterface {
 public:
  // Default constructor for the abstract interface.
  StreamInterface() {}

  // Default destructor for the abstract interface.
  virtual ~StreamInterface() {}
  // virtual bool Init() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(StreamInterface);
};

class EventInterface {
 public:
  EventInterface() {}
  virtual ~EventInterface() {}
  // virtual bool Init() = 0;
  // virtual GetEventStatus() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(EventInterface);
};

}  // namespace internal
}  // namespace stream
}  // namespace platform
}  // namespace paddle
