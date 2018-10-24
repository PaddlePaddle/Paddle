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

#ifndef PADDLE_FLUID_FRAMEWORK_DATAFEED_CREATOR_H_
#define PADDLE_FLUID_FRAMEWORK_DATAFEED_CREATOR_H_
#include <memory>
#include "paddle/fluid/framework/data_feed.h"

std::shared_ptr<paddle::framework::DataFeed> create_datafeed(
    const char* datafeed_class);
#endif  // PADDLE_FLUID_FRAMEWORK_DATAFEED_CREATOR_H_
