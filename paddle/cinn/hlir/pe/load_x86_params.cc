// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/load_x86_params.h"

#include <glog/logging.h>

#include "paddle/common/enforce.h"

namespace cinn {
namespace hlir {
namespace pe {

void InputX86Param(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data,
    const std::string &key,
    const absl::flat_hash_map<std::string, std::vector<int>> &schedule_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  (*model_data)[key] = schedule_data;
}

void LoadX86DefaultParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  // resnet 1
  InputX86Param(model_data,
                "X86ScheduleConv input 1 3 224 224 weight 64 3 7 7 stride 2 2 "
                "padding 3 3 dilation 1 1",
                {{"ic_bn", {1, 3}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {14, 8}},
                 {"unroll_kw", {0}}});
  // resnet 3 4 5 6
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 56 56 weight 64 64 3 3 stride 1 1 "
                "padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 64}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {8, 7}},
                 {"unroll_kw", {1}}});
  // resnet 8
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 56 56 weight 128 64 3 3 stride 2 2 "
                "padding 1 1 dilation 1 1",
                {{"ic_bn", {2, 32}},
                 {"oc_bn", {2, 64}},
                 {"ow_bn", {7, 4}},
                 {"unroll_kw", {0}}});
  // resnet 9 10 11
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 28 28 weight 128 128 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 128}},
                 {"oc_bn", {4, 32}},
                 {"ow_bn", {4, 7}},
                 {"unroll_kw", {1}}});
  // resnet 7
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 56 56 weight 128 64 1 1 stride 2 2 "
                "padding 0 0 dilation 1 1",
                {{"ic_bn", {8, 8}},
                 {"oc_bn", {4, 32}},
                 {"ow_bn", {7, 4}},
                 {"oh_bn", {1}}});
  // resnet 13
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 28 28 weight 256 128 3 3 stride 2 "
                "2 padding 1 1 dilation 1 1",
                {{"ic_bn", {16, 8}},
                 {"oc_bn", {8, 32}},
                 {"ow_bn", {2, 7}},
                 {"unroll_kw", {1}}});
  // resnet 14 15 16
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 14 14 weight 256 256 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {2, 128}},
                 {"oc_bn", {16, 16}},
                 {"ow_bn", {1, 14}},
                 {"unroll_kw", {1}}});
  // resnet 12
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 28 28 weight 256 128 1 1 stride 2 "
                "2 padding 0 0 dilation 1 1",
                {{"ic_bn", {2, 64}},
                 {"oc_bn", {16, 16}},
                 {"ow_bn", {1, 14}},
                 {"oh_bn", {1}}});
  // resnet 18
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 14 14 weight 512 256 3 3 stride 2 "
                "2 padding 1 1 dilation 1 1",
                {{"ic_bn", {32, 8}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {1, 7}},
                 {"unroll_kw", {1}}});
  // resnet 19 20 21
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 7 7 weight 512 512 3 3 stride 1 1 "
                "padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 512}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {1, 7}},
                 {"unroll_kw", {1}}});
  // resnet 17
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 14 14 weight 512 256 1 1 stride 2 "
                "2 padding 0 0 dilation 1 1",
                {{"ic_bn", {2, 128}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {1, 7}},
                 {"oh_bn", {1}}});
  // resnet 2
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 56 56 weight 64 64 1 1 stride 1 1 "
                "padding 0 0 dilation 1 1",
                {{"ic_bn", {4, 16}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {4, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 56 56 weight 256 64 1 1 stride 1 1 "
                "padding 0 0 dilation 1 1",
                {{"ic_bn", {16, 4}},
                 {"oc_bn", {8, 32}},
                 {"ow_bn", {8, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 56 56 weight 64 256 1 1 stride 1 "
                "1 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {8, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 56 56 weight 128 256 1 1 stride 2 "
                "2 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {4, 32}},
                 {"ow_bn", {4, 7}},
                 {"oh_bn", {1}}});
  // resnet 50
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 56 56 weight 512 256 1 1 stride 2 "
                "2 padding 0 0 dilation 1 1",
                // Todo: temporary fix, enhance alterlayout and test performance
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {7, 4}},
                 {"oh_bn", {1}}});
  // {{"ic_bn", {1, 256}}, {"oc_bn", {8, 64}}, {"ow_bn", {7, 4}}, {"oh_bn",
  // {1}}}); resnet50
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 28 28 weight 512 128 1 1 stride 1 "
                "1 padding 0 0 dilation 1 1",
                {{"ic_bn", {32, 4}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {4, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 28 28 weight 128 512 1 1 stride 1 "
                "1 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 512}},
                 {"oc_bn", {2, 64}},
                 {"ow_bn", {7, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 28 28 weight 256 512 1 1 stride 2 "
                "2 padding 0 0 dilation 1 1",
                {{"ic_bn", {8, 64}},
                 {"oc_bn", {4, 64}},
                 {"ow_bn", {7, 2}},
                 {"oh_bn", {2}}});
  // resnet 50
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 28 28 weight 1024 512 1 1 stride "
                "2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 512}},
                 {"oc_bn", {16, 64}},
                 {"ow_bn", {7, 2}},
                 {"oh_bn", {2}}});
  // resnet 50
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 14 14 weight 1024 256 1 1 stride "
                "1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {16, 64}},
                 {"ow_bn", {7, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 1024 14 14 weight 256 1024 1 1 stride "
                "2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {2, 512}},
                 {"oc_bn", {4, 64}},
                 {"ow_bn", {7, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 1024 14 14 weight 512 1024 1 1 stride "
                "2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {2, 512}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 1024 14 14 weight 2048 1024 1 1 "
                "stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {1, 1024}},
                 {"oc_bn", {64, 32}},
                 {"ow_bn", {1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 7 7 weight 2048 512 1 1 stride 1 "
                "1 padding 0 0 dilation 1 1",
                {{"ic_bn", {128, 4}},
                 {"oc_bn", {64, 32}},
                 {"ow_bn", {1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 2048 7 7 weight 512 2048 1 1 stride 1 "
                "1 padding 0 0 dilation 1 1",
                {{"ic_bn", {512, 4}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 3 224 224 weight 64 3 3 3 stride 1 1 "
                "padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 3}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {28, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 224 224 weight 64 64 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {4, 16}},
                 {"oc_bn", {2, 32}},
                 {"ow_bn", {28, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 64 112 112 weight 128 64 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {2, 32}},
                 {"oc_bn", {2, 64}},
                 {"ow_bn", {28, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 112 112 weight 128 128 3 3 stride "
                "1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {2, 64}},
                 {"oc_bn", {2, 64}},
                 {"ow_bn", {28, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 128 56 56 weight 256 128 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {4, 32}},
                 {"oc_bn", {8, 32}},
                 {"ow_bn", {7, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 56 56 weight 256 256 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {8, 32}},
                 {"ow_bn", {7, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 256 28 28 weight 512 256 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 256}},
                 {"oc_bn", {16, 32}},
                 {"ow_bn", {4, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 28 28 weight 512 512 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 512}},
                 {"oc_bn", {32, 16}},
                 {"ow_bn", {2, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "X86ScheduleConv input 1 512 14 14 weight 512 512 3 3 stride 1 "
                "1 padding 1 1 dilation 1 1",
                {{"ic_bn", {1, 512}},
                 {"oc_bn", {32, 16}},
                 {"ow_bn", {1, 14}},
                 {"unroll_kw", {1}}});
}

void LoadResNet18Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "resnet18 index 0 X86ScheduleConv input 1 3 224 224 weight 64 "
                "3 7 7 stride 2 2 padding 3 3 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "resnet18 index 1 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet18 index 2 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 3 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 4 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 5 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 6 X86ScheduleConv input 1 64 56 56 weight 128 "
                "64 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet18 index 7 X86ScheduleConv input 1 64 56 56 weight 128 "
                "64 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 8 X86ScheduleConv input 1 128 28 28 weight 128 "
                "128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 9 X86ScheduleConv input 1 128 28 28 weight 128 "
                "128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 10 X86ScheduleConv input 1 128 28 28 weight "
                "128 128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 11 X86ScheduleConv input 1 128 28 28 weight "
                "256 128 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet18 index 12 X86ScheduleConv input 1 128 28 28 weight "
                "256 128 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 13 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 14 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 15 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 16 X86ScheduleConv input 1 256 14 14 weight "
                "512 256 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet18 index 17 X86ScheduleConv input 1 256 14 14 weight "
                "512 256 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 18 X86ScheduleConv input 1 512 7 7 weight 512 "
                "512 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 19 X86ScheduleConv input 1 512 7 7 weight 512 "
                "512 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet18 index 20 X86ScheduleConv input 1 512 7 7 weight 512 "
                "512 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
}

void LoadResNet50Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "resnet50 index 0 X86ScheduleConv input 1 3 224 224 weight 64 "
                "3 7 7 stride 2 2 padding 3 3 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "resnet50 index 1 X86ScheduleConv input 1 64 56 56 weight 256 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 2 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 3 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 4 X86ScheduleConv input 1 64 56 56 weight 256 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 5 X86ScheduleConv input 1 256 56 56 weight 64 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 6 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 7 X86ScheduleConv input 1 64 56 56 weight 256 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 8 X86ScheduleConv input 1 256 56 56 weight 64 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 9 X86ScheduleConv input 1 64 56 56 weight 64 "
                "64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 10 X86ScheduleConv input 1 64 56 56 weight 256 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 11 X86ScheduleConv input 1 256 56 56 weight "
                "512 256 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 12 X86ScheduleConv input 1 256 56 56 weight "
                "128 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 13 X86ScheduleConv input 1 128 56 56 weight "
                "128 128 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 14 X86ScheduleConv input 1 128 28 28 weight "
                "512 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 15 X86ScheduleConv input 1 512 28 28 weight "
                "128 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 16 X86ScheduleConv input 1 128 28 28 weight "
                "128 128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 17 X86ScheduleConv input 1 128 28 28 weight "
                "512 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 18 X86ScheduleConv input 1 512 28 28 weight "
                "128 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 19 X86ScheduleConv input 1 128 28 28 weight "
                "128 128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 20 X86ScheduleConv input 1 128 28 28 weight "
                "512 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 21 X86ScheduleConv input 1 512 28 28 weight "
                "128 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 22 X86ScheduleConv input 1 128 28 28 weight "
                "128 128 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 23 X86ScheduleConv input 1 128 28 28 weight "
                "512 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 24 X86ScheduleConv input 1 512 28 28 weight "
                "1024 512 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 128}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 25 X86ScheduleConv input 1 512 28 28 weight "
                "256 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 26 X86ScheduleConv input 1 256 28 28 weight "
                "256 256 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 27 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 28 X86ScheduleConv input 1 1024 14 14 weight "
                "256 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 29 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 30 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 31 X86ScheduleConv input 1 1024 14 14 weight "
                "256 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 32 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 33 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 34 X86ScheduleConv input 1 1024 14 14 weight "
                "256 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 35 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 36 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 37 X86ScheduleConv input 1 1024 14 14 weight "
                "256 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 38 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 39 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 40 X86ScheduleConv input 1 1024 14 14 weight "
                "256 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "resnet50 index 41 X86ScheduleConv input 1 256 14 14 weight "
                "256 256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 42 X86ScheduleConv input 1 256 14 14 weight "
                "1024 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 43 X86ScheduleConv input 1 1024 14 14 weight "
                "2048 1024 1 1 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 44 X86ScheduleConv input 1 1024 14 14 weight "
                "512 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 128}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 45 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 46 X86ScheduleConv input 1 512 7 7 weight 2048 "
                "512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 47 X86ScheduleConv input 1 2048 7 7 weight 512 "
                "2048 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 48 X86ScheduleConv input 1 512 7 7 weight 512 "
                "512 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 49 X86ScheduleConv input 1 512 7 7 weight 2048 "
                "512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 50 X86ScheduleConv input 1 2048 7 7 weight 512 "
                "2048 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "resnet50 index 51 X86ScheduleConv input 1 512 7 7 weight 512 "
                "512 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "resnet50 index 52 X86ScheduleConv input 1 512 7 7 weight 2048 "
                "512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
}

void LoadMobileNetV1Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "mobilenetv1 index 0 X86ScheduleConv input 1 3 224 224 weight "
                "32 3 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv1 index 1 X86ScheduleConv input 1 32 112 112 weight "
                "32 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 2 X86ScheduleConv input 1 32 112 112 weight "
                "64 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 3 X86ScheduleConv input 1 64 112 112 weight "
                "64 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 4 X86ScheduleConv input 1 64 56 56 weight "
                "128 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 5 X86ScheduleConv input 1 128 56 56 weight "
                "128 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 6 X86ScheduleConv input 1 128 56 56 weight "
                "128 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 7 X86ScheduleConv input 1 128 56 56 weight "
                "128 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv1 index 8 X86ScheduleConv input 1 128 28 28 weight "
                "256 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 9 X86ScheduleConv input 1 256 28 28 weight "
                "256 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 10 X86ScheduleConv input 1 256 28 28 weight "
                "256 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv1 index 11 X86ScheduleConv input 1 256 28 28 weight "
                "256 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 12 X86ScheduleConv input 1 256 14 14 weight "
                "512 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 13 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 14 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 15 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 16 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 17 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 18 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 19 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 20 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 21 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 22 X86ScheduleConv input 1 512 14 14 weight "
                "512 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 23 X86ScheduleConv input 1 512 14 14 weight "
                "512 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 4}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv1 index 24 X86ScheduleConv input 1 512 7 7 weight "
                "1024 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv1 index 25 X86ScheduleConv input 1 1024 7 7 weight "
                "1024 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv1 index 26 X86ScheduleConv input 1 1024 7 7 weight "
                "1024 1024 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
}

void LoadMobileNetV2Params(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "mobilenetv2 index 0 X86ScheduleConv input 1 3 224 224 weight "
                "32 3 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 1 X86ScheduleConv input 1 32 112 112 weight "
                "32 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 2 X86ScheduleConv input 1 32 112 112 weight "
                "32 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 3 X86ScheduleConv input 1 32 112 112 weight "
                "16 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 16}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 4 X86ScheduleConv input 1 16 112 112 weight "
                "96 16 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 5 X86ScheduleConv input 1 96 112 112 weight "
                "96 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 6 X86ScheduleConv input 1 96 56 56 weight "
                "24 96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 7 X86ScheduleConv input 1 24 56 56 weight "
                "144 24 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 8 X86ScheduleConv input 1 144 56 56 weight "
                "144 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 9 X86ScheduleConv input 1 144 56 56 weight "
                "24 144 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 28}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 10 X86ScheduleConv input 1 24 56 56 weight "
                "144 24 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 11 X86ScheduleConv input 1 144 56 56 weight "
                "144 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 12 X86ScheduleConv input 1 144 28 28 weight "
                "32 144 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 13 X86ScheduleConv input 1 32 28 28 weight "
                "192 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 14 X86ScheduleConv input 1 192 28 28 weight "
                "192 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 15 X86ScheduleConv input 1 192 28 28 weight "
                "32 192 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 16 X86ScheduleConv input 1 32 28 28 weight "
                "192 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 17 X86ScheduleConv input 1 192 28 28 weight "
                "192 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 18 X86ScheduleConv input 1 192 28 28 weight "
                "32 192 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 19 X86ScheduleConv input 1 32 28 28 weight "
                "192 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 20 X86ScheduleConv input 1 192 28 28 weight "
                "192 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 21 X86ScheduleConv input 1 192 14 14 weight "
                "64 192 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 96}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 22 X86ScheduleConv input 1 64 14 14 weight "
                "384 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 23 X86ScheduleConv input 1 384 14 14 weight "
                "384 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 24 X86ScheduleConv input 1 384 14 14 weight "
                "64 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 25 X86ScheduleConv input 1 64 14 14 weight "
                "384 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 26 X86ScheduleConv input 1 384 14 14 weight "
                "384 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 27 X86ScheduleConv input 1 384 14 14 weight "
                "64 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 28 X86ScheduleConv input 1 64 14 14 weight "
                "384 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 29 X86ScheduleConv input 1 384 14 14 weight "
                "384 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 30 X86ScheduleConv input 1 384 14 14 weight "
                "64 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 31 X86ScheduleConv input 1 64 14 14 weight "
                "384 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 32 X86ScheduleConv input 1 384 14 14 weight "
                "384 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 33 X86ScheduleConv input 1 384 14 14 weight "
                "96 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 34 X86ScheduleConv input 1 96 14 14 weight "
                "576 96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 35 X86ScheduleConv input 1 576 14 14 weight "
                "576 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 36 X86ScheduleConv input 1 576 14 14 weight "
                "96 576 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 37 X86ScheduleConv input 1 96 14 14 weight "
                "576 96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 38 X86ScheduleConv input 1 576 14 14 weight "
                "576 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 39 X86ScheduleConv input 1 576 14 14 weight "
                "96 576 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 40 X86ScheduleConv input 1 96 14 14 weight "
                "576 96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 41 X86ScheduleConv input 1 576 14 14 weight "
                "576 1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "mobilenetv2 index 42 X86ScheduleConv input 1 576 7 7 weight "
                "160 576 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 3}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 43 X86ScheduleConv input 1 160 7 7 weight "
                "960 160 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 44 X86ScheduleConv input 1 960 7 7 weight "
                "960 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 192}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  // InputX86Param(model_data, "mobilenetv2 index 45 X86ScheduleConv input 1 960
  // 7 7 weight 160 960 1 1 stride 1 1 padding 0 0 dilation 1 1", {{"ic_bn",
  // {-1, 64}}, {"oc_bn", {-1, 16}}, {"ow_bn", {-1, 7}}, {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 45 X86ScheduleConv input 1 960 7 7 weight "
                "160 960 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 46 X86ScheduleConv input 1 160 7 7 weight "
                "960 160 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 47 X86ScheduleConv input 1 960 7 7 weight "
                "960 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 192}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  // InputX86Param(model_data, "mobilenetv2 index 48 X86ScheduleConv input 1 960
  // 7 7 weight 160 960 1 1 stride 1 1 padding 0 0 dilation 1 1", {{"ic_bn",
  // {-1, 64}}, {"oc_bn", {-1, 16}}, {"ow_bn", {-1, 7}}, {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 48 X86ScheduleConv input 1 960 7 7 weight "
                "160 960 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 49 X86ScheduleConv input 1 160 7 7 weight "
                "960 160 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 50 X86ScheduleConv input 1 960 7 7 weight "
                "960 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 80}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "mobilenetv2 index 51 X86ScheduleConv input 1 960 7 7 weight "
                "320 960 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "mobilenetv2 index 52 X86ScheduleConv input 1 320 7 7 weight "
                "1280 320 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
}

void LoadSqueezeNetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "squeezenet index 0 X86ScheduleConv input 1 3 227 227 weight "
                "64 3 3 3 stride 2 2 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 1 X86ScheduleConv input 1 64 56 56 weight 16 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "squeezenet index 3 X86ScheduleConv input 1 16 56 56 weight 64 "
                "16 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "squeezenet index 2 X86ScheduleConv input 1 16 56 56 weight 64 "
                "16 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 4 X86ScheduleConv input 1 128 56 56 weight "
                "16 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 6 X86ScheduleConv input 1 16 56 56 weight 64 "
                "16 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 5 X86ScheduleConv input 1 16 56 56 weight 64 "
                "16 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 7 X86ScheduleConv input 1 128 28 28 weight "
                "32 128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "squeezenet index 9 X86ScheduleConv input 1 32 28 28 weight "
                "128 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "squeezenet index 8 X86ScheduleConv input 1 32 28 28 weight "
                "128 32 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 10 X86ScheduleConv input 1 256 28 28 weight "
                "32 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 12 X86ScheduleConv input 1 32 28 28 weight "
                "128 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 11 X86ScheduleConv input 1 32 28 28 weight "
                "128 32 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 2}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 13 X86ScheduleConv input 1 256 14 14 weight "
                "48 256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 15 X86ScheduleConv input 1 48 14 14 weight "
                "192 48 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 48}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 8}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 14 X86ScheduleConv input 1 48 14 14 weight "
                "192 48 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 48}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 16 X86ScheduleConv input 1 384 14 14 weight "
                "48 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "squeezenet index 18 X86ScheduleConv input 1 48 14 14 weight "
                "192 48 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 48}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 8}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 17 X86ScheduleConv input 1 48 14 14 weight "
                "192 48 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 48}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 19 X86ScheduleConv input 1 384 14 14 weight "
                "64 384 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 21 X86ScheduleConv input 1 64 14 14 weight "
                "256 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 20 X86ScheduleConv input 1 64 14 14 weight "
                "256 64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 22 X86ScheduleConv input 1 512 14 14 weight "
                "64 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 24 X86ScheduleConv input 1 64 14 14 weight "
                "256 64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "squeezenet index 23 X86ScheduleConv input 1 64 14 14 weight "
                "256 64 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "squeezenet index 25 X86ScheduleConv input 1 512 14 14 weight "
                "1000 512 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 10}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
}

void LoadFaceDetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "facedet index 0 X86ScheduleConv input 1 3 240 320 weight 16 3 "
                "3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 1 X86ScheduleConv input 1 16 120 160 weight 16 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 2 X86ScheduleConv input 1 16 120 160 weight 32 "
                "16 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 20}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 3 X86ScheduleConv input 1 32 120 160 weight 32 "
                "1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 4 X86ScheduleConv input 1 32 60 80 weight 32 32 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "facedet index 5 X86ScheduleConv input 1 32 60 80 weight 32 1 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 6 X86ScheduleConv input 1 32 60 80 weight 32 32 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "facedet index 7 X86ScheduleConv input 1 32 60 80 weight 32 1 "
                "3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 8 X86ScheduleConv input 1 32 30 40 weight 64 32 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 9 X86ScheduleConv input 1 64 30 40 weight 64 1 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 10 X86ScheduleConv input 1 64 30 40 weight 64 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 11 X86ScheduleConv input 1 64 30 40 weight 64 1 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 13 X86ScheduleConv input 1 64 30 40 weight 64 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 4}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 26 X86ScheduleConv input 1 64 30 40 weight 64 1 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 12 X86ScheduleConv input 1 64 30 40 weight 64 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 20}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 14 X86ScheduleConv input 1 64 30 40 weight 8 64 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 40}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 18 X86ScheduleConv input 1 8 30 40 weight 16 8 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 16}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 22 X86ScheduleConv input 1 16 30 40 weight 16 "
                "16 3 3 stride 1 1 padding 2 2 dilation 2 2",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 15 X86ScheduleConv input 1 64 30 40 weight 8 64 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 40}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 19 X86ScheduleConv input 1 8 30 40 weight 16 8 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 21 X86ScheduleConv input 1 16 30 40 weight 16 "
                "16 3 3 stride 1 1 padding 3 3 dilation 3 3",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 16 X86ScheduleConv input 1 64 30 40 weight 8 64 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 40}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 17 X86ScheduleConv input 1 8 30 40 weight 12 8 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 12}},
                 {"ow_bn", {-1, 10}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 20 X86ScheduleConv input 1 12 30 40 weight 16 "
                "12 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 12}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 10}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 23 X86ScheduleConv input 1 16 30 40 weight 16 "
                "16 3 3 stride 1 1 padding 5 5 dilation 5 5",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 24 X86ScheduleConv input 1 48 30 40 weight 64 "
                "48 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 27 X86ScheduleConv input 1 64 30 40 weight 64 1 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 29 X86ScheduleConv input 1 64 30 40 weight 6 64 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 6}},
                 {"ow_bn", {-1, 40}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 25 X86ScheduleConv input 1 64 30 40 weight 64 1 "
                "3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 5}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 30 X86ScheduleConv input 1 64 15 20 weight 128 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 31 X86ScheduleConv input 1 128 15 20 weight 128 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 32 X86ScheduleConv input 1 128 15 20 weight 128 "
                "128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 33 X86ScheduleConv input 1 128 15 20 weight 128 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 34 X86ScheduleConv input 1 128 15 20 weight 128 "
                "128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 36 X86ScheduleConv input 1 128 15 20 weight 128 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 39 X86ScheduleConv input 1 128 15 20 weight 4 "
                "128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 4}},
                 {"ow_bn", {-1, 20}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 35 X86ScheduleConv input 1 128 15 20 weight 128 "
                "1 3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 40 X86ScheduleConv input 1 128 8 10 weight 256 "
                "128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "facedet index 41 X86ScheduleConv input 1 256 8 10 weight 256 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 42 X86ScheduleConv input 1 256 8 10 weight 256 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 44 X86ScheduleConv input 1 256 8 10 weight 256 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 48 X86ScheduleConv input 1 256 8 10 weight 4 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 4}},
                 {"ow_bn", {-1, 10}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 43 X86ScheduleConv input 1 256 8 10 weight 64 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "facedet index 46 X86ScheduleConv input 1 64 8 10 weight 64 1 "
                "3 3 stride 2 2 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "facedet index 49 X86ScheduleConv input 1 64 4 5 weight 256 64 "
                "1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 5}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "facedet index 51 X86ScheduleConv input 1 256 4 5 weight 6 256 "
                "3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 16}},
                 {"oc_bn", {-1, 6}},
                 {"ow_bn", {-1, 5}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 28 X86ScheduleConv input 1 64 30 40 weight 12 "
                "64 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 12}},
                 {"ow_bn", {-1, 40}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 37 X86ScheduleConv input 1 128 15 20 weight 128 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 38 X86ScheduleConv input 1 128 15 20 weight 8 "
                "128 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 20}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 45 X86ScheduleConv input 1 256 8 10 weight 256 "
                "1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "facedet index 47 X86ScheduleConv input 1 256 8 10 weight 8 "
                "256 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 10}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "facedet index 50 X86ScheduleConv input 1 256 4 5 weight 12 "
                "256 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 12}},
                 {"ow_bn", {-1, 5}},
                 {"unroll_kw", {0}}});
}

void LoadEfficientNetParams(
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<int>>>
        *model_data) {
  PADDLE_ENFORCE_NOT_NULL(
      model_data,
      ::common::errors::PreconditionNotMet("model_data should not be null."));
  InputX86Param(model_data,
                "efficientnet index 0 X86ScheduleConv input 1 3 224 224 weight "
                "32 3 3 3 stride 2 2 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 1 X86ScheduleConv input 1 32 112 112 "
                "weight 32 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 2 X86ScheduleConv input 1 32 1 1 weight 8 "
                "32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 32}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  // InputX86Param(model_data, "efficientnet index 3 X86ScheduleConv input 1 8 1
  // 1 weight 32 8 1 1 stride 1 1 padding 0 0 dilation 1 1", {{"ic_bn", {-1,
  // 8}}, {"oc_bn", {-1, 32}}, {"ow_bn", {-1, 1}}, {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 3 X86ScheduleConv input 1 8 1 1 weight 32 "
                "8 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 4 X86ScheduleConv input 1 32 112 112 "
                "weight 16 32 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 2}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 28}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 5 X86ScheduleConv input 1 16 112 112 "
                "weight 96 16 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 2}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 6 X86ScheduleConv input 1 96 112 112 "
                "weight 96 1 3 3 stride 2 2 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 7 X86ScheduleConv input 1 96 1 1 weight 4 "
                "96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 96}},
                 {"oc_bn", {-1, 4}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  // InputX86Param(model_data, "efficientnet index 8 X86ScheduleConv input 1 4 1
  // 1 weight 96 4 1 1 stride 1 1 padding 0 0 dilation 1 1", {{"ic_bn", {-1,
  // 4}}, {"oc_bn", {-1, 96}}, {"ow_bn", {-1, 1}}, {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 8 X86ScheduleConv input 1 4 1 1 weight 96 "
                "4 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 9 X86ScheduleConv input 1 96 56 56 weight "
                "24 96 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 12}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 10 X86ScheduleConv input 1 24 56 56 weight "
                "144 24 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 11 X86ScheduleConv input 1 144 56 56 "
                "weight 144 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 12 X86ScheduleConv input 1 144 1 1 weight "
                "6 144 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 36}},
                 {"oc_bn", {-1, 6}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  // InputX86Param(model_data, "efficientnet index 13 X86ScheduleConv input 1 6
  // 1 1 weight 144 6 1 1 stride 1 1 padding 0 0 dilation 1 1", {{"ic_bn", {-1,
  // 6}}, {"oc_bn", {-1, 144}}, {"ow_bn", {-1, 1}}, {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 13 X86ScheduleConv input 1 6 1 1 weight "
                "144 6 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 14 X86ScheduleConv input 1 144 56 56 "
                "weight 24 144 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 12}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 15 X86ScheduleConv input 1 144 56 56 "
                "weight 144 1 5 5 stride 2 2 padding 3 3 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 29}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 16 X86ScheduleConv input 1 144 28 28 "
                "weight 40 144 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 28}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 17 X86ScheduleConv input 1 40 28 28 weight "
                "240 40 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 18 X86ScheduleConv input 1 240 28 28 "
                "weight 240 1 5 5 stride 1 1 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 4}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 19 X86ScheduleConv input 1 240 1 1 weight "
                "10 240 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 10}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 20 X86ScheduleConv input 1 10 1 1 weight "
                "240 10 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 5}},
                 {"oc_bn", {-1, 4}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 21 X86ScheduleConv input 1 240 28 28 "
                "weight 40 240 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 22 X86ScheduleConv input 1 240 28 28 "
                "weight 240 1 3 3 stride 2 2 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 5}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 23 X86ScheduleConv input 1 240 14 14 "
                "weight 80 240 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 24 X86ScheduleConv input 1 80 14 14 weight "
                "480 80 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 25 X86ScheduleConv input 1 480 14 14 "
                "weight 480 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 26 X86ScheduleConv input 1 480 1 1 weight "
                "20 480 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 24}},
                 {"oc_bn", {-1, 20}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 27 X86ScheduleConv input 1 20 1 1 weight "
                "480 20 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 20}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 28 X86ScheduleConv input 1 480 14 14 "
                "weight 80 480 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 29 X86ScheduleConv input 1 480 14 14 "
                "weight 480 1 5 5 stride 1 1 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 96}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 30 X86ScheduleConv input 1 480 14 14 "
                "weight 112 480 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 80}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 31 X86ScheduleConv input 1 112 14 14 "
                "weight 672 112 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 56}},
                 {"oc_bn", {-1, 32}},
                 {"ow_bn", {-1, 2}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 32 X86ScheduleConv input 1 672 14 14 "
                "weight 672 1 5 5 stride 1 1 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 96}},
                 {"oc_bn", {-1, 48}},
                 {"ow_bn", {-1, 2}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 33 X86ScheduleConv input 1 672 1 1 weight "
                "28 672 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 6}},
                 {"oc_bn", {-1, 14}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 34 X86ScheduleConv input 1 28 1 1 weight "
                "672 28 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 1}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 35 X86ScheduleConv input 1 672 14 14 "
                "weight 112 672 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 96}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 14}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 36 X86ScheduleConv input 1 672 14 14 "
                "weight 672 1 5 5 stride 2 2 padding 3 3 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 8}},
                 {"unroll_kw", {0}}});
  InputX86Param(model_data,
                "efficientnet index 37 X86ScheduleConv input 1 672 7 7 weight "
                "192 672 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 38 X86ScheduleConv input 1 192 7 7 weight "
                "1152 192 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 3}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 39 X86ScheduleConv input 1 1152 7 7 weight "
                "1152 1 5 5 stride 1 1 padding 2 2 dilation 1 1",
                {{"ic_bn", {-1, 8}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 7}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 40 X86ScheduleConv input 1 1152 1 1 weight "
                "48 1152 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 576}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 41 X86ScheduleConv input 1 48 1 1 weight "
                "1152 48 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 12}},
                 {"oc_bn", {-1, 8}},
                 {"ow_bn", {-1, 1}},
                 {"oh_bn", {1}}});
  InputX86Param(model_data,
                "efficientnet index 42 X86ScheduleConv input 1 1152 7 7 weight "
                "192 1152 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 72}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 43 X86ScheduleConv input 1 1152 7 7 weight "
                "1152 1 3 3 stride 1 1 padding 1 1 dilation 1 1",
                {{"ic_bn", {-1, 64}},
                 {"oc_bn", {-1, 64}},
                 {"ow_bn", {-1, 1}},
                 {"unroll_kw", {1}}});
  InputX86Param(model_data,
                "efficientnet index 44 X86ScheduleConv input 1 1152 7 7 weight "
                "320 1152 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 384}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
  InputX86Param(model_data,
                "efficientnet index 45 X86ScheduleConv input 1 320 7 7 weight "
                "1280 320 1 1 stride 1 1 padding 0 0 dilation 1 1",
                {{"ic_bn", {-1, 4}},
                 {"oc_bn", {-1, 16}},
                 {"ow_bn", {-1, 7}},
                 {"oh_bn", {2}}});
}

absl::flat_hash_map<std::string,
                    absl::flat_hash_map<std::string, std::vector<int>>>
CreateX86Params() {
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::vector<int>>>
      model_data;
  LoadX86DefaultParams(&model_data);
  LoadResNet18Params(&model_data);
  LoadResNet50Params(&model_data);
  LoadMobileNetV1Params(&model_data);
  // LoadMobileNetV2Params(model_data);
  // LoadFaceDetParams(model_data);
  LoadEfficientNetParams(&model_data);
  LoadSqueezeNetParams(&model_data);
  return model_data;
}
}  // namespace pe
}  // namespace hlir
}  // namespace cinn
