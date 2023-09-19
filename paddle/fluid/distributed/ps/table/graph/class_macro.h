// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#define DECLARE_GRAPH_FRIEND_CLASS(a) friend class a;
#define DECLARE_1_FRIEND_CLASS(a, ...) DECLARE_GRAPH_FRIEND_CLASS(a)
#define DECLARE_2_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_1_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_3_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_2_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_4_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_3_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_5_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_4_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_6_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_5_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_7_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_6_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_8_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_7_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_9_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_8_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_10_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_9_FRIEND_CLASS(__VA_ARGS__)
#define DECLARE_11_FRIEND_CLASS(a, ...) \
  DECLARE_GRAPH_FRIEND_CLASS(a) DECLARE_10_FRIEND_CLASS(__VA_ARGS__)
#define REGISTER_GRAPH_FRIEND_CLASS(n, ...) \
  PD_DECLARE_##n##_FRIEND_CLASS(__VA_ARGS__)
