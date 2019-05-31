// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define HOMURA_REPEAT1(X) HOMURA_REPEAT_PATTERN(X)
#define HOMURA_REPEAT2(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT1(ARGS)
#define HOMURA_REPEAT3(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT2(ARGS)
#define HOMURA_REPEAT4(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT3(ARGS)
#define HOMURA_REPEAT5(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT4(ARGS)
#define HOMURA_REPEAT6(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT5(ARGS)
#define HOMURA_REPEAT7(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT6(ARGS)
#define HOMURA_REPEAT8(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT7(ARGS)
#define HOMURA_REPEAT9(X, ARGS...) HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT8(ARGS)
#define HOMURA_REPEAT10(X, ARGS...) \
  HOMURA_REPEAT_PATTERN(X) HOMURA_REPEAT9(ARGS)
