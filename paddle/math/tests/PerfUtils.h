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

// Performance Check
#ifdef PADDLE_DISABLE_TIMER

#define EXPRESSION_PERFORMANCE(expression) expression;

#else

#include "paddle/utils/Stat.h"
using namespace paddle;  // NOLINT

#define EXPRESSION_PERFORMANCE(expression)                             \
  do {                                                                 \
    char expr[30];                                                     \
    strncpy(expr, #expression, 30);                                    \
    if (expr[29] != '\0') {                                            \
      expr[27] = '.';                                                  \
      expr[28] = '.';                                                  \
      expr[29] = '\0';                                                 \
    }                                                                  \
    expression;                                                        \
    for (int i = 0; i < 20; i++) {                                     \
      REGISTER_TIMER(expr);                                            \
      expression;                                                      \
    }                                                                  \
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ') \
              << *globalStat.getStat(expr);                            \
    globalStat.reset();                                                \
  } while (0)

#endif
