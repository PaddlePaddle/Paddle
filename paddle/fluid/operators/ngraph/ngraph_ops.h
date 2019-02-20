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

/*
 * This file contains the list of the ngraph operators for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include "ops/accuracy_op.h"
#include "ops/activation_op.h"
#include "ops/batch_norm_op.h"
#include "ops/binary_unary_op.h"
#include "ops/conv2d_op.h"
#include "ops/cross_entropy_op.h"
#include "ops/elementwise_add_op.h"
#include "ops/fill_constant_op.h"
#include "ops/mean_op.h"
#include "ops/mul_op.h"
#include "ops/pool2d_op.h"
#include "ops/scale_op.h"
#include "ops/softmax_op.h"
#include "ops/sum_op.h"
#include "ops/top_k_op.h"
