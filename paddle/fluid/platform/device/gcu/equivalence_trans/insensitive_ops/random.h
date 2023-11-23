/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <limits>
#include <memory>
#include <vector>
#include "paddle/fluid/operators/truncated_gaussian_random_op.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kUniformRandom = "uniform_random";
const char *const kGaussianRandom = "gaussian_random";
const char *const kTruncatedGaussianRandom = "truncated_gaussian_random";
const char *const kRandPerm = "randperm";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               UniformRandomEquivalenceTrans) {
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  // if (map_inputs.count("ShapeTensor") != 0) {
  //   auto dims = map_inputs["ShapeTensor"].at(0)->GetConstData<int>();
  //   shape.clear();
  //   for (auto dim : dims) {
  //     shape.emplace_back(static_cast<int64_t>(dim));
  //   }
  // }
  // if (map_inputs.count("ShapeTensorList") != 0) {
  //   std::vector<int> dims;
  //   for (size_t i = 0; i < map_inputs["ShapeTensorList"].size(); ++i) {
  //     dims.emplace_back(
  //         map_inputs["ShapeTensorList"].at(0)->GetConstData<int>()[0]);
  //   }
  //   shape.clear();
  //   for (auto dim : dims) {
  //     shape.emplace_back(static_cast<int64_t>(dim));
  //   }
  // }
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto min = PADDLE_GET_CONST(float, op->GetAttr("min"));
  auto max = PADDLE_GET_CONST(float, op->GetAttr("max"));
  double scalar = static_cast<double>(max) - static_cast<double>(min);
  double offset = static_cast<double>(min) / scalar;

  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP16) {
    ptype = builder::PrimitiveType::F16();
  } else if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::UINT8) {
    ptype = builder::PrimitiveType::U8();
  } else if (dtype == framework::proto::VarType::INT8) {
    ptype = builder::PrimitiveType::S8();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("uniform_random dtype: %d", dtype));
  }

  builder::Type scalar_type(ptype);
  auto offset_op = builder::Const(gcu_builder, offset, scalar_type);
  auto scalar_op = builder::Const(gcu_builder, scalar, scalar_type);
  auto zero_op = builder::ZerosLike(offset_op);
  auto one_op = builder::OnesLike(offset_op);
  builder::Type shape_type({static_cast<int64_t>(shape.size())},
                           builder::PrimitiveType::S64());
  auto shape_op = builder::Const(gcu_builder, shape, shape_type);

  auto rng = builder::RngUniform(zero_op, one_op, shape_op, builder::nullopt);
  auto ret = (rng + offset_op) * scalar_op;
  return std::make_shared<GcuOp>(ret);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               GaussianRandomEquivalenceTrans) {
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  if (map_inputs.count("ShapeTensor") != 0) {
    auto dims = map_inputs["ShapeTensor"].at(0)->GetConstData<int>();
    shape.clear();
    for (auto dim : dims) {
      shape.emplace_back(static_cast<int64_t>(dim));
    }
  }
  if (map_inputs.count("ShapeTensorList") != 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["ShapeTensorList"].size(); ++i) {
      dims.emplace_back(
          map_inputs["ShapeTensorList"].at(0)->GetConstData<int>()[0]);
    }
    shape.clear();
    for (auto dim : dims) {
      shape.emplace_back(static_cast<int64_t>(dim));
    }
  }
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto mean = PADDLE_GET_CONST(float, op->GetAttr("mean"));
  auto std = PADDLE_GET_CONST(float, op->GetAttr("std"));
  auto seed = PADDLE_GET_CONST(int, op->GetAttr("seed"));

  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP16) {
    ptype = builder::PrimitiveType::F16();
  } else if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::UINT8) {
    ptype = builder::PrimitiveType::U8();
  } else if (dtype == framework::proto::VarType::INT8) {
    ptype = builder::PrimitiveType::S8();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("gaussian_random dtype: %d", dtype));
  }

  builder::Type scalar_type(ptype);
  auto mu = builder::Const(gcu_builder, mean, scalar_type);
  auto sigma = builder::Const(gcu_builder, std, scalar_type);
  builder::Type shape_type({static_cast<int64_t>(shape.size())},
                           builder::PrimitiveType::S64());
  auto shape_op = builder::Const(gcu_builder, shape, shape_type);

  seed = seed == 0 ? 1 : seed;  // sdk cpu_rng_normal use 1 to represent
  // default value.
  return std::make_shared<GcuOp>(builder::RngNormal(mu, sigma, shape_op, seed));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               TruncatedGaussianRandomEquivalenceTrans) {
  auto *op = node->Op();
  auto shape32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  auto mean = PADDLE_GET_CONST(float, op->GetAttr("mean"));
  auto std = PADDLE_GET_CONST(float, op->GetAttr("std"));
  auto seed = PADDLE_GET_CONST(int, op->GetAttr("seed"));
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  std::vector<int64_t> shape(shape32.begin(), shape32.end());

  int64_t size = std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<int64_t>());

  std::vector<double> data(size);

  std::uniform_real_distribution<double> dist(
      std::numeric_limits<double>::min(), 1.0);
  paddle::operators::TruncatedNormal<double> truncated_normal(mean, std);
  auto engine = phi::GetCPURandomEngine(seed);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = truncated_normal(dist(*engine));
  }

  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP16) {
    ptype = builder::PrimitiveType::F16();
  } else if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::UINT8) {
    ptype = builder::PrimitiveType::U8();
  } else if (dtype == framework::proto::VarType::INT8) {
    ptype = builder::PrimitiveType::S8();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "truncated_gaussian_random dtype: %d", dtype));
  }
  auto out_op = builder::Const(gcu_builder, data, builder::Type(shape, ptype));
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RandPermEquivalenceTrans) {
  auto *op = node->Op();
  auto seed = PADDLE_GET_CONST(int, op->GetAttr("seed"));
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  int64_t size = PADDLE_GET_CONST(int, op->GetAttr("n"));

  std::vector<int64_t> data(size);
  std::iota(data.begin(), data.end(), 0);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    std::random_device rd;
    // double has 53 bit significant, so limit uint64 to 53 bits
    uint64_t seed = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
    std::seed_seq seq({seed});
    engine = std::make_shared<std::mt19937_64>(seq);
  }
  std::shuffle(data.begin(), data.end(), *engine);

  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP16) {
    ptype = builder::PrimitiveType::F16();
  } else if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::UINT8) {
    ptype = builder::PrimitiveType::U8();
  } else if (dtype == framework::proto::VarType::INT8) {
    ptype = builder::PrimitiveType::S8();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "truncated_gaussian_random dtype: %d", dtype));
  }
  auto out_op = builder::Const(gcu_builder, data, builder::Type({size}, ptype));
  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kUniformRandom,
                           INSENSITIVE,
                           UniformRandomEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGaussianRandom,
                           INSENSITIVE,
                           GaussianRandomEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTruncatedGaussianRandom,
                           INSENSITIVE,
                           TruncatedGaussianRandomEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRandPerm, INSENSITIVE, RandPermEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
