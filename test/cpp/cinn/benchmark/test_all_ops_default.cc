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

#include <absl/container/flat_hash_map.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/runtime/cpu/use_extern_funcs.h"
#include "test/cpp/cinn/benchmark/test_utils.h"

namespace cinn {
namespace tests {

using cinn::hlir::framework::AttrType;

#define TEST_DEFAULT(op_name__, shape_name__, input_types_, output_types_) \
  TEST(op_defualt, shape_name__) {                                         \
    std::vector<std::vector<int>> input_shapes = shapes_##shape_name__;    \
    std::string op_name = #op_name__;                                      \
    hlir::framework::NodeAttr attrs;                                       \
    OpBenchmarkTester tester(op_name, input_shapes);                       \
    auto input_tensors = tester.CreateInputTensors<float>();               \
    tester.TestOp(common::UniqName(#op_name__),                            \
                  input_tensors,                                           \
                  attrs,                                                   \
                  input_types_,                                            \
                  output_types_);                                          \
  }

#define TEST_DEFAULT1(                                                  \
    op_name__, shape_name__, input_types_, output_types_, attr_store__) \
  TEST(op_defualt1, shape_name__) {                                     \
    std::vector<std::vector<int>> input_shapes = shapes_##shape_name__; \
    std::string op_name = #op_name__;                                   \
    OpBenchmarkTester tester(op_name, input_shapes);                    \
    hlir::framework::NodeAttr attrs;                                    \
    attrs.attr_store = attr_store__;                                    \
    auto input_tensors = tester.CreateInputTensors<float>();            \
    std::vector<Type> input_types{Float(32), Float(32)};                \
    tester.TestOp(common::UniqName(#op_name__),                         \
                  input_tensors,                                        \
                  attrs,                                                \
                  input_types_,                                         \
                  output_types_);                                       \
  }

#define TEST_DEFAULT_INT(op_name__, shape_name__, input_types_, output_types_) \
  TEST(op_defualt, shape_name__) {                                             \
    std::vector<std::vector<int>> input_shapes = shapes_##shape_name__;        \
    std::string op_name = #op_name__;                                          \
    hlir::framework::NodeAttr attrs;                                           \
    OpBenchmarkTester tester(op_name, input_shapes);                           \
    auto input_tensors = tester.CreateInputTensors<int>();                     \
    tester.TestOp(common::UniqName(#op_name__),                                \
                  input_tensors,                                               \
                  attrs,                                                       \
                  input_types_,                                                \
                  output_types_);                                              \
  }

std::vector<Type> type = {Float(32)};
std::vector<Type> type1 = {Float(32), Float(32)};
std::vector<Type> type2 = {Int(32)};
std::vector<Type> type3 = {Bool()};
std::vector<Type> type4 = {
    Float(32), Float(32), Float(32), Float(32), Float(32)};
std::vector<Type> type5 = {Int(32), Int(32)};
std::vector<Type> type6 = {Float(32), Void()};
std::vector<Type> type7 = {Float(32), Float(32), Float(32), Float(32)};
std::vector<Type> type8 = {Float(32), Float(32), Float(32)};

// broadcast_to
std::vector<std::vector<int>> shapes_broadcast_to = {{32}};
std::vector<int> out_shape = {100, 32};
std::vector<int> broadcast_axes = {1};
absl::flat_hash_map<std::string, AttrType> attr_store_broadcast_to = {
    {"out_shape", out_shape}, {"broadcast_axes", broadcast_axes}};
TEST_DEFAULT1(broadcast_to, broadcast_to, type, type, attr_store_broadcast_to)

// concat
std::vector<std::vector<int>> shapes_concat = {{2, 2, 3}, {2, 4, 3}};
absl::flat_hash_map<std::string, AttrType> attr_store_concat = {{"axis", 1}};
TEST_DEFAULT1(concat, concat, type1, type, attr_store_concat)

std::vector<std::vector<int>> shapes_concat1 = {
    {2, 2, 3}, {2, 4, 3}, {2, 5, 3}};
absl::flat_hash_map<std::string, AttrType> attr_store_concat1 = {{"axis", -2}};
TEST_DEFAULT1(concat, concat1, type8, type, attr_store_concat1)

// add
std::vector<std::vector<int>> shapes_add = {{1024, 1024, 1024},
                                            {1024, 1024, 1024}};
TEST_DEFAULT(elementwise_add, add, type1, type)
std::vector<std::vector<int>> shapes_add1_0 = {{100, 32}, {100, 32}};
std::vector<std::vector<int>> shapes_add1_1 = {{32, 100}, {32, 100}};
std::vector<std::vector<int>> shapes_add1_2 = {{100, 33}, {100, 33}};
std::vector<std::vector<int>> shapes_add1_3 = {{33, 100}, {33, 100}};
std::vector<std::vector<int>> shapes_add1_4 = {{100, 16}, {100, 16}};
std::vector<std::vector<int>> shapes_add1_5 = {{1, 33}, {1, 33}};
std::vector<std::vector<int>> shapes_add1_6 = {{33}, {33}};
TEST_DEFAULT(elementwise_add, add1_0, type1, type)
TEST_DEFAULT(elementwise_add, add1_1, type1, type)
TEST_DEFAULT(elementwise_add, add1_2, type1, type)
TEST_DEFAULT(elementwise_add, add1_3, type1, type)
TEST_DEFAULT(elementwise_add, add1_4, type1, type)
TEST_DEFAULT(elementwise_add, add1_5, type1, type)
TEST_DEFAULT(elementwise_add, add1_6, type1, type)
std::vector<std::vector<int>> shapes_add2 = {{1024, 14, 14}, {1024, 14, 14}};
TEST_DEFAULT(elementwise_add, add2, type1, type)
std::vector<std::vector<int>> shapes_add3 = {{1}, {1}};
TEST_DEFAULT(elementwise_add, add3, type1, type)
std::vector<std::vector<int>> shapes_add4 = {{1, 8}, {1, 8}};
TEST_DEFAULT(elementwise_add, add4, type1, type)
std::vector<std::vector<int>> shapes_add5_0 = {{1024, 2}, {1024, 2}};
std::vector<std::vector<int>> shapes_add5_1 = {{2, 1024}, {2, 1024}};
std::vector<std::vector<int>> shapes_add5_2 = {{1025, 2}, {1025, 2}};
std::vector<std::vector<int>> shapes_add5_3 = {{2, 1025}, {2, 1025}};
TEST_DEFAULT(elementwise_add, add5_0, type1, type)
TEST_DEFAULT(elementwise_add, add5_1, type1, type)
TEST_DEFAULT(elementwise_add, add5_2, type1, type)
TEST_DEFAULT(elementwise_add, add5_3, type1, type)
// mul
std::vector<std::vector<int>> shapes_elementwise_mul = {{1024, 1024, 1024},
                                                        {1024, 1024, 1024}};
TEST_DEFAULT(elementwise_mul, elementwise_mul, type1, type)
std::vector<std::vector<int>> shapes_elementwise_mul1 = {{100, 32}, {100, 32}};
TEST_DEFAULT(elementwise_mul, elementwise_mul1, type1, type)
std::vector<std::vector<int>> shapes_elementwise_mul2 = {{1024, 14, 14},
                                                         {1024, 14, 14}};
TEST_DEFAULT(elementwise_mul, elementwise_mul2, type1, type)
std::vector<std::vector<int>> shapes_elementwise_mul3 = {{1}, {1}};
TEST_DEFAULT(elementwise_mul, elementwise_mul3, type1, type)

// relu
std::vector<std::vector<int>> shapes_relu = {{2, 512, 7, 7}};
TEST_DEFAULT(relu, relu, type, type)
std::vector<std::vector<int>> shapes_relu1 = {{1024, 14, 14}};
TEST_DEFAULT(relu, relu1, type, type)

// conv2d nchw
std::vector<std::vector<int>> shapes_conv2d_nchw = {{2, 512, 7, 7},
                                                    {512, 512, 3, 3}};
std::vector<int> padding_conv2d({0, 0});
std::vector<int> stride_conv2d({1, 1});
std::vector<int> dilation_conv2d({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d = {
    {"padding", padding_conv2d},
    {"stride", stride_conv2d},
    {"dilation", dilation_conv2d}};
TEST_DEFAULT1(conv2d, conv2d_nchw, type1, type8, attr_store_conv2d)
std::vector<std::vector<int>> shapes_conv2d_nchw1 = {{2, 1024, 14, 14},
                                                     {256, 1024, 1, 1}};
TEST_DEFAULT1(conv2d, conv2d_nchw1, type1, type8, attr_store_conv2d)
std::vector<std::vector<int>> shapes_conv2d_nchw2 = {{8, 32, 1, 1},
                                                     {8, 32, 1, 1}};
TEST_DEFAULT1(conv2d, conv2d_nchw2, type1, type8, attr_store_conv2d)

// resnet18
std::vector<std::vector<int>> shapes_conv2d_nchw3 = {{1, 3, 224, 224},
                                                     {64, 3, 7, 7}};
std::vector<int> padding_conv2d1({3, 3});
std::vector<int> stride_conv2d1({2, 2});
std::vector<int> dilation_conv2d1({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d1 = {
    {"padding", padding_conv2d1},
    {"stride", stride_conv2d1},
    {"dilation", dilation_conv2d1}};
TEST_DEFAULT1(conv2d, conv2d_nchw3, type1, type7, attr_store_conv2d1)

// resnet18 1*1
std::vector<std::vector<int>> shapes_conv2d_nchw4 = {{1, 64, 56, 56},
                                                     {64, 64, 1, 1}};
std::vector<int> padding_conv2d4({0, 0});
std::vector<int> stride_conv2d4({1, 1});
std::vector<int> dilation_conv2d4({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d4 = {
    {"padding", padding_conv2d4},
    {"stride", stride_conv2d4},
    {"dilation", dilation_conv2d4}};
TEST_DEFAULT1(conv2d, conv2d_nchw4, type1, type8, attr_store_conv2d4)

// mobilenet 1*1
std::vector<std::vector<int>> shapes_conv2d_nchw5 = {{1, 16, 112, 112},
                                                     {96, 16, 1, 1}};
std::vector<int> padding_conv2d5({0, 0});
std::vector<int> stride_conv2d5({1, 1});
std::vector<int> dilation_conv2d5({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d5 = {
    {"padding", padding_conv2d5},
    {"stride", stride_conv2d5},
    {"dilation", dilation_conv2d5}};
TEST_DEFAULT1(conv2d, conv2d_nchw5, type1, type8, attr_store_conv2d5)

// effi
std::vector<std::vector<int>> shapes_conv2d_nchw6 = {{1, 3, 224, 224},
                                                     {32, 3, 3, 3}};
std::vector<int> padding_conv2d6({2, 2});
std::vector<int> stride_conv2d6({2, 2});
std::vector<int> dilation_conv2d6({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d6 = {
    {"padding", padding_conv2d6},
    {"stride", stride_conv2d6},
    {"dilation", dilation_conv2d6}};
TEST_DEFAULT1(conv2d, conv2d_nchw6, type1, type7, attr_store_conv2d6)

// test_op_nn
std::vector<std::vector<int>> shapes_conv2d_nchw7 = {{1, 3, 10, 10},
                                                     {2, 3, 2, 2}};
std::vector<int> padding_conv2d7({1, 1});
std::vector<int> stride_conv2d7({2, 2});
std::vector<int> dilation_conv2d7({2, 2});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d7 = {
    {"padding", padding_conv2d7},
    {"stride", stride_conv2d7},
    {"dilation", dilation_conv2d7}};
TEST_DEFAULT1(conv2d, conv2d_nchw7, type1, type7, attr_store_conv2d7)

// conv2d_NCHWc
// resnet18
std::vector<std::vector<int>> shapes_conv2d_nchwc = {{1, 1, 224, 224, 3},
                                                     {4, 1, 7, 7, 3, 16}};
std::vector<int> padding_conv2d_nchwc({3, 3});
std::vector<int> stride_conv2d_nchwc({2, 2});
std::vector<int> dilation_conv2d_nchwc({1, 1});
absl::flat_hash_map<std::string, AttrType> attr_store_conv2d_nchwc = {
    {"padding", padding_conv2d_nchwc},
    {"stride", stride_conv2d_nchwc},
    {"dilation", dilation_conv2d_nchwc}};
TEST_DEFAULT1(conv2d_NCHWc, conv2d_nchwc, type1, type8, attr_store_conv2d_nchwc)

// depthwise_conv2d nchw
std::vector<std::vector<int>> shapes_depthwise_conv2d_nchw = {{2, 32, 112, 112},
                                                              {32, 1, 3, 3}};
std::vector<int> stride_depthwise_conv2d = {1, 1};
std::vector<int> padding_depthwise_conv2d = {1, 1};
std::vector<int> dilation_depthwise_conv2d = {1, 1};
absl::flat_hash_map<std::string, AttrType> attr_store_depthwise_conv2d = {
    {"padding", padding_depthwise_conv2d},
    {"stride", stride_depthwise_conv2d},
    {"dilation", dilation_depthwise_conv2d}};
TEST_DEFAULT1(depthwise_conv2d,
              depthwise_conv2d_nchw,
              type1,
              type7,
              attr_store_depthwise_conv2d)

// layout_transform
std::vector<std::vector<int>> shapes_layout_transform = {{512, 512, 3, 3}};
std::string src_layout = "OIHW";        // NOLINT
std::string dst_layout = "OIHW16i16o";  // NOLINT
absl::flat_hash_map<std::string, AttrType> attr_store_layout_transform = {
    {"src_layout", src_layout}, {"dst_layout", dst_layout}};
TEST_DEFAULT1(
    layout_transform, layout_transform, type, type, attr_store_layout_transform)

std::vector<std::vector<int>> shapes_layout_transform1 = {{64, 3, 7, 7}};
std::string src_layout1 = "OIHW";       // NOLINT
std::string dst_layout1 = "OIHW3i32o";  // NOLINT
absl::flat_hash_map<std::string, AttrType> attr_store_layout_transform1 = {
    {"src_layout", src_layout1}, {"dst_layout", dst_layout1}};
TEST_DEFAULT1(layout_transform,
              layout_transform1,
              type,
              type,
              attr_store_layout_transform1)

// pool2d
hlir::framework::NodeAttr attrs;
std::vector<int> kernel_size = {3, 3};
std::vector<int> stride_size = {2, 2};
std::vector<int> padding_size = {1, 1, 1, 1};
std::string pool_type = "max";  // NOLINT
absl::flat_hash_map<std::string, AttrType> attr_store_pool2d = {
    {"kernel_size", kernel_size},
    {"stride_size", stride_size},
    {"padding_size", padding_size},
    {"pool_type", pool_type}};

std::vector<std::vector<int>> shapes_pool2d = {{2, 64, 112, 112}};
TEST_DEFAULT1(pool2d, pool2d, type, type, attr_store_pool2d)
std::vector<std::vector<int>> shapes_pool2d1 = {{2, 1024, 14, 14}};
TEST_DEFAULT1(pool2d, pool2d1, type, type, attr_store_pool2d)

// softmax
std::vector<std::vector<int>> shapes_softmax = {{1024, 2048}};
TEST_DEFAULT(softmax, softmax, type, type1)
std::vector<std::vector<int>> shapes_softmax1 = {{3, 1000}};
TEST_DEFAULT(softmax, softmax1, type, type1)

// sigmoid
std::vector<std::vector<int>> shapes_sigmoid = {{2, 672, 1, 1}};
TEST_DEFAULT(sigmoid, sigmoid, type, type)
std::vector<std::vector<int>> shapes_sigmoid1 = {{3, 1000}};
TEST_DEFAULT(sigmoid, sigmoid1, type, type)

// matmul
std::vector<std::vector<int>> shapes_matmul = {{32, 32}, {32, 32}};
TEST_DEFAULT(matmul, matmul, type1, type1)
std::vector<std::vector<int>> shapes_matmul1 = {{512, 512}, {512, 512}};
TEST_DEFAULT(matmul, matmul1, type1, type1)
std::vector<std::vector<int>> shapes_matmul2 = {{100, 32}, {32, 100}};
TEST_DEFAULT(matmul, matmul2, type1, type1)
std::vector<std::vector<int>> shapes_matmul3 = {{1024, 1024}, {1024, 1024}};
TEST_DEFAULT(matmul, matmul3, type1, type1)
std::vector<std::vector<int>> shapes_matmul4 = {{1, 1024, 1024},
                                                {1, 1024, 1024}};
TEST_DEFAULT(matmul, matmul4, type1, type1)
std::vector<std::vector<int>> shapes_matmul5 = {{1}, {1}};
TEST_DEFAULT(matmul, matmul5, type1, type1)
std::vector<std::vector<int>> shapes_matmul6 = {{1, 30}, {30}};
TEST_DEFAULT(matmul, matmul6, type1, type1)
std::vector<std::vector<int>> shapes_matmul7 = {{2, 100, 4}, {2, 4, 100}};
TEST_DEFAULT(matmul, matmul7, type1, type1)

// matrix mul
std::vector<std::vector<int>> shapes_mul = {{32, 32}, {32, 32}};
TEST_DEFAULT(mul, mul, type1, type1)
std::vector<std::vector<int>> shapes_mul1 = {{512, 512}, {512, 512}};
TEST_DEFAULT(mul, mul1, type1, type1)
std::vector<std::vector<int>> shapes_mul2 = {{100, 32}, {100, 32}};
TEST_DEFAULT(mul, mul2, type1, type1)
std::vector<std::vector<int>> shapes_mul3 = {{1024, 1024}, {1024, 1024}};
TEST_DEFAULT(mul, mul3, type1, type1)
std::vector<std::vector<int>> shapes_mul4 = {{1}, {1}};
TEST_DEFAULT(mul, mul4, type1, type1)
std::vector<std::vector<int>> shapes_mul5 = {{1, 30}, {1, 30}};
TEST_DEFAULT(mul, mul5, type1, type1)

// batchnorm
std::vector<std::vector<int>> shapes_batchnorm = {
    {2, 32, 112, 112}, {32}, {32}, {32}, {32}};
TEST_DEFAULT(batch_norm, batchnorm, type4, type)

// scale
std::vector<std::vector<int>> shapes_scale = {{2, 1000}};
TEST_DEFAULT(scale, scale, type, type)

// slice
std::vector<std::vector<int>> shapes_slice = {{2, 32, 113, 113}};
std::vector<int> starts({1, 1});
std::vector<int> ends({10000000, 10000000});
std::vector<int> axes({2, 3});
absl::flat_hash_map<std::string, AttrType> attr_store_slice = {
    {"starts", starts}, {"ends", ends}, {"axes", axes}};
TEST_DEFAULT1(slice, slice, type, type, attr_store_slice)

// unary
#define TEST_DEFAULT_UNARY(op__)                                      \
  std::vector<std::vector<int>> shapes_unary_##op__ = {{1024, 2048}}; \
  std::vector<std::vector<int>> shapes_unary_##op__##1 = {{3, 1000}}; \
  TEST_DEFAULT(op__, unary_##op__, type, type)                        \
  TEST_DEFAULT(op__, unary_##op__##1, type, type)

TEST_DEFAULT_UNARY(exp)
TEST_DEFAULT_UNARY(erf)
TEST_DEFAULT_UNARY(sigmoid)
TEST_DEFAULT_UNARY(sqrt)
TEST_DEFAULT_UNARY(log)
TEST_DEFAULT_UNARY(log2)
TEST_DEFAULT_UNARY(log10)
TEST_DEFAULT_UNARY(floor)
TEST_DEFAULT_UNARY(ceil)
TEST_DEFAULT_UNARY(round)
TEST_DEFAULT_UNARY(trunc)
TEST_DEFAULT_UNARY(cos)
TEST_DEFAULT_UNARY(cosh)
TEST_DEFAULT_UNARY(tan)
TEST_DEFAULT_UNARY(tanh)
TEST_DEFAULT_UNARY(sin)
TEST_DEFAULT_UNARY(sinh)
TEST_DEFAULT_UNARY(acos)
TEST_DEFAULT_UNARY(acosh)
TEST_DEFAULT_UNARY(asin)
TEST_DEFAULT_UNARY(asinh)
TEST_DEFAULT_UNARY(atan)
TEST_DEFAULT_UNARY(atanh)

// unary_bool
#define TEST_DEFAULT_UNARY_BOOL(op__)                                 \
  std::vector<std::vector<int>> shapes_unary_##op__ = {{1024, 2048}}; \
  std::vector<std::vector<int>> shapes_unary_##op__##1 = {{3, 1000}}; \
  TEST_DEFAULT(op__, unary_##op__, type, type3)                       \
  TEST_DEFAULT(op__, unary_##op__##1, type, type3)

TEST_DEFAULT_UNARY_BOOL(isnan)
TEST_DEFAULT_UNARY_BOOL(isfinite)
TEST_DEFAULT_UNARY_BOOL(isinf)

// bitwise_not
std::vector<std::vector<int>> shapes_bitwise_not = {{1024, 2048}};
std::vector<std::vector<int>> shapes_bitwise_not1 = {{3, 1000}};
TEST_DEFAULT_INT(bitwise_not, bitwise_not, type2, type2)
TEST_DEFAULT_INT(bitwise_not, bitwise_not1, type2, type2)

// binary bitwise
#define TEST_DEFAULT_BINARY(op__)                                      \
  std::vector<std::vector<int>> shapes_binary_##op__ = {{1024, 2048},  \
                                                        {1024, 2048}}; \
  std::vector<std::vector<int>> shapes_binary_##op__##1 = {{3, 1000},  \
                                                           {3, 1000}}; \
  TEST_DEFAULT_INT(op__, binary_##op__, type5, type2)                  \
  TEST_DEFAULT_INT(op__, binary_##op__##1, type5, type2)

TEST_DEFAULT_BINARY(left_shift)
TEST_DEFAULT_BINARY(right_shift)
TEST_DEFAULT_BINARY(bitwise_or)
TEST_DEFAULT_BINARY(bitwise_and)
TEST_DEFAULT_BINARY(bitwise_xor)

}  // namespace tests
}  // namespace cinn
