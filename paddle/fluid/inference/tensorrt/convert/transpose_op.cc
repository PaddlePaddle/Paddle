/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <bitset>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

// Permutation is valid if it has nbDims unique values from range [0, nbDims-1]
bool IsValidPermutation(int dims, const nvinfer1::Permutation& permutation) {
  std::bitset<nvinfer1::Dims::MAX_DIMS> found;
  for (int i = 0; i < dims; ++i) {
    const int x = permutation.order[i];
    if ((x < 0) || (x >= dims) || found[x])
      return false;  // Out of bounds or duplicate
    found.set(x);
  }
  return true;
}

/*
 * TransposeOp
 */
class TransposeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    int dims = input->getDimensions().nbDims;
    std::vector<int> axis =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"));
    PADDLE_ENFORCE_LT(axis.size(), nvinfer1::Dims::MAX_DIMS,
                      platform::errors::InvalidArgument(
                          "The size of transpose op's axis should not exceed "
                          "the max dims(currently %d) of TRT, but got %d",
                          nvinfer1::Dims::MAX_DIMS, axis.size()));
    if (!engine_->with_dynamic_shape()) {
      PADDLE_ENFORCE_EQ(axis[0], 0, platform::errors::InvalidArgument(
                                        "TRT static shape mode doesn't support "
                                        "transposing the batch dimension."));
      for (size_t i = 1; i < axis.size(); i++) {
        axis[i]--;
      }
    }

    nvinfer1::Permutation perm;
    for (int i = 0; i < dims; i++) {
      int j = engine_->with_dynamic_shape() ? i : i + 1;
      perm.order[i] = axis[j];
    }
    PADDLE_ENFORCE_EQ(
        IsValidPermutation(dims, perm), true,
        platform::errors::InvalidArgument("Invalid permutation in transpose."));
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    layer->setFirstTranspose(perm);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "transpose", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(transpose, TransposeOpConverter);
