/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "Layer.h"

namespace paddle {

/**
 * Organize the classes into a binary tree. At each node, a sigmoid function
 * is used to calculate the probability of belonging to the right branch.
 * This idea is from "F. Morin, Y. Bengio (AISTATS 05):
 * Hierarchical Probabilistic Neural Network Language Model."
 *
 * Here we uses a simple way of making the binary tree.
 * Assuming the number of classes C = 6,
 * The classes are organized as a binary tree in the following way:
 *
 * @code{.py}
 * *-*-*- 2
 * | | |- 3
 * | |
 * | |-*- 4
 * |   |- 5
 * |
 * |-*- 0
 *   |- 1
 * @endcode
 *
 * where * indicates an internal node, and each leaf node represents a class.
 * - Node 0 ... C-2 are internal nodes.
 * - Node C-1 ... 2C-2 are leaf nodes.
 * - Class c is represented by leaf node \f$c+C-1\f$.
 *
 * We assign an id for each node:
 * - the id of root be 0.
 * - the left child of a node i is 2*i+1.
 * - the right child of a node i is 2*i+2.
 *
 * It's easy to see that:
 * - the parent of node i is \f$\left\lfloor(i-1)/2\right\rfloor\f$.
 * - the j-th level ancestor of node i is
 * \f$\left\lfloor(i+1)/2^{j+1}\right\rfloor - 1\f$.
 * - A node i is a left child of its parent if \f$(i-1)\%2==0\f$.
 *
 * The config file api is hsigmod_layer.
 */
class HierarchicalSigmoidLayer : public Layer {
 public:
  explicit HierarchicalSigmoidLayer(const LayerConfig& config)
      : Layer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

 protected:
  /**
   * The last of inputs is label layer.
   */
  LayerPtr getLabelLayer() { return inputLayers_.back(); }

  WeightList weights_;
  std::unique_ptr<Weight> biases_;
  /// number of classes
  size_t numClasses_;
  /// codeLength_ = \f$1 + \left\lfloor log_{2}(numClasses-1)\right\rfloor\f$
  int codeLength_;
  /// temporary result of output_
  Argument preOutput_;

  /// The temporary variables in CPU memory.
  MatrixPtr cpuWeight_;
  MatrixPtr cpuWeightGrad_;
  MatrixPtr cpuInput_;
  MatrixPtr cpuInputGrad_;
  MatrixPtr cpuBias_;
  MatrixPtr cpuOutput_;
  IVectorPtr cpuLabel_;
};

}  // namespace paddle
