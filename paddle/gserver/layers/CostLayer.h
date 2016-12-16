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

#include <memory>
#include <vector>
#include "Layer.h"

namespace paddle {

/**
 * Base class for a particular type of cost layer.
 * This type of cost should have one data layer, one label layer
 * and an optional weight layer as input.
 * The derived class should implemnt forwardImp() and backwardImp()
 * which calculate the cost for data and label. The weight is automatically
 * handled by the base class.
 */
class CostLayer : public Layer {
public:
  explicit CostLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getOutputLayer() { return inputLayers_[0]; }

  LayerPtr getLabelLayer() { return inputLayers_[1]; }

  virtual void forward(PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  virtual void forwardImp(Matrix& outputValue,
                          Argument& label,
                          Matrix& cost) = 0;

  virtual void backwardImp(Matrix& outputValue,
                           Argument& label,
                           Matrix& outputGrad) = 0;

protected:
  LayerPtr weightLayer_;
  real coeff_;
};

/**
 * The cross-entropy loss for multi-class classification task.
 * The loss function is:
 *
 * \f[
 * L = - \sum_{i}{t_{k} * log(P(y=k))}
 * \f]
 */
class MultiClassCrossEntropy : public CostLayer {
public:
  explicit MultiClassCrossEntropy(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

/**
 * The cross-entropy with self-normalization for multi-class classification.
 *
 * The loss function is:
 * \f[
 * L = \sum_{i}[-log(P(x_{i})) + alpha * log(Z(x_{i})^2)]
 * \f]
 *
 * The \f$Z(x)\f$ is the softmax normalizer.
 *
 * [1] Jacob Devlin, Rabih Zbib, Zhongqiang Huang, Thomas Lamar,
 *     Richard Schwartz, and John Makhoul. Fast and robust neural
 *     network joint models for statistical machine translation.
 *     In Proceedings of the ACL 2014 Conference.
 */
class MultiClassCrossEntropyWithSelfNorm : public CostLayer {
public:
  explicit MultiClassCrossEntropyWithSelfNorm(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);

protected:
  MatrixPtr sftMaxSum_;
  MatrixPtr sumInv_;
};

/**
 * The cross-entropy for soft binary class.
 * \f[
 * L = \sum_i (\sum_j -y_j(i)*log(x_j(i))-(1-y_j(i))*log(1-x_j(i)))
 * \f]
 */
class SoftBinaryClassCrossEntropy : public CostLayer {
public:
  explicit SoftBinaryClassCrossEntropy(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);

protected:
  MatrixPtr targetPerDim_;
};

/**
 * This cost layer compute Euclidean (L2) loss for real-valued regression
 * tasks.
 * \f[
 * L = \sum_{i=1}^N {|| \hat{y}_i - y_i||_2^2}
 * \f]
 */
class SumOfSquaresCostLayer : public CostLayer {
public:
  explicit SumOfSquaresCostLayer(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

/**
 * A cost layer for learning to rank (LTR) task. This layer contains at leat
 * three inputs.
 * \f[
 *  C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + log(1 + e^{o_{i,j}}) \\
 *  o_{i,j} =  o_i - o_j  \\
 *  \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
 * \f]
 *
 * [1]. Chris Burges, Tal Shaked, Erin Renshaw, et al. Learning to
 *      Rank useing Gradient Descent.
 */
class RankingCost : public Layer {
public:
  explicit RankingCost(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getOutputLayer(size_t i) { return inputLayers_[i]; }

  LayerPtr getLabelLayer() { return inputLayers_[2]; }

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

  void onPassEnd();

  void forwardImp(Matrix& output, Argument& label, Matrix& cost) {
    (void)output;
    (void)label;
    (void)cost;
  }

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad) {
    (void)outputValue;
    (void)label;
    (void)outputGrad;
  }

private:
  double posPairCount_;
  double negPairCount_;
  MatrixPtr margin_;
  MatrixPtr marginGrad_;
  /// if input label is put in ids (not value), copy to this buffer.
  MatrixPtr labelBuf_;
  LayerPtr weightLayer_;
};

/**
 * LambdaRank os a method for learning arbitrary information retrieval
 * measures. It can be applied to any algorithm that learns through gradient
 * descent. LambdaRank is a listwise method, in that the cost depends on the
 * sorted order of the documents. LambdaRank gives the gradient of cost
 * function:
 *
 * \f[
 * \lambda_{ij} = \frac{1}{1 + e^{o_i - o_j}} \left| \Delta_{NDCG} \right|
 * \f]
 *
 * [1] Christopher J.C. Burges, Robert Ragno, Quoc Viet Le. Learning to Rank
 *     with Nonsmooth Cost Functions.
 */
class LambdaCost : public Layer {
public:
  explicit LambdaCost(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getOutputLayer() { return inputLayers_[0]; }

  LayerPtr getScoreLayer() { return inputLayers_[1]; }

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr);

  void onPassEnd();

  real calcNDCG(const real* outputScore, const real* score, int size);
  void calcGrad(const real* outputScore,
                const real* score,
                real* gradData,
                int size);

private:
  MatrixPtr marginGrad_;
  int truncationSize_;
  int maxSortSize_;
  std::vector<std::pair<real, int>> scorePair_;
  std::vector<std::pair<real, int>> outputScorePair_;
  std::vector<real> scoreVec_;
};

/**
 * Cross entropy for multi binary labels.
 * \f[
 * cost[i] = -sum(label[i][j]*log(output[i][j]) +
 *            (1-label[i][j])*log(1-output[i][j]))
 * \f]
 */
class MultiBinaryLabelCrossEntropy : public CostLayer {
protected:
  MatrixPtr targetPerDim_;

public:
  explicit MultiBinaryLabelCrossEntropy(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

/**
 * Huber loss for robust 2-classes classification.
 *
 * For label={0, 1}, let y=2*label-1. Given output f, the loss is:
 * \f[
 * Loss =
 * \left\{\begin{matrix}
 *  4 * y * f     &   \textit{if}  \ \  y* f < -1 \\
 *  (1 - y * f)^2 &  \textit{if}   \ \  -1 < y * f < 1  \\
 *  0             &                    \textit{otherwise}
 * \end{matrix}\right.
 * \f]
 */
class HuberTwoClass : public CostLayer {
  std::vector<Argument> tmpCpuInput_;

public:
  explicit HuberTwoClass(const LayerConfig& config) : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void forwardImpIn(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);

  void backwardImpIn(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

typedef std::shared_ptr<CostLayer> CostLayerPtr;
}  // namespace paddle
