/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

  virtual void forwardImp(Matrix& outputValue, Argument& label,
                          Matrix& cost) = 0;

  virtual void backwardImp(Matrix& outputValue, Argument& label,
                           Matrix& outputGrad) = 0;

protected:
  LayerPtr weightLayer_;
  real coeff_;
};

/*
 * MultiClassCrossEntropy
 */
class MultiClassCrossEntropy : public CostLayer {
public:
  explicit MultiClassCrossEntropy(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

/*
 * MultiClassCrossEntropyWithSelfNorm
 * \sum_i (-log(x_label(i)) + alpha * log(Z(i)^2)
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

/*
 * SoftBinaryClassCrossEntropy
 *  \sum_i (\sum_j -y_j(i)*log(x_j(i))-(1-y_j(i))*log(1-x_j(i)))
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

class SumOfSquaresCostLayer : public CostLayer {
public:
  explicit SumOfSquaresCostLayer(const LayerConfig& config)
      : CostLayer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forwardImp(Matrix& output, Argument& label, Matrix& cost);

  void backwardImp(Matrix& outputValue, Argument& label, Matrix& outputGrad);
};

/*
 * RankingCost
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
  // if input label is put in ids (not value), copy to this buffer.
  MatrixPtr labelBuf_;
  LayerPtr weightLayer_;
};

/* lambdaRank listwise LTR approach */
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
  void calcGrad(const real* outputScore, const real* score, real* gradData,
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
 * Cross entropy for multi binary labels
 * cost[i] = -sum(label[i][j]*log(output[i][j])
 *                + (1-label[i][j])*log(1-output[i][j]))
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

/*
 * Huber loss for robust 2-classes classification
 *
 * For label={0, 1}, let y=2*label-1. Given output f, the loss is:
 * -4*y*f, if y*f < -1
 * (1-y*f)^2, if -1 < y*f < 1,
 * 0, otherwise
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
