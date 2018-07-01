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

#include "SelectiveFullyConnectedLayer.h"
#include <algorithm>
#include <vector>
#include "paddle/math/SparseMatrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(selective_fc, SelectiveFullyConnectedLayer);

bool SelectiveFullyConnectedLayer::init(const LayerMap& layerMap,
                                        const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  inputNum_ = inputLayers_.size();
  if (config_.has_selected_colums()) {
    inputNum_ -= 1;
  }
  for (size_t i = 0; i < inputNum_; i++) {
    size_t height = inputLayers_[i]->getSize();
    size_t width = getSize();
    // NOTE weight is transpoed
    weights_.emplace_back(new Weight(width, height, parameters_[i]));
  }

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  fullOutput_ = false;

  return true;
}

void SelectiveFullyConnectedLayer::prefetch() {}

void SelectiveFullyConnectedLayer::reserveOutput(size_t height,
                                                 size_t width,
                                                 size_t nnz) {
  bool flag = (passType_ == PASS_TEST &&
               config_.selective_fc_pass_generation() && !fullOutput_);
  SetDevice device(output_.deviceId);
  if (flag) {
    // output_.value is sparse matrix
    if (dynamic_cast<CpuMatrix*>(output_.value.get()) ||
        dynamic_cast<GpuMatrix*>(output_.value.get())) {
      output_.value = nullptr;
    }
    Matrix::resizeOrCreateSparseMatrix(output_.value,
                                       height,
                                       width,
                                       nnz,
                                       FLOAT_VALUE,
                                       SPARSE_CSR,
                                       /*trans=*/false,
                                       /*useGpu=*/useGpu_);
    output_.value->copyFrom(*selCols_);
    interOutput_ = output_.value;
  } else {
    if (fullOutput_) {
      // output_.value is dense matrix
      if (dynamic_cast<CpuSparseMatrix*>(output_.value.get()) ||
          dynamic_cast<GpuSparseMatrix*>(output_.value.get())) {
        output_.value = nullptr;
      }
      Matrix::resizeOrCreate(output_.value,
                             height,
                             width,
                             /*trans=*/false,
                             /*useGpu=*/useGpu_);
      interOutput_ = output_.value;
    } else {
      // output_.value is dense matrix, but width = nnz /height
      CHECK_EQ(nnz % height, 0U);
      CHECK(nnz / height);
      Matrix::resizeOrCreate(output_.value,
                             height,
                             nnz / height,
                             /*trans=*/false,
                             /*useGpu=*/useGpu_);
      interOutput_ = Matrix::createSparseMatrix(output_.value->getData(),
                                                selCols_->getRows(),
                                                selCols_->getCols(),
                                                height,
                                                width,
                                                nnz,
                                                FLOAT_VALUE,
                                                SPARSE_CSR,
                                                /*trans=*/false,
                                                /*useGpu=*/useGpu_);
    }
  }
  interOutput_->zeroMem();

  if (passType_ != PASS_TEST && needGradient()) {
    CHECK_EQ(nnz % height, 0U) << "during training, each sample must have a "
                                  "same number of selected columns.";
    CHECK(nnz / height)
        << "during training, "
           "each sample must have at least one column selected.";
    Matrix::resizeOrCreate(output_.grad,
                           height,
                           nnz / height,
                           /*trans=*/false,
                           /*useGpu=*/useGpu_);
    output_.grad->zeroMem();
  }
}

void SelectiveFullyConnectedLayer::forward(PassType passType) {
  REGISTER_TIMER("selective_fc.forward");
  Layer::forward(passType);

  getSelectiveCols();
  size_t height = getInput(0).getBatchSize();
  size_t width = getSize();
  size_t nnz = height * width;
  if (!fullOutput_) {
    CHECK(selCols_);
    CHECK(height == selCols_->getHeight());
    CHECK(width == selCols_->getWidth());
    nnz = selCols_->getElementCnt();
  }

  // Layer::ResetOutput(), here we set outV/outG as SparseMatrix manually
  // this outV should be used as input of MaxIdLayer and softmax activation
  reserveOutput(height, width, nnz);

  bool flag = true;
  for (size_t i = 0; i < inputNum_; i++) {
    MatrixPtr input = getInputValue(i);
    MatrixPtr weight = weights_[i]->getW();
    size_t hsize = input->getHeight();
    size_t wsize = weight->getHeight();
    real scaleT = i == 0 ? real(0) : real(1);

    flag = nnz < (hsize * wsize) * config_.selective_fc_full_mul_ratio() &&
           !fullOutput_;
    if (flag) {
      // if the indecies are highly sparse,
      // manully compute the multiplication of
      // the input vector and the selected rows.
      REGISTER_TIMER("selective.plain");
      interOutput_->mul(*input, *weight->getTranspose(), 1, scaleT);
    } else {
      // if the indecies is not sparse enough,
      // use full mul instead
      REGISTER_TIMER("selective.mul");
      if (fullOutput_) {
        interOutput_->mul(*input, *weight->getTranspose(), 1, scaleT);
      } else {
        Matrix::resizeOrCreate(mmat_,
                               hsize,
                               wsize,
                               /*trans=*/false,
                               /*useGpu=*/useGpu_);
        mmat_->mul(*input, *weight->getTranspose());
        interOutput_->add3(mmat_);
      }
    }
  }

  if (biases_) {
    interOutput_->addBias(*(biases_->getW()), 1);
  }

  flag = (passType_ == PASS_TEST && config_.selective_fc_pass_generation() &&
          !fullOutput_);
  if (flag) {
    // during generation, output of this layer is a sparse csr matrix,
    // which is probably the input of maxid layer
    // if the model is trained with multi-class-cross-entroy-with-selfnorm,
    // activiation of this layer should be exponential, not softmax.

    Argument arg;
    arg.value = Matrix::create(interOutput_->getData(),
                               1,
                               nnz,
                               /*trans=*/false,
                               /*useGpu=*/useGpu_);
    //! TODO(yuyang18): Why we cannot invoke forwardActivation here?
    activation_->forward(arg).check();
  } else /* train and test in train, not generating */ {
    // during training, this layer output value is *Matrix*, which is input of
    // eg. multi-class-cross-entropy

    // while training, every sample has a equal number of selected
    // columns to be activated.
    // note indices of multi-class-cross-entropy need to be remapped
    // to this index.
    // e.g. sample = [1,3,5] and 3 is gold, then label is 1

    forwardActivation();
  }
}

void SelectiveFullyConnectedLayer::backward(const UpdateCallback& callback) {
  backwardActivation();
  MatrixPtr oGrad = getOutputGrad();
  if (!fullOutput_) {
    interOutGrad_ = Matrix::createSparseMatrix(oGrad->getData(),
                                               interOutput_->getRows(),
                                               interOutput_->getCols(),
                                               interOutput_->getHeight(),
                                               interOutput_->getWidth(),
                                               interOutput_->getElementCnt(),
                                               FLOAT_VALUE,
                                               SPARSE_CSR,
                                               /*trans=*/false,
                                               /*useGpu=*/useGpu_);
  } else {
    interOutGrad_ = Matrix::create(oGrad->getData(),
                                   oGrad->getHeight(),
                                   oGrad->getWidth(),
                                   /*trans=*/false,
                                   /*useGpu=*/useGpu_);
  }

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*interOutGrad_, 1);
    biases_->getParameterPtr()->incUpdate(callback);
  }

  // backward is different from FullyConnectedLayer
  // because the weight is transposed
  for (size_t i = 0; i < inputNum_; i++) {
    AsyncGpuBlock block;
    MatrixPtr preGrad = getInputGrad(i);
    if (preGrad) {
      REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      preGrad->mul(*interOutGrad_, *weights_[i]->getW(), 1, 1);
    }

    MatrixPtr wGrad = weights_[i]->getWGrad();
    if (wGrad) {
      REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
      MatrixPtr input = getInputValue(i);
      wGrad->mul(*interOutGrad_->getTranspose(), *input, 1, 1);
    }

    {
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void paddle::SelectiveFullyConnectedLayer::fillSelectiveData(
    const std::shared_ptr<std::vector<std::pair<int*, size_t>>>& candidates) {
  if (candidates == nullptr) {
    fillFullySelectiveData();
    return;
  }

  size_t sampleNum = candidates->size();
  size_t outputWidth = getSize();
  size_t nnz =
      std::accumulate(candidates->begin(),
                      candidates->end(),
                      0UL,
                      [](size_t a, const std::pair<int*, size_t>& arr) {
                        return a + arr.second;
                      });

  Matrix::resizeOrCreateSparseMatrix(this->cpuSelCols_,
                                     sampleNum,
                                     outputWidth,
                                     nnz,
                                     NO_VALUE,
                                     SPARSE_CSR,
                                     false,
                                     false);
  CHECK(this->cpuSelCols_ != nullptr);
  CpuSparseMatrixPtr selCols =
      std::dynamic_pointer_cast<CpuSparseMatrix>(cpuSelCols_);
  int* rowOffsets = selCols->getRows();
  int* colIndices = selCols->getCols();

  rowOffsets[0] = 0;
  int idx = 0;
  for (size_t i = 0; i < sampleNum; ++i) {
    if ((*candidates)[i].second > 0) {
      rowOffsets[i + 1] = rowOffsets[i] + (*candidates)[i].second;
      for (size_t j = 0; j < (*candidates)[i].second; ++j) {
        colIndices[idx] = (*candidates)[i].first[j];
        idx++;
      }
    } else {
      rowOffsets[i + 1] = rowOffsets[i];
    }
  }

  CHECK_EQ(static_cast<size_t>(rowOffsets[sampleNum]), nnz);
  if (!useGpu_) {
    this->selCols_ = this->cpuSelCols_;
  } else {
    Matrix::resizeOrCreateSparseMatrix(this->selCols_,
                                       sampleNum,
                                       outputWidth,
                                       nnz,
                                       NO_VALUE,
                                       SPARSE_CSR,
                                       false,
                                       true);
    this->selCols_->copyFrom(*cpuSelCols_, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);
  }

  fullOutput_ = false;
}

void paddle::SelectiveFullyConnectedLayer::getSelectiveCols() {
  if (config_.has_selected_colums()) {
    this->selCols_ = inputLayers_[inputNum_]->getOutputValue();
    fullOutput_ = false;
  } else if (!config_.selective_fc_pass_generation() || selCols_ == nullptr) {
    this->fillFullySelectiveData();
  }  // else selCols_ is initialized by fillSelectiveData
}

}  // namespace paddle
