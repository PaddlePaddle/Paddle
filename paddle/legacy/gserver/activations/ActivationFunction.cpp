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

#include "ActivationFunction.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include "paddle/parameter/Argument.h"
#include "paddle/utils/ClassRegistrar.h"
#include "paddle/utils/Logging.h"

#ifdef PADDLE_WITH_MKLDNN
#include "MKLDNNActivation.h"
#endif

namespace paddle {

static ClassRegistrar<ActivationFunction> gActivationRegistrar;
/**
 * @def ACTIVATION_CLASS_NAME
 * @brief Macro for getting derived activation class name
 * @note ACTIVATION_CLASS_NAME(softmax) softmax_;
 * means softmaxActivation softmax_;
 */
#define ACTIVATION_CLASS_NAME(ACTIVATION_NAME) ACTIVATION_NAME##Activation
/**
 * @def BEGIN_DEFINE_ACTIVATION
 * @brief Macro for defining a devried activation class
 */
#define BEGIN_DEFINE_ACTIVATION(ACTIVATION_NAME)                             \
  class ACTIVATION_CLASS_NAME(ACTIVATION_NAME) : public ActivationFunction { \
   private:                                                                  \
    static const std::string name;                                           \
                                                                             \
   public:                                                                   \
    const std::string& getName() const { return name; }
/**
 * @def END_DEFINE_ACTIVATION
 * @brief Macro for registering a derived activation class
 */
#define END_DEFINE_ACTIVATION(ACTIVATION_NAME)                     \
  }                                                                \
  ;                                                                \
  const std::string ACTIVATION_CLASS_NAME(ACTIVATION_NAME)::name = \
      #ACTIVATION_NAME;                                            \
  static InitFunction __reg_activation__##ACTIVATION_NAME([] {     \
    gActivationRegistrar                                           \
        .registerClass<ACTIVATION_CLASS_NAME(ACTIVATION_NAME)>(    \
            #ACTIVATION_NAME);                                     \
  });

/**
 * @brief The IdentityActivation class
 *
 * Do nothing when forward/backward.
 */
class IdentityActivation : public ActivationFunction {
 public:
  static const std::string name;
  Error __must_check forward(Argument& act) {
    (void)act;
    return Error();
  }
  Error __must_check backward(Argument& act) {
    (void)act;
    return Error();
  }
  const std::string& getName() const { return name; }
};
const std::string IdentityActivation::name = "";
static InitFunction __reg_activation__identity([] {
  gActivationRegistrar.registerClass<IdentityActivation>("");
  gActivationRegistrar.registerClass<IdentityActivation>("linear");
});

/**
 * @brief Sigmoid Activation
 * \f[
 * f(z) = \frac{1}{1+exp(-z)}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(sigmoid)
Error __must_check forward(Argument& act) {
  act.value->sigmoid(*act.value);
  return Error();
}
Error __must_check backward(Argument& act) {
  act.grad->sigmoidDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(sigmoid)

/**
 * @brief Softmax Activation
 * \f[
 * P(y=j|x) = \frac{e^{x^Tw_j}}{\sum^K_{k=1}e^{x^Tw_k}}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(softmax)
private:
MatrixPtr sftMaxSum_;
MatrixPtr sftMaxDot_;

public:
Error __must_check forward(Argument& act) {
  act.value->softmax(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  MatrixPtr outputV = act.value;
  MatrixPtr outputG = act.grad;

  if (outputG->useGpu()) {
    outputG->softmaxBackward(*outputV);
  } else {
    SetDevice device(act.deviceId);
    Matrix::resizeOrCreate(sftMaxDot_,
                           outputG->getHeight(),
                           outputG->getWidth(),
                           /* trans */ false,
                           useGpu(act.deviceId));
    Matrix::resizeOrCreate(sftMaxSum_,
                           outputG->getHeight(),
                           1,
                           /* trans */ false,
                           useGpu(act.deviceId));

    sftMaxDot_->dotMul(*outputG, *outputV);
    sftMaxSum_->colMerge(*sftMaxDot_);

    act.grad->softmaxDerivative(*act.value, *sftMaxSum_);
  }
  return Error();
}
END_DEFINE_ACTIVATION(softmax)

/**
 * @brief Sequence_softmax Activation
 * @note Softmax on all frames of one sequence.
 * Width of frame must be one.
 */
BEGIN_DEFINE_ACTIVATION(sequence_softmax)
private:
ACTIVATION_CLASS_NAME(softmax) softmax_;
Argument argument_;

public:
Error __must_check forward(Argument& act) {
  if (act.value->getWidth() != 1UL) {
    return Error(
        "Input width for each timestep of sequence softmax should be 1");
  }

  if (!argument_.value) {
    argument_.value = Matrix::create(nullptr,
                                     /* height= */ 1,
                                     1,
                                     /* trans= */ false,
                                     useGpu(act.deviceId));
    argument_.grad = Matrix::create(nullptr,
                                    /* height= */ 1,
                                    1,
                                    /* trans= */ false,
                                    useGpu(act.deviceId));
  }

  auto starts =
      act.hasSubseq()
          ? act.subSequenceStartPositions->getVector(useGpu(act.deviceId))
          : act.sequenceStartPositions->getVector(useGpu(act.deviceId));
  act.value->sequenceSoftmax(*act.value, *starts);
  return Error();
}

Error __must_check backward(Argument& act) {
  if (act.value->getWidth() != 1UL) {
    return Error(
        "Input width for each timestep of sequence softmax should be 1");
  }

  size_t numSequences =
      act.hasSubseq() ? act.getNumSubSequences() : act.getNumSequences();
  const int* starts = act.getCpuStartPositions();

  for (size_t i = 0; i < numSequences; ++i) {
    // TODO(Dangqingqing) optimization for GPU
    size_t offset = starts[i];
    size_t size = starts[i + 1] - starts[i];
    argument_.value->setData(act.value->getData() + offset, 1UL, size);
    argument_.grad->setData(act.grad->getData() + offset, 1UL, size);

    Error err = softmax_.backward(argument_);
    if (!err.isOK()) return err;
  }
  return Error();
}
END_DEFINE_ACTIVATION(sequence_softmax)

/*
 * @brief SoftSign Activation.
 * \f[
 * f(z) = \frac{z}{1 + |z|}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(softsign)
private:
MatrixPtr denominator_;

Error __must_check forward(Argument& act) {
  size_t height = act.value->getHeight();
  size_t width = act.value->getWidth();
  Matrix::resizeOrCreate(
      denominator_, height, width, false, useGpu(act.deviceId));
  denominator_->assign(*act.value);
  denominator_->abs2();
  denominator_->add(1.);

  act.value->dotDiv(*act.value, *denominator_);
  return Error();
}

Error __must_check backward(Argument& act) {
  denominator_->square2();
  denominator_->scalarDiv(*denominator_, 1.);
  act.grad->dotMul(*act.grad, *denominator_);
  return Error();
}
END_DEFINE_ACTIVATION(softsign)

/**
 * @brief Relu Activation.
 * forward. y = max(0, z)
 *
 * derivative of relu is:
 *
 *    1 if z > 0
 *
 *    0 otherwise.
 */
BEGIN_DEFINE_ACTIVATION(relu)
Error __must_check forward(Argument& act) {
  act.value->relu(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->reluDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(relu)

/**
 * @brief BRelu Activation.
 *
 * forward. y = min(24, max(0, z))
 *
 * derivative of brelu is:
 *
 *    1 if 0 < z < 24
 *
 *    0 otherwise.
 *
 * TODO(yuyang18): Remove magic number 24 or make it configuable.
 */
BEGIN_DEFINE_ACTIVATION(brelu)
Error __must_check forward(Argument& act) {
  act.value->brelu(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->breluDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(brelu)

/**
 * @brief Tanh Activation.
 * \f[
 * f(z) = tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(tanh)
Error __must_check forward(Argument& act) {
  act.value->tanh(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->tanhDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(tanh)

/**
 * @brief Scaled Tanh Activation
 * \f[
 * f(z) = 1.7159 * tanh(2/3*z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(stanh)
private:
real a, b;

public:
ACTIVATION_CLASS_NAME(stanh)() : a(1.7159), b(2. / 3.) {}
Error __must_check forward(Argument& act) {
  act.value->scaledTanh(*act.value, a, b);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->scaledTanhDerivative(*act.value, a, b);
  return Error();
}
END_DEFINE_ACTIVATION(stanh)

/**
 * @brief Soft Relu Activation.
 * \f[
 * f(z) = ln(1+e^z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(softrelu)
Error __must_check forward(Argument& act) {
  act.value->softrelu(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->softreluDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(softrelu)

/**
 * @brief Abs Activation.
 * Forward: f(z) = abs(z)
 *
 * Derivative:
 *
 *     1   if z>0
 *
 *    -1   if z<0
 *
 *     0   if z=0
 */
BEGIN_DEFINE_ACTIVATION(abs)
Error __must_check forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in,
                         act.value->getHeight(),
                         act.value->getWidth(),
                         /* trans */ false,
                         useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->abs2(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->absDerivative(*act.in);
  return Error();
}
END_DEFINE_ACTIVATION(abs)

/**
 * @brief Square Activation.
 * \f[
 * f(z) = z^2.
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(square)
Error __must_check forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in,
                         act.value->getHeight(),
                         act.value->getWidth(),
                         /* trans */ false,
                         useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->square2(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->squareDerivative(*act.in);
  return Error();
}
END_DEFINE_ACTIVATION(square)

/**
 * @brief Exponential Activation.
 * \f[
 * f(z) = e^z
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(exponential)
Error __must_check forward(Argument& act) {
  act.value->exp2(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->expDerivative(*act.value);
  return Error();
}
END_DEFINE_ACTIVATION(exponential)

/**
 * @brief Reciprocal Activation.
 * \f[
 * f(z) = 1/z
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(reciprocal)
Error __must_check forward(Argument& act) {
  act.value->reciprocal2();
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->dotMulSquare(*act.value);
  act.grad->neg();
  return Error();
}
END_DEFINE_ACTIVATION(reciprocal)

/**
 * @brief Square Root Activation.
 * \f[
 * f(z) = sqrt(z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(sqrt)
Error __must_check forward(Argument& act) {
  act.value->sqrt2();
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->dotDiv(*act.grad, *act.value);
  act.grad->mulScalar(0.5);
  return Error();
}
END_DEFINE_ACTIVATION(sqrt)

/**
 * @brief Logarithm Activation.
 * \f[
 * f(z) = log(z)
 * \f]
 */
BEGIN_DEFINE_ACTIVATION(log)
Error __must_check forward(Argument& act) {
  SetDevice device(act.deviceId);
  Matrix::resizeOrCreate(act.in,
                         act.value->getHeight(),
                         act.value->getWidth(),
                         /* trans */ false,
                         useGpu(act.deviceId));

  act.in->copyFrom(*act.value);
  act.value->log2(*act.value);
  return Error();
}

Error __must_check backward(Argument& act) {
  act.grad->dotDiv(*act.grad, *act.in);
  return Error();
}
END_DEFINE_ACTIVATION(log)

ActivationFunction* ActivationFunction::create(const std::string& type) {
#ifdef PADDLE_WITH_MKLDNN
  if (!type.empty() && type.compare(0, 7, "mkldnn_") == 0) {
    return MKLDNNActivation::create(type);
  }
#endif

  return gActivationRegistrar.createByType(type);
}

std::vector<std::string> ActivationFunction::getAllRegisteredTypes() {
  std::vector<std::string> types;
  gActivationRegistrar.forEachType(
      [&](const std::string& type) { types.push_back(type); });
  return types;
}

}  // namespace paddle
