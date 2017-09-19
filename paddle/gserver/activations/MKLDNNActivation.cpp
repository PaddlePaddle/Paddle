/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MKLDNNActivation.h"
#include "mkldnn.hpp"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {

static ClassRegistrar<ActivationFunction> gMKLDNNActivationRegistrar;
/**
 * @def MKLDNN_ACTIVATION_CLASS_NAME
 * @note MKLDNN_ACTIVATION_CLASS_NAME(relu) relu_;
 * means mkldnn_reluActivation relu_;
 */
#define MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE) mkldnn_##ACT_TYPE##Activation

/**
 * @def DEFINE_MKLDNN_ELTWISE_ACTIVATION
 */
#define DEFINE_MKLDNN_ELTWISE_ACTIVATION(ACT_TYPE, ALPHA, BWD_ALPHA)        \
  class MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)                              \
      : public MKLDNNEltwiseActivation {                                    \
  private:                                                                  \
    static const std::string name;                                          \
    static const float alpha;                                               \
    static const float bwdAlpha;                                            \
                                                                            \
  public:                                                                   \
    const std::string& getName() const { return name; }                     \
    float getAlpha() const { return alpha; }                                \
    float getBwdAlpha() const { return bwdAlpha; }                          \
  };                                                                        \
  const std::string MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::name =          \
      "mkldnn_" #ACT_TYPE;                                                  \
  const float MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::alpha = ALPHA;        \
  const float MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::bwdAlpha = BWD_ALPHA; \
  static InitFunction __reg_activation__mkldnn_##ACT_TYPE([] {              \
    gMKLDNNActivationRegistrar                                              \
        .registerClass<MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)>(             \
            "mkldnn_" #ACT_TYPE);                                           \
  });

/**
 * @brief MKLDNN Relu Activation.
 * Actually mkldnn_relu is Leaky Relu.
 *  f(x) = x                   (x >= 0)
 *  f(x) = negative_slope * x  (x <  0)
 * @note the negative_slope should be -0.f in forward
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(relu, -0.f, 0.f)

/**
 * @brief MKLDNN Tanh Activation.
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(tanh, 0.f, 0.f)

/**
 * @brief MKLDNN ELU(Exponential Linear Unit) Activation.
 *  f(x) = x                              (x >= 0)
 *  f(x) = negative_slope * (exp(x) - 1)  (x <  0)
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(elu, 0.f, 0.f)

ActivationFunction* MKLDNNActivation::create(const std::string& type) {
  return gMKLDNNActivationRegistrar.createByType(type);
}

std::vector<std::string> MKLDNNActivation::getAllRegisteredTypes() {
  std::vector<std::string> types;
  gMKLDNNActivationRegistrar.forEachType(
      [&](const std::string& type) { types.push_back(type); });
  return types;
}

}  // namespace paddle
