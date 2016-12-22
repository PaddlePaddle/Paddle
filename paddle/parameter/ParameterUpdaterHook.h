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

#include "ParameterConfig.pb.h"

namespace paddle {

class Parameter;

/**
 * The parameter updater hook interface.
 *
 * The Parameter Updater hooks is a group of methods invoke before
 * ParameterUpdater::updateImpl. It can modify gradient/momentum/etc before
 * parameter optimization.
 */
class IParameterUpdaterHook {
public:
  virtual ~IParameterUpdaterHook();

  /**
   * Create A ParameterUpdaterHook.
   *
   * The same parameter shared the same hooks. So it returns shared_ptr.
   *
   * @param param_config The parameter config.
   * @param idx  The element index of param_config.updater_hooks() array.
   */
  static std::shared_ptr<IParameterUpdaterHook> create(
      const ParameterConfig& paramConfig, int idx);

  /**
   * The update hook method. Invoke before ParameterUpdater::updateImpl
   */
  virtual void update(Parameter* para) = 0;

  /**
   * The init hook method. Invoke in ParameterUpdater::init
   */
  virtual void init(Parameter* para) = 0;

protected:
  /**
   * Ctor.
   */
  IParameterUpdaterHook();
};

}  // namespace paddle
