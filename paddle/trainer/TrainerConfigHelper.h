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

#include <paddle/utils/Logging.h>
#include <paddle/utils/Util.h>
#include <memory>

namespace paddle {

class TrainerConfig;
class OptimizationConfig;
struct TrainerConfigHelperPrivate;
class ModelConfig;
class DataConfig;

/**
 * @brief TrainerConfig Helper. A class wrap protobuf's TrainerConfig Object,
 * simplize the usage for TrainerConfig.
 *
 * The all operation to TrainerConfig object should use this object. It remove
 * many copy & paste code in trainer.
 *
 * @TODO(yuyang18): Make cmake check compiler support keyword 'final' or not.
 * Define a macro to unify 'final' keyword
 */
class TrainerConfigHelper /*final*/ {
 public:
  DISABLE_COPY(TrainerConfigHelper);

  /**
   * @brief Ctor, Create a TrainerConfig from config file
   * @param configFilePath Config file path.
   */
  explicit TrainerConfigHelper(const std::string& configFilePath);
  explicit TrainerConfigHelper(const TrainerConfig& config);

  /**
   * Dtor
   * @warning this class is a final class. Should not be inherited.
   */
  ~TrainerConfigHelper();

  /**
   * @brief Get Trainer Config itself.
   */
  const TrainerConfig& getConfig() const;

  TrainerConfig& getMutableConfig();

  /**
   * @brief Get Optimizer Config.
   */
  const OptimizationConfig& getOptConfig() const;

  /**
   * @brief Get Model Config.
   */
  const ModelConfig& getModelConfig() const;

  /**
   * @brief Get Train Data Config Pointer.
   * @return nullptr if there is no train data. Else will return pointer
   */
  const DataConfig* getDataConfigPtr() const;

  /**
   * @brief Get Tain Data Config.
   * @warning Core when there is no train data.
   */
  const DataConfig& getDataConfig() const {
    CHECK(this->hasDataConfig());
    auto conf = this->getDataConfigPtr();
    return *conf;
  }

  /**
   * @brief Get test data config
   * @warning Core when there is no test data.
   */
  const DataConfig& getTestDataConfig() const;

  /**
   * @brief Has train data config or not.
   * @return true if has train data.
   */
  bool hasDataConfig() const;

  /**
   * @brief Has test data config or not.
   * @return true if has test data.
   */
  bool hasTestDataConfig() const;

  /**
   * @brief Update trainer config from command line flags.
   *        Override config's (save_dir, init_model_path, start_pass) if command
   *        flags is existed.
   */
  void updateConfigFromFlags();

  /**
   * @brief Disable optimization's sparse remote update.
   */
  void disableRemoteSparseUpdater();

  /**
   * @brief Disable optimization and each parameter's sparse remote update.
   */
  void disableRemoteSparseUpdaterForEachParams();

  /**
   * @brief implicit conversion.
   */
  inline operator const TrainerConfig&() const { return this->getConfig(); }

  /**
   * @brief implicit conversion.
   */
  inline operator const OptimizationConfig&() const {
    return this->getOptConfig();
  }

  /**
   * @brief implicit conversion.
   */
  inline operator const DataConfig&() const { return this->getDataConfig(); }

  /**
   * @brief implicit conversion.
   */
  inline operator const ModelConfig&() const { return this->getModelConfig(); }

  /**
   * @brief Get mutable optimization config.
   */
  OptimizationConfig& getOptConfig();

  /**
   * @brief set model save directory.
   * @param saveDir Directory path.
   */
  void setSaveDir(const std::string& saveDir);

  /**
   * @brief get model save directory.
   * @return save directory path.
   */
  const std::string& getSaveDir() const;

  /**
   * @brief Get config file name from model path.
   *
   * Paddle save model to a directory, and write a file 'path.txt' which save
   * config filename.
   *
   * @param modelPath model saved directory.
   * @return config file name.
   */
  static std::string getConfigNameFromPath(const std::string& modelPath);

  /**
   * @brief Get config file name from this config instance.
   * @param[out] ok true if no error.
   * @return config file name.
   */
  std::string getConfigName(bool* ok = nullptr) const;

  /**
   * @brief Try to create TrainerConfigHelper from all command line flags.
   *        Try to load from --config, --init_model_path, --start_pass one by
   *        one. Return nullptr if cannot load TrainerConfigHelper from all
   *        these place.
   * @return nullptr if cannot load, otherwise return a TrainerConfigHelper.
   */
  static std::shared_ptr<TrainerConfigHelper> createFromFlags();

  /**
   * @brief Try to create TrainerConfigHelper only from '--config' flag.
   * @return nullptr if cannot load, otherwise return a TrainerConfigHelper.
   */
  static std::shared_ptr<TrainerConfigHelper> createFromFlagConfig();

 private:
  static std::string getConfigNameFromPassId(int passId,
                                             const std::string& modelPath);

  TrainerConfigHelperPrivate* m;
};

typedef std::shared_ptr<TrainerConfigHelper> TrainerConfigHelperPtr;

}  // namespace paddle
