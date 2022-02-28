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

#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"

namespace {

template <typename Value, typename Lambda>
void RegisterSetter(
    std::map<std::string, std::function<void(Value)>>& options,  // NOLINT
    const std::string& name, Lambda setter) {
  options[name] = setter;
}

template <typename Value, typename Lambda>
void RegisterGetter(
    std::map<std::string, std::function<Value()>>& options,  // NOLINT
    std::map<std::string, std::string>& options_type,        // NOLINT
    const std::string& name, const std::string& type_str, Lambda getter) {
  options[name] = getter;
  options_type[name] = type_str;
}

}  // namespace

namespace paddle {
namespace platform {
namespace ipu {

IpuStrategy::IpuStrategy() {
#define ADD_BOOL_OPTION(name)                                             \
  RegisterSetter(bool_options, #name, [&](bool value) { name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "bool",             \
                 [&]() { return std::to_string(name); })

#define ADD_UINT64_OPTION(name)                                 \
  RegisterSetter(uint64_options, #name,                         \
                 [&](std::uint64_t value) { name = value; });   \
  RegisterGetter(options_getter, options_type, #name, "uint64", \
                 [&]() { return std::to_string(name); })

#define ADD_DOUBLE_OPTION(name)                                               \
  RegisterSetter(double_options, #name, [&](double value) { name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "double",               \
                 [&]() { return std::to_string(name); })

#define ADD_STRING_OPTION(name)                                    \
  RegisterSetter(string_options, #name,                            \
                 [&](const std::string& value) { name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "string",    \
                 [&]() { return name; })

  ADD_BOOL_OPTION(is_training);
  ADD_BOOL_OPTION(save_init_onnx);
  ADD_BOOL_OPTION(save_onnx_checkpoint);
  ADD_BOOL_OPTION(need_avg_shard);
  ADD_BOOL_OPTION(enable_fp16);
  ADD_UINT64_OPTION(num_ipus);
  ADD_UINT64_OPTION(batches_per_step);
  ADD_UINT64_OPTION(micro_batch_size);
  ADD_UINT64_OPTION(save_per_n_step);
  ADD_DOUBLE_OPTION(available_memory_proportion);
  ADD_DOUBLE_OPTION(loss_scaling);
  ADD_DOUBLE_OPTION(max_weight_norm);

#undef ADD_STRING_OPTION
#undef ADD_DOUBLE_OPTION
#undef ADD_UINT64_OPTION
#undef ADD_BOOL_OPTION

#define ADD_POPART_ENUM_OPTION_ALIAS(name, aliased_name, EnumType)        \
  RegisterSetter(uint64_options, #name, [&](std::uint64_t value) {        \
    PADDLE_ENFORCE_LT(                                                    \
        value, static_cast<std::uint64_t>(popart::EnumType::N),           \
        errors::InvalidArgument("Value for %s out of range", #EnumType)); \
    popart_options.aliased_name = static_cast<popart::EnumType>(value);   \
  });                                                                     \
  RegisterGetter(options_getter, options_type, #name, "uint64", [&]() {   \
    return std::to_string(                                                \
        static_cast<std::uint64_t>(popart_options.aliased_name));         \
  })

#define ADD_POPART_BOOL_OPTION_ALIAS(name, aliased_name)                    \
  RegisterSetter(bool_options, #name,                                       \
                 [&](bool value) { popart_options.aliased_name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "bool", [&]() {       \
    return std::to_string(popart_options.aliased_name);                     \
  })

#define ADD_POPART_UINT64_OPTION_ALIAS(name, aliased_name)              \
  RegisterSetter(uint64_options, #name, [&](std::uint64_t value) {      \
    popart_options.aliased_name = value;                                \
  });                                                                   \
  RegisterGetter(options_getter, options_type, #name, "uint64", [&]() { \
    return std::to_string(popart_options.aliased_name);                 \
  })

#define ADD_POPART_DOUBLE_OPTION_ALIAS(name, aliased_name)                    \
  RegisterSetter(double_options, #name,                                       \
                 [&](double value) { popart_options.aliased_name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "double", [&]() {       \
    return std::to_string(popart_options.aliased_name);                       \
  })

#define ADD_POPART_STRING_OPTION_ALIAS(name, aliased_name)              \
  RegisterSetter(string_options, #name, [&](const std::string& value) { \
    popart_options.aliased_name = value;                                \
  });                                                                   \
  RegisterGetter(options_getter, options_type, #name, "string",         \
                 [&]() { return popart_options.aliased_name; })

#define ADD_POPART_ENUM_OPTION(name, EnumType) \
  ADD_POPART_ENUM_OPTION_ALIAS(name, name, EnumType)

#define ADD_POPART_BOOL_OPTION(name) ADD_POPART_BOOL_OPTION_ALIAS(name, name)

#define ADD_POPART_UINT64_OPTION(name) \
  ADD_POPART_UINT64_OPTION_ALIAS(name, name)

#define ADD_POPART_DOUBLE_OPTION(name) \
  ADD_POPART_DOUBLE_OPTION_ALIAS(name, name)

#define ADD_POPART_STRING_OPTION(name) \
  ADD_POPART_STRING_OPTION_ALIAS(name, name)

  ADD_POPART_ENUM_OPTION(autodiffSettings.stitchStrategy,
                         AutodiffStitchStrategy);
  ADD_POPART_ENUM_OPTION(batchSerializationSettings.transformContext,
                         BatchSerializationTransformContext);
  ADD_POPART_ENUM_OPTION(batchSerializationSettings.method,
                         BatchSerializationMethod);
  ADD_POPART_ENUM_OPTION(batchSerializationSettings.batchSchedule,
                         BatchSerializationBatchSchedule);
  ADD_POPART_ENUM_OPTION(autoRecomputation, RecomputationType);
  ADD_POPART_ENUM_OPTION(mergeVarUpdate, MergeVarUpdateType);
  ADD_POPART_ENUM_OPTION(virtualGraphMode, VirtualGraphMode);
  ADD_POPART_ENUM_OPTION(syntheticDataMode, SyntheticDataMode);
  ADD_POPART_ENUM_OPTION(subgraphCopyingStrategy, SubgraphCopyingStrategy);
  ADD_POPART_ENUM_OPTION(accumulationAndReplicationReductionType,
                         ReductionType);
  ADD_POPART_ENUM_OPTION(meanAccumulationAndReplicationReductionStrategy,
                         MeanReductionStrategy);

  ADD_POPART_STRING_OPTION(logDir);
  ADD_POPART_STRING_OPTION(cachePath);
  ADD_POPART_STRING_OPTION(partialsTypeMatMuls);
  ADD_POPART_STRING_OPTION(customCodeletCompileFlags);
  ADD_POPART_STRING_OPTION(serializedPoprithmsShiftGraphsDir);
  ADD_POPART_STRING_OPTION(kahnTieBreaker);

  ADD_POPART_UINT64_OPTION(executionPhaseSettings.phases);
  ADD_POPART_UINT64_OPTION(executionPhaseSettings.stages);
  ADD_POPART_UINT64_OPTION(batchSerializationSettings.factor);
  ADD_POPART_UINT64_OPTION(firstDotOp);
  ADD_POPART_UINT64_OPTION(finalDotOp);
  ADD_POPART_UINT64_OPTION(numIOTiles);
  ADD_POPART_UINT64_OPTION(mergeVarUpdateMemThreshold);
  ADD_POPART_UINT64_OPTION(looseThresholdAtPeak);
  ADD_POPART_UINT64_OPTION(accumulationFactor);
  ADD_POPART_UINT64_OPTION(swapLimitScheduler);
  ADD_POPART_UINT64_OPTION(globalReplicationFactor);
  ADD_POPART_UINT64_OPTION(globalReplicaOffset);
  ADD_POPART_UINT64_OPTION(defaultPrefetchBufferingDepth);
  ADD_POPART_UINT64_OPTION(compilationProgressTotal);
  ADD_POPART_UINT64_OPTION(transitiveClosureOptimizationThreshold);

  ADD_POPART_BOOL_OPTION(batchSerializationSettings.concatOnVirtualGraphChange);
  ADD_POPART_BOOL_OPTION(
      batchSerializationSettings.concatOnExecutionPhaseChange);
  ADD_POPART_BOOL_OPTION(
      batchSerializationSettings.concatOnPipelineStageChange);
  ADD_POPART_BOOL_OPTION(strictOpVersions);
  ADD_POPART_BOOL_OPTION(opxAliasChecking);
  ADD_POPART_BOOL_OPTION(opxModifyChecking);
  ADD_POPART_BOOL_OPTION(dotOpNames);
  ADD_POPART_BOOL_OPTION(exportPoplarComputationGraph);
  ADD_POPART_BOOL_OPTION(exportPoplarVertexGraph);
  ADD_POPART_BOOL_OPTION(separateCallOpPdfs);
  ADD_POPART_BOOL_OPTION(enableOutlining);
  ADD_POPART_BOOL_OPTION(enableOutliningCopyCostPruning);
  ADD_POPART_BOOL_OPTION(rearrangeAnchorsOnHost);
  ADD_POPART_BOOL_OPTION(enablePrefetchDatastreams);
  ADD_POPART_BOOL_OPTION(enableNonStableSoftmax);
  ADD_POPART_BOOL_OPTION(enableReplicatedGraphs);
  ADD_POPART_BOOL_OPTION(enableGradientAccumulation);
  ADD_POPART_BOOL_OPTION(instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION(enablePipelining);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_pipelining, enablePipelining);
  ADD_POPART_BOOL_OPTION(disableGradAccumulationTensorStreams);
  ADD_POPART_BOOL_OPTION(compileEngine);
  ADD_POPART_BOOL_OPTION(constantWeights);
  ADD_POPART_BOOL_OPTION(enableEngineCaching);
  ADD_POPART_BOOL_OPTION(enableMergeExchange);
  ADD_POPART_BOOL_OPTION(enableFloatingPointChecks);
  ADD_POPART_BOOL_OPTION(enableStochasticRounding);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_stochastic_rounding,
                               enableStochasticRounding);
  ADD_POPART_BOOL_OPTION(explicitRecomputation);
  ADD_POPART_BOOL_OPTION(enableExplicitMainLoops);
  ADD_POPART_BOOL_OPTION(useHostCopyOps);
  ADD_POPART_BOOL_OPTION(aliasZeroCopy);
  ADD_POPART_BOOL_OPTION(delayVarUpdates);
  ADD_POPART_BOOL_OPTION(enableFullyConnectedPass);
  ADD_POPART_BOOL_OPTION(enableSerializedMatmuls);
  ADD_POPART_BOOL_OPTION(enableStableNorm);
  ADD_POPART_BOOL_OPTION(decomposeGradSum);
  ADD_POPART_BOOL_OPTION(enableDistributedReplicatedGraphs);
  ADD_POPART_BOOL_OPTION(groupHostSync);
  ADD_POPART_BOOL_OPTION(automaticLossScalingSettings.enabled);
  ADD_POPART_BOOL_OPTION(instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION(enableSupportedDataTypeCasting);
  ADD_POPART_BOOL_OPTION(groupNormStridedChannelGrouping);
  ADD_POPART_BOOL_OPTION(scheduleNonWeightUpdateGradientConsumersEarly);

  ADD_POPART_DOUBLE_OPTION(outlineSequenceBreakCost);
  ADD_POPART_DOUBLE_OPTION(outlineThreshold);
  ADD_POPART_DOUBLE_OPTION(timeLimitScheduler);
  ADD_POPART_DOUBLE_OPTION(automaticLossScalingSettings.binEdgeLocation);
  ADD_POPART_DOUBLE_OPTION(
      automaticLossScalingSettings.thresholdUpperCountProportion);

#undef ADD_POPART_STRING_OPTION
#undef ADD_POPART_DOUBLE_OPTION
#undef ADD_POPART_UINT64_OPTION
#undef ADD_POPART_BOOL_OPTION
#undef ADD_POPART_ENUM_OPTION
#undef ADD_POPART_STRING_OPTION_ALIAS
#undef ADD_POPART_DOUBLE_OPTION_ALIAS
#undef ADD_POPART_UINT64_OPTION_ALIAS
#undef ADD_POPART_BOOL_OPTION_ALIAS
#undef ADD_POPART_ENUM_OPTION_ALIAS

  RegisterGetter(vector_options_getter, options_type, "custom_ops", "vector",
                 [&]() {
                   std::vector<std::string> res;
                   for (auto x : custom_ops) {
                     res.push_back(x.repr());
                   }
                   return res;
                 });

  RegisterSetter(bool_options, "enable_manual_shard", [&](bool value) {
    if (value) {
      popart_options.virtualGraphMode = popart::VirtualGraphMode::Manual;
    } else {
      popart_options.virtualGraphMode = popart::VirtualGraphMode::Off;
    }
  });

  RegisterGetter(options_getter, options_type, "enable_manual_shard", "bool",
                 [&]() {
                   return std::to_string(popart_options.virtualGraphMode ==
                                         popart::VirtualGraphMode::Manual);
                 });

  RegisterSetter(bool_options, "enable_half_partial", [&](bool value) {
    if (value) {
      popart_options.partialsTypeMatMuls = "half";
    } else {
      popart_options.partialsTypeMatMuls = "float";
    }
  });

  RegisterGetter(
      options_getter, options_type, "enable_half_partial", "bool", [&]() {
        return std::to_string(popart_options.partialsTypeMatMuls == "half");
      });

  RegisterSetter(
      container_options, "dotChecks",
      [&](const std::pair<std::string, std::string>& p) {
        std::uint64_t value = std::stoul(p.first);
        popart_options.dotChecks.insert(static_cast<popart::DotCheck>(value));
      });

  RegisterGetter(
      vector_options_getter, options_type, "dotChecks", "vector", [&]() {
        std::vector<std::string> res;
        for (auto x : popart_options.dotChecks) {
          res.push_back(std::to_string(static_cast<std::uint64_t>(x)));
        }
        return res;
      });

  RegisterSetter(container_options, "hardwareInstrumentations",
                 [&](const std::pair<std::string, std::string>& p) {
                   std::uint64_t value = std::stoul(p.first);
                   popart_options.hardwareInstrumentations.insert(
                       static_cast<popart::Instrumentation>(value));
                 });

  RegisterGetter(
      vector_options_getter, options_type, "hardwareInstrumentations", "vector",
      [&]() {
        std::vector<std::string> res;
        for (auto x : popart_options.hardwareInstrumentations) {
          res.push_back(std::to_string(static_cast<std::uint64_t>(x)));
        }
        return res;
      });

  RegisterSetter(container_options, "customCodelets",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.customCodelets.push_back(p.first);
                 });

  RegisterGetter(vector_options_getter, options_type, "customCodelets",
                 "vector", [&]() {
                   std::vector<std::string> res;
                   for (auto x : popart_options.customCodelets) {
                     res.push_back(x);
                   }
                   return res;
                 });

  RegisterSetter(container_options, "engineOptions",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.engineOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "engineOptions", "map",
                 [&]() { return popart_options.engineOptions; });

  RegisterSetter(container_options, "reportOptions",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.reportOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "reportOptions", "map",
                 [&]() { return popart_options.reportOptions; });

  RegisterSetter(container_options, "convolutionOptions",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.convolutionOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "convolutionOptions", "map",
                 [&]() { return popart_options.convolutionOptions; });

  RegisterSetter(container_options, "lstmOptions",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.lstmOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "lstmOptions", "map",
                 [&]() { return popart_options.lstmOptions; });

  RegisterSetter(container_options, "gclOptions",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.gclOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "gclOptions", "map",
                 [&]() { return popart_options.gclOptions; });
}

void IpuStrategy::AddBoolOption(const std::string& option, bool value) {
  set(option, value, bool_options, "bool");
}

void IpuStrategy::AddUint64Option(const std::string& option,
                                  std::uint64_t value) {
  set(option, value, uint64_options, "uint64");
}

void IpuStrategy::AddDoubleOption(const std::string& option, double value) {
  set(option, value, double_options, "double");
}

void IpuStrategy::AddStringOption(const std::string& option,
                                  const std::string& value) {
  set(option, value, string_options, "string");
}

void IpuStrategy::InsertStringOption(const std::string& option,
                                     const std::string& value) {
  set(option, std::pair<std::string, std::string>(value, ""), container_options,
      "vector");
}

void IpuStrategy::InsertStringPairOption(const std::string& option,
                                         const std::string& key,
                                         const std::string& value) {
  set(option, std::pair<std::string, std::string>(key, value),
      container_options, "map");
}

void IpuStrategy::SetTensorLocation(const std::string& tensor,
                                    const std::string& opt,
                                    std::uint64_t value) {
  VLOG(10) << "Setting " << opt << " to " << value << " for location "
           << tensor;
  popart::TensorLocationSettings* settings;
  if (tensor == "location_activation") {
    settings = &popart_options.activationTensorLocationSettings;
  } else if (tensor == "location_weight") {
    settings = &popart_options.weightTensorLocationSettings;
  } else if (tensor == "location_optimizer") {
    settings = &popart_options.optimizerStateTensorLocationSettings;
  } else if (tensor == "location_accumulator") {
    settings = &popart_options.accumulatorTensorLocationSettings;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unknown tensor location: %s", tensor));
  }

  if (opt == "minElementsForOffChip") {
    settings->minElementsForOffChip = value;
  } else if (opt == "minElementsForReplicatedTensorSharding") {
    settings->minElementsForReplicatedTensorSharding = value;
  } else if (opt == "onChip") {
    settings->location.storage = value > 0 ? popart::TensorStorage::OnChip
                                           : popart::TensorStorage::OffChip;
  } else if (opt == "useReplicatedTensorSharding") {
    settings->location.replicatedTensorSharding =
        value > 0 ? popart::ReplicatedTensorSharding::On
                  : popart::ReplicatedTensorSharding::Off;
  } else if (opt == "useIOTilesToLoad") {
    settings->location.loadTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else if (opt == "useIOTilesToStore") {
    settings->location.storageTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unknown option ' %s' for tensor location: %s", opt, tensor));
  }
}

void IpuStrategy::AddCustomOp(const std::string& paddle_op,
                              const std::string& popart_op,
                              const std::string& domain, int version) {
  LOG(INFO) << "IpuStrategy add custom op: " << paddle_op;
  custom_ops.push_back(
      IpuCustomOpIdentifier(paddle_op, popart_op, domain, version));
}

std::string IpuStrategy::GetOption(const std::string& option) {
  return get(option, options_getter);
}

std::vector<std::string> IpuStrategy::GetVectorOption(
    const std::string& option) {
  return get(option, vector_options_getter);
}

std::map<std::string, std::string> IpuStrategy::GetMapOption(
    const std::string& option) {
  return get(option, map_options_getter);
}

std::string IpuStrategy::GetOptionType(const std::string& option) {
  return options_type[option];
}

void IpuStrategy::EnablePattern(const std::string& t) {
  VLOG(10) << "enable popart pattern: " << t;
  popart_patterns.enablePattern(t, true);
}

void IpuStrategy::DisablePattern(const std::string& t) {
  VLOG(10) << "disable popart pattern: " << t;
  popart_patterns.enablePattern(t, false);
}

const bool IpuStrategy::IsPatternEnabled(const std::string& t) {
  return popart_patterns.isPatternEnabled(t);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
