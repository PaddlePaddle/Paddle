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
  ADD_BOOL_OPTION(need_avg_shard);
  ADD_BOOL_OPTION(enable_fp16);
  ADD_BOOL_OPTION(transfer_cast_op);
  ADD_BOOL_OPTION(use_no_bias_optimizer);
  ADD_BOOL_OPTION(enable_distribution);
  ADD_UINT64_OPTION(num_ipus);
  ADD_UINT64_OPTION(batches_per_step);
  ADD_UINT64_OPTION(micro_batch_size);
  ADD_UINT64_OPTION(random_seed);
  ADD_DOUBLE_OPTION(available_memory_proportion);
  ADD_DOUBLE_OPTION(loss_scaling);
  ADD_DOUBLE_OPTION(max_weight_norm);
  ADD_STRING_OPTION(accl1_type);
  ADD_STRING_OPTION(accl2_type);
  ADD_STRING_OPTION(accl3_type);
  ADD_STRING_OPTION(onnx_dump_path);
  ADD_STRING_OPTION(weight_decay_mode);

#undef ADD_STRING_OPTION
#undef ADD_DOUBLE_OPTION
#undef ADD_UINT64_OPTION
#undef ADD_BOOL_OPTION

#define ADD_RUNTIME_BOOL_OPTION(name, aliased_name)                          \
  RegisterSetter(bool_options, #name,                                        \
                 [&](bool value) { runtime_options.aliased_name = value; }); \
  RegisterGetter(options_getter, options_type, #name, "bool", [&]() {        \
    return std::to_string(runtime_options.aliased_name);                     \
  })

  ADD_RUNTIME_BOOL_OPTION(runtime_options.enable_eval, enable_eval);

#undef ADD_RUNTIME_BOOL_OPTION

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

  ADD_POPART_ENUM_OPTION_ALIAS(autodiff_settings.stitch_strategy,
                               autodiffSettings.stitchStrategy,
                               AutodiffStitchStrategy);
  ADD_POPART_ENUM_OPTION_ALIAS(batch_serialization_settings.transform_context,
                               batchSerializationSettings.transformContext,
                               BatchSerializationTransformContext);
  ADD_POPART_ENUM_OPTION_ALIAS(batch_serialization_settings.method,
                               batchSerializationSettings.method,
                               BatchSerializationMethod);
  ADD_POPART_ENUM_OPTION_ALIAS(batch_serialization_settings.batch_schedule,
                               batchSerializationSettings.batchSchedule,
                               BatchSerializationBatchSchedule);
  ADD_POPART_ENUM_OPTION_ALIAS(auto_recomputation, autoRecomputation,
                               RecomputationType);
  ADD_POPART_ENUM_OPTION_ALIAS(merge_var_update, mergeVarUpdate,
                               MergeVarUpdateType);
  ADD_POPART_ENUM_OPTION_ALIAS(virtual_graph_mode, virtualGraphMode,
                               VirtualGraphMode);
  ADD_POPART_ENUM_OPTION_ALIAS(synthetic_data_mode, syntheticDataMode,
                               SyntheticDataMode);
  ADD_POPART_ENUM_OPTION_ALIAS(subgraph_copying_strategy,
                               subgraphCopyingStrategy,
                               SubgraphCopyingStrategy);
  ADD_POPART_ENUM_OPTION_ALIAS(accumulation_and_replication_reduction_type,
                               accumulationAndReplicationReductionType,
                               ReductionType);
  ADD_POPART_ENUM_OPTION_ALIAS(
      mean_accumulation_and_replication_reduction_strategy,
      meanAccumulationAndReplicationReductionStrategy, MeanReductionStrategy);

  ADD_POPART_STRING_OPTION_ALIAS(log_dir, logDir);
  ADD_POPART_STRING_OPTION_ALIAS(cache_path, cachePath);
  ADD_POPART_STRING_OPTION_ALIAS(partials_type_matmuls, partialsTypeMatMuls);
  ADD_POPART_STRING_OPTION_ALIAS(custom_codelet_compile_flags,
                                 customCodeletCompileFlags);
  ADD_POPART_STRING_OPTION_ALIAS(serialized_poprithms_shift_graphs_dir,
                                 serializedPoprithmsShiftGraphsDir);
  ADD_POPART_STRING_OPTION_ALIAS(kahn_tie_breaker, kahnTieBreaker);

  ADD_POPART_UINT64_OPTION_ALIAS(execution_phase_settings.phases,
                                 executionPhaseSettings.phases);
  ADD_POPART_UINT64_OPTION_ALIAS(execution_phase_settings.stages,
                                 executionPhaseSettings.stages);
  ADD_POPART_UINT64_OPTION_ALIAS(batch_serialization_settings.factor,
                                 batchSerializationSettings.factor);
  ADD_POPART_UINT64_OPTION_ALIAS(first_dot_op, firstDotOp);
  ADD_POPART_UINT64_OPTION_ALIAS(final_dot_op, finalDotOp);
  ADD_POPART_UINT64_OPTION_ALIAS(num_io_tiles, numIOTiles);
  ADD_POPART_UINT64_OPTION_ALIAS(merge_var_update_mem_threshold,
                                 mergeVarUpdateMemThreshold);
  ADD_POPART_UINT64_OPTION_ALIAS(loose_threshold_at_peak, looseThresholdAtPeak);
  ADD_POPART_UINT64_OPTION_ALIAS(replicated_graph_count, replicatedGraphCount);
  ADD_POPART_UINT64_OPTION_ALIAS(accumulation_factor, accumulationFactor);
  ADD_POPART_UINT64_OPTION_ALIAS(swap_limit_scheduler, swapLimitScheduler);
  ADD_POPART_UINT64_OPTION_ALIAS(global_replication_factor,
                                 globalReplicationFactor);
  ADD_POPART_UINT64_OPTION_ALIAS(global_replica_offset, globalReplicaOffset);
  ADD_POPART_UINT64_OPTION_ALIAS(default_prefetch_buffering_depth,
                                 defaultPrefetchBufferingDepth);
  ADD_POPART_UINT64_OPTION_ALIAS(compilation_progress_total,
                                 compilationProgressTotal);
  ADD_POPART_UINT64_OPTION_ALIAS(transitive_closure_optimization_threshold,
                                 transitiveClosureOptimizationThreshold);

  ADD_POPART_BOOL_OPTION_ALIAS(
      batch_serialization_settings.concat_on_virtual_graph_change,
      batchSerializationSettings.concatOnVirtualGraphChange);
  ADD_POPART_BOOL_OPTION_ALIAS(
      batch_serialization_settings.concat_on_execution_phase_change,
      batchSerializationSettings.concatOnExecutionPhaseChange);
  ADD_POPART_BOOL_OPTION_ALIAS(
      batch_serialization_settings.concat_on_pipeline_stage_change,
      batchSerializationSettings.concatOnPipelineStageChange);
  ADD_POPART_BOOL_OPTION_ALIAS(strict_op_versions, strictOpVersions);
  ADD_POPART_BOOL_OPTION_ALIAS(opx_alias_checking, opxAliasChecking);
  ADD_POPART_BOOL_OPTION_ALIAS(opx_modify_checking, opxModifyChecking);
  ADD_POPART_BOOL_OPTION_ALIAS(dot_op_names, dotOpNames);
  ADD_POPART_BOOL_OPTION_ALIAS(export_poplar_computation_graph,
                               exportPoplarComputationGraph);
  ADD_POPART_BOOL_OPTION_ALIAS(export_poplar_vertex_graph,
                               exportPoplarVertexGraph);
  ADD_POPART_BOOL_OPTION_ALIAS(separate_call_op_pdfs, separateCallOpPdfs);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_outlining, enableOutlining);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_outlining_copy_cost_pruning,
                               enableOutliningCopyCostPruning);
  ADD_POPART_BOOL_OPTION_ALIAS(rearrange_anchors_on_host,
                               rearrangeAnchorsOnHost);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_prefetch_datastreams,
                               enablePrefetchDatastreams);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_non_stable_softmax,
                               enableNonStableSoftmax);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_replicated_graphs,
                               enableReplicatedGraphs);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_gradient_accumulation,
                               enableGradientAccumulation);
  ADD_POPART_BOOL_OPTION_ALIAS(instrument_with_hardware_cycle_counter,
                               instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_pipelining, enablePipelining);
  ADD_POPART_BOOL_OPTION_ALIAS(disable_grad_accumulation_tensor_streams,
                               disableGradAccumulationTensorStreams);
  ADD_POPART_BOOL_OPTION_ALIAS(compile_engine, compileEngine);
  ADD_POPART_BOOL_OPTION_ALIAS(constant_weights, constantWeights);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_engine_caching, enableEngineCaching);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_merge_exchange, enableMergeExchange);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_floating_point_checks,
                               enableFloatingPointChecks);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_stochastic_rounding,
                               enableStochasticRounding);
  ADD_POPART_BOOL_OPTION_ALIAS(explicit_recomputation, explicitRecomputation);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_explicit_main_loops,
                               enableExplicitMainLoops);
  ADD_POPART_BOOL_OPTION_ALIAS(use_host_copy_ops, useHostCopyOps);
  ADD_POPART_BOOL_OPTION_ALIAS(alias_zero_copy, aliasZeroCopy);
  ADD_POPART_BOOL_OPTION_ALIAS(delay_var_updates, delayVarUpdates);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_fully_connected_pass,
                               enableFullyConnectedPass);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_serialized_matmuls,
                               enableSerializedMatmuls);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_stable_norm, enableStableNorm);
  ADD_POPART_BOOL_OPTION_ALIAS(decompose_grad_sum, decomposeGradSum);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_distributed_replicated_graphs,
                               enableDistributedReplicatedGraphs);
  ADD_POPART_BOOL_OPTION_ALIAS(group_host_sync, groupHostSync);
  ADD_POPART_BOOL_OPTION_ALIAS(automatic_loss_scaling_settings.enabled,
                               automaticLossScalingSettings.enabled);
  ADD_POPART_BOOL_OPTION_ALIAS(instrument_with_hardware_cycle_counter,
                               instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION_ALIAS(enable_supported_data_type_casting,
                               enableSupportedDataTypeCasting);
  ADD_POPART_BOOL_OPTION_ALIAS(group_norm_strided_channel_grouping,
                               groupNormStridedChannelGrouping);
  ADD_POPART_BOOL_OPTION_ALIAS(
      schedule_non_weight_update_gradient_consumers_early,
      scheduleNonWeightUpdateGradientConsumersEarly);

  ADD_POPART_DOUBLE_OPTION_ALIAS(outline_sequence_break_cost,
                                 outlineSequenceBreakCost);
  ADD_POPART_DOUBLE_OPTION_ALIAS(outline_threshold, outlineThreshold);
  ADD_POPART_DOUBLE_OPTION_ALIAS(time_limit_scheduler, timeLimitScheduler);
  ADD_POPART_DOUBLE_OPTION_ALIAS(
      automatic_loss_scaling_settings.bin_edge_location,
      automaticLossScalingSettings.binEdgeLocation);
  ADD_POPART_DOUBLE_OPTION_ALIAS(
      automatic_loss_scaling_settings.threshold_upper_count_proportion,
      automaticLossScalingSettings.thresholdUpperCountProportion);

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
      container_options, "dot_checks",
      [&](const std::pair<std::string, std::string>& p) {
        std::uint64_t value = std::stoul(p.first);
        popart_options.dotChecks.insert(static_cast<popart::DotCheck>(value));
      });

  RegisterGetter(
      vector_options_getter, options_type, "dot_checks", "vector", [&]() {
        std::vector<std::string> res;
        for (auto x : popart_options.dotChecks) {
          res.push_back(std::to_string(static_cast<std::uint64_t>(x)));
        }
        return res;
      });

  RegisterSetter(container_options, "hardware_instrumentations",
                 [&](const std::pair<std::string, std::string>& p) {
                   std::uint64_t value = std::stoul(p.first);
                   popart_options.hardwareInstrumentations.insert(
                       static_cast<popart::Instrumentation>(value));
                 });

  RegisterGetter(
      vector_options_getter, options_type, "hardware_instrumentations",
      "vector", [&]() {
        std::vector<std::string> res;
        for (auto x : popart_options.hardwareInstrumentations) {
          res.push_back(std::to_string(static_cast<std::uint64_t>(x)));
        }
        return res;
      });

  RegisterSetter(container_options, "custom_codelets",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.customCodelets.push_back(p.first);
                 });

  RegisterGetter(vector_options_getter, options_type, "custom_codelets",
                 "vector", [&]() {
                   std::vector<std::string> res;
                   for (auto x : popart_options.customCodelets) {
                     res.push_back(x);
                   }
                   return res;
                 });

  RegisterSetter(container_options, "engine_options",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.engineOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "engine_options", "map",
                 [&]() { return popart_options.engineOptions; });

  RegisterSetter(container_options, "report_options",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.reportOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "report_options", "map",
                 [&]() { return popart_options.reportOptions; });

  RegisterSetter(container_options, "convolution_options",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.convolutionOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "convolution_options", "map",
                 [&]() { return popart_options.convolutionOptions; });

  RegisterSetter(container_options, "lstm_options",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.lstmOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "lstm_options", "map",
                 [&]() { return popart_options.lstmOptions; });

  RegisterSetter(container_options, "gcl_options",
                 [&](const std::pair<std::string, std::string>& p) {
                   popart_options.gclOptions.emplace(p);
                 });

  RegisterGetter(map_options_getter, options_type, "gcl_options", "map",
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

  if (opt == "min_elements_for_off_chip") {
    settings->minElementsForOffChip = value;
  } else if (opt == "min_elements_for_replicated_tensor_sharding") {
    settings->minElementsForReplicatedTensorSharding = value;
  } else if (opt == "on_chip") {
    settings->location.storage = value > 0 ? popart::TensorStorage::OnChip
                                           : popart::TensorStorage::OffChip;
  } else if (opt == "use_replicated_tensor_sharding") {
    settings->location.replicatedTensorSharding =
        value > 0 ? popart::ReplicatedTensorSharding::On
                  : popart::ReplicatedTensorSharding::Off;
  } else if (opt == "use_io_tiles_to_load") {
    settings->location.loadTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else if (opt == "use_io_tiles_to_store") {
    settings->location.storageTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else if (opt == "sharding_domain_with_all") {
    settings->location.shardingDomain =
        popart::CommGroup(popart::CommGroupType::All, value);
  } else if (opt == "sharding_domain_with_consecutive") {
    settings->location.shardingDomain =
        popart::CommGroup(popart::CommGroupType::Consecutive, value);
  } else if (opt == "sharding_domain_with_orthogonal") {
    settings->location.shardingDomain =
        popart::CommGroup(popart::CommGroupType::Orthogonal, value);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unknown option ' %s' for tensor location: %s", opt, tensor));
  }
}

void IpuStrategy::SetAccumulateOuterFragmentSettings(
    const std::uint64_t& schedule, const std::vector<int>& values) {
  VLOG(10) << "SetAccumulateOuterFragmentSettings schedule:" << schedule;
  auto schedule_ =
      static_cast<popart::AccumulateOuterFragmentSchedule>(schedule);
  popart_options.accumulateOuterFragmentSettings =
      popart::AccumulateOuterFragmentSettings(schedule_, values);
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

std::vector<std::string> IpuStrategy::GetAllOptionNames() {
  std::vector<std::string> names;
  for (auto& option : options_getter) {
    names.push_back(option.first);
  }
  for (auto& option : vector_options_getter) {
    names.push_back(option.first);
  }
  for (auto& option : map_options_getter) {
    names.push_back(option.first);
  }
  return names;
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
