// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/ir/group_schedule/config/filedatabase.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <fstream>

#include "paddle/cinn/utils/multi_threading.h"

PD_DECLARE_string(cinn_tile_config_filename_label);

namespace cinn {
namespace ir {

bool TileConfigToProto(group_schedule::config::proto::TileData* tile_data,
                       const TileConfigMap& tile_config_map,
                       const int& priority) {
  for (auto& it : tile_config_map) {
    // prepare key---convert bucket info to proto::bucket_info
    BucketInfo bucket_info = it.first;
    int dims = bucket_info.space.size();
    for (int i = 0; i < dims; i++) {
      group_schedule::config::proto::Dimension cur_dimension;
      cur_dimension.set_lower_bound(bucket_info.space[i].lower_bound);
      cur_dimension.set_upper_bound(bucket_info.space[i].upper_bound);
      cur_dimension.set_iter_type(bucket_info.space[i].iter_type);
      cur_dimension.set_is_dynamic(bucket_info.space[i].is_dynamic);
      *(tile_data->mutable_bucket_info()->add_dimension()) = cur_dimension;
    }

    // prepare value---transfer tile_config to proto::tile_config
    group_schedule::config::proto::TileConfig tc;
    tc.set_warp_num(it.second.warp_num);
    tc.set_tree_reduce_num(it.second.tree_reduce_num);
    tc.set_spatial_inner_num(it.second.spatial_inner_num);
    *(tile_data->mutable_tile_config()) = tc;
    tile_data->set_priority(priority);
  }
  return true;
}

void AppendLineToFile(const std::string& file_path,
                      const std::vector<std::string>& lines) {
  std::ofstream os(file_path, std::ofstream::app);
  PADDLE_ENFORCE_EQ(os.good(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Cannot open the file to write:  %s", file_path));
  VLOG(3) << "Start append tile_config to json";
  for (auto& line : lines) {
    os << line << std::endl;
    VLOG(3) << "Add config line: " << line;
  }
}

std::string IterSpaceTypeToDir(const common::Target target,
                               const IterSpaceType& iter_space_type) {
  std::string dirname = "";
  std::string filename = "";
  for (auto i : iter_space_type) {
    dirname += i.first;
    dirname += "_";
    filename += i.first + i.second;
    filename += "_";
  }
  dirname = dirname.substr(0, dirname.size() - 1);
  filename = filename.substr(0, filename.size() - 1);

  auto checkexist = [](std::string test_path) {
    if (!PathExists(test_path)) {
      PADDLE_ENFORCE_NE(MKDIR(test_path.c_str()),
                        -1,
                        ::common::errors::PreconditionNotMet(
                            "Can not create directory: %s, Make sure you "
                            "have permission to write",
                            test_path));
    }
  };
  std::string root_path = FLAGS_cinn_tile_config_filename_label;
  checkexist(root_path);
  checkexist(root_path + target.arch_str());
  checkexist(root_path + target.arch_str() + "/" + dirname);
  VLOG(3) << "Dump_path is " << root_path + dirname + "/" + filename + ".json";
  return root_path + target.arch_str() + "/" + dirname + "/" + filename +
         ".json";
}

bool FileTileConfigDatabase::Tofile(const common::Target& target,
                                    int priority) {
  // Step1. To proto
  TileConfigMap& tile_config_map = target_config_data_;
  group_schedule::config::proto::TileData tile_data;
  auto is_success = TileConfigToProto(&tile_data, tile_config_map, priority);
  if (is_success == false) {
    PADDLE_THROW(::common::errors::Unavailable(
        "Can't convert tile_config_map to its proto message."));
  }
  // Step2. ToJson
  IterSpaceType iter_space_type = [&] {
    std::vector<std::pair<std::string, std::string>> res;
    for (const auto& dim : bucket_info_.space) {
      res.emplace_back(dim.iter_type, (dim.is_dynamic ? "dynamic" : "static"));
    }
    return res;
  }();
  std::string dump_path = IterSpaceTypeToDir(target, iter_space_type);
  size_t length = tile_config_map.size();
  std::vector<std::string> json_lines(length);
  for (int i = 0; i < length; i++) {
    std::string json_string;
    auto status =
        google::protobuf::util::MessageToJsonString(tile_data, &json_string);
    PADDLE_ENFORCE_EQ(
        status.ok(),
        true,
        ::common::errors::InvalidArgument(
            "Failed to serialize fileconfig database to JSON, task key = %s-%s",
            target,
            dump_path));
    json_lines[i] = json_string;
  }
  // Step3. CommitJson
  AppendLineToFile(dump_path, json_lines);
  return true;
}

std::vector<std::string> ReadLinesFromFile(const std::string& file_path) {
  std::ifstream is(file_path);
  if (is.good()) {
    std::vector<std::string> json_strs;
    for (std::string str; std::getline(is, str);) {
      if (str != "") {
        json_strs.push_back(str);
      }
    }
    VLOG(3) << "The size of json_lines is: " << json_strs.size();
    return json_strs;
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("File doesn't exist: %s", file_path));
  }
}

void JsonStringToMessageOfTileConfig(
    std::vector<group_schedule::config::proto::TileData>* tile_database,
    const std::vector<std::string>& json_lines) {
  // convert JSON string to proto object
  auto worker_fn = [&json_lines, &tile_database](int index) {
    group_schedule::config::proto::TileData tile_data;
    auto status = google::protobuf::util::JsonStringToMessage(json_lines[index],
                                                              &tile_data);
    CHECK(status.ok()) << "Failed to parse JSON: " << json_lines[index];
    (*tile_database)[index] = tile_data;
  };
  utils::parallel_run(
      worker_fn, utils::SequenceDispatcher(0, json_lines.size()), -1);
  return;
}

bool comparepriority(group_schedule::config::proto::TileData tile_data1,
                     group_schedule::config::proto::TileData tile_data2) {
  return tile_data1.priority() >= tile_data2.priority();
}

TileConfigMap FileTileConfigDatabase::GetConfigs(
    const common::Target& target, const IterSpaceType& iter_space_type) const {
  // Step 1: Read from json file and convert json to proto message
  std::string file_path = IterSpaceTypeToDir(target, iter_space_type);
  auto json_lines = ReadLinesFromFile(file_path);
  size_t line_length = json_lines.size();

  std::vector<group_schedule::config::proto::TileData> tile_database(
      line_length);
  JsonStringToMessageOfTileConfig(&tile_database, json_lines);

  // Step 2: Parse from proto message
  TileConfigMap tile_config_map;
  // order tile_database according to priority
  std::sort(tile_database.begin(), tile_database.end(), comparepriority);
  for (const auto& piece_tileconfig : tile_database) {
    group_schedule::config::proto::BucketInfo its =
        piece_tileconfig.bucket_info();
    //  Step 2.1: Convert proto bucketinfo to source bucketinfo
    int dims = its.dimension_size();
    BucketInfo bucket_info(static_cast<size_t>(dims));
    for (int i = 0; i < dims; i++) {
      bucket_info.space.[i].lower_bound = its.dimension(i).lower_bound();
      bucket_info.space.[i].upper_bound = its.dimension(i).upper_bound();
      bucket_info.space.[i].iter_type = its.dimension(i).iter_type();
      bucket_info.space.[i].is_dynamic = its.dimension(i).is_dynamic();
    }
    //  Step 2.2: Convert proto tile_config to source tile_config
    ScheduleConfig::TileConfig tconfig;
    tconfig.tree_reduce_num = piece_tileconfig.tile_config().tree_reduce_num();
    tconfig.spatial_inner_num =
        piece_tileconfig.tile_config().spatial_inner_num();
    tconfig.warp_num = piece_tileconfig.tile_config().warp_num();
    tile_config_map[bucket_info] = tconfig;
    // TODO(XiaZichao): Add function to cut one lattice into smaller ones
  }
  // TODO(XiaZichao): update json file using top view of tileconfigMap
  return tile_config_map;
}

void FileTileConfigDatabase::AddConfig(const common::Target& target,
                                       const BucketInfo& bucket_info,
                                       const ScheduleConfig::TileConfig& config,
                                       int priority) {
  target_config_data_[bucket_info] = config;
  auto status = FileTileConfigDatabase::Tofile(target, priority);
  if (status == true) {
    target_config_data_.clear();
    return;
  } else {
    PADDLE_THROW(
        ::common::errors::Unavailable("Can't add tile config to json file."));
  }
}
}  // namespace ir
}  // namespace cinn
