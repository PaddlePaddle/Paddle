#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");

using namespace paddle;

float random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void split(const std::string &str, char sep, std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

void split_to_float(const std::string &str, char sep, std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
}

template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  for (const auto &c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec) {
  std::stringstream ss;
  for (const auto &piece : vec) {
    ss << to_string(piece) << "\n";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec) {
  std::stringstream ss;
  for (const auto &line : vec) {
    for (const auto &rcd : line) {
      ss << to_string(rcd) << ";\t";
    }
    ss << '\n';
  }
  return ss.str();
}

// clang-format off
void TensorAssignData(PaddleTensor *tensor, const std::vector<std::vector<float>> &data) {
  // Assign buffer
  int dim = std::accumulate(tensor->shape.begin(), tensor->shape.end(), 1, [](int a, int b) { return a * b; });
  tensor->data.Resize(sizeof(float) * dim);
  int c = 0;
  for (const auto& f : data) {
    for (float v : f) { static_cast<float*>(tensor->data.data())[c++] = v; }
  }
}
// clang-format on

std::string DescribeTensor(const PaddleTensor &tensor) {
  std::stringstream os;
  os << "Tensor [" << tensor.name << "]\n";
  os << " - type: ";
  switch (tensor.dtype) {
    case PaddleDType::FLOAT32:
      os << "float32";
      break;
    case PaddleDType::INT64:
      os << "int64";
      break;
    default:
      os << "unset";
  }
  os << '\n';

  os << " - shape: " << to_string(tensor.shape) << '\n';
  os << " - lod: ";
  for (auto& l : tensor.lod) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  os << " - data: ";

  // clang-format off
  int dim = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, [](int a, int b) { return a * b; });  // clang-format on
  for (size_t i = 0; i < dim; i++) {
    os << static_cast<float *>(tensor.data.data())[i] << " ";
  }
  os << '\n';
  return os.str();
}

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<std::vector<float>> week_data_all, minute_data_all;
  std::vector<size_t> lod1, lod2, lod3;
  std::vector<std::vector<float>> rnn_link_data, rnn_week_datas,
      rnn_minute_datas;

  size_t batch_iter{0};
  size_t batch_size{1};

  DataRecord() = default;
  DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }

  DataRecord NextBatch() {
    DataRecord data;
    LOG(INFO) << "link all " << link_step_data_all.size();
    size_t batch_end = batch_iter + batch_size;

    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= link_step_data_all.size()) {
      data.link_step_data_all.assign(link_step_data_all.begin() + batch_iter,
                                     link_step_data_all.begin() + batch_end);
      data.week_data_all.assign(week_data_all.begin() + batch_iter,
                                week_data_all.begin() + batch_end);
      data.minute_data_all.assign(minute_data_all.begin() + batch_iter,
                                  minute_data_all.begin() + batch_end);

      // Prepare LoDs
      data.lod1.emplace_back(0);
      data.lod2.emplace_back(0);
      data.lod3.emplace_back(0);

      LOG(INFO) << "size: " << data.link_step_data_all.size();
      LOG(INFO) << "size: " << data.week_data_all.size();
      LOG(INFO) << "size: " << data.minute_data_all.size();

      CHECK(!data.link_step_data_all.empty()) << "empty";
      CHECK(!data.week_data_all.empty());
      CHECK(!data.minute_data_all.empty());
      CHECK_EQ(data.link_step_data_all.size(), data.week_data_all.size());
      CHECK_EQ(data.minute_data_all.size(), data.link_step_data_all.size());

      for (size_t j = 0; j < data.link_step_data_all.size(); j++) {
        for (const auto &d : data.link_step_data_all[j]) {
          data.rnn_link_data.push_back(d);
        }
        LOG(INFO) << "push back";
        data.rnn_week_datas.push_back(data.week_data_all[j]);
        data.rnn_minute_datas.push_back(data.minute_data_all[j]);
        // calculate lod
        data.lod1.push_back(data.lod1.back() + data.link_step_data_all[j].size());
        data.lod3.push_back(data.lod3.back() + 1);
        LOG(INFO) << "push back";
        for (size_t i = 1; i < data.link_step_data_all[j].size() + 1; i++) {
          data.lod2.push_back(data.lod2.back() + data.link_step_data_all[j].size());
        }
      }
    }
    LOG(INFO) << "finish one batch";

    batch_iter += batch_size;
    return data;
  }

  void Load(const std::string &path) {
    std::ifstream file(FLAGS_datapath);
    std::string line;

    int num_lines = 0;
    while (std::getline(file, line)) {
      if (++num_lines > 1) break;
      std::vector<std::string> data;
      split(line, ':', &data);

      std::vector<std::vector<float>> link_step_data;
      std::vector<std::string> link_datas;
      split(data[0], '|', &link_datas);

      for (auto &step_data : link_datas) {
        std::vector<float> tmp;
        split_to_float(step_data, ',', &tmp);
        link_step_data.emplace_back(tmp);
      }

      // load week data
      std::vector<float> week_data;
      split_to_float(data[2], ',', &week_data);

      // load minute data
      std::vector<float> minute_data;
      split_to_float(data[1], ',', &minute_data);

      link_step_data_all.emplace_back(std::move(link_step_data));
      week_data_all.emplace_back(std::move(week_data));
      minute_data_all.emplace_back(std::move(minute_data));
    }
  }
};

void Main(int batch_size) {
  if (FLAGS_modeldir.empty() || FLAGS_datapath.empty()) {
    LOG(ERROR) << "./cmd --modeldir=path/to/model";
    exit(1);
  }

  DataRecord data(FLAGS_datapath, batch_size);
  LOG(INFO) << "link_step";
  LOG(INFO) << to_string(data.link_step_data_all);
  LOG(INFO) << "week_data";
  LOG(INFO) << to_string(data.week_data_all);
  LOG(INFO) << "minute_data";
  LOG(INFO) << to_string(data.minute_data_all);

  std::string modeldir = FLAGS_modeldir;
  NativeConfig config;
  config.prog_file = modeldir + "/__model__";
  config.param_file = modeldir + "/param";
  config.use_gpu = false;
  config.device = 0;

  PaddleTensor lod_attention_tensor, init_zero_tensor, lod_tensor_tensor,
      week_tensor, minute_tensor;
  lod_attention_tensor.name = "lod_attention";
  init_zero_tensor.name = "init_zero";
  lod_tensor_tensor.name = "lod_tensor";
  week_tensor.name = "week";
  minute_tensor.name = "minute";

  LOG(INFO) << "get one batch";
  auto one_batch = data.NextBatch();

  LOG(INFO) << "set shape";
  // clang-format off
  LOG(INFO) << one_batch.rnn_minute_datas.size();
  LOG(INFO) << one_batch.rnn_minute_datas.front().size();
  std::vector<int> rnn_link_data_shape({static_cast<int>(one_batch.rnn_link_data.size()), static_cast<int>(one_batch.rnn_link_data.front().size())});
  LOG(INFO) << "set 1";
  lod_attention_tensor.shape.assign({1,2}); lod_attention_tensor.lod.assign({one_batch.lod1, one_batch.lod2});
  LOG(INFO) << "set 1";
  init_zero_tensor.shape.assign({batch_size, 15}); init_zero_tensor.lod.assign({one_batch.lod3});
  LOG(INFO) << "set 1";
  lod_tensor_tensor.shape = rnn_link_data_shape; lod_tensor_tensor.lod.assign({one_batch.lod1});
  LOG(INFO) << "set 1";
  week_tensor.shape.assign({(int)one_batch.rnn_week_datas.size(), (int)one_batch.rnn_week_datas.front().size()}); week_tensor.lod.assign({one_batch.lod3});
  LOG(INFO) << "set 1";
  minute_tensor.shape.assign({(int)one_batch.rnn_minute_datas.size(), (int)one_batch.rnn_minute_datas.front().size()}); minute_tensor.lod.assign({one_batch.lod3});

  LOG(INFO) << "assign data";
  // assign data
  LOG(INFO) << "assign 0";
  TensorAssignData(&lod_attention_tensor, std::vector<std::vector<float>>({{0, 0}}));
  LOG(INFO) << "assign 0";
  std::vector<float> tmp_zeros(batch_size*15, 0.); TensorAssignData(&init_zero_tensor, {tmp_zeros});
  LOG(INFO) << "assign 0";
  TensorAssignData(&lod_tensor_tensor, one_batch.rnn_link_data);
  LOG(INFO) << "assign 0";
  TensorAssignData(&week_tensor, one_batch.rnn_week_datas);
  LOG(INFO) << "assign 0";
  TensorAssignData(&minute_tensor, one_batch.rnn_minute_datas);
  // clang-format on

  LOG(INFO) << "to set input_slots";
  std::vector<PaddleTensor> input_slots({
      lod_tensor_tensor,
      lod_attention_tensor,
      init_zero_tensor,
      init_zero_tensor,
      week_tensor,
      minute_tensor
  });

  LOG(INFO) << "set type";
  for (auto& tensor : input_slots) {
    tensor.dtype = PaddleDType ::FLOAT32;
    LOG(INFO) << DescribeTensor(tensor);
  }

  LOG(INFO) << "create predictor";
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  // { batch begin

  std::vector<PaddleTensor> outputs;

  LOG(INFO) << "to execute";
  predictor->Run(input_slots, &outputs);
  LOG(INFO) << "output.size() " << outputs.size();

  LOG(INFO) << "output";
  LOG(INFO) << DescribeTensor(outputs.front());

  // } batch end
}

TEST(ditu, main) { Main(1); }
