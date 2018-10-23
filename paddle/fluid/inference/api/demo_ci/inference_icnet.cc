// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "inference_icnet.h"

// 数据格式
// "<space splitted floats as data>\t<space splitted ints as shape"
// 1. 存储为float32格式。
// 2. 必须减去均值。 CHW三个通道为 mean = 112.15, 109.41, 185.42
using namespace paddle;

class Predictor {
private:
	std::unique_ptr<PaddlePredictor> predictor;
	struct Record
	{
		std::vector<float> data;
		std::vector<int32_t> shape;
	};

	const int C = 3; // image channel
	const int H = 449; // image height
	const int W = 581; // image width

	using Time = decltype(std::chrono::high_resolution_clock::now());

	Time time() { return std::chrono::high_resolution_clock::now(); };

	double time_diff(Time t1, Time t2) {
		typedef std::chrono::microseconds ms;
		auto diff = t2 - t1;
		ms counter = std::chrono::duration_cast<ms>(diff);
		return counter.count() / 1000.0;
	}

	static void split(const std::string& str, char sep,
		std::vector<std::string>* pieces) {
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

	Record ProcessALine(const std::string& line) {
		std::vector<std::string> columns;
		split(line, '\t', &columns);

		Record record;
		std::vector<std::string> data_strs;
		split(columns[0], ' ', &data_strs);
		for (auto& d : data_strs) {
			record.data.push_back(std::stof(d));
		}

		std::vector<std::string> shape_strs;
		split(columns[1], ' ', &shape_strs);
		for (auto& s : shape_strs) {
			record.shape.push_back(std::stoi(s));
		}
		return record;
	}

public:
	Predictor (const char* prog_file,
		const char* param_file, const float fraction_of_gpu_memory,
		const bool use_gpu, const int device) {

		NativeConfig config;
		config.prog_file = prog_file;
		config.param_file = param_file;
		config.fraction_of_gpu_memory = fraction_of_gpu_memory;
		config.use_gpu = use_gpu;
		config.device = device;

		predictor = CreatePaddlePredictor<NativeConfig>(config);
	}

	void predict(float* input, const int channel, const int height, const int width, 
		int64_t** output, int* output_length, int batch_size) {
		std::vector<float> data;
		int intput_length = channel * height * width * batch_size;
		for (int i = 0; i < intput_length; i++) {
			data.push_back(*((float*)input + i));
		}

		// initialize the input data 
		PaddleTensor tensor;
		tensor.shape = std::vector<int>({ batch_size, channel, height, width });
		tensor.data.Resize(sizeof(float) * batch_size * channel * height * width);
		std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));

		tensor.dtype = PaddleDType::FLOAT32;
		std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

		// initialize the output data
		PaddleTensor tensor_out;
		std::vector<PaddleTensor> outputs(1, tensor_out);
		predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
		*output_length = (int)outputs[0].data.length();
		std::memcpy(static_cast<void *>(*output), outputs[0].data.data(), outputs[0].data.length());
		int64_t sum_out = 0;
		for(int i=0; i < outputs[0].data.length()/sizeof(int64_t); ++i) {
			int64_t item = static_cast<int64_t*>(outputs[0].data.data())[i];
			sum_out += item;
			if (item != 0) {
				std::cout << item << std::endl;
			}
		}

		std::cout << "sum_out" << sum_out << std::endl;
	}
};

API_REFERENCE void * init_predictor(const char* prog_file,
	const char* param_file, const float fraction_of_gpu_memory,
	const bool use_gpu, const int device) {
	return new Predictor(prog_file, param_file, fraction_of_gpu_memory, use_gpu, device);
}

API_REFERENCE void predict(void* handle, float* input, const int channel, const int height, const int width, 
	int64_t** output, int* output_length, int batch_size) {
	assert(handle != nullptr);
	((Predictor*)handle)->predict(input, channel, height, width, output, output_length, batch_size);
}

API_REFERENCE void destory_predictor(void *handle) {
	if (handle) {
		delete handle;
		handle = nullptr;
	}
}
