/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains a simple demo for how to take a model for inference.
 */
#include <cassert>
#include <cctype>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>  //NOLINT
#include "paddle/fluid/inference/paddle_inference_api.h"

std::string MODELDIR = ""; /* "Directory of the inference model." */ // NOLINT
std::string REFER = "";
/*"path to reference result for comparison."*/ //NOTLINT
/*path of data; each line is a record, format:
<space splitted floats as data>\t<space splitted ints as shape>

Please check the demo data of data.txt for details.
 */
std::string DATA = "";  
bool USE_GPU = true;     /*"Whether use gpu."*/

auto message_err = []()
{
  std::cout << "Copyright (c) 2018 PaddlePaddle Authors." << std::endl;
  std::cout << "Demo Case for windows inference. "
            << "\n"
            << "Usage: Input your model path and use_gpu as the guide requires,"
            << "then run the demo inference, and will get a result."
            << std::endl;
  std::cout << std::endl;
};

namespace paddle
{
	namespace demo
	{
		void split(const std::string& str, char sep,
			std::vector<std::string>* pieces)
		{
			pieces->clear();
			if (str.empty())
			{
				return;
			}
			size_t pos = 0;
			size_t next = str.find(sep, pos);
			while (next != std::string::npos)
			{
				pieces->push_back(str.substr(pos, next - pos));
				pos = next + 1;
				next = str.find(sep, pos);
			}
			if (!str.substr(pos).empty())
			{
				pieces->push_back(str.substr(pos));
			}
		}

		/*
		 * Get a summary of a PaddleTensor content.
		 */
		std::string SummaryTensor(const PaddleTensor& tensor)
		{
			std::stringstream ss;
			int num_elems = tensor.data.length() / PaddleDtypeSize(tensor.dtype);

			ss << "data[:10]\t";
			switch (tensor.dtype)
			{
			case PaddleDType::INT64:
				for (int i = 0; i < std::min(num_elems, 10); i++)
				{
					ss << static_cast<int64_t*>(tensor.data.data())[i] << " ";
				}
				break;
			case PaddleDType::FLOAT32:
				for (int i = 0; i < std::min(num_elems, 10); i++)
				{
					ss << static_cast<float*>(tensor.data.data())[i] << " ";
				}
				break;
			}
			return ss.str();
		}

		std::string ToString(const NativeConfig& config)
		{
			std::stringstream ss;
			ss << "Use GPU : " << (config.use_gpu ? "True" : "False") << "\n"
				<< "Device : " << config.device << "\n"
				<< "fraction_of_gpu_memory : " << config.fraction_of_gpu_memory << "\n"
				<< "specify_input_name : "
				<< (config.specify_input_name ? "True" : "False") << "\n"
				<< "Program File : " << config.prog_file << "\n"
				<< "Param File : " << config.param_file;
			return ss.str();
		}

		struct Record
		{
			std::vector<float> data;
			std::vector<int32_t> shape;
		};

		Record ProcessALine(const std::string& line)
		{
			std::cout << "process a line" << std::endl;
			std::vector<std::string> columns;
			split(line, '\t', &columns);
			assert(columns.size() == 2UL, "data format error, should be <data>\t<shape>");

			Record record;
			std::vector<std::string> data_strs;
			split(columns[0], ' ', &data_strs);
			//将数据字符串转换为整型数据并放到record.data中
			for (auto& d : data_strs)
			{
				record.data.push_back(std::stof(d));
			} 

			std::vector<std::string> shape_strs;
			split(columns[1], ' ', &shape_strs);
			for (auto& s : shape_strs)
			{
				record.shape.push_back(std::stoi(s));
			}
			std::cout << "data size " << record.data.size() << std::endl;
			std::cout << "data shape size " << record.shape.size() << std::endl;
			return record;
		}

		void CheckOutput(const std::string& referfile, const PaddleTensor& output)
		{
			std::string line;
			std::ifstream file(referfile);
			std::getline(file, line);
			auto refer = ProcessALine(line);
			file.close();

			size_t numel = output.data.length() / PaddleDtypeSize(output.dtype);
			std::cout << "predictor output numel " << numel << std::endl;
			std::cout << "reference output numel " << refer.data.size() << std::endl;
			assert(numel == refer.data.size());
			switch (output.dtype)
			{
			case PaddleDType::INT64:
				for (size_t i = 0; i < numel; ++i)
				{
					assert(static_cast<int64_t*>(output.data.data())[i] == refer.data[i]);
				}
				break;
			case PaddleDType::FLOAT32:
				for (size_t i = 0; i < numel; ++i)
				{
					assert(fabs(static_cast<float*>(output.data.data())[i] - refer.data[i]) <= 1e-5);
				}
				break;
			}
		}

		/*
		 * Use the native fluid engine to inference the demo.
		 */
		void Main(bool use_gpu)
		{
			NativeConfig config;
			config.model_dir = MODELDIR;
			//config.param_file = MODELDIR + "/__params__";
			//config.prog_file = MODELDIR + "/__model__";
			config.use_gpu = USE_GPU;
			config.device = 0;
			if (USE_GPU)
			{
				config.fraction_of_gpu_memory = 0.1f;  // set by yourself
			}
			std::cout << ToString(config) << std::endl;
			std::cout << "init predictor" << std::endl;
			auto predictor = CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

			std::cout << "begin to process data" << std::endl;
			// Just a single batch of data.
			std::string line;
			std::cout << "data : " << std::endl;
			std::ifstream file(DATA);
			if (!file.is_open()) 
			{
				std::cout << "failed open data" << DATA << std::endl;
				exit(0);
			}
			std::getline(file, line);
			auto record = ProcessALine(line);
			file.close();

			// Inference.
			PaddleTensor input;
			input.shape = record.shape;
			input.data =
				PaddleBuf(record.data.data(), record.data.size() * sizeof(float));
			input.dtype = PaddleDType::FLOAT32;

			std::cout << "run executor" << std::endl;
			std::vector<PaddleTensor> output;
			predictor->Run({ input }, &output);

			std::cout << "output.size " << output.size() << std::endl;
			auto& tensor = output.front();
			std::cout << "output: " << SummaryTensor(tensor) << std::endl;

			// compare with reference result
			std::cout << "refer result : " << REFER << std::endl;
			CheckOutput(REFER, tensor);
		}
	}
}

int main(int argc, char** argv)
{
	MODELDIR = "./LB_icnet_model";
	//DATA = "./icnet_image.txt";
	DATA = "./1.png.txt";
	REFER = "./icnet_label.txt";
	paddle::demo::Main(USE_GPU);

	system("pause");
	return 0;
}
