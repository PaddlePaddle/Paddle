
#include<windows.h>
#include <fstream>
#include "inference_icnet.h"
#include <thread>
#include <vector>
#include <string>
#include <iostream>

#include <sstream>
using namespace std;


template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

void test_imgs() {
	void *h = init_predictor("./lb/__model__", "./lb/__params__", 0.3f, true, 0);

	std::ifstream infile("new_file.list");
	std::ofstream ofs("./1.png.output.txt");

	std::string temp_s;
	std::vector<std::string> all_files;
	while (!infile.eof()) {
		infile >> temp_s;
		all_files.push_back(temp_s);
	}
	// size_t file_num = all_files.size();
	infile.close();
	// =============read file list =============
	for (size_t f_k = 0; f_k < 1; f_k++) {
		// std::string path = "D:\\Paddle\\paddle\\fluid\\inference\\api\\demo_ci\\build\\Release\\";
		// std::ifstream in_img(path + all_files[f_k]);
		std::string mypath = "D:\\Paddle\\paddle\\fluid\\inference\\api\\demo_ci\\build\\Release\\1.png.txt";
		std::cout << "file" << mypath << std::endl;
		std::ifstream in_img(mypath);
		//std::cout << path + all_files[f_k] << std::endl;
		double temp_v;
		const int size = 3 * 449 * 581 * 1;
		float * data = new float[size];
		std::string value;

		if (!in_img.is_open()) {
			cout << "open failed" << endl;
		}
		double sum_input = .0;
		for (auto i = 0; i < size; i++) {
			getline(in_img, value, '\n');
			double v = stringToNum<double>(value);
			data[i] = static_cast<float>(v);
			sum_input += v;
		}  
		std::cout << "sum_input" << sum_input << std::endl;

		in_img.close();
		const int SIZE = 449 * 581 * 1;
		int64_t * p = new int64_t[SIZE]();
		int out_size = 0;
		//memset(p, 0, size);
		predict(h, data, 3, 449, 581, &p, &out_size, 1);
		std::cout << "out_size = " << out_size << std::endl;
	
		double out_sum = .0;
		for (auto i = 0; i < out_size / sizeof(int64_t); i++) {
			out_sum += p[i];
			ofs << p[i] << " ";
		}
		ofs.close();

		std::cout << "inferece out sum" << out_sum << std::endl;
		delete p;
	}

	destory_predictor(h);
}

int main(int argc, char** argv) {
	//if (true) {
	//	std::thread t1(func, init_predictor("./infer_model/__model__", "./infer_model/__params__", 0.1f, true, 0));
	//	std::thread t2(func, init_predictor("./infer_model/__model__", "./infer_model/__params__", 0.1f, true, 0));
	//	//std::thread t3(func, init_predictor("./infer_model/__model__", "./infer_model/__params__", 0.1f, true, 0));
	//	//std::thread t4(func, init_predictor("./infer_model/__model__", "./infer_model/__params__", 0.1f, true, 0));
	//	t1.join();
	//	t2.join();
	//	//t3.join();
	//	//t4.join();
	//	//Sleep(1);
	//}
	test_imgs();

  return 0;
}
