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

#include "paddle/fluid/inference/tests/api/data_reader.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include "paddle/fluid/inference/api/helper.h"

namespace paddle {

namespace {

// Images are resized to ResizeSize x ResizeSize ...
const int ResizeSize = 256;
// ... and then cropped to CropSize x CropSize
const int CropSize = 224;

cv::Mat center_crop_image(cv::Mat img, int width, int height) {
  auto w_start = (img.cols - width) / 2;
  auto h_start = (img.rows - height) / 2;

  cv::Rect roi(w_start, h_start, width, height);
  return img(roi);
}

cv::Mat resize_short(cv::Mat img, int target_size) {
  auto percent = static_cast<float>(target_size) / std::min(img.cols, img.rows);
  auto resized_width = static_cast<int>(round(img.cols * percent));
  auto resized_height = static_cast<int>(round(img.rows * percent));
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(resized_width, resized_height), 0, 0,
             cv::INTER_LANCZOS4);
  return resized_img;
}

}  // namespace

void DataReader::drawImages(float* input, bool is_rgb, int batch_size,
                            int channels, int width, int height) {
  for (int b = 0; b < batch_size; b++) {
    std::vector<cv::Mat> fimage_channels;
    for (int c = 0; c < channels; c++) {
      fimage_channels.emplace_back(
          cv::Size(width, height), CV_32FC1,
          input + width * height * c + width * height * channels * b);
    }
    cv::Mat mat;
    if (is_rgb) {
      std::swap(fimage_channels[0], fimage_channels[2]);
    }
    cv::merge(fimage_channels, mat);
    cv::imshow(std::to_string(b) + " output image", mat);
  }
  std::cout << "Press any key in image window or close it to continue"
            << std::endl;
  cv::waitKey(0);
}

DataReader::DataReader(const std::string& data_list_path,
                       const std::string& data_dir_path, int width, int height,
                       int channels, bool convert_to_rgb)
    : data_list_path(data_list_path),
      data_dir_path(data_dir_path),
      file(data_list_path),
      width(width),
      height(height),
      channels(channels),
      convert_to_rgb(convert_to_rgb) {
  if (!file.is_open()) {
    throw std::invalid_argument("Cannot open data list file " + data_list_path);
  }

  if (data_dir_path.empty()) {
    throw std::invalid_argument("Data directory must be set to use imagenet.");
  }

  if (channels != 3) {
    throw std::invalid_argument("Only 3 channel image loading supported");
  }

  if (!(width == height && width == CropSize)) {
    std::stringstream ss;
    ss << "Width and heigth must be both " << CropSize
       << " because this reader is for "
          "validation which does resize of smaller edge to "
       << ResizeSize << " and "
                        "center crop of ("
       << CropSize << ", " << CropSize << "). Your width and heigth are: ("
       << width << ", " << height << ")." << std::endl;
    throw std::invalid_argument(ss.str());
  }
}

// return true if separator works or false otherwise
bool DataReader::SetSeparator(char separator) {
  sep = separator;

  std::string line;
  auto position = file.tellg();
  std::getline(file, line);
  file.clear();
  file.seekg(position);

  // test out
  std::vector<std::string> pieces;
  inference::split(line, separator, &pieces);

  return (pieces.size() == 2);
}

bool DataReader::NextBatch(float* input, int64_t* label, int batch_size,
                           bool debug_display_images) {
  std::string line;

  for (int i = 0; i < batch_size; i++) {
    if (!std::getline(file, line)) return false;

    std::vector<std::string> pieces;
    inference::split(line, sep, &pieces);
    if (pieces.size() != 2) {
      std::stringstream ss;
      ss << "invalid number of separators '" << sep << "' found in line " << i
         << ":'" << line << "' of file " << data_list_path;
      throw std::runtime_error(ss.str());
    }

    auto filename = data_dir_path + pieces.at(0);
    label[i] = std::stoi(pieces.at(1));

    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (image.data == nullptr) {
      std::string error_msg = "Couldn't open file " + filename;
      throw std::runtime_error(error_msg);
    }

    if (convert_to_rgb) {
      cv::cvtColor(image, image, CV_BGR2RGB);
    }

    if (debug_display_images)
      cv::imshow(std::to_string(i) + " input image", image);

    cv::Mat image_resized = resize_short(image, ResizeSize);
    cv::Mat image_cropped = center_crop_image(image_resized, width, height);

    cv::Mat fimage;
    image_cropped.convertTo(fimage, CV_32FC3);

    fimage /= 255.f;

    cv::Scalar mean(0.406f, 0.456f, 0.485f);
    cv::Scalar std(0.225f, 0.224f, 0.229f);

    if (convert_to_rgb) {
      std::swap(mean[0], mean[2]);
      std::swap(std[0], std[2]);
    }

    std::vector<cv::Mat> fimage_channels;
    cv::split(fimage, fimage_channels);

    for (int c = 0; c < channels; c++) {
      fimage_channels[c] -= mean[c];
      fimage_channels[c] /= std[c];
      for (int row = 0; row < fimage.rows; ++row) {
        const float* fimage_begin = fimage_channels[c].ptr<const float>(row);
        const float* fimage_end = fimage_begin + fimage.cols;
        std::copy(fimage_begin, fimage_end,
                  input + row * fimage.cols + c * fimage.cols * fimage.rows +
                      i * 3 * fimage.cols * fimage.rows);
      }
    }
  }
  return true;
}

}  // namespace paddle
