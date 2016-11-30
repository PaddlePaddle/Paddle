/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "DataTransformer.h"
#include <time.h>
#include <limits>

DataTransformer::DataTransformer(int threadNum,
                                 int capacity,
                                 bool isTest,
                                 bool isColor,
                                 int cropHeight,
                                 int cropWidth,
                                 int imgSize,
                                 bool isEltMean,
                                 bool isChannelMean,
                                 float* meanValues)
    : isTest_(isTest),
      isColor_(isColor),
      cropHeight_(cropHeight),
      cropWidth_(cropWidth),
      imgSize_(imgSize),
      capacity_(capacity),
      prefetchFree_(capacity),
      prefetchFull_(capacity) {
  fetchCount_ = -1;
  scale_ = 1.0;
  isChannelMean_ = isChannelMean;
  isEltMean_ = isEltMean;
  loadMean(meanValues);

  imgPixels_ = cropHeight * cropWidth * (isColor_ ? 3 : 1);

  prefetch_.reserve(capacity);
  for (int i = 0; i < capacity; i++) {
    auto d = std::make_shared<DataType>(new float[imgPixels_ * 3], 0);
    prefetch_.push_back(d);
    memset(prefetch_[i]->first, 0, imgPixels_ * sizeof(float));
    prefetchFree_.enqueue(prefetch_[i]);
  }

  numThreads_ = 12;
  syncThreadPool_.reset(new SyncThreadPool(numThreads_, false));
}

void DataTransformer::loadMean(float* values) {
  if (values) {
    int c = isColor_ ? 3 : 1;
    int sz = isChannelMean_ ? c : cropHeight_ * cropWidth_ * c;
    meanValues_ = new float[sz];
    memcpy(meanValues_, values, sz * sizeof(float));
  }
}

void DataTransformer::startFetching(const char* src,
                                    const int size,
                                    float* trg) {
  vector<char> imbuf(src, src + size);
  int cvFlag = (isColor_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat im = cv::imdecode(cv::Mat(imbuf), cvFlag);
  if (!im.data) {
    LOG(ERROR) << "Could not decode image";
    LOG(ERROR) << im.channels() << " " << im.rows << " " << im.cols;
  }
  this->transform(im, trg);
}

int DataTransformer::Rand(int min, int max) {
  std::random_device source;
  std::mt19937 rng(source());
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

void DataTransformer::transform(Mat& cvImgOri, float* target) {
  const int imgChannels = cvImgOri.channels();
  const int imgHeight = cvImgOri.rows;
  const int imgWidth = cvImgOri.cols;
  const bool doMirror = (!isTest_) && Rand(0, 1);
  int h_off = 0;
  int w_off = 0;
  int th = imgHeight;
  int tw = imgWidth;
  cv::Mat img;
  if (imgSize_ > 0) {
    if (imgHeight > imgWidth) {
      tw = imgSize_;
      th = int(double(imgHeight) / imgWidth * tw);
      th = th > imgSize_ ? th : imgSize_;
    } else {
      th = imgSize_;
      tw = int(double(imgWidth) / imgHeight * th);
      tw = tw > imgSize_ ? tw : imgSize_;
    }
    cv::resize(cvImgOri, img, cv::Size(tw, th));
  } else {
    cv::Mat img = cvImgOri;
  }

  cv::Mat cv_cropped_img = img;
  if (cropHeight_ && cropWidth_) {
    if (!isTest_) {
      h_off = Rand(0, th - cropHeight_);
      w_off = Rand(0, tw - cropWidth_);
    } else {
      h_off = (th - cropHeight_) / 2;
      w_off = (tw - cropWidth_) / 2;
    }
    cv::Rect roi(w_off, h_off, cropWidth_, cropHeight_);
    cv_cropped_img = img(roi);
  } else {
    CHECK_EQ(cropHeight_, imgHeight);
    CHECK_EQ(cropWidth_, imgWidth);
  }
  int height = cropHeight_;
  int width = cropWidth_;
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < imgChannels; ++c) {
        if (doMirror) {
          top_index = (c * height + h) * width + width - 1 - w;
        } else {
          top_index = (c * height + h) * width + w;
        }
        float pixel = static_cast<float>(ptr[img_index++]);
        if (isEltMean_) {
          int mean_index = (c * imgHeight + h) * imgWidth + w;
          target[top_index] = (pixel - meanValues_[mean_index]) * scale_;
        } else {
          if (isChannelMean_) {
            target[top_index] = (pixel - meanValues_[c]) * scale_;
          } else {
            target[top_index] = pixel * scale_;
          }
        }
      }
    }
  }  // target: BGR
}

void DataTransformer::start(vector<char*>& data, int* datalen, int* labels) {
  auto job = [&](int tid, int numThreads) {
    for (int i = tid; i < data.size(); i += numThreads) {
      DataTypePtr ret = prefetchFree_.dequeue();
      char* buf = data[i];
      int size = datalen[i];
      ret->second = labels[i];
      this->startFetching(buf, size, ret->first);
      prefetchFull_.enqueue(ret);
    }
  };
  syncThreadPool_->exec(job);
  fetchCount_ = data.size();
}

void DataTransformer::obtain(float* data, int* label) {
  fetchCount_--;
  if (fetchCount_ < 0) {
    LOG(FATAL) << "Empty data";
  }
  DataTypePtr ret = prefetchFull_.dequeue();
  *label = ret->second;
  memcpy(data, ret->first, sizeof(float) * imgPixels_);
  prefetchFree_.enqueue(ret);
}
