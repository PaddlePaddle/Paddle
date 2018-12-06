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

#include <Windows.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// note the image pre-process works for icnet only
const int C = 3;    // image channel
const int H = 449;  // image height
const int W = 581;  // image width
const float Image_Mean[3] = {112.15f, 109.41f, 185.42f};

struct Record {
  float* data;
  std::vector<int32_t> shape;
  Record() { data = new float[C * H * W]; }
  ~Record() {
    if (data != nullptr) {
      delete data;
      data = nullptr;
    }
  }
};

class ImageProcess {
 public:
  static bool image_read(std::string imagename, unsigned char*& imagebuf,
                         int& imagewidth, int& imageheight,
                         int& imagebitcount) {
    int imagelinebyte = 0;
    FILE* filep;
    errno_t filerror = fopen_s(&filep, &imagename[0], "rb");
    if (filerror == 0) {
      if (!fseek(filep, sizeof(BITMAPFILEHEADER), 0)) {
        BITMAPINFOHEADER Image_Header;
        fread(&Image_Header, sizeof(BITMAPINFOHEADER), 1, filep);
        imagewidth = Image_Header.biWidth;
        imageheight = Image_Header.biHeight;
        imagebitcount = Image_Header.biBitCount;
        imagelinebyte = (imagewidth * imagebitcount / 8 + 3) / 4 * 4;
        if (imagebitcount == 8) {
          fread(new RGBQUAD[256], sizeof(RGBQUAD), 256, filep);
        }
        imagebuf = new unsigned char[imagelinebyte * imageheight];
        fread(imagebuf, 1, imagelinebyte * imageheight, filep);
        unsigned char* imagedatabuf =
            new unsigned char[imagewidth * imageheight * 3];
        for (int i = 0; i < imageheight; i++) {
          for (int j = 0; j < imagewidth; j++) {
            for (int k = 0; k < 3; k++) {
              imagedatabuf[k * imagewidth * imageheight + i * imagewidth + j] =
                  imagebuf[i * imagelinebyte + j * 3 + k];
            }
          }
        }
        delete imagebuf;
        imagebuf = imagedatabuf;
      }
      fclose(filep);

      return true;
    }

    return false;
  }

  // save the data into BMP file
  static bool image_save(const char* imagename, unsigned char* imagebuf,
                         int imagewidth, int imageheight, int imagebitcount) {
    if (!imagebuf) return 0;
    int colorTablesize = 0;
    if (imagebitcount == 8) {
      colorTablesize = 256;
    }
    int imagelinebyte = (imagewidth * imagebitcount / 8 + 3) / 4 * 4;
    FILE* fp;
    fopen_s(&fp, imagename, "wb");
    if (fp == 0) {
      return false;
    }
    BITMAPFILEHEADER fileHead;
    fileHead.bfType = 0x4D42;
    fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) +
                      colorTablesize + imagelinebyte * imageheight;
    fileHead.bfReserved1 = 0;
    fileHead.bfReserved2 = 0;
    fileHead.bfOffBits =
        sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize;
    fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
    BITMAPINFOHEADER head;
    head.biBitCount = imagebitcount;
    head.biClrImportant = 0;
    head.biClrUsed = 0;
    head.biCompression = 0;
    head.biHeight = imageheight;
    head.biPlanes = 1;
    head.biSize = 40;
    head.biSizeImage = imagelinebyte * imageheight;
    head.biWidth = imagewidth;
    head.biXPelsPerMeter = 0;
    head.biYPelsPerMeter = 0;
    // write the header
    fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
    if (imagebitcount == 8) {
      RGBQUAD rgbquad[256];
      for (int i = 0; i < 256; i++) {
        rgbquad[i].rgbBlue = i;
        rgbquad[i].rgbGreen = i;
        rgbquad[i].rgbRed = i;
        rgbquad[i].rgbReserved = 0;
      }
      fwrite(rgbquad, sizeof(RGBQUAD), 256, fp);
    }
    fwrite(imagebuf, imagelinebyte * imageheight, 1, fp);
    fclose(fp);

    return true;
  }

  static void bmp_save(const char* filename, int64_t* resultimage,
                       int image_size) {
    int imagelinebyte = (W * 8 / 8 + 3) / 4 * 4;
    unsigned char* imagebuffer = new unsigned char[imagelinebyte * H];

    unsigned char temp = 0;
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        temp = (unsigned char)resultimage[i * W + j];
        if (resultimage[i * W + j] != 0)
          imagebuffer[i * imagelinebyte + j] = 255;
        else
          imagebuffer[i * imagelinebyte + j] = 0;
      }
    }

    image_save(filename, imagebuffer, W, H, 8);
  }

  static bool preprocess_image(Record& record, const char* filename) {
    unsigned char* Image_Buf = nullptr;
    int Image_Width = 0;
    int Image_Height = 0;
    int Image_Bitcount = 0;
    int i = 0;

    if (!image_read(filename, Image_Buf, Image_Width, Image_Height,
                    Image_Bitcount)) {
      return false;
    }
    for (int channel = 0; channel < 3; channel++) {
      for (int height = 0; height < Image_Height; height++) {
        for (int width = 0; width < Image_Width; width++) {
          record.data[i++] =
              float((float)Image_Buf[channel * Image_Height * Image_Width +
                                     height * Image_Width + width] -
                    Image_Mean[channel]);
        }
      }
    }
    if (Image_Buf != nullptr) {
      delete Image_Buf;
    }

    // NCHW
    record.shape.push_back(1);
    record.shape.push_back(C);
    record.shape.push_back(H);
    record.shape.push_back(W);

    return true;
  }
};
