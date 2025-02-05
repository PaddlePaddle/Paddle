/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstring>
#include <memory>

#include "paddle/common/hostdevice.h"
#include "paddle/common/macros.h"

namespace phi {
namespace strings {

HOSTDEVICE inline bool IsSpace(uint32_t chr) { return (chr & 16) > 0; }

HOSTDEVICE inline bool IsAlpha(uint32_t chr) { return (chr & 8) > 0; }

HOSTDEVICE inline bool IsDigit(uint32_t chr) { return (chr & 4) > 0; }

HOSTDEVICE inline bool IsNumeric(uint32_t chr) { return (chr & 2) > 0; }

HOSTDEVICE inline bool IsDecimal(uint32_t chr) { return (chr & 1) > 0; }

HOSTDEVICE inline bool IsAlphaNum(uint32_t chr) { return (chr & 15) > 0; }

HOSTDEVICE inline bool IsUpper(uint32_t chr) { return (chr & 32) > 0; }

HOSTDEVICE inline bool IsLower(uint32_t chr) { return (chr & 64) > 0; }

HOSTDEVICE inline uint32_t BytesInUtf8Char(uint8_t byte) {
  unsigned int count = 1;
  // no if-statements means no divergence
  count += static_cast<int>((byte & 0xF0) == 0xF0);
  count += static_cast<int>((byte & 0xE0) == 0xE0);
  count += static_cast<int>((byte & 0xC0) == 0xC0);
  count -= static_cast<int>((byte & 0xC0) == 0x80);
  return count;
}

HOSTDEVICE inline uint32_t UTF8ToUInt32(const char* pSrc, uint32_t* chr) {
  uint32_t chwidth = BytesInUtf8Char(static_cast<uint8_t>(*pSrc));
  *chr = static_cast<uint32_t>(*pSrc++) & 0xFF;
  if (chwidth > 1) {
    *chr = (*chr) << 8;
    *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 8;
    if (chwidth > 2) {
      *chr = (*chr) << 8;
      *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 16;
      if (chwidth > 3) {
        *chr = (*chr) << 8;
        *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 24;
      }
    }
  }
  return chwidth;
}

HOSTDEVICE inline uint32_t UTF8ToUnicode(uint32_t utf8) {
  uint32_t unchr = 0;
  if (utf8 < 0x00000080) {
    unchr = utf8;
  } else if (utf8 < 0x0000E000) {
    unchr = (utf8 & 0x1F00) >> 2;
    unchr |= (utf8 & 0x003F);
  } else if (utf8 < 0x00F00000) {
    unchr = (utf8 & 0x0F0000) >> 4;
    unchr |= (utf8 & 0x003F00) >> 2;
    unchr |= (utf8 & 0x00003F);
  } else if (utf8 <= static_cast<uint32_t>(0xF8000000)) {
    unchr = (utf8 & 0x03000000) >> 6;
    unchr |= (utf8 & 0x003F0000) >> 4;
    unchr |= (utf8 & 0x00003F00) >> 2;
    unchr |= (utf8 & 0x0000003F);
  }
  return unchr;
}

HOSTDEVICE inline uint32_t UnicodeToUTF8(uint32_t unchr) {
  uint32_t utf8 = 0;
  if (unchr < 0x00000080) {
    utf8 = unchr;
  } else if (unchr < 0x00000800) {
    utf8 = (unchr << 2) & 0x1F00;
    utf8 |= (unchr & 0x3F);
    utf8 |= 0x0000C080;
  } else if (unchr < 0x00010000) {
    utf8 = (unchr << 4) & 0x0F0000;   // upper 4 bits
    utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);           // last 6 bits
    utf8 |= 0x00E08080;
  } else if (unchr < 0x00110000) {      // 3-byte unicode
    utf8 = (unchr << 6) & 0x07000000;   // upper 3 bits
    utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
    utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);             // last 6 bits
    utf8 |= static_cast<uint32_t>(0xF0808080);
  }
  return utf8;
}

HOSTDEVICE inline uint32_t BytesInUnicodeChar(uint32_t chr) {
  uint32_t count = 1;
  // no if-statements means no divergence
  count += static_cast<int>((chr & static_cast<uint32_t>(0x0000FF00)) > 0);
  count += static_cast<int>((chr & static_cast<uint32_t>(0x00FF0000)) > 0);
  count += static_cast<int>((chr & static_cast<uint32_t>(0xFF000000)) > 0);
  return count;
}

HOSTDEVICE inline uint32_t UnicodeToUTF8Char(uint32_t chr, char* dst) {
  uint32_t chwidth = BytesInUnicodeChar(chr);
  for (uint32_t idx = 0; idx < chwidth; ++idx) {
    dst[chwidth - idx - 1] = static_cast<char>(chr & 0xFF);
    chr = chr >> 8;
  }
  return chwidth;
}

HOSTDEVICE inline void GetUnicodeStr(const char* pSrc,
                                     uint32_t* unicode_str,
                                     size_t unicode_len) {
  uint32_t curr_unicode_char;
  uint32_t count = UTF8ToUInt32(pSrc, &curr_unicode_char);
  curr_unicode_char = UTF8ToUnicode(curr_unicode_char);
  for (size_t i = 0; i < unicode_len; ++i) {
    unicode_str[i] = curr_unicode_char;
    pSrc += count;
    count = UTF8ToUInt32(pSrc, &curr_unicode_char);
    curr_unicode_char = UTF8ToUnicode(curr_unicode_char);
  }
}

HOSTDEVICE inline uint32_t GetUnicodeStrLen(const char* pSrc, size_t size) {
  uint32_t curr_unicode_char;
  uint32_t count = 0;
  uint32_t offset = 0;
  uint32_t curr_count = 0;
  while (offset < size) {
    curr_count = UTF8ToUInt32(pSrc, &curr_unicode_char);
    offset += curr_count;
    pSrc += curr_count;
    if (curr_count == 0) {
      break;
    }
    ++count;
  }
  return count;
}

HOSTDEVICE inline uint32_t GetUTF8StrLen(const uint32_t* unicode_str,
                                         size_t unicode_len) {
  uint32_t utf8_str_count = 0;
  for (size_t i = 0; i < unicode_len; ++i) {
    uint32_t utf8_uint32 = UnicodeToUTF8(unicode_str[i]);
    utf8_str_count += BytesInUnicodeChar(utf8_uint32);
  }
  // +1 means '\0'
  return utf8_str_count + 1;
}
// Need to guarantee utf8_str has enough memory

HOSTDEVICE inline void GetUTF8Str(const uint32_t* unicode_str,
                                  char* utf8_str,
                                  size_t unicode_len) {
  char dst_char[5] = {0};
  for (size_t i = 0; i < unicode_len; ++i) {
    uint32_t utf8_uint32 = UnicodeToUTF8(unicode_str[i]);
    uint32_t utf8_char_count = UnicodeToUTF8Char(utf8_uint32, dst_char);
    dst_char[utf8_char_count] = '\0';
    memcpy(utf8_str, dst_char, utf8_char_count);
    utf8_str += utf8_char_count;
  }
  *utf8_str = '\0';
}

const uint8_t* GetUniFlagMap();
const uint16_t* GetCharCasesMap();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

const uint8_t* GetGPUUniflagMap();
const uint16_t* GetGPUCharCasesMap();
#endif

}  // namespace strings
}  // namespace phi
