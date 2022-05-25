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

// NOTE:
// GetMKLDNNFormat function is here temporarily. It is
// needed because without them forward declaration was causing an error when
// building with "-DWITH_TESTING=ON". This file will be deleted after completing
// md-related refactoring

namespace paddle {
namespace platform {

inline dnnl::memory::format_tag GetMKLDNNFormat(dnnl::memory::desc mem_desc) {
  auto ndims = mem_desc.data.ndims;
  auto strides = mem_desc.data.format_desc.blocking.strides;
  auto inner_nblks = mem_desc.data.format_desc.blocking.inner_nblks;
  auto inner_blks = mem_desc.data.format_desc.blocking.inner_blks;
  auto inner_idxs = mem_desc.data.format_desc.blocking.inner_idxs;

  if (ndims == 1) {
    return dnnl::memory::format_tag::x;
  } else if (ndims == 2) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1]) {
        return dnnl::memory::format_tag::nc;
      } else {
        return dnnl::memory::format_tag::cn;
      }
    }
  } else if (ndims == 3) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2]) {
        return dnnl::memory::format_tag::ncw;
      } else if (strides[1] >= strides[0] && strides[0] >= strides[2]) {
        return dnnl::memory::format_tag::ntc;
      } else {
        return dnnl::memory::format_tag::nwc;
      }
    }
  } else if (ndims == 4) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3]) {
        return dnnl::memory::format_tag::nchw;
      } else if (strides[2] >= strides[3] && strides[3] >= strides[1] &&
                 strides[1] >= strides[0]) {
        return dnnl::memory::format_tag::cdba;
      } else if (strides[3] >= strides[2] && strides[2] >= strides[0] &&
                 strides[0] >= strides[1]) {
        return dnnl::memory::format_tag::dcab;
      } else {
        return dnnl::memory::format_tag::nhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw16c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw8c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return dnnl::memory::format_tag::Acdb8a;
        }
      } else if (inner_blks[0] == 4 && inner_idxs[0] == 1) {
        return dnnl::memory::format_tag::nChw4c;
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return dnnl::memory::format_tag::Acdb16a;
        }
      }
    } else if (inner_nblks == 2) {
      if (inner_blks[0] == 16 && inner_blks[1] == 16) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return dnnl::memory::format_tag::OIhw16i16o;
        }
      } else if (inner_blks[0] == 8 && inner_blks[1] == 8) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return dnnl::memory::format_tag::OIhw8i8o;
        }
      }
    }
  } else if (ndims == 5) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4]) {
        return dnnl::memory::format_tag::abcde;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[1] &&
                 strides[1] >= strides[3] && strides[3] >= strides[4]) {
        return dnnl::memory::format_tag::acbde;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
                 strides[3] >= strides[4] && strides[4] >= strides[1]) {
        return dnnl::memory::format_tag::acdeb;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return dnnl::memory::format_tag::Acdeb8a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::Abcde8a;
        }
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::aBcde8b;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return dnnl::memory::format_tag::Acdeb16a;
        }
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::Abcde16a;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return dnnl::memory::format_tag::aBcde16b;
        }
      }
    }
  } else if (ndims == 6) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4] &&
          strides[4] >= strides[5]) {
        return dnnl::memory::format_tag::abcdef;
      } else if (strides[0] >= strides[2] && strides[2] >= strides[1] &&
                 strides[1] >= strides[3] && strides[3] >= strides[4] &&
                 strides[4] >= strides[5]) {
        return dnnl::memory::format_tag::acbdef;
      }
    }
  }
  // DEBUG CODE - KEEP UNTILL TENSOR.MEMORY_DESC IMPLEMENTED
  // std::cout<<"@@@@@@@@@@ UNDEFINED FORMAT @@@@@@@@@@@@@@@@@@@"<<std::endl;
  // std::cout<<"NDIMS: "<<ndims<<std::endl;
  // std::cout<<"INNER_NBLKS: "<<inner_nblks<<std::endl;
  // for (int i=0;i<ndims;++i) {
  //   std::cout<<"STRIDE["<<i<<"]: "<<strides[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_BLKS["<<i<<"]: "<<inner_blks[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_IDXS["<<i<<"]: "<<inner_idxs[i]<<std::endl;
  // }
  return dnnl::memory::format_tag::undef;
}

}  // namespace platform
}  // namespace paddle
