/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>
#include "paddle/utils/Logging.h"
#include "mkldnn.hpp"

namespace paddle {
/**
 * @brief A MKLDNN memory buffer class
 *
 * 
 */

typedef enum { 
  dnnUser2Intl = 0,
  dnnIntl2User, 
} dnnCvtType_t;

using dnnfmt = mkldnn::memory::format;
// This format should be matched with the version of MKLDNN
const static std::map<dnnfmt, std::string> DNN_FORMAT_STR = {
  {dnnfmt::format_undef, "format_undef"}, {dnnfmt::any, "any"},
  {dnnfmt::blocked, "blocked"}, {dnnfmt::x, "x"}, {dnnfmt::nc, "nc"},
  {dnnfmt::nchw, "nchw"}, {dnnfmt::nhwc, "nhwc"}, {dnnfmt::chwn, "chwn"}, 
  {dnnfmt::nChw8c, "nChw8c"}, {dnnfmt::nChw16c, "nChw16c"}, {dnnfmt::oi, "oi"},
  {dnnfmt::io, "io"}, {dnnfmt::oihw, "oihw"}, {dnnfmt::ihwo, "ihwo"}, 
  {dnnfmt::oIhw8i, "oIhw8i"}, {dnnfmt::oIhw16i, "oIhw16i"},
  {dnnfmt::OIhw8i8o, "OIhw8i8o"}, {dnnfmt::OIhw16i16o, "OIhw16i16o"},
  {dnnfmt::OIhw8o8i, "OIhw8o8i"}, {dnnfmt::OIhw16o16i, "OIhw16o16i"},
  {dnnfmt::OIhw8i16o2i, "OIhw8i16o2i"}, {dnnfmt::Ohwi8o, "Ohwi8o"},
  {dnnfmt::Ohwi16o, "Ohwi16o"}, {dnnfmt::OhIw16o4i, "OhIw16o4i"},
  {dnnfmt::goihw, "goihw"}, {dnnfmt::gOIhw8i8o, "gOIhw8i8o"},
  {dnnfmt::gOIhw16i16o, "gOIhw16i16o"}, {dnnfmt::gOIhw8i16o2i, "gOIhw8i16o2i"},
  // {dnnfmt::gOhwi8o, "gOhwi8o"}, {dnnfmt::gOhwi16o, "gOhwi16o"},  // later ver
  {dnnfmt::gOIhw8o8i, "gOIhw8o8i"}, {dnnfmt::gOIhw16o16i, "gOIhw16o16i"},
  {dnnfmt::gOhIw16o4i, "gOhIw16o4i"}};


class MkldnnBuffer {
using mem = mkldnn::memory;

protected:
  /// user and internal memory
  std::shared_ptr<mem> pUser_;
  std::shared_ptr<mem> pIntl_;

  /// data type: only support f32 yet
  mem::data_type tp_;

  /// conversion handle
  std::shared_ptr<mkldnn::primitive> pReorder;


public:
  explicit MkldnnBuffer(
    mem::data_type type = mem::data_type::f32) :
    pUser_(nullptr),
    pIntl_(nullptr),
    pReorder(nullptr) {
    tp_ = type;
    if (tp_ != mem::data_type::f32)
      LOG(FATAL) << "only support float 32 so far";
  }

  ~MkldnnBuffer() {}

  /**
   * get memory desc from details info
   */
  static inline mem::desc getMD(const mem::dims& dm,
    const mem::format& fmt = mem::format::any,
    const mem::data_type &tp = mem::data_type::f32) {
    return mem::desc({dm}, tp, fmt);
  }

  /**
   * reset user buffer functions
   */
  void resetUser(void *pd, const mem::dims& dm,
    const mem::format& fmt, const mkldnn::engine& eg) {
    resetUser(pd, mem::desc({dm}, tp_, fmt), eg);
  }

  void resetUser(
    void *pdata, const mem::desc& md, const mkldnn::engine& eg) {
    resetUser(pdata, mem::primitive_desc(md, eg));
  }

  void resetUser(void *pdata, const mem::primitive_desc& pd) {
    resetMem(pUser_, pd, pdata);
  }

  void resetUser(const std::shared_ptr<mem>& user) {
    pUser_ = user;
  }

  /**
   * update user data handle
   */
  void updateUserData(void* data) {
    setData(pUser_, data);
  }

  /**
   * reset internal buffer functions
   * generally, it should be after resetUser
   */
  void resetIntl(const mem::primitive_desc& intlPD, void *pdata = NULL) {
    if (nullptr != pUser_ && 
      getUserPD() == intlPD && (pdata == NULL || pdata == getUserData())) {
      pIntl_ = pUser_;
      return;
    }

    if (nullptr == pUser_) {
      LOG(WARNING) << "Generally, internal buffer should be after resetUser";
    }
    resetMem(pIntl_, intlPD, pdata);
  }
  
  void resetIntl(const std::shared_ptr<mem>& intl) {
    pIntl_ = intl;
  }

  /**
   * Other get functions of user
   */
  const std::shared_ptr<mem>& getUser() {
    return pUser_;
  }

  // get primitive desc
  // do not use const &
  mem::primitive_desc getUserPD() {
    return getPD(pUser_);
  }

  // get memory desc
  mem::desc getUserMD() {
    return getMD(pUser_);
  }

  // get user memory format
  std::string getUserFmt() {
    dnnfmt fmt = dnnfmt(getFmt(pUser_));
    std::string fmtStr;
    CHECK(mapGet(fmt, DNN_FORMAT_STR, &fmtStr)) << "invalid format: " << fmt;
    return fmtStr;
  }

  const void* getUserData() {
    return getData(pUser_);
  }

  /// it's the element size not memory size
  size_t getUserSize() {
    return getElementCnt(pUser_);
  }

  /**
   * Other get functions of internal
   */
  const std::shared_ptr<mem>& getIntl() {
     return pIntl_;
  }

  // get primitive desc
  mem::primitive_desc getIntlPD() {
    return getPD(pIntl_);
  }

  // get memory desc
  mem::desc getIntlMD() {
    return getMD(pIntl_);
  }

  // get internal memory format
  std::string getIntlFmt() {
    CHECK(pIntl_) << "shoud have inited internal buffer";
    dnnfmt fmt = dnnfmt(getFmt(pIntl_));
    std::string fmtStr;
    CHECK(mapGet(fmt, DNN_FORMAT_STR, &fmtStr)) << "invalid format: " << fmt;
    return fmtStr;
  }

  const void* getIntlData() {
    return getData(pIntl_);
  }
  
  /// it's the element size not memory size
  size_t getIntlSize() {
    return getElementCnt(pIntl_);
  }

  /****************************************************************************/
  /**
   * Functions for reorder
   */
  bool needReorder() {
    if (pUser_ == pIntl_) {
      return false;
    }
    if (getIntlPD() == getUserPD() && getIntlData() == getUserData()) {
      return false;
    }
    return true;
  }

  void resetReorder(dnnCvtType_t type) {
    CHECK(type == dnnUser2Intl || type == dnnIntl2User)
      << "please specify one type of reorder";

    if (!needReorder()) {
      pReorder = nullptr;
      return;
    }

    if (type == dnnUser2Intl) {
      pReorder.reset(new mkldnn::reorder(*pUser_, *pIntl_));
    } else {
      pReorder.reset(new mkldnn::reorder(*pIntl_, *pUser_));
    }
  }

  const std::shared_ptr<mkldnn::primitive>& getReorder() {
    if (nullptr == pReorder) {
      LOG(WARNING) << "reoder should not be empty! reset it before get";
    }
    return pReorder;
  }


  /*************** protected functions ***************************************************/
protected:
  /**
   * Reset memory buffer
   */
  void resetMem(
    std::shared_ptr<mem>& pMem, const mem::primitive_desc& pd, void* pdata = NULL) {
    checkType(pd);
    mem::primitive_desc tmp = pd;
    if (pdata == NULL) {
      pMem.reset(new mem(tmp));
    } else {
      pMem.reset(new mem(tmp, pdata));
    }
  }

  /**
   * Get primitive desc
   * caution: get_primitive_desc() return a local var
   */
  inline mem::primitive_desc getPD(const std::shared_ptr<mem>& pMem) {
    CHECK(pMem) << "should have mkldnn memory";
    return pMem->get_primitive_desc();
  }

  // get memory desc from mkldnn memory
  inline mem::desc getMD(const std::shared_ptr<mem>& pMem) {
    return getMD(getPD(pMem));
  }

  // get memory desc from mkldnn memory
  inline mem::desc getMD(const mem::primitive_desc& pd) {
    mem::primitive_desc tmp = pd;
    return tmp.desc();
  }

  // get format from memory desc
  inline int getFmt(const mem::desc& md) {
    return md.data.format;
  }

  // get format from primitive desc
  inline int getFmt(const mem::primitive_desc& pd) {
    return getFmt(getMD(pd));
  }

  // get format from memory
  inline int getFmt(const std::shared_ptr<mem>& pMem) {
    return getFmt(getMD(pMem));
  }

  // update the data handle of memory
  inline void setData(const std::shared_ptr<mem>& pMem, void* data) {
    CHECK(pMem) << "should have mkldnn memory";
    pMem->set_data_handle(data);
  }

  // get the data handle of  memory
  inline const void* getData(const std::shared_ptr<mem>& pMem) {
    CHECK(pMem) << "should have mkldnn memory";
    return pMem->get_data_handle();
  }

  size_t getElementCnt(const std::shared_ptr<mem>& pMem) {
    size_t bufferSize = getPD(pMem).get_size();
    size_t unit;
    switch (tp_) {
      case mem::data_type::f32:
        unit = sizeof(float);
        break;
      case mem::data_type::s32:
        unit = sizeof(signed int);
        break;
      default:
        LOG(ERROR) << "Error data type";
        return 0;
    }
    return bufferSize / unit;
  }

  // get dims from memory desc
  inline std::vector<int> getDims(const mem::desc& md) {
    const int* dm = md.data.dims;
    int ndims = md.data.ndims;
    std::vector<int> v(dm, dm + ndims);
    return v;
  }

  void checkType(const mem::desc& md) {
    CHECK_EQ(int(md.data.data_type), int(tp_))
      << "input data type does not match: "
      << md.data.data_type << " vs " << tp_;
  }
  void checkType(const mem::primitive_desc& pd) {
     mem::primitive_desc tmp = pd;
    checkType(tmp.desc());
  }

};

typedef std::shared_ptr<MkldnnBuffer> MkldnnBufferPtr;

}  // namespace paddle
