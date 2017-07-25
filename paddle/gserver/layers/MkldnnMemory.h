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
const static std::map<dnnfmt, std::string> DNN_FMT_STR = {
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
  // {dnnfmt::gOhwi8o, "gOhwi8o"}, {dnnfmt::gOhwi16o, "gOhwi16o"},
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

  void initUser(void *pd, const mem::dims& dm,
    const mem::format& fmt, const mkldnn::engine& eg) {
    resetUser(pd, dm, fmt, eg);
  }

  void initIntl(mem::primitive_desc intlPD) {
    resetIntl(intlPD);
  }

  void initIntl(const std::shared_ptr<mem>& intl) {
    resetIntl(intl);
  }

  void resetUser(void *pd, const mem::dims& dm,
    const mem::format& fmt, const mkldnn::engine& eg) {
    resetUser(pd, mem::desc({dm}, tp_, fmt), eg);
  }

  void checkType(mem::primitive_desc& pd) {
    CHECK_EQ(int(pd.desc().data.data_type), int(tp_))
        << "input data type does not match: "
        << pd.desc().data.data_type << " vs " << tp_;
  }

  void checkType(const mem::desc& md) {
    CHECK_EQ(int(md.data.data_type), int(tp_))
      << "input data type does not match: "
      << md.data.data_type << " vs " << tp_;
  }

  void resetUser(void *pdata, mem::primitive_desc pd) {
    if (pdata == NULL) {
      pUser_.reset(new mem(pd));
    } else {
      checkType(pd);
      pUser_.reset(new mem(pd, pdata));
    }
  }

  void resetUser(
    void *pd, const mem::desc& md, const mkldnn::engine& eg) {
    checkType(md);
    pUser_.reset(new mem(mem::primitive_desc(md, eg), pd));
  }

  void resetUser(const std::shared_ptr<mem>& user) {
    pUser_ = user;
  }

  const std::shared_ptr<mem>& getUser() {
    return pUser_;
  }

  // get internal data handle
  void* getIntlData() {
    CHECK(pIntl_) << "shoud have inited internal buffer";
    return pIntl_->get_data_handle();
  }

  // get internal data handle
  void* getUserData() {
    CHECK(pUser_) << "shoud have inited user buffer";
    return pUser_->get_data_handle();
  }

  void resetIntl(mem::primitive_desc intlPD, void *pdata = NULL) {
    if (nullptr != pUser_
        && pUser_->get_primitive_desc() == intlPD
        && (pdata == NULL || pdata == getUserData())) {
      pIntl_ = pUser_;
    } else {
      checkType(intlPD);
      pIntl_.reset(new mem(intlPD));
    }
  }

  void resetIntl(const std::shared_ptr<mem>& intl) {
    pIntl_ = intl;
  }

  /// functions for getting infos
  size_t getSize(size_t sz) {
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
    return sz / unit;
  }

  /// it's the element size not memory size
  size_t getIntlSize() {
    CHECK(pIntl_) << "haven't init internal layout, call initUser then initCvt";
    return getSize(pIntl_->get_primitive_desc().get_size());
  }

  /// it's the element size not memory size
  size_t getUserSize() {
    CHECK(pUser_) << "haven't init user layout";
    return getSize(pUser_->get_primitive_desc().get_size());
  }

  const std::shared_ptr<mem>& getIntl() {
     return pIntl_;
  }

  // get user primitive desc
  mem::primitive_desc getUserPD() {
    CHECK(pUser_) << "shoud have inited user buffer";
    return pUser_->get_primitive_desc();
  }

  // get internal primitive desc
  mem::primitive_desc getIntlPD() {
    CHECK(pIntl_) << "shoud have inited internal buffer";
    return pIntl_->get_primitive_desc();
  }

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
    if (pUser_ == pIntl_) {
      return;
    }
    if (getIntlPD() == getUserPD() && getIntlData() == getUserData()) {
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

  void updateUserData(void * data) {
    CHECK(pUser_) << "shoud have inited user buffer";
    pUser_->set_data_handle(data);
  }

  static mem::desc getMD(const mem::dims& dm,
    const mem::format& fmt = mem::format::any,
    const mem::data_type &tp = mem::data_type::f32) {
    return mem::desc({dm}, tp, fmt);
  }

  // get user memory desc
  mem::desc getUserMD() {
    CHECK(pUser_) << "shoud have inited user buffer";
    return pUser_->get_primitive_desc().desc();
  }

  // get internal memory desc
  mem::desc getIntlMD() {
    CHECK(pIntl_) << "shoud have inited internal buffer";
    return pIntl_->get_primitive_desc().desc();
  }

  // get format from MD
  static int getMDFmt(const mem::desc& md) {
    return md.data.format;
  }

  // get internal memory format
  std::string getIntlFmt() {
    CHECK(pIntl_) << "shoud have inited internal buffer";
    dnnfmt fmt = dnnfmt(getMDFmt(getIntlMD()));
    std::string fmtStr;
    CHECK(mapGet(fmt, DNN_FMT_STR, &fmtStr)) << "invalid format: " << fmt;
    return fmtStr;
  }

  // get user memory format
  std::string getUserFmt() {
    CHECK(pUser_) << "shoud have inited user buffer";
    dnnfmt fmt = dnnfmt(getMDFmt(getUserMD()));
    std::string fmtStr;
    CHECK(mapGet(fmt, DNN_FMT_STR, &fmtStr)) << "invalid format: " << fmt;
    return fmtStr;
  }













/*
  const std::shared_ptr<mem::desc> getUserMD() {
    CHECK(pUser_) << "haven't init user layout";
    return std::shared_ptr<mem::desc> &(pUser_->get_primitive_desc().desc());
  }



//  std::shared_ptr<memory> getUserMem() {
//     return this->pUser_;
//  }


  static mem::dims getMDDims(const mem::desc& md) {
    const int* dm = md.data.dims;
    int ndims = md.data.ndims;
    std::vector<int> v(dm, dm + ndims);
    return v;
  }


  // get format from PD
  static int getPDFmt(mem::primitive_desc pd) {
    return pd.desc().data.format;
  }

  mem::dims getUserDims() {
    return getMDDims(getUserMD());
  }




  */


/*
  void clearCvtFlag() {
    hasCvted_ = false;
  }

  // init conversion(reorder), will create internal buffer if needed
  // return true if need cvt.
  bool initCvt(mem::primitive_desc intlPD, int cvtType) {
    CHECK(cvtType == dnnUser2Intl || cvtType == dnnIntl2User
      || cvtType == dnnCvtNoNeed) << "please specify one type of conversion";
    CHECK(pUser_)
      << "call initUser before init internal layout and conversion";
    CHECK(nullptr == pIntl_)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    cvtType_ = cvtType;
    clearCvtFlag();
    if (cvtType == dnnCvtNoNeed || intlPD == getUserPD()) {
      cvtType_ = dnnCvtNoNeed;
      return false;
    } else {
      // allocate internal src memory from user
      this->pIntl_.reset(new mem(intlPD));
      // create a reorder
      if (cvtType == dnnUser2Intl) {
        this->pReorder.reset(new mkldnn::reorder(*pUser_, *pIntl_));
      } else {
        this->pReorder.reset(new mkldnn::reorder(*pIntl_, *pUser_));
      }
      return true;
    }
  }

  // init with dnnCvtNoNeed
  bool initCvt() {
    CHECK(pUser_)
      << "call initUser before init internal layout and conversion";
    CHECK(nullptr == pIntl_)
      << "internal memory should be empty before initCvt";
    pIntl_ = pUser_;
    cvtType_ = dnnCvtNoNeed;
    clearCvtFlag();
    return false;
  }

  bool needCvt() {
    CHECK(cvtType_) << "init conversion firstly";
    if (cvtType_ == dnnCvtNoNeed) {
      return false;
    } else {
      return nullptr == pReorder ? false : true;
    }
  }
  void submitCvt(std::vector<mkldnn::primitive> &net,
                      void* userData = NULL) {
    CHECK(cvtType_) << "init conversion firstly";
    // set user data handle, whether if need reorder or not
    if (userData) {
      if (userData != pUser_->get_data_handle()) {
        pUser_->set_data_handle(userData);
        // data changed, so donot care hasCvted_
      } else {  // user data do not change
        if (hasCvted_)  return;
      }
    } else {  // user data do not change
      if (hasCvted_)  return;
    }
    if (cvtType_ == dnnCvtNoNeed)
      return;
    CHECK(pReorder) << "init conversion firstly";
    net.push_back(*pReorder);
    hasCvted_ = true;
  }
  */
};

typedef std::shared_ptr<MkldnnBuffer> MkldnnBufferPtr;

}  // namespace paddle
