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

#pragma once

#include <memory>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

class ExceptionHolder {
 public:
  void Catch(std::exception_ptr eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (platform::EOFException& exp) {
      Catch(exp);
    } catch (platform::EnforceNotMet& exp) {
      Catch(exp);
    } catch (std::exception& ex) {
      LOG(FATAL) << "std::exception caught, " << ex.what();
    } catch (...) {
      LOG(FATAL) << "Unknown exception caught";
    }
  }

  bool IsCaught() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exception_.get() != nullptr;
  }

  void ReThrow() {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type_) {
      case kNone:
        break;
      case kEnforceNotMet: {
        auto e = *static_cast<platform::EnforceNotMet*>(exception_.get());
        throw e;
      }
      case kEOF: {
        auto e = *static_cast<platform::EOFException*>(exception_.get());
        throw e;
      }
    }
    ClearImpl();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    ClearImpl();
  }

  std::string Type() {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type_) {
      case kNone:
        return "None";
      case kEnforceNotMet: {
        return "EnforceNotMet";
      }
      case kEOF: {
        return "EOF";
      }
    }
    return "unknown";
  }

 private:
  void ClearImpl() {
    exception_.reset();
    type_ = kNone;
  }

  void Catch(const platform::EnforceNotMet& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    exception_.reset(new platform::EnforceNotMet(exp));
    type_ = kEnforceNotMet;
  }

  void Catch(const platform::EOFException& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    // EOFException will not cover up existing EnforceNotMet.
    if (exception_.get() == nullptr) {
      exception_.reset(new platform::EOFException(exp));
      type_ = kEOF;
    }
  }

  enum ExceptionType { kNone, kEnforceNotMet, kEOF };
  ExceptionType type_{kNone};

  std::unique_ptr<std::exception> exception_;
  mutable std::mutex mu_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
