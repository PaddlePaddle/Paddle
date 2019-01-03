
#include "debug_support.h"

namespace paddle {
    namespace platform {

        std::once_flag DebugSupport::init_flag_;
        std::unique_ptr<DebugSupport> debugSupport_(nullptr);

        DebugSupport *DebugSupport::GetInstance() {
          std::call_once(init_flag_,
                         [&]() { debugSupport_.reset(new DebugSupport()); });
          return debugSupport_.get();
        }

        std::string DebugSupport::getActiveOperator() {
          return infos[TOperaor];
        }

        void DebugSupport::setActiveOperator(std::string info){
          infos.at(TOperaor) = info;
        }

    }  // namespace platform
}  // namespace paddle
