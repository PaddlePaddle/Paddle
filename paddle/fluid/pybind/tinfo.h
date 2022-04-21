//
// Created by linxu on 2022/4/21.
//

#ifndef PADDLE_TINFO_H
#define PADDLE_TINFO_H

#include "paddle/fluid/framework/type_info.h"

namespace paddle {
namespace pybind {

void BindFinfoVarDsec(pybind11::module *m);
void BindIinfoVarDsec(pybind11::module *m);

}
}


#endif  // PADDLE_TINFO_H
