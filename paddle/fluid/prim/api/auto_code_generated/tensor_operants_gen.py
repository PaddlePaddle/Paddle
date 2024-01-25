# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import yaml
from eager_gen import BaseAPI

indent = "  "

eager_header_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/common/macros.h"
#include "paddle/utils/test_macros.h"

"""

eager_header_start = """
namespace paddle {

namespace prim {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

class TEST_API EagerTensorOperants : public TensorOperantsBase {
 private:
  DISABLE_COPY_AND_ASSIGN(EagerTensorOperants);

 public:
  EagerTensorOperants() = default;

  Tensor add(const Tensor& x, const Scalar& y);

  Tensor subtract(const Tensor& x, const Scalar& y);

  Tensor multiply(const Tensor& x, const Scalar& y);

  Tensor divide(const Tensor& x, const Scalar& y);

  Tensor add(const Scalar& x, const Tensor& y);

  Tensor subtract(const Scalar& x, const Tensor& y);

  Tensor multiply(const Scalar& x, const Tensor& y);

  Tensor divide(const Scalar& x, const Tensor& y);

  Tensor pow(const Tensor& x, const Tensor& y);

  Tensor pow(const Tensor& x, const Scalar& y);

"""


eager_header_end = """};

}  // namespace prim
}  // namespace paddle

"""


eager_source_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#include "paddle/fluid/prim/utils/eager/eager_tensor_operants.h"

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

"""


eager_source_start = """
namespace paddle {

namespace prim {

Tensor EagerTensorOperants::add(const Tensor& x, const Scalar& y) {
  return ::add_ad_func(x, ::full_like_ad_func(x, y));
}

Tensor EagerTensorOperants::subtract(const Tensor& x, const Scalar& y) {
  return ::subtract_ad_func(x, ::full_like_ad_func(x, y));
}

Tensor EagerTensorOperants::multiply(const Tensor& x, const Scalar& y) {
  return ::scale_ad_func(x, y, 0.0f, true);
}

Tensor EagerTensorOperants::divide(const Tensor& x, const Scalar& y) {
  return ::divide_ad_func(x, ::full_like_ad_func(x, y));
}

Tensor EagerTensorOperants::add(const Scalar& x, const Tensor& y) {
  return ::add_ad_func(::full_like_ad_func(y, x), y);
}

Tensor EagerTensorOperants::subtract(const Scalar& x, const Tensor& y) {
  return ::subtract_ad_func(::full_like_ad_func(y, x), y);
}

Tensor EagerTensorOperants::multiply(const Scalar& x, const Tensor& y) {
  return ::scale_ad_func(y, x, 0.0f, true);
}

Tensor EagerTensorOperants::divide(const Scalar& x, const Tensor& y) {
  return ::divide_ad_func(::full_like_ad_func(y, x), y);
}

Tensor EagerTensorOperants::pow(const Tensor& x, const Tensor& y) {
  return ::elementwise_pow_ad_func(x, y);
}

Tensor EagerTensorOperants::pow(const Tensor& x, const Scalar& y) {
  return ::elementwise_pow_ad_func(x, ::full_like_ad_func(x, y));
}

"""


eager_source_end = """
}  // namespace prim
}  // namespace paddle

"""


static_header_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/common/macros.h"
#include "paddle/utils/test_macros.h"
"""

static_header_start = """
namespace paddle {

namespace prim {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

class TEST_API StaticTensorOperants : public TensorOperantsBase {
 private:
  DISABLE_COPY_AND_ASSIGN(StaticTensorOperants);

 public:
  StaticTensorOperants() = default;

  Tensor add(const Tensor& x, const Scalar& y);

  Tensor subtract(const Tensor& x, const Scalar& y);

  Tensor multiply(const Tensor& x, const Scalar& y);

  Tensor divide(const Tensor& x, const Scalar& y);

  Tensor add(const Scalar& x, const Tensor& y);

  Tensor subtract(const Scalar& x, const Tensor& y);

  Tensor multiply(const Scalar& x, const Tensor& y);

  Tensor divide(const Scalar& x, const Tensor& y);

  Tensor pow(const Tensor& x, const Tensor& y);

  Tensor pow(const Tensor& x, const Scalar& y);

"""


static_header_end = """};

}  // namespace prim
}  // namespace paddle

"""


static_source_include = """// Generated by paddle/fluid/prim/api/auto_code_generated/tensor_operants_gen.py

#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"

#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/fluid/prim/api/manual_prim/prim_manual_api.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

COMMON_DECLARE_bool(enable_pir_api);
COMMON_DECLARE_bool(enable_pir_in_executor);

"""


static_source_start = """
namespace paddle {

namespace prim {
using DescTensor = paddle::prim::DescTensor;
using LazyTensor = paddle::primitive::LazyTensor;

Tensor StaticTensorOperants::add(const Tensor& x, const Scalar& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::add<LazyTensor>(x, paddle::primitive::backend::full<LazyTensor>(x.shape(), y, x.dtype(), x.place()));
  } else {
    return paddle::prim::add<DescTensor>(x, paddle::prim::full<DescTensor>(x.shape(), y, x.dtype(), x.place()));
  }
}

Tensor StaticTensorOperants::subtract(const Tensor& x, const Scalar& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::subtract<LazyTensor>(x, paddle::primitive::backend::full<LazyTensor>(x.shape(), y, x.dtype(), x.place()));
  } else {
    return paddle::prim::subtract<DescTensor>(x, paddle::prim::full<DescTensor>(x.shape(), y, x.dtype(), x.place()));
  }
}

Tensor StaticTensorOperants::multiply(const Tensor& x, const Scalar& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::scale<LazyTensor>(x, y, 0.0f, true);
  } else {
    return paddle::prim::scale<DescTensor>(x, y, 0.0f, true);
  }
}

Tensor StaticTensorOperants::divide(const Tensor& x, const Scalar& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::divide<LazyTensor>(x, paddle::primitive::backend::full<LazyTensor>(x.shape(), y, x.dtype(), x.place()));
  } else {
    return paddle::prim::divide<DescTensor>(x, paddle::prim::full<DescTensor>(x.shape(), y, x.dtype(), x.place()));
  }
}

Tensor StaticTensorOperants::add(const Scalar& x, const Tensor& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::add<LazyTensor>(paddle::primitive::backend::full<LazyTensor>(y.shape(), x, y.dtype(), y.place()), y);
  } else {
    return paddle::prim::add<DescTensor>(paddle::prim::full<DescTensor>(y.shape(), x, y.dtype(), y.place()), y);
  }
}


Tensor StaticTensorOperants::subtract(const Scalar& x, const Tensor& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::subtract<LazyTensor>(paddle::primitive::backend::full<LazyTensor>(y.shape(), x, y.dtype(), y.place()), y);
  } else {
    return paddle::prim::subtract<DescTensor>(paddle::prim::full<DescTensor>(y.shape(), x, y.dtype(), y.place()), y);
  }
}

Tensor StaticTensorOperants::multiply(const Scalar& x, const Tensor& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::scale<LazyTensor>(y, x, 0.0f, true);
  } else {
    return paddle::prim::scale<DescTensor>(y, x, 0.0f, true);
  }
}

Tensor StaticTensorOperants::divide(const Scalar& x, const Tensor& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::divide<LazyTensor>(paddle::primitive::backend::full<LazyTensor>(y.shape(), x, y.dtype(), y.place()), y);
  } else {
    return paddle::prim::divide<DescTensor>(paddle::prim::full<DescTensor>(y.shape(), x, y.dtype(), y.place()), y);
  }
}

Tensor StaticTensorOperants::pow(const Tensor& x, const Tensor& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::elementwise_pow<LazyTensor>(x, y);
  } else {
    return paddle::prim::elementwise_pow<DescTensor>(x, y);
  }
}

Tensor StaticTensorOperants::pow(const Tensor& x, const Scalar& y) {
  if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {
    return paddle::primitive::backend::elementwise_pow<LazyTensor>(x, paddle::primitive::backend::full<LazyTensor>(x.shape(), y, x.dtype(), x.place()));
  } else {
    return paddle::prim::elementwise_pow<DescTensor>(x, paddle::prim::full<DescTensor>(x.shape(), y, x.dtype(), x.place()));
  }
}
"""


static_source_end = """
}  // namespace prim
}  // namespace paddle

"""


class PrimTensorAPI(BaseAPI):
    def __init__(self, api_item_yaml, prims=()):
        super().__init__(api_item_yaml, prims)

    def get_api_func_name(self):
        return self.api

    # def is_inplace(self):
    #     if self.inplace_map
    #         return True
    #     return False

    def gene_tensor_operants_declaration(self):
        api_func_name = self.get_api_func_name()

        if api_func_name[-1] != '_':
            return f"""{indent}{self.get_return_type()} {api_func_name}({self.get_declare_args()});\n
"""
        else:
            return f"""{indent}{self.get_return_type(inplace_flag=True)} {api_func_name}({self.get_declare_args(inplace_flag=True)});\n
"""

    def get_func_input_args(self, inplace_flag=False):
        input_args = []
        for name in self.inputs['names']:
            name = name.split('@')[0]
            if inplace_flag and name in self.inplace_map.values():
                input_args.append(name)
            else:
                input_args.append(name)
        return input_args

    def get_func_args(self, inplace_flag=False):
        ad_func_args = self.get_func_input_args(inplace_flag)
        for name in self.attrs['names']:
            default_value = ''
            if self.attrs['attr_info'][name][1] is not None:
                default_value = ' = ' + self.attrs['attr_info'][name][1]
            ad_func_args.append(name)

        ad_func_args_str = ", ".join(ad_func_args)
        return ad_func_args_str

    def gene_eager_tensor_func_call(self):
        api_func_name = self.get_api_func_name()

        dygraph_ad_func_name = '::' + api_func_name + '_ad_func'
        dygraph_ad_func_parameters = self.get_func_args()

        return (
            f"""return {dygraph_ad_func_name}({dygraph_ad_func_parameters});"""
        )

    def gene_eager_tensor_operants_implementation(self):
        api_func_name = self.get_api_func_name()
        # func decalaration
        if api_func_name[-1] != '_':
            api_code = f"""{self.get_return_type()} EagerTensorOperants::{api_func_name}({self.get_declare_args_nodefault()}) {{"""
        else:
            api_code = f"""{self.get_return_type(inplace_flag=True)} EagerTensorOperants::{api_func_name}({self.get_declare_args_nodefault(inplace_flag=True)}) {{"""

        # func code
        api_code += f"""
{indent}{self.gene_eager_tensor_func_call()}\n}}\n
"""
        return api_code

    def gene_static_tensor_func_call(self):
        api_func_name = self.get_api_func_name()
        backend_static_func_name = (
            'paddle::primitive::backend::' + api_func_name + '<LazyTensor>'
        )
        prim_static_func_name = (
            'paddle::prim::' + api_func_name + '<DescTensor>'
        )
        static_func_parameters = self.get_func_args()

        static_tensor_func_call = f"""if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {{
    return {backend_static_func_name}({static_func_parameters});
  }} else {{
    return {prim_static_func_name}({static_func_parameters});
  }}"""

        return static_tensor_func_call

    def gene_static_tensor_operants_implementation(self):
        api_code = ""
        indent = "  "
        api_func_name = self.get_api_func_name()
        # func decalaration
        if api_func_name[-1] != '_':
            api_code = f"""{self.get_return_type()} StaticTensorOperants::{api_func_name}({self.get_declare_args_nodefault()}) {{"""
        else:
            api_code = f"""{self.get_return_type(inplace_flag=True)} StaticTensorOperants::{api_func_name}({self.get_declare_args_nodefault(inplace_flag=True)}) {{"""

        function_call = self.gene_static_tensor_func_call()
        # func code
        api_code += f"""
{indent}{function_call}\n}}\n
"""

        return api_code


def generate_tensor_operants_api(
    api_yaml_path,
    eager_header_path,
    eager_source_path,
    static_header_path,
    static_source_path,
    api_prim_path,
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    eager_header_file = open(eager_header_path, 'w')
    eager_source_file = open(eager_source_path, 'w')
    static_header_file = open(static_header_path, 'w')
    static_source_file = open(static_source_path, 'w')

    eager_header_file.write(eager_header_include)
    eager_header_file.write(eager_header_start)
    eager_source_file.write(eager_source_include)
    eager_source_file.write(eager_source_start)
    static_header_file.write(static_header_include)
    static_header_file.write(static_header_start)
    static_source_file.write(static_source_include)
    static_source_file.write(static_source_start)

    with open(api_prim_path, 'rt') as f:
        api_prims = yaml.safe_load(f)

    for api in apis:
        eager_api = PrimTensorAPI(api, api_prims)
        if eager_api.is_prim_api:
            eager_header_file.write(
                eager_api.gene_tensor_operants_declaration()
            )
            eager_source_file.write(
                eager_api.gene_eager_tensor_operants_implementation()
            )
            static_header_file.write(
                eager_api.gene_tensor_operants_declaration()
            )
            static_source_file.write(
                eager_api.gene_static_tensor_operants_implementation()
            )

    eager_header_file.write(eager_header_end)
    eager_source_file.write(eager_source_end)
    static_header_file.write(static_header_end)
    static_source_file.write(static_source_end)

    eager_header_file.close()
    eager_source_file.close()
    static_header_file.close()
    static_source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--eager_tensor_operants_header_path',
        help='output of generated eager_tensor_operants header code file',
        default='paddle/fluid/prim/utils/eager/eager_tensor_operants.h.tmp',
    )

    parser.add_argument(
        '--eager_tensor_operants_source_path',
        help='output of generated eager_tensor_operants source code file',
        default='paddle/fluid/prim/utils/eager/eager_tensor_operants.cc.tmp',
    )

    parser.add_argument(
        '--static_tensor_operants_header_path',
        help='output of generated eager_tensor_operants header code file',
        default='paddle/fluid/prim/utils/static/static_tensor_operants.h.tmp',
    )

    parser.add_argument(
        '--static_tensor_operants_source_path',
        help='output of generated eager_tensor_operants source code file',
        default='paddle/fluid/prim/utils/static/static_tensor_operants.cc.tmp',
    )

    parser.add_argument(
        '--api_prim_yaml_path',
        help='Primitive API list yaml file.',
        default='paddle/fluid/prim/api/auto_code_generated/api.yaml',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    api_prim_yaml_path = options.api_prim_yaml_path
    eager_tensor_operants_header_path = (
        options.eager_tensor_operants_header_path
    )
    eager_tensor_operants_source_path = (
        options.eager_tensor_operants_source_path
    )
    static_tensor_operants_header_path = (
        options.static_tensor_operants_header_path
    )
    static_tensor_operants_source_path = (
        options.static_tensor_operants_source_path
    )

    generate_tensor_operants_api(
        api_yaml_path,
        eager_tensor_operants_header_path,
        eager_tensor_operants_source_path,
        static_tensor_operants_header_path,
        static_tensor_operants_source_path,
        api_prim_yaml_path,
    )


if __name__ == '__main__':
    main()
