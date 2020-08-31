#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# Utility script to print the python tag + the abi tag for a Python
# See PEP 425 for exactly what these are, but an example would be:
#   cp27-cp27mu

from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

print("{0}{1}-{2}".format(get_abbr_impl(), get_impl_ver(), get_abi_tag()))
