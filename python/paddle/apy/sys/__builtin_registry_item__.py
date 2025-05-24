# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import __builtin__


class RegistryEntry:

    def __init__(self):
        # list[RegistryObject]
        self.__values__ = __builtin__.MutableList()
        self.__child_register_item_name2value__ = (
            __builtin__.MutableOrderedDict()
        )

    def __get_valid_values__(self):
        # TODO(lixinqi)
        # def IsValid(elt):
        #     return elt.value != None
        #
        # return __builtin__.filter(IsValid, self.__values__)
        return self.__values__

    def __getattr__(self, attrname):
        def contains():
            return self.__child_register_item_name2value__.contains(attrname)

        def find():
            return self.__child_register_item_name2value__[attrname]

        def create():
            register_entry = RegistryEntry()
            self.__child_register_item_name2value__[attrname] = register_entry
            return register_entry

        return find() if contains() else create()

    # tag_name: str
    # order_value: int
    def __call__(self, tag_name, order_value):
        registry_obj = RegistryObject(tag_name, order_value)
        self.__values__.append(registry_obj)
        return RegisterItemDecorator(registry_obj)


class RegistryObject:

    def __init__(self, tag_name, order_value):
        self.tag_name = tag_name
        self.order_value = order_value
        self.value = None


class RegisterItemDecorator:

    def __init__(self, register_obj):
        self.register_obj = register_obj

    def __call__(self, value):
        self.register_obj.value = value
        return value
