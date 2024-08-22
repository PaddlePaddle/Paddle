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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

if TYPE_CHECKING:
    from .mutable_data import DataGetter, MutableData
    from .pycode_generator import PyCodeGen
    from .variables import VariableBase

    MutableDataT = TypeVar("MutableDataT", bound=MutableData)


class SideEffectsState(NamedTuple):
    data_id_to_proxy: dict[int, MutableData]
    proxy_variables: list[VariableBase]
    mutable_variables: list[VariableBase]
    proxy_versions: list[int]
    mutable_attrs: list[dict[str, Any]]


class SideEffects:
    def __init__(self):
        self.data_id_to_proxy: dict[int, MutableData] = {}
        self.proxy_variables: list[VariableBase] = []
        self.mutable_variables: list[VariableBase] = []

    def record_proxy_variable(self, variable: VariableBase):
        if variable not in self.proxy_variables:
            self.proxy_variables.append(variable)

    def record_mutable_variable(self, variable: VariableBase):
        if variable not in self.mutable_variables:
            self.mutable_variables.append(variable)

    def get_proxy(
        self,
        proxy_type: type[MutableDataT],
        data: Any,
        getter: DataGetter,
    ) -> MutableDataT:
        data_id = id(data)
        if data_id not in self.data_id_to_proxy:
            self.data_id_to_proxy[data_id] = proxy_type(data, getter)
        return self.data_id_to_proxy[data_id]  # type: ignore

    def get_state(self):
        return SideEffectsState(
            self.data_id_to_proxy.copy(),
            self.proxy_variables.copy(),
            self.mutable_variables.copy(),
            [proxy.version for proxy in self.data_id_to_proxy.values()],
            [
                {attr: getattr(var, attr)}
                for var in self.mutable_variables
                for attr in var.mutable_attrs
            ],
        )

    def restore_state(self, state: SideEffectsState):
        self.data_id_to_proxy = state.data_id_to_proxy
        self.proxy_variables = state.proxy_variables
        self.mutable_variables = state.mutable_variables
        # NOTE(SigureMo): We can use the `strict=True` option in zip after
        # Python 3.10.
        assert len(self.data_id_to_proxy.values()) == len(
            state.proxy_versions
        ), "proxy_versions length not match"
        assert len(self.mutable_variables) == len(
            state.mutable_attrs
        ), "mutable_attrs length not match"

        for proxy, version in zip(
            self.data_id_to_proxy.values(), state.proxy_versions
        ):
            proxy.rollback(version)

        for (variable, attr), attr_dict in zip(
            (
                (var, attr)
                for var in self.mutable_variables
                for attr in var.mutable_attrs
            ),
            (attr_dict for attr_dict in state.mutable_attrs),
        ):
            setattr(variable, attr, attr_dict[attr])


class SideEffectRestorer:
    def pre_gen(self, codegen: PyCodeGen):
        raise NotImplementedError

    def post_gen(self, codegen: PyCodeGen):
        raise NotImplementedError


class DictSideEffectRestorer(SideEffectRestorer):
    """
    old_dict.clear()
    old_dict.update(new_dict)
    """

    def __init__(self, var: VariableBase):
        super().__init__()
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        # Reference to the original dict.
        # load old_dict.update and new_dict to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_method("update")
        # Generate dict by each key-value pair.
        self.var.reconstruct(codegen, use_tracker=False)
        # load old_dict.clear to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_method("clear")

    def post_gen(self, codegen: PyCodeGen):
        # Call methods to apply side effects.
        codegen.gen_call_method(0)  # call clear
        codegen.gen_pop_top()
        codegen.gen_call_method(1)  # call update
        codegen.gen_pop_top()


class ListSideEffectRestorer(SideEffectRestorer):
    """
    old_list[:] = new_list
    """

    def __init__(self, var: VariableBase):
        super().__init__()
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        # Reference to the original list.
        # load new_list to stack.
        self.var.reconstruct(codegen, use_tracker=False)
        # load old_list[:] to stack.
        self.var.reconstruct(codegen)
        codegen.gen_load_const(None)
        codegen.gen_load_const(None)
        codegen.gen_build_slice(2)

    def post_gen(self, codegen: PyCodeGen):
        # Call STORE_SUBSCR to apply side effects.
        codegen.gen_store_subscr()


class GlobalSetSideEffectRestorer(SideEffectRestorer):
    """
    global_var = new_value
    """

    def __init__(self, name: str, var: VariableBase):
        super().__init__()
        self.name = name
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        self.var.reconstruct(codegen)

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_store_global(self.name)


class GlobalDelSideEffectRestorer(SideEffectRestorer):
    """
    del global_var
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def pre_gen(self, codegen: PyCodeGen):
        # do nothing
        ...

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_delete_global(self.name)


class ObjSetSideEffectRestorer(SideEffectRestorer):
    """
    obj.attr = new_value
    """

    def __init__(self, obj: VariableBase, name: str, var: VariableBase):
        super().__init__()
        self.obj = obj
        self.name = name
        self.var = var

    def pre_gen(self, codegen: PyCodeGen):
        # value
        self.var.reconstruct(codegen)
        # obj
        self.obj.reconstruct(codegen)

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_store_attr(self.name)


class ObjDelSideEffectRestorer(SideEffectRestorer):
    """
    del obj.attr
    """

    def __init__(self, obj: VariableBase, name: str):
        super().__init__()
        self.obj = obj
        self.name = name

    def pre_gen(self, codegen: PyCodeGen):
        self.obj.reconstruct(codegen)

    def post_gen(self, codegen: PyCodeGen):
        codegen.gen_delete_attr(self.name)
