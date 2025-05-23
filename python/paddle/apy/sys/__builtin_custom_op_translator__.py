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

import ap
import axpr


def LambdaJsonStrToPyCode(lambda_json_str, lambda_name):
    axpr_value = axpr.axpr_json_str_to_axpr(lambda_json_str)
    axpr_atomic = axpr_value.match(axpr_atomic=lambda x: x)
    translator = ApyAxprToPyAxprNativeTranslator(
        CallTranslator(BuiltinCallTranslator()),
    )
    axpr_atomic = translator.translate(axpr_atomic)
    function = axpr.axpr_atomic_to_function(axpr_atomic)
    return function.to_string(lambda_name)


class Env:

    def __init__(self):
        self.var2producer_name = ap.MutableOrderedDict()

    def set_var2producer_name(self, var, producer_name):
        self.var2producer_name[var] = producer_name

    def producer_name_eq(self, var, producer_name):
        return (
            self.var2producer_name[var] == producer_name
            if self.var2producer_name.contains(var)
            else False
        )


class ApyAxprToPyAxprNativeTranslator:

    def __init__(self, call_translator):
        self.call_translator = call_translator
        self.env = Env()

    def translate(self, axpr_atomic):
        ret = self.translate_axpr_atomic(axpr_atomic)
        return ret

    def translate_axpr_atomic(self, axpr_atomic):
        return axpr_atomic.match(
            axpr_lambda=self.translate_axpr_lambda, _=lambda: axpr_atomic
        )

    def translate_axpr_lambda(self, lambda_args, lambda_body):
        return axpr.axpr_lambda(
            lambda_args, self.translate_axpr_obj(lambda_body)
        )

    def translate_axpr_obj(self, axpr_obj):
        return axpr_obj.match(
            axpr_atomic=lambda x: axpr_obj, axpr_call=self.translate_axpr_call
        )

    def translate_axpr_call(self, outer_func, inner_func, call_args):
        pair = self.translate_inner_func_call(
            inner_func, ap.map(self.translate_func_args_axpr_atomic, call_args)
        )
        self._set_var2producer_name(outer_func, inner_func)
        return axpr.axpr_call(
            self.translate_axpr_atomic(outer_func), pair[0], pair[1]
        )

    def translate_func_args_axpr_atomic(self, axpr_atomic):
        return axpr_atomic.match(
            axpr_lambda=self.translate_func_args_axpr_lambda,
            _=lambda: axpr_atomic,
        )

    def translate_func_args_axpr_lambda(self, lambda_args, lambda_body):
        new_translator = ApyAxprToPyAxprNativeTranslator(self.call_translator)
        return axpr.axpr_lambda(
            lambda_args, new_translator.translate_axpr_obj(lambda_body)
        )

    def _set_var2producer_name(self, outer_func, inner_func):

        def update(var_name, producer_name):
            self.env.set_var2producer_name(
                axpr.axpr_symbol(var_name), producer_name
            )

        def set_producer_name(var_name):
            inner_func.match(
                axpr_symbol=lambda producer_name: update(
                    var_name, producer_name
                ),
                _=ap.do_nothing,
            )

        outer_func.match(
            axpr_lambda=lambda arg_names, body: set_producer_name(arg_names[0]),
            _=ap.do_nothing,
        )

    def translate_inner_func_call(self, inner_func, args):
        ret = inner_func.match(
            axpr_symbol=lambda x: self.translate_inner_func_axpr_symbol(
                x, args
            ),
            _=lambda: self.translate_axpr_atomic(inner_func),
        )
        return ret

    def translate_inner_func_axpr_symbol(self, symbol_name, args):
        return self.call_translator.translate(self.env, symbol_name, args)


class CallTranslator:

    def __init__(self, builtin_call_translator):
        self.builtin_call_translator = builtin_call_translator

    def translate(self, env, func_name, args):
        translatable = lambda: self.translatable(func_name, args)
        translate = lambda: self.builtin_call_translator.translate(
            env, func_name, args
        )
        return (
            translate()
            if translatable()
            else [
                axpr.axpr_symbol(func_name),
                self._get_default_args(env, args),
            ]
        )

    def translatable(self, func_name, args):
        func_translatable = lambda: self.builtin_call_translator.translatable(
            func_name
        )
        args_translatable = lambda: self._args_translatable(args)
        return func_translatable() and args_translatable()

    def _get_default_args(self, env, args):
        def is_packed_args():
            return len(args) == 1 and env.producer_name_eq(
                args[0], '__builtin_PackedArgs__'
            )

        return self._get_unpacked_args(args) if is_packed_args() else args

    def _get_unpacked_args(self, args):
        return self.builtin_call_translator.get_unpacked_args(args)

    def _args_translatable(self, args):
        def is_lambda(arg):
            return arg.get_type_name() == 'axpr_lambda'

        return len(ap.filter(is_lambda, args)) == 0


class BuiltinCallTranslator:

    def __init__(self):
        self.translator_factory = self._get_translator_factory()

    def translatable(self, func_name):
        return self.translator_factory.contains(func_name)

    def translate(self, env, func_name, args):
        return self.translator_factory[func_name](env, func_name, args)

    def get_unpacked_args(self, args):
        var_name = self._axpr_atomic_to_str(args[0])
        return [
            axpr.axpr_symbol(f"*{var_name}[0]"),
            axpr.axpr_symbol(f"**{var_name}[1]"),
        ]

    def _axpr_atomic_to_str(self, axpr_atomic):
        return axpr_atomic.match(
            axpr_none=lambda: "None",
            axpr_bool=lambda x: f"{x}",
            axpr_int=lambda x: f"{x}",
            axpr_float=lambda x: f"{x}",
            axpr_str=lambda x: ap.quoted(x),
            axpr_symbol=lambda x: x,
        )

    def _translate_import(self, env, func_name, args):
        return [axpr.axpr_symbol("__builtin__import"), args]

    def _translate_raise(self, env, func_name, args):
        return [axpr.axpr_symbol("__builtin__raise"), args]

    def _translator_getattr(self, env, func_name, args):
        obj = args[0].match(axpr_symbol=lambda x: x)
        field = args[1].match(axpr_str=lambda x: x)
        return [
            axpr.axpr_symbol("__builtin_identity__"),
            [axpr.axpr_symbol(f"{obj}.{field}")],
        ]

    def _translator_getitem(self, env, func_name, args):
        obj = args[0].match(axpr_symbol=lambda x: x)
        key = self._axpr_atomic_to_str(args[1])
        return [
            axpr.axpr_symbol("__builtin_identity__"),
            [axpr.axpr_symbol(f"{obj}[{key}]")],
        ]

    def _translator_list(self, env, func_name, args):
        def translator_list_elt(elt):
            axpr_atomic_str = self._axpr_atomic_to_str(elt)
            is_star = env.producer_name_eq(elt, "__builtin_starred__")
            return f"*{axpr_atomic_str}" if is_star else axpr_atomic_str

        args_str = ", ".join(ap.map(translator_list_elt, args))
        opt_comma = "," if len(args) == 1 else ""
        return [
            axpr.axpr_symbol("__builtin_identity__"),
            [axpr.axpr_symbol(f"({args_str}{opt_comma})")],
        ]

    def _translator_starred(self, env, func_name, args):
        return [axpr.axpr_symbol("__builtin_identity__"), args]

    def _translator_packed_args(self, env, func_name, args):
        pos_args_str = self._axpr_atomic_to_str(args[0])
        keyword_args_str = self._axpr_atomic_to_str(args[1])
        l_brace = '{'
        r_brace = '}'
        return [
            axpr.axpr_symbol("__builtin_identity__"),
            [
                axpr.axpr_symbol(
                    f"({pos_args_str}, {l_brace}k:v for k, v in {keyword_args_str}{r_brace})"
                )
            ],
        ]

    def _get_translator_factory(self):
        return ap.OrderedDict(
            [
                ["import", self._translate_import],
                ["raise", self._translate_raise],
                ["__builtin_getattr__", self._translator_getattr],
                ["__builtin_getitem__", self._translator_getitem],
                ["__builtin_list__", self._translator_list],
                ["__builtin_starred__", self._translator_starred],
                ["__builtin_PackedArgs__", self._translator_packed_args],
            ]
        )
