import re
import yaml
from copy import copy
from typing import Dict, Any, List, Tuple
from tests import is_attr, is_input, is_output


def parse_arg(api_name: str, s: str) -> Dict[str, str]:
    """parse an argument in following formats:
    1. typename name
    2. typename name = default_value
    """
    typename, rest = [item.strip() for item in s.split(" ", 1)]
    assert len(typename) > 0, f"The arg typename should not be empty. Please check the args of {api_name} in yaml."
    
    assert rest.count("=") <= 1, f"There is more than 1 = in an arg in {api_name}"
    if rest.count("=") == 1:
        name, default_value = [item.strip() for item in rest.split("=", 1)]
        assert len(name) > 0, f"The arg name should not be empty. Please check the args of {api_name} in yaml."
        assert len(default_value) > 0, f"The default value should not be empty. Please check the args of {api_name} in yaml."
        return {"typename": typename, "name": name, "default_value": default_value}
    else:
        name = rest.strip()
        assert len(name) > 0, f"The arg name should not be empty. Please check the args of {api_name} in yaml."
        return {"typename": typename, "name": name}


def parse_input_and_attr(
    api_name: str,
    arguments: str) -> Tuple[List, List, Dict, Dict]:
    args_str = arguments.strip()
    assert args_str.startswith('(') and args_str.endswith(')'), \
        (f"Args declaration should start with '(' and end with ')', "
         f"please check the args of {api_name} in yaml.")
    args_str = args_str[1:-1]
    args = parse_plain_list(args_str)

    inputs = []
    attrs = []

    met_attr_with_default_value = False
    
    for arg in args:
        item = parse_arg(api_name, arg)
        typename = item["typename"]
        name = item["name"]
        if is_input(typename):
            assert len(attrs) == 0, \
                (f"The input Tensor should appear before attributes. "
                f"please check the position of {api_name}:input({name}) "
                f"in yaml.")
            inputs.append(item)
        elif is_attr(typename):
            if met_attr_with_default_value:
                assert "default_value" in item, f"{api_name}: Arguments with default value should not precede those without default value"
            elif "default_value" in item:
                met_attr_with_default_value = True
            attrs.append(item)
        else:
            raise KeyError(f"{api_name}: Invalid argument type {typename}.")
    return inputs, attrs


def parse_output(api_name: str, s: str) -> Dict[str, str]:
    """parse an output, typename or typename(name)."""
    if re.search(r'\([a-zA-Z0-9_@]*\)', s):
        match = re.search(
            r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*\((?P<name>[a-zA-Z0-9_@]+)\)", s)
        typename = match.group("out_type")
        name = match.group("name")
    else:
        typename = s.strip()
        name = "out"

    assert is_output(typename), \
        (f"Invalid output type: {typename} in api: {api_name}."
            f"Supported types are Tensor and Tensor[]")
    return {"typename": typename, "name": name}


def parse_outputs(api_name: str, outputs: str) -> List[Dict]:
    outputs = parse_plain_list(outputs, sep=",")
    output_items = []
    for output in outputs:
        output_items.append(parse_output(api_name, output))
    return output_items


def parse_infer_meta(infer_meta: Dict[str, Any]) -> Dict[str, Any]:
    infer_meta = copy(infer_meta)  # to prevent mutating the input
    if "param" not in infer_meta:
        infer_meta["param"] = None
    return infer_meta


def parse_candidates(s: str) -> Dict[str, Any]:
    "parse candidates joined by either '>'(ordered) or ','(unordered)"
    delimiter = ">" if ">" in s else ","
    ordered = delimiter == ">"
    candidates = parse_plain_list(s, delimiter)
    return {"ordered": ordered, "candidates": candidates}


def parse_plain_list(s: str, sep=",") -> List[str]:
    items = [item.strip() for item in s.strip().split(sep)]
    return items


def parse_kernel(api_name: str, kernel_config: Dict[str, Any]) -> Dict[str, Any]:
    # kernel :
    #    func : [], Kernel functions (example: scale, scale_sr)
    #    param : [], Input params of kernel
    #    backend : str, the names of param to choose the kernel backend, default is None
    #    layout : str, the names of param to choose the kernel layout, default is None
    #    data_type : str, the names of param to choose the kernel data_type, default is None
    kernel = {
        'func': None,  # up to 2 function names
        'param': None,
        'backend': None,
        'layout': None,
        'data_type': None
    }
    kernel['func'] = parse_plain_list(kernel_config['func'])
    if 'param' in kernel_config:
        kernel['param'] = kernel_config['param']
        
    if 'backend' in kernel_config:
        kernel['backend'] = parse_candidates(kernel_config["backend"])

    if 'layout' in kernel_config:
        kernel['layout'] = parse_candidates(kernel_config["layout"])

    if 'data_type' in kernel_config:
        kernel['data_type'] = parse_candidates(kernel_config["data_type"])
    return kernel


def parse_inplace(api_name: str, inplace_cfg: str) -> Dict[str, str]:
    inplace_map = {}
    inplace_cfg = inplace_cfg.lstrip("(").rstrip(")")
    pairs = parse_plain_list(inplace_cfg)
    for pair in pairs:
        in_name, out_name = parse_plain_list(pair, sep="->")
        inplace_map[out_name] = in_name
    return inplace_map


def parse_invoke(api_name: str, invoke_config: str) -> Dict[str, Any]:
    invoke_config = invoke_config.strip()
    func, rest = invoke_config.split("(", 1)
    func = func.strip()
    args = rest.rstrip(")").strip()
    invocation = {"func": func, "args": args}
    return invocation


def extract_type_and_name(records: List[Dict]) -> List[Dict]:
    extracted = [{
        "name": item["name"],
        "typename": item["typename"]
    } for item in records]
    return extracted


def parse_forward(api_name: str, forward_config: str) -> Dict[str, Any]:
    # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
    result = re.search(
        r"(?P<api>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
        forward_config)
    api = result.group("api")
    outputs = parse_outputs(api_name, result.group("outputs"))
    outputs = extract_type_and_name(outputs)

    inputs, attrs = parse_input_and_attr(api_name, result.group("args"))
    inputs = extract_type_and_name(inputs)
    attrs = extract_type_and_name(attrs)
    forward_cfg = {
        "name": api,
        "inputs": inputs,
        "attrs": attrs,
        "outputs": outputs
    }
    return forward_cfg



def parse_api_entry(api_entry: Dict[str, Any], name_field="api"):
    api_name = api_entry[name_field]
    inputs, attrs = parse_input_and_attr(api_name, api_entry["args"])
    outputs = parse_outputs(api_name, api_entry["output"])

    input_names = [item["name"] for item in inputs]
    attr_names = [item["name"] for item in attrs]
    
    # add optional tag for every inputs
    for input in inputs:
        input["optional"] = False
    if "optional" in api_entry:
        optional_args = parse_plain_list(api_entry["optional"])
        for name in optional_args:
            assert name in input_names
        for input in inputs:
            if input["name"] in optional_args:
                input["optional"] = True
    
    api = {
        "name": api_name,
        "inputs": inputs,
        "attrs": attrs,
        "outputs": outputs
    }

    # invokes another api?
    is_base_api = "invoke" not in api_entry

    if is_base_api:
        # kernel
        kernel = parse_kernel(api_name, api_entry["kernel"])
        if kernel["param"] is None:
            kernel["param"] = input_names + attr_names

        # infer meta
        infer_meta = parse_infer_meta(api_entry["infer_meta"])
        if infer_meta["param"] is None:
            infer_meta["param"] = copy(kernel["param"])

        # inplace
        if "inplace" in api_entry:
            inplace_pairs = parse_inplace(api_name, api_entry["inplace"])
        else:
            inplace_pairs = None
        api.update({
            "infer_meta": infer_meta,
            "kernel": kernel,
            "inplace": inplace_pairs
        })
    else:
        # invoke
        invoke = parse_invoke(api_name, api_entry["invoke"])
        api["invoke"] = invoke

    # forward or backward 
    is_backward_api = name_field == "backward_api"
    if not is_backward_api:
        if "backward" in api_entry:
            backward = api_entry["backward"]
        else:
            backward = None
        api["backward"] = backward
    else:
        if "forward" in api_entry:
            forward = parse_forward(api_name, api_entry["forward"])
        else:
            forward = None
        api["forward"] = forward
    return api