from typing import List, Dict
from type_mapping import (input_types_map, optional_input_types_map,
                          attr_types_map, opmaker_attr_types_map,
                          output_type_map)
from type_mapping import (dense_input_types_map,
                          dense_optional_input_types_map,
                          dense_output_types_map, sr_input_types_map,
                          sr_optional_input_types_map, sr_output_types_map,
                          phi_attr_types_map)

def to_named_dict(items: List[Dict]) -> Dict[str, Dict]:
    named_dict = {}
    for item in items:
        if "name" not in item:
            raise KeyError(f"name not in {item}")
        name = item["name"]
        named_dict[name] = item
    return named_dict


# ------------------------------ attr -------------------------------------
def to_phi_attr_type(s):
    return phi_attr_types_map[s]


def to_op_attr_type(s):
    return opmaker_attr_types_map[s]


def to_paddle_attr_type(s):
    "Convert type tag for attributes in yaml to c++ types"
    return attr_types_map[s]


# ------------------------------ input ----------------------------------
def to_paddle_input_type(s, optional=False):
    "Convert type tag for inputs in yaml to c++ types"
    if optional:
        return optional_input_types_map[s]
    else:
        return input_types_map[s]


def to_dense_input_type(s, optional=False):
    "Convert types in yaml to dense tensor type in phi"
    if optional:
        return dense_input_types_map[s]
    else:
        return dense_optional_input_types_map[s]


# ------------------------------ output  ----------------------------------
def to_paddle_output_type(s):
    return output_type_map[s]


def to_dense_output_type(s):
    "Convert types in yaml to dense tensor type in phi"
    return dense_output_types_map[s]


def to_sr_output_type(s):
    "Convert types in yaml to selected rows type in phi"
    return sr_output_types_map[s]

# -------------- transform argument names from yaml to opmaker ------------
def to_opmaker_name(s):
    if s.endswith("_grad"):
        return 'GradVarName("")'.format(s.rstrip("_grad").capitalize())
    else:
        return to_pascal_case(s)
    
def to_pascal_case(s):
    words = s.split("_")
    return "".join([word.capitalize() for word in words])