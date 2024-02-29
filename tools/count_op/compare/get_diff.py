import glob
import os
import re
import yaml

import sys
sys.path.append(
    r'/home/aistudio/test/Paddle/tools/count_op/op_compat'
)

import op_compat_info

def old_attr_type_to_new(old_attr_type):
    attr_type_dict ={
        "INT" : "int",
        "FLOAT" : "float",
        "STRING" : "str",
        "INTS" : "int[]",
        "STRINGS" : "str[]",
        "BOOLEAN" : "bool",
        "BLOCK" : "BLOCK",
        "LONG" : "int64_t",
        "BLOCKS" : "BLOCKS",
        "LONGS" : "int64_t[]",
        "FLOAT64S" : "FLAOT64S",
        "FLOAT64" : "double",
        "VAR" : "VAR",
        "VARS" : "VARS",
        "SCALAR" : "Scalar",
        "SCALARS" : "Scalar[]",
        "FLOATS" : "float[]",
        "BOOLEANS" : "BOOLEANS",
    }
    return attr_type_dict[old_attr_type]



def load_ops_yaml(yaml_path):
    parsed_ops = []
    with open(yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
    for op in ops:
        op_name = op["op"]
        inputs_list = []
        attrs_list = []
        outputs_list = []
        optional_list = []
        inplace_list = []
        
        inputs_list_str = op["inputs"]
        inputs_split_result = inputs_list_str[1:-1].split(', ')
        for i in inputs_split_result:
            if len(i) != 0:
                inputs_list.append(i.split(' '))

        attrs_list_str = op["attrs"]
        attrs_split_result = attrs_list_str[1:-1].split(', ')
        for i in attrs_split_result:
            if len(i) != 0:
                attrs_list.append(i.split(' '))
        
        outputs_list_str = op["outputs"]
        outputs_split_result = outputs_list_str.split(', ')
        for i in outputs_split_result:
            if len(i) != 0:
                outputs_list.append(i[:-1].split('('))
            
        optional_list_str=""
        try:
            optional_list_str = op["optionals"]
        except KeyError:
            pass

        if len(optional_list_str) != 0:
            optional_list = optional_list_str.split(',')

        inplace_list_str = ""
        try:
            inplace_list_str = op["inplaces"]
        except KeyError:
            pass
        
        inplace_split_result = []
        if len(inplace_list_str) != 0:
            inplace_split_result = inplace_list_str.split(',')
        for i in inplace_split_result:
            inplace_list.append(i.split('->'))

        input_types = []
        input_names = [] 
        attr_types = []
        attr_names = []
        output_types = []
        output_names = []
        optional_names = optional_list
        inplaces = inplace_list

        for i in inputs_list:
            input_types.append(i[0])
            input_names.append(i[1])

        for i in attrs_list:
            attr_types.append(i[0])
            attr_names.append(i[1])

        for i in outputs_list:
            output_types.append(i[0])
            output_names.append(i[1])



        parsed_ops.append({"op" : op_name, "input_types" : input_types,"input_names" : input_names,
                            "attr_types" : attr_types, "attr_names" : attr_names,
                            "output_types" : output_types,"output_names" : output_names, 
                            "optionals" : optional_names,"inplaces" : inplaces})
     
    return parsed_ops


def get_op_arg_mapping_list(op_name):
    op_arg_name_mappings = op_compat_info.op_arg_name_mappings
    for op_arg_name_mapping in op_arg_name_mappings:
        if op_name in op_arg_name_mapping:
            return op_arg_name_mapping[op_name]
        if op_name[-1] == "_" and op_name[:-1] in op_arg_name_mapping:
            return op_arg_name_mapping[op_name[:-1]]
    return None

def get_op_name_mapping(old_ir_name):
    op_name_mappings = op_compat_info.op_name_mappings
    for op_name_mapping in op_name_mappings:
        if old_ir_name in op_name_mapping:
            return op_name_mapping[old_ir_name]
    return None


def convert_old_ir_to_pir_form(parsed_old_ir_ops):

    op_name_mappings = op_compat_info.op_name_mappings
    op_arg_name_mappings = op_compat_info.op_arg_name_mappings
    
    for op in parsed_old_ir_ops:
        attr_types = op["attr_types"]
        for i in range(len(attr_types)):
            attr_types[i] = old_attr_type_to_new(attr_types[i])

        op["attr_types"] = attr_types
        

    for op in parsed_old_ir_ops:
       op_name = op["op"]
       arg_mapping_list = get_op_arg_mapping_list(op_name)
       if arg_mapping_list is not None:
           for arg_mapping in arg_mapping_list:
                for i in range(len(op["input_names"])):
                   if op["input_names"][i] in arg_mapping:
                        op["input_names"][i] = arg_mapping[op["input_names"][i]]
                for i in range(len(op["attr_names"])):
                    if op["attr_names"][i] in arg_mapping:
                        op["attr_names"][i] = arg_mapping[op["attr_names"][i]]
                for i in range(len(op["output_names"])):
                    if op["output_names"][i] in arg_mapping:
                        op["output_names"][i] = arg_mapping[op["output_names"][i]]
                for i in range(len(op["optionals"])):
                    if op["optionals"][i] in arg_mapping:
                        op["optionals"][i] = arg_mapping[op["optionals"][i]]
                for i in range(len(op["inplaces"])):
                    if op["inplaces"][i][0] in arg_mapping:
                        op["inplaces"][i][0] = arg_mapping[op["inplaces"][i][0]]
                    if op["inplaces"][i][1] in arg_mapping:
                        op["inplaces"][i][1] = arg_mapping[op["inplaces"][i][1]]
                pir_op_name = get_op_name_mapping(op["op"])

       pir_op_name = get_op_name_mapping(op["op"])
       if pir_op_name is not None:
            op["op"] = pir_op_name
            print(op["op"])

def convert_opinfo_to_dict(
    op_name,
    input_types,
    input_names,
    attr_types,
    attr_names,
    output_types,
    output_names,
    optional_names,
    inplaces,
):
    inputs = ""
    attrs = ""
    outputs = ""
    optionals = ""

    for i in range(len(input_names)):
        if i < len(input_names) - 1:
            inputs = (
                inputs + str(input_types[i]) + " " + str(input_names[i]) + ", "
            )
        else:
            inputs = inputs + str(input_types[i]) + " " + str(input_names[i])

    for i in range(len(attr_names)):
        if i < len(attr_names) - 1:
            attrs = attrs + str(attr_types[i]) + " " + str(attr_names[i]) + ", "
        else:
            attrs = attrs + str(attr_types[i]) + " " + str(attr_names[i])

    for i in range(len(output_names)):
        if i < len(output_names) - 1:
            outputs = (
                outputs
                + str(output_types[i])
                +"(" + str(output_names[i]) + ")"
                + ", "
            )
        else:
            outputs = (
                outputs + str(output_types[i])  +"(" + str(output_names[i]) + ")"
            )

    for i in range(len(optional_names)):
        if i < len(optional_names) - 1:
           optionals = optionals + optional_names[i] + ","
        else:
           optionals = optionals + optional_names[i] 
        

    inputs = "(" + inputs + ")"
    attrs = "(" + attrs + ")"
    outputs = outputs
    optionals = optionals
    inplaces = inplaces

    if len(optionals) == 0 and len(inplaces) == 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs}
    if len(optionals) > 0 and len(inplaces) == 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"optionals" : optionals}
    if len(optionals) == 0 and len(inplaces) > 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"inpalces" : inplaces}
    if  len(optionals) > 0 and len(inplaces) > 0:
        data = {"op": op_name, "inputs": inputs, "attrs": attrs, "outputs": outputs,"optionals" : optionals,"inpalces" : inplaces}
    return data

def applied_compat_info(old_ir_yaml_path,save_path):
    old_ir_ops = load_ops_yaml(old_ir_yaml_path)
    convert_old_ir_to_pir_form(old_ir_ops)
    save_yaml_ops = []
    for op in old_ir_ops:
        inplaces = op["inplaces"]
        inplaces_str = ""
        for i in inplaces:
            inplaces_str = inplaces_str + i[0] + "->" + i[1] + ","
        op["inplaces"] = inplaces_str[:-1]
        data = convert_opinfo_to_dict(op["op"],op["input_types"],op["input_names"],op["attr_types"],op["attr_names"],op["output_types"],
                                        op["output_names"],op["optionals"],op["inplaces"])
        save_yaml_ops.append(data)
    with open(save_path, 'w') as file:
        for op_info in save_yaml_ops:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")

def convert_inplaces_list_to_str(inplaces):
    inplaces_str = ""
    for i in inplaces:
        inplaces_str = inplaces_str + i[0] + "->" + i[1] + ","
    return inplaces_str[:-1]

def get_exits_difference(old_ir_yaml_path,new_ir_yaml_path,save_paths):
    old_ir_ops = load_ops_yaml(old_ir_yaml_path)
    new_ir_ops = load_ops_yaml(new_ir_yaml_path)

    new_ir_op_set = set()
    for op in new_ir_ops:
        if op["op"] in new_ir_op_set:
            print(op["op"])
        new_ir_op_set.add(op["op"])
    
    old_ir_op_set = set()
    for op in old_ir_ops:
        old_ir_op_set.add(op["op"])
    

    in_old_not_in_pir = set()
    for op in old_ir_op_set:
        if op not in new_ir_op_set:
            in_old_not_in_pir.add(op)
       
    in_new_not_in_old = set()
    for op in new_ir_op_set:
        if op not in old_ir_op_set:
            in_new_not_in_old.add(op)

    in_old_not_in_pir_ops = []
    for op in old_ir_ops:
        if op["op"] in in_old_not_in_pir:
            op["inplaces"] = convert_inplaces_list_to_str(op["inplaces"])
            data = convert_opinfo_to_dict(op["op"],op["input_types"],op["input_names"],op["attr_types"],op["attr_names"],op["output_types"],
                                        op["output_names"],op["optionals"],op["inplaces"])
            in_old_not_in_pir_ops.append(data)
    
    in_new_not_in_old_ops = []
    for op in new_ir_ops:
        if op["op"] in in_new_not_in_old:
            op["inplaces"] = convert_inplaces_list_to_str(op["inplaces"])
            data = convert_opinfo_to_dict(op["op"],op["input_types"],op["input_names"],op["attr_types"],op["attr_names"],op["output_types"],
                                        op["output_names"],op["optionals"],op["inplaces"])
            in_new_not_in_old_ops.append(data)

    in_old_in_new_new_ir_ops =[]
    for op in new_ir_ops:
        if op["op"] in new_ir_op_set and op["op"] in old_ir_op_set:
            op["inplaces"] = convert_inplaces_list_to_str(op["inplaces"])
            data = convert_opinfo_to_dict(op["op"],op["input_types"],op["input_names"],op["attr_types"],op["attr_names"],op["output_types"],
                                        op["output_names"],op["optionals"],op["inplaces"])
            in_old_in_new_new_ir_ops.append(data)

    in_old_in_new_old_ir_ops =[]
    for op in old_ir_ops:
        if op["op"] in new_ir_op_set and op["op"] in old_ir_op_set:
            op["inplaces"] = convert_inplaces_list_to_str(op["inplaces"])
            data = convert_opinfo_to_dict(op["op"],op["input_types"],op["input_names"],op["attr_types"],op["attr_names"],op["output_types"],
                                        op["output_names"],op["optionals"],op["inplaces"])
            in_old_in_new_old_ir_ops.append(data)

    with open(save_paths[0], 'w') as file:
        for op_info in in_old_not_in_pir_ops:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")

    with open(save_paths[1], 'w') as file:
        for op_info in in_new_not_in_old_ops:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")

    with open(save_paths[2], 'w') as file:
        for op_info in in_old_in_new_new_ir_ops:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")

    with open(save_paths[3], 'w') as file:
        for op_info in in_old_in_new_old_ir_ops:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")

            
if __name__ == '__main__':
    old_ir_ops_path = r"/home/aistudio/test/Paddle/tools/count_op/fluid/old_ir_ops(op_compat_applied).yaml"
    new_ir_ops_path = r"/home/aistudio/test/Paddle/tools/count_op/pir_ops/pir_ops.yaml"
    save_paths = [r"/home/aistudio/test/Paddle/tools/count_op/compare/in_old_not_in_new_ops.yaml",
                 r"/home/aistudio/test/Paddle/tools/count_op/compare/in_new_not_in_old_ops.yaml",
                 r"/home/aistudio/test/Paddle/tools/count_op/compare/in_old_in_new_new_ir_ops.yaml",
                 r"/home/aistudio/test/Paddle/tools/count_op/compare/in_old_in_new_old_ir_ops.yaml",
    ]
    get_exits_difference(old_ir_ops_path,new_ir_ops_path,save_paths)

    #applied_compat_info(r"/home/aistudio/test/Paddle/tools/count_op/fluid/old_ir_ops.yaml",r"/home/aistudio/test/Paddle/tools/count_op/fluid/old_ir_ops(op_compat_applied).yaml")


    

    
   


        



        



