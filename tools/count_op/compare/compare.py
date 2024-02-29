import glob
import os
import re
import yaml


def load_ops_yaml(yaml_path):
    parsed_ops = dict()
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
        inplace_list.sort(key=lambda x: x[1])

        attrs_list_str = op["attrs"]
        attrs_split_result = attrs_list_str[1:-1].split(', ')
        for i in attrs_split_result:
            if len(i) != 0:
                attrs_list.append(i.split(' '))
        attrs_list.sort(key=lambda x: x[1])
        
        outputs_list_str = op["outputs"]
        outputs_split_result = outputs_list_str.split(', ')
        for i in outputs_split_result:
            if len(i) != 0:
                outputs_list.append(i[:-1].split('('))
        outputs_list.sort(key=lambda x: x[1])
            
        optional_list_str=""
        try:
            optional_list_str = op["optionals"]
        except KeyError:
            pass

        if len(optional_list_str) != 0:
            optional_list = optional_list_str.split(',')
        optional_list.sort()

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

        inplace_list.sort(key=lambda x: x[1])

        parsed_ops[op_name] = {"op" : op_name, "inputs" : inputs_list, "attrs" : attrs_list, "outputs" : outputs_list, "optionals" : optional_list, "inplaces" : inplace_list}
    return parsed_ops



def compare(new_ir_yaml_path,old_ir_yaml_path):
    new_ir_ops = load_ops_yaml(new_ir_yaml_path)
    old_ir_ops = load_ops_yaml(old_ir_yaml_path)

    op_compare_result = []
    for op_name in old_ir_ops:
        print(op_name)
        old_op_inputs = old_ir_ops[op_name]["inputs"]
        old_op_attrs = old_ir_ops[op_name]["attrs"]
        old_op_outputs = old_ir_ops[op_name]["outputs"]
        old_op_optionals = old_ir_ops[op_name]["optionals"]
        old_op_inplaces = old_ir_ops[op_name]["inplaces"]

        new_op_inputs = new_ir_ops[op_name]["inputs"]
        new_op_attrs = new_ir_ops[op_name]["attrs"]
        new_op_outputs = new_ir_ops[op_name]["outputs"]
        new_op_optionals = new_ir_ops[op_name]["optionals"]
        new_op_inplaces = new_ir_ops[op_name]["inplaces"]

        diff_old_new_inputs = []
        for inp in old_op_inputs:
            if inp not in new_op_inputs:
                diff_old_new_inputs.append(inp)
        
        diff_new_old_inputs = []
        for inp in new_op_inputs:
            if inp not in old_op_inputs:
                diff_new_old_inputs.append(inp)
        
        diff_old_new_attrs = []
        for attr in old_op_attrs:
            if attr not in new_op_attrs:
                diff_old_new_attrs.append(attr)
        
        diff_new_old_attrs = []
        for attr in new_op_attrs:
            if attr not in old_op_attrs:
                diff_new_old_attrs.append(attr)

        diff_old_new_outputs = []
        for output in old_op_outputs:
            if output not in new_op_outputs:
                diff_old_new_outputs.append(output)
        
        diff_new_old_outputs = []
        for output in new_op_outputs:
            if output not in old_op_outputs:
                diff_new_old_outputs.append(output)

        diff_old_new_optionals = []
        for optional in old_op_optionals:
            if optional not in new_op_optionals:
                diff_old_new_optionals.append(optional)

        diff_new_old_optionals = []
        for optional  in new_op_optionals:
            if optional not in old_op_optionals:
                diff_new_old_optionals.append(optional)

        diff_old_new_inplaces = []
        for inplace in old_op_inplaces:
            if inplace not in new_op_inplaces:
                diff_old_new_inplaces.append(inplace)

        diff_new_old_inplaces = []
        for inplace in new_op_inplaces:
            if inplace not in old_op_inplaces:
                diff_new_old_inplaces.append(inplace)


        diff_inputs_str = ""
        
        for old_input in diff_old_new_inputs:
            diff_inputs_str = diff_inputs_str + old_input[0] + " " + old_input[1] + ","
        if len(diff_inputs_str) > 0 and diff_inputs_str[-1] == ',':
            diff_inputs_str = diff_inputs_str[:-1]
        
        diff_inputs_str = "(" + diff_inputs_str + ") -> ("
        
        for new_input in diff_new_old_inputs:
            diff_inputs_str = diff_inputs_str + new_input[0] + " " + new_input[1] + ","
        if len(diff_inputs_str) > 0 and diff_inputs_str[-1] == ',':
            diff_inputs_str = diff_inputs_str[:-1]
        diff_inputs_str  = diff_inputs_str + ")"

        print(diff_inputs_str)
        
        diff_attrs_str = ""
        masked_attr_set = {'op_callstack','op_device','op_namescope','op_role','op_role_var','with_quant_attr'}
        
        for old_attr in diff_old_new_attrs:
            if old_attr[1] not in masked_attr_set:
                diff_attrs_str = diff_attrs_str + old_attr[0] + " " + old_attr[1] + ","
        if len(diff_attrs_str) > 0 and diff_attrs_str[-1] == ',':
            diff_attrs_str = diff_attrs_str[:-1]
        
        diff_attrs_str = "(" + diff_attrs_str + ") -> ("
        
        for new_attr in diff_new_old_attrs:
            diff_attrs_str = diff_attrs_str + new_attr[0] + " " + new_attr[1] + ","
        if len(diff_attrs_str) > 0 and diff_attrs_str[-1] == ',':
            diff_attrs_str = diff_attrs_str[:-1]
        diff_attrs_str  = diff_attrs_str + ")"

        print(diff_attrs_str)

        diff_outputs_str = ""
        
        for old_output in diff_old_new_outputs:
            diff_outputs_str = diff_outputs_str + old_output[0] + "(" + old_output[1] + "), "
        if len(diff_outputs_str) > 1 and diff_outputs_str[-1] == ' ':
            diff_outputs_str = diff_outputs_str[:-2]
        
        diff_outputs_str = diff_outputs_str + "->"
        
        for new_output in diff_new_old_outputs:
            diff_outputs_str = diff_outputs_str + new_output[0] + "(" + new_output[1] + "), "
        if len(diff_outputs_str) > 1 and diff_outputs_str[-1] == ' ':
            diff_outputs_str = diff_outputs_str[:-2]

        print(diff_outputs_str)

        diff_optionals_str= ""
        for old_optional in diff_old_new_optionals:
            diff_optionals_str = diff_optionals_str + old_optional + ', '
        if len(diff_optionals_str) > 2 and diff_optionals_str[-1] == ' ':
            diff_optionals_str = diff_optionals_str[:-2]
            
        diff_optionals_str = "(" + diff_optionals_str + ") -> ("

        for new_optional in diff_new_old_optionals:
            diff_optionals_str = diff_optionals_str + new_optional + ', '
        if len(diff_optionals_str) > 2 and diff_optionals_str[-1] == ' ':
            diff_optionals_str = diff_optionals_str[:-2]

        diff_optionals_str = diff_optionals_str + ")"

        print(diff_optionals_str)


        diff_inplaces_str = ""
        
        for old_inplace in diff_old_new_inplaces:
            print(diff_old_new_inplaces)
            diff_inplaces_str = diff_inplaces_str + old_inplace[0] + "->" + old_inplace[1] + ","
        if len(diff_inplaces_str) > 0 and diff_inplaces_str[-1] == ',':
            diff_inplaces_str = diff_inplaces_str[:-1]
        
        diff_inplaces_str = "(" + diff_inplaces_str + ") -> ("
        
        for new_inplace in diff_new_old_inplaces:
            diff_inplaces_str = diff_inplaces_str + new_inplace[0] + " " + new_inplace[1] + ","
        if len(diff_inplaces_str) > 0 and diff_inplaces_str[-1] == ',':
            diff_inplaces_str = diff_inplaces_str[:-1]
        diff_inplaces_str  = diff_inplaces_str + ")"

        print(diff_inplaces_str)

        
        op_compare_result.append({"op":op_name,"inputs":diff_inputs_str,"attrs":diff_attrs_str,"outputs":diff_outputs_str,"optional":diff_optionals_str,"inplaces":diff_inplaces_str})

    for result in op_compare_result:
        if result['inputs'] == "() -> ()":
            result.pop('inputs')
        if result['attrs'] == "() -> ()":
            result.pop('attrs')
        if result['outputs'] == "->":
            result.pop('outputs')
        if result['optional'] == "() -> ()":
            result.pop('optional')
        if result['inplaces'] == "() -> ()":
            result.pop('inplaces')

    op_compare_result = [d for d in op_compare_result if len(d) > 1]
    return op_compare_result

if __name__ == '__main__':
    new_ir_yaml_path = r"/home/aistudio/test/Paddle/tools/count_op/compare/in_old_in_new_new_ir_ops.yaml"
    old_ir_yaml_path = r"/home/aistudio/test/Paddle/tools/count_op/compare/in_old_in_new_old_ir_ops.yaml"
    compare_result_path = r"/home/aistudio/test/Paddle/tools/count_op/compare/compare_results_new.yaml"

    results = compare(new_ir_yaml_path,old_ir_yaml_path)

    
    with open(compare_result_path, 'w') as file:
        for op_info in results:
            temp = [op_info]
            yaml.dump(
                temp,
                file,
                default_flow_style=False,
                sort_keys=False,
                indent=1,
            )
            file.write("\n")