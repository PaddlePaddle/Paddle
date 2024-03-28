from argparse import ArgumentParser
import re

def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


code_template = """
#include "paddle/extension.h"
#include "${triton_kernel_header_file}"




void ${custom_op_name}_func(${para}) {


auto get_tensor_ptr = [](const paddle::Tensor& input) -> CUdeviceptr {
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
};



  auto status = ${triton_kernel}(${invoke_para});
  assert(status == CUDA_SUCCESS);
}

PD_BUILD_OP(${custom_op_name})
    .Inputs({${op_inputs}})
    .Outputs({${op_outputs}})
    .SetInplaceMap({${inplace_map}})
    .Attrs({${op_attrs}}) 
    .SetKernelFn(PD_KERNEL(${custom_op_name}_func));
"""



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--custom-op-name", "-on", type=str, default="", help="Name of the paddle custom op", required=True)
    parser.add_argument("--custom-op-file", "-cof", type=str, help="The file.cu of generated .cu file", required=True)
    parser.add_argument("--kernel-name", "-kn", type=str, default="", help="Name of the triton kernel you will invoke", required=True)
    parser.add_argument("--signature", "-s", type=str, help="Signature of the triton kernel", required=True)
    parser.add_argument("--output-ids", "-oi", type=str, help="Output ids in your signature", required=True)
    parser.add_argument("--triton-kernel-header-file", "-header", type=str, help="The header file of the triton kernel", required=True)

    args = parser.parse_args()

    template_dict = {}
    template_dict["triton_kernel"] = args.kernel_name
    template_dict["custom_op_name"] = args.custom_op_name
    signature = args.signature
    signature = signature.split(",")

    
    def convert_to_ctype(sig):
        if ("i32" in sig):
          return "int"
        elif ("fp32" in sig):
          return "float"

    op_inputs_len = 0
    attrs_type = []

    for i in range(len(signature)):
      sig = signature[i]
      sig = sig.strip(" ")
      if ("*" in sig):
        op_inputs_len += 1
      elif (sig.isdigit()):
        print("this is a mata-parameter")
        pass
      else:
        attrs_type.append(convert_to_ctype(sig))

    para = ""
    for i in range(op_inputs_len):
      para += "const paddle::Tensor& para{id},".format(id=i)
    for i in range(len(attrs_type)):
      para += attrs_type[i] + " attr{id},".format(id=i)
    para = para[:-1]

    template_dict["para"] = para    
    
    invoke_para = "para0.stream(),"
    for i in range(op_inputs_len):
      invoke_para += "get_tensor_ptr(para{id}),".format(id=i)
    for i in range(len(attrs_type)):
      invoke_para += "attr{id},".format(id=i)
    invoke_para += "0"
    template_dict["invoke_para"] = invoke_para

    op_inputs = ""
    for i in range(op_inputs_len):
      op_inputs += "\"para{id}\", ".format(id=i)
    op_inputs = op_inputs[:-2]
    template_dict["op_inputs"] = op_inputs
    
    op_attrs = ""
    for i in range(len(attrs_type)):
      op_attrs += ("\"attr{id} : {type}\", ").format(id=i, type=attrs_type[i])
    # remove the last ","
    op_attrs = op_attrs[:-1]

    template_dict["op_attrs"] = op_attrs

    output_ids = args.output_ids
    output_ids = output_ids.split(",")
    op_outputs = ""
    for id in output_ids:
      op_outputs += "\"out{id}\",".format(id=id)
    op_outputs = op_outputs[:-1]
    template_dict["op_outputs"] = op_outputs

    inplace_map = ""
    for id in output_ids:
      inplace_map += "{{\"para{id}\",\"out{id}\"}},".format(id=id)
    inplace_map = inplace_map[:-1]
    
    template_dict["inplace_map"] = inplace_map
    template_dict["triton_kernel_header_file"] = args.triton_kernel_header_file


    file_path = args.custom_op_file
    with open(file_path, "w") as f:
        f.write(SubstituteTemplate(code_template, template_dict))
        f.close()
