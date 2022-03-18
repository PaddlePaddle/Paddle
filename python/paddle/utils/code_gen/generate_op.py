import pathlib
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from filters import to_op_attr_type, to_opmaker_name, to_pascal_case
from tests import is_base_api

file_loader = FileSystemLoader(pathlib.Path(__file__).parent / "templates")
env = Environment(loader=file_loader,
                  keep_trailing_newline=True,
                  trim_blocks=True,
                  lstrip_blocks=True,
                  undefined=StrictUndefined,
                  extensions=['jinja2.ext.do'])
env.filters["to_op_attr_type"] = to_op_attr_type
env.filters["to_opmaker_name"] = to_opmaker_name
env.filters["to_pascal_case"] = to_pascal_case
env.tests["base_api"] = is_base_api


with open("api.parsed.yaml", "rt") as f:
    apis = yaml.safe_load(f)
    
with open("backward.parsed.yaml", 'rt') as f:
    backward_apis = yaml.safe_load(f)

# # op2.cc
template = env.get_template('op.c.j2')
with open("op2.cc", "wt") as f:
    for api in apis:
        msg = template.render(api=api)
        f.write(msg)
        
# # backward.op2.cc
template = env.get_template('op.c.j2')
with open("backward.op2.cc", "wt") as f:
    for api in apis:
        msg = template.render(api=api)
        f.write(msg)

# ks2.cc
template = env.get_template('ks.c.j2')
with open("ks2.cc", "wt") as f:
    for api in apis:
        msg = template.render(api=api)
        f.write(msg)

# backward.ks2.cc
template = env.get_template('ks.c.j2')
with open("backward.ks2.cc", "wt") as f:
    for api in backward_apis:
        msg = template.render(api=api)
        f.write(msg)
