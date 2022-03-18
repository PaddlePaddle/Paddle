from parse_utils import parse_api_entry
import yaml
import json

def main():
    with open("/Users/chenfeiyu/projects/Paddle/python/paddle/utils/code_gen/api.yaml", "rt") as f:
        apis = yaml.safe_load(f)
        apis = [parse_api_entry(api) for api in apis]
    
    with open("api.parsed.yaml", "wt") as f:
        yaml.safe_dump(apis, f, default_flow_style=None, sort_keys=False)
        
    # with open("api.parsed.jsonl", "wt") as f:
    #     for api in apis:
    #         f.write(json.dumps(api))
    #         f.write("\n")
            
    with open("/Users/chenfeiyu/projects/Paddle/python/paddle/utils/code_gen/backward.yaml", "rt") as f:
        apis = yaml.safe_load(f)
        apis = [parse_api_entry(api, "backward_api") for api in apis]
    
    with open("backward.parsed.yaml", "wt") as f:
        yaml.safe_dump(apis, f, default_flow_style=None, sort_keys=False)
        
    # with open("backward.parsed.jsonl", "wt") as f:
    #     for api in apis:
    #         f.write(json.dumps(api))
    #         f.write("\n")
        
if __name__ == "__main__":
    main()
