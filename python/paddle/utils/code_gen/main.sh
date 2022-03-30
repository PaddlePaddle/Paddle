python parse_api.py \
  --api_yaml_path ./api.yaml \
  --output_path ./temp/api.parsed.yaml
python parse_api.py \
  --api_yaml_path ./backward.yaml \
  --output_path ./temp/backward_api.parsed.yaml \
  --backward

python generate_op.py \
  --api_yaml_path ./temp/api.parsed.yaml \
  --backward_api_yaml_path ./temp/backward_api.parsed.yaml \
  --output_dir ./temp