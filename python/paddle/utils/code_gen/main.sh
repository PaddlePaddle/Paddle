python parse_api.py \
  --api_yaml_path ./api.yaml --output_path ./temp/api.parsed.yaml
python generate_op.py \
  --api_yaml_path ./temp/api.parsed.yaml --output_dir ./temp