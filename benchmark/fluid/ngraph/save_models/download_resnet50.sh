#/bin/bash
root_url=http://paddle-imagenet-models-name.bj.bcebos.com
train_tar=ResNet50_pretrained.zip
train_dir=ResNet50_pretrained
echo "Download imagenet training data..."
wget -nd -c ${root_url}/${train_tar}
unzip ${train_tar}
cp __model__ ${train_dir}/
