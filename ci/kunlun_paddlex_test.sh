# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export FLAGS_enable_pir_api=0
#install paddlex
git clone --depth 1000 https://gitee.com/paddlepaddle/PaddleX.git
cd PaddleX
pip install -e .

#install paddle x dependency
paddlex --install PaddleClas

#download paddle dataset
wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/

#train Reset50
echo "Starting to train ResNet50 model..."
python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.output=resnet50_output \
    -o Global.device="xpu:${CUDA_VISIBLE_DEVICES}"
echo "Training Resnet50 completed!"

#inference Reset50
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
echo ${DEVICES[0]}

echo "Starting to predict ResNet50 model..."
python main.py -c paddlex/configs/modules/image_classification/ResNet50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./resnet50_output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg" \
    -o Global.device="xpu:${DEVICES[0]}"
echo "Predicting Resnet50 completed!"
cd ..
export FLAGS_enable_pir_api=1
