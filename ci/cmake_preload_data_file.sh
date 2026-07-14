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

set -e

echo "=== preload data file from CFS_DIR ==="

BASE_SRC=${CFS_DIR}/preload_file
BASE_DST=/paddle/build/third_party/inference_demo

declare -A FILE_MAP=(
  ["word2vec.inference.model.tar.gz"]="word2vec"
  ["image_classification_resnet.inference.model.tgz"]="image_classification_resnet"

  ########################
  # Quant1
  ########################
  ["ResNet50_qat_model.tar.gz"]="quant/ResNet50_quant"
  ["ResNet101_qat_model.tar.gz"]="quant/ResNet101_quant"
  ["GoogleNet_qat_model.tar.gz"]="quant/GoogleNet_quant"
  ["MobileNetV1_qat_model.tar.gz"]="quant/MobileNetV1_quant"
  ["MobileNetV2_qat_model.tar.gz"]="quant/MobileNetV2_quant"
  ["VGG16_qat_model.tar.gz"]="quant/VGG16_quant"
  ["VGG19_qat_model.tar.gz"]="quant/VGG19_quant"

  ########################
  # Quant2
  ########################
  ["ResNet50_qat_perf.tar.gz"]="quant/ResNet50_quant2"
  ["ResNet50_qat_range.tar.gz"]="quant/ResNet50_quant2_range"
  ["ResNet50_qat_channelwise.tar.gz"]="quant/ResNet50_quant2_channelwise"
  ["MobileNet_qat_perf.tar.gz"]="quant/MobileNetV1_quant2"

  ########################
  # NLP
  ########################
  ["Ernie_dataset.tar.gz"]="Ernie_dataset"
  ["ernie_qat.tar.gz"]="quant/Ernie_quant2"
  ["ernie_fp32_model.tar.gz"]="quant/Ernie_float"

  ########################
  # LSTM
  ########################
  ["lstm_quant.tar.gz"]="quant/lstm_quant_test"
  ["quant_lstm_input_data.tar.gz"]="quant/lstm_quant2_int8"
  ["lstm_fp32_model.tar.gz"]="quant/lstm_quant2_int8"
)

for file in "${!FILE_MAP[@]}"; do
    src="${BASE_SRC}/${file}"
    dst="${BASE_DST}/${FILE_MAP[$file]}"

    mkdir -p "${dst}"

    if [ -f "${src}" ]; then
        echo "[HIT] copy ${src} -> ${dst}"
        cp -f "${src}" "${dst}/"
    else
        echo "[MISS] ${src} not found"
    fi
done

echo "=== preload finished ==="
