#!/usr/bin/env bash
set -e
work_dir=$(dirname $(dirname $(realpath $0)))
data_dir=data

# md5 info
# e6ecbe86fff843acb1ae254789cd8c6d data/1k_utt_feats_pdf.ark
# ec62e2287702ed03456ed18e1a1c3d64 data/9k_utt_feats_pdf.ark
cd $data_dir
for item in 1k_utt_feats_pdf.ark 9k_utt_feats_pdf.ark.tar.001 \
    9k_utt_feats_pdf.ark.tar.002 9k_utt_feats_pdf.ark.tar.003;do
  if [ ! -f $item ];then
    wget -P . http://dl.bintray.com/pineking/atlas/$item
  fi
done
cat 9k_utt_feats_pdf.ark.tar.* | tar -x
cd -

# Use 9000 utterances to train, 1000 utterances to evaluation
echo "$data_dir/9k_utt_feats_pdf.ark" > $data_dir/train.list
echo "$data_dir/1k_utt_feats_pdf.ark" > $data_dir/test.list

nvidia-docker run -it --rm \
  --privileged \
  -v $work_dir:/paddle/demo \
  docker.io/paddledev/paddle:gpu-latest /bin/bash
