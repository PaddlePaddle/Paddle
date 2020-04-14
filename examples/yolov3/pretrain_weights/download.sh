DIR="$( cd "$(dirname "$0")" ; pwd -P   )"
cd "$DIR"

echo "Downloading https://paddlemodels.bj.bcebos.com/hapi/darknet53_pretrained.pdparams"
wget https://paddlemodels.bj.bcebos.com/hapi/darknet53_pretrained.pdparams
