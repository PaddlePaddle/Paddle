import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hapi.model import Model, Loss

from hapi.vision.models import vgg16
from hapi.vision.transforms import transforms
from paddle import fluid
from paddle.fluid.io import Dataset

import cv2
import copy


def load_image(image_path, max_size=400, shape=None):
    image = cv2.imread(image_path)
    image = image.astype('float32') / 255.0
    size = shape if shape is not None else max_size if max(
        image.shape[:2]) > max_size else max(image.shape[:2])

    transform = transforms.Compose([
        transforms.Resize(size), transforms.Permute(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)[np.newaxis, :3, :, :]
    image = fluid.dygraph.to_variable(image)
    return image


def image_restore(image):
    image = np.squeeze(image.numpy(), 0)
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))

    image = image.clip(0, 1)
    return image


class StyleTransferModel(Model):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # pretrained设置为true，会自动下载imagenet上的预训练权重并加载
        vgg = vgg16(pretrained=True)
        self.base_model = vgg.features
        for p in self.base_model.parameters():
            p.stop_gradient = True
        self.layers = {
            '0': 'conv1_1',
            '3': 'conv2_1',
            '6': 'conv3_1',
            '10': 'conv4_1',
            '11': 'conv4_2',  ## content representation
            '14': 'conv5_1'
        }

    def forward(self, image):
        outputs = []
        for name, layer in self.base_model.named_sublayers():
            image = layer(image)
            if name in self.layers:
                outputs.append(image)
        return outputs


class StyleTransferLoss(Loss):
    def __init__(self,
                 content_loss_weight=1,
                 style_loss_weight=1e5,
                 style_weights=[1.0, 0.8, 0.5, 0.3, 0.1]):
        super(StyleTransferLoss, self).__init__()
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.style_weights = style_weights

    def forward(self, outputs, labels):
        content_features = labels[-1]
        style_features = labels[:-1]

        # 计算图像内容相似度的loss
        content_loss = fluid.layers.mean((outputs[-2] - content_features)**2)

        # 计算风格相似度的loss
        style_loss = 0
        style_grams = [self.gram_matrix(feat) for feat in style_features]
        style_weights = self.style_weights
        for i, weight in enumerate(style_weights):
            target_gram = self.gram_matrix(outputs[i])
            layer_loss = weight * fluid.layers.mean((target_gram - style_grams[
                i])**2)
            b, d, h, w = outputs[i].shape
            style_loss += layer_loss / (d * h * w)

        total_loss = self.content_loss_weight * content_loss + self.style_loss_weight * style_loss
        return total_loss

    def gram_matrix(self, A):
        if len(A.shape) == 4:
            _, c, h, w = A.shape
            A = fluid.layers.reshape(A, (c, h * w))
        GA = fluid.layers.matmul(A, fluid.layers.transpose(A, [1, 0]))

        return GA


def main():
    # 启动动态图模式
    fluid.enable_dygraph()

    content = load_image(FLAGS.content_image)
    style = load_image(FLAGS.style_image, shape=tuple(content.shape[-2:]))

    model = StyleTransferModel()
    style_loss = StyleTransferLoss()

    # 使用内容图像初始化要生成的图像
    target = Model.create_parameter(model, shape=content.shape)
    target.set_value(content.numpy())

    optimizer = fluid.optimizer.Adam(
        parameter_list=[target], learning_rate=FLAGS.lr)
    model.prepare(optimizer, style_loss)

    content_fetures = model.test_batch(content)
    style_features = model.test_batch(style)

    # 将两个特征组合，作为损失函数的label传给模型
    feats = style_features + [content_fetures[-2]]

    # 训练5000个step，每500个step画一下生成的图像查看效果
    steps = FLAGS.steps
    for i in range(steps):
        outs = model.train_batch(target, feats)
        if i % 500 == 0:
            print('iters:', i, 'loss:', outs[0][0])

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # 保存生成好的图像
    name = FLAGS.content_image.split(os.sep)[-1]
    output_path = os.path.join(FLAGS.save_dir, 'generated_' + name)
    cv2.imwrite(output_path,
                cv2.cvtColor((image_restore(target) * 255).astype('uint8'),
                             cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument(
        "--content-image",
        type=str,
        default='./images/chicago_cropped.jpg',
        help="content image")
    parser.add_argument(
        "--style-image",
        type=str,
        default='./images/Starry-Night-by-Vincent-Van-Gogh-painting.jpg',
        help="style image")
    parser.add_argument(
        "--save-dir", type=str, default='./output', help="output dir")
    parser.add_argument(
        "--steps", default=5000, type=int, help="number of steps to run")
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    FLAGS = parser.parse_args()
    main()
