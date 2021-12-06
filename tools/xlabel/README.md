<h1 align="center">
  <img src="./images/logo.png"><br/>LabelImage
</h1>
<h4 align="center">
  一款在线的深度学习图像分割标注工具
</h4>
<div align="center">
  <img src="./images/readme/example.png" width="70%">
</div>

# 描述
LabelImage 是一款用于深度学习分割模型训练的图像标注工具（生成.json文件），可以对你将要训练的模型提供帮助。
<br/>
[在线演示](https://paddlecv-sig.github.io/label_image/)

本地运行：
<br/>

```bash
git clone git@github.com:PaddleCV-SIG/label_image.git
cd label_image
# For Python3
python server/flask_server.py
```

<br/>
然后打开浏览器访问 http://localhost:627

<br/>

# 如何开发

建议使用Visual Studio Code进行开发，并安装Prettier扩展。

# 后端API
<a href="./backend.md">这里</a>
<br/>

# 功能清单
- [x] 上传多个文件，可切换不同图片
- [x] 矩形标注工具、多边形标注工具 （持续新增工具中……）
- [x] 图片拖拽，缩放，缩略图显示
- [x] 标签管理：增、删、改、查（存储在本地）
- [x] 标注结果编辑修改、删除
- [x] 操作管理历史记录
- [ ] 掩膜标注
- [ ] 对接后端
- [ ] 修改色调

# 功能演示
#### 矩形工具操作示例
<div>
  <img src="./images/readme/rectExample.gif">
</div>

#### 多边形工具操作示例
<div>
  <img src="./images/readme/polygonExample.gif">
</div>

#### 拖拽操作示例
 - 图片拖拽可以选择拖拽工具或者按住鼠标右键快捷拖拽
 - 对坐标点进行拖拽更新
<div>
  <img src="./images/readme/dragExample1.gif">
  <img src="./images/readme/dragExample2.gif">
</div>

#### 历史记录操作示例
<div>
  <img src="./images/readme/historyExample.gif">
</div>


# 要求
- Ubuntu / macOS / Windows
- Chrome v51+ / Firefox v53+

# 问题
如有疑问或建议，随时向我留言
Email: rachelcao277@hotmail.com

# 结语
感谢你的使用，希望能对你有所帮助

