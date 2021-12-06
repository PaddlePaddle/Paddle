/*
  Canvas handle 主函数
 */

class LabelImage {
  constructor(options) {
    // Dataset ID
    this.dataset_id = options.dataset_id,
      // 后端请求工具
      this.Backend = options.Backend,
      // 画布宽度
      this.cWidth = options.canvas.clientWidth;
    // 画布高度
    this.cHeight = options.canvas.clientHeight;
    // 缩略图宽度
    this.sWidth = 0;
    // 缩略图高度
    this.sHeight = 0;
    // 图片宽度
    this.iWidth = 0;
    // 图片高度
    this.iHeight = 0;
    // 图片拖拽至边缘最小显示
    this.appearSize = 180;
    // 缩放布进
    this.scaleStep = 0.02;
    // 最小缩放比例
    this.minScale = 0.2;
    // 最大缩放比例
    this.maxScale = 9;
    // 图片在画板中的横坐标
    this.x = 0;
    // 图片在画板中的纵坐标
    this.y = 0;
    // 鼠标当前画板中的横坐标
    this.mouseX = 0;
    // 鼠标当前画板中的纵坐标
    this.mouseY = 0;
    // 拖动过程中，鼠标前一次移动的横坐标
    this.prevX = 0;
    // 拖动过程中，鼠标前一次移动的纵坐标
    this.prevY = 0;
    // 缩放比例
    this.scale = 0;
    // 鼠标在图片中的横坐标
    this.ix = 0;
    // 鼠标在图片中的纵坐标
    this.iy = 0;
    // 矩形框起点横坐标
    this.rectX = 0;
    // 矩形框起点纵坐标
    this.rectY = 0;
    // 绘制多边形的圆点半径
    this.radius = 6;

    // 绘制线段宽度
    this.lineWidth = 1;

    //绘制区域模块透明度
    this.opacity = 0.45;

    // 定时器
    this.timer = null;

    // 结果是否被修改
    this.isModify = false;

    // 是否全屏
    this.isFullScreen = false;

    // 当前点击圆点index
    this.snapCircleIndex = 0;

    // 用于在拖拽或者缩放时，让绘制至存储面板的数据，只绘制一次
    this.drawFlag = true;

    // 监听滚动条缩放是否结束的定时器
    this.mousewheelTimer = null;

    // 历史记录下标
    this.historyIndex = 0;
    // 当前默认绘制图形
    this.defaultDraw = "rectOn";

    this.Arrays = {
      // 标定历史保存标签记录
      history: [],

      // 图片标注展示数据集
      imageAnnotateShower: [],

      // 图片标注存储数据集
      imageAnnotateMemory: [],

      // 标注集操作 result list index
      resultIndex: 0,
    };
    this.Nodes = {
      // 图片节点
      image: null,
      // 画布节点
      canvas: options.canvas,
      // 缩略图节点
      scaleCanvas: options.scaleCanvas,
      // 缩放比例面板
      scalePanel: options.scalePanel,
      // 画布上下文
      ctx: options.canvas.getContext("2d"),
      // 缩略图画板上下文
      sCtx: null,
      // 缩略图方框
      scaleRect: null,
      // 存储图像数据的画布
      bCanvas: null,
      // 存储图像数据的上下文
      bCtx: null,
      // 绘图部分主外层函数
      canvasMain: options.canvasMain,
      // 标记结果数据集
      resultGroup: options.resultGroup,
      // 十字线开关按钮
      crossLine: options.crossLine,
      // 标注结果开关按钮
      labelShower: options.labelShower,
      // 屏幕快照按钮
      screenShot: options.screenShot,
      // 全屏按钮
      screenFull: options.screenFull,
      // 颜色选取器节点数据
      colorHex: options.colorHex,
      // 清空标注内容
      // clearScreen: options.clearScreen,
      // 撤销，返回上一步
      // returnUp: options.returnUp,
      // 标签管理
      toolTagsManager: options.toolTagsManager,
      // 历史记录列表
      historyGroup: options.historyGroup,
    };
    this.Features = {
      // 拖动开关
      dragOn: true,
      // 矩形标注开关
      rectOn: false,
      // 多边形标注开关
      polygonOn: false,
      // 掩膜标注开关
      maskOn: false,
      // 标签管理工具
      tagsOn: false,
      // 十字线开关
      crossOn: true,
      // 标注结果显示
      labelOn: true,
    };
    this.Initial();
  }

  //----初始化节点参数，绑定各自事件
  Initial = async () => {
    let _nodes = this.Nodes;
    _nodes.scaleRect = document.createElement("div");
    _nodes.scaleRect.className = "scaleWindow";
    Object.assign(_nodes.scaleRect.style, {
      position: "absolute",
      border: "1px solid red",
      boxSizing: "border-box",
    });
    // _nodes.scaleCanvas.appendChild(_nodes.scaleRect);
    _nodes.canvas.addEventListener("mousedown", this.CanvasMouseDown);
    // _nodes.canvas.addEventListener("mousewheel", this.MouseWheel);
    // _nodes.canvas.addEventListener("DOMMouseScroll", this.MouseWheel); // 兼容Firefox 滚动条事件
    _nodes.canvas.addEventListener(
      "contextmenu",
      LabelImage.NoRightMenu.bind(this)
    );
    // _nodes.scaleCanvas.addEventListener("click", this.ScaleCanvasClick);
    // _nodes.crossLine.addEventListener('click', this.CrossHairSwitch);
    // _nodes.labelShower.addEventListener('click', this.IsShowLabels);
    // _nodes.screenShot.addEventListener('click', this.ScreenShot);
    // _nodes.screenFull.addEventListener('click', this.IsScreenFull);
    _nodes.historyGroup.addEventListener("click", this.HistoryClick);
    document.addEventListener("fullscreenchange", this.ScreenViewChange);
    document.addEventListener("webkitfullscreenchange", this.ScreenViewChange);
    document.addEventListener("mozfullscreenchange", this.ScreenViewChange);
    document.addEventListener("msfullscreenchange", this.ScreenViewChange);
    _nodes.canvas.addEventListener("mousemove", this.CanvasMouseMove);
    _nodes.resultGroup.addEventListener("mouseover", this.ResultListOperation);
    _nodes.toolTagsManager.addEventListener("click", this.ManageLabels);
  };

  //----设置图片并初始化画板信息
  // tip: point
  SetImage = async (img_url, img_name) => {
    let _nodes = this.Nodes;
    _nodes.image = new Image();
    _nodes.image.crossOrigin = "anonymous";
    _nodes.image.src = img_url;
    //监听图片加载
    _nodes.image.addEventListener("load", async () => {
      openBox("#loading", false);
      this.iWidth = _nodes.image.width;
      this.iHeight = _nodes.image.height;
      //获取原有节点
      // let beforeCanvas = _nodes.scaleCanvas.querySelectorAll("canvas");
      let bodyCanvas = document.querySelector(".bodyCanvas");

      //删除原有节点
      // if (beforeCanvas.length > 0) {
      //   _nodes.scaleCanvas.removeChild(beforeCanvas[0]);
      // }
      if (bodyCanvas) {
        document.body.removeChild(bodyCanvas);
      }

      //初始化上一张图片标注数据
      for (let i = this.Nodes.resultGroup.children.length - 1; i >= 0; i--) {
        this.Nodes.resultGroup.removeChild(this.Nodes.resultGroup.children[i]);
      }
      for (let i = this.Nodes.historyGroup.children.length - 1; i >= 0; i--) {
        this.Nodes.historyGroup.removeChild(
          this.Nodes.historyGroup.children[i]
        );
      }
      document.querySelector(".resultLength").innerHTML = "0";
      this.Arrays.imageAnnotateShower.splice(
        0,
        this.Arrays.imageAnnotateShower.length
      );
      this.Arrays.imageAnnotateMemory.splice(
        0,
        this.Arrays.imageAnnotateMemory.length
      );
      this.Arrays.history.splice(0, this.Arrays.history.length);

      //创建缩略图画板
      // let sCanvas = document.createElement("canvas");
      // _nodes.sCtx = sCanvas.getContext("2d");
      // sCanvas.style.display = "block";
      // this.sWidth = parseInt(_nodes.scaleCanvas.getBoundingClientRect().width);
      // this.sHeight = parseInt((this.sWidth * this.iHeight) / this.iWidth);
      // sCanvas.width = this.sWidth;
      // sCanvas.height = this.sHeight;
      // _nodes.scaleCanvas.appendChild(sCanvas);

      // 创建数据存储面板
      _nodes.bCanvas = document.createElement("canvas");
      _nodes.bCanvas.width = this.iWidth;
      _nodes.bCanvas.height = this.iHeight;
      _nodes.bCanvas.style.display = "none";
      _nodes.bCanvas.className = "bodyCanvas";
      _nodes.bCtx = _nodes.bCanvas.getContext("2d");
      _nodes.bCtx.drawImage(_nodes.image, 0, 0, this.iWidth, this.iHeight);
      _nodes.bCtx.translate(0.5, 0.5);
      document.body.appendChild(_nodes.bCanvas);

      this.scale = 1;

      // 图片初始定位
      // 初始化自适应缩放图片并居中
      if (this.iWidth > this.cWidth || this.iHeight > this.cHeight) {
        this.scale =
          this.iWidth - this.cWidth > this.iHeight - this.cHeight
            ? this.cWidth / this.iWidth
            : this.cHeight / this.iHeight;
      }
      let initImgX = (this.cWidth - this.iWidth * this.scale) / 2;
      let initImgY = (this.cHeight - this.iHeight * this.scale) / 2;
      this.SetXY(initImgX, initImgY);

      this.historyIndex = 0;
      let annotations = await Backend.getAnnotation(img_name);
      this.Arrays.imageAnnotateMemory = annotations.data.data;
      this.ReplaceAnnotateShow();
      this.RepaintResultList();
      this.Arrays.imageAnnotateMemory.forEach((memory, index) => {
        this.RecordOperation("add", "绘制", index, JSON.stringify(memory));
      });
    });
  };

  //----设置功能参数
  SetFeatures = (f, value) => {
    if (f === "crossOn" || f === "labelOn") {
      this.Features[f] = value;
    } else {
      for (let key in this.Features) {
        if (key !== "crossOn" && key !== "labelOn") {
          this.Features[key] = false;
        }
      }
    }
    this.Features[f] = value;

    // 清空标注结果列表中classList
    let resultList =
      this.Nodes.resultGroup.getElementsByClassName("result_list");
    for (let i = 0; i < resultList.length; i++) {
      resultList[i].classList.remove("active");
    }
    this.Arrays.resultIndex = 0;
    this.DrawSavedAnnotateInfoToShow(this.Arrays.resultIndex);
    this.defaultDraw = f;
  };

  //----更新画板数据, 将存储面板数据绘制到展示面板已经缩略图面板
  UpdateCanvas = () => {
    let _nodes = this.Nodes;
    _nodes.ctx.clearRect(0, 0, this.cWidth, this.cHeight);
    // _nodes.sCtx.clearRect(
    //   0,
    //   0,
    //   this.sWidth,
    //   (this.sWidth * this.iHeight) / this.iHeight
    // );

    _nodes.ctx.drawImage(
      _nodes.bCanvas,
      -this.x / this.scale,
      -this.y / this.scale,
      this.cWidth / this.scale,
      this.cHeight / this.scale,
      0,
      0,
      this.cWidth,
      this.cHeight
    );
    // _nodes.sCtx.drawImage(
    //   _nodes.bCanvas,
    //   0,
    //   0,
    //   this.iWidth,
    //   this.iHeight,
    //   0,
    //   0,
    //   this.sWidth,
    //   this.sHeight
    // );

    // 将缩略图方框区域绘制到画布
    let width = (this.sWidth * this.cWidth) / this.iWidth / this.scale;
    let height = (width * this.cHeight) / this.cWidth;
    let left = (-this.x * this.sWidth) / (this.iWidth * this.scale);
    let top = (-this.y * this.sWidth) / (this.iWidth * this.scale);
    // 将方框宽度固定在缩略图面板中
    if (width + left >= this.sWidth) {
      width = this.sWidth - left;
      left = this.sWidth - width;
      if (width >= this.sWidth) {
        width = this.sWidth;
        left = 0;
      }
    } else if (left <= 0) {
      width += left;
      left = 0;
    }

    // 将方框高度固定在缩略图面板中
    if (height + top >= this.sHeight) {
      height = this.sHeight - top;
      top = this.sHeight - height;
      if (height >= this.sHeight) {
        height = this.sHeight;
        top = 0;
      }
    } else if (top <= 0) {
      height += top;
      top = 0;
    }

    _nodes.scaleRect.style.left = left + "px";
    _nodes.scaleRect.style.top = top + "px";
    if (width !== Number(_nodes.scaleRect.style.width)) {
      _nodes.scaleRect.style.width = width + "px";
      _nodes.scaleRect.style.height = height + "px";
    }

    // _nodes.scalePanel.innerText = (this.scale * 100).toFixed(2) + "%";
  };

  //----画板跟随鼠标十字线绘制函数
  MouseMoveCrossHair = () => {
    let _nodes = this.Nodes;
    _nodes.ctx.setLineDash([6, 3]);
    _nodes.ctx.lineWidth = 1;
    _nodes.ctx.strokeStyle = "#333";
    _nodes.ctx.beginPath();
    // 横线
    _nodes.ctx.moveTo(0, this.mouseY);
    _nodes.ctx.lineTo(this.cWidth, this.mouseY);
    _nodes.ctx.stroke();
    // 纵线
    _nodes.ctx.moveTo(this.mouseX, 0);
    _nodes.ctx.lineTo(this.mouseX, this.cHeight);
    _nodes.ctx.stroke();
    _nodes.ctx.closePath();
  };

  //----鼠标跟随十字线开关按钮操作函数
  CrossHairSwitch = () => {
    let _nodes = this.Nodes;
    if (_nodes.crossLine.className.indexOf("focus") > -1) {
      _nodes.crossLine.childNodes[1].checked = false;
      _nodes.crossLine.classList.remove("focus");
      this.SetFeatures("crossOn", false);
      _nodes.canvas.removeEventListener(
        "mousemove",
        this.MouseMoveCrossHairLocation
      );
    } else {
      _nodes.crossLine.childNodes[1].checked = true;
      _nodes.crossLine.classList.add("focus");
      this.SetFeatures("crossOn", true);
      _nodes.canvas.addEventListener(
        "mousemove",
        this.MouseMoveCrossHairLocation
      );
    }
  };

  //----鼠标移动十字线定位函数
  MouseMoveCrossHairLocation = () => {
    // 更新鼠标当前位置十字线
    if (this.Features.crossOn) {
      this.DrawSavedAnnotateInfoToShow();
      this.MouseMoveCrossHair();
    }
  };

  //----监听画板鼠标移动
  CanvasMouseMove = (e) => {
    let _nodes = this.Nodes;
    let _arrays = this.Arrays;
    this.GetMouseInCanvasLocation(e);
    if (_arrays.resultIndex !== 0) {
      let imageIndexShow =
        _arrays.imageAnnotateShower[_arrays.resultIndex - 1].content;
      if (imageIndexShow.length > 0) {
        for (let i = 0; i < imageIndexShow.length; i++) {
          // 使用勾股定理计算鼠标当前位置是否处于当前点上
          let distanceFromCenter = Math.sqrt(
            Math.pow(imageIndexShow[i].x - this.mouseX, 2) +
            Math.pow(imageIndexShow[i].y - this.mouseY, 2)
          );
          // 改变圆点颜色动画
          if (distanceFromCenter <= this.radius) {
            _nodes.canvas.style.cursor = "grabbing";
            return;
          } else {
            _nodes.canvas.style.cursor = "crosshair";
          }
        }
      }
    }
  };

  //----监听画板鼠标点击
  CanvasMouseDown = (e) => {
    let _nodes = this.Nodes;
    let _arrays = this.Arrays;
    this.GetMouseInCanvasLocation(e);
    if (e.button === 0) {
      if (_arrays.resultIndex !== 0) {
        let imageIndex =
          _arrays.imageAnnotateShower[_arrays.resultIndex - 1].content;
        if (imageIndex.length > 0) {
          for (let i = 0; i < imageIndex.length; i++) {
            // 使用勾股定理计算鼠标当前位置是否处于当前点上
            let distanceFromCenter = Math.sqrt(
              Math.pow(imageIndex[i].x - this.mouseX, 2) +
              Math.pow(imageIndex[i].y - this.mouseY, 2)
            );
            if (distanceFromCenter <= this.radius) {
              this.snapCircleIndex = i;
              if (
                _arrays.imageAnnotateShower[_arrays.resultIndex - 1]
                  .contentType === "rect"
              ) {
                this.Nodes.canvas.addEventListener(
                  "mousemove",
                  this.DragRectCircleRepaintRect
                );
                this.Nodes.canvas.addEventListener(
                  "mouseup",
                  this.RemoveDragRectCircle
                );
              } else if (
                _arrays.imageAnnotateShower[_arrays.resultIndex - 1]
                  .contentType === "polygon"
              ) {
                this.Nodes.canvas.addEventListener(
                  "mousemove",
                  this.CircleDrag
                );
                this.Nodes.canvas.addEventListener(
                  "mouseup",
                  this.RemoveCircleDrag
                );
              }
              return;
            }
          }
        }
      }

      if (this.Features.dragOn) {
        // 是否开启拖拽模式
        let prevP = this.CalculateChange(e, _nodes.canvas);
        this.prevX = prevP.x;
        this.prevY = prevP.y;
        _nodes.canvas.addEventListener("mousemove", this.ImageDrag);
        _nodes.canvas.addEventListener("mouseup", this.RemoveImageDrag);
        return;
      }
      if (this.Features.rectOn) {
        // 是否开启绘制矩形功能
        this.SetFeatures("rectOn", true);
        // 判断是否点击了某标注结果
        let clicked = false;
        for (let i = 0; i < _arrays.imageAnnotateShower.length; i++) {
          // 对于每个多边形
          let polygonPoints = _arrays.imageAnnotateShower[i].content;
          let point = {
            x: this.mouseX,
            y: this.mouseY,
          };
          if (this.InsidePolygon(point, polygonPoints)) {
            _arrays.resultIndex = i + 1;
            this.selectAnnotation(i);
            clicked = true;
            break;
          }
        }
        if (clicked) {
        }
        if (this.Arrays.resultIndex === 0) {
          _nodes.ctx.lineWidth = 1;
          _nodes.ctx.strokeStyle = "#ff0000";
          _nodes.ctx.fillStyle = "rgba(255,0,0," + this.opacity + ")";
          this.rectX = this.mouseX;
          this.rectY = this.mouseY;
          this.Nodes.canvas.addEventListener(
            "mousemove",
            this.MouseMoveDrawRect
          );
          this.Nodes.canvas.addEventListener(
            "mouseup",
            this.MouseUpRemoveDrawRect
          );
        }
        return;
      }
      if (this.Features.polygonOn) {
        // 是否开启绘制多边形功能
        let resultList =
          _nodes.resultGroup.getElementsByClassName("result_list");
        let isActive = false;
        for (let i = 0; i < resultList.length; i++) {
          // 循环结果列表判断是否点击某一个结果，若是，则改变焦点
          if (resultList[i].className.indexOf("active") > -1) {
            _arrays.resultIndex = resultList[i].id;
            isActive = true;
          }
        }
        if (!isActive) {
          _arrays.resultIndex = 0;
        }
        if (_arrays.resultIndex === 0) {
          // 未选定标签结果，创建新标签
          this.CreateNewResultList(this.mouseX, this.mouseY, "polygon");
        }
        let index = _arrays.resultIndex - 1;
        // 保存坐标点
        _arrays.imageAnnotateShower[index].content.push({
          x: this.mouseX,
          y: this.mouseY,
        });
        this.CalcRectMask(_arrays.imageAnnotateShower[index].content);
        this.ReplaceAnnotateMemory();
        this.DrawSavedAnnotateInfoToShow();
        this.RecordOperation(
          "addPoint",
          "添加坐标点",
          index,
          JSON.stringify(_arrays.imageAnnotateMemory[index])
        );
        return;
      }
      if (this.Features.maskOn) {
        // 是否开启掩膜标注功能
      }
    } else if (e.button === 2) {
      // 长按右击直接开启拖拽模式
      let prevP = this.CalculateChange(e, _nodes.canvas);
      this.prevX = prevP.x;
      this.prevY = prevP.y;
      _nodes.canvas.addEventListener("mousemove", this.ImageDrag);
      _nodes.canvas.addEventListener("mouseup", this.RemoveImageDrag);
    }
  };

  //----判断点是否在多边形内: https://blog.csdn.net/hjh2005/article/details/9246967
  InsidePolygon = (point, polyPoints) => {
    let crossLineNum = 0;
    for (let j = 0, k = 1; j < polyPoints.length; j++, k++) {
      if (k == polyPoints.length) {
        k = 0;
      }
      let startPoint = polyPoints[j],
        endPoint = polyPoints[k];
      // 只取点左侧线判断
      if (startPoint.x > point.x && endPoint.x > point.x) {
        continue;
      }
      if (
        (startPoint.y > point.y && endPoint.y < point.y) ||
        (startPoint.y < point.y && endPoint.y >= point.y)
      ) {
        crossLineNum++;
      }
    }
    // 点左侧与奇数条边相交，则在此多边形内
    return crossLineNum % 2 == 1;
  };

  //----通过已保存的坐标点计算矩形蒙层位置与大小，以及标签位置, 添加至数组列表中
  CalcRectMask = (arrays) => {
    if (arrays.length >= 2) {
      // 保存边缘矩形框坐标点
      let xMin = arrays[0].x,
        xMax = arrays[0].x,
        yMin = arrays[0].y,
        yMax = arrays[0].y;
      arrays.forEach((item) => {
        xMin = xMin < item.x ? xMin : item.x;
        xMax = xMax > item.x ? xMax : item.x;
        yMin = yMin < item.y ? yMin : item.y;
        yMax = yMax > item.y ? yMax : item.y;
      });
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].rectMask = {
        xMin: xMin,
        yMin: yMin,
        width: xMax - xMin,
        height: yMax - yMin,
      };
      // 计算已创建的标签居中显示
      let labelX = (xMax - xMin) / 2 + xMin;
      let labelY = (yMax - yMin) / 2 + yMin;
      this.Arrays.imageAnnotateShower[
        this.Arrays.resultIndex - 1
      ].labelLocation.x = labelX;
      this.Arrays.imageAnnotateShower[
        this.Arrays.resultIndex - 1
      ].labelLocation.y = labelY;
    }
  };

  //----绘制矩形的方法
  DrawRect = (ctx, x, y, width, height, color, rgb) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(" + rgb + "," + this.opacity + ")";
    ctx.strokeRect(x, y, width, height);
    ctx.fillRect(x, y, width, height);
  };

  //----绘制圆点的方法
  DrawCircle = (ctx, x, y, color) => {
    ctx.beginPath();
    ctx.fillStyle = "#000";
    ctx.arc(x, y, this.radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, this.radius / 3, 0, 2 * Math.PI);
    ctx.fill();
  };

  //----绘制标签的方法
  DrawRectLabel = (ctx, x, y, color, name, index) => {
    ctx.font = "12px Verdana";
    let txtWidth = ctx.measureText(name).width;
    ctx.fillStyle = "rgba(255,255,255, 0.7)";
    ctx.fillRect(x - txtWidth / 2 - 8, y - 10, txtWidth + 16, 20);
    ctx.fillStyle = color;
    ctx.fillText(name, x - txtWidth / 2, y + 4);
  };

  //----绘制已保存的标定信息（在数据操作更新时渲染）绘至数据展示画板
  DrawSavedAnnotateInfoToShow = (resultIndex) => {
    let _arrays = this.Arrays;
    let _nodes = this.Nodes;
    _nodes.ctx.clearRect(0, 0, this.cWidth, this.cHeight);
    _nodes.ctx.drawImage(
      _nodes.bCanvas,
      -this.x / this.scale,
      -this.y / this.scale,
      this.cWidth / this.scale,
      this.cHeight / this.scale,
      0,
      0,
      this.cWidth,
      this.cHeight
    );
    _nodes.ctx.setLineDash([0, 0]);
    _arrays.imageAnnotateShower.forEach((item, index) => {
      if (item.contentType === "polygon") {
        // 绘制闭合线条
        _nodes.ctx.beginPath();
        _nodes.ctx.lineWidth = this.lineWidth;
        _nodes.ctx.moveTo(item.content[0].x, item.content[0].y);
        item.content.forEach((line) => {
          _nodes.ctx.lineTo(line.x, line.y);
        });
        _nodes.ctx.fillStyle =
          "rgba(" + item.labels.labelColorRGB + "," + this.opacity + ")";
        _nodes.ctx.fill();
        _nodes.ctx.strokeStyle =
          "rgba(" + item.labels.labelColorRGB + "," + this.opacity + ")";
        _nodes.ctx.stroke();
      } else if (item.contentType === "rect") {
        this.DrawRect(
          _nodes.ctx,
          item.rectMask.xMin,
          item.rectMask.yMin,
          item.rectMask.width,
          item.rectMask.height,
          item.labels.labelColor,
          item.labels.labelColorRGB
        );
      }
      if (_arrays.resultIndex !== 0 && _arrays.resultIndex - 1 === index) {
        item.content.forEach((circle) => {
          // 绘制圆点
          this.DrawCircle(_nodes.ctx, circle.x, circle.y, "#20c3f9");
        });
      }
      if (item.content.length >= 2 && item.labels.visibility) {
        // 绘制标签
        this.DrawRectLabel(
          _nodes.ctx,
          item.labelLocation.x,
          item.labelLocation.y,
          item.labels.labelColor,
          item.labels.labelName,
          index + 1
        );
      }
      // 绘制矩形蒙层
      if (resultIndex && resultIndex - 1 === index) {
        _nodes.ctx.beginPath();
        _nodes.ctx.lineWidth = 2;
        _nodes.ctx.strokeStyle = "#fffd4d";
        _nodes.ctx.fillStyle = "rgba(255, 253, 77, 0.3)";
        _nodes.ctx.strokeRect(
          item.rectMask.xMin,
          item.rectMask.yMin,
          item.rectMask.width,
          item.rectMask.height
        );
        _nodes.ctx.fillRect(
          item.rectMask.xMin,
          item.rectMask.yMin,
          item.rectMask.width,
          item.rectMask.height
        );
        _nodes.ctx.closePath();
      }
    });
  };

  //----绘制已保存的标定信息（只在拖拽和缩放画布时渲染）绘画至数据存储面板
  DrawSavedAnnotateInfoToMemory = (isRender) => {
    let _arrays = this.Arrays;
    let _nodes = this.Nodes;
    _nodes.bCtx.clearRect(0, 0, this.iWidth, this.iHeight);
    _nodes.bCtx.drawImage(_nodes.image, 0, 0, this.iWidth, this.iHeight);
    if (isRender) {
      _arrays.imageAnnotateMemory.forEach((item, index) => {
        if (item.contentType === "polygon") {
          // 绘制闭合线条
          _nodes.bCtx.beginPath();
          _nodes.bCtx.lineWidth = this.lineWidth;
          _nodes.bCtx.moveTo(item.content[0].x, item.content[0].y);
          item.content.forEach((line) => {
            _nodes.bCtx.lineTo(line.x, line.y);
          });
          _nodes.bCtx.fillStyle =
            "rgba(" + item.labels.labelColorRGB + "," + this.opacity + ")";
          _nodes.bCtx.fill();
          _nodes.bCtx.strokeStyle =
            "rgba(" + item.labels.labelColorRGB + "," + this.opacity + ")";
          _nodes.bCtx.stroke();
        } else if (item.contentType === "rect") {
          this.DrawRect(
            _nodes.bCtx,
            item.rectMask.xMin,
            item.rectMask.yMin,
            item.rectMask.width,
            item.rectMask.height,
            item.labels.labelColor,
            item.labels.labelColorRGB
          );
        }
        if (_arrays.resultIndex !== 0 && _arrays.resultIndex - 1 === index) {
          item.content.forEach((circle) => {
            // 绘制圆点
            this.DrawCircle(_nodes.bCtx, circle.x, circle.y, "#20c3f9");
          });
        }
        if (item.content.length >= 2 && item.labels.visibility) {
          // 绘制标签
          this.DrawRectLabel(
            _nodes.bCtx,
            item.labelLocation.x,
            item.labelLocation.y,
            item.labels.labelColor,
            item.labels.labelName,
            index + 1
          );
        }
      });
    }
    this.UpdateCanvas();
    !isRender && this.DrawSavedAnnotateInfoToShow();
  };

  //----圆点拖拽事件，并且重新绘制边缘轨迹点
  CircleDrag = (e) => {
    this.GetMouseInCanvasLocation(e);
    let imageIndex =
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].content;
    imageIndex[this.snapCircleIndex].x = this.mouseX;
    imageIndex[this.snapCircleIndex].y = this.mouseY;
    this.DrawSavedAnnotateInfoToShow();
  };

  //----移除圆点拖拽事件, 并重新绘制一遍最新状态
  RemoveCircleDrag = () => {
    let index = this.Arrays.resultIndex - 1;
    this.Nodes.canvas.removeEventListener("mousemove", this.CircleDrag);
    this.Nodes.canvas.removeEventListener("mouseup", this.RemoveCircleDrag);
    // 移除圆点拖拽事件之后，改变被拖拽圆点在矩形蒙层数据中的坐标
    this.CalcRectMask(this.Arrays.imageAnnotateShower[index].content);
    this.DrawSavedAnnotateInfoToShow();
    this.ReplaceAnnotateMemory();
    this.RecordOperation(
      "modify",
      "拖拽更新多边形边缘点",
      index,
      JSON.stringify(this.Arrays.imageAnnotateMemory[index])
    );
  };

  //----矩形拖拽事件
  RectDrag = (e) => {
    let _nodes = this.Nodes;
    let p = this.CalculateChange(e, _nodes.canvas);
    let imageIndex =
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].content;
    for (let point of imageIndex) {
      point.x -= p.x;
      point.y -= p.y;
    }
    this.DrawSavedAnnotateInfoToShow();
  };
  //----矩形拖拽事件
  RemoveRectDrag = (e) => {
    let _nodes = this.Nodes;
    let p = this.CalculateChange(e, _nodes.canvas);
    let imageIndex =
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].content;
    for (let point of imageIndex) {
      point.x -= p.x;
      point.y -= p.y;
    }
    this.DrawSavedAnnotateInfoToShow();
  };

  //----图片拖拽事件函数
  ImageDrag = (e) => {
    let _nodes = this.Nodes;
    let p = this.CalculateChange(e, _nodes.canvas);
    let offsetX = p.x - this.prevX;
    let offsetY = p.y - this.prevY;
    this.SetXY(this.x + offsetX, this.y + offsetY);
    this.prevX = p.x;
    this.prevY = p.y;
    if (this.drawFlag) {
      this.DrawSavedAnnotateInfoToMemory(true);
      this.drawFlag = false;
    }
  };

  //----移除鼠标拖拽图片事件函数, 并将最新数据绘制到存储面板中
  RemoveImageDrag = () => {
    this.ReplaceAnnotateShow();
    this.DrawSavedAnnotateInfoToMemory(false);
    this.drawFlag = true;
    this.Nodes.canvas.removeEventListener("mousemove", this.ImageDrag);
    this.Nodes.canvas.removeEventListener("mouseup", this.RemoveImageDrag);
  };

  //----鼠标移动绘制矩形事件
  MouseMoveDrawRect = (e) => {
    this.GetMouseInCanvasLocation(e);
    this.DrawSavedAnnotateInfoToShow();
    this.Nodes.ctx.strokeStyle = "#ff0000";
    this.Nodes.ctx.fillStyle = "rgba(255,0,0," + this.opacity + ")";
    this.Nodes.ctx.strokeRect(
      this.rectX,
      this.rectY,
      this.mouseX - this.rectX,
      this.mouseY - this.rectY
    );
    this.Nodes.ctx.fillRect(
      this.rectX,
      this.rectY,
      this.mouseX - this.rectX,
      this.mouseY - this.rectY
    );
  };

  //----绘制矩形时鼠标抬起后移除监听函数
  MouseUpRemoveDrawRect = () => {
    if (this.mouseX - this.rectX >= 5 || this.rectX - this.mouseX >= 5) {
      // 判断矩形绘制距离大于五才认定为有效绘制
      // 保存绘图数据
      this.CreateNewResultList(this.mouseX, this.mouseY, "rect");
      this.DrawSavedAnnotateInfoToShow();
      this.ReplaceAnnotateMemory();
      let index = this.Arrays.resultIndex - 1;
      this.RecordOperation(
        "add",
        "绘制矩形框",
        index,
        JSON.stringify(this.Arrays.imageAnnotateMemory[index])
      );
    }
    this.Nodes.canvas.removeEventListener("mousemove", this.MouseMoveDrawRect);
    this.Nodes.canvas.removeEventListener(
      "mouseup",
      this.MouseUpRemoveDrawRect
    );
  };

  //----拖拽矩形圆点时改变矩形十个点坐标
  DragRectCircleChangeLocation = (content, circleIndex) => {
    switch (circleIndex) {
      case 0:
        content[1].y = this.mouseY;
        content[3].x = this.mouseX;
        break;
      case 1:
        content[0].y = this.mouseY;
        content[2].x = this.mouseX;
        break;
      case 2:
        content[1].x = this.mouseX;
        content[3].y = this.mouseY;
        break;
      case 3:
        content[0].x = this.mouseX;
        content[2].y = this.mouseY;
        break;
      default:
        break;
    }
  };

  //----拖拽矩形圆点时重新绘制矩形事件
  DragRectCircleRepaintRect = (e) => {
    this.GetMouseInCanvasLocation(e);
    let imageIndex =
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].content;
    this.Nodes.ctx.fillStyle =
      "rgba(" +
      this.Arrays.imageAnnotateShower[this.Arrays.resultIndex - 1].labels
        .labelColorRGB +
      "," +
      this.opacity +
      ")";
    imageIndex[this.snapCircleIndex].x = this.mouseX;
    imageIndex[this.snapCircleIndex].y = this.mouseY;
    this.DragRectCircleChangeLocation(imageIndex, this.snapCircleIndex);
    this.CalcRectMask(imageIndex);
    this.DrawSavedAnnotateInfoToShow();
  };

  //----移除矩形圆点拖拽事件，并将最新数据绘制到存储面板中
  RemoveDragRectCircle = () => {
    this.ReplaceAnnotateMemory();
    this.DrawSavedAnnotateInfoToMemory(false);
    this.drawFlag = true;
    this.Nodes.canvas.removeEventListener(
      "mousemove",
      this.DragRectCircleRepaintRect
    );
    this.Nodes.canvas.removeEventListener("mouseup", this.RemoveDragRectCircle);
    let index = this.Arrays.resultIndex - 1;
    this.RecordOperation(
      "modify",
      "拖拽更新矩形框",
      index,
      JSON.stringify(this.Arrays.imageAnnotateMemory[index])
    );
  };

  //----重新绘制已保存的图像标注记录与标签（删除修改之后重新渲染整体模块）
  RepaintResultList = () => {
    // 先清空标签, 之后再重新渲染
    this.Arrays.resultIndex = 0;
    this.DrawSavedAnnotateInfoToShow();
    this.ReplaceAnnotateMemory();
    this.Nodes.resultGroup.innerHTML = "";
    if (this.Arrays.imageAnnotateShower.length > 0) {
      let _index = 0;
      this.Arrays.imageAnnotateShower.forEach((item, index) => {
        // 创建结果标签
        _index = ++index;
        let eyeIconClass = item.labels.visibility
          ? "icon-eye-open"
          : "icon-eye-close";
        let resultListBody = document.createElement("div");
        resultListBody.className = "result_list";
        resultListBody.id = _index;
        resultListBody.innerHTML =
          '<span class="result_no">' +
          _index +
          "</span>" +
          '<span class="result_color" style="background: ' +
          item.labels.labelColor +
          ';"></span>' +
          '<input class="result_Name" value="' +
          item.labels.labelName +
          '" disabled>' +
          '<i class="editLabelName icon-pencil"></i>' +
          '<i class="deleteLabel icon-trash"></i>' +
          '<i class="isShowLabel ' +
          eyeIconClass +
          '"></i>';
        this.Nodes.resultGroup.appendChild(resultListBody);
      });
      document.querySelector(".resultLength").innerHTML = _index;
    }
  };

  //----创建新的标定结果标签
  CreateNewResultList = (lx, ly, contentType) => {
    let _nodes = this.Nodes;
    let _arrays = this.Arrays;
    let eyeIconClass = _nodes.labelShower ? "icon-eye-open" : "icon-eye-close";
    let resultLength = document.querySelectorAll(".result_list").length + 1;
    let resultListBody = document.createElement("div");
    resultListBody.className = "result_list active";
    resultListBody.id = resultLength;
    resultListBody.innerHTML =
      '<span class="result_no">' +
      resultLength +
      "</span>" +
      '<span class="result_color"></span>' +
      '<input class="result_Name" value="未命名" disabled>' +
      '<i class="editLabelName icon-pencil"></i>' +
      '<i class="deleteLabel icon-trash"></i>' +
      '<i class="isShowLabel ' +
      eyeIconClass +
      '"></i>';
    _nodes.resultGroup.appendChild(resultListBody);

    // 轮询获取当前ResultIndex;
    let resultList = _nodes.resultGroup.getElementsByClassName("result_list");
    for (let i = 0; i < resultList.length; i++) {
      if (resultList[i].className.indexOf("active") > -1) {
        _arrays.resultIndex = resultList[i].id;
      }
    }

    if (contentType === "rect") {
      let rectMask = {
        xMin: this.rectX,
        yMin: this.rectY,
        width: this.mouseX - this.rectX,
        height: this.mouseY - this.rectY,
      };
      this.Arrays.imageAnnotateShower.push({
        content: [
          { x: this.rectX, y: this.rectY },
          { x: this.mouseX, y: this.rectY },
          { x: this.mouseX, y: this.mouseY },
          { x: this.rectX, y: this.mouseY },
        ],
        labels: {
          labelName: "未命名",
          labelColor: "red",
          labelColorRGB: "255,0,0",
          visibility: _nodes.labelShower,
        },
        labelLocation: this.ComputerLabelLocation(rectMask),
        rectMask,
        contentType: contentType,
      });
      this.ReplaceAnnotateMemory();
    } else if (contentType === "polygon") {
      this.Arrays.imageAnnotateShower.push({
        labels: {
          labelName: "未命名",
          labelColor: "red",
          labelColorRGB: "255,0,0",
          visibility: _nodes.labelShower,
        },
        labelLocation: {
          x: lx,
          y: ly,
        },
        contentType: contentType,
        content: [],
        rectMask: {},
      });
    }
    document.querySelector(".resultLength").innerHTML = resultLength;
  };

  //----删除某个已标定结果标签
  DeleteSomeResultLabel = (index) => {
    this.ReplaceAnnotateMemory();
    this.RecordOperation(
      "delete",
      "删除标定标签",
      index,
      JSON.stringify(this.Arrays.imageAnnotateMemory[index])
    );
    this.Arrays.imageAnnotateShower.splice(index, 1);
    this.RepaintResultList();
  };

  //----已标定结果列表交互操作
  ResultListOperation = () => {
    let _self = this;
    let resultList =
      this.Nodes.resultGroup.getElementsByClassName("result_list");
    for (let i = 0; i < resultList.length; i++) {
      resultList[i].index = i;
      resultList[i].onmouseover = function () {
        let hoverIndex = resultList[this.index].id;
        _self.DrawSavedAnnotateInfoToShow(hoverIndex);
      };
      resultList[i].onmouseout = function () {
        _self.DrawSavedAnnotateInfoToShow();
      };
      resultList[i].onclick = function (event) {
        let target = event.target;
        let pageY = event.pageY + 21;
        switch (target.classList[0]) {
          case "deleteLabel":
            _self.DeleteSomeResultLabel(i);
            break;
          case "editLabelName":
            _self.getCreatedLabels(resultList[i], pageY, i);
            break;
          case "result_Name":
            _self.selectAnnotation(i);
            break;
          case "isShowLabel":
            if (target.classList.value.indexOf("icon-eye-open") > -1) {
              target.className = "isShowLabel icon-eye-close";
              _self.Arrays.imageAnnotateShower[
                this.index
              ].labels.visibility = false;
            } else {
              target.className = "isShowLabel icon-eye-open";
              _self.Arrays.imageAnnotateShower[
                this.index
              ].labels.visibility = true;
            }
            _self.DrawSavedAnnotateInfoToShow();
            break;
          default:
            break;
        }
      };
    }
  };

  selectAnnotation = (i) => {
    let resultList =
      this.Nodes.resultGroup.getElementsByClassName("result_list");
    for (let j = 0; j < resultList.length; j++) {
      resultList[j].classList.remove("active");
    }
    resultList[i].classList.add("active");
    this.Arrays.resultIndex = i + 1;
    this.DrawSavedAnnotateInfoToShow();
  };

  //----获取已经创建的标签列表
  getCreatedLabels = async (node, pageY, resultIndex) => {
    let _self = this;
    let resultSelectLabel = document.querySelector(".resultSelectLabel");
    let selectLabelUL = resultSelectLabel.querySelector(".selectLabel-ul"); //标签选择UL
    let closeLabel = resultSelectLabel.querySelector(".closeLabelManage");
    let selectLabelTip = resultSelectLabel.querySelector(".selectLabelTip");
    //加载标签数据
    selectLabelUL.innerHTML = "";
    let labels = await Backend.getTags();
    labels = labels.data.data;
    if (Object.keys(labels).length > 0) {
      selectLabelTip.style.display = "none";
      let fragment = document.createDocumentFragment();
      Object.entries(labels).forEach(entry => {
        const [index, item] = entry;
        let labelLi = document.createElement("li");
        labelLi.innerText = item.labelName;
        labelLi.value = item.labelColor;
        labelLi.setAttribute("data-index", index);
        labelLi.setAttribute("data-r", item.labelColorR);
        labelLi.setAttribute("data-g", item.labelColorG);
        labelLi.setAttribute("data-b", item.labelColorB);
        labelLi.style.color = item.labelColor;
        labelLi.style.borderColor = item.labelColor;
        fragment.appendChild(labelLi);
        labelLi.onmouseover = function () {
          labelLi.style.color = "#fff";
          labelLi.style.background = item.labelColor;
        };
        labelLi.onmouseleave = function () {
          labelLi.style.color = item.labelColor;
          labelLi.style.background = "transparent";
        };
        labelLi.onclick = function () {
          _self.Arrays.imageAnnotateShower[resultIndex].labels.labelName =
            item.labelName;
          _self.Arrays.imageAnnotateShower[resultIndex].labels.labelColor =
            item.labelColor;
          _self.Arrays.imageAnnotateShower[resultIndex].labels.labelColorRGB =
            item.labelColorR + "," + item.labelColorG + "," + item.labelColorB;
          resultSelectLabel.classList.remove("focus");
          resultSelectLabel.classList.add("blur");
          _self.Arrays.resultIndex = 0;
          _self.RepaintResultList();
          _self.RecordOperation(
            "modify",
            "修改标签",
            resultIndex,
            JSON.stringify(_self.Arrays.imageAnnotateMemory[resultIndex])
          );
        };
      });
      selectLabelUL.appendChild(fragment);
    } else {
      selectLabelTip.style.display = "block";
    }
    // 判断是否显示标签管理
    if (resultSelectLabel.className.indexOf("focus") === -1) {
      resultSelectLabel.classList.remove("blur");
      resultSelectLabel.classList.add("focus");
      resultSelectLabel.style.top = pageY + "px";
    } else {
      resultSelectLabel.classList.remove("focus");
      resultSelectLabel.classList.add("blur");
    }

    // 关闭标签管理
    closeLabel.onclick = function () {
      resultSelectLabel.classList.remove("focus");
      resultSelectLabel.classList.add("blur");
    };
  };

  //----标签管理
  ManageLabels = async () => {
    let labelSearch = document.querySelector(".labelSearch-input"); // 标签搜索
    let labelManage = document.querySelector(".labelManage"); // 标签管理父节点
    let closeLabel = labelManage.querySelector(".closeLabelManage"); // 关闭标签管理窗口节点
    let labelManegeUL = labelManage.querySelector(".labelManage-ul"); // 标签管理列表父节点
    let labelManageInfo = labelManage.querySelector(".labelManage-Info"); // 标签列表模块
    let labelManageCreateInfo = labelManage.querySelector(
      ".labelManage-create"
    ); // 标签编辑模块
    let addLabel = labelManage.querySelector(".addLabel");
    let addLabelName = labelManage.querySelector(".labelCreate-nameInput");
    let addLabelColor = labelManage.querySelector("#colorHex");
    let labelManageCreate = labelManage.querySelector(
      ".labelManage-createButton"
    ); // 添加标签节点
    let closeAdd = labelManage.querySelector(".closeAdd"); // 取消添加标签
    let removeLabel = labelManage.querySelector(".removeLabel"); // 删除标签
    let colorPicker = labelManage.querySelector("#colorPicker"); // 选取颜色模块
    let labelTip = labelManage.querySelector(".labelTip"); // 标签提示
    let input = document.getElementById("colorHex");
    let labelManageTitle = document.querySelector(".labelManage-Title");
    let flag = false;
    let flagIndex = 0;
    let labels = [];
    drawLabels();
    async function drawLabels() {
      let labelsResponse = await Backend.getTags();
      if (labelsResponse.data.data && Object.keys(labelsResponse.data.data).length) {
        labels = labelsResponse.data.data;
        eachLabels(labels);
      } else {
        labelTip.style.display = "block";
      }
    }
    function eachLabels(labelList) {
      //加载标签数据
      labelTip.style.display = "none";
      labelManegeUL.innerHTML = "";
      let fragment = document.createDocumentFragment();
      Object.entries(labelList).forEach(entry => {
        const [index, item] = entry;
        let labelLi = document.createElement("li");
        labelLi.innerText = item.labelName;
        labelLi.value = item.labelName;
        labelLi.setAttribute("data-index", index);
        labelLi.setAttribute("data-r", item.labelColorR);
        labelLi.setAttribute("data-g", item.labelColorG);
        labelLi.setAttribute("data-b", item.labelColorB);
        labelLi.style.color = item.labelColor;
        labelLi.style.borderColor = item.labelColor;
        fragment.appendChild(labelLi);

        labelLi.onmouseover = function () {
          labelLi.style.color = "#fff";
          labelLi.style.background = item.labelColor;
        };
        labelLi.onmouseleave = function () {
          labelLi.style.color = item.labelColor;
          labelLi.style.background = "transparent";
        };
        labelLi.onclick = function () {
          addLabelName.value = item.labelName;
          colorPicker.style.background = item.labelColor;
          input.value = item.labelColor;
          flag = true;
          flagIndex = index;
          labelManageTitle.innerText = "编辑标签";
          labelManageInfo.style.display = "none";
          labelManageCreateInfo.style.display = "block";
          removeLabel.style.display = "block";
        };
      });
      labelManegeUL.appendChild(fragment);
    }
    // 添加标签事件
    labelManageCreate.onclick = function () {
      flag = false;
      labelManageTitle.innerText = "创建标签";
      addLabelName.value = "";
      labelManageInfo.style.display = "none";
      labelManageCreateInfo.style.display = "block";
      removeLabel.style.display = "none";
    };
    closeAdd.onclick = function () {
      labelManageInfo.style.display = "block";
      labelManageCreateInfo.style.display = "none";
      drawLabels();
    };

    removeLabel.onclick = async function () {
      if (confirm('确定删除 "' + addLabelName.value + '" 标签吗？')) {
        labelManageInfo.style.display = "block";
        labelManageCreateInfo.style.display = "none";
        const res = await Backend.deleteTag(addLabelName.value, addLabelColor.value);
        drawLabels();
      }
    };

    colorPicker.onclick = function () {
      let colorDiv = document.querySelector(".colorDiv");
      if (!colorDiv) {
        Colorpicker.create({
          bindClass: "colorPicker",
          change: function (elem, hex, rgb) {
            elem.style.backgroundColor = hex;
            input.value = hex;
            input.setAttribute("data-r", rgb.r);
            input.setAttribute("data-g", rgb.g);
            input.setAttribute("data-b", rgb.b);
          },
        });
        document.querySelector(".colorDiv").style.display = "block";
      }
    };

    addLabel.onclick = async function () {
      if (!!addLabelName.value) {
        if (flag) {
          const res = await Backend.addTag(
            addLabelName.value,
            addLabelColor.value
          );
          alert("修改成功");
          labelManageInfo.style.display = "block";
          labelManageCreateInfo.style.display = "none";
          drawLabels();
        } else {
          const res = await Backend.addTag(addLabelName.value, addLabelColor.value);
          addLabelName.value = "";
          alert("添加成功");
          drawLabels();
        }
      } else {
        alert("请填写标签名称");
      }
    };

    // 判断是否显示标签管理
    if (labelManage.className.indexOf("focus") === -1) {
      labelManage.classList.remove("blur");
      labelManage.classList.add("focus");
    } else {
      labelManage.classList.remove("focus");
      labelManage.classList.add("blur");
    }

    labelSearch.onchange = function (e) {
      let filterLabel = labels.filter((label) => {
        return label.name.indexOf(e.currentTarget.value) > -1;
      });
      eachLabels(filterLabel);
    };

    // 关闭标签管理
    closeLabel.onclick = function () {
      labelManage.classList.remove("focus");
      labelManage.classList.add("blur");
    };
  };

  //----历史记录点击事件
  HistoryClick = (e) => {
    let index = e.target.dataset.index;
    if (index) {
      this.historyIndex = parseInt(index);
      this.RenderHistoryState(this.historyIndex);
    }
  };

  //----渲染至指定历史记录状态
  RenderHistoryState = (index) => {
    let history = this.Arrays.history;
    let historyNodes = this.Nodes.historyGroup.children;
    let prevIndex = -1;
    for (let i = 0; i < historyNodes.length; i++) {
      if (historyNodes[i].classList.value.indexOf("active") > -1) {
        prevIndex = i;
        break;
      }
    }
    // 移除上一个历史记录列表焦点
    prevIndex !== -1 && historyNodes[prevIndex].classList.remove("active");
    this.Arrays.imageAnnotateMemory.splice(
      0,
      this.Arrays.imageAnnotateMemory.length
    );
    for (let i = history.length - 1; i > index; i--) {
      historyNodes[i].classList.add("record");
    }
    for (let i = 0; i <= index; i++) {
      historyNodes[i].classList.remove("record");
      this.HistoryTypeOperation(
        history[i].type,
        history[i].index,
        history[i].content
      );
    }
    historyNodes[index].classList.add("active");
    this.ReplaceAnnotateShow();
    this.RepaintResultList();
  };

  //----历史记录类型判断处理
  HistoryTypeOperation = (type, index, content) => {
    switch (type) {
      case "add":
        this.Arrays.imageAnnotateMemory.splice(index, 0, JSON.parse(content));
        break;
      case "addPoint":
        this.Arrays.imageAnnotateMemory[index] = JSON.parse(content);
        break;
      case "delete":
        this.Arrays.imageAnnotateMemory.splice(index, 1);
        break;
      default:
        this.Arrays.imageAnnotateMemory[index] = JSON.parse(content);
        break;
    }
  };

  //----记录每步操作存储在内存中
  RecordOperation = (type, desc, index, content) => {
    // 渲染到页面上
    if (this.historyIndex < this.Arrays.history.length) {
      this.RenderHistory(type, desc, this.historyIndex + 1);
      this.Arrays.history.splice(
        this.historyIndex + 1,
        this.Arrays.history.length
      );
    } else {
      this.RenderHistory(type, desc, this.historyIndex);
      this.Arrays.history.splice(this.historyIndex, this.Arrays.history.length);
    }
    let historyData = {
      type: type,
      desc: desc,
      index: index,
      content: content,
    };
    this.Arrays.history.push(historyData);
    this.historyIndex++;
  };

  //----将历史记录渲染到页面上
  RenderHistory = (type, desc, index) => {
    let children = this.Nodes.historyGroup.children;
    children.length > 0 && children[index - 1].classList.remove("active");
    for (let i = children.length - 1; i >= 0; i--) {
      children[i].classList.value.indexOf("record") > -1 &&
        this.Nodes.historyGroup.removeChild(children[i]);
    }
    let history = document.createElement("p");
    history.setAttribute("data-type", type);
    history.setAttribute("data-index", index);
    history.innerText = desc;
    history.classList.add("active");
    this.Nodes.historyGroup.appendChild(history);
  };

  //----控制是否显示标签
  IsShowLabels = () => {
    let _nodes = this.Nodes;
    let annotates = this.Arrays.imageAnnotateShower;
    let resultList = document.querySelectorAll(".result_list");
    if (resultList.length > 0) {
      if (_nodes.labelShower.className.indexOf("focus") > -1) {
        // 隐藏标注结果
        _nodes.labelShower.children[0].checked = false;
        _nodes.labelShower.classList.remove("focus");
        resultList.forEach((item, index) => {
          item.childNodes[5].className = "isShowLabel icon-eye-close";
          annotates[index].labels.visibility = false;
        });
      } else {
        // 显示标注结果
        _nodes.labelShower.children[0].checked = true;
        _nodes.labelShower.classList.add("focus");
        resultList.forEach((item, index) => {
          item.childNodes[5].className = "isShowLabel icon-eye-open";
          annotates[index].labels.visibility = true;
        });
      }
    }
    this.DrawSavedAnnotateInfoToShow();
  };

  //----屏幕快照事件
  ScreenShot = () => {
    let imgData = this.Nodes.bCanvas.toDataURL("image/jpeg");
    let windowOpen = window.open("about:blank", "image from canvas");
    windowOpen.document.write("<img alt='' src='" + imgData + "'>");
  };

  //----全屏显示事件
  IsScreenFull = () => {
    if (this.isFullScreen) {
      // 取消全屏显示事件
      let el = document;
      let cfs =
        el.cancelFullScreen ||
        el.webkitCancelFullScreen ||
        el.mozCancelFullScreen ||
        el.msCancelFullScreen;
      if (typeof cfs != "undefined" && cfs) {
        cfs.call(el);
      } else if (typeof window.ActiveXObject != "undefined") {
        // IE 下模拟按下键盘F11事件 取消退出全屏
        let wScript = new ActiveXObject("WScript.Shell");
        if (wScript != null) {
          wScript.SendKeys("{F11}");
        }
      }
      this.Nodes.screenFull.childNodes[3].innerText = "全屏";
      this.Nodes.screenFull.childNodes[1].style.backgroundPosition = "0 -480px";
    } else {
      let el = document.documentElement;
      let rfs =
        el.requestFullScreen ||
        el.webkitRequestFullScreen ||
        el.mozRequestFullScreen ||
        el.msRequestFullScreen;
      if (typeof rfs != "undefined" && rfs) {
        rfs.call(el);
      } else if (typeof window.ActiveXObject != "undefined") {
        // IE 下模拟按下键盘F11事件
        let wScript = new ActiveXObject("WScript.Shell");
        if (wScript != null) {
          wScript.SendKeys("{F11}");
        }
      }
      this.Nodes.screenFull.childNodes[3].innerText = "退出全屏";
      this.Nodes.screenFull.childNodes[1].style.backgroundPosition =
        "5px -480px";
    }
  };

  //----监听浏览器是否全屏, 并调整尺寸
  ScreenViewChange = () => {
    if (
      document.webkitIsFullScreen ||
      document.fullscreen ||
      document.mozFullScreen ||
      document.msFullscreenElement
    ) {
      // 全屏后调整节点尺寸
      let sFullHeight = window.screen.height;
      this.Nodes.canvasMain.style.height = sFullHeight - 60 + "px";
      this.Nodes.canvas.height = this.Nodes.canvasMain.offsetHeight;
      this.cHeight = this.Nodes.canvasMain.offsetHeight;
      this.UpdateCanvas();
      this.isFullScreen = true;
    } else {
      // 取消全屏后调整节点尺寸
      let sNormalHeight = window.innerHeight;
      this.Nodes.canvasMain.style.height = sNormalHeight - 60 + "px";
      this.Nodes.canvas.height = this.Nodes.canvasMain.offsetHeight;
      this.cHeight = this.Nodes.canvasMain.offsetHeight;
      this.UpdateCanvas();
      this.isFullScreen = false;
    }
  };

  //----缩略图画布点击定位函数
  ScaleCanvasClick = (e) => {
    let p = this.CalculateChange(e, this.Nodes.scaleCanvas);
    let tmpX = this.cWidth / 2 - (this.iWidth * this.scale * p.x) / this.sWidth;
    let tmpY =
      this.cHeight / 2 -
      (((this.iWidth * this.scale * p.x) / this.sWidth) * p.y) / p.x;
    this.SetXY(tmpX, tmpY);
    this.ReplaceAnnotateShow();
  };

  //----滚动条缩放事件
  MouseWheel = (e) => {
    let wd = e.wheelDelta || e.detail;
    let newScale =
      this.scale * (1 + (wd > 0 ? this.scaleStep : -this.scaleStep));
    newScale = newScale < this.minScale ? this.minScale : newScale;
    newScale = newScale > this.maxScale ? this.maxScale : newScale;

    if (newScale !== this.scale) {
      let p = this.CalculateChange(e, this.Nodes.canvas);
      let newX = ((this.x - p.x) * newScale) / this.scale + p.x;
      let newY = ((this.y - p.y) * newScale) / this.scale + p.y;
      this.scale = newScale;
      this.SetXY(newX, newY);
    }
    clearTimeout(this.mousewheelTimer);
    this.mousewheelTimer = setTimeout(() => {
      this.IsMouseWheelEnd();
    }, 500);
    if (this.drawFlag) {
      this.DrawSavedAnnotateInfoToMemory(true);
      this.drawFlag = false;
    }
  };
  //----监听滚动条缩放是否结束
  IsMouseWheelEnd = () => {
    this.ReplaceAnnotateShow();
    this.DrawSavedAnnotateInfoToMemory(false);
    this.drawFlag = true;
  };

  //----设置图片位置，防止图片被拖出画布
  SetXY = (vx, vy) => {
    if (vx < this.appearSize - this.iWidth * this.scale) {
      this.x = this.appearSize - this.iWidth * this.scale;
    } else if (vx > this.cWidth - this.appearSize) {
      this.x = this.cWidth - this.appearSize;
    } else {
      this.x = vx;
    }

    if (vy < this.appearSize - this.iHeight * this.scale) {
      this.y = this.appearSize - this.iHeight * this.scale;
    } else if (vy > this.cHeight - this.appearSize) {
      this.y = this.cHeight - this.appearSize;
    } else {
      this.y = vy;
    }
    this.UpdateCanvas();
  };

  //----Y坐标点装换， 防止绘制到图片外
  YPointReplace = (y) => {
    if (y < this.y) {
      y = this.y;
    } else if (y > this.iHeight * this.scale + this.y) {
      y = this.iHeight * this.scale + this.y;
    }
    return y;
  };
  //----X坐标点装换， 防止绘制到图片外
  XPointReplace = (x) => {
    if (x < this.x) {
      x = this.x;
    } else if (x > this.iWidth * this.scale + this.x) {
      x = this.iWidth * this.scale + this.x;
    }
    return x;
  };

  //----获取更新鼠标在当前展示画板中的位置
  GetMouseInCanvasLocation = (e) => {
    this.mouseX = this.XPointReplace(e.offsetX);
    this.mouseY = this.YPointReplace(e.offsetY);
  };

  //----获取鼠标当前相对所在存储面板图片中的位置
  GetMouseInImageLocation = (location) => {
    let prevP = this.CalculateChange(location, this.Nodes.canvas);
    // 鼠标点击在当前图像的位置
    this.ix = Math.floor((prevP.x - this.x) / this.scale);
    if (this.ix < 0) {
      this.ix = 0;
    } else if (this.ix > this.iWidth) {
      this.ix = this.iWidth;
    }
    this.iy = Math.floor((prevP.y - this.y) / this.scale);
    if (this.iy < 0) {
      this.iy = 0;
    } else if (this.iy > this.iHeight) {
      this.iy = this.iHeight;
    }
  };

  //----计算更新鼠标相对容器的位置
  CalculateChange = (e, container, skip) => {
    !skip && e.preventDefault();
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const x = typeof e.pageX === "number" ? e.pageX : e.touches[0].pageX;
    const y = typeof e.pageY === "number" ? e.pageY : e.touches[0].pageY;
    let left =
      x - (container.getBoundingClientRect().left + window.pageXOffset);
    let top = y - (container.getBoundingClientRect().top + window.pageYOffset);

    if (left < 0) {
      left = 0;
    } else if (left > containerWidth) {
      left = containerWidth;
    }

    if (top < 0) {
      top = 0;
    } else if (top > containerHeight) {
      top = containerHeight;
    }

    return {
      x: left,
      y: top,
    };
  };

  //----计算标签相对于当前标定范围的位置
  ComputerLabelLocation = (rectMask) => {
    let x = rectMask.width / 2 + rectMask.xMin;
    let y = rectMask.height / 2 + rectMask.yMin;
    return { x, y };
  };

  //----按缩放程度修改数据存储面板数据
  ReplaceAnnotateMemory = () => {
    this.Arrays.imageAnnotateMemory.splice(
      0,
      this.Arrays.imageAnnotateMemory.length
    );
    this.Arrays.imageAnnotateShower.map((item) => {
      let content = [];
      item.content.forEach((contents) => {
        content.push({
          x: (contents.x - this.x) / this.scale,
          y: (contents.y - this.y) / this.scale,
        });
      });
      let rectMask = {
        xMin: (item.rectMask.xMin - this.x) / this.scale,
        yMin: (item.rectMask.yMin - this.y) / this.scale,
        width: item.rectMask.width / this.scale,
        height: item.rectMask.height / this.scale,
      };
      this.Arrays.imageAnnotateMemory.push({
        content,
        rectMask,
        labels: item.labels,
        labelLocation: this.ComputerLabelLocation(rectMask),
        contentType: item.contentType,
      });
    });
  };
  //----按缩放程度修改数据展示面板数据
  ReplaceAnnotateShow = () => {
    this.Arrays.imageAnnotateShower.splice(
      0,
      this.Arrays.imageAnnotateShower.length
    );
    this.Arrays.imageAnnotateMemory.map((item, index) => {
      let content = [];
      item.content.forEach((contents) => {
        content.push({
          x: contents.x * this.scale + this.x,
          y: contents.y * this.scale + this.y,
        });
      });
      let rectMask = {
        xMin: item.rectMask.xMin * this.scale + this.x,
        yMin: item.rectMask.yMin * this.scale + this.y,
        width: item.rectMask.width * this.scale,
        height: item.rectMask.height * this.scale,
      };
      this.Arrays.imageAnnotateShower.push({
        content,
        rectMask,
        labels: item.labels,
        labelLocation: this.ComputerLabelLocation(rectMask),
        contentType: item.contentType,
      });
    });
  };

  /*
      画板禁止触发右键菜单事件
     */
  static NoRightMenu(event) {
    event.preventDefault();
  }
}
