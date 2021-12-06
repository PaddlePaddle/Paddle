// 设置画布初始属性
const canvasMain = document.querySelector(".canvasMain");
const canvas = document.getElementById("canvas");
const resultGroup = document.querySelector(".resultGroup");

// 设置画布宽高背景色
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

const annotate = new LabelImage({
  canvas: canvas,
  scaleCanvas: document.querySelector(".scaleCanvas"),
  scalePanel: document.querySelector(".scalePanel"),
  annotateState: document.querySelector(".annotateState"),
  canvasMain: canvasMain,
  resultGroup: resultGroup,
  // crossLine: document.querySelector('.crossLine'),
  // 标注结果显示开关
  labelShower: true,
  screenShot: document.querySelector(".screenShot"),
  screenFull: document.querySelector(".screenFull"),
  colorHex: document.querySelector("#colorHex"),
  toolTagsManager: document.querySelector(".toolTagsManager"),
  historyGroup: document.querySelector(".historyGroup"),
});

// 初始化交互操作节点
const prevBtn = document.querySelector(".pagePrev"); // 上一张
const nextBtn = document.querySelector(".pageNext"); // 下一张
const taskName = document.querySelector(".pageName"); // 标注任务名称
const processIndex = document.querySelector(".processIndex"); // 当前标注进度
const processSum = document.querySelector(".processSum"); // 当前标注任务总数
const transferBtn = document.querySelector(".transfer"); // 格式转换

let imgFiles = []; //目录中的文件数据集
let imgIndex = 1; //标定图片默认下标;
let imgSum = 0; // 选择图片总数;

const http = axios.create({
  baseURL: "http://localhost:627",
});
const Backend = {
  getTags: async () => {
    return http.get(`/tag/${annotate.dataset_id}`);
  },
  addTag: async (name, color) => {
    let rgb = hexToRgb(color);
    return http.post(`/tag/${annotate.dataset_id}/add`, {
      labelName: name,
      labelColor: color,
      labelColorR: rgb.r,
      labelColorG: rgb.g,
      labelColorB: rgb.b,
    });
  },
  deleteTag: async (name, color) => {
    return http.post(`/tag/${annotate.dataset_id}/delete`, {
      labelName: name,
      labelColor: color
    });
  },
  uploadDataset: async (label_type, path) => {
    return http.post(`/upload/${label_type}/${annotate.dataset_id}`, {
      pics_path: path,
    });
  },
  getPicURL: (pic_name) => {
    return `/get/picture/${annotate.dataset_id}/${pic_name}`;
  },
  getPic: async (pic_name) => {
    return http.get(Backend.getPicURL(pic_name));
  },
  setAnnotation: async (pic_name, annotations) => {
    return http.post(`/set/annotation/${annotate.dataset_id}/${pic_name}`, annotations);
  },
  getAnnotation: async (pic_name) => {
    return http.get(`/get/annotation/${annotate.dataset_id}/${pic_name}`);
  },
  transfer: async () => {
    return http.get(`/transform/submit/${annotate.dataset_id}`);
  }
};

// 获取URL参数
const UrlParamHash = () => {
  const url = window.location.toString();
  var params = [],
    h;
  var hash = url.slice(url.indexOf("?") + 1).split("&");
  for (var i = 0; i < hash.length; i++) {
    h = hash[i].split("=");
    params[h[0]] = h[1];
  }
  return params;
};
// 16进制转RGB
const hexToRgb = (hex) => {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
};
// 初始化数据集
const InitDataset = async () => {
  let params = UrlParamHash();
  let type = params["type"];
  let path = params["path"];
  let dataset_id = params["dataset_id"];
  if (!type || !path || !dataset_id) {
    alert("数据有误，请关闭本页，回到PaddleX，选择数据集后重新打开。");
  }
  annotate.dataset_id = dataset_id;
  let dataset_info = await Backend.uploadDataset(type, path, dataset_id);
  imgFiles = dataset_info.data.data.image_names;
  imgSum = imgFiles.size;
  initImage();
};
// 初始化数据集
InitDataset();

// 初始化图片状态
function initImage() {
  selectImage(0);
  processSum.innerText = imgSum;
}

//切换操作选项卡
let tool = document.getElementById("tools");
tool.addEventListener("click", function (e) {
  for (let i = 0; i < tool.children.length; i++) {
    tool.children[i].classList.remove("focus");
  }
  e.target.classList.add("focus");
  switch (true) {
    case e.target.className.indexOf("toolDrag") > -1: // 拖拽
      annotate.SetFeatures("dragOn", true);
      break;
    case e.target.className.indexOf("toolRect") > -1: // 矩形
      annotate.SetFeatures("rectOn", true);
      break;
    case e.target.className.indexOf("toolPolygon") > -1: // 多边形
      annotate.SetFeatures("polygonOn", true);
      break;
    case e.target.className.indexOf("toolMask") > -1: // 多边形
      annotate.SetFeatures("polygonOn", true);
      break;
    case e.target.className.indexOf("toolTagsManager") > -1: // 标签管理工具
      annotate.SetFeatures("tagsOn", true);
      break;
    default:
      break;
  }
});

// 获取下一张图片
nextBtn.onclick = function () {
  // 保存已标定的图片信息
  if (annotate.Arrays.imageAnnotateMemory.length > 0) {
    Backend.setAnnotation(taskName.textContent, annotate.Arrays.imageAnnotateMemory);
  }
  if (imgIndex >= imgSum) {
    imgIndex = 1;
    selectImage(0);
  } else {
    imgIndex++;
    selectImage(imgIndex - 1);
  }
};

// 获取上一张图片
prevBtn.onclick = function () {
  // 保存已标定的图片信息
  if (annotate.Arrays.imageAnnotateMemory.length > 0) {
    Backend.setAnnotation(taskName.textContent, annotate.Arrays.imageAnnotateMemory);
  }
  if (imgIndex === 1) {
    imgIndex = imgSum;
    selectImage(imgSum - 1);
  } else {
    imgIndex--;
    selectImage(imgIndex - 1);
  }
};

transferBtn.onclick = async function () {
  try {
    let res = await Backend.transfer();
    alert('转换成功');
  } catch (e) {
    alert(`转换失败！${e}`);
  }
}

function changeFolder(e) {
  imgFiles = e.files;
  imgSum = imgFiles.length;
  processSum.innerText = imgSum;
  imgIndex = 1;
  selectImage(0);
}

function getInitImage(index) {
  return imgFiles[index];
}

async function selectImage(index) {
  openBox("#loading", true);
  processIndex.innerText = imgIndex;
  taskName.innerText = getInitImage(index);
  let imgURL = Backend.getPicURL(imgFiles[index]);
  annotate.SetImage(imgURL, imgFiles[index]);
}

document.querySelector(".saveJson").addEventListener("click", function () {
  let filename = taskName.textContent.split(".")[0] + ".json";
  annotate.Arrays.imageAnnotateMemory.length > 0
    ? saveJson(annotate.Arrays.imageAnnotateMemory, filename)
    : alert("当前图片未有有效的标定数据");
});

function saveJson(data, filename) {
  if (!data) {
    alert("保存的数据为空");
    return false;
  }
  if (!filename) {
    filename = "json.json";
  }
  if (typeof data === "object") {
    data = JSON.stringify(data, undefined, 4);
  }
  let blob = new Blob([data], { type: "text/json" }),
    e = document.createEvent("MouseEvent"),
    a = document.createElement("a");
  a.download = filename;
  a.href = window.URL.createObjectURL(blob);
  a.dataset.downloadurl = ["text/json", a.download, a.href].join(":");
  e.initMouseEvent(
    "click",
    true,
    false,
    window,
    0,
    0,
    0,
    0,
    0,
    false,
    false,
    false,
    false,
    0,
    null
  );
  a.dispatchEvent(e);
}

//弹出框
function openBox(e, isOpen) {
  let el = document.querySelector(e);
  let maskBox = document.querySelector(".mask_box");
  if (isOpen) {
    maskBox.style.display = "block";
    el.style.display = "block";
  } else {
    maskBox.style.display = "none";
    el.style.display = "none";
  }
}

// 倍数缩放
function resize(newScale) {
  let newX = (annotate.x * newScale) / annotate.scale;
  let newY = (annotate.y * newScale) / annotate.scale;
  annotate.scale = newScale;
  annotate.SetXY(newX, newY);
}

// 编辑标签
function toolTagsManager() {
  annotate.SetFeatures("tagsOn", true);
}
