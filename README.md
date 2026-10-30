# 青铜器智能导览系统

基于中国国家博物馆青铜器器型分类、重点文物图像匹配、本地知识检索与千问 API 的课程设计原型。支持图片上传、浏览器摄像头单帧识别、展品问答和来源显示；暂不包含语音。

## ⚠️ 本仓库不包含以下内容

出于版权与体积考虑，以下资源**未随仓库发布**，需自行准备：

| 资源 | 路径 | 获取方式 |
|------|------|----------|
| 器型训练图片 | `data/dataset/` | 自行采集或整理 8 类（鼎/簋/爵/觚/尊/卣/盉/壶）青铜器图片 |
| 重点文物图库 | `data/exhibit_gallery/` | 自行采集 8 件文物的多视角图片（每件约 20 张） |
| 器型分类权重 | `models/vessel_classifier/best.pt` | 运行 `scripts/train_classifier.py` 训练生成 |
| API 密钥 | `.env` | 复制 `.env.example` 为 `.env`，填入千问 API Key |
| Hugging Face 模型 | `.model_cache/` | 首次运行自动下载 ResNet18 / Sentence-BERT |

知识库文本（`data/exhibits.json`、`data/vessel_types.json`）与训练结果（`models/vessel_classifier/*.json`、`*.csv`）已包含在仓库中。

## 技术方案

- Python 3.10+
- FastAPI + HTML/CSS/JavaScript
- 第一阶段：微调 MobileNetV3 Small，识别鼎、簋、爵、觚、尊、卣、盉、壶八类器型
- 第二阶段：`microsoft/resnet-18` 图像向量，在 `data/exhibit_gallery/` 的重点文物图库中匹配具体展品
- RAG：Chroma 持久向量库；本地 Sentence Transformers 可用时采用语义向量，否则使用离线中文字符 n-gram 向量
- LLM：阿里云百炼千问 OpenAI 兼容接口

项目不要求部署本地大语言模型。图片和知识库检索由本机处理，只有问答文本与检索到的资料会发送到配置的千问 API。

## 快速开始

1. 安装 Python 3.10 或更高版本。
2. 在项目目录运行 `python -m venv .venv`，再激活虚拟环境。
3. 安装依赖：`python -m pip install -r requirements.txt`
4. 复制 `.env.example` 为 `.env`，填入百炼 API Key、模型 ID 和服务地址。
5. 下载模型（首次运行会自动尝试从 Hugging Face 下载到 `.model_cache/`）：
   - 图像匹配：`microsoft/resnet-18`
   - 中文知识检索：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
6. 按下方步骤准备数据并训练器型分类器。
7. 运行 `python -m uvicorn app.main:app --reload`，浏览器打开 `http://127.0.0.1:8000`。

## 器型分类训练

项目已验证 CUDA 版 PyTorch 可识别 RTX 4060 Laptop GPU。执行：

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset.py
.venv\Scripts\python.exe scripts\train_classifier.py --epochs 30 --batch-size 16
```

最佳权重保存为 `models/vessel_classifier/best.pt`。RTX 4060 Laptop 8GB 建议从 MobileNetV3 Small、224×224、batch size 16 开始；出现显存不足时将 batch size 改为 8。

当前器型数据共 8 类，每类 65 张；重点文物图库共 8 件，每件 20 张。同一器物的多视角不能拆到训练集与测试集两边。若发现 `raw_dataset` 中少量图片属于同一器物，应在正式实验前补充分组清单。

识别回退策略：器型和展品匹配都可靠时介绍具体展品；只有器型可靠时只介绍器型；器型置信度低时要求重新拍摄并展示 Top-3 器型候选。

当前训练结果：器型分类独立测试准确率 91.14%；重点文物保留图 Top-1 为 90.63%、Top-3 为 96.88%。阈值决策回放中，所有进入“介绍具体文物”的样本均识别正确。详见 [docs/TRAINING_RESULTS.md](docs/TRAINING_RESULTS.md)。

模型缓存默认写入项目内的 `.model_cache/`，不会依赖用户目录的模型缓存权限。视觉接口不会在请求期间自动下载大文件；缺少 ResNet18 权重时会立即返回配置提示。

如果模型下载受网络限制，可在 `models/` 下手动放置模型，并在 `.env` 中设置 `VISION_WEIGHTS_PATH` 和 `TEXT_EMBEDDING_MODEL_PATH`。模型缓存目录和手动下载步骤见 [docs/MODELS.md](docs/MODELS.md)。

## 数据与授权

知识库包含 8 件重点文物和 8 类器型通识，主要依据中国国家博物馆官方藏品页面整理。每条记录保留来源 URL。参见 [data/exhibits.json](data/exhibits.json)、[data/vessel_types.json](data/vessel_types.json) 和 [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)。

> **数据集图片不包含在仓库中。** 需自行准备：
> - `data/dataset/{train,val,test}/{ding,gui,jue,gu,zun,you,he,hu}/` — 器型分类图片，每类约 65 张
> - `data/exhibit_gallery/{01_xxx,02_xxx,...}/` — 每件文物 20 张多视角图片
> - `data/exhibit_gallery_split.json` — 图库参考图/测试图划分（已包含）

## 当前功能范围

- 上传图片或拍摄摄像头单帧。
- MobileNetV3 输出器型概率，再在同器型馆藏图库中做 ResNet18 特征相似度检索；图库相似度不是校准概率。
- 按识别展品和用户问题检索本地知识条目。
- 调用千问 API 生成基于检索材料的回答，并返回参考来源。
- 支持围绕当前展品连续追问。

本项目是课程设计演示原型，不提供文物鉴定或学术考证结论。识别失败或分数过低时应由用户确认。
