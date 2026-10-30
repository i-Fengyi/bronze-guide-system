# 本地模型

应用使用两个预训练模型：

- Hugging Face `microsoft/resnet-18`：提取图像特征，用于用户图片与本地展品图库的最近邻匹配。模型约 47 MB，项目当前已保存于 `models/resnet-18/`。
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`：中文/多语言文本向量，用于 RAG 语义检索。

模型缓存默认位于项目 `.model_cache/`。如果需要重新下载 ResNet18：

1. 从 `microsoft/resnet-18` 下载 `config.json`、`preprocessor_config.json` 和 `model.safetensors`。
2. 按原文件名保存到 `models/resnet-18/`。
3. 在 `.env` 设置：

```env
VISION_WEIGHTS_PATH=models/resnet-18
```

如果文本嵌入模型无法自动下载：

1. 手动下载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 的完整模型快照。
2. 保留配置、权重、tokenizer 和子目录结构，放到 `models/paraphrase-multilingual-MiniLM-L12-v2/`。
3. 在 `.env` 设置：

```env
TEXT_EMBEDDING_MODEL_PATH=models/paraphrase-multilingual-MiniLM-L12-v2
```

模型权重体积较大，项目 `.gitignore` 已排除 `models/` 与 `.model_cache/`。
