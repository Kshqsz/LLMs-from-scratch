# LLMs from Scratch

从零开始学习和实现大语言模型。

## 🚀 快速开始

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📂 项目进度

### Ch02：文本数据处理 (Working with Text Data)

- ✅ `simpleTokenizer.py` - 简单分词器实现
- ✅ `tokenizer.py` - GPT2分词器测试
- ✅ `dataLoader.py` - PyTorch数据加载器
- ✅ `embedding.py` - 文本嵌入示例
- ✅ `text-prepare.py` - 文本预处理
- ✅ `test.py` - 基础测试

### Ch03：注意力机制 (Coding Attention Mechanisms)

- ✅ `simpleSelfAttention.py` - 简单自注意力机制
- ✅ `causalAttention.py` - 因果注意力实现
- ✅ `multiHeadAttention.py` - 多头注意力机制

### Ch04：实现GPT模型 (Implementing GPT Model)

- ✅ `gptModel.py` - GPT模型架构实现

**GPT模型整体架构：**

<img src="image/README/1766658357442.png" width="600" alt="GPT Model Architecture"/>

**Transformer Block内部结构：**

<img src="image/README/1766658431287.png" width="600" alt="Transformer Block Structure"/>

- ✅ `previous_chapters.py` - 前面章节的集成

## 📝 要求

- Python 3.10
