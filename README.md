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

**项目整体目标流程图：**

<img src="image/README/1766739664902.png" width="700" alt="Project Overview"/>

### Ch02：文本数据处理 (Working with Text Data)

- ✅ `simpleTokenizer.py` - 简单分词器实现
- ✅ `tokenizer.py` - GPT2分词器测试
- ✅ `dataLoader.py` - PyTorch数据加载器
- ✅ `embedding.py` - 文本嵌入示例
- ✅ `text-prepare.py` - 文本预处理
- ✅ `test.py` - 基础测试

**Token嵌入层 (Embedding Layer)：**

<img src="image/README/1766658803148.png" width="600" alt="Embedding Layer"/>

### Ch03：注意力机制 (Coding Attention Mechanisms)

**注意力机制基础概念：**

<img src="image/README/1766658617366.png" width="600" alt="Attention Mechanism"/>

**因果注意力机制 (Causal Attention) 流程：**

<img src="image/README/1766658678226.png" width="600" alt="Causal Attention Flow"/>

**多头注意力机制 (Multi-Head Attention)：**

<img src="image/README/1766658727531.png" width="600" alt="Multi-Head Attention"/>

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

### Ch05：无标签数据预训练 (Pretraining on Unlabeled Data)

- ✅ `previous_chapters.py` - 前四章代码集成优化
- ✅ `generate_test.py` - 文本生成测试
- ✅ `loss_calc.py` - 损失函数计算
- ✅ `loss_test.py` - 损失函数测试
- ✅ `gpt_download.py` - GPT模型下载工具
- ✅ `pretraining.py` - 预训练主程序
- ✅ `load.py` - 模型加载工具
- ✅ `temperature_scaling_and_top-k.py` - 文本生成策略优化

**预训练目标：**

<img src="image/README/1766739399269.png" width="600" alt="Loss Calculation and Optimization"/>

**损失函数计算详解：**

<img src="image/README/1766739380311.png" width="600" alt="Loss Calculation Details"/>

**预训练流程详解：**

<img src="image/README/1766910153313.png" width="600" alt="Pretraining Process"/>

**Top-k采样策略示例：**

<img src="image/README/1766910243825.png" width="600" alt="Top-k Sampling Example"/>

### Ch06：文本分类微调 (Finetuning for Text Classification)

**项目目标：** 基于预训练的GPT-2模型，使用SMS垃圾短信数据集进行文本分类微调

#### 6.2 数据准备 (Preparing the dataset)

- ✅ 下载并解析SMS垃圾短信数据集
- ✅ 类别平衡：欠采样多数类（ham）以匹配少数类（spam）数量
- ✅ 类别编码：将字符串标签映射为整数（ham: 0, spam: 1）
- ✅ 数据分割：70% 训练集、10% 验证集、20% 测试集

#### 6.3 数据加载器 (Creating data loaders)

- ✅ `SpamDataset` 类实现：
  - 文本分词和编码
  - 动态最大长度计算
  - 填充到统一长度
- ✅ DataLoader 配置（batch_size=8, shuffle=True）
- ✅ 数据验证（输入维度: [batch_size, sequence_length]，标签维度: [batch_size]）

**数据集处理流程：**

<img src="image/README/1767000838511.png" width="600" alt="Dataset Processing Flow"/>

**数据加载器验证：**

<img src="image/README/1767000860575.png" width="600" alt="DataLoader Verification"/>

#### 6.4 模型初始化 (Initializing a model with pretrained weights)

- ✅ 加载预训练GPT-2小模型（124M参数）
- ✅ 权重加载验证：生成文本测试
- ✅ 关键修复：添加 `model.eval()` 确保LayerNorm使用运行时统计

**GPT-2模型配置：**

- vocab_size: 50257
- context_length: 1024
- emb_dim: 768
- n_layers: 12
- n_heads: 12

#### 6.5 分类头添加 (Adding a classification head)

- ✅ 冻结所有预训练参数：`param.requires_grad = False`
- ✅ 替换输出层为分类层：`Linear(emb_dim=768, num_classes=2)`
- ✅ 解冻最后一层和LayerNorm：允许微调

**转移学习策略：** 冻结大部分参数，仅微调最后的transformer块和分类头，这样既能利用预训练知识，又能快速适应新任务

**分类头架构：**

<img src="image/README/1767001053566.png" width="600" alt="Classification Head Architecture"/>

该图展示了GPT-2模型经过冻结参数和添加分类头后的结构，（最后一层transformer块和分类头）可参与梯度更新

#### 6.6 损失和精度计算 (Calculating loss and accuracy)

- ✅ `calc_accuracy_loader()`：计算分类精度

  - 使用最后一个token的logits
  - 通过argmax获取预测标签
  - 对比target计算正确率
- ✅ `calc_loss_batch()`：计算单个batch的交叉熵损失
- ✅ `calc_loss_loader()`：计算整个data loader的平均损失

**损失和精度计算结果：**

<img src="image/README/1767000933805.png" width="600" alt="Loss and Accuracy Calculation Results"/>
