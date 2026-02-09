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

**文本分类微调工作流程：**

<img src="image/README/1767163855228.png" width="700" alt="Finetuning Workflow"/>

该流程图展示了从数据准备到模型推理的完整微调过程，包括数据处理、模型初始化、参数冻结、微调训练和最终分类等关键步骤。

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

#### 6.7 模型微调 (Finetuning on supervised data)

- ✅ `train_classifier_simple()`：完整的训练循环实现

  - 支持自定义epoch数、评估频率、评估迭代次数
  - 返回训练/验证损失和精度曲线
  - 每个epoch后显示精度统计
- ✅ `evaluate_model()`：评估阶段的模型切换

  - 在评估时切换到eval模式（禁用dropout和BatchNorm更新）
  - 评估后重新切换回train模式
- ✅ 优化器配置：`AdamW`（权重衰减0.1，学习率5e-5）
- ✅ 训练配置：5个epoch，每50步评估一次

**训练循环详细流程：**

<img src="image/README/1767005284011.png" width="600" alt="Training Loop Flow Diagram"/>

上图展示了完整的训练流程：

1. **初始化阶段**：设置模型、优化器、epoch数等
2. **训练循环**：对每个epoch进行以下操作
   - 将模型设置为train模式
   - 遍历所有batch数据
   - 计算loss → 反向传播 → 优化器更新
   - 定期进行验证评估
3. **评估阶段**：在验证集上评估当前模型性能
4. **统计输出**：每个epoch后打印训练/验证精度

**训练结果汇总：**

- 最终训练精度：~97-99%（模型充分学习训练数据特征）
- 最终验证精度：~93-95%（在未见数据上的泛化能力）
- 最终测试精度：~92-94%（最终模型评估指标）

#### 6.8 使用微调模型进行分类 (Using the LLM as a spam classifier)

- ✅ `classify_review()`：单条文本分类函数

  - **输入**：任意长度的文本字符串
  - **处理流程**：
    1. 文本分词和编码
    2. 截断或填充至指定长度
    3. 转换为tensor并推送到设备
    4. 提取最后token的logits
    5. argmax确定预测类别
  - **输出**："spam" 或 "not spam"

### Ch07：指令微调 (Finetuning to Follow Instructions)

**项目目标：** 基于预训练的GPT-2中型模型，使用指令数据集进行有监督微调，使模型能够遵循用户指令

**指令微调的完整流程：**

<img src="image/README/1767602476986.png" width="700" alt="Instruction Finetuning Workflow"/>

该流程图展示了从数据准备、模型加载、参数冻结、到模型微调和评估的完整指令微调过程。

#### 7.2 准备指令微调数据集 (Preparing a dataset for supervised instruction finetuning)

- ✅ 下载并加载指令数据集（alpaca-like format）
- ✅ 数据格式处理：包含instruction、input和output三个字段
- ✅ 文本格式化：按照标准模板组织instruction、input和response
- ✅ 数据分割：85% 训练集、10% 测试集、5% 验证集

**指令数据格式示例：**

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
[用户的指令]

### Input:
[可选的输入数据]

### Response:
[期望的模型输出]
```

#### 7.3 组织数据成训练批次 (Organizing data into training batches)

**数据集结构与批处理流程：**

<img src="image/README/1767602517877.png" width="600" alt="Dataset Structure"/>

该图展示了指令数据集的基本结构，包含instruction、input和output三个关键字段。

**Batch处理中的目标标记策略：**

<img src="image/README/1767602576497.png" width="600" alt="Batch Target Masking"/>

该图说明了在批处理中如何标记目标张量：仅对response部分计算loss，instruction和padding部分被设置为ignore_index（-100），这样模型只学习生成正确的response，而不对instruction部分进行优化。

- ✅ `InstructionDataset` 类实现：
  - 加载指令数据集
  - 文本分词和编码
  - 动态最大长度计算
- ✅ `custom_collate_fn` 实现：
  - 自定义填充策略（pad_token_id = 50256）
  - 为instruction部分（非response部分）的padding token设置ignore_index = -100
  - 限制最大长度为1024 tokens
  - 返回输入和目标张量对

#### 7.4 创建数据加载器 (Creating data loaders for an instruction dataset)

- ✅ 设备检测：支持CUDA、MPS或CPU
- ✅ 训练/验证/测试数据加载器配置：
  - batch_size: 8
  - 训练集：shuffle=True, drop_last=True
  - 验证/测试集：shuffle=False, drop_last=False
- ✅ 数据验证（inputs和targets维度匹配）

#### 7.5 加载预训练模型 (Loading a pretrained LLM)

- ✅ 加载预训练GPT-2中型模型（355M参数）
- ✅ 模型配置选项：

  - GPT-2 Small (124M)
  - GPT-2 Medium (355M) ← 本实验使用
  - GPT-2 Large (774M)
  - GPT-2 XL (1558M)

**GPT-2中型模型配置：**

- vocab_size: 50257
- context_length: 1024
- emb_dim: 1024
- n_layers: 24
- n_heads: 16

#### 7.6 在指令数据上微调模型 (Finetuning the LLM on instruction data)

- ✅ `train_model_simple()`：完整的微调训练循环

  - 支持自定义epoch数、评估频率、评估迭代次数
  - 定期在验证集上评估模型性能
  - 返回训练/验证损失和token计数
  - 支持生成示例输出进行定性评估

#### 7.7 提取和保存响应 (Extracting and saving responses)

- ✅ 在测试集上生成模型响应：

  - 使用微调后的模型进行推理
  - 生成max_new_tokens=256的输出
  - 清理response文本（移除模板部分）
- ✅ 响应保存：将测试数据和模型生成的响应保存为JSON文件
- ✅ 模型保存：将微调后的模型权重保存为 `.pth`文件

#### 7.8 评估微调模型 (Evaluating the finetuned LLM)

- ✅ 使用Ollama运行本地LLM进行评分：

  - 集成Llama3作为评判模型
  - 为每个模型响应生成0-100的评分
- ✅ 评分过程：

  - 对每条测试样本生成模型响应
  - 使用Llama3对比正确答案和模型答案
  - 计算平均评分
- ✅ 评估指标：

  - 个体样本评分：0-100分
  - 整体平均评分：汇总所有测试样本的评分
  - 定性评估：逐条对比正确答案和模型答案

---

### Ch07-DPO：偏好优化 (Preference Tuning with DPO)

**项目目标：** 使用Direct Preference Optimization (DPO)方法，基于人类偏好数据对已微调的模型进行进一步优化

**DPO优化流程概览：**

<img src="image/README/1770624092896.png" width="700" alt="DPO Optimization Workflow"/>

该流程图展示了DPO方法的核心过程：使用偏好数据集（包含chosen和rejected响应）训练策略模型，同时使用参考模型提供基准，通过优化相对概率来对齐人类偏好。



#### 1. 准备DPO偏好数据集 (Preparing a preference dataset for DPO)

- ✅ 下载并加载偏好数据集（包含chosen和rejected响应）
- ✅ 数据格式：包含instruction、input、output、chosen和rejected五个字段
  - `chosen`：人类偏好的响应
  - `rejected`：人类不偏好的响应
- ✅ 数据分割：85% 训练集、10% 测试集、5% 验证集

**偏好数据集结构：**

```json
{
  "instruction": "用户指令",
  "input": "可选输入",
  "output": "标准输出",
  "chosen": "偏好响应（更好的回答）",
  "rejected": "非偏好响应（较差的回答）"
}
```

#### 2. 创建偏好数据加载器 (Creating preference data loaders)

- ✅ `PreferenceDataset` 类实现：

  - 分别编码prompt、chosen和rejected响应
  - 存储三种token序列供训练使用
- ✅ `custom_collate_fn` 实现：

  - 为chosen和rejected响应分别创建mask
  - `mask_prompt_tokens=True`：仅对response部分计算loss
  - 动态padding到batch内最大长度
  - 返回包含prompt、chosen、rejected及对应mask的字典
- ✅ 数据验证：

  - 检查mask正确标记了response部分
  - 验证padding token被正确处理

#### 3. 加载策略模型和参考模型 (Loading policy and reference models)

- ✅ Policy Model（策略模型）：

  - 加载Ch07微调后的模型权重
  - 训练过程中参数会被更新
  - 用于生成最终的响应
- ✅ Reference Model（参考模型）：

  - 加载相同的Ch07微调后的模型权重
  - 参数保持冻结，不参与训练
  - 用于计算KL散度，防止策略模型偏离太远

**策略模型与参考模型的协同机制：**

<img src="image/README/1770624112406.png" width="650" alt="Policy and Reference Model Interaction"/>

该图说明了DPO训练中策略模型和参考模型的关系：策略模型负责学习并生成更符合人类偏好的响应，而参考模型作为固定基准，确保策略模型不会过度偏离原始的指令遵循能力。两者共同计算DPO损失，实现偏好对齐。

#### 4. 实现DPO损失函数 (Implementing DPO loss functions)

- ✅ `compute_logprobs()`：计算对数概率

  - 使用log_softmax获取token的对数概率
  - 通过torch.gather选择对应label的概率
  - 使用mask仅对response部分计算平均概率
- ✅ `compute_dpo_loss()`：核心DPO损失计算

  - 计算策略模型的log ratio：chosen vs rejected
  - 计算参考模型的log ratio：chosen vs rejected
  - 损失函数：`-log_sigmoid(β * (model_logratios - ref_logratios))`
  - 返回损失、chosen奖励和rejected奖励

**DPO损失公式核心思想：**

- 最大化chosen响应相对于rejected响应的概率比
- 使用参考模型作为基准，避免过度优化
- β参数控制优化强度（本实验使用0.1）

#### 5. DPO模型训练 (Training with DPO)

- ✅ `train_model_dpo_simple()`：DPO训练循环

  - 在每个batch上计算DPO损失
  - 同时跟踪chosen和rejected的奖励边际
  - 定期评估训练和验证的损失及奖励
  - 生成示例输出进行定性评估
- ✅ 训练配置：

  - 优化器：AdamW（lr=5e-6, weight_decay=0.01）
  - Epochs：1
  - Beta：0.1
  - 评估频率：每5步
- ✅ 关键指标：

  - **训练/验证损失**：DPO损失值
  - **Reward Margin**：chosen_reward - rejected_reward
  - 更高的reward margin表示模型更好地区分偏好

#### 6. 模型评估与对比 (Evaluation and comparison)

- ✅ 对比三种输出：

  - **正确答案**：数据集中的标准输出
  - **Reference Model**：SFT微调后的基础模型
  - **Policy Model**：DPO优化后的模型
- ✅ 评估维度：

  - 响应质量：是否更符合人类偏好
  - 响应风格：是否更加准确和有帮助
  - 对比分析：DPO前后的改进
- ✅ 损失曲线可视化：

  - 绘制训练/验证损失随epoch变化
  - 观察模型收敛情况

**DPO优势总结：**

- ✅ 无需显式奖励模型
- ✅ 训练更稳定（相比RLHF）
- ✅ 计算效率更高
- ✅ 直接优化策略模型以匹配人类偏好
