# 学习清单

## linux 基础操作

- [x] 了解什么是bash和.bashrc (推荐使用zsh，并安装命令补全插件)
- [x] 学习ssh的使用，选择合适的终端工具，学习用rsakey无密码登陆，学习.ssh/config的配置
- [x] 学习vscode或者pycharm等IDE的远程开发功能，学习其中的Jupyter notebooks的使用（数据分析或小型实验）
- [x] 了解什么是frp穿透
- [x] 常用bash命令：ls, cd, cp, mv, rm, mkdir, cat, head, tail, df, du
- [x] 学习frp命令
- [x] 学习ln，在有大文件、大数据集的时候很有用
- [x] `ps aux | grep xxx`：查找进程
- [x] `nvidia-smi`：查看显卡信息，显存使用情况 学习screen命令（如果你会tmux也可以）
- [x] 学习安装miniconda和安装conda虚拟环境，conda常用命令

## 论文阅读

### 论文1：现代深度学习基础

- [x] 了解深度学习中的基本概念和经典任务
- [ ] **Transformer：Attention is all your need**
- [ ] **BERT: Bidirectional Encoder Representations from Transformers**
- [ ] **MCAN: Deep modular co-attention networks for visual question answering**
- [ ] **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
- [ ] **CLIP: Learning Transferable Visual Models From Natural Language Supervision**

### 论文2：大模型相关

#### 模型与训练

- [ ] **GPT-3: Language Models are Few-Shot Learners**
- [ ] InstructGPT: Training language models to follow instructions with human feedback
- [ ] Llama 2: Open Foundation and Fine-Tuned Chat Models
- [ ] Transformer升级之路：2、博采众长的旋转式位置编码
- [ ] **LoRA: Low-rank adaptation of large language models**
- [ ] Alpaca: A Strong, Replicable Instruction-Following Model
- [ ] DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Mode

#### Prompt技术

- [ ] **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
- [ ] **ReAct: Synergizing Reasoning and Acting in Language Models**
- [ ] Retrieval Augmented Generation (RAG)
- [ ] Introduction to LLM Agents

### 论文3：多模态相关

- [ ] **Prompting large language models with answer heuristics for knowledge-based visual question answering**
- [ ] **LLaVA-1: Visual Instruction Tuning**
- [ ] **LLaVA-1.5: Improved Baselines with Visual Instruction Tuning**
- [ ] NExT-Chat: An LMM for Chat, Detection and Segmentation
- [ ] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models
- [ ] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face
- [ ] Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
- [ ] Segment Anything
- [ ] Video-LLaVA: Learning United Visual Representation by Alignment Before Projection
- [ ] VideoChat: Chat-Centric Video Understanding
- [ ] OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation
- [ ] Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning

### 论文4：选读

#### Diffusion

#### 3D

#### SSM

## 代码练习

### Exercise 1: PyTorch基础

- [ ] 手写一个MNIST数字识别的10层左右的小型resnet模型
- [ ] 可视化loss和accuracy变化曲线
- [ ] 存储训练得到的最好的模型
- [ ] 自己拿一张白纸写一个数字，输入模型实现准确预测
- [ ] 尝试在多卡环境下分别使用data parallel、distributed data parallel和FSDP ZeRO-3进行训练，该实验不在jupyter notebook中进行，直接写python脚本

### Exercise 2: Transformers基础和基础和LLM

- [ ] 下载qwen2-1.5b-instruct模型
- [ ] 观察local_folder_path有哪些文件，打开看看.json文件有哪些信息，在后续实验中体会这些文件和参数的作用
- [ ] load模型，打印模型的config信息model.config，打印模型的结构print(model)，打印所有模型参数的shape、device和dtype
- [ ] 使用model.generate()实现一次生成，查看源码或查阅资料了解generate()有哪些参数，作用是什么
- [ ] 不使用generate()，使用model.model(input_ids)等底层接口自己实现基于beam search的生成，循环过程中请使用past_key_values
- [ ] 比较 贪心和beam search的生成结果的困惑度（exp(NLL per token)）的区别

### Exercise 3: OpenAI API

- [ ] 使用OpenAI API调用gpt-4o模型实现一次文本生成
- [ ] 使用gpt-4o的图像接口对图像提问
- [ ] 利用gpt-4o的图像接口完成一个图像定位任务
- [ ] 】持续一周在网页端和ChatGPT、Claude3.5和Gemini（任选）中进行各种各样的对话交互，体会其擅长的问题和能力边界

### Exercise 4: OpenAI CLIP实验

- [ ] 下载huggingface上的模型openai/clip-vit-base-patch32到本地文件夹（注意，flax和tf的权重不需要下载，只需要下载pytorch_model.bin），观察配置文件内容
- [ ] 使用该模型实现一次图片分类（注意，是不需要分类的，依靠prompt计算和相似度计算实现）
- [ ] 观察输入输出等步骤中的tensor的shape，获取视觉vit计算过程中的hidden states和attention map，观察他们的shape
- [ ] 可视化vit的倒数第二个SA层的第一个head的attention map

------

# 
