# XXG通义千问离线图片中文打标工具-离线版(Qwen3-VL) Ver.2.3
> **本地模型 · 100%离线运行 · 隐私安全 · 自然中文语句描述 · 训练中文提示词模型专用**
> 
> **By 西小瓜 / 使用问题和AI交流请联系 / Wechat:priest-mos**
> 
![](https://img.shields.io/badge/Python-10.0-blue.svg?style=flat#crop=0&crop=0&crop=1&crop=1&id=ebVxY&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=jYpxH&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=OveOV&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
![](https://img.shields.io/badge/Pytorch-2.4.1-brightgreen.svg?style=flat#crop=0&crop=0&crop=1&crop=1&id=ebVxY&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=jYpxH&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=OveOV&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) ![](https://img.shields.io/badge/Torchvision-0.19.1-brightgreen.svg?style=flat#crop=0&crop=0&crop=1&crop=1&id=ebVxY&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=jYpxH&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=OveOV&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) ![](https://img.shields.io/badge/Torchaudio-2.4.1-brightgreen.svg?style=flat#crop=0&crop=0&crop=1&crop=1&id=ebVxY&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=jYpxH&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=OveOV&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) ![](https://img.shields.io/badge/gradio-4.44.1-red.svg?style=flat#crop=0&crop=0&crop=1&crop=1&id=ebVxY&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=jYpxH&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=#crop=0&crop=0&crop=1&crop=1&id=OveOV&originHeight=20&originWidth=86&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

<!-- 这是示例 -->
![](./assets/demo_01.jpg)
![](./assets/demo_02.jpg)

## 🌟 核心特性
+ **完全离线**: 无网络请求，数据永不离开您的设备
+ **专业中文描述**: 详细场景分析，包含人物、物体、风景、设计、电商等
+ **隐私保护**: 适合处理敏感/私人/商业化图片
+ **智能资源管理**: 4-bit量化支持，低显存需求
+ **用户友好**: 简洁Web界面，实时进度显示

## 📋 系统要求
### 最低配置
+ **操作系统**: Windows 10/11, macOS 12+, Linux
+ **CPU**: 4核以上
+ **内存**: 8GB RAM
+ **磁盘空间**: 20GB+ 空闲
+ **Python**: 3.10

### 推荐配置
+ **GPU**: NVIDIA GPU (8GB+显存)
+ **内存**: 16GB+ RAM
+ **磁盘**: SSD (加快IO速度)

### 技术规格
+ **基础模型**: Qwen3-VL-8B-Instruct (通义千问视觉语言模型)
+ **中文打标质量**: 专业级描述，包含人物、物体、环境、颜色、动作、文字、排版等细节
+ **资源需求**:
  + 磁盘空间: 20GB+ (模型14GB + 缓存)
  + GPU: NVIDIA GPU 6GB+显存 (推荐8GB+)
  + CPU模式: 8GB+ RAM
+ **处理速度**:
  + GPU 4-bit: 每张2-4秒 (RTX 4090)
  + GPU 标准: 每张1.5-3秒 (RTX 4090)
  + CPU: 每张15-30秒 (16核) _注：cpu性能越差生成时间越长，如macos m1芯片需要150秒左右/张_

###  选择适合您的启动脚本:
+ **Linux/Mac用户**:
  + chmod +x run.sh && ./run.sh
+ **Windows用户**:
  + run.bat
###  首次运行:
  + 脚本会自动创建环境、安装依赖（失败后请根据提示使用pip install安装）
  + 提示下载Qwen3-VL模型（约14GB）
  + 模型下载完成后完全离线运行


## 🚀 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/moskoo/qwen-caption.git
cd qwen-caption
```

### <font style="color:rgb(29, 29, 31);">2. 下载模型 (需要网络)</font>
```bash
# Linux/Mac用户
chmod +x download_model.sh && ./download_model.sh


# Windows用户
download_model.bat
```

### <font style="color:rgb(29, 29, 31);">3. 运行应用 (完全离线)</font>
```bash
# Linux/Mac
chmod +x run.sh && ./run.sh

# Windows
run.bat
```

### <font style="color:rgb(29, 29, 31);">4. 使用Web界面</font>
1. <font style="color:rgb(29, 29, 31);">访问 </font><font style="color:rgb(97, 92, 237);">`http://127.0.0.1:9527`</font>
2. <font style="color:rgb(29, 29, 31);">输入图片文件夹路径</font>
3. <font style="color:rgb(29, 29, 31);">选择运行模式 (4-bit量化/CPU模式)</font>
4. <font style="color:rgb(29, 29, 31);">点击"</font><font style="color:rgb(29, 29, 31);">🚀</font><font style="color:rgb(29, 29, 31);"> 开始中文打标"</font>
5. <font style="color:rgb(29, 29, 31);">查看结果和生成的txt文件</font>

## <font style="color:rgb(29, 29, 31);">⚙️</font><font style="color:rgb(29, 29, 31);"> 高级用法</font>
### <font style="color:rgb(29, 29, 31);">低显存模式 (6GB以下显存GPU)</font>
```bash
python app.py --4bit
```

### <font style="color:rgb(29, 29, 31);">CPU模式 (无GPU)</font>
```bash
python app.py --cpu
```

### <font style="color:rgb(29, 29, 31);">自定义端口</font>
```bash
python app.py --port 7860
```

## <font style="color:rgb(29, 29, 31);">📦</font><font style="color:rgb(29, 29, 31);"> 项目结构</font>
```bash
qwen-caption/
├── app.py                     # 主应用程序
├── requirements.txt           # 依赖文件
├── download_model.sh          # linux/macos下载qwen3-vl模型脚本
├── download_model.bat         # win下载qwen3-vl模型脚本
├── run.sh                     # linux/macos 主启动脚本
├── run.bat                    # win主启动脚本
├── qwen3_vl_models/           # 模型目录（自动创建）
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
├── .venv/                     # 虚拟环境
├── datasets/demo/             # 推理结果示例
└── README.md                  # 使用指南
```

## <font style="color:rgb(29, 29, 31);">🛠️</font><font style="color:rgb(29, 29, 31);"> 常见问题</font>
### <font style="color:rgb(29, 29, 31);">Q: 首次运行需要多长时间?</font>
<font style="color:rgb(29, 29, 31);">A:</font>

+ <font style="color:rgb(29, 29, 31);">模型下载: 10-60分钟 (取决于网络速度)</font>
+ <font style="color:rgb(29, 29, 31);">首次启动: 2-5分钟 (加载模型到内存)</font>
+ <font style="color:rgb(29, 29, 31);">之后启动: 10-30秒</font>

### <font style="color:rgb(29, 29, 31);">Q: 处理速度如何?</font>
<font style="color:rgb(29, 29, 31);">A:</font>

| <font style="color:rgb(29, 29, 31);">模式</font> | <font style="color:rgb(29, 29, 31);">每张图片耗时</font>  | <font style="color:rgb(29, 29, 31);">100张图片预计</font> |
| --- |-----------------------------------------------------|------------------------------------------------------|
| <font style="color:rgb(29, 29, 31);">GPU (8GB+)</font> | <font style="color:rgb(29, 29, 31);">1.5-3秒</font>  | <font style="color:rgb(29, 29, 31);">3-5分钟</font>    |
| <font style="color:rgb(29, 29, 31);">4-bit量化</font> | <font style="color:rgb(29, 29, 31);">2-4秒</font>    | <font style="color:rgb(29, 29, 31);">5-7分钟</font>    |
| <font style="color:rgb(29, 29, 31);">CPU模式</font> | <font style="color:rgb(29, 29, 31);">15-180秒</font> | <font style="color:rgb(29, 29, 31);">20-70分钟</font>  |

### <font style="color:rgb(29, 29, 31);">Q: 模型文件太大怎么办?（约14GB）</font>
<font style="color:rgb(29, 29, 31);">A:</font>

+ <font style="color:rgb(29, 29, 31);">使用4-bit量化 (显存需求减少75%)</font>
+ <font style="color:rgb(29, 29, 31);">仅保留必要文件 (删除示例、文档等)</font>
+ <font style="color:rgb(29, 29, 31);">使用外部硬盘存储模型</font>

## 🚀 不使用一键脚本使用指南

### 1. **首次设置 (需要网络)**
```bash
# 1. 克隆仓库
git clone https://github.com/moskoo/qwen-caption.git
cd qwen-caption

# 2. 安装依赖
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 步骤2-1: 安装基础依赖 (无transformers)
pip install -r requirements.txt

# 步骤2-2: 安装编译工具
pip install -r requirements-transformers.txt

# 3. 下载模型 (约14GB)
# Linux/Mac用户
chmod +x download_model.sh && ./download_model.sh
# Windows用户
download_model.bat

# 4. 运行主程序
python app.py
```

**<font style="color:rgb(140, 141, 155);">重要提示</font>**<font style="color:rgb(140, 141, 155);">: 首次下载完成后，所有操作完全离线进行。模型文件约14GB，确保有足够磁盘空间。有任何疑问可以联系小瓜。</font>


## <font style="color:rgb(29, 29, 31);">📜</font><font style="color:rgb(29, 29, 31);"> 许可证</font>
<font style="color:rgb(29, 29, 31);">本项目使用 </font>[<font style="color:rgb(97, 92, 237);">Apache 2.0 License</font>](https://chat.qwen.ai/c/LICENSE)<font style="color:rgb(29, 29, 31);">，基于通义实验室开源的Qwen3-VL模型。</font>

**<font style="color:rgb(140, 141, 155);">注意</font>**<font style="color:rgb(140, 141, 155);">: 本工具仅用于个人学习和研究目的。商业使用请遵守Qwen模型的许可协议。</font>

## <font style="color:rgb(29, 29, 31);">🙏</font><font style="color:rgb(29, 29, 31);"> 致谢</font>
+ [<font style="color:rgb(97, 92, 237);">通义实验室</font>](https://www.aliyun.com/product/tongyi)<font style="color:rgb(29, 29, 31);"> - **Qwen3-VL**模型</font>
+ [<font style="color:rgb(97, 92, 237);">Hugging Face</font>](https://huggingface.co/)<font style="color:rgb(29, 29, 31);"> - 模型托管平台</font>
+ [<font style="color:rgb(97, 92, 237);">Gradio</font>](https://www.gradio.app/)<font style="color:rgb(29, 29, 31);"> - Web界面框架</font>

---

**<font style="color:rgb(29, 29, 31);">©</font>XXG<font style="color:rgb(29, 29, 31);"> 2026 通义千问离线图片打标工具 | 完全离线 · 隐私安全 · 开源免费</font>**
