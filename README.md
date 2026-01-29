# 🖼️ 通义千问离线图片中文打标工具
> **100%离线运行 · 隐私安全 · 专业级中文描述**
>

<!-- 这是一张图片 -->
![](https://dashscope.oss-cn-beijing.aliyuncs.com/images/qwen-offline-ui.png)

## 🌟 核心特性
+ **完全离线**: 无网络请求，数据永不离开您的设备
+ **专业中文描述**: 详细场景分析，包含物体、环境、颜色、动作等
+ **隐私保护**: 适合处理敏感/私人图片
+ **智能资源管理**: 4-bit量化支持，低显存需求
+ **用户友好**: 简洁Web界面，实时进度显示

## 📋 系统要求
### 最低配置
+ **操作系统**: Windows 10/11, macOS 12+, Linux
+ **CPU**: 4核以上
+ **内存**: 8GB RAM
+ **磁盘空间**: 10GB+ 空闲
+ **Python**: 3.8+

### 推荐配置
+ **GPU**: NVIDIA GPU (8GB+显存)
+ **内存**: 16GB+ RAM
+ **磁盘**: SSD (加快IO速度)

### 技术规格
+ **基础模型**: Qwen-VL-Chat (通义千问视觉语言模型)
+ **中文打标质量**: 专业级描述，包含物体、场景、颜色、动作等细节
+ **资源需求**:
  + 磁盘空间: 20GB+ (模型18GB + 缓存)
  + GPU: NVIDIA GPU 6GB+显存 (推荐8GB+)
  + CPU模式: 8GB+ RAM
+ **处理速度**:
  + GPU模式: 每张3-5秒
  + CPU模式: 每张20-40秒

###  选择适合您的启动脚本:
+ **Linux/Mac用户**:
  + 非Conda: chmod +x run.sh && ./run.sh
  + Conda: chmod +x run-conda.sh && ./run-conda.sh
+ **Windows用户**:
  + 非Conda: run.bat
  + Conda: run-conda.bat
###  首次运行:
  + 脚本会自动创建环境、安装依赖
  + 提示下载Qwen-VL-Chat模型（约18GB）
  + 模型下载完成后完全离线运行

### 基本下载
```bash
python download_models.py
```

### 中国用户加速下载
```bash
python download_models.py --mirror
```

### 自定义保存目录
```bash
python download_models.py --dir ./models/qwen-vl
```

### 增加重试次数（网络不稳定时）
```bash
python download_models.py --retry 5
```

## 🚀 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/moskoo/qwen-caption.git
cd qwen-caption
```



### <font style="color:rgb(29, 29, 31);">2. 下载模型 (需要网络)</font>
```bash
# Linux/Mac
./run.sh --download

# Windows
run.bat --download
```

### <font style="color:rgb(29, 29, 31);">3. 运行应用 (完全离线)</font>
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### <font style="color:rgb(29, 29, 31);">4. 使用Web界面</font>
1. <font style="color:rgb(29, 29, 31);">访问 </font>`<font style="color:rgb(97, 92, 237);">http://127.0.0.1:9527</font>`
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

### <font style="color:rgb(29, 29, 31);">中国用户加速下载</font>
```bash
python download_models.py --mirror
```

## <font style="color:rgb(29, 29, 31);">📦</font><font style="color:rgb(29, 29, 31);"> 项目结构</font>
```bash
qwen-caption/
├── app.py                     # 主应用程序
├── requirements.txt           # 依赖文件
├── run.sh                     # 非Conda版启动脚本 (Linux/Mac)
├── run.bat                    # 非Conda版启动脚本 (Windows)
├── run-conda.sh               # Conda版启动脚本 (Linux/Mac)
├── run-conda.bat              # Conda版启动脚本 (Windows)
├── download_models.py         # 模型下载脚本
├── fix_dependencies.py        # 依赖修复工具
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

| <font style="color:rgb(29, 29, 31);">模式</font> | <font style="color:rgb(29, 29, 31);">每张图片耗时</font> | <font style="color:rgb(29, 29, 31);">100张图片预计</font> |
| --- | --- | --- |
| <font style="color:rgb(29, 29, 31);">GPU (8GB+)</font> | <font style="color:rgb(29, 29, 31);">3-5秒</font> | <font style="color:rgb(29, 29, 31);">5-8分钟</font> |
| <font style="color:rgb(29, 29, 31);">4-bit量化</font> | <font style="color:rgb(29, 29, 31);">4-7秒</font> | <font style="color:rgb(29, 29, 31);">7-12分钟</font> |
| <font style="color:rgb(29, 29, 31);">CPU模式</font> | <font style="color:rgb(29, 29, 31);">20-40秒</font> | <font style="color:rgb(29, 29, 31);">35-70分钟</font> |


### <font style="color:rgb(29, 29, 31);">Q: 如何离线部署到多台机器?</font>
<font style="color:rgb(29, 29, 31);">A:</font>

1. <font style="color:rgb(29, 29, 31);">在一台机器上完成下载和验证</font>
2. <font style="color:rgb(29, 29, 31);">复制整个项目目录 (包含 </font>`<font style="color:rgb(97, 92, 237);">qwen_models</font>`<font style="color:rgb(29, 29, 31);"> 文件夹)</font>
3. <font style="color:rgb(29, 29, 31);">在目标机器上运行 </font>`<font style="color:rgb(97, 92, 237);">fix_dependencies.py</font>`
4. <font style="color:rgb(29, 29, 31);">使用 </font>`<font style="color:rgb(97, 92, 237);">run.sh</font>`<font style="color:rgb(29, 29, 31);">/</font>`<font style="color:rgb(97, 92, 237);">run.bat</font>`<font style="color:rgb(29, 29, 31);"> 启动</font>

### <font style="color:rgb(29, 29, 31);">Q: 模型文件太大怎么办?</font>
<font style="color:rgb(29, 29, 31);">A:</font>

+ <font style="color:rgb(29, 29, 31);">使用4-bit量化 (显存需求减少75%)</font>
+ <font style="color:rgb(29, 29, 31);">仅保留必要文件 (删除示例、文档等)</font>
+ <font style="color:rgb(29, 29, 31);">使用外部硬盘存储模型</font>

## <font style="color:rgb(29, 29, 31);">📜</font><font style="color:rgb(29, 29, 31);"> 许可证</font>
<font style="color:rgb(29, 29, 31);">本项目使用 </font>[<font style="color:rgb(97, 92, 237);">Apache 2.0 License</font>](https://chat.qwen.ai/c/LICENSE)<font style="color:rgb(29, 29, 31);">，基于通义实验室开源的Qwen-VL-Chat模型。</font>

**<font style="color:rgb(140, 141, 155);">注意</font>**<font style="color:rgb(140, 141, 155);">: 本工具仅用于个人学习和研究目的。商业使用请遵守Qwen模型的许可协议。</font>

## <font style="color:rgb(29, 29, 31);">🙏</font><font style="color:rgb(29, 29, 31);"> 致谢</font>
+ [<font style="color:rgb(97, 92, 237);">通义实验室</font>](https://www.aliyun.com/product/tongyi)<font style="color:rgb(29, 29, 31);"> - Qwen-VL-Chat模型</font>
+ [<font style="color:rgb(97, 92, 237);">Hugging Face</font>](https://huggingface.co/)<font style="color:rgb(29, 29, 31);"> - 模型托管平台</font>
+ [<font style="color:rgb(97, 92, 237);">Gradio</font>](https://www.gradio.app/)<font style="color:rgb(29, 29, 31);"> - Web界面框架</font>

---

**<font style="color:rgb(29, 29, 31);">©</font>****<font style="color:rgb(29, 29, 31);"> 2026 通义千问离线图片打标工具 | 完全离线 · 隐私安全 · 开源免费</font>**

```bash
## 🚀 安装与使用指南

### 1. **首次设置 (需要网络)**
```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/qwen-captioner.git
cd qwen-captioner

# 2. 安装依赖
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# 3. 下载模型 (约5GB)
python download_models.py --mirror  # 中国用户加--mirror参数
```

### <font style="color:rgb(29, 29, 31);">2. </font>**<font style="color:rgb(29, 29, 31);">运行应用 (完全离线)</font>**
```bash
# Linux/Mac
chmod +x run.sh
./run.sh

# Windows
run.bat
```

### <font style="color:rgb(29, 29, 31);">3. </font>**<font style="color:rgb(29, 29, 31);">离线部署到其他机器</font>**
```bash
# 1. 在已配置好的机器上打包
zip -r qwen-caption.zip qwen-caption/

# 2. 复制到目标机器
# 3. 解压并运行修复脚本
python fix_dependencies.py --force

# 4. 启动应用
./run.sh  # 或 run.bat
```

**<font style="color:rgb(140, 141, 155);">重要提示</font>**<font style="color:rgb(140, 141, 155);">: 首次下载完成后，所有操作完全离线进行。模型文件约5GB，确保有足够磁盘空间。对于企业级部署，建议联系阿里云获取商业支持。</font>



