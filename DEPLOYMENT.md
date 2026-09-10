# 本地部署与恢复

## 已部署电脑

保持目录关系：

```text
工作目录/
├─ photo-to-print/                  # 此仓库
├─ photo-to-print-runtime/          # 服务与几何处理 Python 环境
├─ photo-to-print-xpu-conda/         # 当前单图推理环境
└─ photo-to-print-ai-conda/          # 保留的 CPU 实验/渲染环境
```

双击 `启动应用.cmd`，打开 `http://127.0.0.1:8765`。控制台关闭后服务结束。仓库名可与本机目录名不同；运行环境、模型和 `data` 中本机验收记录需要保留。已有模型分色无需运行混元。

## 在另一台 Windows 电脑准备基础应用

需要 Python 3.12 x64、Node.js（满足 `package.json` 的 engines）、Git。终端进入仓库后运行：

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe start_app.py
```

`setup.ps1` 安装固定的几何/服务依赖并构建界面，**不包含单图 AI 环境与权重**。若系统策略阻止执行脚本，可依次执行其中的 Python、pip、npm 命令；无需关闭系统安全策略。启动器优先使用 `.venv`，否则使用上一级 `photo-to-print-runtime`。

## 单图引擎配置记录

当前工作版本针对 Windows / Intel Arc XPU 验证，不能据此声称 CUDA、其他显卡或其他系统已经适配。

| 项目 | 固定版本/位置 |
|---|---|
| Miniconda | 本机启动路径回退为 `C:/miniconda3/Scripts/conda.exe`；部分实验脚本需要按实际安装位置调整 |
| Python 环境 | 上一级 `photo-to-print-xpu-conda` |
| PyTorch / torchvision | `2.6.0+xpu` / `0.21.0+xpu`，来源 `https://download.pytorch.org/whl/xpu` |
| 其他包 | `requirements-xpu-installed.txt`；基础应用 `requirements-lock.txt` |
| 混元源代码 | `tools/hunyuan3d`，官方提交 `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| 模型仓库 | `tencent/Hunyuan3D-2mini`，提交 `f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6` |
| 模型目录 | `models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini-turbo` |
| 抠图 | `models/rembg/u2net.onnx`；官方 MD5 `60024c5c889badc19c04ad937298a77b` |

Turbo `model.fp16.safetensors` 为 3,822,584,202 字节，SHA-256：

```text
bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4
```

对应 `config.yaml` SHA-256：

```text
be28205844da01bd5d3c5ba5160f5886fb9765d542483f9d12bda8f17324db5e
```

worker 加载前校验这两个文件，全程离线推理。用官方仓库取得匹配文件及许可证，不要替换为来历不明的整合包。

旧 `scripts/prepare_single_photo.py` 和 `download_hunyuan_weight.py` 默认准备的是**标准 Mini / CPU 实验环境**，不能作为当前 Turbo 的一键安装器。`prepare_xpu_dependencies.py` 只补齐 XPU 环境的其他依赖，不安装显卡驱动。仓库还没有统一的跨设备单图安装/验收入口。

新机器安装后先运行 `check_xpu_runtime.py`，再用自有照片运行 `foreground.py` 和 `single_photo_turbo.py`，检查实际网格并完成打印处理。浏览器是否开放单图由 `backend/single_photo.py` 检查本机真实验收记录；**不能通过手写“验证成功”绕过检查**。历史猫照/人物照验收脚本依赖未随仓库分发的私有样本，不能直接当作通用安装检查。

## 开发与测试

```powershell
npm.cmd ci
npm.cmd run build
.\.venv\Scripts\python.exe -m pytest -q
```

需要热更新时，在基础服务运行的同时执行 `npm.cmd run dev`，访问输出的本地地址。测试需要可写临时目录；必要时将 pytest 的 `--basetemp` 指向工作区专用测试目录，勿指向含真实资料的文件夹。

多角度路线另见 `安装重建引擎.ps1`。模型许可及官方入口见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
