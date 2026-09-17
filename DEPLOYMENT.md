# 部署、启动与局域网使用

PhotoForm 是有 Python 后端的本地应用。浏览器只是界面，模型电脑必须开机并运行服务。GitHub Pages 不能代替推理后端。完成首次安装和检查后，生成使用本地文件；同一电脑可以离线使用。

## 已部署电脑

- 本机使用：双击 `启动应用.cmd`，打开 `http://127.0.0.1:8765`。
- 手机或另一台电脑：在模型电脑双击 `启动局域网.cmd`，使用窗口显示的局域网地址与访问码。
- 提示模型未就绪：运行 `检查模型环境.cmd`，结果见 `data/engine-check.json`。
- 关机或结束服务后地址不能继续使用；再次运行启动器即可，历史任务仍在 `data/`。
- 已运行本机模式时，先在原服务窗口按 Ctrl+C，再启动局域网模式。重复启动会打开现有服务，不启动第二个推理服务。

## 另一台 Windows 电脑：基础应用

需要 Python 3.12 x64、Node.js 22.13 或更高版本、Git；单图 AI 还需要 Miniconda3 和兼容的 Intel XPU 硬件。实测机器为 Core Ultra 9 185H / Intel Arc 集显 / 32 GB 内存。**当前安装器针对 Intel XPU，尚未在 NVIDIA CUDA、AMD 或其他系统上验收。** 其他设备可以作为浏览器客户端连接兼容的模型电脑，或单独使用基础网格处理。

```powershell
git clone https://github.com/jinmianlee/photofprm-local.git
cd photofprm-local
.\setup.ps1
```

基础安装不包括 AI 环境和权重。如果执行策略阻止脚本，按以下等价命令操作，无需更改安全策略：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
npm.cmd ci
npm.cmd run build
```

此时可导入 PLY / GLB / STL。单图生成还需下面的 AI 环境。

## A. 接入已有本地模型和 Conda 环境

本机路径保存在 `photoform.json`，此文件不上传 GitHub。参考 `photoform.example.json`；相对路径以项目根目录为基准。可用命令配置并进行实际检查，不重复下载：

```powershell
.\.venv\Scripts\python.exe scripts/configure_engine.py `
  --conda "C:/miniconda3/Scripts/conda.exe" `
  --ai-env "D:/AI/photoform-xpu" `
  --model-dir "D:/AI/models/hunyuan3d-dit-v2-mini-turbo" `
  --source-dir "D:/AI/Hunyuan3D-2" `
  --rembg-dir "D:/AI/models/rembg"
```

替换为真实路径。`model_dir` 直接包含 Turbo `config.yaml` 和 `model.fp16.safetensors`；`source_dir` 包含官方 `hy3dgen/shapegen`；`rembg_dir` 包含 `u2net.onnx`。AI 环境需要本项目固定依赖；这不是 Ollama 模型目录。

加 `--no-check` 可以仅保存路径，之后运行 `检查模型环境.cmd`。检查会校验模型哈希和官方源代码，导入推理管线，并比较实际 XPU 注意力计算与 CPU 参考结果。**通过只说明运行条件具备，不代表相似度、生成质量或实物打印已验收。** 不再要求复制旧猫照或手写历史验收文件。

未配置时自动兼容原目录旁的 `photo-to-print-xpu-conda`，新部署默认使用仓库内 `.ai-env`。默认模型和源码位于 `models/`、`tools/hunyuan3d`。路径或文件变化后需要重新检查；推理时仍会完整校验模型哈希。

## B. 从官方来源安装单图环境

先安装 Miniconda3。如果 `conda` 不在 PATH，用上节的 `--conda` 和 `--no-check` 保存正确路径，再运行：

```powershell
# 可先查看计划；不下载、不修改环境
.\.venv\Scripts\python.exe scripts/setup_turbo.py --dry-run

# 安装环境、官方源码、Turbo/抠图权重并检查
.\.venv\Scripts\python.exe scripts/setup_turbo.py
```

安装器从官方 PyTorch XPU 索引安装 `torch==2.6.0+xpu`、`torchvision==0.21.0+xpu`，其他包固定于 `requirements-xpu-installed.txt`。使用官方 Conda 频道；如果 Conda 要求确认许可，按其提示自行完成，再重跑命令。本应用不自动同意协议或修改驱动。

Turbo 单个权重约 3.82 GB，还需运行环境、下载缓存、抠图模型和生成结果空间。网络失败后可按阶段重跑，`hf download` 会复用缓存；不会关闭 TLS 校验或添加杀毒白名单。

```powershell
.\.venv\Scripts\python.exe scripts/setup_turbo.py --stage dependencies
.\.venv\Scripts\python.exe scripts/setup_turbo.py --stage source
.\.venv\Scripts\python.exe scripts/setup_turbo.py --stage models
.\.venv\Scripts\python.exe scripts/setup_turbo.py --stage check
```

本轮用已有官方文件完成了检查和生成。提供了全新安装脚本与固定版本，**但没有在另一台干净机器重新下载全部依赖**。旧 `prepare_single_photo.py` 默认准备标准 Mini CPU 实验，不是当前 Turbo 安装入口。

## 可选：2.1 精细形状模型

安装完成后，网页“形状模型”会出现“2.1 精细”；Mini 仍是快速默认。结果和限制见 [CAT_QUALITY.md](CAT_QUALITY.md)。先完成上面的 XPU 环境安装；此处不安装纹理模型、CUDA 渲染扩展或新驱动。模型另有腾讯社区许可，请阅读官方 `LICENSE`。

```powershell
# 仅检出官方形状 Python 包，避开训练样本的 Windows 长路径
git clone --filter=blob:none --no-checkout https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git tools/Hunyuan3D-2.1
git -C tools/Hunyuan3D-2.1 sparse-checkout set hy3dshape/hy3dshape
git -C tools/Hunyuan3D-2.1 fetch origin 82920d643c0dc2f7bfd7255f45f62d386edfe60c
git -C tools/Hunyuan3D-2.1 checkout --detach 82920d643c0dc2f7bfd7255f45f62d386edfe60c

conda run --prefix .ai-env python -m pip install --no-deps --only-binary=:all: --index-url https://pypi.org/simple -r requirements-shape21.txt
conda run --prefix .ai-env hf download tencent/Hunyuan3D-2.1 LICENSE Notice.txt README.md hunyuan3d-dit-v2-1/config.yaml hunyuan3d-dit-v2-1/model.fp16.ckpt --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 --local-dir models/hunyuan3d-2.1 --max-workers 1

# 使用自己的透明前景照片，输出仍是未经打印验收的原始模型
conda run --prefix .ai-env --no-capture-output python scripts/compare_shape21.py --input data/your-foreground.png --output data/shape21-trial --steps 50 --seed 1234 --resolution 255 --resume
```

已有 Conda 环境时，将 `.ai-env` 换为 `photoform.json` 中配置的路径。源码/模型放在别处可传 `--source` / `--model`。整包下载持续无进度时先正常取消，再用同一环境运行 `scripts/download_shape21.py`；只从原官方 HTTPS 地址分段下载，完成哈希校验前不提供可加载文件，分段缓存会额外占用约一份权重空间。不要同时运行两个下载器写同一目标。

成功推理仍不代表忠实还原原照片，更不代表已可直接打印。对照正、侧、背面后，再在本应用导入修复后的网格进行打印检查与分色。本文未声称其他硬件上的端到端部署已通过。

## 局域网连接

```powershell
.\.venv\Scripts\python.exe start_app.py --lan
# 多网卡未显示所需地址时，加入本机真实局域网 IP：
.\.venv\Scripts\python.exe start_app.py --lan --allow-host 192.168.1.25
```

客户端与模型电脑连接同一可信家庭/办公网络，打开窗口显示的 `http://局域网IP:8765`，输入访问码。客户端不需要 Python、Conda 或模型。访问码保存在 `data/lan-key.txt`；它授予所有任务的查看和管理权限，不是多用户隔离系统。

当前局域网 HTTP 没有链路加密，只用于可信私有网络，不要公网端口转发。互联网远程使用需另配 HTTPS、VPN/访问控制，本版不提供公网托管。Windows 若询问网络权限，仅允许预期的私有网络；应用不自动修改防火墙。无法连接时检查同网段、访客网络隔离及私有网络端口权限。

浏览器摄像头通常要求安全上下文，普通局域网 HTTP 可能禁用摄像头。此时先用手机相机拍好，再“上传主体照片”；本机 localhost 拍摄入口仍可用。

## 任务与文件

右上角“任务与文件”显示全部任务，可搜索编号、筛选状态、查看磁盘占用及文件清单。已完成任务直接进入第三步；重新分色会创建新任务，保留旧结果。

- **取消生成**：终止该任务的计算进程及子进程，保留照片、断点和已有模型。
- **删除任务及文件**：任务停止后经确认永久删除其整个目录，包含输入、模型、缓存和日志。共享权重、其他任务和已下载到别处的副本不受影响。
- 文件被切片器占用时，删除可能失败；关闭占用程序后重试。不自动清理已有任务。
- 历史实验目录不属于正式任务库，其空间不计入任务总量，也不会在这里删除。

## 固定文件与验证

| 项目 | 固定值 |
|---|---|
| 官方源码 Tencent-Hunyuan/Hunyuan3D-2 | `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| 模型 tencent/Hunyuan3D-2mini | `f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6` |
| Turbo 权重 SHA-256 | `bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4` |
| Turbo 配置 SHA-256 | `be28205844da01bd5d3c5ba5160f5886fb9765d542483f9d12bda8f17324db5e` |
| U2Net 官方 MD5 | `60024c5c889badc19c04ad937298a77b` |

开发检查：

```powershell
npm.cmd run build
.\.venv\Scripts\python.exe -m pytest -q
```

许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，历史与精度限制见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
