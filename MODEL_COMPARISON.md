# 单张照片转 3D：本地选型记录

核查日期：2026-09-16。用途：从用户的橘猫照片生成忠实、精细的头胸像，再转为多色打印的 3MF 和分色 STL。

最新同图对照见 [CAT_QUALITY.md](CAT_QUALITY.md)：五色与对应点校准改善了材料位置；Turbo 15 步和 Standard mini 均未明显解决脸形问题；正在独立验证更强的官方 2.1 形状模型。下面包含早期选型与切片记录，不是相似度已经通过用户验收的证明。

## 结论与当前状态

混元 3D-2 / 2mini / 2.1 公开了代码与权重，可以在社区许可允许的条件下免费本地使用；这不等同于 MIT 或 Apache-2.0 的宽松开源许可。腾讯当前条款包含地域、用途、分发和超大规模商业使用限制，不能把“免费”写成“无限制”。

在以下已核查的候选中，**TRELLIS.2 是高质量、MIT 许可候选**，但本机不满足官方 GPU 要求。按照用户选择，现已部署并实际跑通 **Hunyuan3D-2mini-Turbo / Intel XPU**。这属于 2mini 系列，尚不是在同一硬件上对所有候选实测得出的质量排名。

本机已确认是 Windows、Core Ultra 9 185H、Intel Arc 核显、约 32GB 内存，没有 NVIDIA CUDA 显卡。不能把系统内存当成 NVIDIA 显存。2mini 的官方示例仍使用 CUDA；能传入 `device='cpu'` 不足以证明整条链路可用或速度可接受。需要真实推理验收。

实测结果：标准 Mini 的 CPU / XPU 对照在这张猫照上未得到可接受实体；官方 Mini Turbo 的 3072 latent、5 步 Consistency 流程成功生成可辨认的折耳猫头胸像。正式 worker 与官方流程的最终 latent 完全一致；255 格网格包含 502132 个原始三角面。清理 4 个零面积三角面并小幅裁平底面后，95 mm 最长边的模型有 455766 面，尺寸约 94.668 × 95 × 83.386 mm。四种颜色均导出为封闭实体，Bambu Studio 导入时不需要网格修复，实际四色离线切片也通过。

硬件实测：Intel 的 Windows XPU 路径支持本机 Arc 核显。当前驱动 32.0.101.6314 低于 [PyTorch 2.10 前置条件](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-10.html)，但满足 [PyTorch 2.6 前置条件](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html)。Miniconda 环境 `photo-to-print-xpu-conda` 使用官方 PyTorch 2.6.0+XPU / torchvision 0.21.0+XPU，依赖检查与真实推理通过。Turbo 一次形状生成和网格提取耗时 254.72 秒；同时运行切片软件的另一次实测为 383.56 秒。该比较不等于统一硬件基准。没有修改显卡驱动或安全软件。

对照证据：标准 Mini 在同一 CPU 环境能生成腾讯官方企鹅示例，说明早期猫照失败不能直接归因于整个环境失效。官方源码保持未修改；本地适配位于 `backend/`。注意力按 query 分块，每块保留全部 key，避免本机 GPU 单次大分配失败；CPU / XPU 注意力一致性和 Turbo 断点恢复均已验证。猫照原始结果、正式任务和切片报告保留在 `data/`。颜色来自照片投影与量化，背面使用主色；不是混元纹理模型，也未完成实物试打。

## 主要候选比较

| 模型 | 许可与免费情况 | 对本项目的价值 | 官方硬件 / 本机限制 |
|---|---|---|---|
| TRELLIS.2-4B | 主模型及代码 MIT；依赖各有许可 | 同时生成细致形状和材质，公开训练代码；本项目的高质量候选 | 官方只测试 Linux，要求 NVIDIA 至少 24GB 显存；本机不满足 |
| Hunyuan3D-2.1 | 免费开放权重，腾讯社区许可，有限制 | 形状和纹理两阶段，公开训练代码；完整材质工作流 | 官方给出形状 10GB、纹理 21GB、合计 29GB 显存；本机不满足官方推荐路径 |
| Hunyuan3D-2mini | 免费开放权重，腾讯社区许可 | 0.6B 形状生成器，适合优先探索本机 CPU 适配；颜色需另做 | 0.6B 仅指形状生成器，不包括全部视觉编码和解码模型；CPU 端到端耗时、内存待实测 |
| TripoSG | 代码 MIT，官方权重模型卡也标注 MIT | 强调 SDF 几何细节和形状一致性；需增加颜色处理 | 官方要求 CUDA GPU 至少 8GB；其默认去背景依赖另有许可，不能只看主仓库 MIT |
| TripoSR | 官方明确代码、预训练模型均 MIT | 单图直接输出形状和顶点色，推理代码含 CPU 回退，适合作为轻量对照 | 官方默认约 6GB 显存；CPU 实测待做，不能承诺精细猫脸效果 |
| SAM 3D Objects | 开放权重采用独立 SAM 许可，需申请下载访问 | 适合从自然照片中理解物体形状、纹理和布局 | 官方 Linux、NVIDIA 至少 32GB 显存；本机不满足 |

这覆盖目前与单张照片生成实体相关的主要公开候选，并非宣称穷尽所有项目。官方自报指标来自不同测试设置，不能直接拼成统一排行榜。

## 打印适用性

渲染效果好不代表打印细节好。眼睛、鼻子和口鼻区必须有真实几何起伏，不能只靠贴图；同时检查封闭性、薄壁、孤岛、底面和切片层。TRELLIS.2 可以表达开放和非流形表面，因此输出也需要打印修复。

单张照片看不到侧面和背面，生成模型只能推断这些区域。对这张猫照，以圆脸、折耳、眼鼻位置、口鼻轮廓和橘色 / 浅色毛区作为验收重点。STL 本身不存储标准材质颜色；多色打印优先交付带共同坐标和材料分组的 3MF，另附整体彩色模型及各色 STL。

## Ollama 与训练

Ollama 的导入功能不是通用神经网络执行器。这些 3D 扩散、图像编码和网格解码管线不在其常规支持的模型架构中；即使文件同为 safetensors，也不能直接 `ollama run`。应用需要独立的本地 Python 推理服务，浏览器调用该服务。保留现有 Ollama，无需修改它。

已有免费模型，因此当前不需要从零训练基础模型。如果实测发现猫的相似度不够，先比较预训练候选，再考虑使用有权使用的猫咪 3D 资产和多视角渲染数据进行微调。训练代码存在不等于已有训练资源；也不能用一张猫照从零训练出具备一般三维理解能力的模型。若日后需要训练，再确定用户可提供的 GPU、数据与预算。

## 安装约束

- 只下载官方仓库源代码及权重；权重优先 safetensors，记录固定提交和 SHA-256。
- 在工作区独立环境安装，不改系统 Python、Ollama 或杀毒设置。
- 保持 TLS 证书校验；不使用不明整合包，也不把网络错误当成关闭防护的理由。
- 首次安装需要联网；实际推理须使用本地文件并关闭自动联网下载。
- 正式接入前必须对用户猫照实测：时间、峰值内存、正侧背预览、原始网格以及打印文件。

已核对版本：Hunyuan3D-2 源码 `f8db63096c8282cb27354314d896feba5ba6ff8a`；官方 Hugging Face 权重仓库 `f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6`。腾讯 ModelScope 权重文件版本 `0d1901d5f6300a268615cff3aff885c4fd494ff5`，两端模型文件均为 3,819,958,234 字节，SHA-256 为 `3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b`。版本和参数匹配不等于本机推理效果验收。

## 官方资料

- [腾讯 3D-2 当前许可](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)
- [腾讯 3D-2.1 权重许可](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE)
- [2mini 模型说明](https://huggingface.co/tencent/Hunyuan3D-2mini)
- [2.1 代码、训练和显存说明](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [TRELLIS.2 许可、硬件、训练代码](https://github.com/microsoft/TRELLIS.2)
- [TripoSG 代码及硬件要求](https://github.com/VAST-AI-Research/TripoSG)
- [TripoSG 权重说明](https://huggingface.co/VAST-AI/TripoSG)
- [TripoSR 代码、权重 MIT 声明](https://github.com/VAST-AI-Research/TripoSR)
- [SAM 3D Objects 官方安装要求](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Ollama 官方模型导入说明](https://docs.ollama.com/import)
