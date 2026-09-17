# 第三方来源

本仓库上传应用代码和集成逻辑；不重新分发模型权重、第三方二进制、Python/Conda 环境或用户样本。各依赖与模型的许可应以它们自身仓库、分发包及固定版本的许可证为准。

- Hunyuan3D 源码：[Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)，本机固定提交记录见 `DEPLOYMENT.md`。
- 2mini 模型及许可证：[tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini)。这里没有把其自定义许可改称为统一的 MIT/Apache 许可。
- 抠图：[rembg](https://github.com/danielgatis/rembg)，使用其官方 U2Net 下载与校验信息。
- 基础模型环境：[PyTorch](https://pytorch.org/)、[Transformers](https://github.com/huggingface/transformers)、[Diffusers](https://github.com/huggingface/diffusers)。
- 几何和打印：[trimesh](https://github.com/mikedh/trimesh)、[Manifold](https://github.com/elalish/manifold)、[scikit-image](https://github.com/scikit-image/scikit-image)。
- 多图重建：[COLMAP](https://github.com/colmap/colmap)、[OpenMVS](https://github.com/cdcseacave/openMVS)。
- 前端及服务依赖的完整名称和固定版本见 `package-lock.json`、`requirements-lock.txt`、`requirements-xpu-installed.txt`。

官方 2.1 训练资料提供后续适配参考，本仓库没有声称已完成其训练代码到 Mini Turbo 的移植。另有可选的离线 2.1 形状对照脚本；固定版本与文件哈希见 `CAT_QUALITY.md`，仍受腾讯自定义许可约束。额外依赖 [timm](https://github.com/huggingface/pytorch-image-models) 固定于 `requirements-shape21.txt`；不分发其权重。
