# 阶段 0 环境档案

日期：2026-11-01
机器：Windows 11 企业版 10.0.26300

## 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060，12288 MiB，驱动 610.88（CUDA UMD 13.3） |
| CPU | AMD Ryzen 5 5600G，6 核 12 线程 |
| 内存 | 63.4 GB（空闲 50.4 GB 时测得） |
| 磁盘 | C: 剩 7.6 GB（不使用）；D: 剩 428.9 GB（全部产物在此） |

## 软件

| 项 | 值 |
|---|---|
| Python | 3.11.6（继承自 ComfyUI 的 `.ext` 解释器） |
| 虚拟环境 | `D:\projects\minimind-async-perception\minimind\.venv`（`--system-site-packages`） |
| torch | **2.9.1+cu130**，`torch.cuda.is_available() == True` |
| transformers | **4.57.6**（正好是 MiniMind requirements 指定版本） |
| accelerate | 1.14.0 |
| modelscope | 1.40.2 |
| git | 2.55.0.windows.3 |

### 为什么复用 ComfyUI 的解释器

第一次用系统 Python 3.11.9 建独立 venv，从清华 PyPI 镜像装到的是 **torch 2.6.0+cpu**（`cuda False`），
无法用于 GPU 实验。改用 `D:\Program Files\ComfyUI-aki-v1.4\.ext\python.exe` 建
`--system-site-packages` 虚拟环境后，直接继承到 torch 2.9.1+cu130 与 transformers 4.57.6，
省掉 2.4 GB 下载，也没有污染 ComfyUI 自己的环境（新包只装进本项目 venv）。

注：启动时有一条无害告警 `Failed to find CUDA. Install CUDA Toolkit ... set CUDA_HOME`（来自 triton），
不影响 torch 使用 GPU，实测 `cuda True`。

## MiniMind

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/jingyaogong/minimind |
| commit | `233d00fa068ec2b8c0d4f07b703934d6391a5a73` |
| commit 日期 | 2026-10-27 14:57:05 +0800 |
| 权重 | `gongjy/minimind-3`（ModelScope），128 MB，`model.safetensors` |
| 结构 | Dense，hidden 768 / 8 层 / 词表 6400 / 8 attn heads / 4 kv heads |
| 加载方式 | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`，fp16 |

**26M 说明**：26M 对应的是 `minimind2-small`（2025-04-26 历史版本，dim 512、kv_heads 2），
与 minimind-3 结构不同，属于上一代模型。**降级到 26M 需要切换对应历史 commit 的模型代码**，
不是换一个 checkpoint 名称即可。本阶段未使用 26M，阶段 2 若显存或时间吃紧再评估。

## 基线结果（阶段 0）

- 固定 20 条 prompt，`seed=1234+i`，`temperature=0.85`，`top_p=0.9`，`open_thinking=true`，`max_new_tokens=128`
- 20/20 全部完成，无 OOM、无崩溃
- **显存峰值 0.131 GB**（远低于 11.5 GB 阈值）
- **平均 45.38 tok/s**；首条因预热 20.4 tok/s，其余 37.6–52.4 tok/s
- 总生成耗时 56.53 s
- 产物：`reports/baseline.json`、`logs/baseline.log`

## 关键代码事实（决定阶段 1 可行性）

`model/model_minimind.py` 的 `MiniMindForCausalLM.generate`（第 262–292 行）**不是**
`GenerationMixin` 的高层封装，而是自己写的显式逐 token 解码循环：

- 显式维护 `past_key_values`，用 `input_ids[:, past_len:]` 只喂新增 token；
- 位置编码按 `start_pos = past_len` 递增；
- 支持 `streamer` 与 `return_kv`。

结论：在解码循环里向 `input_ids` 尾部追加注入 token，下一轮 forward 会自然把它们并入 KV cache，
**不需要重算前缀**。这使得「非阻塞中途注入」在实现上是干净的，而不是假的。

## 已知限制

1. 基线只有 20 条、每条约 128 token，统计功效有限。
2. 64M 模型的思考链较浅（部分回答直接以 `</think>` 空思考开头），轨迹差异可能偏小；
   阶段 3 必须同时报「同 seed 重采样的自然波动」作为对照，否则容易把随机噪声当成注入效果。
3. 控制台中文在 GBK 代码页下显示为乱码，但 `logs/` 与 `reports/` 内的文件均为 UTF-8，内容正确。
