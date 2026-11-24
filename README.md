# Munneke2023 心脏模型

Python 复现保留原 PRef 的 20 个心肌 Patch（左右房、右室各 1 个，左室游离壁 12 个，室间隔 5 个）和 52 个动态状态。心脏采用原 C/Lsi 肌节方程、TriSeg、心包、瓣膜及各 Patch 激活时序。这个机械内核通过 TV、PV、SA、PA 四个循环端口连接外部压力，并接受指定胸膜压。参数文件取自原 PRef，沿用原心脏参数。冠脉、供需控制和呼吸方程均在 Python 心脏内核的接口之外。

- `cpmodel/heart.py` 为独立心脏内核，`heart_fast.py` 提供可选的 Numba 实现。
- `data/munneke_parameters.json` 为 PRef 提取参数及初态，`tests/fixtures/munneke_reference.json` 为独立 MATLAB 对照。
- `original_matlab/` 保存原 MATLAB 循环源码及 PRef 快照，包含原论文的冠脉部分。
- `run_reference_beat.m` 从发布的 PRef 运行一个 MATLAB 参考心搏，导出四腔压力和容积、主动脉和肺动脉压力、各端口流量。CSV 中的 `q_vc_ra_ml_s` 是腔静脉到右房的流量。

## 验证 Python 心脏内核

在本目录的 PowerShell 中执行。Numba 仅在使用加速内核时需要。

```powershell
$run = 'D:\data\OneDrive\Desktop\CardioPINN-Research-Agent\runs\desktop_model_exports_20261123\runtime'
New-Item -ItemType Directory -Force -Path $run | Out-Null
$env:TEMP = $env:TMP = $env:MPLCONFIGDIR = $run
$env:PYTHONPYCACHEPREFIX = Join-Path $run 'pycache'
$env:NUMBA_CACHE_DIR = Join-Path $run 'numba'
& 'D:\anaconda\envs\biomodel\python.exe' -m unittest discover -s tests -v
```

## 运行一个原 MATLAB 心搏

在本目录的 MATLAB 中运行。结果保存在项目的 `runs` 目录。

```matlab
run_reference_beat('D:/data/OneDrive/Desktop/CardioPINN-Research-Agent/runs/desktop_model_exports_20261123/runtime/matlab');
```

[Munneke2023 原源码及提交 85d9d8b](https://github.com/CircAdapt/Munneke2023/tree/85d9d8bdaddb6b09902ba3578cba1721caa6e145)
