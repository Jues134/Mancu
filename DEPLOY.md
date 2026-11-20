# 漫簇 Mancu：Windows 部署与运维

本文面向安装、交付、迁移和排错人员。项目代码与启动逻辑保持不变；本文记录实际验证环境、部署步骤和已知兼容性边界。

快速体验请先阅读 [README.md](README.md)。

## 目录

- [部署选择](#部署选择)
- [环境要求](#环境要求)
- [兼容性矩阵](#兼容性矩阵)
- [源码版部署](#源码版部署)
- [便携版部署](#便携版部署)
- [离线部署](#离线部署)
- [模型配置](#模型配置)
- [启动与停止](#启动与停止)
- [局域网与远程访问](#局域网与远程访问)
- [数据、备份与恢复](#数据备份与恢复)
- [日志与健康检查](#日志与健康检查)
- [故障排查](#故障排查)
- [升级、迁移与卸载](#升级迁移与卸载)
- [发布检查清单](#发布检查清单)
- [参数与环境变量](#参数与环境变量)

## 部署选择

| 方案 | 优点 | 限制 | 推荐用途 |
|---|---|---|---|
| 源码版 `Mancu-source` | 在目标电脑建立环境，适合维护和排错 | 需要 Python；可能遇到系统编码或 wheel 问题 | 正式使用、开发、远程电脑 |
| 便携版 `Mancu-portable` | 不要求系统安装 Python | 原生 DLL 对 CPU、运行库和系统环境敏感 | 环境接近验证机的临时使用 |
| 离线源码版 | 内网电脑不访问 PyPI | 需要提前准备相同架构和 Python 版本的 wheel | 内网和受控环境 |

便携版不代表兼容性一定更高。NumPy、SciPy、ONNX Runtime 等原生组件换机后仍可能崩溃。

## 环境要求

### 源码版

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| 操作系统 | Windows 10/11 64 位 | Windows 11 64 位 |
| PowerShell | 5.1 | 5.1 或 7 |
| Python | 3.11 64 位 | 3.13 64 位 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 2 GB，不含媒体结果 | 10 GB 以上 |
| Node.js | 普通启动不需要 | 重建前端时使用 20+ |
| 网络 | 安装依赖和在线模型需要 | 稳定网络 |

### 便携版

便携版不要求安装 Python 或 Node.js，但依赖：

- 64 位 Windows；
- CPU 支持原生依赖使用的指令集；
- 可用的 Microsoft Visual C++ 运行组件；
- 安全软件未隔离 DLL 或 `.pyd` 文件；
- 完整复制的 `runtime` 目录。

### 已验证基线

- Windows 11 64 位
- Windows PowerShell 5.1
- Python 3.13.9 64 位
- 后端测试 298 项通过
- HTTP 首页返回 200

其他 Windows、云电脑和虚拟机需要单独验收。

## 兼容性矩阵

| 场景 | 状态 | 说明 |
|---|---|---|
| Windows 11 x64 + Python 3.13 x64 | 已验证 | 推荐源码版 |
| Windows 10 x64 | 目标支持 | 需要实际测试依赖和运行库 |
| Windows 中文区域设置 | 有已知限制 | pip 可能按 CP936 读取 UTF-8 依赖文件 |
| 远程云电脑 / 虚拟机 | 有条件支持 | CPU 指令集、内存和运行库可能不同 |
| Python 3.11 / 3.12 | 脚本允许 | 依赖版本由 pip 实际解析 |
| Python 3.13 | 推荐 | 当前验证基线 |
| 更新的 Python | 不保证 | 原生包可能尚无对应 wheel |
| 便携版跨电脑复制 | 不保证 | 原生 DLL 可能发生内存访问冲突 |
| Linux / macOS | 未提供入口 | 不属于当前交付范围 |

## 源码版部署

### 1. 放置项目

复制完整 `Mancu-source` 目录，建议使用简短路径：

```text
D:\Mancu-source
```

不要只复制启动脚本。

### 2. 安装 Python

安装 Python 3.13 64 位并勾选 **Add Python to PATH**。重新打开 PowerShell：

```powershell
py -3.13 --version
```

没有 `py` 命令时使用：

```powershell
python --version
```

### 3. 标准启动

双击：

```text
start.cmd
```

或在 PowerShell 中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

脚本会创建 `.venv`，并始终使用这个环境安装依赖和启动服务。

### 4. 国内源顺序

Python 依赖默认尝试：

1. 清华 PyPI 镜像；
2. 阿里云 PyPI 镜像；
3. 腾讯 PyPI 镜像；
4. 官方 PyPI。

前端依赖默认尝试：

1. npmmirror；
2. 腾讯 npm 镜像；
3. 官方 npm。

镜像回退只解决网络和同步问题，不能解决本地文件编码错误。

### 5. CP936/GBK 编码错误

#### 现象

```text
UnicodeDecodeError: 'gbk' codec can't decode byte ...
decoding with 'cp936' codec failed
```

调用栈通常位于 `pip\_internal\req\req_file.py`，切换镜像后会重复出现。

#### 原因

`backend\requirements.txt` 是 UTF-8 文件并包含中文注释。部分 Windows Python/pip 使用 CP936 读取它。错误发生在本地，不是镜像故障。

#### 临时处理

PowerShell：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

CMD：

```bat
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
start.cmd
```

环境变量只影响当前窗口及其子进程。

如果仍然失败，可删除本次失败产生的项目 `.venv`，再用 Python 3.13 64 位重新运行。删除前必须确认目标确实是当前项目中的 `.venv`。

### 6. 手动安装

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip check
.\start.ps1
```

## 便携版部署

### 1. 复制完整目录

必须同时保留：

```text
Mancu-portable/
├─ runtime/
├─ backend/
├─ frontend/
├─ start.cmd
├─ start.ps1
└─ stop.cmd
```

### 2. 启动

双击 `start.cmd`。脚本会检查内置 Python 和关键依赖，然后启动网页服务。

### 3. 已知失败

可能出现：

```text
python.exe - 应用程序错误
该内存不能为 read
```

或：

```text
The bundled Python dependencies are incomplete or damaged.
```

它们可能来自同一根因：NumPy、SciPy、ONNX Runtime 或其他 DLL 在目标电脑加载时崩溃。影响因素包括：

- Visual C++ 运行库缺失或损坏；
- CPU 指令集不匹配；
- 虚拟机未暴露需要的 CPU 能力；
- 安全软件隔离 DLL；
- 文件复制不完整；
- 便携环境与目标系统不兼容。

处理顺序：

1. 查看 `logs\start-powershell.log`；
2. 检查安全软件隔离记录；
3. 确认 `runtime` 完整；
4. 安装或修复 Microsoft Visual C++ x64 运行库；
5. 仍失败时改用源码版。

不要把其他 Python 的 `site-packages` 直接复制进便携目录，也不要混用系统 pip。

## 离线部署

联网电脑准备 wheel：

```powershell
.\start.ps1 -PrepareOffline
```

把源码目录和 `backend\wheels` 复制到相同架构、相同 Python 版本的目标电脑：

```powershell
$env:PYTHONUTF8 = "1"
.\start.ps1 -Offline
```

前端已有 `frontend\dist` 时无需 Node.js。

## 模型配置

启动后访问：

```text
http://127.0.0.1:8100/settings
```

配置供应商、Base URL、API Key、文本模型、视觉模型、生图模型和生视频模型，并执行“测试连接”。

环境变量示例：

```powershell
$env:STUDIO_LLM_BASE_URL = "https://api.example.com/v1"
$env:STUDIO_LLM_MODEL = "your-model"
$env:STUDIO_LLM_API_KEY = "your-api-key"
.\start.ps1
```

`.env.example` 只是模板，程序不会自动加载 `.env`。

只生成提示词：

```powershell
$env:STUDIO_MEDIA_MODE = "prompt"
.\start.ps1
```

真实生成媒体：

```powershell
$env:STUDIO_MEDIA_MODE = "full"
.\start.ps1
```

真实调用可能产生费用，先用小任务验证模型和接口。

## 启动与停止

### 启动

双击 `start.cmd`，或使用：

```powershell
.\start.ps1
.\start.ps1 -Port 8200
.\start.ps1 -NoBrowser
.\start.ps1 -CheckOnly
.\start.ps1 -Build
.\start.ps1 -OfficialSources
.\start.ps1 -Mirror "https://your-mirror.example/simple"
```

`start.cmd` 会先切换控制台到 UTF-8，再调用 `start.ps1`。

### 停止

默认端口：

```bat
stop.cmd
```

自定义端口：

```bat
stop.cmd 8200
```

`stop.cmd` 只停止能够识别为 Mancu 的 Python 进程；端口属于其他程序时会拒绝操作。

## 局域网与远程访问

默认只监听 `127.0.0.1`。

允许局域网访问：

```powershell
.\start.ps1 -ListenHost 0.0.0.0 -NoBrowser
```

其他设备访问：

```text
http://目标电脑局域网IP:8100
```

> [!WARNING]
> 当前服务没有面向公网设计的用户登录和访问控制。不要把 8100 端口直接映射到公网。远程使用应通过带 TLS、认证和访问限制的反向代理或 VPN。

## 数据、备份与恢复

### 数据位置

| 路径 | 内容 | 备份 |
|---|---|---:|
| `backend\userdata\studio.db` | 任务、设置、资料索引 | 必需 |
| `backend\userdata\uploads` | 上传资料 | 必需 |
| `backend\userdata\assets` | 生成媒体 | 必需 |
| `backend\userdata\logs` | 应用日志 | 可选 |
| `backend\config\providers.json` | 供应商配置和 Key | 敏感 |
| `logs\start-powershell.log` | 启动和安装日志 | 排错时保留 |

### 备份

1. 运行 `stop.cmd`；
2. 确认网页无法访问；
3. 复制整个 `backend\userdata`；
4. 加密备份 `backend\config\providers.json`；
5. 不要把备份上传到公开仓库。

### 恢复

1. 在新目录完成一次空环境启动；
2. 停止服务；
3. 用备份覆盖 `backend\userdata`；
4. 恢复本机配置；
5. 启动并检查历史任务和资料库。

不要在服务运行时覆盖 SQLite 数据库。

## 日志与健康检查

| 文件 | 用途 |
|---|---|
| `logs\start-powershell.log` | Python 选择、依赖安装、前端检查和启动 |
| `backend\userdata\logs\app.log` | 请求、任务、配置和运行状态 |
| `backend\userdata\logs\error.log` | 异常与调用栈 |

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8100/ -UseBasicParsing
```

HTTP 200 只表示网页服务已监听，不代表模型 Key、余额和媒体后端全部可用。

## 故障排查

### 安装阶段

| 错误 | 判断 | 处理 |
|---|---|---|
| `UnicodeDecodeError / cp936` | 本地依赖文件编码 | 设置 `PYTHONUTF8=1` 后重试 |
| 所有镜像报同一错误 | 通常不是镜像问题 | 查看第一段错误 |
| `No matching distribution` | Python 没有对应 wheel | 使用 Python 3.13 x64 |
| SSL / 代理错误 | 网络或企业代理 | 配置代理或可访问镜像 |
| `pip check` 失败 | 间接依赖冲突 | 重建项目 `.venv` |

### 启动阶段

| 错误 | 判断 | 处理 |
|---|---|---|
| 双击 `.ps1` 打开文本 | Windows 文件关联 | 双击 `start.cmd` |
| 中文日志乱码 | 入口或控制台编码错误 | 使用当前目录的 `start.cmd` |
| 端口被占用 | 其他程序监听端口 | 改用 `-Port 8200` |
| 浏览器未打开 | 服务未启动或禁用浏览器 | 手动访问并看日志 |
| `python.exe` 内存不能为 read | 原生依赖崩溃 | 便携版改用源码版 |
| 内置依赖损坏 | 可能是 DLL 加载失败 | 检查运行库和安全软件 |

### 运行阶段

| 错误 | 判断 | 处理 |
|---|---|---|
| 网页正常但 Agent 不工作 | 未配置模型 | 在设置中测试供应商 |
| 401 / 403 | Key 无效或权限不足 | 检查 Key 和模型权限 |
| 429 | 限流或余额不足 | 降低频率或更换供应商 |
| 图片或视频失败 | 模型或接口不匹配 | 检查供应商媒体能力 |
| `database is locked` | 多个进程写同一个库 | 运行 `stop.cmd` |
| 页面仍显示旧内容 | 浏览器缓存或未构建 | `Ctrl+F5` 或 `-Build` |

### 提交故障信息

请提供：

1. Windows 版本和系统语言；
2. 物理机、云电脑或虚拟机；
3. CPU 型号和架构；
4. Python 版本和来源；
5. 源码版或便携版；
6. 完整错误文本；
7. `logs\start-powershell.log`；
8. `backend\userdata\logs\error.log`。

API Key 和隐私资料必须打码。

## 升级、迁移与卸载

### 升级

1. 停止旧服务；
2. 备份数据和本机配置；
3. 将新文件放到新目录；
4. 先完成一次新目录启动；
5. 停止新服务并恢复数据；
6. 再次启动和验证；
7. 确认无误后归档旧目录。

### 迁移

源码版不要复制 `.venv` 到另一台电脑。复制源码和数据后，让目标电脑重新建立 `.venv`。

便携版可以复制完整目录尝试，但仍需重新验收原生依赖兼容性。

### 卸载

1. 运行 `stop.cmd`；
2. 备份数据；
3. 删除项目目录。

源码版虚拟环境位于项目内部，不会卸载系统 Python。

## 发布检查清单

- [ ] README 区分源码版和便携版
- [ ] 标明验证环境和兼容性限制
- [ ] 根目录包含 `start.cmd`、`stop.cmd` 和 `start.ps1`
- [ ] 发布文件名使用 ASCII
- [ ] 源码包不含 `.venv`、`node_modules` 和日志
- [ ] 便携包包含完整 `runtime`
- [ ] `frontend\dist\bundle.js` 存在
- [ ] `backend\models` 内置模型存在
- [ ] 不含真实供应商配置、数据库、上传、媒体和 API Key
- [ ] 在干净目录执行启动测试
- [ ] HTTP 首页返回 200
- [ ] `stop.cmd` 能停止服务
- [ ] 中文日志显示正常

## 参数与环境变量

### 启动参数

| 参数 | 作用 |
|---|---|
| `-Port 8200` | 修改服务端口 |
| `-ListenHost 0.0.0.0` | 允许其他电脑访问 |
| `-NoBrowser` | 不自动打开浏览器 |
| `-CheckOnly` | 只检查环境 |
| `-Build` | 重建前端 |
| `-OfficialSources` | 只使用官方源 |
| `-Mirror URL` | 指定 pip 镜像 |
| `-PrepareOffline` | 下载离线 wheel |
| `-Offline` | 从 `backend\wheels` 安装 |

### 常用环境变量

| 变量 | 作用 |
|---|---|
| `PYTHONUTF8` | 强制 Python UTF-8 模式 |
| `PYTHONIOENCODING` | Python 标准输入输出编码 |
| `STUDIO_LLM_BASE_URL` | 文本模型接口 |
| `STUDIO_LLM_API_KEY` | 文本模型 Key |
| `STUDIO_LLM_MODEL` | 文本模型名 |
| `STUDIO_LLM_VISION_MODEL` | 视觉模型名 |
| `STUDIO_IMAGE_MODEL` | 生图模型名 |
| `STUDIO_VIDEO_MODEL` | 生视频模型名 |
| `STUDIO_MEDIA_MODE` | `prompt` 或 `full` |
| `STUDIO_DATA_DIR` | 自定义数据目录 |
| `STUDIO_CONFIG_DIR` | 自定义配置目录 |
| `MANCU_MEDIA_BASE_URL` | 外部媒体后端地址 |

环境变量优先于本机配置。不要在截图、日志或公开仓库中暴露 API Key。

