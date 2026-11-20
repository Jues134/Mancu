# 安装与启动

一句话：解压后双击 `start.cmd`。它会调用 `start.ps1` 选择 Python、建立隔离环境、补齐依赖、构建前端并启动服务。

## 标准启动

普通使用直接双击根目录的 `start.cmd`。需要传递高级参数时，在项目目录打开 PowerShell，然后运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

脚本会自动完成：

1. 检查 64 位 Windows 和 PowerShell 版本。
2. 从 Python Launcher 与 PATH 中选择可用的最高版本 Python（最低 3.11）。
3. 建立项目专用 `.venv`，依赖永远安装到这个环境，并用它启动项目。
4. 比对 `requirements.txt` 指纹，逐项导入核心模块，并执行 `pip check`；任何必需依赖缺失或冲突都不会启动。
5. 缺依赖时优先按清华、阿里云、腾讯、官方 PyPI 的顺序安装。
6. 有预构建前端时直接使用；需要构建时优先使用 npmmirror、腾讯、官方 npm 源安装并构建。
7. 启动后端，端口就绪后打开浏览器。

安装和启动过程同时记录在 `logs\start-powershell.log`。

## 常用参数

```powershell
.\start.ps1 -Help
.\start.ps1 -Port 8200
.\start.ps1 -Build
.\start.ps1 -NoBrowser
.\start.ps1 -CheckOnly
.\start.ps1 -OfficialSources
.\start.ps1 -Mirror "https://your-pypi/simple"
```

`-Mirror` 是 pip 源；前端 npm 仍按 npmmirror、腾讯、官方源回退。

## 离线安装

在一台 Windows 架构和 Python 版本与目标机一致、且能够联网的电脑上运行：

```powershell
.\start.ps1 -PrepareOffline
```

脚本把 Python wheel 下载到 `backend\wheels` 后退出。把整个项目复制到离线电脑，再运行：

```powershell
.\start.ps1 -Offline
```

前端 `dist` 已经随项目提供，所以离线启动不需要 Node.js。若使用 `-Build`，则必须同时带上可用的 `frontend\node_modules`。

## Python 选择和依赖完整性

- 优先选择已安装的最高版本 Python，而不是盲目使用 PATH 中第一项。
- 自动建立 `.venv`，解决“pip 装到 A、程序从 B 启动”的问题。
- 如果机器后来安装了更高版本 Python，下次启动会重建 `.venv` 并迁移到高版本。
- `requirements.txt` 改变后会自动重新安装。
- 启动前同时验证 FastAPI、Uvicorn、Pydantic、HTTPX、NumPy、Pillow、scikit-image、ONNX Runtime、tokenizers，并执行 `pip check` 检查间接依赖。

如果 Python 新版本暂时没有某个三方包的 Windows wheel，安装会明确失败，不会带着残缺依赖继续启动。此时安装 Python 3.13 后重新运行即可。

## 前端与 Node.js

普通发布包已有 `frontend\dist\bundle.js`，因此日常启动不需要 Node.js。以下情况才需要 Node.js 20 或更高版本：

- `frontend\dist\bundle.js` 不存在；
- 显式使用 `-Build`；
- 修改了前端源代码。

## 停止

启动窗口仍在时直接按 `Ctrl+C`。也可以双击根目录的 `stop.cmd`。自定义端口可在命令行中传入：

```bat
stop.cmd
stop.cmd 8200
```

`stop.cmd` 只结束能够识别为 Mancu 的 Python 进程；如果端口属于其他程序，它会拒绝误杀并显示错误。

## 常见问题

| 现象 | 处理 |
|---|---|
| 双击 `.ps1` 只打开文本 | 请双击 `start.cmd`；不要直接双击 `start.ps1` |
| 启动窗口中文乱码 | 请确认使用新版 `start.cmd`；它会自动切换 UTF-8，不要从旧包复制 CMD |
| 找不到 Python | 安装 64 位 Python 3.11+，勾选 Add python.exe to PATH，然后重新打开 PowerShell |
| 所有 pip 源失败 | 检查网络/代理；有代理时用 `-OfficialSources`，有私有源时用 `-Mirror URL` |
| pip 安装结束但仍报依赖不全 | 查看上方具体 import 或 `pip check` 错误，以及 `logs\start-powershell.log` |
| 找不到 Node.js | 只有前端需要重建；安装 Node.js 20+，或者恢复发布包中的 `frontend\dist` |
| 端口被其他程序占用 | 使用 `-Port 8200`，或先关闭占用端口的程序 |
| 中文目录或空格目录 | PowerShell 参数全部按路径对象传递，正常支持；个别三方安装包若自身不支持，会在安装输出中明确报错 |

