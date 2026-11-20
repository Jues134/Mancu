这个目录存放"这台机器"的配置，不是你的作品。

首次启动时程序会自动创建它；里面出现的文件：

  providers.json    模型供应商与 API Key。在界面 /settings 里保存时自动生成。
                    ★ 含真实密钥，绝对不要发给别人。
  feishu.json       飞书应用凭据（App ID / App Secret）。只有接入飞书时才需要。
                    ★ 含真实密钥，绝对不要发给别人。
  feishu_oauth.json 飞书授权状态，同样是机器私有数据。
  studio.json       媒体模式等偏好，例如 {"media_mode": "prompt"}。
  mcp_tools.json    功能开关，例如 {"knowledge": true, "feishu": true}。

本目录中的 *.example.json 只是字段说明模板，不含任何真实凭据，可以随程序分发。
要启用某项配置，请复制模板为去掉 .example 的文件名再填写，或直接在界面里配置
（推荐，程序会处理好权限与格式）。

备份时请把 userdata\ 和 config\providers.json 分开存放：前者是你的作品，
后者是密钥。
