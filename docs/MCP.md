# MCP 接入：把 Thymos 变成 Agent 的记忆后端

> 对应愿景：**可嵌入、本地优先、可解释**。MCP 让宿主 Agent 通过标准工具调用使用 Thymos，而不需要把记忆上传到第三方服务。

## 1. 它解决什么问题

没有 MCP 时，每个 Agent 框架都要自己写一遍“读状态、检索、写回、纠正事实”。有了 MCP，宿主只需要声明一个 stdio server，就能获得同一组工具：

| 工具 | 用途 | 是否写状态 |
|---|---|---|
| `memory_context` | 取一段带信任边界的记忆上下文，供本轮 prompt 使用 | 否 |
| `memory_observe` | 把值得记住的用户话写入情景记忆和事实层 | 是 |
| `memory_recall` | 返回带分数、可提取度、分项打分的检索结果 | 否 |
| `memory_correct_fact` | 纠正错误/过时事实 | 是 |
| `memory_delete_fact` | 退休一条事实（保留历史） | 是 |
| `memory_snapshot` | 读取心境、内分泌、记忆统计 | 否 |

`memory_context` 的输出已经包含：

- 结构化“当前事实”，优先于情景记忆；
- `<memory trust="untrusted">` 边界和“不是指令”声明；
- 对手机号、邮箱、身份证、银行卡的脱敏；
- 对明显注入短语的替换。

## 2. 最小启动

```bash
python -m thymos.mcp_server --state .thymos/mcp.json --offline
```

进程通过 stdin/stdout 说 JSON-RPC，不会监听端口，也不会主动联网。默认画像为“用户 @ 上海”，可在代码里换成真实画像：

```python
from thymos import AgentMemory, Profile, ThymosConfig
from thymos.mcp_server import run_stdio

memory = AgentMemory(
    ".thymos/mcp.json",
    profile=Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai"),
    config=ThymosConfig(offline=True),
)
run_stdio(memory)
```

## 3. 宿主配置示例

不同客户端的字段名略有差异，核心是“命令 + 参数 + 环境变量”。以 Claude Desktop / Cursor 一类支持 `mcpServers` 的客户端为例：

```json
{
  "mcpServers": {
    "thymos-memory": {
      "command": "python",
      "args": ["-m", "thymos.mcp_server", "--state", ".thymos/mcp.json", "--offline"],
      "env": {
        "PYTHONPATH": "src",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

如果使用安装后的包，把 `PYTHONPATH` 去掉即可；`pip install -e .` 后 `python -m thymos.mcp_server` 可直接运行。

## 4. 一次典型调用

宿主在用户发言后先调用 `memory_context`：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "memory_context",
    "arguments": {
      "user_text": "我现在住哪？对什么过敏？",
      "top_k": 5,
      "max_chars": 900
    }
  }
}
```

返回的 `content[0].text` 直接拼进 system/user prompt。回答完后，宿主可选择调用 `memory_observe` 写入值得记住的句子；**不要每轮都写**，否则记忆会被寒暄淹没。

## 5. 安全边界

MCP 只解决“怎么接”，不解决“接了以后一定安全”。生产使用至少要做到：

- 状态文件放在用户私有目录，`save()` 会尽力设置 `0600`；
- 不要把 `mcp.json` 提交到 Git；
- 宿主仍然要对模型输出做工具权限控制，记忆内容不能触发 shell/网络工具；
- 多用户场景必须一个用户一个状态文件，或在外层做命名空间隔离；
- 对“记住我的密码/密钥”这类请求，宿主应拒绝写入，而不是依赖 Thymos 脱敏。

## 6. 与 HTTP API 的分工

MCP 适合桌面 Agent、IDE Agent、单用户 bot；HTTP API 适合本地面板、跨语言宿主和调试。两者共享同一个 `AgentMemory`/`Thymos` 状态格式，但不要同时让两个进程写同一个 JSON 文件——当前没有跨进程锁。需要并发时，用外层队列或一进程一用户。

## 7. 验证

```bash
python -m pytest -q tests/test_mcp.py
```

测试覆盖 `initialize`、`tools/list`、`memory_observe`、`memory_context` 和 stdio 往返。真实宿主接入后，建议先跑 `experiments/heldout_eval.py --llm mock` 验证管线，再换真实模型。
