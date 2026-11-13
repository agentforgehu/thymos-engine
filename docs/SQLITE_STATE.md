# SQLite 状态后端

> 版本：0.3.4（2026-11-13）
>
> 目标不是“把所有东西塞进一个数据库就成熟了”，而是把**记忆条目**和**引擎状态**从一次全量 JSON 重写里拆出来，并给 schema 一个明确的版本与迁移入口。

## 1. 为什么不能只“启用 sqlite_store.py”

旧版 `SQLiteMemoryStore` 只持久化 `MemoryItem` 和关联边；`Thymos.save()` 仍然调用 `to_dict()`，把 `memory` 全量写进 `state.json`。于是：

- 即使传入 SQLite store，`state.json` 还是随记忆条数线性膨胀；
- 每次保存仍然重写整个 JSON；
- 心境、内分泌、事实、事件、审计轨迹仍没有数据库版本；
- 旧状态文件没有 schema 版本，字段一改就可能静默丢数据。

0.3.4 把 SQLite 提升为**状态后端**：同一个数据库里同时保存记忆表、关联边、引擎状态行和 schema 版本。

## 2. 用法

### 2.1 新建 SQLite 状态

```bash
python -m thymos --state .thymos/agent.sqlite --offline init --name 林墨 --city 杭州
python -m thymos --state .thymos/agent.sqlite --offline perceive "我叫林墨，对花生过敏。"
python -m thymos --state .thymos/agent.sqlite --offline snapshot
```

路径以 `.sqlite`、`.sqlite3` 或 `.db` 结尾时自动选择 SQLite；`.json` 保持旧行为，方便回滚。

Python：

```python
from thymos import AgentMemory, Profile

memory = AgentMemory(
    ".thymos/agent.sqlite",
    profile=Profile(name="林墨", location_name="杭州"),
)
memory.observe("我叫林墨，对花生过敏。", importance=0.8)
memory.save()
```

`AgentMemory`、`Thymos`、CLI、HTTP API 和 MCP 共用同一个 `state_path`，不需要额外的开关。

### 2.2 从旧 JSON 迁移

```bash
python -m thymos migrate .thymos/agent.json .thymos/agent.sqlite
```

迁移规则：

- 源 JSON **不删除、不改写**；
- 目标必须是空数据库，或只有表结构但没有 `engine_state`；
- 目标已有 `engine_state` 时拒绝覆盖，避免误合并；
- 记忆写入 `memories`，关联写入 `edges`，其余状态写入 `meta.engine_state`；
- `engine_state.memory` 固定为 `[]`，不再复制一份记忆。

迁移后先用 `snapshot` 或 `recall` 验证，再切换宿主配置。旧 JSON 可以保留一个版本周期。

## 3. Schema

当前 `PRAGMA user_version = 1`：

```sql
CREATE TABLE memories (
  id   TEXT PRIMARY KEY,
  data TEXT NOT NULL          -- MemoryItem.to_dict() 的 JSON
);

CREATE TABLE edges (
  a      TEXT NOT NULL,
  b      TEXT NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY (a, b)
);

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL          -- engine_state 的 JSON
);
```

保存语义：

- `SQLiteMemoryStore.add()`：立即写入该条记忆并 commit；
- `SQLiteMemoryStore.flush()`：把 review、遗忘更新、prune 后的内存对象写回；`Thymos.save()` 会调用；
- `SQLiteMemoryStore.save_state()`：写 `meta.engine_state`；
- `Thymos.save()` 在 SQLite 模式下先 `flush()`，再写状态行，最后返回；
- `close()` 做一次 WAL checkpoint，方便拷贝数据库文件。

这不是“每条记忆一个事务的完整事件溯源”。进程如果在 `observe()` 与 `save()` 之间崩溃，记忆可能已经落库，但心境/事件还没落库；宿主应把 `save()` 放在一轮对话结束或关键状态变更之后。

## 4. 版本与迁移边界

- 数据库 schema 版本用 `PRAGMA user_version` 记录；
- 打开高于当前支持的 schema 会直接报错，不会用旧代码猜测新字段；
- 引擎状态里的 `version` 高于当前 `Thymos.VERSION` 会直接报错；
- 当前只实现 v1；后续新增字段时在 `_ensure_schema()` 里追加 `v1 -> v2` 迁移，而不是改 `CREATE TABLE` 后让旧库自生自灭；
- JSON 状态仍保留 `version` 字段，便于迁移工具判断。

这解决的是**“老状态文件被新版本静默作废”**的一部分；它不承诺跨大版本自动兼容所有语义变化。语义迁移仍需要显式脚本和回归样本。

## 5. 性能与边界

本机合成数据（短中文记忆、Windows、Python 3.12）的一次保存：

| 记忆条数 | JSON 保存 | JSON 大小 | SQLite 保存 | SQLite 主库大小 |
|---:|---:|---:|---:|---:|
| 285 | 6.5 ms | 0.17 MB | 12.2 ms | <0.01 MB* |
| 1000 | 23.5 ms | 0.60 MB | 24.0 ms | 0.46 MB |
| 3000 | 68.2 ms | 1.78 MB | 62.0 ms | 1.52 MB |

\* WAL 尚未 checkpoint 时主库很小，数据在 `-wal` 文件里；`close()` 会 checkpoint。上表不是 benchmark，只是确认没有数量级退化。

必须说清楚 SQLite **没有**解决什么：

- 检索仍然把工作集加载到内存并线性扫描；几千条时真正的瓶颈可能是 `most_similar`/`retrieve`，不是 `json.dump`；
- 每条记忆仍是 JSON blob，没有 FTS5、向量索引或列式字段；
- 单进程内用 `RLock` 串行化，跨进程只依赖 SQLite 的 WAL/`busy_timeout`；没有应用层多写者协议；
- 没有加密、保留期、租户隔离和在线备份策略。

因此它是“可靠的单用户状态底座”，不是“成熟多租户记忆服务”。下一层若要提升召回，应优先加 embedding + FTS/向量索引，而不是继续给 JSON 换壳。

## 6. 验收与回归

新增测试：

- SQLite 状态 round-trip，记忆和心境都保留；
- 重开进程式数据库，记忆仍可召回；
- JSON → SQLite 迁移，事实和记忆都保留；
- 目标已有状态时拒绝覆盖；
- `flush()` 持久化 review 后的稳定度/显著度；
- schema 版本显式记录，拒绝未来版本；
- 状态版本显式检查；
- 两个连接并发写入不丢行；
- SQLite 状态不把记忆复制进 `engine_state`。

完整回归见 `tests/test_sqlite_state.py`。当前全量测试数见 [README](../README.md)。

## 7. 还没做

- FTS5 / embedding 索引；
- 多进程写者与租约；
- 在线备份、加密、保留期；
- 真正的 schema v2 迁移样例；
- 大状态（10k+）的 p95 延迟和崩溃注入测试。

这些应该在“聊天场景参数手册”之前或并行做，但不要用参数调优掩盖存储层的结构问题。
