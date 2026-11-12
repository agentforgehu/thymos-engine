# 固定小上下文预算 A/B 测试

- 模型：`qwen3.6-35b-a3b`
- 轮数：50
- 记忆块预算：600 字符
- 检索 Top-k：5

| 模式 | 完成轮数 | 探针 | 上下文召回 | 回答召回 | 模型错误 | 平均记忆块字符 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| thymos | 50 | 16 | 1.0 | 1.0 | 0 | 417.5 | 5.6s |
| recent | 50 | 16 | 0.375 | 0.438 | 0 | 490.5 | 5.25s |

## 逐探针对照

| 轮 | 探针 | thymos 上下文 | thymos 回答 | recent 上下文 | recent 回答 |
|---:|---|---:|---:|---:|---:|
| 5 | 你还记得我叫什么、住在哪吗？ | True | True | True | True |
| 10 | 我的猫叫什么？是什么品种？ | True | True | True | True |
| 13 | 你还记得我的项目叫什么、什么时候交付吗？ | True | True | True | True |
| 16 | 我是不是对什么食物过敏？ | True | True | False | False |
| 21 | 我晚上一般几点睡？ | True | True | False | False |
| 22 | 我早上一般几点起？ | True | True | False | False |
| 23 | 我喝咖啡加糖吗？ | True | True | False | False |
| 24 | 我是不是独立开发者？ | True | True | False | True |
| 26 | 我现在住在哪个城市？ | True | True | False | False |
| 30 | 我现在的收货地址是哪里？ | True | True | True | True |
| 33 | 你还记得我养了什么猫吗？ | True | True | False | False |
| 35 | 我现在一般几点睡？ | True | True | True | True |
| 45 | 你还记得我对什么过敏吗？ | True | True | False | False |
| 46 | 我搬到上海后，地址是什么？ | True | True | False | False |
| 47 | 我最近是不是睡得越来越晚？ | True | True | True | True |
| 50 | 我叫什么？住在哪？ | True | True | False | False |

## 说明

- `thymos` 的记忆块由结构化事实 + 预算内情景记忆组成；
- `recent` 只放最近的用户/助手对话，截断到同一字符预算；
- 上下文命中不等于回答命中；两者分开统计。
