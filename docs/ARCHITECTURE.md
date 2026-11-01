# 实现细节

这套东西只有一条核心逻辑：**在自回归解码的过程中，往序列尾部追加 token，并让 KV cache 继续复用。**

下面按「为什么这么设计」讲，包括几个必须绕开的坑。

## 一、为什么不能用 `model.generate()`

HuggingFace 的 `generate()` 把解码循环封在内部，调用方只能通过 `LogitsProcessor`、
`StoppingCriteria` 之类的回调影响采样，**没法往 `input_ids` 里追加 token**。

能用的替代方案都不成立：

| 方案 | 为什么不行 |
|---|---|
| 回调里改 logits | 只能影响「下一个 token 选什么」，不能注入一整段新上下文 |
| 停下来拼 prompt 重新生成 | 前缀要全部重算一遍，等于放弃 KV cache，慢一个数量级 |
| 用 `StoppingCriteria` 提前停 | 那是「结束」，不是「插入」 |

所以自己写循环：

```python
for step in range(max_new_tokens):
    # ① 收插话（get_nowait，永不阻塞）
    # ② 到断句点就把插话 token 接到尾部
    input_ids = torch.cat([input_ids, piece], dim=-1)
    attention_mask = torch.cat([attention_mask, ones], dim=-1)

    # ③ 只 forward「上一轮之后新增的 token」
    outputs = model(input_ids[:, past_len:], past_key_values=past, use_cache=True)
    past = outputs.past_key_values

    nxt = sample(outputs.logits[:, -1, :])
    input_ids = torch.cat([input_ids, nxt], dim=-1)
```

`past_len` 由 `get_seq_length()` 给出。插进来的 token 对模型来说就是「新 token」，
下次 forward 顺手并进 cache。**这就是「不中断」的全部秘密。**

## 二、KV cache 长度的坑（会产出乱码）

一开始写成 `past[0][0].shape[1]` 当已缓存长度，结果输出全是乱码。

原因：`DynamicCache` 的内部布局是 `(batch, kv_heads, seq, dim)`，
`shape[1]` 拿到的是 **kv_heads 数（4）**，不是序列长度。于是每步只喂 4 个 token 的位置，
模型看到的前缀完全错位。

正确写法：

```python
def _cache_len(past):
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return past.get_seq_length()
    return past[0][0].shape[1]     # 老式 tuple-list cache 的布局
```

顺带发现：某些模型自带的 `generate()` 实现里也有同样的写法，等于 KV cache 从来没生效过，
每一步都在重算整个前缀 —— 只影响速度，不影响正确性，所以一直没人发现。

## 三、`@torch.inference_mode()` 不是可选项

漏了它的后果：每生成一个 token 都保留一次计算图，显存一路上涨。
实测 12 GB 卡上 PyTorch 分配到 **46 GiB** 后 OOM，报错是
`CUDA out of memory. Tried to allocate 20.00 MiB`。

加上之后峰值稳定在 4.3 GB（Qwen2.5-1.5B fp16 + 激活值）。

## 四、插在哪儿 —— 断句点三级策略

最早是「收到就插」，结果把句子切成两半：

```
我是Qwen，由阿里云我是一个预训练语言模型
```

模型正写「由阿里云开发的…」，插话怼进去，后半句就黏成了残句。

现在的策略分三级：

| 级别 | 触发 | 处理 |
|---|---|---|
| `sentence` | 末尾是 `。！？!?…；;` 或换行 | 立即插入 |
| `comma` | 末尾是 `，、：` 且已等 ≥ `soft_after`（默认 8）个 token | 直接插入，不补标点 |
| `forced` | 等满 `max_wait_tokens`（默认 24） | 先补一个句号，再插入 |

补句号这个兜底是必要的：如果模型正在写长句，用户等太久会以为没反应。
但补句号会产生「肥而不腻，。」这种多余标点，所以第二级（逗号）在两者之间做折中。

实现：

```python
tail = tokenizer.decode(input_ids[0, -6:], skip_special_tokens=True)
for ch in reversed(tail):
    if ch.isspace():          return 2
    if ch in TERMINALS:       return 2
    if ch in SOFT_BREAKS:     return 1
    return 0
```

## 五、注入措辞：位置 > 内容

同样一句「先说明你收到这条补充，再直接照做」：

- 写进 **system prompt**（上下文最前面）：失败。1.5B 要么复读示例，要么学会说
  「请稍等，我将为您创作」然后 EOS，活一点没干。
- 写进 **插话正文**（紧贴用户消息）：有效。极晚插入时交代率从 1/3 提到 3/3。

原因大概是位置衰减：system 与插话之间隔了几十个 token，小模型对远端指令的遵循度掉得很快。

所以 `SUPPLEMENT_STYLES` 有三档，默认 `brief`：

```python
"plain":  "（用户补充：{text}）"
"brief":  "（用户补充：{text}。先说明你收到这条补充，再直接照做。）"
"strict": "（用户补充：{text}。收到后先用一句话说明，然后立刻按新要求继续，不要停下来提问。）"
```

`strict` 反而有副作用：会诱发「请稍等片刻」这种敷衍句，然后草草收尾。

另外刻意**不用方括号**（`[用户补充]`）：小模型会把方括号结构当成模板去模仿，
输出 `[用户说]` `[用户问]` 这类同构垃圾。全角括号读起来像自然语流。

## 六、让「插在哪」看得见

模型没有「位置」概念，人有。所以插入时算一个字符偏移：

```python
chars = len(tokenizer.decode(all_new_tokens))
```

然后输出按每个 `chars` 切片：

```python
segments = [
    {"text": "秋风送爽…", "supp": None},
    {"text": "", "supp": {"id": 3, "chars": 52, "step": 51, "boundary": "sentence"}},
    {"text": "秋天，是一个多情的季节…", "supp": None},
]
```

前端拿到 `segments` 直接画「文字 → 分隔条 → 文字」，编号徽标用同一个 `id`，
就能和左侧的插话列表一一对应。

## 七、其它落地细节

- **多订阅推送**：浏览器可能开多个标签页，每个 SSE 连接一个 `queue.Queue`，广播时各自入队。
- **单会话**：`CHAT` 是全局单例，两人同时开会串台。Demo 够用，多用户要按 `session_id` 拆。
- **一次只能跑一轮**：同一张卡上串行，插话只作用于「正在进行」的那一轮。
- **插话期间不推进生成**：插入发生在两次 forward 之间，模型端感知不到任何「停顿」。
