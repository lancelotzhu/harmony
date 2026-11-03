# Stage 2 设计稿：letter-format 微调（**已暂缓，未执行**）

状态：2026-11-03 评估后**决定暂缓**。本文存档备用，随时可启动。

## 为什么暂缓

原始动机是「漏放和线上持平（5 vs 5），必须靠阈值标定才能胜出」。
Stage 1 把测试集重建之后，这个前提消失了：

| `policy-test-v6`（280 条，130 危险） | 正确 | 漏放 | 误拦 |
| --- | --- | --- | --- |
| `policy-v5-8bit`（已部署） | 275/280 | 5 | 0 |
| 线上 `OptiQ-4bit` | 212/280 | 11 | 57 |

两项都已严格优于线上。原来的「持平」是小样本假象——旧测试集只有 75 条危险样本
（一半被训练集抢走），其中 2 条还是 `#` 注释错标。

所以本方案的收益从「把持平变成胜出」降级为「把 3.8% 漏放再压低一点」，而代价是
把误拦从 0 吐回去一部分。同时 `shadowSample` 已开到 10%，未来几周会积累**真实的**
漏放样本——用真实使用数据驱动下一轮，优先级高于现在用合成测试集上的 5 条去驱动。

**重启本方案的判据**：shadowSample 攒够一批真实漏放样本后，若模型在真实数据上的
漏放率仍明显高于可接受水平，再按下面的步骤执行。

## 核心思路

oMLX 的 `/v1/chat/completions` 不返回 logprobs，`/v1/decisions` 是这台机器上唯一的
概率出口。它对**普通 chat 模型**（Path A）的打分方式是：

```
user:      {input}\n\nQuestion: {instructions}\nOptions:\nA. SAFE\nB. RISKY\nC. DANGEROUS
           \n\nReply with the letter of the best option only.
assistant: Answer:            ← 半截轮，is_partial=True
```

只生成 1 个 token，抓 A/B/C 字母 token 的 logprob（裸写法与带前导空格两种用
`logaddexp` 合并），在字母之间归一化。

所以**不需要换模型、不需要改 oMLX、不需要新服务**，只要把训练数据渲染成这个形状、
标签改成字母，现有的 mlx-lm LoRA → fuse → convert 流程就能产出带概率的模型。

## 关键实现要点

- **assistant 目标必须写成 `"Answer: A"`**，不能是 `" A"`。`chat_template.jinja:82`
  对 content 做 `|trim`，`" A"` 会被 trim 成 `"A"`，`Answer:` 前缀消失，和推理时的
  prompt 结尾对不上。另外 `"Answer:A"`（无空格）会被切成 `:A` 合并 token，同样错位。
  数据生成时断言目标形如 `^Answer: [ABC]$`。
- **规则正文 + `<command>` 块放 `input`，防注入段落放 `instructions`。**
  防注入要出现在命令之后才有效，而 decisions 布局里 `instructions` 正好渲染在命令
  之后；规则正文放最前面才吃得到 prefix cache（每次评审逐字节相同的 ~700 token）。
- **`<command>` 包裹必须保留**：`prepare_policy.py` 的 `COMMAND_BLOCK` 正则在
  `normalise_base()` 和那条「测试集 ∩ 训练集」泄漏断言里都靠它捞命令。
- **删掉收尾句「现在只输出一个单词：」**，它和 decisions 自带的
  `Reply with the letter of the best option only.` 互相矛盾。
- **选项顺序固定 A=SAFE / B=RISKY / C=DANGEROUS，不做随机化**：生产只发这一种顺序，
  随机化等于花 LoRA 容量去学永不发生的变体，直接损害分离度。
- `--mask-prompt` 保持开启，计损落在 `Answer` `:` ` A` `<|im_end|>` `\n` 五个 token 上。

## 投训前的门禁（这是方案里最值钱的部分）

**不过门禁不许进全量训练**，把 10 小时的下注前置成 2 小时验证：

1. **格式对齐**：用 `/v1/tokenize`（`server.py:8113`，接 `continue_final_message: true`）
   拿服务端自己渲染的 token id，和本地渲染逐 token diff，要求完全相等。
2. **决定性校验**：本地用 mlx_lm 对同一串前向一次，复刻 oMLX 的 logaddexp + 三字母
   归一化，要求算出的概率和 `/v1/decisions` 返回值小数点后 4 位完全相同。
   **用现在的 v5 模型就能跑**——不需要模型好，只需要数字对上。
3. **可学性探针**：400 条数据跑 200 步 LoRA，看 valid 上 P(SAFE) 的 AUC 是否 > 0.8。
   若所有行都卡在 0.33 附近，多半是 mask 把答案 token 也 mask 掉了。

## 最大的风险：概率饱和撞上 round4 天花板

`models/decision.py:47` 把概率 `round4`。LoRA 在 one-hot 字母目标上训 2 epoch，
分布很容易塌成 P ≈ 1.0，`0.99997` 和 `1.0` 都变成 `1.0000`。一旦 DANGEROUS 侧最高
P(SAFE) 也是 `1.0000`，阈值区间就消失，方案收益归零。

预警：标定时 `P(SAFE) == 1.0000 或 0.0000` 的行占比 > 30%。
应对：选 checkpoint 的指标从「准确率」改成「分离度」（AUC + `max_danger_p` 离 1.0
的距离），早期 checkpoint 通常更不自信、更可用；或降 LR、砍到 1 epoch。

## 其他要点

- **阈值只在 `data/policy-v6/valid.jsonl` 上定**，测试集只在最后读一次。
- **生产阈值必须在已导入 oMLX 的 8-bit 件上经 HTTP 标定**，不能用本地 fused 的数
  （README 已记录 4-bit 的校准漂移前科）。顺序：训练 → 本地选 checkpoint → 8-bit
  导出 → 导入 oMLX → 再标定。
- valid 建议从 200 扩到 ~600，当前分辨率（100 条 SAFE）撑不起「6-9 条误拦」这种
  量级的决策。
- 插件降级要分三级，每级只往更保守方向掉：decisions+阈值 → chat/completions+partial
  → 交人工。**绝不能降级到旧的 `REVIEW_PROMPT`**（letter-format 模型面对旧提示词是
  另一个任务，而 `parseVerdict` 的 `\bSAFE\b` 很容易被误命中）。
- 回滚靠开关不靠备份：v6 全程不覆盖 v5 的任何产物，回滚 = 改 `useDecisions` 和
  `OMLX_MODEL` 两处。

## ⚠️ 设计稿里一条被实测推翻的说法

设计稿称 `chat_template.jinja:101` 对最后一轮 assistant **无条件**吐
`<think>\n\n</think>\n\n`，因此 `enable_thinking` 是空操作。

**实测否定了这一点。** 280 条测试集、固定 `max_tokens: 10`、只切这一个开关：

- thinking 开 → 275/280（漏放 5 误拦 0）
- thinking 关 → 273/280（漏放 5 误拦 2）

反向固定 thinking 关、只切 `max_tokens`（6 vs 10），结果完全相同。所以 oMLX 对
`enable_thinking` 的处理不止模板那一层，它确实改变了模型看到的东西。

执行本方案时，涉及 `READOUT_TEMPLATE_KWARGS={"enable_thinking": False}` 的推断要
重新验证，不要沿用「它是空操作」这个前提——`/v1/decisions` 的 readout 是写死带这个
参数的，这意味着 decisions 路径看到的 prompt 可能和我们训练时的不一致。
**这是门禁第 1 步必须先 diff 出来的东西。**
