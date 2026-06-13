# CS336 Assignment 5: Alignment Writeup

> `writeup.pdf` 的中文草稿。题目顺序按照 handout 排列。

---

## Problem `prompting_baselines` (5 pts): 在 GSM8K 上运行 OLMo-2-0425-1B

### 实验设置

本题使用三种给定 prompt 模板，在 GSM8K 上评估 `allenai/OLMo-2-0425-1B`：

| Prompt | 模板文件 | Reward 函数 | Stop string |
|---|---|---|---|
| `question_only` | `cs336_alignment/prompts/question_only.prompt` | `question_only_reward_fn` | None |
| `r1_zero` | `cs336_alignment/prompts/r1_zero.prompt` | `r1_zero_reward_fn` | `</answer>` |
| `r1_zero_three_shot` | `cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt` | `r1_zero_reward_fn` | `</answer>` |

生成超参数如下：

| 参数 | 值 |
|---|---|
| Temperature | `1.0` |
| Top-p | `1.0` |
| 最大生成长度 | `512` |

GSM8K 的标准答案从数据集 `answer` 字段中提取：按 `####` 切分，并取后半部分去掉首尾空白。

### (a) 评估指标与解析审查

需要提交几句话评论、评估指标，以及若干 prompt 和 response 的例子。

对每个 prompt，统计模型生成结果落入以下三类的数量：

1. Format reward = `1` 且 correctness reward = `1`。
2. Format reward = `1` 且 correctness reward = `0`。
3. Format reward = `0` 且 correctness reward = `0`。

| Prompt | 样本总数 | Category 1: format 1, correct 1 | Category 2: format 1, correct 0 | Category 3: format 0, correct 0 | Accuracy / total reward |
|---|---:|---:|---:|---:|---:|
| `question_only` | TODO | TODO | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO | TODO | TODO |

对 Category 2 至少人工检查 10 个例子：

| Prompt | 检查样本数 | 实际正确但未被正确解析 | 备注 |
|---|---:|---:|---|
| `question_only` | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO |

对 Category 3 至少人工检查 10 个例子：

| Prompt | 检查样本数 | 实际正确但未被正确解析 | 备注 |
|---|---:|---:|---|
| `question_only` | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO |

评论：

TODO：总结主要定量结果。说明哪个 prompt 的正确率最高，哪个 prompt 最稳定地遵循预期输出格式，以及解析失败是否导致明显低估了模型的真实正确率。

#### 支持性例子

例子 1：

```text
Prompt 模板: TODO
问题: TODO
标准答案: TODO
模型回答:
TODO
Format reward: TODO
Correctness reward: TODO
人工判断: TODO
```

例子 2：

```text
Prompt 模板: TODO
问题: TODO
标准答案: TODO
模型回答:
TODO
Format reward: TODO
Correctness reward: TODO
人工判断: TODO
```

例子 3：

```text
Prompt 模板: TODO
问题: TODO
标准答案: TODO
模型回答:
TODO
Format reward: TODO
Correctness reward: TODO
人工判断: TODO
```

### (b) 不同 prompt 下的模型行为

需要提交几句话评论，并给出支持性例子。

需要讨论：

- 只给 `question_only` 时，模型是否能稳定回答问题，还是会表现出 base model 的其他行为，例如续写文本、生成额外题目、漏掉 `\boxed{}` 答案，或偏离要求的输出格式。
- zero-shot 的 `r1_zero` prompt 如何改变模型行为，尤其是它是否鼓励模型生成推理过程并使用 `<think>` / `<answer>` 标签。
- few-shot 的 `r1_zero_three_shot` 相比 zero-shot `r1_zero` 如何改变行为，尤其是示例是否改善了任务理解、输出格式和最终答案抽取。

评论：

TODO：描述每种 prompt 下的模型行为。这里应使用评估输出中的具体例子，而不仅是汇总指标。

`question_only` 的支持性例子：

```text
问题: TODO
模型回答:
TODO
观察: TODO
```

`r1_zero` 的支持性例子：

```text
问题: TODO
模型回答:
TODO
观察: TODO
```

`r1_zero_three_shot` 的支持性例子：

```text
问题: TODO
模型回答:
TODO
观察: TODO
```

---

<!-- 完成 `prompting_baselines` 后，继续填写下面的 `baseline_calcs`。 -->

## Problem `baseline_calcs` (5 pts): 计算 policy gradient estimator 的方差

令 \(A \sim \mathrm{Bernoulli}(p)\)，其中 \(p = \sigma(\theta)\)。该 policy 的 score function 为

\[
\nabla_\theta \log \pi_\theta(A)
=
\begin{cases}
1-p, & A = 1, \\
-p, & A = 0,
\end{cases}
\]

等价地，\(\nabla_\theta \log \pi_\theta(A) = A - p\)。由于 reward 为 \(r(A)=\mathbf{1}\{A=1\}=A\)，所以单个样本对应的无 baseline policy-gradient 项为

\[
Z = r(A)\nabla_\theta \log \pi_\theta(A) = A(A-p).
\]

因此，\(Z\) 以概率 \(p\) 取值 \(1-p\)，以概率 \(1-p\) 取值 \(0\)。它的期望和二阶矩为

\[
\mathbb{E}[Z] = p(1-p),
\qquad
\mathbb{E}[Z^2] = p(1-p)^2.
\]

所以

\[
\mathrm{Var}(Z)
= p(1-p)^2 - p^2(1-p)^2
= p(1-p)^3.
\]

对于 sample-mean estimator \(\frac{1}{n}\sum_{i=1}^n Z_i\)，其方差为

\[
\boxed{\mathrm{Var}\left(\frac{1}{n}\sum_{i=1}^n r(A_i)\nabla_\theta \log \pi_\theta(A_i)\right)
= \frac{p(1-p)^3}{n}}.
\]

加入常数 baseline \(b\) 后，单个样本对应的项为

\[
Z_b = (r(A)-b)\nabla_\theta \log \pi_\theta(A) = (A-b)(A-p).
\]

当 \(A=1\) 时，\(Z_b=(1-b)(1-p)\)；当 \(A=0\) 时，\(Z_b=bp\)。它的期望保持不变：

\[
\mathbb{E}[Z_b]
= p(1-b)(1-p) + (1-p)bp
= p(1-p).
\]

二阶矩为

\[
\mathbb{E}[Z_b^2]
= p(1-b)^2(1-p)^2 + (1-p)b^2p^2.
\]

因此单样本方差为

\[
\begin{aligned}
\mathrm{Var}(Z_b)
&= p(1-b)^2(1-p)^2 + (1-p)b^2p^2 - p^2(1-p)^2 \\
&= p(1-p)\left((1-p)(1-b)^2 + pb^2 - p(1-p)\right) \\
&= p(1-p)\left(b-(1-p)\right)^2.
\end{aligned}
\]

所以，baseline-adjusted sample-mean estimator 的方差为

\[
\boxed{\mathrm{Var}\left(\frac{1}{n}\sum_{i=1}^n (r(A_i)-b)\nabla_\theta \log \pi_\theta(A_i)\right)
= \frac{p(1-p)\left(b-(1-p)\right)^2}{n}}.
\]

这说明减去 baseline 会保持 estimator 的期望不变，但并不一定降低方差。在这个一维例子中，使方差最小的常数 baseline 是 \(b^\star = 1-p\)，而不是 mean reward \(p\)。

如果代入 population mean reward baseline \(b=p\)，方差变为

\[
\boxed{
\mathrm{Var}_{b=p}
= \frac{p(1-p)(2p-1)^2}{n}
}.
\]

与无 baseline 的方差 \(\frac{p(1-p)^3}{n}\) 相比，\(b=p\) 的方差更低当且仅当

\[
(2p-1)^2 < (1-p)^2
\iff
p(3p-2) < 0.
\]

对于 \(0 < p < 1\)，这意味着 population-mean baseline 在 \(p < 2/3\) 时降低方差，在 \(p=2/3\) 时方差相同，在 \(p > 2/3\) 时反而提高方差。因此，使用 \(b=p\) 并不总是更好；当 policy 已经以较高概率选择有 reward 的动作时，它可能增加方差。
