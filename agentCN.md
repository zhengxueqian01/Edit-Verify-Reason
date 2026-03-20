# Agent 总体流程

## 1. 总览

ChartAgent 接收图表输入和一段统一的用户输入，在需要时先更新图表，再基于更新后的图回答 QA 问题。

整体流程如下：

1. 接收输入文本和图表资源。
2. 将输入拆解为更新语义和 QA 语义。
3. 从 SVG 或图像上下文中感知图表结构。
4. 构造可执行的更新步骤。
5. 逐步执行图表更新。
6. 校验渲染结果。
7. 在更新后的图表上回答 QA 问题。
8. 对困难问题可选地执行视觉工具增强。

## 2. 输入

### 2.1 实际运行时输入

在实际运行场景中，agent 在概念上应接收：

- `input`：一整段统一输入字符串
- `svg_path`：用于更新和推理的图表 SVG

评测场景下可选地还会提供：

- `image_path`：原图 PNG，仅用于原图 QA 测评

核心原则是：agent 应把用户请求视为一个统一输入，而不是多个并列字段输入。

### 2.2 评测场景特例

在评测场景中，数据集可能提供一个单一输入，而输入尾部附带从 JSON 直接拼接的结构化片段，例如：

```text
After adding the category Regional Carriers and deleting the category Charter Flights, in which year does the overall maximum occur?
"operation_target": {
  "add_category": "Regional Carriers",
  "del_category": "Charter Flights"
},
"data_change": {
  "add": {
    "mode": "full_series",
    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
  },
  "del": {
    "category": "Charter Flights"
  }
}
```

它仍然被视为一个统一输入。评测场景唯一的特例是：系统可以直接从结构化尾部提取 `operation_target` 和 `data_change`，并用它们来增强 operation 的解析结果。

## 3. 问题分解

### 3.1 目标

问题分解的目标，是把一个统一输入转换成规范化的中间表示，并在内部区分：

- 更新语义
- QA 语义

问题分解后的输出，在概念上应等价于：

```json
{
  "qa_question": "...",
  "operation_text": "...",
  "operation_target": {},
  "data_change": {}
}
```

这里的 `qa_question` 指的是纯问题部分，不包含更新操作本身。它用于图表已经完成更新之后的问答阶段。

### 3.2 统一原则

实际运行输入和评测输入，最终都应该收敛到同一种内部表示。

两者唯一的区别在于结构化信息的来源不同：

- 实际运行：从自然语言中推断 `operation_target` 和 `data_change`
- 评测场景：从结构化尾部直接解析 `operation_target` 和 `data_change`，再与自然语言中的 operation 合并

### 3.3 评测场景的分解规则

对于评测输入，问题分解逻辑应执行以下步骤：

1. 把整段内容视为一个统一的 `input`
2. 如果存在 `operation_target` 和 `data_change`，则将自然语言部分与结构化尾部分离
3. 从自然语言部分中提取 QA 问题
4. 从自然语言部分中提取或归一化 operation 文本
5. 直接从结构化尾部解析 `operation_target` 和 `data_change`
6. 将结构化字段并入 operation 语义，使最终的 operation 不只是文本，而是“文本 + 明确的目标 + 明确的数据”

这意味着评测场景并没有引入第二条独立输入通道，它只是为 operation grounding 提供了更显式的来源。

### 3.4 示例 A：评测场景输入

原始输入：

```text
After adding the category Regional Carriers and deleting the category Charter Flights, in which year does the overall maximum occur?
"operation_target": {
  "add_category": "Regional Carriers",
  "del_category": "Charter Flights"
},
"data_change": {
  "add": {
    "mode": "full_series",
    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
  },
  "del": {
    "category": "Charter Flights"
  }
}
```

规范化后的内部表示：

```json
{
  "qa_question": "In which year does the overall maximum occur?",
  "operation_text": "After adding the category Regional Carriers and deleting the category Charter Flights",
  "operation_target": {
    "add_category": "Regional Carriers",
    "del_category": "Charter Flights"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
      "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
    },
    "del": {
      "category": "Charter Flights"
    }
  }
}
```

### 3.5 示例 B：真实运行场景输入

原始输入：

```text
After adding a new category "Regional Carriers" with values across the years 2015-2024 (128,598; 186,977; 205,514; 136,129; 226,783; 246,727; 170,089; 154,587; 195,958; 176,685) and deleting the category "Charter Flights," in which year does the overall maximum occur?
```

规范化后的内部表示：

```json
{
  "qa_question": "In which year does the overall maximum occur?",
  "operation_text": "After adding a new category Regional Carriers with values across the years 2015-2024 and deleting the category Charter Flights",
  "operation_target": {
    "add_category": "Regional Carriers",
    "del_category": "Charter Flights"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
      "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
    }
  }
}
```

### 3.6 问题分解的输出

问题分解阶段最终应产出一种可以直接转为执行步骤的 operation 表示，例如：

```json
[
  {
    "operation": "add",
    "target": "Regional Carriers",
    "data_change": {
      "mode": "full_series",
      "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
      "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
    }
  },
  {
    "operation": "delete",
    "target": "Charter Flights"
  }
]
```

换句话说：

- 外部输入形式可以不同
- 内部规范化表示应保持一致
- 执行阶段应基于规范化表示，而不是直接基于原始文本执行

## 4. 感知

问题分解之后，agent 会从 SVG 中感知图表结构，恢复图表类型和映射信息，包括：

- 图表类型
- 坐标轴 ticks
- 图例项
- 可操作图元
- 坐标映射关系

这一阶段为后续正确执行图表更新提供 grounding。

## 5. 规划与执行

规划阶段会把规范化后的 operation 语义转成可执行的更新步骤。

典型执行规则如下：

1. 决定 operation 的执行顺序
2. 对每个步骤，如有必要先对最新 SVG 重新感知
3. 调用对应图表类型的 updater 执行更新
4. 保存中间或最终的 SVG 与渲染后的 PNG

对于 add + delete 这类多操作输入，agent 应在不断演化的 SVG 上按顺序逐步执行。

因此，按图表类型划分的 updater 也可以被视为一层“工具式执行器”，例如：

- `line` updater
- `area` updater
- `scatter` updater

在这种抽象下：

- 模型决定应该调用哪个工具
- 模型提供目标对象和参数
- 底层规则执行器负责真正的 SVG 删除 / 新增 / 修改

换句话说，当前系统可以理解为：

- 模型负责指定“改谁、怎么改”
- 规则负责把修改真正落到 SVG 上

## 6. 校验

执行完成后，agent 会对更新后的渲染结果做校验。

校验可以包括：

- 基础渲染检查
- 图表类型相关的程序化检查
- 当渲染结果无效时进行重试

目的是确保后续 QA 使用的是可信的更新后图表。

## 7. 回答

一旦更新后的图表可用，agent 就应基于更新后的图表回答 QA 问题，而不是基于原图回答。

在评测场景中，系统也可以对原图回答同一个问题，作为 baseline 指标。

三类问答输入应明确区分如下：

1. 原图问答使用完整输入 `full_input`，因为此时图表尚未更新。
2. 修改后图问答使用 `qa_question`，因为更新已经执行完成。
3. 增强后图问答也使用 `qa_question`，因为视觉增强是在图表更新之后进行的。

简化表示如下：

- 原图问答：`full_input` + `original_image_path`
- 修改后图问答：`qa_question` + `updated_image_path`
- 增强后图问答：`qa_question` + `augmented_image_path`

## 8. 视觉工具增强

对于交点计数、聚类判断等较难的 QA 任务，agent 可以执行视觉工具增强阶段。

这一阶段会在已经更新后的图表上添加轻量级视觉辅助标记，以提升回答可靠性，但不会改变图表数据本身。
