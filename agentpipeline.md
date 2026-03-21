# Agent Flow

## 1. Overview

ChartAgent is an update-then-answer agent for chart understanding.

Its end-to-end behavior is:

1. Read one unified user request together with the chart.
2. Separate the update instruction from the QA question.
3. Recover chart structure from the current SVG.
4. Plan the chart edits that need to happen.
5. Execute those edits step by step on the latest chart state.
6. Check whether the updated rendering is usable.
7. Answer the QA question on the updated chart.
8. If the question is visually difficult, add a lightweight visual reasoning stage.

The current system is hybrid:

- the model handles understanding, planning, answer generation, and visual-tool selection
- the programmatic chart pipeline handles SVG grounding, chart editing, and validation

## 2. Runtime Role Of The Agent

The agent is not a single-pass pipeline. It is a staged loop with retries.

In practice, it does four kinds of work:

- interpret what the user wants to change
- understand the current chart structure
- apply edits in the requested order
- answer only after the chart has been updated

For some hard cases, it also performs an additional visual reasoning pass after the first answer.

## 3. Input Understanding

The agent treats the user request as one unified instruction, not as several unrelated channels.

That unified request may contain two kinds of information at the same time:

- natural-language update instructions
- the final QA question to answer after the update

In evaluation-style inputs, the request may also include a structured tail copied from metadata. When this happens, the agent uses that extra structure only as stronger grounding for the same update request.

So the input stage is not just reading text. It is deciding:

- what should be changed in the chart
- what should be answered after the change
- whether any explicit structured clues should be trusted during later execution

## 4. Request Splitting

The first major stage is splitting the unified request into two semantic parts:

- the update side
- the QA side

The update side should contain only the chart-edit intent.

The QA side should contain only the final question that should be answered after the chart has already been changed.

This stage may use model-based splitting first, then fall back to rule-based or heuristic splitting if needed.

The important outcome is not a perfect text rewrite. The important outcome is a stable internal separation between:

- what must be executed
- what must be answered later

If explicit structured information is embedded in the original request, the split stage also preserves it and carries it forward as execution grounding.

But the decomposition does not stop there.

After the system isolates the update side from the QA side, it continues decomposing the update side itself into an ordered sequence of edit steps.

This second layer of decomposition is necessary because many user requests are not single edits. They often contain:

- multiple operations in one sentence
- one operation followed by another dependent operation
- a large composite edit that should be executed as several smaller actions

So it is important to distinguish two layers:

1. request splitting is only responsible for separating the unified request into the update side and the QA side
2. the actual expansion of the update side into executable steps happens later in update planning and step construction

In other words, request splitting may recognize that the update side contains several actions, but it should not be treated as the stage that produces the final step sequence.

## 5. Chart Perception

After splitting, the agent needs to understand the chart it is about to edit.

This perception stage recovers chart structure from the SVG, including things such as:

- chart type
- axis layout
- legend/category information
- primitive layout
- how series, points, line segments, or filled regions are organized
- the correspondence among colors, labels, coordinates, and values
- the mapping needed for later edits

This is the grounding step that lets the agent connect the user’s request to real visual elements in the chart.

More concretely, this stage is not just broad visual understanding. It tries to recover the structure that later execution actually depends on, such as:

- which primitives belong to the same series
- which group of marks corresponds to a named category
- how axis ticks map to visual positions
- which objects already exist in the chart and which ones would need to be inserted or removed

So the output of chart perception is not merely a descriptive summary. It is closer to the working context that the planner and updater need in order to act reliably.

Perception is also not done only once.

It happens:

- once on the initial chart before planning
- again before each execution step on the newest SVG state

That repeated perception is important because the chart changes over time, and later steps should act on the already-updated chart rather than on the original one.

Without this grounding layer, the model may understand the language request but still fail to act reliably on questions like which line to delete, which category to modify, or where a new element should attach in the chart structure.

So in this system, chart perception is both an understanding stage and an execution prerequisite.

## 6. Update Planning

Once the chart has been perceived, the agent plans how to perform the requested edits.

This planning stage converts the update request into an ordered execution plan.

Its main responsibilities are:

- normalize the intended edit into a clearer internal form
- preserve the operation order from the user request
- break multi-operation requests into executable steps
- keep any explicit data payload attached to the correct step

If the request contains several edits, the planner should not collapse them into one vague action. It should preserve the sequence in which they need to happen.

If model planning is weak or incomplete, the system can fall back to more rule-based construction of the edit sequence.

In practice, this is the stage where the update side is operationally decomposed into steps.

So a request such as “delete one category, then add another one, then answer a question” should become a step sequence rather than remain a single block of text.

That step sequence is what later execution consumes.

If the question is simply "where are the steps actually split out?", the answer is:

- request splitting separates update from QA
- update planning interprets the update as an ordered edit plan
- step construction turns that plan into the final runnable step list for execution

In the current implementation, there is one more detail:

- the planner already produces one version of `steps`
- if `SVG_UPDATE_MODE=llm`, the code normalizes that mode to `llm_intent`
- in that mode, the system runs one extra perception-aware step rewriting pass

So from the code’s point of view, there is not just one place where steps appear:

- first in the planner
- then optionally in `llm_intent` step rewrite
- then finally in step construction, which fills gaps, merges structured payloads, and expands atomic steps

## 7. Optional Step Rewrite (called llm_intent in code)

The update pipeline can optionally insert another model stage after planning. In documentation terms this can be described as intent refinement, but in the current code it is more accurately a step rewrite stage.

This extra stage does not perform the edit itself. It runs after planning and uses:

- the normalized update request
- the perceived chart summary
- any structured grounding already extracted from the input

to generate another version of the step-level description; if successful, it replaces the planner’s original `steps`.

It is not triggered for every request automatically. It is controlled by the update mode:

- if `SVG_UPDATE_MODE=rules`, this stage is skipped
- if `SVG_UPDATE_MODE=llm`, the code normalizes it to `llm_intent`
- then the system makes an additional LLM call that rewrites `steps` using the operation text, chart perception summary, and structured context

So `llm_intent` is not a separate end-to-end update pipeline. It only affects the narrow question of how steps are written and whether they are rewritten after planning.

You can think of it this way:

- the planner already knows roughly which steps need to happen
- but in the current implementation this layer does not merely add fields
- it may directly rewrite the whole step list using perception and structured grounding

For example, the planner may produce something like "delete an unnecessary series", while this layer may rewrite that into something closer to execution, such as deleting the series whose legend label is X and whose color is Y, assuming that evidence is available from perception and prior grounding.

Another example is a composite request such as:

- change series A for both 2020 and 2021
- delete categories B and C

After this layer and the later step construction logic work together, the result will often no longer remain as two vague instructions. It will be expanded into more atomic steps such as:

1. Change "A" in 2020 to ...
2. Change "A" in 2021 to ...
3. Delete the category/series "B"
4. Delete the category/series "C"

In other words, updates that mention multiple years or multiple targets are usually expanded into multiple atomic steps rather than forcing the executor to consume one large composite command.

This is especially visible in the later step construction stage:

- one `change` that touches multiple years may be split by year
- one `delete` that names multiple categories may be split by target
- a mixed add / delete / change request is expanded while preserving the original operation order as much as possible

So even in the more model-driven path, the system still follows the same division of labor:

- the model clarifies what to edit
- the chart-specific executor performs the actual SVG change

This stage is optional rather than mandatory. If the mode is not `llm_intent`, or if this stage fails to return usable `steps`, the system continues with the planner output and then moves into step construction and execution.

## 8. Step Construction

Before execution begins, the agent finalizes the step list that will actually run.

This means combining all available evidence:

- planned edit steps
- structured update clues
- fallback interpretation from the original update text

This stage also expands composite requests into atomic actions when needed.

For example, a request that conceptually contains multiple deletions or multiple independent value changes may be converted into a sequence of smaller executable steps.

The purpose is to ensure that execution runs on a clean, ordered list of concrete actions rather than on one large ambiguous instruction.

So this stage is not separate from decomposition in spirit. It is the execution-oriented continuation of decomposition:

- first decompose the user request into update and QA
- then finalize the update plan into the runnable step list used by the executor

## 9. Stepwise Execution

Execution is iterative.

For each step, the agent:

1. turns the step into a concrete execution instruction
2. re-perceives the current chart state
3. calls the chart-type-specific updater
4. produces a new SVG and rendered image

The chart-specific updater depends on the chart type, but the overall pattern is the same:

- interpret the current step against the latest chart structure
- modify the SVG
- save the result

This means multi-step requests are executed on an evolving chart.

So the second step is based on the output of the first step, the third step is based on the output of the second, and so on.

That is the core agent loop for chart updating.

## 10. Validation And Retry

After an execution attempt finishes, the agent checks whether the updated result is trustworthy enough for QA.

This validation stage serves two goals:

- verify that the rendering is still valid
- verify that the intended update likely succeeded in a chart-consistent way

The system uses both generic render checking and some chart-specific programmatic checks.

If validation fails, the agent does not immediately give up. Instead, it can rerun the perceive-plan-execute cycle with retry guidance derived from the previous failure.

So the update phase is not simply:

- plan once
- execute once

It is more like:

- plan
- execute
- validate
- retry if necessary

This retry structure is one of the main reasons the system behaves like an agent rather than a fixed one-shot pipeline.

## 12. Answering On The Updated Chart

Once the updated chart is accepted, the agent answers the QA question using the updated chart rather than the original mixed request.

This stage is important because the answer should reflect the chart after the requested edit has already been applied.

In other words:

- the update instruction should not appear again as something to execute
- the QA stage should now behave like pure chart question answering

The answer is therefore conditioned on the final updated visual result, together with a compact summary of the update process.

## 13. Visual Tool Augmentation

Some chart questions are visually difficult even after the update is complete.

Typical examples include:

- cluster reasoning
- intersection counting
- locating extreme points

For such cases, the agent may run an extra visual tool stage after the first answer.

This stage does not re-edit the chart content. Instead, it adds light visual guides that make reasoning easier, such as:

- highlighting regions
- marking points
- drawing helper lines
- isolating relevant topology

The purpose is not presentation. The purpose is to improve answer reliability on hard visual tasks.

If this stage succeeds, the system can answer again on the visually augmented chart and use that improved answer as the final result.

## 14. What Makes This An Agent

The current system is best described as an agent because it has:

- internal decomposition of one request into multiple semantic tasks
- grounding against the current chart state
- ordered multi-step execution
- repeated perception after state changes
- validation before answering
- retries after failure
- optional tool use for hard reasoning

So it is not just “question in, answer out”.

It is a stateful update-and-reason loop over an evolving chart.

## 15. Summary

In one sentence:

ChartAgent first separates editing intent from QA intent, grounds the edit against the current chart, performs the requested chart changes step by step with repeated perception and validation, then answers on the updated chart, and adds a lightweight visual reasoning phase when the question is difficult.
