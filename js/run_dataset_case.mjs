#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { JSDOM } from "jsdom";

import {
  applyQuestionToSvg,
  createSvgElementFromString,
  detectChartType,
  runLocalModel,
  summarizeSvg,
} from "./chart_ops.mjs";

const PROJECT_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const DOTENV_VARS = loadDotenvVars(path.join(PROJECT_ROOT, ".env"));

function loadDotenvVars(dotenvPath) {
  const out = {};
  if (!fs.existsSync(dotenvPath)) {
    return out;
  }
  const lines = fs.readFileSync(dotenvPath, "utf8").split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) {
      continue;
    }
    const idx = line.indexOf("=");
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim().replace(/^['\"]|['\"]$/g, "");
    if (key) {
      out[key] = value;
    }
  }
  return out;
}

function envOrDotenv(name, defaultValue = "") {
  const val = process.env[name];
  if (val !== undefined && val !== "") {
    return val;
  }
  return DOTENV_VARS[name] ?? defaultValue;
}

function parseArgs(argv) {
  const defaults = {
    qaIndex: 0,
    questionOnly: false,
    useQaAsUpdate: false,
    noAiAnswer: false,
    outDir: path.join(PROJECT_ROOT, "output", "dataset_runner_js"),
    aiModel: envOrDotenv("GPT_MODEL", "gpt-5.2"),
    aiBaseUrl: envOrDotenv("GPT_BASE_URL", "https://aihubmix.com/v1"),
    aiApiKey:
      envOrDotenv("Aihubmix_API_KEY_ZZT") ||
      envOrDotenv("AIHUBMIX_API_KEY") ||
      envOrDotenv("OPENAI_API_KEY") ||
      "",
    resvgBin: process.env.RESVG_BIN || "",
  };

  const args = { ...defaults };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    if (key === "question-only") {
      args.questionOnly = true;
      continue;
    }
    if (key === "use-qa-as-update") {
      args.useQaAsUpdate = true;
      continue;
    }
    if (key === "no-ai-answer") {
      args.noAiAnswer = true;
      continue;
    }

    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    i += 1;

    if (key === "task") args.task = value;
    else if (key === "case") args.case = value;
    else if (key === "qa-index") args.qaIndex = Number.parseInt(value, 10);
    else if (key === "out-dir") args.outDir = value;
    else if (key === "ai-model") args.aiModel = value;
    else if (key === "ai-base-url") args.aiBaseUrl = value;
    else if (key === "ai-api-key") args.aiApiKey = value;
    else if (key === "resvg-bin") args.resvgBin = value;
  }

  if (!args.task || !args.case) {
    throw new Error("Usage: node js/run_dataset_case.mjs --task <task> --case <id> [options]");
  }
  if (!Number.isFinite(args.qaIndex) || args.qaIndex < 0) {
    args.qaIndex = 0;
  }
  return args;
}

function walkDirs(root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        out.push(full);
        stack.push(full);
      }
    }
  }
  return out;
}

function findCaseDir(datasetRoot, task, caseId) {
  const taskDir = path.join(datasetRoot, task);
  if (!fs.existsSync(taskDir)) {
    throw new Error(`Task path not found: ${taskDir}`);
  }
  const dirs = walkDirs(taskDir).filter((p) => path.basename(p) === caseId);
  if (!dirs.length) {
    throw new Error(`Case directory not found for case '${caseId}' under ${taskDir}`);
  }
  return dirs.sort((a, b) => a.length - b.length)[0];
}

function findCaseFile(caseDir, caseId, suffix) {
  const preferred = path.join(caseDir, `${caseId}${suffix}`);
  if (fs.existsSync(preferred)) {
    return preferred;
  }
  const matches = fs
    .readdirSync(caseDir)
    .filter((name) => name.endsWith(suffix))
    .map((name) => path.join(caseDir, name))
    .sort();
  if (!matches.length) {
    throw new Error(`No '${suffix}' file found in ${caseDir}`);
  }
  return matches[0];
}

function chooseQaQuestion(payload, qaIndex) {
  const qa = payload?.QA;
  if (Array.isArray(qa) && qa.length) {
    const idx = Math.max(0, Math.min(qaIndex, qa.length - 1));
    const q = qa[idx]?.question;
    if (typeof q === "string" && q.trim()) {
      return q.trim();
    }
  }
  if (typeof payload?.question === "string" && payload.question.trim()) {
    return payload.question.trim();
  }
  throw new Error("No question found in JSON (expected QA[].question or question).");
}

function formatPoints(points) {
  const out = [];
  for (const p of points) {
    const x = Number.parseFloat(p?.x);
    const y = Number.parseFloat(p?.y);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      out.push(`(${x.toFixed(6)},${y.toFixed(6)})`);
    }
  }
  return out.join(" ");
}

function synthesizeInstruction(payload) {
  const op = String(payload?.operation || "").toLowerCase();
  const target = payload?.operation_target || {};
  const change = payload?.data_change || {};
  const chartType = String(payload?.chart_type || "").toLowerCase();
  const parts = [];

  if (chartType === "scatter") {
    const points = change?.points;
    if (Array.isArray(points) && points.length) {
      const pointsText = formatPoints(points);
      if (pointsText) {
        parts.push(`新增点 ${pointsText}`);
      }
    }
  }

  if (op.includes("delete") || op.includes("del")) {
    const names = target?.category_name || target?.del_category;
    if (Array.isArray(names) && names.length) {
      parts.push(`删除类别 ${names.map((n) => `\"${n}\"`).join(", ")}`);
    } else if (typeof names === "string" && names.trim()) {
      parts.push(`删除类别 \"${names.trim()}\"`);
    }
  }

  const addBlock = typeof change === "object" && change ? change.add : null;
  const addBlocks = Array.isArray(addBlock)
    ? addBlock.filter((item) => item && typeof item === "object")
    : addBlock && typeof addBlock === "object"
      ? [addBlock]
      : [];
  if (addBlocks.length) {
    const addName = target?.add_category;
    const addNames = Array.isArray(addName) ? addName : [addName];
    for (const [idx, block] of addBlocks.entries()) {
      const values = block?.values;
      if (!Array.isArray(values) || !values.length) {
        continue;
      }
      const valuesText = values.join(", ");
      const name = addNames[idx];
      if (typeof name === "string" && name.trim()) {
        parts.push(`新增系列 \"${name.trim()}\" : [${valuesText}]`);
      } else {
        parts.push(`新增系列: [${valuesText}]`);
      }
    }
  }

  const changeBlock = typeof change === "object" && change ? change.change : null;
  if (changeBlock && typeof changeBlock === "object") {
    const changeName = target?.change_category;
    const years = changeBlock?.years;
    const values = changeBlock?.values;
    const year = Array.isArray(years) && years.length ? years[0] : null;
    const value = Array.isArray(values) && values.length ? values[0] : null;
    if (year !== null && value !== null) {
      if (typeof changeName === "string" && changeName.trim()) {
        parts.push(`将 \"${changeName.trim()}\" 在 ${year} 年改为 ${value}`);
      } else {
        parts.push(`在 ${year} 年改为 ${value}`);
      }
    }
  }

  return parts.join("；然后 ").trim();
}

function buildQuestion(payload, qaQuestion, questionOnly, useQaAsUpdate) {
  if (questionOnly || useQaAsUpdate) {
    return { updateQuestion: qaQuestion, qaQuestion };
  }
  const instruction = synthesizeInstruction(payload);
  return { updateQuestion: instruction || qaQuestion, qaQuestion };
}

function ensureScatterUpdateQuestion(question, payload) {
  const text = String(question || "");
  const hasPointPairs = /[\(（]\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*[\)）]/.test(text);
  if (hasPointPairs) {
    return text;
  }
  const points = payload?.data_change?.points;
  if (Array.isArray(points) && points.length) {
    const pointsText = formatPoints(points);
    if (pointsText) {
      return `新增点 ${pointsText}`;
    }
  }
  return text;
}

function splitUpdateCommands(question) {
  const raw = String(question || "").trim();
  if (!raw) {
    return [];
  }
  return raw
    .split(/\s*(?:[;；\n]+|然后|并且|同时|and then|then)\s*/i)
    .map((x) => x.trim())
    .filter(Boolean);
}

function areaCommandRank(command) {
  const text = String(command || "");
  if (/(delete|remove|drop|删|删除|去掉|移除|去除|剔除)/i.test(text)) {
    return 0;
  }
  if (!(/[\[【]/.test(text)) && (/\b(19|20)\d{2}\b/.test(text) || /\d+\s*年/.test(text))) {
    return 1;
  }
  return 2;
}

function resolveResvgBin(explicit = "") {
  if (explicit) {
    const full = path.resolve(explicit);
    if (fs.existsSync(full)) {
      return full;
    }
  }
  const direct = spawnSync("which", ["resvg"], { encoding: "utf8" });
  if (direct.status === 0 && direct.stdout.trim()) {
    return direct.stdout.trim();
  }
  const candidates = [
    "/opt/anaconda3/envs/scatter/bin/resvg",
    "/opt/anaconda3/bin/resvg",
    "/opt/anaconda3/pkgs/resvg-0.46.0-h748bcf4_0/bin/resvg",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return null;
}

function renderSvgToPng(resvgBin, svgPath, pngPath) {
  if (!resvgBin) {
    return { ok: false, error: "resvg unavailable" };
  }
  fs.mkdirSync(path.dirname(pngPath), { recursive: true });
  const proc = spawnSync(resvgBin, [svgPath, pngPath], { encoding: "utf8" });
  if (proc.status !== 0) {
    return { ok: false, error: (proc.stderr || proc.stdout || "resvg failed").trim() };
  }
  return { ok: true };
}

function setupDomGlobals() {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", { contentType: "text/html" });
  const { window } = dom;
  globalThis.window = window;
  globalThis.document = window.document;
  globalThis.DOMParser = window.DOMParser;
  globalThis.XMLSerializer = window.XMLSerializer;
  globalThis.Element = window.Element;
  globalThis.SVGElement = window.SVGElement;
  globalThis.SVGGElement = window.SVGGElement;
}

function serializeSvg(svgElement) {
  return new XMLSerializer().serializeToString(svgElement);
}

function callVisionAnswer({ question, imagePath, model, baseUrl, apiKey }) {
  if (!apiKey || !String(apiKey).trim()) {
    return Promise.resolve({ ok: false, error: "missing api key" });
  }
  if (!imagePath || !fs.existsSync(imagePath)) {
    return Promise.resolve({ ok: false, error: `image not found: ${imagePath}` });
  }
  const ext = path.extname(imagePath).toLowerCase();
  let mime = "application/octet-stream";
  if (ext === ".png") mime = "image/png";
  else if (ext === ".jpg" || ext === ".jpeg") mime = "image/jpeg";
  else if (ext === ".webp") mime = "image/webp";
  else if (ext === ".svg") mime = "image/svg+xml";

  const raw = fs.readFileSync(imagePath);
  const b64 = raw.toString("base64");
  const dataUrl = `data:${mime};base64,${b64}`;

  return fetch(`${String(baseUrl || "").replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: question },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
    }),
  })
    .then(async (res) => {
      if (!res.ok) {
        const detail = await res.text();
        return { ok: false, error: `HTTP ${res.status}: ${detail.slice(0, 300)}` };
      }
      const payload = await res.json();
      const content = payload?.choices?.[0]?.message?.content;
      let answer = "";
      if (typeof content === "string") {
        answer = content.trim();
      } else if (Array.isArray(content)) {
        answer = content
          .map((x) => (typeof x?.text === "string" ? x.text.trim() : ""))
          .filter(Boolean)
          .join("\n");
      }
      if (!answer) {
        return { ok: false, error: "empty model answer", raw: payload };
      }
      return {
        ok: true,
        answer,
        model,
        base_url: baseUrl,
        image_used: imagePath,
        raw: payload,
      };
    })
    .catch((err) => ({ ok: false, error: String(err) }));
}

async function main() {
  const args = parseArgs(process.argv);
  setupDomGlobals();

  const datasetRoot = path.join(PROJECT_ROOT, "dataset");
  const caseDir = findCaseDir(datasetRoot, args.task, args.case);
  const jsonPath = findCaseFile(caseDir, args.case, ".json");
  const svgPath = findCaseFile(caseDir, args.case, ".svg");

  const payload = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const qaQuestion = chooseQaQuestion(payload, args.qaIndex);
  const built = buildQuestion(payload, qaQuestion, args.questionOnly, args.useQaAsUpdate);
  const updateQuestion = built.updateQuestion;

  const outDir = path.resolve(args.outDir, args.task.replaceAll("/", "_"), args.case);
  const outputSvg = path.join(outDir, `${args.case}_updated.svg`);
  const outputPng = path.join(outDir, `${args.case}_updated.png`);
  fs.mkdirSync(outDir, { recursive: true });

  const svgSource = fs.readFileSync(svgPath, "utf8");
  const svgElement = createSvgElementFromString(svgSource);

  const jsonType = String(payload?.chart_type || "").toLowerCase();
  const detectedType = String(detectChartType(svgElement) || "unknown").toLowerCase();
  const chartType = detectedType !== "unknown" ? detectedType : jsonType;

  const result = {
    chart_type: chartType,
    update_question_used: updateQuestion,
    qa_question: qaQuestion,
    output_svg_path: outputSvg,
    output_png_path: outputPng,
    resvg_bin: null,
    ok: false,
    png_ok: false,
  };

  const resolvedResvg = resolveResvgBin(args.resvgBin);
  result.resvg_bin = resolvedResvg;

  let opResult = null;
  try {
    if (chartType === "area") {
      let commands = splitUpdateCommands(updateQuestion);
      if (!commands.length) {
        commands = [updateQuestion];
      }
      if (commands.length > 1) {
        commands = [...commands].sort((a, b) => areaCommandRank(a) - areaCommandRank(b));
      }

      const steps = [];
      for (let i = 0; i < commands.length; i += 1) {
        const command = commands[i];
        opResult = applyQuestionToSvg(svgElement, command);
        steps.push({ step: i + 1, question: command, result: opResult });
        if (commands.length > 1 && i < commands.length - 1) {
          const stepSvg = outputSvg.replace(/\.svg$/i, `_step${i + 1}.svg`);
          fs.writeFileSync(stepSvg, serializeSvg(svgElement), "utf8");
        }
      }
      if (steps.length > 1) {
        opResult = { op: "area_combo", steps };
      }
    } else if (chartType === "line" || chartType === "bar") {
      opResult = applyQuestionToSvg(svgElement, updateQuestion);
    } else if (chartType === "scatter") {
      const scatterQuestion = ensureScatterUpdateQuestion(updateQuestion, payload);
      result.update_question_used = scatterQuestion;
      opResult = applyQuestionToSvg(svgElement, scatterQuestion);
    } else {
      throw new Error(`Unsupported chart_type '${chartType}' for JS runner.`);
    }

    fs.writeFileSync(outputSvg, serializeSvg(svgElement), "utf8");
    result.ok = true;
    result.operation_result = opResult;

    const png = renderSvgToPng(resolvedResvg, outputSvg, outputPng);
    if (png.ok) {
      result.png_ok = true;
      result.output_png_path = outputPng;
    } else {
      result.png_ok = false;
      result.note = "SVG updated successfully, but PNG was not rendered because resvg is unavailable or failed.";
      result.png_error = png.error;
    }
  } catch (error) {
    result.ok = false;
    result.error = error instanceof Error ? error.message : String(error);
    result.note = fs.existsSync(outputSvg)
      ? "SVG may be partially updated, but final output failed."
      : "Update failed before SVG output was generated.";
  }

  const summary = summarizeSvg(svgElement);
  result.summary = summary;
  result.local_answer = runLocalModel(qaQuestion, summary, result.operation_result || null);

  if (!args.noAiAnswer) {
    const imageForAi = fs.existsSync(outputPng) ? outputPng : outputSvg;
    result.ai_answer = await callVisionAnswer({
      question: qaQuestion,
      imagePath: imageForAi,
      model: args.aiModel,
      baseUrl: args.aiBaseUrl,
      apiKey: args.aiApiKey,
    });
  }

  result.task = args.task;
  result.case = args.case;
  result.case_dir = caseDir;
  result.json_path = jsonPath;
  result.svg_path = svgPath;

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

main().catch((err) => {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
});
