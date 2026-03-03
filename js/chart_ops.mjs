const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";

const NUMBER_PATTERN = /-?\d+(?:\.\d+)?/g;
const POINT_PATTERN = /[\(（]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[\)）]/g;
const BRACKET_PAIR_PATTERN = /[\[【]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[\]】]/g;

function getGroupById(root, id) {
  const groups = root.getElementsByTagNameNS(SVG_NS, "g");
  for (const group of groups) {
    if (group.getAttribute("id") === id) {
      return group;
    }
  }
  return null;
}

function getGroupsByPrefix(root, prefix, directParent = null) {
  const source = directParent || root;
  const groups = source.getElementsByTagNameNS(SVG_NS, "g");
  const out = [];
  for (const group of groups) {
    const id = group.getAttribute("id") || "";
    if (!id.startsWith(prefix)) {
      continue;
    }
    if (directParent && group.parentNode !== directParent) {
      continue;
    }
    out.push(group);
  }
  return out;
}

function extractCommentText(element) {
  const serialized = new XMLSerializer().serializeToString(element);
  const match = serialized.match(/<!--\s*([^<]+?)\s*-->/);
  return match ? match[1].trim() : null;
}

function extractNumericComment(element) {
  const text = extractCommentText(element);
  if (text) {
    const value = Number.parseFloat(text);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  const visible = Number.parseFloat(String(element?.textContent || "").trim());
  return Number.isFinite(visible) ? visible : Number.NaN;
}

function extractPathStroke(path) {
  const style = path?.getAttribute("style") || "";
  const match = style.match(/stroke:\s*(#[0-9a-fA-F]{6})/);
  return match ? match[1] : null;
}

function extractPathFill(path) {
  const fill = path?.getAttribute("fill");
  if (fill && /^#[0-9a-fA-F]{6}$/.test(fill)) {
    return fill;
  }
  const style = path?.getAttribute("style") || "";
  const match = style.match(/fill:\s*(#[0-9a-fA-F]{6})/);
  return match ? match[1] : null;
}

function parsePathPoints(dAttr) {
  const points = [];
  const coords = String(dAttr || "").matchAll(/[ML]\s+(-?[\d.]+)\s+(-?[\d.]+)/g);
  for (const match of coords) {
    const x = Number.parseFloat(match[1]);
    const y = Number.parseFloat(match[2]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push([x, y]);
    }
  }
  return points;
}

function formatPath(points, closePath = false) {
  if (!points.length) {
    return "";
  }
  const parts = [`M ${points[0][0].toFixed(6)} ${points[0][1].toFixed(6)}`];
  for (const [x, y] of points.slice(1)) {
    parts.push(`L ${x.toFixed(6)} ${y.toFixed(6)}`);
  }
  if (closePath) {
    parts.push("Z");
  }
  return parts.join(" ");
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function parsePointsFromQuestion(question) {
  const points = [];
  const text = String(question || "");
  for (const match of text.matchAll(POINT_PATTERN)) {
    const x = Number.parseFloat(match[1]);
    const y = Number.parseFloat(match[2]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push({ x, y });
    }
  }
  for (const match of text.matchAll(BRACKET_PAIR_PATTERN)) {
    const x = Number.parseFloat(match[1]);
    const y = Number.parseFloat(match[2]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push({ x, y });
    }
  }
  return points;
}

function parseValuesFromQuestion(question) {
  const text = String(question || "");
  const bracket = text.match(/[\[【]([\s\S]*?)[\]】]/);
  if (bracket) {
    const nums = (bracket[1].match(NUMBER_PATTERN) || []).map((v) => Number.parseFloat(v));
    return nums.filter(Number.isFinite);
  }
  const nums = (text.match(NUMBER_PATTERN) || []).map((v) => Number.parseFloat(v));
  return nums.filter(Number.isFinite);
}

function hasDeleteIntent(question) {
  return /(delete|remove|drop|删|删除|去掉|移除|去除|剔除)/i.test(String(question || ""));
}

function hasYearUpdate(question) {
  const text = String(question || "");
  if (text.includes("[") || text.includes("【")) {
    return false;
  }
  return /\b(19|20)\d{2}\b/.test(text) || /\d+\s*年/.test(text);
}

function extractYear(question) {
  const text = String(question || "");
  const m1 = text.match(/(19|20)\d{2}/);
  if (m1) {
    return Number.parseFloat(m1[0]);
  }
  const m2 = text.match(/(-?\d+(?:\.\d+)?)\s*年/);
  return m2 ? Number.parseFloat(m2[1]) : null;
}

function extractUpdateValue(question) {
  const text = String(question || "");
  const absPatterns = [/(?:=|to|为|变为|改为)\s*(-?\d+(?:\.\d+)?)/i];
  for (const p of absPatterns) {
    const m = text.match(p);
    if (m) {
      return { value: Number.parseFloat(m[1]), mode: "absolute" };
    }
  }
  const relPatterns = [
    /(?:increase|add|increase by|add by|增加|提升|上升|减少|下降|降低)\s*(?:by\s*)?(-?\d+(?:\.\d+)?)/i,
    /([+-]\d+(?:\.\d+)?)/,
  ];
  for (const p of relPatterns) {
    const m = text.match(p);
    if (m) {
      let value = Number.parseFloat(m[1]);
      if (/(减少|下降|降低|decrease|reduce)/i.test(text)) {
        value = -Math.abs(value);
      }
      return { value, mode: "relative" };
    }
  }
  const m = text.match(/(-?\d+(?:\.\d+)?)/);
  if (m) {
    const relative = /(增加|提升|上升|减少|下降|降低|increase|add|decrease|reduce)/i.test(text);
    return { value: Number.parseFloat(m[1]), mode: relative ? "relative" : "absolute" };
  }
  return { value: null, mode: "absolute" };
}

function parseAxisTicks(svgElement, axisId, tickPrefix, isX) {
  const axis = getGroupById(svgElement, axisId);
  if (!axis) {
    return [];
  }
  const ticks = [];
  const tickGroups = getGroupsByPrefix(axis, tickPrefix);
  for (const group of tickGroups) {
    let pixel = Number.NaN;
    const use = group.getElementsByTagNameNS(SVG_NS, "use")[0];
    if (use) {
      const raw = use.getAttribute(isX ? "x" : "y");
      pixel = Number.parseFloat(raw || "");
    }
    if (!Number.isFinite(pixel)) {
      const path = group.getElementsByTagNameNS(SVG_NS, "path")[0];
      const d = path?.getAttribute("d") || "";
      const match = isX ? d.match(/M\s+(-?[\d.]+)/) : d.match(/M\s+-?[\d.]+\s+(-?[\d.]+)/);
      pixel = Number.parseFloat(match?.[1] || "");
    }
    const textGroup = getGroupsByPrefix(group, "text_")[0] || group;
    const value = extractNumericComment(textGroup);
    if (Number.isFinite(pixel) && Number.isFinite(value)) {
      ticks.push([pixel, value]);
    }
  }
  ticks.sort((a, b) => a[1] - b[1]);
  return ticks;
}

function dataToPixel(value, ticks) {
  const sorted = [...ticks].sort((a, b) => a[1] - b[1]);
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const [p1, d1] = sorted[i];
    const [p2, d2] = sorted[i + 1];
    if ((d1 <= value && value <= d2) || (d2 <= value && value <= d1)) {
      if (d2 === d1) {
        return p1;
      }
      return p1 + ((value - d1) / (d2 - d1)) * (p2 - p1);
    }
  }
  const [p1, d1] = sorted[sorted.length - 2];
  const [p2, d2] = sorted[sorted.length - 1];
  if (d2 === d1) {
    return p2;
  }
  return p1 + ((value - d1) / (d2 - d1)) * (p2 - p1);
}

function pixelToData(pixel, ticks) {
  const sorted = [...ticks].sort((a, b) => a[0] - b[0]);
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const [p1, d1] = sorted[i];
    const [p2, d2] = sorted[i + 1];
    if (Math.min(p1, p2) <= pixel && pixel <= Math.max(p1, p2)) {
      if (p2 === p1) {
        return d1;
      }
      return d1 + ((pixel - p1) / (p2 - p1)) * (d2 - d1);
    }
  }
  const [p1, d1] = sorted[sorted.length - 2];
  const [p2, d2] = sorted[sorted.length - 1];
  if (p2 === p1) {
    return d2;
  }
  return d1 + ((pixel - p1) / (p2 - p1)) * (d2 - d1);
}

function ensureUpdateGroup(axes, id) {
  const existing = getGroupById(axes, id);
  if (existing) {
    return existing;
  }
  const node = document.createElementNS(SVG_NS, "g");
  node.setAttribute("id", id);
  axes.appendChild(node);
  return node;
}

function extractLegendItems(svgElement, mode) {
  const legend = getGroupById(svgElement, "legend_1");
  if (!legend) {
    return { legend: null, items: [] };
  }
  const items = [];
  let pendingColor = null;
  let pendingPatch = null;
  const children = Array.from(legend.children);
  for (const child of children) {
    const path = child.getElementsByTagNameNS(SVG_NS, "path")[0] || null;
    const color = mode === "line" ? extractPathStroke(path) : extractPathFill(path);
    if (color) {
      pendingColor = color;
      pendingPatch = child;
      continue;
    }
    const id = child.getAttribute("id") || "";
    if (id.startsWith("text_")) {
      const label = extractCommentText(child) || child.textContent?.trim() || "";
      if (label) {
        items.push({
          label,
          color: pendingColor,
          text: child,
          patch: pendingPatch,
        });
      }
      pendingColor = null;
      pendingPatch = null;
    }
  }
  return { legend, items };
}

function appendLegendText(legend, label) {
  const text = document.createElementNS(SVG_NS, "text");
  text.setAttribute("x", "80");
  text.setAttribute("y", `${80 + legend.children.length * 14}`);
  text.setAttribute("font-size", "10");
  text.setAttribute("fill", "#000000");
  text.textContent = label;
  legend.appendChild(text);
}

function removeLegendItem(legend, items, label) {
  const found = items.find((item) => item.label === label);
  if (!found) {
    return;
  }
  if (found.patch && found.patch.parentNode === legend) {
    legend.removeChild(found.patch);
  }
  if (found.text && found.text.parentNode === legend) {
    legend.removeChild(found.text);
  }
}

function matchLabel(question, labels) {
  const lower = String(question || "").toLowerCase();
  for (const label of labels) {
    if (lower.includes(label.toLowerCase())) {
      return label;
    }
  }
  return null;
}

function matchLabels(question, labels) {
  const lower = String(question || "").toLowerCase();
  const matched = labels.filter((label) => lower.includes(label.toLowerCase()));
  if (matched.length) {
    return matched;
  }
  const quoted = [...String(question || "").matchAll(/[\"“”']([^\"“”']+)[\"“”']/g)].map((m) =>
    m[1].trim().toLowerCase()
  );
  return labels.filter((label) => quoted.includes(label.toLowerCase()));
}

function detectLineGroups(axes) {
  const lines = [];
  for (const child of Array.from(axes.children)) {
    if (!(child instanceof SVGGElement)) {
      continue;
    }
    const id = child.getAttribute("id") || "";
    if (!id.startsWith("line2d_")) {
      continue;
    }
    const path = child.querySelector("path");
    const style = path?.getAttribute("style") || "";
    if (path && /stroke:/.test(style) && /fill:\s*none/.test(style)) {
      lines.push({ group: child, path });
    }
  }
  return lines;
}

function computeXPositions(values, xTicks) {
  if (values.length === xTicks.length) {
    return xTicks.map((t) => t[0]);
  }
  const xMin = Math.min(...xTicks.map((t) => t[0]));
  const xMax = Math.max(...xTicks.map((t) => t[0]));
  if (values.length === 1) {
    return [(xMin + xMax) / 2];
  }
  const step = (xMax - xMin) / (values.length - 1);
  return values.map((_, i) => xMin + i * step);
}

function extractSeriesLabel(question, mode) {
  const quoted = String(question || "").match(/[\"“”']([^\"“”']+)[\"“”']/);
  if (quoted) {
    return quoted[1].trim();
  }
  const text = String(question || "");
  const split = text.includes("：") ? text.split("：", 1)[0] : text.includes(":") ? text.split(":", 1)[0] : "";
  if (!split) {
    return null;
  }
  const pattern =
    mode === "line"
      ? /^(新增|添加|add|new)\s*(一个|一条)?\s*(类别|折线|series|line)?\s*/i
      : /^(新增|添加|add|new)\s*(一个|一条)?\s*(类别|系列|area|series)?\s*/i;
  const cleaned = split.replace(pattern, "").trim().replace(/[-:\s]+$/g, "");
  if (!cleaned) {
    return null;
  }
  return cleaned;
}

function getAxes(svgElement) {
  const axes = getGroupById(svgElement, "axes_1");
  if (!axes) {
    throw new Error("SVG axes group not found.");
  }
  return axes;
}

function hasClusterIntent(question) {
  return /(cluster|dbscan|聚类|簇)/i.test(String(question || ""));
}

function parseClusterParams(question) {
  const text = String(question || "");
  const epsMatch = text.match(/eps\s*=\s*(-?\d+(?:\.\d+)?)/i);
  const minMatch = text.match(/min[_\s-]*samples\s*=\s*(\d+)/i);
  const eps = epsMatch ? Number.parseFloat(epsMatch[1]) : 5.0;
  const minSamples = minMatch ? Number.parseInt(minMatch[1], 10) : 5;
  return {
    eps: Number.isFinite(eps) && eps > 0 ? eps : 5.0,
    minSamples: Number.isFinite(minSamples) && minSamples > 0 ? minSamples : 5,
  };
}

function extractScatterPixelPoints(svgElement) {
  const axes = getAxes(svgElement);
  const out = [];
  const pushPointsFrom = (group) => {
    if (!group) {
      return;
    }
    for (const use of group.querySelectorAll("use")) {
      const x = Number.parseFloat(use.getAttribute("x") || "");
      const y = Number.parseFloat(use.getAttribute("y") || "");
      if (Number.isFinite(x) && Number.isFinite(y)) {
        out.push([x, y]);
      }
    }
    for (const circle of group.querySelectorAll("circle")) {
      const x = Number.parseFloat(circle.getAttribute("cx") || "");
      const y = Number.parseFloat(circle.getAttribute("cy") || "");
      if (Number.isFinite(x) && Number.isFinite(y)) {
        out.push([x, y]);
      }
    }
  };
  pushPointsFrom(getGroupById(axes, "PathCollection_1"));
  pushPointsFrom(getGroupById(axes, "PathCollection_update"));
  return out;
}

function dbscan(points, eps, minSamples) {
  const labels = Array.from({ length: points.length }, () => -1);
  const visited = Array.from({ length: points.length }, () => false);
  let clusterId = 0;

  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const neighborsOf = (idx) => {
    const out = [];
    for (let i = 0; i < points.length; i += 1) {
      if (distance(points[idx], points[i]) <= eps) {
        out.push(i);
      }
    }
    return out;
  };

  for (let i = 0; i < points.length; i += 1) {
    if (visited[i]) {
      continue;
    }
    visited[i] = true;
    const neighbors = neighborsOf(i);
    if (neighbors.length < minSamples) {
      labels[i] = -1;
      continue;
    }
    labels[i] = clusterId;
    for (let k = 0; k < neighbors.length; k += 1) {
      const n = neighbors[k];
      if (!visited[n]) {
        visited[n] = true;
        const nNeighbors = neighborsOf(n);
        if (nNeighbors.length >= minSamples) {
          for (const candidate of nNeighbors) {
            if (!neighbors.includes(candidate)) {
              neighbors.push(candidate);
            }
          }
        }
      }
      if (labels[n] === -1) {
        labels[n] = clusterId;
      }
    }
    clusterId += 1;
  }
  return labels;
}

function updateScatter(svgElement, question) {
  const xTicks = parseAxisTicks(svgElement, "matplotlib.axis_1", "xtick_", true);
  const yTicks = parseAxisTicks(svgElement, "matplotlib.axis_2", "ytick_", false);
  if (xTicks.length < 2 || yTicks.length < 2) {
    throw new Error("Insufficient ticks to map data points.");
  }

  if (hasClusterIntent(question)) {
    const pixelPoints = extractScatterPixelPoints(svgElement);
    if (!pixelPoints.length) {
      throw new Error("No scatter points found for clustering.");
    }
    const dataPoints = pixelPoints.map(([x, y]) => [pixelToData(x, xTicks), pixelToData(y, yTicks)]);
    const { eps, minSamples } = parseClusterParams(question);
    const labels = dbscan(dataPoints, eps, minSamples);
    const clusters = new Set(labels.filter((x) => x !== -1)).size;
    const noise = labels.filter((x) => x === -1).length;
    return { op: "scatter_cluster", points: dataPoints.length, clusters, noise, eps, min_samples: minSamples };
  }

  const points = parsePointsFromQuestion(question);
  if (!points.length) {
    throw new Error("No new points parsed from question.");
  }
  const axes = getAxes(svgElement);
  const baseCollection = getGroupById(axes, "PathCollection_1");
  let href = null;
  let clipPath = null;
  if (baseCollection) {
    const firstUse = baseCollection.querySelector("g use, use");
    href =
      firstUse?.getAttribute("href") ||
      firstUse?.getAttribute("xlink:href") ||
      firstUse?.getAttributeNS(XLINK_NS, "href") ||
      null;
    clipPath = firstUse?.parentElement?.getAttribute("clip-path") || null;
  }

  const group = ensureUpdateGroup(axes, "PathCollection_update");
  clearChildren(group);

  for (const point of points) {
    const x = dataToPixel(point.x, xTicks);
    const y = dataToPixel(point.y, yTicks);
    if (href) {
      const g = document.createElementNS(SVG_NS, "g");
      if (clipPath) {
        g.setAttribute("clip-path", clipPath);
      }
      const use = document.createElementNS(SVG_NS, "use");
      use.setAttribute("href", href);
      use.setAttributeNS(XLINK_NS, "xlink:href", href);
      use.setAttribute("x", x.toFixed(6));
      use.setAttribute("y", y.toFixed(6));
      use.setAttribute("style", "fill: #ff1493; fill-opacity: 0.85");
      g.appendChild(use);
      group.appendChild(g);
    } else {
      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", x.toFixed(6));
      circle.setAttribute("cy", y.toFixed(6));
      circle.setAttribute("r", "3.5");
      circle.setAttribute("style", "fill: #ff1493; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85");
      group.appendChild(circle);
    }
  }
  return { op: "scatter_add", count: points.length };
}

function updateLine(svgElement, question) {
  const axes = getAxes(svgElement);
  const xTicks = parseAxisTicks(svgElement, "matplotlib.axis_1", "xtick_", true);
  const yTicks = parseAxisTicks(svgElement, "matplotlib.axis_2", "ytick_", false);
  if (xTicks.length < 2 || yTicks.length < 2) {
    throw new Error("Insufficient axis ticks for line mapping.");
  }
  const { legend, items } = extractLegendItems(svgElement, "line");
  const labels = items.map((x) => x.label);

  if (hasDeleteIntent(question)) {
    const targets = matchLabels(question, labels);
    if (!targets.length) {
      throw new Error("No matching line series found in question.");
    }
    const lineGroups = detectLineGroups(axes);
    for (const label of targets) {
      const color = items.find((x) => x.label === label)?.color;
      if (!color) {
        continue;
      }
      const target = lineGroups.find((item) => (extractPathStroke(item.path) || "").toLowerCase() === color.toLowerCase());
      if (target && target.group.parentNode === axes) {
        axes.removeChild(target.group);
      }
      if (legend) {
        removeLegendItem(legend, items, label);
      }
    }
    rescaleLineAfterRemoval(svgElement, axes, yTicks);
    return { op: "line_remove", labels: targets };
  }

  if (hasYearUpdate(question)) {
    const label = matchLabel(question, labels);
    const year = extractYear(question);
    const { value, mode } = extractUpdateValue(question);
    if (!label || year === null || value === null) {
      throw new Error("No valid line update request found in question.");
    }
    const color = items.find((x) => x.label === label)?.color;
    if (!color) {
      throw new Error("No matching legend color for selected series.");
    }
    const targetGroup = detectLineGroups(axes).find(
      (item) => (extractPathStroke(item.path) || "").toLowerCase() === color.toLowerCase()
    );
    if (!targetGroup) {
      throw new Error("No line series matches selected legend color.");
    }
    const points = parsePathPoints(targetGroup.path.getAttribute("d") || "");
    if (!points.length) {
      throw new Error("Line path contains no points.");
    }
    const targetX = dataToPixel(year, xTicks);
    let idx = 0;
    let dist = Number.POSITIVE_INFINITY;
    points.forEach((p, i) => {
      const d = Math.abs(p[0] - targetX);
      if (d < dist) {
        dist = d;
        idx = i;
      }
    });
    const currentData = pixelToData(points[idx][1], yTicks);
    const newData = mode === "relative" ? currentData + value : value;
    points[idx][1] = dataToPixel(newData, yTicks);
    targetGroup.path.setAttribute("d", formatPath(points, false));
    return { op: "line_year_update", label, year, value: newData };
  }

  const values = parseValuesFromQuestion(question);
  if (!values.length) {
    throw new Error("No new series values found in question.");
  }
  const xPos = computeXPositions(values, xTicks);
  const points = values.map((v, i) => [xPos[i], dataToPixel(v, yTicks)]);

  const lineGroups = detectLineGroups(axes);
  const baseStroke = extractPathStroke(lineGroups[0]?.path) || "#dc143c";
  const baseWidth = (() => {
    const style = lineGroups[0]?.path?.getAttribute("style") || "";
    const m = style.match(/stroke-width:\s*([\d.]+)/);
    return m ? Number.parseFloat(m[1]) : 2;
  })();

  const lineGroup = ensureUpdateGroup(axes, "line2d_update");
  let path = lineGroup.querySelector("path");
  if (!path) {
    path = document.createElementNS(SVG_NS, "path");
    lineGroup.appendChild(path);
  }
  path.setAttribute("style", `fill: none; stroke: ${baseStroke}; stroke-width: ${baseWidth}`);
  path.setAttribute("d", formatPath(points, false));

  const markerGroup = ensureUpdateGroup(axes, "line2d_update_markers");
  clearChildren(markerGroup);
  for (const [x, y] of points) {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", x.toFixed(6));
    circle.setAttribute("cy", y.toFixed(6));
    circle.setAttribute("r", "3");
    circle.setAttribute("style", `fill: ${baseStroke}; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85`);
    markerGroup.appendChild(circle);
  }

  const label = extractSeriesLabel(question, "line");
  if (label && legend) {
    appendLegendText(legend, label);
  }
  return { op: "line_add", count: values.length, label: label || null };
}

function extractTickPosition(tickGroup, isX) {
  const use = tickGroup.querySelector("use");
  if (use) {
    const raw = use.getAttribute(isX ? "x" : "y");
    const num = Number.parseFloat(raw || "");
    if (Number.isFinite(num)) {
      return num;
    }
  }
  const path = tickGroup.querySelector("path");
  const d = path?.getAttribute("d") || "";
  const match = isX ? d.match(/M\s+(-?[\d.]+)/) : d.match(/M\s+-?[\d.]+\s+(-?[\d.]+)/);
  const value = Number.parseFloat(match?.[1] || "");
  return Number.isFinite(value) ? value : null;
}

function pickYMin(yTicks, maxValue) {
  const hasZero = yTicks.some((tick) => Math.abs(tick[1]) < 1e-9);
  if (hasZero && maxValue >= 0) {
    return 0;
  }
  return Math.min(...yTicks.map((t) => t[1]));
}

function niceNumber(value, roundUp) {
  if (value === 0) {
    return 0;
  }
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const fraction = Math.abs(value) / 10 ** exponent;
  let niceFraction;
  if (roundUp) {
    if (fraction <= 1) {
      niceFraction = 1;
    } else if (fraction <= 2) {
      niceFraction = 2;
    } else if (fraction <= 5) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  } else if (fraction < 1.5) {
    niceFraction = 1;
  } else if (fraction < 3) {
    niceFraction = 2;
  } else if (fraction < 7) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

function niceUpperBound(minValue, maxValue, tickCount) {
  if (tickCount < 2) {
    return maxValue;
  }
  const span = maxValue - minValue;
  if (span <= 0) {
    return maxValue;
  }
  const step = niceNumber(span / (tickCount - 1), true);
  return minValue + step * (tickCount - 1);
}

function buildTicks(minValue, maxValue, tickCount) {
  if (tickCount < 2) {
    return [minValue, maxValue];
  }
  const step = (maxValue - minValue) / (tickCount - 1);
  return Array.from({ length: tickCount }, (_, i) => minValue + step * i);
}

function mapDataToPixel(value, dataMin, dataMax, pixelMin, pixelMax) {
  if (dataMax === dataMin) {
    return pixelMin;
  }
  const ratio = (value - dataMin) / (dataMax - dataMin);
  return pixelMin + ratio * (pixelMax - pixelMin);
}

function formatTickLabel(value) {
  if (Math.abs(value - Math.round(value)) < 1e-6) {
    return `${Math.round(value)}`;
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function fallbackTickTextX(tickGroup) {
  const use = tickGroup.querySelector("use");
  const x = Number.parseFloat(use?.getAttribute("x") || "");
  return Number.isFinite(x) ? x - 12 : 0;
}

function extractTickTextAnchor(tickGroup) {
  const textGroup = getGroupsByPrefix(tickGroup, "text_")[0];
  if (!textGroup) {
    return { x: null, offset: null };
  }
  const nested = textGroup.querySelector("g[transform]");
  const transform = nested?.getAttribute("transform") || "";
  const match = transform.match(/translate\(([-\d.]+)\s+([-\d.]+)\)/);
  if (!match) {
    return { x: null, offset: null };
  }
  const textX = Number.parseFloat(match[1]);
  const textY = Number.parseFloat(match[2]);
  const oldY = extractTickPosition(tickGroup, false);
  if (!Number.isFinite(textX) || !Number.isFinite(textY)) {
    return { x: null, offset: null };
  }
  return {
    x: textX,
    offset: Number.isFinite(oldY) ? textY - oldY : null,
  };
}

function updateTickLinePosition(tickGroup, newY) {
  for (const use of tickGroup.querySelectorAll("use")) {
    if (use.hasAttribute("y")) {
      use.setAttribute("y", newY.toFixed(6));
    }
  }
}

function updateTickLabel(tickGroup, value, textX, textY) {
  let textGroup = getGroupsByPrefix(tickGroup, "text_")[0];
  if (!textGroup) {
    textGroup = document.createElementNS(SVG_NS, "g");
    textGroup.setAttribute("id", "text_update");
    tickGroup.appendChild(textGroup);
  }
  clearChildren(textGroup);
  const label = formatTickLabel(value);
  const text = document.createElementNS(SVG_NS, "text");
  text.setAttribute("x", (Number.isFinite(textX) ? textX : fallbackTickTextX(tickGroup)).toFixed(6));
  text.setAttribute("y", textY.toFixed(6));
  text.setAttribute("font-size", "10");
  text.setAttribute("font-family", "DejaVu Sans");
  text.setAttribute("fill", "#000000");
  text.textContent = label;
  textGroup.appendChild(text);
}

function updateYAxisTicks(svgElement, newTicks) {
  const axis = getGroupById(svgElement, "matplotlib.axis_2");
  if (!axis) {
    return;
  }
  const tickGroups = getGroupsByPrefix(axis, "ytick_");
  if (!tickGroups.length) {
    return;
  }
  tickGroups.sort((a, b) => {
    const ay = extractTickPosition(a, false) ?? 0;
    const by = extractTickPosition(b, false) ?? 0;
    return by - ay;
  });
  const newTicksSorted = [...newTicks].sort((a, b) => a[1] - b[1]);
  const count = Math.min(tickGroups.length, newTicksSorted.length);
  for (let i = 0; i < count; i += 1) {
    const tickGroup = tickGroups[i];
    const [newY, newValue] = newTicksSorted[i];
    const anchor = extractTickTextAnchor(tickGroup);
    updateTickLinePosition(tickGroup, newY);
    updateTickLabel(
      tickGroup,
      newValue,
      anchor.x,
      newY + (Number.isFinite(anchor.offset) ? anchor.offset : 0)
    );
  }
}

function rescaleSeriesGroups(lineGroups, oldTicks, newMin, newMax, pixelMin, pixelMax) {
  for (const item of lineGroups) {
    const points = parsePathPoints(item.path.getAttribute("d") || "");
    if (!points.length) {
      continue;
    }
    const remapped = points.map(([x, y]) => {
      const data = pixelToData(y, oldTicks);
      const newY = mapDataToPixel(data, newMin, newMax, pixelMin, pixelMax);
      return [x, newY];
    });
    item.path.setAttribute("d", formatPath(remapped, false));
    for (const node of item.group.querySelectorAll("*")) {
      if (node.hasAttribute("y")) {
        const y = Number.parseFloat(node.getAttribute("y") || "");
        if (Number.isFinite(y)) {
          const data = pixelToData(y, oldTicks);
          const newY = mapDataToPixel(data, newMin, newMax, pixelMin, pixelMax);
          node.setAttribute("y", newY.toFixed(6));
        }
      }
      if (node.hasAttribute("cy")) {
        const y = Number.parseFloat(node.getAttribute("cy") || "");
        if (Number.isFinite(y)) {
          const data = pixelToData(y, oldTicks);
          const newY = mapDataToPixel(data, newMin, newMax, pixelMin, pixelMax);
          node.setAttribute("cy", newY.toFixed(6));
        }
      }
    }
  }
}

function rescaleLineAfterRemoval(svgElement, axes, yTicks) {
  if (!Array.isArray(yTicks) || yTicks.length < 2) {
    return;
  }
  const lineGroups = detectLineGroups(axes);
  if (!lineGroups.length) {
    return;
  }
  let maxValue = null;
  for (const item of lineGroups) {
    const points = parsePathPoints(item.path.getAttribute("d") || "");
    for (const [, y] of points) {
      const data = pixelToData(y, yTicks);
      if (!Number.isFinite(data)) {
        continue;
      }
      if (maxValue === null || data > maxValue) {
        maxValue = data;
      }
    }
  }
  if (maxValue === null) {
    return;
  }
  const newMin = pickYMin(yTicks, maxValue);
  let newMax = niceUpperBound(newMin, maxValue, yTicks.length);
  if (newMax === newMin) {
    newMax = newMin + 1;
  }
  const oldTicksSorted = [...yTicks].sort((a, b) => a[1] - b[1]);
  const oldMinPixel = oldTicksSorted[0][0];
  const oldMaxPixel = oldTicksSorted[oldTicksSorted.length - 1][0];
  const newValues = buildTicks(newMin, newMax, yTicks.length);
  const newTicks = newValues.map((v) => [mapDataToPixel(v, newMin, newMax, oldMinPixel, oldMaxPixel), v]);
  updateYAxisTicks(svgElement, newTicks);
  rescaleSeriesGroups(lineGroups, yTicks, newMin, newMax, oldMinPixel, oldMaxPixel);
}

function extractAreaGroups(svgElement) {
  const axes = getAxes(svgElement);
  const groups = getGroupsByPrefix(axes, "FillBetweenPolyCollection_");
  const areas = [];
  for (const group of groups) {
    const id = group.getAttribute("id") || "";
    const path = group.querySelector("path");
    if (!path) {
      continue;
    }
    const fill = (extractPathFill(path) || "").toLowerCase();
    areas.push({
      id,
      group,
      path,
      fill,
      points: parsePathPoints(path.getAttribute("d") || ""),
    });
  }
  areas.sort((a, b) => {
    const ai = Number.parseInt((a.id.match(/_(\d+)$/) || [0, "0"])[1], 10);
    const bi = Number.parseInt((b.id.match(/_(\d+)$/) || [0, "0"])[1], 10);
    return ai - bi;
  });
  return areas;
}

function topBottomByX(points) {
  const byX = new Map();
  for (const [x, y] of points) {
    const arr = byX.get(x) || [];
    arr.push(y);
    byX.set(x, arr);
  }
  const out = new Map();
  for (const [x, ys] of byX.entries()) {
    out.set(x, [Math.min(...ys), Math.max(...ys)]);
  }
  return out;
}

function areaSeriesValues(areas, yTicks) {
  if (!areas.length) {
    return { xValues: [], seriesValues: [] };
  }
  const base = topBottomByX(areas[0].points);
  const xValues = [...base.keys()].sort((a, b) => a - b);
  const seriesValues = [];
  for (const area of areas) {
    const bounds = topBottomByX(area.points);
    const values = [];
    for (const x of xValues) {
      const b = bounds.get(x);
      if (!b) {
        throw new Error("Area series do not share a common x grid.");
      }
      const topData = pixelToData(b[0], yTicks);
      const bottomData = pixelToData(b[1], yTicks);
      values.push(topData - bottomData);
    }
    seriesValues.push(values);
  }
  return { xValues, seriesValues };
}

function chooseFill(existingFills) {
  const palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];
  for (const color of palette) {
    if (!existingFills.includes(color.toLowerCase())) {
      return color;
    }
  }
  return palette[palette.length - 1];
}

function updateAreaSingle(svgElement, question) {
  const axes = getAxes(svgElement);
  const xTicks = parseAxisTicks(svgElement, "matplotlib.axis_1", "xtick_", true);
  const yTicks = parseAxisTicks(svgElement, "matplotlib.axis_2", "ytick_", false);
  if (xTicks.length < 2 || yTicks.length < 2) {
    throw new Error("Insufficient ticks to map area data.");
  }
  const areas = extractAreaGroups(svgElement);
  if (!areas.length) {
    throw new Error("No stacked area collections found in SVG.");
  }
  const { legend, items } = extractLegendItems(svgElement, "area");
  const labels = items.map((x) => x.label);

  if (hasDeleteIntent(question)) {
    const label = matchLabel(question, labels);
    if (!label) {
      throw new Error("No matching area series found in question.");
    }
    const fill = (items.find((x) => x.label === label)?.color || "").toLowerCase();
    const idx = areas.findIndex((x) => x.fill === fill);
    if (idx < 0) {
      throw new Error("No stacked area series matches selected legend color.");
    }
    if (areas.length <= 1) {
      throw new Error("Cannot remove the only remaining area series.");
    }
    const { xValues, seriesValues } = areaSeriesValues(areas, yTicks);
    const remaining = areas.filter((_, i) => i !== idx);
    let cumulative = xValues.map(() => 0);
    remaining.forEach((area, pos) => {
      const srcIdx = pos >= idx ? pos + 1 : pos;
      const values = seriesValues[srcIdx];
      const top = [];
      const bottom = [];
      xValues.forEach((x, i) => {
        const bottomData = cumulative[i];
        const topData = cumulative[i] + values[i];
        bottom.push([x, dataToPixel(bottomData, yTicks)]);
        top.push([x, dataToPixel(topData, yTicks)]);
        cumulative[i] += values[i];
      });
      area.path.setAttribute("d", formatPath([...top, ...bottom.reverse()], true));
    });
    if (areas[idx].group.parentNode === axes) {
      axes.removeChild(areas[idx].group);
    }
    if (legend) {
      removeLegendItem(legend, items, label);
    }
    return { op: "area_remove", label };
  }

  if (hasYearUpdate(question)) {
    const label = matchLabel(question, labels);
    const year = extractYear(question);
    const { value, mode } = extractUpdateValue(question);
    if (!label || year === null || value === null) {
      throw new Error("No valid area update request found in question.");
    }
    const fill = (items.find((x) => x.label === label)?.color || "").toLowerCase();
    const idx = areas.findIndex((x) => x.fill === fill);
    if (idx < 0) {
      throw new Error("No stacked area series matches selected legend color.");
    }
    const { xValues, seriesValues } = areaSeriesValues(areas, yTicks);
    const targetX = dataToPixel(year, xTicks);
    let yearIdx = 0;
    let dist = Number.POSITIVE_INFINITY;
    xValues.forEach((x, i) => {
      const d = Math.abs(x - targetX);
      if (d < dist) {
        dist = d;
        yearIdx = i;
      }
    });
    const current = seriesValues[idx][yearIdx];
    seriesValues[idx][yearIdx] = Math.max(0, mode === "relative" ? current + value : value);

    let cumulative = xValues.map(() => 0);
    areas.forEach((area, i) => {
      const vals = seriesValues[i];
      const top = [];
      const bottom = [];
      xValues.forEach((x, j) => {
        const b = cumulative[j];
        const t = b + vals[j];
        bottom.push([x, dataToPixel(b, yTicks)]);
        top.push([x, dataToPixel(t, yTicks)]);
        cumulative[j] += vals[j];
      });
      area.path.setAttribute("d", formatPath([...top, ...bottom.reverse()], true));
    });
    return { op: "area_year_update", label, year };
  }

  const values = parseValuesFromQuestion(question);
  if (!values.length) {
    throw new Error("No new series values found in question.");
  }
  const existingFills = areas.map((a) => a.fill);
  const fill = chooseFill(existingFills);
  const topMap = new Map();
  for (const area of areas) {
    for (const [x, y] of area.points) {
      const prev = topMap.get(x);
      if (prev === undefined || y < prev) {
        topMap.set(x, y);
      }
    }
  }
  const topBoundary = [...topMap.entries()].sort((a, b) => a[0] - b[0]);
  if (!topBoundary.length) {
    throw new Error("Insufficient area mapping info.");
  }

  const resampled = (() => {
    if (values.length === topBoundary.length) {
      return values;
    }
    if (values.length === 1) {
      return topBoundary.map(() => values[0]);
    }
    const out = [];
    const srcLen = values.length;
    for (let i = 0; i < topBoundary.length; i += 1) {
      const t = (i * (srcLen - 1)) / (topBoundary.length - 1);
      const left = Math.floor(t);
      const right = Math.min(left + 1, srcLen - 1);
      const ratio = t - left;
      out.push(values[left] + ratio * (values[right] - values[left]));
    }
    return out;
  })();

  const baseData = topBoundary.map((p) => pixelToData(p[1], yTicks));
  const newTop = topBoundary.map(([x], i) => [x, dataToPixel(baseData[i] + resampled[i], yTicks)]);
  const polygon = [...newTop, ...[...topBoundary].reverse()];

  const group = ensureUpdateGroup(axes, "FillBetweenPolyCollection_update");
  let path = group.querySelector("path");
  if (!path) {
    path = document.createElementNS(SVG_NS, "path");
    group.appendChild(path);
  }
  const clip = areas[0].path.getAttribute("clip-path");
  if (clip) {
    path.setAttribute("clip-path", clip);
  }
  path.setAttribute("d", formatPath(polygon, true));
  path.setAttribute("style", `fill: ${fill}; fill-opacity: 0.75; stroke: #000000; stroke-width: 0.5`);

  const label = extractSeriesLabel(question, "area");
  if (label && legend) {
    appendLegendText(legend, label);
  }
  return { op: "area_add", count: resampled.length, label: label || null };
}

function splitAreaCommands(question) {
  const raw = String(question || "").trim();
  if (!raw) {
    return [];
  }
  return raw
    .split(/\s*(?:[;；\n]+|然后|并且|同时|and then|then)\s*/i)
    .map((x) => x.trim())
    .filter(Boolean);
}

function updateArea(svgElement, question) {
  const commands = splitAreaCommands(question);
  if (commands.length <= 1) {
    return updateAreaSingle(svgElement, question);
  }
  const steps = [];
  for (const command of commands) {
    steps.push(updateAreaSingle(svgElement, command));
  }
  return { op: "area_combo", steps };
}

function extractXTickLabels(svgElement) {
  const axis = getGroupById(svgElement, "matplotlib.axis_1");
  if (!axis) {
    return [];
  }
  const labels = [];
  for (const tick of getGroupsByPrefix(axis, "xtick_")) {
    const use = tick.querySelector("use");
    const x = Number.parseFloat(use?.getAttribute("x") || "");
    const textGroup = getGroupsByPrefix(tick, "text_")[0] || tick;
    const label = extractCommentText(textGroup) || "";
    if (Number.isFinite(x) && label) {
      labels.push([x, label]);
    }
  }
  return labels;
}

function closestLabel(x, labels) {
  if (!labels.length) {
    return null;
  }
  let best = labels[0];
  let dist = Math.abs(x - best[0]);
  for (const candidate of labels.slice(1)) {
    const d = Math.abs(x - candidate[0]);
    if (d < dist) {
      best = candidate;
      dist = d;
    }
  }
  return best[1];
}

function extractBars(svgElement) {
  const axes = getAxes(svgElement);
  const labels = extractXTickLabels(svgElement);
  const out = [];
  for (const group of getGroupsByPrefix(axes, "patch_")) {
    const id = group.getAttribute("id") || "";
    if (id === "patch_1" || id === "patch_2") {
      continue;
    }
    const path = group.querySelector("path");
    if (!path) {
      continue;
    }
    const style = path.getAttribute("style") || "";
    const clip = path.getAttribute("clip-path");
    if (!clip || /fill:\s*none/.test(style)) {
      continue;
    }
    const points = parsePathPoints(path.getAttribute("d") || "");
    if (points.length < 4) {
      continue;
    }
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    out.push({
      group,
      path,
      xMin,
      xMax,
      yMin,
      yMax,
      fill: extractPathFill(path) || "#1f77b4",
      label: closestLabel((xMin + xMax) / 2, labels),
    });
  }
  return out.filter((b) => b.label);
}

function updateBar(svgElement, question) {
  const bars = extractBars(svgElement);
  const yTicks = parseAxisTicks(svgElement, "matplotlib.axis_2", "ytick_", false);
  if (!bars.length || yTicks.length < 2) {
    throw new Error("Insufficient bar mapping info.");
  }
  const labels = bars.map((b) => b.label);
  const label = matchLabel(question, labels);
  const { value, mode } = extractUpdateValue(question);
  if (!label) {
    throw new Error("No matching category label found in question.");
  }
  if (value === null) {
    throw new Error("No update value found in question.");
  }
  const bar = bars.find((b) => b.label === label);
  if (!bar) {
    throw new Error("Target bar not found.");
  }
  const current = pixelToData(bar.yMin, yTicks);
  const newValue = mode === "absolute" ? value : current + value;
  if (newValue <= current) {
    throw new Error("Only positive bar increases are supported.");
  }
  const newTop = dataToPixel(newValue, yTicks);
  const d = `M ${bar.xMin.toFixed(6)} ${bar.yMax.toFixed(6)} L ${bar.xMax.toFixed(6)} ${bar.yMax.toFixed(
    6
  )} L ${bar.xMax.toFixed(6)} ${newTop.toFixed(6)} L ${bar.xMin.toFixed(6)} ${newTop.toFixed(6)} z`;
  bar.path.setAttribute("d", d);
  return { op: "bar_update", label, value: newValue };
}

export function detectChartType(svgElement) {
  const areas = getGroupsByPrefix(svgElement, "FillBetweenPolyCollection_").length;
  if (areas > 0) {
    return "area";
  }
  const bars = extractBars(svgElement).length;
  if (bars > 0) {
    return "bar";
  }
  const lines = detectLineGroups(getAxes(svgElement)).length;
  if (lines > 0) {
    return "line";
  }
  const points = getGroupById(svgElement, "PathCollection_1")?.querySelectorAll("use").length || 0;
  if (points > 0) {
    return "scatter";
  }
  return "unknown";
}

export function applyQuestionToSvg(svgElement, question) {
  if (!(svgElement instanceof SVGElement)) {
    throw new TypeError("svgElement must be an SVGElement");
  }
  const chartType = detectChartType(svgElement);
  if (chartType === "scatter") {
    return { chart_type: chartType, ...updateScatter(svgElement, question) };
  }
  if (chartType === "line") {
    return { chart_type: chartType, ...updateLine(svgElement, question) };
  }
  if (chartType === "area") {
    return { chart_type: chartType, ...updateArea(svgElement, question) };
  }
  if (chartType === "bar") {
    return { chart_type: chartType, ...updateBar(svgElement, question) };
  }
  throw new Error("Unsupported chart type for text-driven update.");
}

function escapeForTemplateLiteral(input) {
  return input.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
}

export function buildJsModuleFromSvg(svgElement) {
  if (!(svgElement instanceof SVGElement)) {
    throw new TypeError("svgElement must be an SVGElement");
  }
  const serialized = new XMLSerializer().serializeToString(svgElement);
  const escaped = escapeForTemplateLiteral(serialized);
  return `const SVG_STRING = \`${escaped}\`;

export function createSvgElement() {
  const parser = new DOMParser();
  const doc = parser.parseFromString(SVG_STRING, "image/svg+xml");
  const parserError = doc.querySelector("parsererror");
  if (parserError) {
    throw new Error("Invalid SVG content: " + (parserError.textContent || "").trim());
  }
  return document.importNode(doc.documentElement, true);
}

export function mountSvg(target) {
  if (!(target instanceof Element)) {
    throw new TypeError("target must be a DOM Element");
  }
  const svg = createSvgElement();
  target.replaceChildren(svg);
  return svg;
}

export { SVG_STRING };
`;
}

export function createSvgElementFromString(svgString) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(String(svgString || ""), "image/svg+xml");
  const parserError = doc.querySelector("parsererror");
  if (parserError) {
    throw new Error("Invalid SVG content: " + (parserError.textContent || "").trim());
  }
  return document.importNode(doc.documentElement, true);
}

function countScatterPoints(svgElement) {
  const base = getGroupById(svgElement, "PathCollection_1")?.querySelectorAll("use").length || 0;
  const update = getGroupById(svgElement, "PathCollection_update")?.querySelectorAll("use,circle").length || 0;
  return { base, update, total: base + update };
}

function extractLineSeries(svgElement) {
  const axes = getAxes(svgElement);
  const series = [];
  for (const item of detectLineGroups(axes)) {
    const points = parsePathPoints(item.path.getAttribute("d") || "");
    const stroke = extractPathStroke(item.path) || "#000000";
    series.push({ stroke, points_count: points.length });
  }
  return series;
}

function extractAreaSeries(svgElement) {
  return extractAreaGroups(svgElement).map((area) => ({
    id: area.id,
    fill: area.fill,
    points_count: area.points.length,
  }));
}

export function summarizeSvg(svgElement) {
  const chartType = detectChartType(svgElement);
  const xTicks = parseAxisTicks(svgElement, "matplotlib.axis_1", "xtick_", true);
  const yTicks = parseAxisTicks(svgElement, "matplotlib.axis_2", "ytick_", false);
  const bars = chartType === "bar" ? extractBars(svgElement) : [];
  const scatter = chartType === "scatter" ? countScatterPoints(svgElement) : null;
  const lines = chartType === "line" ? extractLineSeries(svgElement) : [];
  const areas = chartType === "area" ? extractAreaSeries(svgElement) : [];

  return {
    chart_type: chartType,
    x_tick_count: xTicks.length,
    y_tick_count: yTicks.length,
    x_range: xTicks.length ? [xTicks[0][1], xTicks[xTicks.length - 1][1]] : null,
    y_range: yTicks.length ? [Math.min(...yTicks.map((t) => t[1])), Math.max(...yTicks.map((t) => t[1]))] : null,
    scatter,
    bars: bars.map((b) => ({ label: b.label, fill: b.fill })),
    lines,
    areas,
  };
}

export function runLocalModel(question, summary, operationResult = null) {
  const q = String(question || "").trim();
  if (!q) {
    return {
      answer: "未提供问题。已返回当前图表摘要。",
      confidence: 0.4,
      used_rule: "empty_question",
      summary,
      operation_result: operationResult,
    };
  }

  const lower = q.toLowerCase();
  if (lower.includes("图表类型") || lower.includes("chart type") || lower.includes("是什么图")) {
    return {
      answer: `当前图表类型是 ${summary.chart_type}。`,
      confidence: 0.95,
      used_rule: "chart_type",
      summary,
      operation_result: operationResult,
    };
  }

  if ((lower.includes("点") || lower.includes("point")) && summary.scatter) {
    return {
      answer: `当前散点总数约为 ${summary.scatter.total}（基础 ${summary.scatter.base}，新增 ${summary.scatter.update}）。`,
      confidence: 0.9,
      used_rule: "scatter_count",
      summary,
      operation_result: operationResult,
    };
  }

  if ((lower.includes("聚类") || lower.includes("cluster") || lower.includes("簇")) && operationResult?.op === "scatter_cluster") {
    return {
      answer: `聚类结果：${operationResult.clusters} 个簇，噪声点 ${operationResult.noise} 个。`,
      confidence: 0.9,
      used_rule: "scatter_cluster",
      summary,
      operation_result: operationResult,
    };
  }

  if ((lower.includes("柱") || lower.includes("bar")) && summary.chart_type === "bar") {
    return {
      answer: `当前柱状图类别数为 ${summary.bars.length}。`,
      confidence: 0.85,
      used_rule: "bar_count",
      summary,
      operation_result: operationResult,
    };
  }

  if ((lower.includes("线") || lower.includes("line")) && summary.chart_type === "line") {
    return {
      answer: `当前折线系列数为 ${summary.lines.length}。`,
      confidence: 0.85,
      used_rule: "line_count",
      summary,
      operation_result: operationResult,
    };
  }

  if ((lower.includes("面积") || lower.includes("area")) && summary.chart_type === "area") {
    return {
      answer: `当前面积系列数为 ${summary.areas.length}。`,
      confidence: 0.85,
      used_rule: "area_count",
      summary,
      operation_result: operationResult,
    };
  }

  return {
    answer: "已完成修改并返回当前图表摘要；该问题暂无专门规则回答。",
    confidence: 0.6,
    used_rule: "fallback_summary",
    summary,
    operation_result: operationResult,
  };
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    const match = String(text || "").match(/\{[\s\S]*\}/);
    if (!match) {
      return null;
    }
    try {
      return JSON.parse(match[0]);
    } catch (_) {
      return null;
    }
  }
}

export async function runRemoteModel(
  question,
  summary,
  operationResult = null,
  config = {}
) {
  const q = String(question || "").trim();
  if (!q) {
    return runLocalModel(question, summary, operationResult);
  }

  const baseUrl = String(config.baseUrl || "").trim().replace(/\/+$/, "");
  const apiKey = String(config.apiKey || "").trim();
  const model = String(config.model || "").trim();
  if (!baseUrl || !apiKey || !model) {
    throw new Error("Model config missing: baseUrl/apiKey/model");
  }

  const prompt = [
    "You are a chart QA assistant.",
    "Use ONLY the provided chart summary and operation result to answer.",
    "Return JSON only: {\"answer\": string, \"confidence\": number, \"issues\": [string]}",
    `Question: ${q}`,
    `Summary: ${JSON.stringify(summary)}`,
    `OperationResult: ${JSON.stringify(operationResult)}`,
  ].join("\n");

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "Answer chart questions with strict JSON output.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`LLM HTTP ${response.status}: ${detail.slice(0, 300)}`);
  }

  const payload = await response.json();
  const content = payload?.choices?.[0]?.message?.content ?? "";
  const parsed = safeJsonParse(content);
  if (parsed && typeof parsed.answer === "string") {
    return {
      answer: parsed.answer,
      confidence:
        typeof parsed.confidence === "number" && Number.isFinite(parsed.confidence)
          ? parsed.confidence
          : 0.7,
      issues: Array.isArray(parsed.issues) ? parsed.issues : [],
      used_rule: "remote_llm_json",
      summary,
      operation_result: operationResult,
      llm_raw: content,
    };
  }

  return {
    answer: String(content || "").trim() || "模型返回为空。",
    confidence: 0.65,
    issues: ["llm_response_not_json"],
    used_rule: "remote_llm_text",
    summary,
    operation_result: operationResult,
    llm_raw: content,
  };
}
