// Inline SVG charts (no library). Each function returns an SVG markup string.
const W = 560, H = 300, M = { l: 52, r: 16, t: 14, b: 40 };
const fmt = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : +v.toPrecision(3) + "");

function scale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range, span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}
function ticks(a, b, n = 5) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(a + ((b - a) * i) / n);
  return out;
}
function frame(sx, sy, xd, yd, xl, yl, xt) {
  let s = "";
  for (const t of ticks(yd[0], yd[1])) {
    s += `<line class="grid-l" x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${M.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`;
  }
  for (const t of xt || ticks(xd[0], xd[1])) {
    s += `<text x="${sx(t)}" y="${H - M.b + 16}" text-anchor="middle">${fmt(t)}</text>`;
  }
  s += `<line class="ax" x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}"/><line class="ax" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${H - M.b}"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${xl}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
  return s;
}
const svg = (inner, label) => `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">${inner}</svg>`;

// lines: [{pts:[[x,y]...], cls:"ln"}]; marker: [x,y] optional
export function lineChart({ lines, xLabel, yLabel, label, marker, xTicks, yMin, yMax, xMin, xMax, dots = true }) {
  const all = lines.flatMap((l) => l.pts);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const xd = [xMin ?? Math.min(...xs), xMax ?? Math.max(...xs)];
  const yd = [yMin ?? Math.min(0, ...ys), yMax ?? Math.max(...ys) * 1.05];
  const sx = scale(xd, [M.l, W - M.r]), sy = scale(yd, [H - M.b, M.t]);
  let s = frame(sx, sy, xd, yd, xLabel, yLabel, xTicks);
  for (const l of lines) {
    s += `<polyline class="${l.cls || "ln"}" points="${l.pts.map((p) => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" ")}"/>`;
    if (dots) for (const p of l.pts) s += `<circle class="pt" r="3" cx="${sx(p[0])}" cy="${sy(p[1])}"><title>${xLabel} ${fmt(p[0])}: ${fmt(p[1])}</title></circle>`;
  }
  if (marker) s += `<circle class="mark" r="6" cx="${sx(marker[0])}" cy="${sy(marker[1])}"/>`;
  return svg(s, label);
}

// ROC: curve points [{fpr,tpr}], current point
export function rocChart(roc, cur) {
  const sx = scale([0, 1], [M.l, W - M.r]), sy = scale([0, 1], [H - M.b, M.t]);
  let s = frame(sx, sy, [0, 1], [0, 1], "False positive rate", "True positive rate (recall)");
  s += `<line class="grid-l" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>`;
  s += `<polyline class="ln" points="${roc.map((p) => sx(p.fpr).toFixed(1) + "," + sy(p.tpr).toFixed(1)).join(" ")}"/>`;
  if (cur) s += `<circle class="mark" r="6" cx="${sx(cur.fpr)}" cy="${sy(cur.tpr)}"><title>current threshold</title></circle>`;
  return svg(s, "ROC curve of the logistic model on the test rows with the current threshold marked");
}

// Bars: items [{label,value,cls?}]
export function barChart({ items, yLabel, label, signed = false }) {
  const vals = items.map((i) => i.value);
  const yd = [Math.min(0, ...vals) * (signed ? 1.1 : 1), Math.max(0, ...vals) * 1.1 || 1];
  const sy = scale(yd, [H - M.b, M.t]);
  const bw = (W - M.l - M.r) / items.length;
  let s = "";
  for (const t of ticks(yd[0], yd[1])) s += `<line class="grid-l" x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${M.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`;
  s += `<line class="ax" x1="${M.l}" x2="${W - M.r}" y1="${sy(0)}" y2="${sy(0)}"/>`;
  items.forEach((it, i) => {
    const x = M.l + i * bw + bw * 0.18, y0 = sy(0), y1 = sy(it.value);
    s += `<rect class="${it.cls || "bar"}" x="${x}" y="${Math.min(y0, y1)}" width="${bw * 0.64}" height="${Math.abs(y1 - y0)}" rx="3"><title>${it.label}: ${fmt(it.value)}</title></rect>`;
    s += `<text x="${x + bw * 0.32}" y="${H - M.b + 16}" text-anchor="middle">${it.label}</text>`;
  });
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
  return svg(s, label);
}
