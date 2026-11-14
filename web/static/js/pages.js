import { api, debounce } from "./api.js";
import { renderMarkdown } from "./md.js";
import { lineChart, rocChart, barChart } from "./charts.js";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const f4 = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : esc(v));
const inr = (v) => "INR " + Math.round(v).toLocaleString("en-US");
const $ = (root, sel) => root.querySelector(sel);

function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; }
const card = (title, body, cls = "") => `<section class="card ${cls}"><h3>${title}</h3>${body}</section>`;
function table(header, rows) {
  return `<div class="tablewrap"><table><thead><tr>${header.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((c) => `<td>${f4(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
}
const keyTable = (results, filter) => table(["Metric", "Value"], results.filter((r) => filter(r.key)).map((r) => [r.key, r.value]));
function confusionHtml(c) {
  return `<div class="cm" role="table" aria-label="Confusion matrix">
    <div></div><div class="h">Predicted flag</div><div class="h">Predicted no flag</div>
    <div class="h">Actual default</div><div class="c ok"><b>${c.tp}</b>caught (TP)</div><div class="c no"><b>${c.fn}</b>missed (FN)</div>
    <div class="h">Actual paid</div><div class="c no"><b>${c.fp}</b>false alarm (FP)</div><div class="c ok"><b>${c.tn}</b>correct pass (TN)</div></div>`;
}
const metricsHtml = (m) => `<div class="kv">${[["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1", m.f1], ["Specificity", m.specificity]]
  .map(([l, v]) => `<div><b>${v.toFixed(4)}</b><span>${l}</span></div>`).join("")}</div>`;
const findings = (md) => (md ? card("Findings", `<div class="prose">${renderMarkdown(md)}</div>`) : "");
const errBox = (e) => `<div class="error" role="alert">${esc(e.message || e)}</div>`;

// ------------------------------------------------------------------ Overview
export async function overview(root) {
  const d = await api("/api/overview");
  root.innerHTML = `<h1>${esc(d.title)}</h1>
    <p class="lead">A lending-risk analytics study for a UPI payments platform: test a spend assumption, flag borrowers who may default, and segment the book. Every figure below is computed from the project's own outputs.</p>
    <div class="grid">${d.headline.map((h) => `<div class="card stat"><div class="v">${esc(typeof h.value === "number" && !Number.isInteger(h.value) ? h.value.toFixed(4) : h.value)}</div><div class="l">${esc(h.label)}</div><div class="n">${esc(h.note)}</div></div>`).join("")}</div>
    <h2>The problem</h2><div class="card"><ol class="clean">${d.problem.map((p) => `<li>${esc(p)}</li>`).join("")}</ol></div>
    <h2>Key findings</h2><div class="card"><ul class="clean">${d.findings.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>
    <h2>Explore</h2><div class="grid">
    ${[["#/part2", "Threshold lab", "Drag the alert threshold and watch the confusion matrix, ROC point and INR cost move."],
       ["#/part2", "Score a borrower", "Enter seven features and get a default probability with per-feature contributions."],
       ["#/part3", "Segment explorer", "Compare the three borrower segments feature by feature."],
       ["#/part4", "Ensemble lab", "Combine logistic, KNN and stump models by hard or soft vote."]]
      .map(([h, t, n]) => `<a class="card" style="text-decoration:none;color:inherit" href="${h}"><h3>${t}</h3><span class="muted">${n}</span></a>`).join("")}</div>`;
}

// ------------------------------------------------------------------ Part 1
export async function part1(root) {
  const d = await api("/api/part/1");
  root.innerHTML = `<h1>${esc(d.title)}</h1><p class="lead">Is monthly spend still INR 8,000, how many records give a tight estimate, and how far does a regression get?</p>
    <div class="grid two">
      ${card("Estimation and test", keyTable(d.results, (k) => /^(n|mean_value|sd_value|se_value|ci95_|z_stat|reject_h0|n_required)/.test(k)))}
      ${card("Regression baselines", keyTable(d.results, (k) => /^(reg_|r2|rmse|mae|loglog_|mr_r2|mr_adj_r2)/.test(k)))}
      ${card("Gradient descent cost", lineChart({ lines: [{ pts: d.series.gd_loss }], xLabel: "Updates", yLabel: "Cost J", label: "Gradient descent cost against number of updates" }))}
      ${card("Residual spread (funnel check)", barChart({ items: [
        { label: "raw", value: d.results.find((r) => r.key === "resid_sd_ratio_raw").value },
        { label: "log-log", value: d.results.find((r) => r.key === "resid_sd_ratio_log").value }], yLabel: "Upper / lower residual SD", label: "Residual spread ratio, raw versus log-log model" }) +
        `<p class="muted">A ratio near 1 means no funnel; the log-log model fixes it.</p>`)}
    </div>${findings(d.findings)}`;
}

// ------------------------------------------------------------------ Part 2
export async function part2(root) {
  const [d, model, curve] = await Promise.all([api("/api/part/2"), api("/api/model"), api("/api/curve")]);
  const costTau = d.results.find((r) => r.key === "cost_tau").value;
  root.innerHTML = `<h1>${esc(d.title)}</h1>
    <p class="lead">Logistic regression trained from scratch on ${model.train_rows} rows and judged on ${model.test_rows} held-out rows (${model.test_positives} defaults).</p>
    <h2>Threshold lab</h2>
    <div class="grid two">
      <section class="card controls">
        <div><label for="tau">Flag a borrower when probability is at least <output id="tauv">0.30</output></label>
          <input id="tau" type="range" min="0" max="1" step="0.01" value="0.3" aria-describedby="taunote"></div>
        <div class="chips"><button type="button" class="ghost" data-tau="0.5">0.50 default</button><button type="button" class="ghost" data-tau="0.3">0.30</button><button type="button" class="ghost" data-tau="${costTau}">${costTau} cost-optimal</button></div>
        <p id="taunote" class="muted">Cost rule: INR 50 per false alarm, INR 400 per missed default. Model AUC ${curve.auc.toFixed(3)}.</p>
        <div id="tl-out" aria-live="polite"></div>
      </section>
      ${card("ROC curve", '<div id="roc"></div>')}
      ${card("INR cost against threshold", '<div id="costc"></div><div class="legend"><span><i></i>test rows</span><span><i class="b"></i>training rows (used to pick the threshold)</span></div>')}
      ${card("Cost comparison", '<div id="cmp"></div>')}
    </div>
    <h2>KNN K selector</h2>
    <div class="grid two"><section class="card controls">
      <div><label for="kk">Neighbours K: <output id="kkv">5</output></label><input id="kk" type="range" min="1" max="21" step="2" value="${5}"></div>
      <div id="knn-out" aria-live="polite"></div></section>
      ${card("5-fold cross-validated F1 by K", '<div id="cvc"></div>')}</div>
    <h2>Score a borrower</h2>
    <div class="grid two"><form class="card controls" id="scoreform" novalidate>${model.features.map((f) => fieldHtml(f)).join("")}
      <div><label for="s-tau">Flag threshold</label><input id="s-tau" name="tau" type="number" min="0" max="1" step="0.01" value="${model.default_tau}"></div>
      <button type="submit">Score</button></form>
      <section class="card" aria-live="polite" id="score-out"><p class="muted">Submit the form to see the predicted default probability.</p></section></div>
    <h2>Model and results</h2>
    <div class="grid two">${card("Weights and odds ratios", table(["Feature", "Weight", "Odds ratio"], model.features.map((f) => [f.code + " " + f.column, f.weight, f.odds_ratio])) + `<p class="muted">Bias ${model.bias}. Features standardised with training mean and SD.</p>`)}
      ${card("Training loss", lineChart({ lines: [{ pts: d.series.train_loss }], xLabel: "Updates", yLabel: "Cross-entropy", label: "Training loss", xTicks: [0, 250, 500, 750, 1000] }))}
      ${card("Threshold sweep (test rows)", table(d.tables.part2_thresholds.header, d.tables.part2_thresholds.rows))}
      ${card("Learning rate effect (loss after 200 updates)", barChart({ items: d.series.lr_loss.map((p) => ({ label: "eta " + p[0], value: p[1] })), yLabel: "Loss", label: "Loss by learning rate" }))}
    </div>${findings(d.findings)}`;

  // threshold lab
  const tau = $(root, "#tau"), tauv = $(root, "#tauv");
  const update = async () => {
    const out = $(root, "#tl-out");
    try {
      const r = await api("/api/threshold?tau=" + tau.value);
      out.innerHTML = confusionHtml(r.confusion) + metricsHtml(r.metrics) +
        `<p>Flagged ${r.flagged} of ${model.test_rows}. Cost <span class="big">${inr(r.cost_inr)}</span></p>`;
      const pt = curve.roc[Math.round(r.tau * 100)];
      $(root, "#roc").innerHTML = rocChart(curve.roc, { fpr: r.metrics.fpr, tpr: r.metrics.recall });
      $(root, "#costc").innerHTML = lineChart({ lines: [{ pts: curve.cost.map((c) => [c.tau, c.test]) }, { pts: curve.cost.map((c) => [c.tau, c.train]), cls: "ln2" }],
        xLabel: "Threshold", yLabel: "INR cost", label: "Cost against threshold", marker: [r.tau, r.cost_inr], dots: false });
      const b = r.baselines;
      $(root, "#cmp").innerHTML = barChart({ items: [{ label: "flag nobody", value: b.flag_nobody, cls: "bar2" }, { label: "tau 0.5", value: b.at_0_5, cls: "bar2" },
        { label: "tau " + b.cost_tau, value: b.at_cost_tau }, { label: "yours", value: r.cost_inr }], yLabel: "INR cost (test)", label: "Cost comparison" });
    } catch (e) { out.innerHTML = errBox(e); }
  };
  const upd = debounce(update, 60);
  tau.addEventListener("input", () => { tauv.textContent = (+tau.value).toFixed(2); upd(); });
  root.querySelectorAll("[data-tau]").forEach((b) => b.addEventListener("click", () => { tau.value = b.dataset.tau; tauv.textContent = (+tau.value).toFixed(2); update(); }));
  update();

  // KNN
  const kk = $(root, "#kk");
  const knnUpdate = async () => {
    $(root, "#kkv").textContent = kk.value;
    try {
      const r = await api("/api/knn?k=" + kk.value);
      $(root, "#knn-out").innerHTML = confusionHtml(r.confusion) + metricsHtml(r.metrics) + `<p class="muted">Best K by cross-validation: ${r.best_k}.</p>`;
      $(root, "#cvc").innerHTML = barChart({ items: r.cv.map((c) => ({ label: String(c.k), value: c.f1, cls: c.k === +kk.value ? "bar" : "bar2" })), yLabel: "Mean CV F1", label: "Cross-validated F1 for each K" });
    } catch (e) { $(root, "#knn-out").innerHTML = errBox(e); }
  };
  kk.addEventListener("input", debounce(knnUpdate, 60));
  knnUpdate();

  // score form
  $(root, "#scoreform").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const out = $(root, "#score-out"), fd = new FormData(ev.target), body = {};
    for (const [k, v] of fd.entries()) body[k] = v === "" ? null : Number(v);
    try {
      const r = await api("/api/score", body);
      out.innerHTML = `<h3>Predicted default probability</h3><div class="big">${(r.probability * 100).toFixed(1)}%</div>
        <p class="${r.flagged ? "neg" : "pos"}"><strong>${r.flagged ? "Flag for a collections nudge" : "No flag"}</strong> at threshold ${r.tau}.</p>
        ${barChart({ items: r.contributions.map((c) => ({ label: c.feature, value: c.contribution, cls: c.contribution >= 0 ? "bar" : "bar2" })), yLabel: "Contribution to log-odds", label: "Contribution of each feature to the log-odds", signed: true })}
        <p class="muted">Log-odds = bias ${r.bias} + sum of contributions = ${r.logit.toFixed(3)}. Teal bars raise risk, grey bars lower it.</p>`;
    } catch (e) { out.innerHTML = errBox(e); }
  });
}

function fieldHtml(f) {
  const id = "f-" + f.code;
  if (f.kind === "flag") {
    return `<div><label for="${id}">${esc(f.label)}</label><select id="${id}" name="${f.code}"><option value="0">0 - no</option><option value="1" ${f.mean >= 0.5 ? "selected" : ""}>1 - yes</option></select></div>`;
  }
  const step = f.kind === "count" ? 1 : f.code === "X3" ? 0.01 : 0.1;
  const max = f.code === "X3" ? 1 : f.code === "X5" ? 100 : "";
  const def = f.kind === "count" ? Math.round(f.mean) : f.mean;
  return `<div><label for="${id}">${esc(f.label)}</label><input id="${id}" name="${f.code}" type="number" min="0" ${max !== "" ? `max="${max}"` : ""} step="${step}" value="${def}" required>
    <span class="muted" style="font-size:.8rem">typical range ${f.min.toFixed(f.kind === "count" ? 0 : 1)} to ${f.max.toFixed(f.kind === "count" ? 0 : 1)}</span></div>`;
}

// ------------------------------------------------------------------ Part 3
export async function part3(root) {
  const [d, seg] = await Promise.all([api("/api/part/3"), api("/api/segments")]);
  const metrics = [["mean_X1", "Account age (months)"], ["mean_X2", "Failed transactions"], ["mean_X3", "Credit utilisation"],
    ["mean_X4", "Monthly inflow (INR thousands)"], ["mean_X5", "Repayment score"], ["outcome_rate", "Default rate"], ["mean_value", "Mean monthly spend"], ["size", "Segment size"]];
  const bookKey = (m) => (m.startsWith("mean_X") ? m.replace("mean_", "") : m);
  root.innerHTML = `<h1>${esc(d.title)}</h1><p class="lead">K-means on five standardised features finds three borrower segments; PCA and hierarchical clustering give two further views.</p>
    <h2>Segment explorer</h2>
    <div class="grid two"><section class="card controls"><div><label for="seg-m">Compare segments on</label><select id="seg-m">${metrics.map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select></div>
      <div id="seg-chart"></div><div class="legend"><span><i></i>segment</span><span><i class="b"></i>whole book (grey bar)</span></div></section>
      ${card("Segment profiles", table(seg.segments.length ? Object.keys(seg.segments[0]) : [], seg.segments.map((s) => Object.values(s))) + `<p class="muted">A: steady veterans, B: busy high-inflow accounts, C: new heavy borrowers.</p>`)}
      <form class="card controls" id="assignform"><h3>Which segment does a borrower belong to?</h3>
      ${seg.features.map((f, j) => `<div><label for="a-${f.code}">${esc(f.label)}</label><input id="a-${f.code}" name="${f.code}" type="number" step="any" min="0" value="${seg.book_mean[f.code]}" required></div>`).join("")}
      <button type="submit">Assign</button><div id="assign-out" aria-live="polite"></div></form>
      ${card("K-means elbow (within-cluster sum of squares)", lineChart({ lines: [{ pts: d.series.wcss }], xLabel: "K", yLabel: "WCSS", label: "Within-cluster sum of squares by K", xTicks: [2, 3, 4, 5, 6] }) + `<p class="muted">The drop flattens after K = 3.</p>`)}
      ${card("PCA scree", lineChart({ lines: [{ pts: d.series.explained }, { pts: d.series.cumulative, cls: "ln2" }], xLabel: "Component", yLabel: "Variance share", label: "PCA explained and cumulative variance", xTicks: [1, 2, 3, 4, 5], yMax: 1.05 }) + `<div class="legend"><span><i></i>explained</span><span><i class="b"></i>cumulative</span></div>`)}
    </div>
    <h2>Hierarchical clustering of six cohorts</h2>
    <div class="card controls"><div><label for="link">Linkage</label><select id="link"><option>single</option><option>complete</option><option>average</option></select></div><div id="link-out"></div></div>
    ${findings(d.findings)}`;

  const drawSeg = () => {
    const m = $(root, "#seg-m").value, bk = bookKey(m);
    const items = seg.segments.map((s) => ({ label: s.segment, value: s[m] }));
    if (seg.book_mean[bk] !== undefined) items.push({ label: "book", value: seg.book_mean[bk], cls: "bar2" });
    $(root, "#seg-chart").innerHTML = barChart({ items, yLabel: metrics.find((x) => x[0] === m)[1], label: "Segments compared on the chosen feature" });
  };
  $(root, "#seg-m").addEventListener("change", drawSeg); drawSeg();
  $(root, "#assignform").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const body = {}; for (const [k, v] of new FormData(ev.target).entries()) body[k] = v === "" ? null : Number(v);
    const out = $(root, "#assign-out");
    try {
      const r = await api("/api/segment/assign", body);
      out.innerHTML = `<p>Nearest segment: <span class="big">${r.segment}</span></p><p class="muted">Distances in standard deviations: ${Object.entries(r.distances).map(([k, v]) => k + " " + v.toFixed(2)).join(", ")}</p>`;
    } catch (e) { out.innerHTML = errBox(e); }
  });
  const link = () => {
    const t = d.tables.part3_hierarchical, rows = t.rows.filter((r) => r[0] === $(root, "#link").value);
    $(root, "#link-out").innerHTML = table(t.header, rows);
  };
  $(root, "#link").addEventListener("change", link); link();
}

// ------------------------------------------------------------------ Part 4
export async function part4(root) {
  const d = await api("/api/part/4");
  const sp = d.tables.part4_splits, ada = d.tables.part4_adaboost, arch = d.tables.part4_architecture;
  const res = (k) => d.results.find((r) => r.key === k).value;
  const rounds = [...new Set(ada.rows.map((r) => r[0]))];
  root.innerHTML = `<h1>${esc(d.title)}</h1><p class="lead">Which structured model to ship, how boosting reweights mistakes, and how big later neural models would be.</p>
    <h2>Split purity</h2>
    <div class="grid two"><section class="card controls"><div role="group" aria-label="Impurity measure"><button type="button" id="m-info" aria-pressed="true">Information gain</button> <button type="button" id="m-gini" class="ghost" aria-pressed="false">Gini gain</button></div>
      <div id="sp-chart"></div></section>
      ${card("Candidate splits (training rows)", table(sp.header, sp.rows) + `<p class="muted">S1 irregular salary, S2 autopay, S3 utilisation above 0.5, S4 account age at most 24 months. Best: ${res("best_split_by_info_gain")}.</p>`)}</div>
    <h2>Ensemble lab</h2>
    <div class="grid two"><section class="card controls"><fieldset style="border:0;padding:0;margin:0"><legend class="sr">Models to combine</legend><div class="chips">
      ${["logistic", "knn", "stump"].map((m) => `<label class="chip"><input type="checkbox" name="m" value="${m}" checked> ${m}</label>`).join("")}</div></fieldset>
      <div><label for="mode">Vote</label><select id="mode"><option value="hard">hard (majority)</option><option value="soft">soft (average probability)</option></select></div>
      <div><label for="etau">Threshold <output id="etauv">0.30</output></label><input id="etau" type="range" min="0" max="1" step="0.01" value="0.3"></div>
      <div id="ens-out" aria-live="polite"></div></section>
      ${card("Voting results at 0.3 (from outputs)", table(d.tables.part4_voting.header, d.tables.part4_voting.rows))}</div>
    <h2>AdaBoost by hand</h2>
    <div class="grid two"><section class="card controls"><div><label for="round">Round</label><select id="round">${rounds.map((r) => `<option>${r}</option>`).join("")}</select></div><div id="ada-chart"></div>
      <p class="muted">Round 1 error ${res("round1_error")}, alpha ${res("round1_alpha")}. Round 2 error ${res("round2_error")}, alpha ${res("round2_alpha")}. Ensemble accuracy ${res("ensemble_accuracy")}.</p></section>
      ${card("Weights table", '<div id="ada-table"></div>')}</div>
    <h2>Neural network sizing</h2>
    <div class="grid two">${card("Layer parameters", table(arch.header, arch.rows))}
      ${card("Totals", table(["Model", "Parameters"], [["Fully connected (MLP)", res("mlp_total")], ["Convolutional (CNN)", res("cnn_total")], ["Recurrent (RNN)", res("rnn_total")], ["Dense layer on 128x128x3 input", res("fc_layer_params")], ["64 conv filters 3x3", res("conv_layer_params")]]) + `<p class="muted">Convolution shares weights across positions, so it needs far fewer parameters.</p>`)}</div>
    ${findings(d.findings)}`;

  let metricIdx = 5;
  const drawSp = () => { $(root, "#sp-chart").innerHTML = barChart({ items: sp.rows.map((r) => ({ label: r[0], value: r[metricIdx] })), yLabel: sp.header[metricIdx], label: "Impurity gain by candidate split" });
    $(root, "#m-info").setAttribute("aria-pressed", metricIdx === 5); $(root, "#m-gini").setAttribute("aria-pressed", metricIdx === 6);
    $(root, "#m-info").className = metricIdx === 5 ? "" : "ghost"; $(root, "#m-gini").className = metricIdx === 6 ? "" : "ghost"; };
  $(root, "#m-info").onclick = () => { metricIdx = 5; drawSp(); }; $(root, "#m-gini").onclick = () => { metricIdx = 6; drawSp(); }; drawSp();

  const ens = async () => {
    const models = [...root.querySelectorAll('input[name="m"]:checked')].map((c) => c.value), out = $(root, "#ens-out");
    $(root, "#etauv").textContent = (+$(root, "#etau").value).toFixed(2);
    if (!models.length) { out.innerHTML = `<div class="notice">Select at least one model.</div>`; return; }
    try {
      const r = await api("/api/ensemble", { models, mode: $(root, "#mode").value, tau: +$(root, "#etau").value });
      out.innerHTML = confusionHtml(r.confusion) + metricsHtml(r.metrics) + `<p>Cost <span class="big">${inr(r.cost_inr)}</span></p>`;
    } catch (e) { out.innerHTML = errBox(e); }
  };
  const ensD = debounce(ens, 60);
  root.querySelectorAll('input[name="m"], #mode, #etau').forEach((n) => n.addEventListener("input", ensD)); ens();

  const drawAda = () => {
    const rows = ada.rows.filter((r) => String(r[0]) === $(root, "#round").value);
    $(root, "#ada-chart").innerHTML = barChart({ items: rows.flatMap((r) => [{ label: "p" + r[1], value: r[4] }]), yLabel: "Weight after round", label: "Point weights after the selected round" });
    $(root, "#ada-table").innerHTML = table(ada.header, rows);
  };
  $(root, "#round").addEventListener("change", drawAda); drawAda();
}

// ------------------------------------------------------------------ Memo, limitations
export async function memo(root) {
  const d = await api("/api/memo");
  root.innerHTML = `<h1>${esc(d.title)}</h1><div class="card prose">${renderMarkdown(d.text)}</div>`;
}
export async function limitations(root) {
  const d = await api("/api/limitations");
  root.innerHTML = `<h1>${esc(d.title)}</h1><p class="lead">What this analysis cannot tell you.</p><div class="card prose">${renderMarkdown(d.text)}</div>`;
}
