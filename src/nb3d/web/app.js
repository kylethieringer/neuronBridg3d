import { Viewer } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const viewer = new Viewer($("viewer"));
const state = { params: null, results: [], cells: [], selected: null, view: "list", sort: "rank" };
// Column sort: ranks ascending, scores descending; ties keep nb3d rank order.
const SORTS = {
  rank: (a, b) => a.rank - b.rank, score: (a, b) => b.score - a.score || a.rank - b.rank,
  forward: (a, b) => b.forward - a.forward || a.rank - b.rank, reverse: (a, b) => b.reverse - a.reverse || a.rank - b.rank,
  nb_rank: (a, b) => a.nb_rank - b.nb_rank,
};
const sorted = () => [...state.results].sort(SORTS[state.sort]);

function setStatus(msg, error = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", error);
}

function searchParams() {
  const libs = [...document.querySelectorAll("input[name=lib]:checked")].map(i => i.value);
  return {
    dataset: $("dataset").value, query: $("query").value.trim(), side: $("side").value,
    segment: $("segment").value, penalty: $("penalty").checked,
    weight: $("weight").value, libraries: libs.join(","),
  };
}

async function runSearch(params) {
  state.params = params;
  const qs = new URLSearchParams(params).toString();
  setStatus("Starting search…");
  for (;;) {
    const r = await fetch(`/api/search?${qs}`);
    const j = await r.json();
    if (!r.ok) { setStatus(j.error || `Error ${r.status}`, true); return; }
    state.cells = j.group.cells;
    const miss = j.group.missing.length ? ` (${j.group.missing.length} not in NeuronBridge)` : "";
    $("groupInfo").textContent = `${j.group.label}: ${j.group.cells.length} cell(s)${miss}`;
    if (j.status === "done") {
      state.results = j.results;
      setStatus(`${j.results.length} lines ranked. Click a row to inspect it in 3D.`);
      render();
      if (j.results.length) select(j.results[0].line);
      return;
    }
    setStatus(`Scoring candidate lines in 3D… ${j.done}/${j.total || "?"} (first search of a neuron downloads images; later ones are cached)`);
    await new Promise(res => setTimeout(res, 1000));
  }
}

function fmtPenalty(p) {
  if (p === null || p === undefined) return "";
  if (p === "n/a") return "n/a";
  return `${Math.round(p * 100)}% extra`;
}

function verdictButtons(r) {
  return ["good", "bad", "unsure"].map(v =>
    `<button type="button" class="${v}${r.verdict === v ? " on" : ""}" data-v="${v}" title="${v}">${{ good: "✓", bad: "✗", unsure: "?" }[v]}</button>`
  ).join("");
}

function render() {
  const showList = state.view === "list";
  $("results").hidden = !showList;
  $("grid").hidden = showList;
  $("viewList").classList.toggle("on", showList);
  $("viewGrid").classList.toggle("on", !showList);
  if (showList) {
    document.querySelectorAll("#results th[data-sort]").forEach(th => th.classList.toggle("on", th.dataset.sort === state.sort));
    $("results").querySelector("tbody").innerHTML = sorted().map(r => `
      <tr data-line="${r.line}" class="${r.line === state.selected ? "sel" : ""}">
        <td>${r.rank}</td>
        <td><strong>${r.line}</strong></td>
        <td>${r.score.toFixed(3)}</td>
        <td class="sub">${r.forward.toFixed(2)}</td>
        <td class="sub">${r.reverse.toFixed(2)}</td>
        <td><span class="badge ${r.source === "3d" ? "d3" : ""}">${r.source === "3d" ? "3D" : "2.5D"}</span>${r.mirrored ? ' <span class="mir" title="matched on the mirrored side">↔</span>' : ""}</td>
        <td title="NeuronBridge score ${r.nb_score}">${r.nb_rank}</td>
        <td>${fmtPenalty(r.penalty)}</td>
        <td class="verdict">${verdictButtons(r)}</td>
        <td class="links"><a href="${r.links.flylight}" target="_blank" rel="noopener">FlyLight</a><a href="${r.links.neuronbridge}" target="_blank" rel="noopener">NB</a></td>
      </tr>`).join("");
  } else {
    $("grid").innerHTML = sorted().map(r => `
      <div class="card${r.line === state.selected ? " sel" : ""}" data-line="${r.line}">
        <img loading="lazy" src="${r.thumbnail_url || ""}" alt="${r.line} colour-depth MIP">
        <div><strong>${r.rank}. ${r.line}</strong> · ${r.score.toFixed(3)}
          <span class="badge ${r.source === "3d" ? "d3" : ""}">${r.source === "3d" ? "3D" : "2.5D"}</span><br>
          fwd ${r.forward.toFixed(2)} · rev ${r.reverse.toFixed(2)} · NB #${r.nb_rank} ${fmtPenalty(r.penalty)}</div>
      </div>`).join("");
  }
}

async function loadGeometry() {
  const r = state.results.find(x => x.line === state.selected);
  if (!r) return;
  const ids = state.cells.map(c => c.body_id).join(",");
  const thr = Number($("threshold").value);
  const qs = new URLSearchParams({ dataset: state.params.dataset, body_ids: ids, image_id: r.best_image_id });
  if (thr > 0) qs.set("threshold", thr);
  if ($("showExpr").checked) qs.set("expression_line", r.line);
  $("viewerTitle").textContent = `${r.line} — loading…`;
  const res = await fetch(`/api/geometry?${qs}`);
  const g = await res.json();
  if (!res.ok) { $("viewerTitle").textContent = g.error; return; }
  viewer.setData({ em: g.em, lm: g.lm, template_outline: g.template_outline, expression: g.expression });
  const img = r.images.find(i => i.image_id === r.best_image_id) || {};
  $("viewerTitle").textContent = `${r.line} · ${img.library || ""} ${img.slide_code || ""} ch${img.channel ?? ""} · ${g.lm_source === "3d" ? "3D stack" : "2.5D from CDM"}${g.expression ? " · full expression shown" : ""}`;
  $("cellToggles").innerHTML = state.cells.map(c => `
    <label title="${c.cell_type || ""}"><input type="checkbox" data-cell="${c.body_id}" checked> ${c.body_id}${c.side ? " " + c.side : ""}${c.segment ? " " + c.segment : ""}${String(r.best_cell) === c.body_id ? " ★" : ""}</label>`).join("");
}

function select(line) {
  state.selected = line;
  if (state.view === "grid") state.view = "list";
  render();
  document.querySelector(`#results tr[data-line="${CSS.escape(line)}"]`)?.scrollIntoView({ block: "nearest" });
  loadGeometry();
}

async function saveVerdict(line, verdict) {
  const res = await fetch("/api/verdict", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: state.params.dataset, query: state.params.query, line, verdict, note: "" }),
  });
  if (!res.ok) { setStatus((await res.json()).error, true); return; }
  state.results.find(r => r.line === line).verdict = verdict;
  render();
}

$("search").addEventListener("submit", (e) => { e.preventDefault(); runSearch(searchParams()); });
$("penalty").addEventListener("change", () => { $("weightWrap").hidden = !$("penalty").checked; });
$("weight").addEventListener("input", () => { $("weightOut").textContent = Number($("weight").value).toFixed(2); });
$("dataset").addEventListener("change", () => { $("segment").disabled = $("dataset").value === "manc"; if ($("segment").disabled) $("segment").value = ""; });
$("segment").disabled = true;
$("viewList").addEventListener("click", () => { state.view = "list"; render(); });
$("viewGrid").addEventListener("click", () => { state.view = "grid"; render(); });
$("results").addEventListener("click", (e) => {
  const th = e.target.closest("th[data-sort]");
  if (th) { state.sort = th.dataset.sort; render(); return; }
  const tr = e.target.closest("tr[data-line]");
  if (!tr) return;
  if (e.target.closest("a")) return;
  const vb = e.target.closest(".verdict button");
  if (vb) { saveVerdict(tr.dataset.line, vb.dataset.v); return; }
  select(tr.dataset.line);
});
$("grid").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) select(c.dataset.line); });
$("cellToggles").addEventListener("change", (e) => { const id = e.target.dataset.cell; if (id) viewer.setCellVisible(id, e.target.checked); });
$("threshold").addEventListener("change", () => {
  $("thrOut").textContent = Number($("threshold").value) > 0 ? $("threshold").value : "auto";
  loadGeometry();
});
$("psize").addEventListener("input", () => viewer.setPointSize(Number($("psize").value)));
$("opacity").addEventListener("input", () => viewer.setOpacity(Number($("opacity").value)));
$("mirror").addEventListener("change", () => viewer.setMirror($("mirror").checked));
$("showExpr").addEventListener("change", loadGeometry);
$("resetView").addEventListener("click", () => viewer.resetView());
