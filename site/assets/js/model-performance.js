"use strict";

const COLORS = { ink: "#162126", muted: "#5f6d72", teal: "#0f766e", amber: "#df8f38", blue: "#3f6f8f", green: "#62a680", red: "#c85c55", grid: "#e6e9e5" };
const VIEW_TEXT = {
  validation: {
    title: "The initial winner depends on the available sensors",
    copy: "Histogram gradient boosting has the highest mean chronological-validation F1 with all five baseline inputs. Without Light, logistic regression leads, but its mean F1 is lower. The large error bars reflect strong fold-to-fold variability, including an all-unoccupied weekend fold.",
    boundary: "These are fixed baseline configurations rather than exhaustively tuned models; small mean differences should not be treated as decisive.",
  },
  heldout: {
    title: "Removing Light is especially costly in Test 2",
    copy: "The all-sensor baseline remains strong in both held-out source periods. The no-Light baseline declines modestly in Test 1 and sharply in Test 2, where lower precision means many occupied predictions are false alarms.",
    boundary: "Test 1 precedes the training window and Test 2 follows it. Both are separate source periods, not evidence from another building.",
  },
  progression: {
    title: "Later experiments use a simpler three-sensor candidate",
    copy: "The initial baseline selected all-sensor histogram boosting. The sensor-budget experiment later selected Temperature + Light + CO2 logistic regression by training-only validation, and this candidate achieved higher clean F1 in both supplied test periods.",
    boundary: "This comparison documents the project progression; repeatedly observed test results must not be used to redesign and reselect candidates.",
  },
  curves: {
    title: "Both candidates rank occupied rows strongly",
    copy: "Each curve shows the precision–recall trade-off as the classification threshold changes. Hovering reveals the underlying threshold. Curves nearer the upper-right retain high precision while recovering more occupied observations.",
    boundary: "A strong ranking curve does not select an operating threshold or attach real energy and comfort costs to errors.",
  },
  errors: {
    title: "The three-sensor candidate makes different numbers of errors across periods",
    copy: "At threshold 0.50, the chart separates false occupied predictions from missed occupied observations. Counts must be interpreted with the number of rows and occupancy prevalence in each source period.",
    boundary: "Error counts describe clean source data. The robustness dashboards separately test what happens when sensor behaviour changes.",
  },
};
const VIEW_SUMMARY = {
  validation: "Histogram boosting leads the all-sensor validation comparison, while logistic regression leads when Light is excluded.",
  heldout: "Removing Light reduces performance in both evaluation periods and produces the larger loss in the second period.",
  progression: "The later three-sensor logistic candidate outperforms the initial all-sensor baseline on both clean evaluation periods.",
  curves: "Both candidates rank occupied observations strongly across thresholds, with hover values exposing the precision–recall trade-off.",
  errors: "At threshold 0.50, the three-sensor candidate makes few errors, but their composition differs across evaluation periods.",
};

const baseLayout = (title) => ({
  title: { text: title, x: 0.02, xanchor: "left", font: { size: 20 } },
  paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
  font: { family: "DM Sans, sans-serif", color: COLORS.ink },
  margin: { l: 116, r: 48, t: 90, b: 94 },
  hoverlabel: { bgcolor: COLORS.ink, bordercolor: COLORS.ink, font: { color: "white" } },
  legend: { orientation: "h", y: 1.1, x: 1, xanchor: "right" },
});
const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
function styleBarLabels(traces) { traces.filter((trace) => trace.type === "bar").forEach((trace) => { trace.textposition = "auto"; trace.insidetextfont = { color: "#ffffff" }; trace.outsidetextfont = { color: COLORS.ink }; }); }

function validationFigure(data) {
  const modelOrder = ["Dummy prior", "Decision tree", "Random forest", "Logistic regression", "Histogram gradient boosting"];
  const buildTrace = (featureLabel, color) => {
    const lookup = new Map(data.cv_summary.filter((row) => row.feature_label === featureLabel).map((row) => [row.model_label, row]));
    return {
      type: "bar", orientation: "h", name: featureLabel, y: modelOrder,
      x: modelOrder.map((model) => lookup.get(model).f1_mean),
      error_x: { type: "data", symmetric: false, array: modelOrder.map((model) => Math.min(lookup.get(model).f1_std, 1 - lookup.get(model).f1_mean)), arrayminus: modelOrder.map((model) => Math.min(lookup.get(model).f1_std, lookup.get(model).f1_mean)), visible: true, color: COLORS.ink, thickness: 1.5, width: 5 },
      marker: { color },
      texttemplate: "%{x:.3f}",
      customdata: modelOrder.map((model) => [lookup.get(model).precision_mean, lookup.get(model).recall_mean, lookup.get(model).f1_std]),
      hovertemplate: "%{y}<br>Mean F1: %{x:.3f}<br>F1 standard deviation: %{customdata[2]:.3f}<br>Mean precision: %{customdata[0]:.3f}<br>Mean recall: %{customdata[1]:.3f}<extra></extra>",
    };
  };
  return { traces: [buildTrace("All-sensor baseline", COLORS.teal), buildTrace("No-Light baseline", COLORS.amber)], layout: { ...baseLayout("Five-fold chronological-validation comparison"), barmode: "group", xaxis: { title: "Mean validation F1 ± one standard deviation (display clipped to 0–1)", range: [0, 1.01], gridcolor: COLORS.grid, automargin: true, fixedrange: true }, yaxis: { title: "", automargin: true, fixedrange: true } } };
}

function heldoutFigure(data) {
  const rows = data.baseline_heldout;
  const categories = rows.map((row) => `${row.split_label}<br>${row.feature_label}`);
  const metric = (name, label, color) => ({ type: "bar", name: label, x: categories, y: rows.map((row) => row[name]), marker: { color }, texttemplate: "%{y:.3f}", hovertemplate: `%{x}<br>${label}: %{y:.3f}<extra></extra>` });
  return { traces: [metric("f1", "F1", COLORS.teal), metric("precision", "Precision", COLORS.blue), metric("recall", "Recall", COLORS.amber)], layout: { ...baseLayout("Held-out baseline metrics at threshold 0.50"), barmode: "group", xaxis: { title: "Source period and baseline", automargin: true, fixedrange: true }, yaxis: { title: "Score", range: [0, 1.01], gridcolor: COLORS.grid, automargin: true, fixedrange: true } } };
}

function progressionFigure(data) {
  const labels = ["All-sensor baseline", "Three-sensor candidate"];
  const traces = ["Test 1", "Test 2"].map((splitLabel, index) => ({
    type: "bar", name: splitLabel, x: labels,
    y: labels.map((label) => data.candidate_comparison.find((row) => row.feature_label === label && row.split_label === splitLabel).f1),
    marker: { color: index === 0 ? COLORS.blue : COLORS.teal },
    texttemplate: "%{y:.3f}", textposition: "outside", cliponaxis: false,
    hovertemplate: "%{x}<br>" + splitLabel + " F1: %{y:.3f}<extra></extra>",
  }));
  return { traces, layout: { ...baseLayout("Clean F1 across the two candidate stages"), barmode: "group", xaxis: { title: "Project candidate", automargin: true, fixedrange: true }, yaxis: { title: "F1", range: [0, 1.04], gridcolor: COLORS.grid, automargin: true, fixedrange: true } } };
}

function curvesFigure(data) {
  const palette = { "All-sensor baseline|Test 1": COLORS.blue, "All-sensor baseline|Test 2": COLORS.amber, "Three-sensor candidate|Test 1": COLORS.green, "Three-sensor candidate|Test 2": COLORS.teal };
  const traces = data.precision_recall_curves.map((curve) => ({
    type: "scatter", mode: "lines", name: `${curve.candidate} · ${curve.split_label}`,
    x: curve.points.map((point) => point.recall), y: curve.points.map((point) => point.precision),
    customdata: curve.points.map((point) => point.threshold),
    line: { width: curve.candidate === "Three-sensor candidate" ? 3.5 : 2.5, color: palette[`${curve.candidate}|${curve.split_label}`] },
    hovertemplate: "Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{customdata:.3f}<extra></extra>",
  }));
  return { traces, layout: { ...baseLayout("Held-out precision–recall curves"), xaxis: { title: "Recall", range: [0, 1.01], gridcolor: COLORS.grid, automargin: true, fixedrange: true }, yaxis: { title: "Precision", range: [0, 1.01], gridcolor: COLORS.grid, automargin: true, fixedrange: true } } };
}

function errorsFigure(data) {
  const rows = data.primary_confusion;
  const errorTrace = (field, name, color) => ({ type: "bar", name, x: rows.map((row) => row.split_label), y: rows.map((row) => row[field]), marker: { color }, texttemplate: "%{y:,}", hovertemplate: "%{x}<br>" + name + ": %{y:,}<extra></extra>" });
  return { traces: [errorTrace("false_positive", "False occupied", COLORS.amber), errorTrace("false_negative", "Missed occupied", COLORS.red)], layout: { ...baseLayout("Three-sensor candidate error counts at threshold 0.50"), barmode: "group", xaxis: { title: "Held-out source period", automargin: true, fixedrange: true }, yaxis: { title: "Rows", rangemode: "tozero", gridcolor: COLORS.grid, automargin: true, fixedrange: true } } };
}

const builders = { validation: validationFigure, heldout: heldoutFigure, progression: progressionFigure, curves: curvesFigure, errors: errorsFigure };
function updateText(view) { const text = VIEW_TEXT[view]; document.querySelector("#interpretation-title").textContent = text.title; document.querySelector("#interpretation-copy").textContent = VIEW_SUMMARY[view]; document.querySelector("#interpretation-boundary").textContent = `Limit: ${text.boundary}`; }
function activate(view) { document.querySelectorAll(".view-tab").forEach((button) => { const selected = button.dataset.view === view; button.classList.toggle("is-active", selected); button.setAttribute("aria-selected", String(selected)); }); }

async function initialise() {
  const response = await fetch("../data/model-performance.json");
  if (!response.ok) throw new Error(`Model data request failed: ${response.status}`);
  const data = await response.json(); const chart = document.querySelector("#model-chart");
  const render = (requested) => { const view = builders[requested] ? requested : "progression"; const figure = builders[view](data); styleBarLabels(figure.traces); window.formatDashboardFigure(figure); Plotly.react(chart, figure.traces, figure.layout, config); activate(view); updateText(view); if (window.location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`); };
  document.querySelectorAll(".view-tab").forEach((button) => button.addEventListener("click", () => render(button.dataset.view)));
  window.addEventListener("hashchange", () => render(window.location.hash.slice(1)));
  render(window.location.hash.slice(1));
}
window.addEventListener("DOMContentLoaded", () => initialise().catch((error) => { console.error(error); document.querySelector("#model-chart").hidden = true; document.querySelector("#chart-error").hidden = false; }));
