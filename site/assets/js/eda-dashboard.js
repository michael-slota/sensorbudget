"use strict";

const COLORS = {
  ink: "#162126",
  muted: "#5f6d72",
  teal: "#0f766e",
  amber: "#df8f38",
  blue: "#3f6f8f",
  green: "#62a680",
  red: "#c85c55",
  grid: "#e6e9e5",
};

const SPLIT_COLORS = { train: COLORS.blue, test_1: COLORS.amber, test_2: COLORS.green };
const SPLIT_ORDER = ["train", "test_1", "test_2"];

const VIEW_TEXT = {
  balance: {
    title: "Occupancy prevalence changes across periods",
    copy: "Occupancy is the minority class in training and Test 2, but it is substantially more common in Test 1. That makes raw accuracy hard to compare across periods and motivates class-sensitive metrics such as precision, recall, and F1.",
    boundary: "The bars describe the supplied periods; they do not estimate the long-run occupancy rate of this room or another building.",
  },
  schedule: {
    title: "The target contains a strong office schedule",
    copy: "Occupancy concentrates during working hours and is absent overnight in the observed data. A model could exploit that regularity, but a short collection window cannot establish that the same schedule will persist.",
    boundary: "Calendar patterns are descriptive context, not proof that time-of-day features would generalize safely.",
  },
  correlation: {
    title: "Light is the dominant linear signal",
    copy: "Light has by far the strongest Pearson correlation with occupancy. CO2 and Temperature also move with occupancy, while Humidity is weakly correlated. HumidityRatio is derived from Temperature and Humidity rather than measured by another physical sensor.",
    boundary: "Correlation measures association, not causation, reliability under faults, or standalone predictive value.",
  },
  light: {
    title: "Darkness nearly rules out occupancy—but light does not guarantee it",
    copy: "Only one occupied observation across all periods is completely dark. By contrast, many unoccupied observations still have positive Light readings. This asymmetric relationship makes Light powerful while warning that lighting behaviour can act as a shortcut.",
    boundary: "A changed lighting policy, daylight pattern, or failed sensor could invalidate the relationship learned from this room.",
  },
  drift: {
    title: "The held-out periods do not match the training distribution",
    copy: "Test 1 differs most strongly in Temperature. Test 2 shows the largest shifts in Humidity and derived HumidityRatio. Reporting the periods separately exposes changes that a random row-level split could obscure.",
    boundary: "Standardized mean difference compares means relative to spread; it does not capture every kind of distribution change.",
  },
};

const baseLayout = (title) => ({
  title: { text: title, x: 0.02, xanchor: "left", font: { size: 20 } },
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: { family: "DM Sans, sans-serif", color: COLORS.ink },
  margin: { l: 92, r: 42, t: 86, b: 82 },
  hoverlabel: { bgcolor: COLORS.ink, bordercolor: COLORS.ink, font: { color: "white" } },
  legend: { orientation: "h", y: 1.08, x: 1, xanchor: "right" },
});

const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
function styleBarLabels(traces) { traces.filter((trace) => trace.type === "bar").forEach((trace) => { trace.textposition = "auto"; trace.insidetextfont = { color: "#ffffff" }; trace.outsidetextfont = { color: COLORS.ink }; }); }

function balanceFigure(data) {
  const rows = data.split_summary;
  return {
    traces: [
      {
        type: "bar",
        name: "Unoccupied",
        x: rows.map((row) => row.label),
        y: rows.map((row) => row.unoccupied_rows),
        marker: { color: "#cfd8d5" },
        texttemplate: "%{y:,}",
        customdata: rows.map((row) => [row.unoccupied_rows / row.rows, row.rows]),
        hovertemplate: "%{x}<br>Unoccupied: %{y:,}<br>Share: %{customdata[0]:.1%}<br>Total rows: %{customdata[1]:,}<extra></extra>",
      },
      {
        type: "bar",
        name: "Occupied",
        x: rows.map((row) => row.label),
        y: rows.map((row) => row.occupied_rows),
        marker: { color: COLORS.teal },
        texttemplate: "%{y:,}",
        customdata: rows.map((row) => [row.occupancy_rate, row.rows]),
        hovertemplate: "%{x}<br>Occupied: %{y:,}<br>Share: %{customdata[0]:.1%}<br>Total rows: %{customdata[1]:,}<extra></extra>",
      },
    ],
    layout: {
      ...baseLayout("Class balance by supplied period"),
      barmode: "stack",
      xaxis: { title: "Source period", automargin: true, fixedrange: true },
      yaxis: { title: "Observations", gridcolor: COLORS.grid, rangemode: "tozero", automargin: true, fixedrange: true },
    },
  };
}

function scheduleFigure(data) {
  const traces = SPLIT_ORDER.map((split) => {
    const rows = data.hourly_occupancy.filter((row) => row.split === split);
    return {
      type: "scatter",
      mode: "lines+markers",
      name: rows[0].label,
      x: rows.map((row) => row.hour),
      y: rows.map((row) => row.occupancy_rate),
      customdata: rows.map((row) => row.observations),
      line: { color: SPLIT_COLORS[split], width: 3 },
      marker: { size: 7 },
      hovertemplate: "%{fullData.name}<br>Hour: %{x}:00<br>Occupied: %{y:.1%}<br>Observations: %{customdata:,}<extra></extra>",
    };
  });
  return {
    traces,
    layout: {
      ...baseLayout("Occupancy rate by hour and supplied period"),
      hovermode: "x unified",
      xaxis: { title: "Hour of day", dtick: 2, range: [0, 23], automargin: true, fixedrange: true },
      yaxis: { title: "Occupancy rate", tickformat: ".0%", range: [0, 1.01], gridcolor: COLORS.grid, automargin: true, fixedrange: true },
    },
  };
}

function correlationFigure(data) {
  const rows = [...data.target_correlations].reverse();
  return {
    traces: [{
      type: "bar",
      orientation: "h",
      x: rows.map((row) => row.correlation),
      y: rows.map((row) => row.sensor === "HumidityRatio" ? "Humidity ratio" : row.sensor),
      marker: { color: rows.map((row) => row.sensor === "Light" ? COLORS.teal : COLORS.blue) },
      text: rows.map((row) => row.correlation.toFixed(3)),
      textposition: "outside",
      cliponaxis: false,
      hovertemplate: "%{y}<br>Pearson correlation: %{x:.3f}<extra></extra>",
    }],
    layout: {
      ...baseLayout("Sensor correlation with occupancy"),
      showlegend: false,
      xaxis: { title: "Pearson correlation", range: [-0.05, 1.02], gridcolor: COLORS.grid, zerolinecolor: COLORS.ink, automargin: true, fixedrange: true },
      yaxis: { title: "", automargin: true, fixedrange: true },
    },
  };
}

function lightFigure(data) {
  const rows = data.light_exceptions;
  return {
    traces: [
      {
        type: "bar",
        name: "Occupied while dark",
        x: rows.map((row) => row.label),
        y: rows.map((row) => row.occupied_while_dark_rate),
        marker: { color: COLORS.red },
        texttemplate: "%{y:.1%}",
        customdata: rows.map((row) => [row.occupied_while_dark, row.occupied_rows]),
        hovertemplate: "%{x}<br>Occupied while dark: %{customdata[0]:,} / %{customdata[1]:,}<br>Rate: %{y:.3%}<extra></extra>",
      },
      {
        type: "bar",
        name: "Unoccupied while lit",
        x: rows.map((row) => row.label),
        y: rows.map((row) => row.unoccupied_while_lit_rate),
        marker: { color: COLORS.teal },
        texttemplate: "%{y:.1%}",
        customdata: rows.map((row) => [row.unoccupied_while_lit, row.unoccupied_rows]),
        hovertemplate: "%{x}<br>Unoccupied while lit: %{customdata[0]:,} / %{customdata[1]:,}<br>Rate: %{y:.1%}<extra></extra>",
      },
    ],
    layout: {
      ...baseLayout("Exceptions to the Light–occupancy relationship"),
      barmode: "group",
      xaxis: { title: "Source period", automargin: true, fixedrange: true },
      yaxis: { title: "Rate within occupancy state", tickformat: ".0%", range: [0, 0.26], gridcolor: COLORS.grid, automargin: true, fixedrange: true },
    },
  };
}

function driftFigure(data) {
  const sensors = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"];
  const tests = ["test_1", "test_2"];
  const value = (split, sensor) => data.standardized_drift.find((row) => row.split === split && row.sensor === sensor).standardized_mean_difference;
  const z = tests.map((split) => sensors.map((sensor) => value(split, sensor)));
  return {
    traces: [{
      type: "heatmap",
      x: sensors.map((sensor) => sensor === "HumidityRatio" ? "Humidity ratio" : sensor),
      y: ["Test 1", "Test 2"],
      z,
      zmid: 0,
      zmin: -1.1,
      zmax: 1.1,
      colorscale: [[0, "#3f6f8f"], [0.5, "#f6f3ec"], [1, "#c85c55"]],
      text: z.map((row) => row.map((item) => item.toFixed(2))),
      texttemplate: "%{text}",
      hovertemplate: "%{y} vs training<br>%{x}: %{z:.2f} SD<extra></extra>",
      colorbar: { title: "SMD", thickness: 14 },
    }],
    layout: {
      ...baseLayout("Standardized mean difference relative to training"),
      xaxis: { title: "Sensor", automargin: true, fixedrange: true },
      yaxis: { title: "Held-out source period", autorange: "reversed", automargin: true, fixedrange: true },
    },
  };
}

const figureBuilders = { balance: balanceFigure, schedule: scheduleFigure, correlation: correlationFigure, light: lightFigure, drift: driftFigure };

function updateInterpretation(view) {
  const content = VIEW_TEXT[view];
  document.querySelector("#interpretation-title").textContent = content.title;
  document.querySelector("#interpretation-copy").textContent = content.copy;
  document.querySelector("#interpretation-boundary").textContent = `Interpretation boundary: ${content.boundary}`;
}

function activateButton(view) {
  document.querySelectorAll(".view-tab").forEach((button) => {
    const selected = button.dataset.view === view;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}

async function initialiseDashboard() {
  const response = await fetch("../data/eda.json");
  if (!response.ok) throw new Error(`EDA data request failed: ${response.status}`);
  const data = await response.json();
  const chart = document.querySelector("#eda-chart");

  const render = (requestedView) => {
    const view = figureBuilders[requestedView] ? requestedView : "balance";
    const figure = figureBuilders[view](data);
    styleBarLabels(figure.traces);
    Plotly.react(chart, figure.traces, figure.layout, config);
    activateButton(view);
    updateInterpretation(view);
    if (window.location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  };

  document.querySelectorAll(".view-tab").forEach((button) => {
    button.addEventListener("click", () => render(button.dataset.view));
  });
  window.addEventListener("hashchange", () => render(window.location.hash.slice(1)));
  render(window.location.hash.slice(1));
}

window.addEventListener("DOMContentLoaded", () => {
  initialiseDashboard().catch((error) => {
    console.error(error);
    document.querySelector("#eda-chart").hidden = true;
    document.querySelector("#chart-error").hidden = false;
  });
});
