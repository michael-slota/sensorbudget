"use strict";

function enhanceDashboardReadingPath() {
  if (!window.location.pathname.includes("/dashboards/")) return;

  const hero = document.querySelector(".dashboard-hero");
  const currentPage = window.location.pathname.split("/").pop();
  // All visible header, answer, and method content is stored in the HTML so it
  // appears on first paint. The code below remains only as a fallback for an
  // older or incomplete dashboard page.
  if (hero?.querySelector(".dashboard-switcher") && document.querySelector(".dashboard-answer") && document.querySelector(".dashboard-method")) return;
  const dashboardPages = [
    ["model-performance.html", "Model performance", "core"],
    ["sensor-selection.html", "Sensor trade-offs", "core"],
    ["robustness.html", "Robustness", "core"],
    ["decision-explainability.html", "Decision analysis", "core"],
    ["eda.html", "EDA", "supporting"],
    ["fault-mitigation.html", "Fault mitigation", "supporting"],
  ];
  const methods = {
    "eda.html": {
      title: "Audit the source data before modeling",
      answer: "The source data is structurally clean, but Light is a dominant occupancy proxy and the single-room sample limits broader generalization.",
      steps: [
        "Validate all three supplied files, timestamps, target values, and sensor ranges.",
        "Compare class balance, schedules, distributions, and drift across the training and test periods.",
        "Identify redundancy and proxy risks, especially the relationship between Light and occupancy.",
      ],
    },
    "model-performance.html": {
      title: "Compare fixed classifiers without random shuffling",
      answer: "Chronological evaluation produces strong candidates, but performance changes significantly across feature sets and source periods.",
      steps: [
        "Test a dummy prior, logistic regression, decision tree, random forest, and histogram gradient boosting.",
        "Select models with five expanding chronological folds so every validation block follows its training rows.",
        "Separate the initial all-sensor baseline from the later three-sensor logistic research candidate.",
      ],
    },
    "sensor-selection.html": {
      title: "Repeat model selection for every physical-sensor subset",
      answer: "The sensor combination Temperature + Light + CO2 leads validation, while three Light-containing configurations remain Pareto-efficient across the tested costs, so the evidence does not justify one hardware choice.",
      steps: [
        "Evaluate all 15 non-empty combinations of Temperature, Humidity, Light, and CO2.",
        "Select the strongest fixed classifier for each combination using chronological validation only.",
        "Compare validation F1 with five transparent relative-cost scenarios and identify Pareto-efficient options.",
      ],
      assumption: "Reference sensor-cost example (relative points): Temperature 1, Humidity 1, Light 0.5, and CO2 4. Sensitivity cases also test equal costs, CO2 at 1.5 or 8, and Light at 5.",
    },
    "robustness.html": {
      title: "Simulate three families of sensor and operating failures",
      answer: "Strong clean performance hides a critical dependence on Light, with complete Light loss and occupied darkness reducing F1 to approximately zero.",
      steps: [
        "Data availability: randomly missing cells represent intermittent gaps, while complete sensor loss represents a sustained outage replaced by the training median.",
        "Measurement quality: Gaussian noise represents unstable readings, stuck-low or stuck-high values represent a frozen sensor, and gradual drift represents calibration bias building over time.",
        "Changed room behaviour: occupied darkness and an unoccupied-but-lit room break the historical lighting pattern that the fitted models learned as an occupancy proxy.",
      ],
      assumption: "The three frontier candidates remain fixed and are not retrained for each failure. These controlled scenarios diagnose sensitivity; they do not estimate how frequently real hardware faults occur.",
    },
    "fault-mitigation.html": {
      title: "Compare three responses to unreliable Light readings",
      answer: "Detector routing is the strongest tested mitigation overall, but false alarms, missed plausible faults, and a weaker fallback prevent full recovery.",
      steps: [
        "Fallback model: select a Temperature + CO2 model that never uses Light. It sacrifices some clean performance but remains available when Light cannot be trusted.",
        "Detection and routing: monitor Light for missing, out-of-range, frozen, or abrupt readings and send flagged rows to the fallback. Oracle routing first shows the best recovery possible with perfect fault knowledge.",
        "Fault-aware training: inject simulated Light faults during training, replace missing values with the training median, and optionally add an indicator telling the model that replacement occurred.",
      ],
      assumption: "The comparison measures both recovery and side effects. A useful mitigation must improve faulted predictions without routing too many healthy rows or creating an unacceptable clean-data penalty.",
    },
    "decision-explainability.html": {
      title: "Audit how a fixed model becomes an operating decision",
      answer: "Under the stated validation cost assumptions, threshold 0.86 appears most cost-effective, but that advantage does not hold on the held-out periods; model explanations also confirm that Light drives the prediction most strongly.",
      steps: [
        "Select probability thresholds using chronological validation and explicit illustrative error costs.",
        "Check whether the chosen threshold and probability calibration remain stable on both held-out periods.",
        "Explain global coefficients, representative predictions, and recall after occupancy transitions.",
      ],
      assumption: "Illustrative error-cost ratios (false occupied:false unoccupied): equal 1:1; comfort-focused 1:5, where missing occupancy costs more; energy-focused 5:1, where responding to an empty room costs more.",
    },
  };
  const method = methods[currentPage];
  const currentIndex = dashboardPages.findIndex(([path]) => path === currentPage);
  if (hero && currentIndex >= 0 && !hero.querySelector(".dashboard-switcher")) {
    const navigation = document.createElement("nav");
    navigation.className = "dashboard-switcher";
    navigation.setAttribute("aria-label", "Dashboard pages");
    const renderGroup = (group, label) => `<div class="dashboard-nav-group"><span>${label}</span><div>${dashboardPages.map(([path, pageLabel, pageGroup], index) => pageGroup === group ? `<a href="${path}"${index === currentIndex ? ' class="is-current" aria-current="page"' : ""}>${pageLabel}</a>` : "").join("")}</div></div>`;
    navigation.innerHTML = renderGroup("core", "Main review") + renderGroup("supporting", "Deep dives");
    const breadcrumb = hero.querySelector(".breadcrumb");
    if (breadcrumb) breadcrumb.insertAdjacentElement("afterend", navigation);
    else hero.querySelector("div")?.prepend(navigation);
  }
  let answerPanel = document.querySelector(".dashboard-answer");
  if (method && hero && !answerPanel) {
    answerPanel = document.createElement("aside");
    answerPanel.className = "dashboard-answer";
    answerPanel.innerHTML = `<span>Answer</span><p>${method.answer}</p>`;
    hero.insertAdjacentElement("afterend", answerPanel);
  }
  if (method && hero && !document.querySelector(".dashboard-method")) {
    const panel = document.createElement("aside");
    panel.className = "dashboard-method";
    panel.innerHTML = `<details><summary><span><small>Method</small>${method.title}</span><span class="method-toggle" aria-hidden="true">Details +</span></summary><div class="method-details">${method.assumption ? `<p class="method-assumption"><strong>Assumptions</strong>${method.assumption}</p>` : ""}<ol>${method.steps.map((step, index) => `<li><span>0${index + 1}</span>${step}</li>`).join("")}</ol></div></details>`;
    (document.querySelector(".finding-strip") || answerPanel || hero).insertAdjacentElement("afterend", panel);
  }

  const highlights = document.querySelector(".finding-strip");
  if (answerPanel && highlights) {
    highlights.classList.add("dashboard-summary");
    highlights.prepend(answerPanel);
    hero.insertAdjacentElement("afterend", highlights);
  }

}

function formatDashboardFigure(value) {
  if (typeof value === "string") return value.replaceAll("Test 1", "First period").replaceAll("Test 2", "Second period");
  if (Array.isArray(value)) return value.map(formatDashboardFigure);
  if (value && typeof value === "object") {
    Object.keys(value).forEach((key) => { value[key] = formatDashboardFigure(value[key]); });
  }
  return value;
}

function initialiseScrollRail() {
  const defaultSections = ["overview", "highlights", "analysis", "evidence"];
  const mainSections = [...document.querySelectorAll("main > section")].slice(0, defaultSections.length);
  if (!document.querySelector("[data-page-section]") && mainSections.length) {
    mainSections.forEach((section, index) => {
      section.id = defaultSections[index];
      section.setAttribute("data-page-section", "");
    });
  }

  if (!document.querySelector(".scroll-rail") && mainSections.length) {
    const rail = document.createElement("nav");
    rail.className = "scroll-rail";
    rail.setAttribute("aria-label", "Dashboard sections");
    rail.innerHTML = `<div class="scroll-rail-track" aria-hidden="true"><span id="scroll-rail-progress"></span></div><div class="scroll-rail-links">${defaultSections.map((id, index) => `<a href="#${id}" data-scroll-section="${id}"${index === 0 ? ' aria-current="true"' : ""}><span class="scroll-rail-dot"></span>${id[0].toUpperCase() + id.slice(1)}</a>`).join("")}</div><span class="scroll-rail-more" aria-hidden="true">↓</span>`;
    document.body.prepend(rail);
  }

  const progress = document.querySelector("#scroll-rail-progress");
  const rail = document.querySelector(".scroll-rail");
  const sections = [...document.querySelectorAll("[data-page-section]")];
  const links = [...document.querySelectorAll("[data-scroll-section]")];
  if (!progress || !rail || sections.length === 0) return;

  let scheduled = false;

  const update = () => {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const fraction = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
    progress.style.height = `${fraction * 100}%`;
    rail.style.setProperty("--page-progress", fraction.toFixed(3));

    // Anchor navigation scrolls a section to the top of the viewport. Keep the
    // active-section marker near that same position so short sections such as
    // Short result sections are not skipped in favour of the section beneath them.
    const marker = window.scrollY + Math.min(96, window.innerHeight * 0.12);
    let active = sections[0].id;
    sections.forEach((section) => {
      if (section.offsetTop <= marker) active = section.id;
    });
    // The final section cannot always reach the top when it is shorter than the
    // viewport. At the bottom of the page it should nevertheless be active.
    if (fraction > 0.985) active = sections.at(-1).id;

    links.forEach((link) => {
      const selected = link.dataset.scrollSection === active;
      link.classList.toggle("is-active", selected);
      if (selected) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });

    rail.classList.toggle("is-complete", fraction > 0.985);
    scheduled = false;
  };

  const requestUpdate = () => {
    if (!scheduled) {
      scheduled = true;
      window.requestAnimationFrame(update);
    }
  };

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  update();
}

function initialisePageNavigation() {
  enhanceDashboardReadingPath();
  initialiseScrollRail();
}

window.initialiseScrollRail = initialiseScrollRail;
window.formatDashboardFigure = formatDashboardFigure;
if (document.readyState === "loading") window.addEventListener("DOMContentLoaded", initialisePageNavigation);
else initialisePageNavigation();
