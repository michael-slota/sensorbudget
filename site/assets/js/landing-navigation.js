"use strict";

function enhanceDashboardReadingPath() {
  if (!window.location.pathname.includes("/dashboards/")) return;

  const hero = document.querySelector(".dashboard-hero");
  let periodKey = document.querySelector(".period-key");
  if (hero && !document.querySelector(".period-key")) {
    periodKey = document.createElement("aside");
    periodKey.className = "period-key";
    periodKey.setAttribute("aria-label", "Evaluation period definitions");
    periodKey.innerHTML = `<strong>Evaluation periods</strong><span><b>Test 1</b> = earlier source period, before training</span><span><b>Test 2</b> = later source period, after training</span>`;
    hero.insertAdjacentElement("afterend", periodKey);
  }

  const currentPage = window.location.pathname.split("/").pop();
  const methods = {
    "eda.html": {
      title: "Audit the source data before modeling",
      steps: [
        "Validate all three supplied files, timestamps, target values, and sensor ranges.",
        "Compare class balance, schedules, distributions, and drift across the training and test periods.",
        "Identify redundancy and proxy risks, especially the relationship between Light and occupancy.",
      ],
    },
    "model-performance.html": {
      title: "Compare fixed classifiers without random shuffling",
      steps: [
        "Test a dummy prior, logistic regression, decision tree, random forest, and histogram gradient boosting.",
        "Select models with five expanding chronological folds so every validation block follows its training rows.",
        "Separate the initial all-sensor baseline from the later three-sensor logistic research candidate.",
      ],
    },
    "sensor-selection.html": {
      title: "Repeat model selection for every physical-sensor subset",
      steps: [
        "Evaluate all 15 non-empty combinations of Temperature, Humidity, Light, and CO2.",
        "Select the strongest fixed classifier for each combination using chronological validation only.",
        "Compare validation F1 with five transparent relative-cost scenarios and identify Pareto-efficient options.",
      ],
    },
    "robustness.html": {
      title: "Stress-test fitted candidates without redesigning them",
      steps: [
        "Keep the three validation-frontier configurations fixed.",
        "Inject missingness, noise, complete loss, stuck readings, drift, and changed lighting behaviour.",
        "Measure how the existing models perform on both held-out periods under each controlled fault.",
      ],
    },
    "fault-mitigation.html": {
      title: "Compare three responses to unreliable Light readings",
      steps: [
        "Select a no-Light fallback using training-period validation.",
        "Compare perfect oracle routing with causal fault detection and end-to-end routing.",
        "Test whether fault augmentation or a missingness indicator improves robustness without excessive clean-data cost.",
      ],
    },
    "decision-explainability.html": {
      title: "Audit how a fixed model becomes an operating decision",
      steps: [
        "Select probability thresholds using chronological validation and explicit illustrative error costs.",
        "Check whether the chosen threshold and probability calibration remain stable on both held-out periods.",
        "Explain global coefficients, representative predictions, and recall after occupancy transitions.",
      ],
    },
  };
  const method = methods[currentPage];
  if (method && periodKey && !document.querySelector(".dashboard-method")) {
    const panel = document.createElement("aside");
    panel.className = "dashboard-method";
    panel.innerHTML = `<div><p class="eyebrow">Method</p><h2>${method.title}</h2></div><ol>${method.steps.map((step, index) => `<li><span>0${index + 1}</span>${step}</li>`).join("")}</ol>`;
    periodKey.insertAdjacentElement("afterend", panel);
  }

  const nextPages = {
    "eda.html": ["model-performance.html", "Model performance", "See how candidate models were selected chronologically."],
    "model-performance.html": ["sensor-selection.html", "Sensor trade-offs", "Compare all physical-sensor combinations and cost assumptions."],
    "sensor-selection.html": ["robustness.html", "Robustness", "Test whether the leading clean configurations remain reliable under faults."],
    "robustness.html": ["decision-explainability.html", "Decision analysis", "Review threshold stability, calibration, and model explanations."],
    "fault-mitigation.html": ["decision-explainability.html", "Decision analysis", "Continue to the operating-threshold and explanation evidence."],
    "decision-explainability.html": ["../index.html#summary", "Project summary", "Return to the consolidated conclusion and evidence limits."],
  };
  const next = nextPages[currentPage];
  const footer = document.querySelector(".site-footer");
  if (next && footer && !document.querySelector(".dashboard-next")) {
    const panel = document.createElement("aside");
    panel.className = "dashboard-next";
    panel.innerHTML = `<div><p class="eyebrow">Continue the review</p><h2>Next: ${next[1]}</h2><p>${next[2]}</p></div><a class="button button-primary" href="${next[0]}">Continue →</a>`;
    footer.insertAdjacentElement("beforebegin", panel);
  }
}

function initialiseScrollRail() {
  const defaultSections = ["overview", "results", "analysis", "evidence"];
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
if (document.readyState === "loading") window.addEventListener("DOMContentLoaded", initialisePageNavigation);
else initialisePageNavigation();
