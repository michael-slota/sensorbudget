"use strict";

function initialiseScrollRail() {
  const defaultSections = ["overview", "findings", "analysis", "evidence"];
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
    // Findings are not skipped in favour of the section beneath them.
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

window.initialiseScrollRail = initialiseScrollRail;
if (document.readyState === "loading") window.addEventListener("DOMContentLoaded", initialiseScrollRail);
else initialiseScrollRail();
