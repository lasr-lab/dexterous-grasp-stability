"use strict";

const copyButton = document.querySelector(".copy-bibtex");
const citation = document.getElementById("bibtex-code");
const copyStatus = document.getElementById("copy-status");

if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  const label = copyButton.querySelector("span");
  let resetLabel;

  copyButton.addEventListener("click", async () => {
    clearTimeout(resetLabel);
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      label.textContent = "Copied!";
      copyStatus.textContent = "BibTeX copied to clipboard.";
    } catch {
      // Keep manual copying available when clipboard access is unavailable.
      citation.closest("pre").focus();
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = "Selected";
      copyStatus.textContent = "Citation selected. Use your device's copy command.";
    }
    resetLabel = setTimeout(() => { label.textContent = "Copy"; }, 2200);
  });
}

// Play the short clips only while they are on screen. Audio stays off until the
// viewer unmutes with the controls, and reduced-motion preferences are honored.
const clips = document.querySelectorAll("video[data-clip]");
if (clips.length && "IntersectionObserver" in window) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(({ target, isIntersecting }) => {
      if (isIntersecting && !reduceMotion.matches) {
        // Autoplay can still be blocked; the visible controls remain the fallback.
        target.play().catch(() => {});
      } else if (!target.paused) {
        target.pause();
      }
    });
  }, { threshold: 0.35 });

  clips.forEach((clip) => observer.observe(clip));
}
