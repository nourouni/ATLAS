(function () {
  const page = document.getElementById("page");
  const setTheme = (mode) => {
    page.dataset.theme = mode;
    document.querySelectorAll("[data-theme-choice]").forEach((button) => button.classList.toggle("active", button.dataset.themeChoice === mode));
    window.localStorage.setItem("atlas_demo-theme", mode);
  };
  setTheme(window.localStorage.getItem("atlas_demo-theme") === "dark" ? "dark" : "light");
  document.querySelectorAll("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.themeChoice)));

  const password = document.getElementById("password");
  const toggle = document.getElementById("toggle-password");
  toggle.addEventListener("click", () => {
    const visible = password.type === "text";
    password.type = visible ? "password" : "text";
    toggle.setAttribute("aria-label", visible ? "Show password" : "Hide password");
  });

  const scatter = document.getElementById("scatter");
  let seed = 42;
  const random = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280);
  const points = Array.from({ length: 160 }, () => ({ x: random() * 1200, y: random() * 800, r: 0.6 + random() * 1.1, tint: random() > 0.96 ? "tint-a" : (random() > 0.93 ? "tint-b" : "") }));
  let links = "", linkCount = 0;
  points.forEach((a, i) => points.slice(i + 1).forEach((b) => {
    if (linkCount < 45 && Math.hypot(a.x - b.x, a.y - b.y) < 55 && random() > 0.55) {
      links += `<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}"/>`;
      linkCount += 1;
    }
  }));
  scatter.innerHTML = links + points.map((p) => `<circle class="${p.tint}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${p.r.toFixed(2)}"/>`).join("");

  const layer = document.getElementById("parallax");
  let frame;
  window.addEventListener("mousemove", (event) => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      const x = (event.clientX / window.innerWidth - 0.5) * 24;
      const y = (event.clientY / window.innerHeight - 0.5) * 24;
      layer.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      frame = null;
    });
  });
})();
