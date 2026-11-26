(function () {
  const root = document.documentElement;
  const storedTheme = window.localStorage.getItem("atlas_demo-theme");
  if (storedTheme === "dark") {
    root.classList.add("dark");
  }

  const setThemeIcon = () => {
    const dark = root.classList.contains("dark");
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      const icon = button.querySelector("[data-theme-icon]");
      if (icon) {
        icon.innerHTML = dark ? "&#9788;" : "&#9789;";
      }
    });
  };

  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      root.classList.toggle("dark");
      window.localStorage.setItem("atlas_demo-theme", root.classList.contains("dark") ? "dark" : "light");
      setThemeIcon();
    });
  });
  setThemeIcon();

  const sidebar = document.querySelector(".nav-sidebar");
  document.querySelectorAll("[data-sidebar-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      if (sidebar) {
        sidebar.classList.toggle("is-open");
      }
    });
  });

  const routes = {
    overview: "/dashboard/",
    dashboard: "/dashboard/",
    customer: "/dashboard/customers/",
    customers: "/dashboard/customers/",
    client: "/dashboard/customers/",
    clients: "/dashboard/customers/",
    segment: "/dashboard/segments/",
    segments: "/dashboard/segments/",
    forecast: "/dashboard/forecasting/",
    forecasting: "/dashboard/forecasting/",
    product: "/dashboard/products/",
    products: "/dashboard/products/",
    propensity: "/dashboard/products/",
    recommendation: "/dashboard/recommendations/",
    recommendations: "/dashboard/recommendations/",
    recs: "/dashboard/recommendations/",
    assistant: "/chatbot/",
  };

  document.querySelectorAll("[data-dashboard-search]").forEach((form) => {
    const input = form.querySelector("input[type='search']");
    const status = form.querySelector("[data-dashboard-search-status]");
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = (input && input.value ? input.value : "").trim();
      if (!query) {
        if (status) status.textContent = "Enter a customer ID, page, segment, product, forecast, or recommendation.";
        return;
      }

      const lower = query.toLowerCase();
      const numeric = lower.match(/(?:^|\b)(?:#?\s*)(\d{2,})(?:\b|$)/);
      if (numeric) {
        window.location.assign(`/dashboard/customers/?customer_id=${encodeURIComponent(numeric[1])}`);
        return;
      }

      const matchedKey = Object.keys(routes).find((key) => lower.includes(key));
      if (matchedKey) {
        window.location.assign(routes[matchedKey]);
        return;
      }

      if (status) status.textContent = `No dashboard match for "${query}".`;
    });
  });

  document.querySelectorAll("[data-dashboard-export]").forEach((button) => {
    button.addEventListener("click", () => window.print());
  });

  document.querySelectorAll("[data-panel-target]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) return;
      const target = button.getAttribute("data-panel-target");
      const group = button.closest("[data-panel-group]")?.getAttribute("data-panel-group");
      if (!group) return;
      document.querySelectorAll(`[data-panel-group="${group}"] [data-panel-target]`).forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(`[data-panel-group="${group}"][data-panel]`).forEach((panel) => {
        panel.hidden = panel.getAttribute("data-panel") !== target;
      });
    });
  });
})();
