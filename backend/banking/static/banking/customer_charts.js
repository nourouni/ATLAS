(function () {
  const atlasChartPalette = ["#9067C6", "#FFC2E2", "#50AECE", "#F05212", "#FECD04", "#3B6F9E"];
  const parseData = (id) => {
    const node = document.getElementById(id);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (_error) {
      return null;
    }
  };

  const valueFor = (point, key) => {
    const value = point[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };

  const formatNumber = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
  };

  const formatDate = (value) => {
    if (!value) return "";
    const date = new Date(`${value}T00:00:00`);
    return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat(undefined, { day: "2-digit", month: "short" }).format(date);
  };

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));

  const customerSegmentFeatureLabels = { Frequency: "Transaction frequency", Monetary: "Monetary value", avg_balance: "Average balance", net_cash_flow: "Net cash flow", channel_entropy: "Channel diversity", positive_cashflow_week_share: "Positive cash-flow weeks", low_balance_week_share: "Low-balance weeks", digital_tx_pct: "Digital transactions", atm_tx_pct: "ATM transactions" };
  const customerSegmentFeatureOrder = ["Frequency", "Monetary", "avg_balance", "net_cash_flow", "channel_entropy", "positive_cashflow_week_share", "low_balance_week_share", "digital_tx_pct", "atm_tx_pct"];
  const customerSegmentSignal = (label, key, value, color) => {
    const display = key === "Frequency" ? formatNumber(Math.round(value)) : formatNumber(value);
    return `<div class="customer-segment-signal" style="--segment-signal-color:${color}"><div class="customer-segment-signal__heading"><strong>${escapeHtml(label)}</strong><b>${escapeHtml(display)}</b></div><small>${escapeHtml(key)} · actual customer value</small></div>`;
  };
  const enhanceCustomerSegment = (section) => {
    const technical = section.querySelector(".selected-customer-technical-segment");
    const list = technical?.querySelector(".detail-list");
    if (!technical || !list || technical.dataset.enhanced) return;
    technical.dataset.enhanced = "true";
    const fields = Array.from(list.querySelectorAll("dt"));
    const values = Object.fromEntries(fields.map((field) => [field.textContent.trim(), field.nextElementSibling?.textContent.trim() || ""]));
    const name = values.Name || "Assigned segment";
    const clusteringFeatures = parseData("customer-segment-clustering-features-data") || {};
    const segmentMedianFeatures = parseData("customer-segment-median-features-data") || {};
    const header = section.querySelector(".card__header > div");
    if (header) header.innerHTML = `<h2 class="text-card-title">Segment context</h2><h3 class="customer-segment-name">${escapeHtml(name)}</h3><p class="customer-segment-kind">Behavioral segment</p>`;
    const visual = document.createElement("div");
    visual.className = "customer-segment-visual";
    const featureColors = { Frequency: "#50AECE", Monetary: "#9067C6", avg_balance: "#FECD04", net_cash_flow: "#F05212", channel_entropy: "#3B6F9E" };
    const availableFeatures = customerSegmentFeatureOrder.filter((key) => Number.isFinite(Number(clusteringFeatures[key])));
    visual.innerHTML = `<p class="customer-segment-data-note">Actual clustering inputs stored for this customer.</p><div class="customer-segment-signals">${availableFeatures.map((key) => customerSegmentSignal(customerSegmentFeatureLabels[key], key, Number(clusteringFeatures[key]), featureColors[key] || "#50AECE")).join("") || '<p class="empty-state__body">No customer-level clustering inputs are available.</p>'}</div><p class="customer-segment-group-note">These are the variables used to assign the static behavioral cluster.</p>`;
    const dataNote = visual.querySelector(".customer-segment-data-note");
    if (dataNote) dataNote.textContent = "Customer values compared with the median customer in the assigned behavioral segment.";
    const legend = document.createElement("div");
    legend.className = "customer-segment-chart-legend";
    legend.innerHTML = '<span><i class="customer-segment-chart-legend__customer"></i>This customer</span><span><i class="customer-segment-chart-legend__median"></i>Segment median</span>';
    visual.querySelector(".customer-segment-signals")?.before(legend);
    Array.from(visual.querySelectorAll(".customer-segment-signal")).forEach((signal, index) => {
      const key = availableFeatures[index];
      const value = Number(clusteringFeatures[key]);
      const median = Number(segmentMedianFeatures[key]);
      const hasMedian = Number.isFinite(median);
      const maxMagnitude = Math.max(1, Math.abs(value), hasMedian ? Math.abs(median) : 0);
      const bar = document.createElement("div");
      bar.className = "customer-segment-signal__bar";
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", `${customerSegmentFeatureLabels[availableFeatures[index]]}: ${formatNumber(value)}`);
      const zero = document.createElement("span");
      zero.className = "customer-segment-signal__zero";
      const fill = document.createElement("span");
      fill.className = "customer-segment-signal__fill";
      fill.dataset.direction = value < 0 ? "negative" : "positive";
      const barWidth = value === 0 ? 0 : Math.max(6, Math.abs(value) / maxMagnitude * 50);
      fill.style.width = `${Math.min(50, barWidth).toFixed(2)}%`;
      fill.style.left = value < 0 ? `calc(50% - ${fill.style.width})` : "50%";
      bar.append(zero, fill);
      if (hasMedian) {
        const marker = document.createElement("span");
        marker.className = "customer-segment-signal__median";
        marker.style.left = `${(50 + (median / maxMagnitude * 50)).toFixed(2)}%`;
        marker.title = `Segment median: ${formatNumber(median)}`;
        bar.append(marker);
      }
      const note = signal.querySelector("small");
      if (note) note.textContent = `${key} · customer vs segment median`;
      signal.querySelector("small")?.before(bar);
    });
    technical.parentNode.insertBefore(visual, technical);
    technical.querySelector("summary").textContent = "Technical segment evidence";
    const featureGrid = document.createElement("div");
    featureGrid.className = "customer-segment-feature-grid";
    customerSegmentFeatureOrder.forEach((key) => {
      const raw = Number.isFinite(Number(clusteringFeatures[key])) ? Number(clusteringFeatures[key]) : null;
      if (!Number.isFinite(raw)) return;
      const display = /(?:share|_pct)$/i.test(key) ? `${(raw * 100).toFixed(0)}%` : new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(raw);
      featureGrid.insertAdjacentHTML("beforeend", `<article class="customer-segment-feature" title="${escapeHtml(key)} — raw value: ${escapeHtml(raw)}"><span>${escapeHtml(customerSegmentFeatureLabels[key])}</span><strong>${escapeHtml(display)}</strong><small>${escapeHtml(key)}</small></article>`);
    });
    list.replaceWith(featureGrid);
  };

  const renderLineChart = (container) => {
    const data = parseData(container.dataset.chartSource);
    const points = data && Array.isArray(data.points) ? data.points : [];
    const series = (container.dataset.series || "").split(",").map((item) => item.trim()).filter(Boolean);
    const labels = (container.dataset.labels || "").split(",").map((item) => item.trim());
    const title = container.dataset.valueLabel || "Value";
    const usable = points.filter((point) => series.some((key) => valueFor(point, key) !== null));
    if (!usable.length || !series.length) {
      container.innerHTML = '<p class="empty-state__body">No chart data is available.</p>';
      return;
    }

    const isSelectedCustomerForecast = Boolean(container.closest(".selected-customer-outlook"));
    const width = isSelectedCustomerForecast ? 920 : 760;
    const height = isSelectedCustomerForecast ? 380 : 260;
    const pad = isSelectedCustomerForecast
      ? { top: 26, right: 24, bottom: 48, left: 82 }
      : { top: 18, right: 18, bottom: 38, left: 62 };
    const xs = usable.map((_point, index) => index);
    const ys = [];
    usable.forEach((point) => series.forEach((key) => {
      const value = valueFor(point, key);
      if (value !== null) ys.push(value);
    }));
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanY = maxY - minY || 1;
    const xScale = (index) => pad.left + (index / Math.max(1, xs.length - 1)) * (width - pad.left - pad.right);
    const yScale = (value) => pad.top + ((maxY - value) / spanY) * (height - pad.top - pad.bottom);
    const pathFor = (key) => {
      let path = "";
      let open = false;
      usable.forEach((point, index) => {
        const value = valueFor(point, key);
        if (value === null) {
          open = false;
          return;
        }
        path += `${open ? "L" : "M"} ${xScale(index).toFixed(1)} ${yScale(value).toFixed(1)} `;
        open = true;
      });
      return path.trim();
    };

    const seriesMarkup = series.map((key, index) => {
      const label = labels[index] || key;
      const dashed = /^forecast$/i.test(key) ? ' stroke-dasharray="6 5"' : "";
      return `<path class="customer-chart__line customer-chart__line--${key}"${dashed} d="${pathFor(key)}"><title>${label}</title></path>`;
    }).join("");

    const pointMarkup = usable.map((point, index) => series.map((key, seriesIndex) => {
      const value = valueFor(point, key);
      if (value === null) return "";
      const label = labels[seriesIndex] || key;
      const status = key === "forecast" ? "Projected" : "Actual";
      return `<circle class="customer-chart__point customer-chart__point--${key}" cx="${xScale(index).toFixed(1)}" cy="${yScale(value).toFixed(1)}" r="1.5"><title>${escapeHtml(point.date)} - ${escapeHtml(label)}: ${formatNumber(value)} (${status})</title></circle>`;
    }).join("")).join("");

    const yTicks = [minY, minY + spanY / 2, maxY];
    const tickMarkup = yTicks.map((tick) => `<g><line class="customer-chart__grid" x1="${pad.left}" x2="${width - pad.right}" y1="${yScale(tick).toFixed(1)}" y2="${yScale(tick).toFixed(1)}"></line><text class="customer-chart__axis" x="8" y="${(yScale(tick) + 4).toFixed(1)}">${formatNumber(tick)}</text></g>`).join("");
    const firstDate = formatDate(usable[0].date);
    const lastDate = formatDate(usable[usable.length - 1].date);
    const legend = series.length > 1 ? series.map((key) => `<span><i class="customer-chart__legend-swatch customer-chart__legend-swatch--${key}"></i>${labels[series.indexOf(key)] || key}</span>`).join("") : "";
    const forecastIndex = usable.findIndex((point) => valueFor(point, "forecast") !== null);
    const transition = forecastIndex >= 0
      ? `<g><line class="customer-chart__transition" x1="${xScale(forecastIndex).toFixed(1)}" x2="${xScale(forecastIndex).toFixed(1)}" y1="${pad.top}" y2="${height - pad.bottom}"></line><text class="customer-chart__axis" text-anchor="middle" x="${xScale(forecastIndex).toFixed(1)}" y="${pad.top + 12}">Forecast</text></g>`
      : "";

    container.innerHTML = `
      <div class="customer-chart__legend">${legend}</div>
      <svg class="customer-chart__svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${title} over time">
        ${tickMarkup}
        ${transition}
        <line class="customer-chart__axis-line" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"></line>
        <text class="customer-chart__axis" x="${pad.left}" y="${height - 10}">${escapeHtml(firstDate)}</text>
        <text class="customer-chart__axis" text-anchor="end" x="${width - pad.right}" y="${height - 10}">${escapeHtml(lastDate)}</text>
        ${seriesMarkup}
        ${pointMarkup}
      </svg>
    `;
  };

  const renderChannelMix = (container) => {
    const data = parseData(container.dataset.chartSource);
    const items = data && Array.isArray(data.channel_mix) ? data.channel_mix : [];
    if (!items.length) {
      container.innerHTML = '<p class="empty-state__body">No channel mix data is available.</p>';
      return;
    }
    container.innerHTML = items.map((item, index) => {
      const label = String(item.label || "").toLowerCase();
      const channelClass = label.includes("digital") ? "digital" : label.includes("atm") ? "atm" : label.includes("branch") ? "branch" : `extra-${index}`;
      return `
      <div class="channel-mix__row">
        <span>${item.label}</span>
        <div class="bar-track"><span class="channel-mix__bar channel-mix__bar--${channelClass}" style="width: ${Math.max(0, Math.min(100, (item.value || 0) * 100)).toFixed(1)}%"></span></div>
        <strong>${item.display}</strong>
      </div>
    `; }).join("");
  };

  const renderDonutChart = (container) => {
    const source = parseData(container.dataset.chartSource);
    const population = container.dataset.population;
    const rows = source && Array.isArray(source[population]) ? source[population] : [];
    const total = rows.reduce((sum, row) => sum + (Number(row.customer_count) || 0), 0);
    if (!rows.length || !total) {
      container.innerHTML = '<p class="empty-state__body">No segment distribution is available.</p>';
      return;
    }

    let offset = 0;
    const colors = atlasChartPalette;
    const slices = rows.map((row, index) => {
      const count = Number(row.customer_count) || 0;
      const share = count / total;
      const dash = `${(share * 100).toFixed(3)} ${Math.max(0, 100 - share * 100).toFixed(3)}`;
      const slice = `<circle class="donut-chart__slice" r="15.9155" cx="18" cy="18" style="stroke:${colors[index % colors.length]};stroke-dasharray:${dash};stroke-dashoffset:${(-offset).toFixed(3)}"><title>${escapeHtml(row.segment_name)}: ${count} customers, ${row.percentage_display}. ${escapeHtml(row.description || "")}</title></circle>`;
      offset += share * 100;
      return slice;
    }).join("");
    const legend = rows.map((row, index) => `
      <div class="donut-chart__legend-row">
        <i style="background:${colors[index % colors.length]}"></i>
        <span>${escapeHtml(row.segment_name)}</span>
        <strong>${escapeHtml(row.percentage_display)}</strong>
      </div>
    `).join("");

    container.innerHTML = `
      <div class="donut-chart__visual">
        <svg viewBox="0 0 36 36" role="img" aria-label="${population} segment distribution">
          <circle class="donut-chart__track" r="15.9155" cx="18" cy="18"></circle>
          ${slices}
        </svg>
        <div><strong>${total}</strong><span>customers</span></div>
      </div>
      <div class="donut-chart__legend">${legend}</div>
    `;
  };

  const renderOwnershipBars = (chart, items, mode) => {
    if (!items.length) {
      chart.innerHTML = '<p class="empty-state__body">No ownership data is available.</p>';
      return;
    }
    chart.innerHTML = items.map((item) => {
      const label = mode === "segment" ? item.product_name : item.segment_name;
      const pct = Math.max(0, Math.min(100, (Number(item.penetration_value) || 0) * 100));
      return `
        <div class="ownership-bar-row">
          <span>${escapeHtml(label)}</span>
          <div class="bar-track"><span style="width:${pct.toFixed(1)}%"><title>${escapeHtml(item.product_name)} / ${escapeHtml(item.segment_name)}: ${item.owner_count} of ${item.segment_customers} customers (${escapeHtml(item.penetration)})</title></span></div>
          <strong>${escapeHtml(item.penetration)}</strong>
        </div>
      `;
    }).join("");
  };

  const renderOwnershipExplorer = (container) => {
    const source = parseData(container.dataset.chartSource);
    const population = container.dataset.population;
    const data = source && source[population] ? source[population] : null;
    if (!data) {
      container.innerHTML = '<p class="empty-state__body">No ownership data is available.</p>';
      return;
    }
    const segmentEntries = Object.entries(data.by_segment || {});
    const productEntries = Object.entries(data.by_product || {});
    if (!segmentEntries.length && !productEntries.length) {
      container.innerHTML = '<p class="empty-state__body">No ownership data is available.</p>';
      return;
    }

    container.innerHTML = `
      <div class="ownership-explorer__controls">
        <div class="segmented-control" role="tablist">
          <button class="segmented-control__item is-active" type="button" role="tab" aria-selected="true" data-mode="segment">By segment</button>
          <button class="segmented-control__item" type="button" role="tab" aria-selected="false" data-mode="product">By product</button>
        </div>
        <select class="form-select" data-picker></select>
      </div>
      <div class="ownership-bars"></div>
    `;
    const buttons = Array.from(container.querySelectorAll("[data-mode]"));
    const picker = container.querySelector("[data-picker]");
    const chart = container.querySelector(".ownership-bars");
    let mode = "segment";

    const fillPicker = () => {
      const entries = mode === "segment" ? segmentEntries : productEntries;
      picker.innerHTML = entries.map(([id, group]) => `<option value="${escapeHtml(id)}">${escapeHtml(group.segment_name || group.product_name || id)}</option>`).join("");
    };
    const draw = () => {
      const groups = mode === "segment" ? data.by_segment : data.by_product;
      const group = groups[picker.value] || {};
      renderOwnershipBars(chart, group.items || [], mode);
    };

    buttons.forEach((button) => button.addEventListener("click", () => {
      mode = button.dataset.mode;
      buttons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      fillPicker();
      draw();
    }));
    picker.addEventListener("change", draw);
    fillPicker();
    draw();
  };

  const metricButtons = () => `
    <div class="segmented-control" role="tablist">
      <button class="segmented-control__item is-active" type="button" role="tab" aria-selected="true" data-rec-metric="count">Recommendation count</button>
      <button class="segmented-control__item" type="button" role="tab" aria-selected="false" data-rec-metric="score">Average score</button>
    </div>
  `;

  const renderRecommendationBars = (chart, rows, metric) => {
    const sorted = [...rows].sort((a, b) => {
      const ak = metric === "score" ? Number(a.average_score_value) || 0 : Number(a.recommendation_count) || 0;
      const bk = metric === "score" ? Number(b.average_score_value) || 0 : Number(b.recommendation_count) || 0;
      return bk - ak;
    });
    const maxCount = Math.max(1, ...sorted.map((row) => Number(row.recommendation_count) || 0));
    chart.innerHTML = sorted.map((row) => {
      const isScore = metric === "score";
      const value = isScore ? row.average_score : row.recommendation_count;
      const width = isScore
        ? `${Math.max(2, Math.min(100, (Number(row.average_score_value) || 0) * 100)).toFixed(1)}%`
        : `${((Number(row.recommendation_count) || 0) / maxCount * 100).toFixed(1)}%`;
      const segment = row.segment_name ? `${row.segment_name} / ` : "";
      return `
        <article class="analytics-bar-row">
          <span>${escapeHtml(row.product_name)}</span>
          <div class="bar-track"><span style="width:${width}"><title>${escapeHtml(segment)}${escapeHtml(row.product_name)}: ${row.recommendation_count} recommendations, average score ${escapeHtml(row.average_score)}</title></span></div>
          <strong>${escapeHtml(value)}</strong>
        </article>
      `;
    }).join("") || '<p class="empty-state__body">No recommendation data is available.</p>';
  };

  const bindMetricButtons = (container, onChange) => {
    const buttons = Array.from(container.querySelectorAll("[data-rec-metric]"));
    buttons.forEach((button) => button.addEventListener("click", () => {
      buttons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      onChange(button.dataset.recMetric);
    }));
  };

  const renderRecommendationProductChart = (container) => {
    const rows = parseData(container.dataset.chartSource) || [];
    container.innerHTML = `${metricButtons()}<div class="analytics-bars analytics-bars--scroll"></div>`;
    const chart = container.querySelector(".analytics-bars");
    bindMetricButtons(container, (metric) => renderRecommendationBars(chart, rows, metric));
    renderRecommendationBars(chart, rows, "count");
  };

  const renderRecommendationSegmentChart = (container) => {
    const source = parseData(container.dataset.chartSource) || {};
    container.innerHTML = `
      <div class="ownership-explorer__controls">
        <div class="segmented-control" role="tablist">
          <button class="segmented-control__item is-active" type="button" role="tab" aria-selected="true" data-population="retail">Retail</button>
          <button class="segmented-control__item" type="button" role="tab" aria-selected="false" data-population="enterprise">Enterprise</button>
        </div>
        ${metricButtons()}
        <select class="form-select" data-segment-picker></select>
      </div>
      <div class="analytics-bars analytics-bars--scroll"></div>
    `;
    const populationButtons = Array.from(container.querySelectorAll("[data-population]"));
    const picker = container.querySelector("[data-segment-picker]");
    const chart = container.querySelector(".analytics-bars");
    let population = "retail";
    let metric = "count";

    populationButtons.forEach((button) => {
      if (!Object.keys(source[button.dataset.population] || {}).length) {
        button.disabled = true;
        button.title = `${button.textContent.trim()} data unavailable`;
      }
    });

    const fillPicker = () => {
      const entries = Object.entries(source[population] || {});
      picker.innerHTML = entries.map(([id, group]) => `<option value="${escapeHtml(id)}">${escapeHtml(group.segment_name || id)}</option>`).join("");
      picker.disabled = !entries.length;
    };
    const draw = () => {
      const group = (source[population] || {})[picker.value];
      renderRecommendationBars(chart, group ? group.items || [] : [], metric);
    };
    populationButtons.forEach((button) => button.addEventListener("click", () => {
      population = button.dataset.population;
      populationButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      fillPicker();
      draw();
    }));
    bindMetricButtons(container, (nextMetric) => {
      metric = nextMetric;
      draw();
    });
    picker.addEventListener("change", draw);
    fillPicker();
    draw();
  };

  const renderForecastComparisonChart = (container) => {
    const source = parseData(container.dataset.chartSource) || {};
    const series = Array.isArray(source[container.dataset.population]) ? source[container.dataset.population] : [];
    const usable = series.filter((item) => Array.isArray(item.points) && item.points.length);
    if (!usable.length) {
      container.innerHTML = '<p class="empty-state__body">No forecast comparison data is available.</p>';
      return;
    }

    const width = 920;
    const height = 360;
    const pad = { top: 22, right: 24, bottom: 42, left: 82 };
    const dates = [...new Set(usable.flatMap((item) => item.points.map((point) => point.date)))].sort();
    const dateIndex = new Map(dates.map((date, index) => [date, index]));
    const values = usable.flatMap((item) => item.points.map((point) => Number(point.value)).filter(Number.isFinite));
    const minY = Math.min(...values);
    const maxY = Math.max(...values);
    const spanY = maxY - minY || 1;
    const xScale = (date) => pad.left + ((dateIndex.get(date) || 0) / Math.max(1, dates.length - 1)) * (width - pad.left - pad.right);
    const yScale = (value) => pad.top + ((maxY - value) / spanY) * (height - pad.top - pad.bottom);
    const colors = atlasChartPalette;
    const pathFor = (item) => item.points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.date).toFixed(1)} ${yScale(Number(point.value)).toFixed(1)}`)
      .join(" ");
    const lineMarkup = usable.map((item, index) => `
      <path class="forecast-curve__line" d="${pathFor(item)}" style="stroke:${colors[index % colors.length]}">
        <title>${escapeHtml(item.label)}</title>
      </path>
    `).join("");
    const pointMarkup = usable.map((item, index) => item.points.map((point) => `
      <circle class="forecast-curve__point" cx="${xScale(point.date).toFixed(1)}" cy="${yScale(Number(point.value)).toFixed(1)}" r="3" style="fill:${colors[index % colors.length]}">
        <title>${escapeHtml(item.label)} / ${escapeHtml(point.date)}: ${formatNumber(Number(point.value))}</title>
      </circle>
    `).join("")).join("");
    const yTicks = [minY, minY + spanY / 2, maxY];
    const tickMarkup = yTicks.map((tick) => `<g><line class="customer-chart__grid" x1="${pad.left}" x2="${width - pad.right}" y1="${yScale(tick).toFixed(1)}" y2="${yScale(tick).toFixed(1)}"></line><text class="customer-chart__axis" x="8" y="${(yScale(tick) + 4).toFixed(1)}">${formatNumber(tick)}</text></g>`).join("");
    const legend = usable.map((item, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${escapeHtml(item.label)}</span>`).join("");

    container.innerHTML = `
      <div class="forecast-curve__legend">${legend}</div>
      <svg class="customer-chart__svg forecast-curve__svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Forecast curves by behavioral segment">
        ${tickMarkup}
        <line class="customer-chart__axis-line" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"></line>
        <text class="customer-chart__axis" x="${pad.left}" y="${height - 10}">${dates[0]}</text>
        <text class="customer-chart__axis" text-anchor="end" x="${width - pad.right}" y="${height - 10}">${dates[dates.length - 1]}</text>
        ${lineMarkup}
        ${pointMarkup}
      </svg>
    `;
  };

  document.querySelectorAll(".selected-customer-outlook .table-wrap").forEach((table) => {
    const details = document.createElement("details");
    details.className = "selected-customer-forecast-details";
    details.innerHTML = "<summary>Forecast details</summary>";
    table.parentNode.insertBefore(details, table);
    details.appendChild(table);
  });
  document.querySelectorAll(".selected-customer-segment .detail-list").forEach((list) => {
    const details = document.createElement("details");
    details.className = "selected-customer-technical-segment";
    details.innerHTML = "<summary>Technical segment details</summary>";
    list.parentNode.insertBefore(details, list);
    details.appendChild(list);
  });
  document.querySelectorAll(".selected-customer-segment").forEach(enhanceCustomerSegment);
  document.querySelectorAll(".selected-customer-owned table").forEach((table) => {
    const cards = document.createElement("div");
    cards.className = "owned-product-card-grid";
    table.querySelectorAll("tbody tr").forEach((row) => {
      const cells = row.querySelectorAll("td");
      if (cells.length < 3) return;
      const family = (cells[2].textContent.trim() || cells[1].textContent.trim() || "Product").replaceAll("_", " ");
      cards.insertAdjacentHTML("beforeend", `<article class="owned-product-card"><strong>${escapeHtml(cells[0].textContent.trim())}</strong><span>${escapeHtml(family)}</span><b>Owned</b></article>`);
    });
    if (cards.children.length) table.replaceWith(cards);
  });

  document.querySelectorAll(".selected-customer-outlook").forEach((section) => {
    const data = parseData("customer-forecast-chart-data");
    const points = data && Array.isArray(data.points) ? data.points : [];
    const actual = [...points].reverse().find((point) => valueFor(point, "actual") !== null);
    const forecast = points.find((point) => valueFor(point, "forecast") !== null);
    if (!actual && !forecast) return;
    const current = actual ? valueFor(actual, "actual") : null;
    const projected = forecast ? valueFor(forecast, "forecast") : null;
    const change = current !== null && projected !== null && current !== 0 ? ((projected - current) / Math.abs(current)) * 100 : null;
    const compact = (value) => {
      if (value === null) return "n/a";
      return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
    };
    const direction = change === null ? "No change signal is available." : change > 0 ? "Expected group outlook is increasing." : change < 0 ? "Expected group outlook is decreasing." : "Expected group outlook is stable.";
    const summary = document.createElement("div");
    summary.className = `selected-customer-forecast-summary balance-outlook__metrics${change !== null && change < 0 ? " selected-customer-forecast-summary--negative" : ""}`;
    summary.innerHTML = `
      <div><small>Current group average</small><strong>${escapeHtml(compact(current))}</strong></div>
      <div><small>Forecast group average</small><strong>${escapeHtml(compact(projected))}</strong></div>
      <div class="selected-customer-forecast-summary__change"><small>Expected change</small><strong>${change === null ? "n/a" : `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`}</strong></div>
      <p>${escapeHtml(direction)}</p>
    `;
    const chart = section.querySelector("[data-line-chart]");
    if (chart) chart.parentNode.insertBefore(summary, chart);
  });

  document.querySelectorAll("[data-line-chart]").forEach(renderLineChart);
  document.querySelectorAll("[data-channel-mix]").forEach(renderChannelMix);
  document.querySelectorAll("[data-donut-chart]").forEach(renderDonutChart);
  document.querySelectorAll("[data-ownership-explorer]").forEach(renderOwnershipExplorer);
  document.querySelectorAll("[data-recommendation-product-chart]").forEach(renderRecommendationProductChart);
  document.querySelectorAll("[data-recommendation-segment-chart]").forEach(renderRecommendationSegmentChart);
  document.querySelectorAll("[data-forecast-comparison-chart]").forEach(renderForecastComparisonChart);
  document.querySelectorAll(".customer-recommendations__chart .bar-track span").forEach((bar) => {
    const width = parseFloat(bar.style.width);
    if (Number.isFinite(width) && width > 0 && width < 2) bar.style.width = `${Math.min(100, width * 100)}%`;
  });
})();
