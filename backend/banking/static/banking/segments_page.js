(function () {
  const dataNode = document.getElementById("segments-workspace-data");
  if (!dataNode) return;
  let workspace;
  try { workspace = JSON.parse(dataNode.textContent); } catch (_error) { return; }
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  const format = (value) => value === null || value === undefined ? "n/a" : new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
  const formatDate = (value) => { const date = new Date(`${value}T00:00:00`); return Number.isNaN(date.getTime()) ? String(value || "") : new Intl.DateTimeFormat(undefined, { day: "2-digit", month: "short" }).format(date); };
  const atlasChartPalette = ["#9067C6", "#FFC2E2", "#50AECE", "#F05212", "#FECD04", "#3B6F9E"];
  const segmentColors = { "Active Low-Balance": "#F05212", "Stable Multi-Product": "#50AECE", "Moderate Low-Balance": "#FECD04", "Low-Activity ATM": "#3B6F9E", "ATM & Study Services": "#9067C6", "Active Digital": "#FFC2E2", "Credit & Cash-Flow Pressure": "#F05212", "Irregular Cash Flow": "#3B6F9E", "Stable High-Balance": "#50AECE", "Atypical Retail": "#9067C6", "Atypical Enterprise": "#9067C6" };
  const colorFor = (name, index) => segmentColors[name] || atlasChartPalette[index % atlasChartPalette.length];

  function renderDistribution(population) {
    const container = document.querySelector(`[data-distribution-chart="${population}"]`);
    const rows = workspace[population]?.distribution || [];
    if (!container) return;
    if (!rows.length) { container.innerHTML = '<p class="empty-state__body">No segment distribution is available.</p>'; return; }
    let offset = -Math.PI / 2;
    const point = (angle) => [100 + 94 * Math.cos(angle), 100 + 94 * Math.sin(angle)];
    const slices = rows.map((row, index) => { const share = Number(row.percentage || 0); const end = offset + share * Math.PI * 2; const startPoint = point(offset), endPoint = point(end); const largeArc = share > .5 ? 1 : 0; const markup = `<path class="segments-pie-slice" d="M 100 100 L ${startPoint[0].toFixed(2)} ${startPoint[1].toFixed(2)} A 94 94 0 ${largeArc} 1 ${endPoint[0].toFixed(2)} ${endPoint[1].toFixed(2)} Z" fill="${colorFor(row.segment_name, index)}"><title>${esc(row.segment_name)}: ${esc(row.percentage_display)}</title></path>`; offset = end; return markup; }).join("");
    const legend = rows.map((row, index) => `<div class="segments-donut-legend-row"><i style="background:${colorFor(row.segment_name, index)}"></i><span><strong>${esc(row.segment_name)}</strong><small>${esc(row.customer_count)} customers</small></span><b>${esc(row.percentage_display)}</b></div>`).join("");
    const total = rows.reduce((sum, row) => sum + (Number(row.customer_count) || 0), 0);
    container.innerHTML = `<div class="segments-distribution-layout"><div class="segments-pie-visual"><svg viewBox="0 0 200 200" role="img" aria-label="${population} segment distribution">${slices}</svg></div><div class="segments-donut-legend">${legend}</div></div>`;
  }

  function numberFrom(text, key) { const match = String(text || "").match(new RegExp(`${key}[^0-9+-]*([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\\.[0-9]+)?)`, "i")); return match ? Number(match[1].replaceAll(",", "")) : null; }
  function renderComparison(population) {
    const container = document.querySelector(`[data-comparison-chart="${population}"]`);
    const segments = workspace[population]?.segments || [];
    if (!container || !segments.length) return;
    const dimensions = [
      ["Monetary", (s) => Number(s.clustering_features?.Monetary)],
      ["Frequency", (s) => Number(s.clustering_features?.Frequency)],
      ["Average balance", (s) => Number(s.clustering_features?.avg_balance)],
      ["Net cash flow", (s) => Number(s.clustering_features?.net_cash_flow)],
      ["Channel entropy", (s) => Number(s.clustering_features?.channel_entropy)],
    ];
    const values = dimensions.map(([, getter]) => segments.map(getter));
    const normalise = (valuesForDimension, value) => { const usable = valuesForDimension.filter((item) => item !== null && Number.isFinite(item)); if (value === null || !usable.length) return null; const min = Math.min(...usable), max = Math.max(...usable); return max === min ? 50 : ((value - min) / (max - min)) * 100; };
    const polar = (angle, radius) => [50 + radius * Math.cos(angle), 52 + radius * Math.sin(angle)];
    const cards = segments.map((segment, rowIndex) => {
      const color = colorFor(segment.display_name, rowIndex);
      const scores = dimensions.map(([, getter], dimensionIndex) => normalise(values[dimensionIndex], getter(segment)));
      const points = scores.map((score, index) => { const angle = -Math.PI / 2 + index * (Math.PI * 2 / dimensions.length); return polar(angle, 36 * ((score === null ? 0 : score) / 100)); });
      const grid = [25, 50, 75, 100].map((level) => { const ring = dimensions.map(([,], index) => { const angle = -Math.PI / 2 + index * (Math.PI * 2 / dimensions.length); return polar(angle, 36 * level / 100).join(","); }).join(" "); return `<polygon class="segments-radar-grid" points="${ring}"></polygon>`; }).join("");
      const axisLabels = ["Monetary", "Frequency", "Average balance", "Net cash flow", "Channel entropy"];
      const axes = dimensions.map(([,], index) => { const angle = -Math.PI / 2 + index * (Math.PI * 2 / dimensions.length); const [x, y] = polar(angle, 45); return `<line class="segments-radar-axis" x1="50" y1="52" x2="${x}" y2="${y}"></line><text class="segments-radar-label" x="${x}" y="${y}" text-anchor="middle">${esc(axisLabels[index])}</text>`; }).join("");
      const valuesText = scores.map((score) => score === null ? "n/a" : `${score.toFixed(0)}`).join(" · ");
      const markers = points.map((point, index) => scores[index] === null ? "" : `<circle class="segments-radar-point" cx="${point[0]}" cy="${point[1]}" r="1.5" style="--radar-color:${color}"><title>${esc(segment.display_name)} — ${esc(axisLabels[index])}: ${scores[index].toFixed(0)}</title></circle>`).join("");
      const polygon = scores.every((score) => score !== null) ? `<polygon class="segments-radar-shape" points="${points.map((point) => point.join(",")).join(" ")}" style="--radar-color:${color}"></polygon>` : "";
      return `<article class="segments-radar-card"><h3><i style="background:${color}"></i>${esc(segment.display_name)}</h3><svg viewBox="0 0 100 104" role="img" aria-label="${esc(segment.display_name)} behavioral radar">${grid}${axes}${polygon}${markers}</svg></article>`;
    }).join("");
    container.innerHTML = `<div class="segments-radar-grid-wrap">${cards}</div><p class="segments-comparison-note">Scores are normalized from 0–100 within the selected population for comparison; they are not absolute performance scores.</p>`;
  }

  function forecastChart(forecast) {
    const points = (forecast?.points || []).filter((point) => point.actual !== null || point.forecast !== null);
    if (!points.length) return '<p class="empty-state__body">No forecast chart is available for this segment.</p>';
    const width = 920, height = 380, pad = { top: 28, right: 24, bottom: 48, left: 82 };
    const values = points.flatMap((point) => [point.actual, point.forecast]).filter((value) => Number.isFinite(Number(value))).map(Number);
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
    const x = (index) => pad.left + index / Math.max(1, points.length - 1) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (max - value) / span * (height - pad.top - pad.bottom);
    const path = (key) => { let open = false; return points.map((point, index) => { const value = Number(point[key]); if (!Number.isFinite(value)) { open = false; return ""; } const command = `${open ? "L" : "M"} ${x(index).toFixed(1)} ${y(value).toFixed(1)}`; open = true; return command; }).filter(Boolean).join(" "); };
    const forecastIndex = points.findIndex((point) => Number.isFinite(Number(point.forecast)));
    const pointMarkup = points.flatMap((point, index) => ["actual", "forecast"].map((key) => Number.isFinite(Number(point[key])) ? `<circle class="segments-chart-point segments-chart-point--${key}" cx="${x(index)}" cy="${y(Number(point[key]))}" r="4" data-date="${esc(point.date)}" data-series="${key}" data-value="${Number(point[key])}"></circle>` : "")).join("");
    return `<div class="segments-chart-legend"><span><i class="segments-chart-swatch segments-chart-swatch--actual"></i>Observed average balance</span><span><i class="segments-chart-swatch segments-chart-swatch--forecast"></i>Forecast average balance</span></div><div class="segments-chart-tooltip" role="status" aria-live="polite" hidden></div><svg class="segments-forecast-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Average balance forecast"><line class="segments-chart-grid" x1="${pad.left}" x2="${width-pad.right}" y1="${y(min)}" y2="${y(min)}"></line><line class="segments-chart-grid" x1="${pad.left}" x2="${width-pad.right}" y1="${y(max)}" y2="${y(max)}"></line>${forecastIndex >= 0 ? `<line class="segments-chart-boundary" x1="${x(forecastIndex)}" x2="${x(forecastIndex)}" y1="${pad.top}" y2="${height-pad.bottom}"></line>` : ""}<path class="segments-chart-line segments-chart-line--actual" d="${path("actual")}"></path><path class="segments-chart-line segments-chart-line--forecast" d="${path("forecast")}"></path>${pointMarkup}<text class="segments-chart-axis" x="${pad.left}" y="${height-8}">${esc(formatDate(points[0].date))}</text><text class="segments-chart-axis" text-anchor="end" x="${width-pad.right}" y="${height-8}">${esc(formatDate(points[points.length-1].date))}</text><text class="segments-chart-axis" x="8" y="${y(max)+4}">${format(max)}</text><text class="segments-chart-axis" x="8" y="${y(min)+4}">${format(min)}</text></svg>`;
  }

  const technicalFeatureLabels = { Monetary: "Monetary value", Frequency: "Transaction frequency", avg_balance: "Average balance", net_cash_flow: "Net cash flow", positive_cashflow_week_share: "Positive cash-flow weeks", low_balance_week_share: "Low-balance weeks", channel_entropy: "Channel diversity", digital_tx_pct: "Digital transactions", atm_tx_pct: "ATM transactions", branch_tx_pct: "Branch transactions" };
  const technicalFeatureOrder = ["Frequency", "Monetary", "avg_balance", "net_cash_flow", "positive_cashflow_week_share", "low_balance_week_share", "channel_entropy", "digital_tx_pct", "atm_tx_pct", "branch_tx_pct"];
  const populationLabel = (population) => String(population || "").replace(/^[a-z]/, (character) => character.toUpperCase());
  const technicalFeatureMarkup = (segment) => {
    const features = { ...(segment.clustering_features || {}) };
    const sourceText = [segment.transaction_behavior, segment.balance_and_cashflow_behavior, segment.digital_channel_behavior].filter(Boolean).join("; ");
    technicalFeatureOrder.forEach((key) => {
      if (features[key] !== undefined && features[key] !== null) return;
      const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const match = sourceText.match(new RegExp(`\\b${escapedKey}\\b\\s*[:=]?\\s*([+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)`, "i"));
      if (match) features[key] = Number(match[1].replaceAll(",", ""));
    });
    return technicalFeatureOrder.filter((key) => Number.isFinite(Number(features[key]))).map((key) => {
      const raw = Number(features[key]);
      const display = /(?:share|_pct)$/i.test(key) ? `${(raw * 100).toFixed(0)}%` : new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(raw);
      const context = sourceText.match(new RegExp(`[^.;]*\\b${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b[^.;]*`, "i"))?.[0] || "Stored clustering feature";
      const semantic = /highest|most positive/i.test(context) ? "technical-feature--positive" : /lowest|most negative/i.test(context) ? "technical-feature--attention" : "";
      return `<article class="technical-feature ${semantic}" title="${esc(key)} — raw value: ${esc(raw)}"><span>${esc(technicalFeatureLabels[key] || key.replaceAll("_", " "))}</span><strong>${esc(display)}</strong><small>${esc(key)}</small></article>`;
    }).join("") || '<p class="empty-state__body">No raw clustering feature values are available.</p>';
  };
  const technicalDetailsMarkup = (segment, population) => {
    const details = segment.technical_details || {};
    const confidence = String(details.confidence || "Not recorded");
    const behavior = confidence.match(/(?:behavior|behaviour)[^A-Za-z]*(medium|high|low)/i)?.[1] || confidence.match(/^(medium|high|low)/i)?.[1] || "Not recorded";
    const magnitude = confidence.match(/magnitude[^A-Za-z]*(medium|high|low)/i)?.[1] || "Not recorded";
    return `<details class="segments-technical"><summary>Technical segment details</summary><div class="segments-technical-body"><div class="segments-technical-meta"><div><dt>Internal segment ID</dt><dd>${esc(details.segment_id || "Not recorded")}</dd></div><div><dt>Source cluster</dt><dd>${esc(details.source_name || "Not recorded")}<small>${esc(populationLabel(population))} customers · Cluster #${esc(details.cluster_id ?? "not recorded")}</small></dd></div><div><dt>Interpretation confidence</dt><dd><span>Behavior interpretation <b class="technical-confidence technical-confidence--${esc(behavior.toLowerCase())}">${esc(behavior.toUpperCase())}</b></span><span>Magnitude interpretation <b class="technical-confidence technical-confidence--${esc(magnitude.toLowerCase())}">${esc(magnitude.toUpperCase())}</b></span></dd></div></div><section class="segments-raw-features"><h4>Raw behavioral features</h4><div class="technical-feature-grid">${technicalFeatureMarkup(segment)}</div></section></div></details>`;
  };
  const humanizeTechnicalName = (value) => ({ ets_damped: "Damped ETS", trend_seasonal: "Trend + seasonal" }[String(value || "")] || String(value || "Not recorded"));
  const forecastTechnicalMarkup = (forecast) => `<section class="segments-model-block segments-model-block--forecast"><h4><span aria-hidden="true">↗</span>Forecast model</h4><div class="segments-model-fields"><div><dt>Forecasting method</dt><dd>${esc(humanizeTechnicalName(forecast?.model))}<small>${esc(forecast?.model || "Not recorded")}</small></dd></div><div><dt>Scenario</dt><dd>${esc(humanizeTechnicalName(forecast?.scenario))}<small>${esc(forecast?.scenario || "Not recorded")}</small></dd></div><div><dt>Forecast uncertainty</dt><dd><b class="technical-uncertainty">POINT ESTIMATE ONLY</b><small>No stored confidence interval</small></dd></div></div></section>`;

  function bindForecastTooltips() {
    document.querySelectorAll(".segments-selected-forecast").forEach((section) => {
      const chart = section.querySelector(".segments-forecast-chart");
      const tooltip = section.querySelector(".segments-chart-tooltip");
      if (!chart || !tooltip) return;
      chart.querySelectorAll(".segments-chart-point").forEach((point) => {
        point.addEventListener("mouseenter", (event) => {
          const date = point.dataset.date;
          const sameDate = Array.from(chart.querySelectorAll(".segments-chart-point")).filter((item) => item.dataset.date === date);
          const rows = sameDate.map((item) => `<span><i class="segments-chart-tooltip__marker segments-chart-tooltip__marker--${item.dataset.series}"></i>${item.dataset.series === "actual" ? "Observed average" : "Forecast average"}<b>${format(Number(item.dataset.value))}</b></span>`).join("");
          const parsed = new Date(`${date}T00:00:00`);
          const dateLabel = Number.isNaN(parsed.getTime()) ? date : new Intl.DateTimeFormat(undefined, { day: "2-digit", month: "short", year: "numeric" }).format(parsed);
          tooltip.innerHTML = `<strong>${esc(dateLabel)}</strong>${rows}`;
          tooltip.hidden = false;
          const sectionRect = section.getBoundingClientRect();
          tooltip.style.left = `${Math.max(8, Math.min(section.clientWidth - 218, event.clientX - sectionRect.left + 12))}px`;
          tooltip.style.top = `${Math.max(8, event.clientY - sectionRect.top - 92)}px`;
        });
        point.addEventListener("mouseleave", () => { tooltip.hidden = true; });
      });
    });
  }

  function renderSelected(population, id) {
    const target = document.querySelector(`[data-selected-segment="${population}"]`);
    const segment = (workspace[population]?.segments || []).find((item) => item.segment_id === id) || workspace[population]?.segments?.[0];
    if (!target || !segment) { if (target) target.innerHTML = '<p class="empty-state__body">No segment profile is available.</p>'; return; }
    const forecast = segment.forecast || {};
    const products = segment.products || [];
    const forecastMessage = forecast.status || "Forecast unavailable for this segment; no static forecast output is available.";
    const signals = (items) => items?.length ? `<ul class="segments-signal-list">${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : '<p class="empty-state__body">Not available</p>';
    target.innerHTML = `<article class="segments-selected-profile"><div class="segments-profile-intro"><div><span class="segments-eyebrow">${esc(population)} segment</span><h3>${esc(segment.display_name || segment.segment_name)}</h3><p class="segments-profile-size"><strong>${esc(segment.customer_count)}</strong> customers · <strong>${esc(segment.percentage_display)}</strong> of classified ${esc(population)} customers</p><p>${esc(segment.description || "A behavioral group described from observed portfolio statistics.")}</p></div><div class="segments-profile-share">${esc(segment.percentage_display)}<small>portfolio share</small></div></div><div class="segments-detail-grid"><section><h4>Behavior vs peers</h4><dl class="segments-facts"><div><dt>Transaction activity</dt><dd>${esc(segment.transaction_behavior || "Not available")}</dd></div><div><dt>Balance &amp; cash flow</dt><dd>${esc(segment.balance_and_cashflow_behavior || "Not available")}</dd></div><div><dt>Digital / channel behavior</dt><dd>${esc(segment.digital_channel_behavior || "Not available")}</dd></div></dl></section><section><h4>Products held by this group</h4><p class="text-muted">Highest observed ownership values for this segment; these are descriptive, not causal.</p>${products.length ? `<ul class="segments-product-list">${products.map((item) => `<li><span>${esc(item.product_name)}</span><strong>${esc(item.penetration)}</strong></li>`).join("")}</ul>` : '<p class="empty-state__body">No segment ownership data is available.</p>'}</section></div><div class="segments-detail-grid segments-detail-grid--signals"><section><h4>Typical strengths</h4>${signals(segment.strengths)}</section><section><h4>Things to watch</h4>${signals(segment.watch_outs)}</section></div><section class="segments-selected-forecast"><div class="segments-section-heading"><div><h4>Segment Balance Outlook</h4><p class="text-muted">This estimates how the average balance of customers in this static behavioral segment may evolve over the forecast horizon.</p></div>${forecast.available ? `<div class="segments-forecast-summary"><span><small>Current</small>${esc(forecast.current_average)}</span><span><small>Forecast end</small>${esc(forecast.forecast_end)}</span><span><small>Expected change</small>${esc(forecast.change)} · ${esc(forecast.direction)}</span></div>` : ""}</div>${forecast.available ? forecastChart(forecast) : `<p class="empty-state__body">${esc(forecastMessage)}</p>`}</section><details class="segments-technical"><summary>Technical segment details</summary><dl><div><dt>Internal segment ID</dt><dd>${esc(segment.technical_details?.segment_id)}</dd></div><div><dt>Source name / cluster</dt><dd>${esc(segment.technical_details?.source_name)} / ${esc(segment.technical_details?.cluster_id)}</dd></div><div><dt>Interpretation confidence</dt><dd>${esc(segment.technical_details?.confidence)}</dd></div></dl></details>${forecast.available ? `<details class="segments-technical"><summary>Forecast methodology &amp; accuracy</summary><dl><div><dt>Stored forecasting model</dt><dd>${esc(forecast.model || "Not recorded")}</dd></div><div><dt>Scenario</dt><dd>${esc(forecast.scenario || "Not recorded")}</dd></div><div><dt>Point-estimate limitation</dt><dd>Confidence intervals are not available in the stored output.</dd></div></dl></details>` : ""}</article>`;
  }

  function enhanceSelectedProfile(population, segment, target) {
    if (!target || !segment) return;
    const intro = target.querySelector(".segments-profile-intro");
    if (intro) {
      const copy = intro.querySelector(".segments-profile-intro > div");
      const description = copy?.querySelector("p:last-child");
      if (description) description.remove();
      intro.querySelector(".segments-profile-share")?.remove();
    }
    const sections = target.querySelectorAll(".segments-detail-grid > section");
    const peers = workspace[population]?.segments || [];
    const valueFor = (item, kind) => {
      if (kind === "frequency") return Number(item.clustering_features?.Frequency);
      if (kind === "low_balance") return numberFrom(item.balance_and_cashflow_behavior, "low_balance_week_share");
      if (kind === "cash_flow") return numberFrom(item.balance_and_cashflow_behavior, "net_cash_flow");
      return numberFrom(item.digital_channel_behavior, "digital_tx_pct");
    };
    const insight = (kind, label, reverse = false) => {
      const value = valueFor(segment, kind);
      const all = peers.map((item) => valueFor(item, kind)).filter(Number.isFinite);
      if (!Number.isFinite(value) || !all.length) return `<li><strong>→ ${label}</strong><small>Not available</small></li>`;
      const rank = all.length === 1 ? .5 : (value - Math.min(...all)) / (Math.max(...all) - Math.min(...all) || 1);
      const level = rank >= .67 ? "high" : rank <= .33 ? "low" : "mid";
      const effective = reverse ? (level === "high" ? "low" : level === "low" ? "high" : "mid") : level;
      const arrow = effective === "high" ? "↑" : effective === "low" ? "↓" : "→";
      const wording = effective === "high" ? `Very high ${label}` : effective === "low" ? `Low ${label}` : `Moderate ${label}`;
      const secondary = effective === "high" ? "Highest among groups" : effective === "low" ? "Below peers" : "Mid-range among groups";
      return `<li><strong>${arrow} ${wording}</strong><small>${secondary}</small></li>`;
    };
    const peerScale = (kind, label, reverse, color) => {
      const value = valueFor(segment, kind);
      const all = peers.map((item) => valueFor(item, kind)).filter(Number.isFinite);
      if (!Number.isFinite(value) || !all.length) return `<li class="peer-scale peer-scale--unavailable" style="--peer-color:${color}"><div class="peer-scale__heading"><strong>${label}</strong><b>NOT AVAILABLE</b></div><div class="peer-scale__track"></div><small>Peer position unavailable</small></li>`;
      const rank = all.length === 1 ? .5 : (value - Math.min(...all)) / (Math.max(...all) - Math.min(...all) || 1);
      const effectiveRank = reverse ? 1 - rank : rank;
      const level = effectiveRank >= .67 ? "VERY HIGH" : effectiveRank <= .33 ? "LOW" : "MODERATE";
      const interpretation = effectiveRank >= .67 ? "Highest among groups" : effectiveRank <= .33 ? "Below peers" : "Mid-range among groups";
      return `<li class="peer-scale" style="--peer-position:${Math.max(0, Math.min(1, effectiveRank))};--peer-color:${color}"><div class="peer-scale__heading"><strong>${label}</strong><b>${level}</b></div><div class="peer-scale__track"><span></span><i></i></div><small>${interpretation}</small></li>`;
    };
    if (sections[0]) sections[0].innerHTML = `<h4>How this group differs</h4><div class="peer-scale-orientation"><span>LOW / BELOW PEERS</span><span>HIGH / ABOVE PEERS</span></div><ul class="segments-difference-list">${peerScale("frequency", "Transaction activity", false, "#50AECE")}${peerScale("low_balance", "Balance stability", true, "#9067C6")}${peerScale("cash_flow", "Cash-flow position", false, "#F05212")}${peerScale("digital", "Digital usage", false, "#FECD04")}</ul>`;
    if (sections[1]) {
      const heading = sections[1].querySelector("h4");
      if (heading) heading.textContent = "Most common products";
      const note = sections[1].querySelector("p.text-muted");
      if (note) note.textContent = "Share of customers in this segment who hold each product.";
      const list = sections[1].querySelector(".segments-product-list");
      const products = (segment.products || []).slice(0, 3);
      if (list && products.length) list.innerHTML = products.map((item, index) => `<li style="--product-color:${atlasChartPalette[index % atlasChartPalette.length]}"><div><span>${esc(item.product_name)}</span><div class="bar-track"><span style="width:${Math.max(0, Math.min(100, Number(item.penetration_value || 0) * 100)).toFixed(1)}%"></span></div></div><strong>${esc(item.penetration)}</strong></li>`).join("");
    }
    const forecast = segment.forecast || {};
    const forecastSection = target.querySelector(".segments-selected-forecast");
    forecastSection?.classList.add("balance-outlook");
    target.querySelector(".segments-forecast-summary")?.classList.add("balance-outlook__metrics");
    const technical = target.querySelector(".segments-technical");
    if (technical) {
      const fragment = document.createRange().createContextualFragment(technicalDetailsMarkup(segment, population));
      const combined = fragment.firstElementChild;
      combined.querySelector("summary").textContent = "Model & technical information";
      combined.querySelector(".segments-raw-features")?.remove();
      const body = combined.querySelector(".segments-technical-body");
      const meta = body?.querySelector(".segments-technical-meta");
      if (body && meta) {
        const segmentModel = document.createElement("section");
        segmentModel.className = "segments-model-block segments-model-block--segment";
        segmentModel.innerHTML = '<h4><span aria-hidden="true">◇</span>Segment model</h4>';
        body.insertBefore(segmentModel, meta);
        segmentModel.appendChild(meta);
      }
      body?.insertAdjacentHTML("beforeend", forecastTechnicalMarkup(forecast));
      technical.replaceWith(combined);
      target.querySelectorAll(".segments-technical")[1]?.remove();
    }
    const summary = target.querySelector(".segments-forecast-summary");
    if (summary && forecast.available) {
      const last = (forecast.points || []).at(-1) || {};
      const current = Number(last.actual), end = Number(last.forecast);
      const compact = (value) => { if (!Number.isFinite(value)) return "n/a"; if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(2)}M`; if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)}K`; return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value); };
      const direction = String(forecast.direction || "Stable").toLowerCase();
      summary.innerHTML = `<span title="Exact value: ${esc(forecast.current_average)}"><small>Current average</small><b>${compact(current)}</b></span><span title="Exact value: ${esc(forecast.forecast_end)}"><small>Forecast average</small><b>${compact(end)}</b></span><span class="segments-summary-change segments-summary-change--${direction}" title="Exact change: ${esc(forecast.change)}"><small>Expected change</small><b>${direction.includes("increas") ? "↑" : direction.includes("decreas") ? "↓" : "→"} ${esc(forecast.change)}</b><em>${esc(forecast.direction)}</em></span>`;
    }
    if (forecastSection && forecast.available && !forecastSection.querySelector(".segments-forecast-interpretation")) {
      const lastPoint = (forecast.points || []).at(-1) || {};
      const current = Number(lastPoint.actual), end = Number(lastPoint.forecast);
      let text = "Average balance is expected to remain broadly stable.";
      if (Number.isFinite(current) && Number.isFinite(end)) text = end > current ? (end < 0 ? "Average balance is expected to improve, although it remains negative." : "Average balance is expected to improve over the forecast horizon.") : end < current ? (end < 0 ? "Average balance is expected to decline and remains negative." : "Average balance is expected to decline over the forecast horizon.") : text;
      const note = document.createElement("p"); note.className = "segments-forecast-interpretation"; note.textContent = text;
      forecastSection.querySelector(".segments-forecast-chart")?.before(note);
    }
  }

  ["retail", "enterprise"].forEach((population) => {
    renderDistribution(population);
    renderComparison(population);
    const picker = document.querySelector(`[data-segment-picker="${population}"]`);
    if (picker) { const draw = () => { renderSelected(population, picker.value); const target = document.querySelector(`[data-selected-segment="${population}"]`); const segment = (workspace[population]?.segments || []).find((item) => item.segment_id === picker.value) || workspace[population]?.segments?.[0]; enhanceSelectedProfile(population, segment, target); bindForecastTooltips(); }; picker.addEventListener("change", draw); draw(); }
  });
  const setPopulation = (population) => {
      document.querySelectorAll(".segment-population-switcher [data-population]").forEach((item) => { const active = item.dataset.population === population; item.classList.toggle("is-active", active); item.setAttribute("aria-selected", active ? "true" : "false"); });
    document.querySelectorAll("[data-segment-panel]").forEach((panel) => {
      const active = panel.dataset.segmentPanel === population;
      panel.hidden = !active;
      panel.style.display = active ? "grid" : "none";
    });
  };
  document.querySelectorAll(".segment-population-switcher [data-population]").forEach((button) => button.addEventListener("click", () => setPopulation(button.dataset.population)));
  setPopulation("retail");
})();
