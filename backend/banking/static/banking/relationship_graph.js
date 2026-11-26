(function () {
  "use strict";

  const COLORS = {
    Product: "#71717a",
    Package: "#c2410c",
    Account: "#0369a1",
    Card: "#7c3aed",
    FinancingSolution: "#b45309",
    DigitalService: "#0891b2",
    Investment: "#15803d",
    Insurance: "#be123c",
    Service: "#4338ca",
    ProductFamily: "#525252",
    CustomerNeed: "#ca8a04",
    BehavioralSegment: "#065f46",
    EligibilityRule: "#166534",
    OwnershipConstraint: "#7c2d12",
    EvidenceSource: "#334155",
    Customer: "#111827",
    OwnedProduct: "#2563eb",
    RecommendedProduct: "#be123c",
    OwnedAndRecommendedProduct: "#7c3aed",
    Unknown: "#9ca3af",
  };

  const PRIORITY = [
    "Customer",
    "BehavioralSegment",
    "CustomerNeed",
    "OwnedProduct",
    "RecommendedProduct",
    "OwnedAndRecommendedProduct",
    "Package",
    "Account",
    "Card",
    "FinancingSolution",
    "DigitalService",
    "Investment",
    "Insurance",
    "Service",
    "ProductFamily",
    "EligibilityRule",
    "OwnershipConstraint",
    "EvidenceSource",
    "Product",
  ];

  function escape(value) {
    return String(value || "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[ch]);
  }

  function label(value, size) {
    const text = String(value || "");
    return text.length > size ? text.slice(0, size - 3) + "..." : text;
  }

  function normalizeSearch(value) {
    return String(value || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/gi, " ")
      .trim()
      .toLowerCase();
  }

  function isSubsequence(needle, haystack) {
    if (!needle) return true;
    let index = 0;
    for (const char of haystack) {
      if (char === needle[index]) index += 1;
      if (index === needle.length) return true;
    }
    return false;
  }

  function looseWordMatch(token, words) {
    if (token.length < 4) return words.some((word) => word.includes(token));
    return words.some((word) => {
      if (word.includes(token) || isSubsequence(token, word)) return true;
      const shared = [...new Set(token)].filter((char) => word.includes(char)).length;
      return word.startsWith(token.slice(0, 2)) && shared >= Math.min(4, token.length - 1);
    });
  }

  function nodeMatchesQuery(node, query) {
    const properties = Object.entries(node.properties || {})
      .flatMap(([key, value]) => [key, value])
      .join(" ");
    const searchable = normalizeSearch(`${node.label} ${node.id} ${node.group} ${node.description || ""} ${properties}`);
    const compact = searchable.replace(/\s+/g, "");
    const compactQuery = query.replace(/\s+/g, "");
    if (searchable.includes(query) || compact.includes(compactQuery)) return true;
    const words = searchable.split(/\s+/);
    return query.split(/\s+/).every((token) => looseWordMatch(token, words));
  }

  function relationshipLabel(value, dir) {
    const phrase = String(value || "relatedTo")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replace(/_/g, " ")
      .toLowerCase();
    const readable = {
      "addresses need": dir === "out" ? "addresses customer need" : "is addressed by",
      "belongs to family": dir === "out" ? "belongs to product family" : "contains product",
      "has eligibility rule": dir === "out" ? "has eligibility rule" : "eligibility applies to",
      "has ownership constraint": dir === "out" ? "has ownership constraint" : "constraint applies to",
      "supported by evidence": dir === "out" ? "supported by evidence" : "supports",
      "targets segment": dir === "out" ? "targets behavioral segment" : "is targeted by",
    };
    return readable[phrase] || (dir === "out" ? phrase : `linked by ${phrase}`);
  }

  function sortGroups(groups) {
    return groups.sort((a, b) => {
      const ia = PRIORITY.indexOf(a);
      const ib = PRIORITY.indexOf(b);
      if (ia === -1 && ib === -1) return a.localeCompare(b);
      if (ia === -1) return 1;
      if (ib === -1) return -1;
      return ia - ib;
    });
  }

  class RelationshipGraph {
    constructor(container, graph) {
      this.container = container;
      this.allNodes = graph.nodes || [];
      this.allEdges = graph.edges || [];
      this.nodeById = Object.fromEntries(this.allNodes.map((node) => [node.id, node]));
      this.activeGroups = new Set(this.allNodes.map((node) => node.group || "Unknown"));
      this.depth = 1;
      this.focusedId = null;
      this.selectedId = null;
      this.scale = 1;
      this.tx = 0;
      this.ty = 0;
      this.drag = null;
      this.nodeDrag = null;
      this.positions = new Map();
      this.buildAdjacency();
      this.mount();
      this.layout();
      this.showAll();
    }

    buildAdjacency() {
      this.adj = {};
      this.allEdges.forEach((edge) => {
        (this.adj[edge.from] ||= []).push({ to: edge.to, label: edge.label, dir: "out" });
        (this.adj[edge.to] ||= []).push({ to: edge.from, label: edge.label, dir: "in" });
      });
    }

    mount() {
      this.container.innerHTML = `
        <aside class="graph-sidebar">
          <input class="graph-search" type="search" placeholder="Search product, need, rule..." autocomplete="off">
          <div class="graph-search-examples" aria-label="Search examples"></div>
          <div class="graph-results"></div>
          <div class="graph-control-row">
            <button type="button" class="graph-btn is-active" data-depth="1">1 hop</button>
            <button type="button" class="graph-btn" data-depth="2">2 hops</button>
            <button type="button" class="graph-btn" data-reset>Show all</button>
          </div>
          <div class="graph-section-label">Classes</div>
          <div class="graph-legend"></div>
          <div class="graph-detail">
            <strong>Ontology graph</strong>
            <small>${this.allNodes.length.toLocaleString()} nodes, ${this.allEdges.length.toLocaleString()} edges</small>
            <p>Hover for a preview. Click a product or related node for details.</p>
          </div>
        </aside>
        <section class="graph-canvas">
          <svg class="relationship-graph__svg" role="img" aria-label="Relationship graph">
            <g class="graph-viewport">
              <g class="graph-edges"></g>
              <g class="graph-edge-labels"></g>
              <g class="graph-nodes"></g>
            </g>
          </svg>
          <div class="graph-tooltip" hidden></div>
          <div class="graph-hint">Drag to pan. Scroll to zoom. Click a node to focus.</div>
        </section>
      `;
      this.searchEl = this.container.querySelector(".graph-search");
      this.resultsEl = this.container.querySelector(".graph-results");
      this.legendEl = this.container.querySelector(".graph-legend");
      this.detailEl = this.container.querySelector(".graph-detail");
      this.svg = this.container.querySelector("svg");
      this.viewport = this.container.querySelector(".graph-viewport");
      this.edgeLayer = this.container.querySelector(".graph-edges");
      this.edgeLabelLayer = this.container.querySelector(".graph-edge-labels");
      this.nodeLayer = this.container.querySelector(".graph-nodes");
      this.tooltipEl = this.container.querySelector(".graph-tooltip");
      this.examplesEl = this.container.querySelector(".graph-search-examples");
      this.renderSearchExamples();
      this.renderSearch();

      this.searchEl.addEventListener("input", () => this.renderSearch());
      this.searchEl.addEventListener("keyup", () => this.renderSearch());
      this.searchEl.addEventListener("search", () => this.renderSearch());
      this.searchEl.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const first = this.resultsEl.querySelector("[data-node-id]");
        if (first) {
          event.preventDefault();
          this.focusNode(first.dataset.nodeId);
        }
      });
      this.resultsEl.addEventListener("click", (event) => {
        const result = event.target.closest("[data-node-id]");
        if (result) this.focusNode(result.dataset.nodeId);
      });
      this.examplesEl.addEventListener("click", (event) => {
        const example = event.target.closest("[data-search-example]");
        if (!example) return;
        this.searchEl.value = example.dataset.searchExample;
        this.renderSearch();
        this.searchEl.focus();
      });
      this.container.querySelectorAll("[data-depth]").forEach((button) => {
        button.addEventListener("click", () => {
          this.container.querySelectorAll("[data-depth]").forEach((item) => item.classList.remove("is-active"));
          button.classList.add("is-active");
          this.depth = Number(button.dataset.depth);
          if (this.focusedId) this.focusNode(this.focusedId);
        });
      });
      this.container.querySelector("[data-reset]").addEventListener("click", () => this.showAll());
      this.bindPanZoom();
      this.renderLegend();
    }

    bindPanZoom() {
      this.svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = this.svg.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        const old = this.scale;
        const factor = event.deltaY < 0 ? 1.12 : 0.89;
        this.scale = Math.max(0.12, Math.min(5, this.scale * factor));
        this.tx = mx - (mx - this.tx) * (this.scale / old);
        this.ty = my - (my - this.ty) * (this.scale / old);
        this.applyTransform();
      }, { passive: false });

      this.svg.addEventListener("pointerdown", (event) => {
        if (event.target.closest(".graph-node")) return;
        this.drag = { x: event.clientX, y: event.clientY, tx: this.tx, ty: this.ty };
        this.svg.setPointerCapture(event.pointerId);
      });
      this.svg.addEventListener("pointermove", (event) => {
        if (this.nodeDrag) {
          const dx = (event.clientX - this.nodeDrag.x) / this.scale;
          const dy = (event.clientY - this.nodeDrag.y) / this.scale;
          this.positions.set(this.nodeDrag.id, { x: this.nodeDrag.px + dx, y: this.nodeDrag.py + dy });
          this.render();
          return;
        }
        if (!this.drag) return;
        this.tx = this.drag.tx + event.clientX - this.drag.x;
        this.ty = this.drag.ty + event.clientY - this.drag.y;
        this.applyTransform();
      });
      this.svg.addEventListener("pointerup", () => {
        this.drag = null;
        this.nodeDrag = null;
      });
      this.svg.addEventListener("pointercancel", () => {
        this.drag = null;
        this.nodeDrag = null;
      });
    }

    layout() {
      const canvas = this.container.querySelector(".graph-canvas");
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(rect.width || 900, 900);
      const height = Math.max(rect.height || 620, 620);
      this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const groups = sortGroups([...new Set(this.allNodes.map((node) => node.group || "Unknown"))]);
      const cx = width / 2;
      const cy = height / 2;
      const groupRadius = Math.min(width, height) * 0.32;
      groups.forEach((group, groupIndex) => {
        const groupNodes = this.allNodes.filter((node) => (node.group || "Unknown") === group);
        const ga = (2 * Math.PI * groupIndex) / groups.length - Math.PI / 2;
        const gcx = cx + Math.cos(ga) * groupRadius;
        const gcy = cy + Math.sin(ga) * groupRadius;
        const radius = Math.max(28, Math.min(125, 16 + groupNodes.length * 3.2));
        groupNodes.forEach((node, index) => {
          const ring = Math.floor(Math.sqrt(index));
          const start = ring * ring;
          const count = Math.max(1, (ring + 1) * (ring + 1) - start);
          const angle = 2 * Math.PI * ((index - start) / count) + groupIndex * 0.31;
          const rr = ((ring + 1) / (Math.ceil(Math.sqrt(groupNodes.length)) + 1)) * radius;
          this.positions.set(node.id, { x: gcx + Math.cos(angle) * rr, y: gcy + Math.sin(angle) * rr });
        });
      });
    }

    visibleNodeIds() {
      if (!this.focusedId) {
        return new Set(this.allNodes.filter((node) => this.activeGroups.has(node.group || "Unknown")).map((node) => node.id));
      }
      const seen = new Set([this.focusedId]);
      let frontier = [this.focusedId];
      for (let step = 0; step < this.depth; step += 1) {
        const next = [];
        frontier.forEach((id) => {
          (this.adj[id] || []).forEach((edge) => {
            if (!seen.has(edge.to)) {
              seen.add(edge.to);
              next.push(edge.to);
            }
          });
        });
        frontier = next;
      }
      return seen;
    }

    render() {
      const visible = this.visibleNodeIds();
      const showLabels = this.container.dataset.showLabels === "true" || visible.size <= 55;
      const showEdgeLabels = this.container.dataset.showEdgeLabels === "true" || visible.size <= 45;
      this.edgeLayer.innerHTML = "";
      this.edgeLabelLayer.innerHTML = "";
      this.nodeLayer.innerHTML = "";

      this.allEdges.forEach((edge) => {
        if (!visible.has(edge.from) || !visible.has(edge.to)) return;
        const a = this.positions.get(edge.from);
        const b = this.positions.get(edge.to);
        if (!a || !b) return;
        const related = this.selectedId && (edge.from === this.selectedId || edge.to === this.selectedId) ? " is-related" : "";
        this.edgeLayer.insertAdjacentHTML("beforeend", `<line class="graph-edge${related}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" />`);
        if (showEdgeLabels) {
          this.edgeLabelLayer.insertAdjacentHTML(
            "beforeend",
            `<text class="graph-edge-label" x="${(a.x + b.x) / 2}" y="${(a.y + b.y) / 2 - 3}">${escape(edge.label)}</text>`
          );
        }
      });

      this.allNodes.forEach((node) => {
        if (!visible.has(node.id)) return;
        const pos = this.positions.get(node.id);
        const color = COLORS[node.group] || COLORS.Unknown;
        const selected = node.id === this.selectedId ? " is-selected" : "";
        const related = this.selectedId && (this.adj[this.selectedId] || []).some((rel) => rel.to === node.id) ? " is-related" : "";
        this.nodeLayer.insertAdjacentHTML("beforeend", `
          <g class="graph-node${selected}${related}" data-node-id="${escape(node.id)}">
            <circle cx="${pos.x}" cy="${pos.y}" r="${selected ? 13 : 10}" fill="${color}">
              <title>${escape(node.label)} - ${escape(node.group)}${node.description ? `: ${escape(label(node.description, 120))}` : ""}</title>
            </circle>
            ${showLabels || selected ? `<text x="${pos.x + 14}" y="${pos.y + 4}">${escape(label(node.label, 34))}</text>` : ""}
          </g>
        `);
      });

      this.nodeLayer.querySelectorAll(".graph-node").forEach((el) => {
        el.addEventListener("click", () => this.focusNode(el.dataset.nodeId));
        el.addEventListener("pointerenter", (event) => this.showTooltip(el.dataset.nodeId, event));
        el.addEventListener("pointermove", (event) => this.moveTooltip(event));
        el.addEventListener("pointerleave", () => this.hideTooltip());
        el.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
          const pos = this.positions.get(el.dataset.nodeId);
          this.nodeDrag = { id: el.dataset.nodeId, x: event.clientX, y: event.clientY, px: pos.x, py: pos.y };
        });
      });
      this.applyTransform();
    }

    focusNode(id) {
      if (!this.nodeById[id]) return;
      this.focusedId = id;
      this.selectedId = id;
      this.render();
      this.fit([...this.visibleNodeIds()]);
      this.renderDetail(id);
    }

    showAll() {
      this.focusedId = null;
      this.selectedId = null;
      this.resultsEl.innerHTML = "";
      this.searchEl.value = "";
      this.render();
      this.fit([...this.visibleNodeIds()]);
      this.detailEl.innerHTML = `
        <strong>Ontology graph</strong>
        <small>${this.allNodes.length.toLocaleString()} nodes, ${this.allEdges.length.toLocaleString()} edges</small>
        <p>Hover for a preview. Click a product or related node for details.</p>
      `;
    }

    fit(ids) {
      const points = ids.map((id) => this.positions.get(id)).filter(Boolean);
      if (!points.length) return;
      const rect = this.svg.getBoundingClientRect();
      const minX = Math.min(...points.map((p) => p.x));
      const maxX = Math.max(...points.map((p) => p.x));
      const minY = Math.min(...points.map((p) => p.y));
      const maxY = Math.max(...points.map((p) => p.y));
      const pad = 100;
      const width = Math.max(120, maxX - minX + pad * 2);
      const height = Math.max(120, maxY - minY + pad * 2);
      this.scale = Math.max(0.15, Math.min(2.5, Math.min(rect.width / width, rect.height / height)));
      this.tx = rect.width / 2 - ((minX + maxX) / 2) * this.scale;
      this.ty = rect.height / 2 - ((minY + maxY) / 2) * this.scale;
      this.applyTransform();
    }

    applyTransform() {
      this.viewport.setAttribute("transform", `translate(${this.tx} ${this.ty}) scale(${this.scale})`);
    }

    renderDetail(id) {
      const node = this.nodeById[id];
      const rels = (this.adj[id] || []).slice().sort((a, b) => a.label.localeCompare(b.label));
      const items = rels.map((rel) => {
        const arrow = rel.dir === "out" ? "to" : "from";
        return `<li><span>${escape(relationshipLabel(rel.label, rel.dir))}</span>${escape(this.nodeById[rel.to]?.label || rel.to)} <small>${arrow} ${escape(this.nodeById[rel.to]?.group || "node")}</small></li>`;
      }).join("");
      const props = Object.entries(node.properties || {})
        .filter(([key, value]) => value && !["label", "description"].includes(key))
        .slice(0, 8)
        .map(([key, value]) => `<li><span>${escape(relationshipLabel(key, "out"))}</span>${escape(value)}</li>`)
        .join("");
      this.detailEl.innerHTML = `
        <strong>${escape(node.label)}</strong>
        <small>${escape(node.group)} · ${escape(node.id)}</small>
        ${node.description ? `<p>${escape(node.description)}</p>` : ""}
        ${props ? `<ul class="graph-detail__props">${props}</ul>` : ""}
        ${items ? `<ul>${items}</ul>` : "<p>No relationships found for this node.</p>"}
      `;
    }

    showTooltip(id, event) {
      const node = this.nodeById[id];
      if (!node || !this.tooltipEl) return;
      const count = (this.adj[id] || []).length;
      this.tooltipEl.innerHTML = `
        <strong>${escape(node.label)}</strong>
        <small>${escape(node.group)} / ${count} relationships</small>
        ${node.description ? `<p>${escape(label(node.description, 150))}</p>` : ""}
      `;
      this.tooltipEl.hidden = false;
      this.moveTooltip(event);
    }

    moveTooltip(event) {
      if (!this.tooltipEl || this.tooltipEl.hidden) return;
      const rect = this.container.querySelector(".graph-canvas").getBoundingClientRect();
      this.tooltipEl.style.left = `${Math.min(rect.width - 280, Math.max(12, event.clientX - rect.left + 14))}px`;
      this.tooltipEl.style.top = `${Math.min(rect.height - 150, Math.max(12, event.clientY - rect.top + 14))}px`;
    }

    hideTooltip() {
      if (this.tooltipEl) this.tooltipEl.hidden = true;
    }

    renderLegend() {
      const groups = sortGroups([...new Set(this.allNodes.map((node) => node.group || "Unknown"))]);
      this.legendEl.innerHTML = groups.map((group) => {
        const count = this.allNodes.filter((node) => (node.group || "Unknown") === group).length;
        return `
          <button type="button" class="graph-legend__item" data-group="${escape(group)}">
            <i style="background:${COLORS[group] || COLORS.Unknown}"></i>${escape(group)} (${count})
          </button>
        `;
      }).join("");
      this.legendEl.querySelectorAll("[data-group]").forEach((button) => {
        button.addEventListener("click", () => {
          const group = button.dataset.group;
          if (this.activeGroups.has(group)) this.activeGroups.delete(group);
          else this.activeGroups.add(group);
          button.classList.toggle("is-dim", !this.activeGroups.has(group));
          this.showAll();
        });
      });
    }

    renderSearch() {
      const q = normalizeSearch(this.searchEl.value);
      this.resultsEl.innerHTML = "";
      const preferredGroups = new Set(["Product", "OwnedProduct", "RecommendedProduct", "Package", "Account", "Card"]);
      const matches = q.length < 2
        ? this.allNodes.filter((node) => preferredGroups.has(node.group) && node.label).slice(0, 6)
        : this.allNodes.filter((node) => nodeMatchesQuery(node, q)).slice(0, 20);
      if (!matches.length) {
        if (q.length < 2) return;
        this.resultsEl.innerHTML = '<div class="graph-no-results">No matching nodes found.</div>';
        return;
      }
      this.resultsEl.innerHTML = `
        <div class="graph-results__count">${q.length < 2 ? "Example results" : `${matches.length} result${matches.length === 1 ? "" : "s"}`}</div>
        ${matches.map((node) => `
        <button type="button" class="graph-result" data-node-id="${escape(node.id)}">
          <strong>${escape(label(node.label, 34))}</strong>
          <small>${escape(node.group)}</small>
        </button>
      `).join("")}
      `;
    }

    renderSearchExamples() {
      const preferredGroups = new Set(["Product", "OwnedProduct", "RecommendedProduct", "Package", "Account", "Card"]);
      const examples = this.allNodes
        .filter((node) => preferredGroups.has(node.group) && node.label)
        .map((node) => node.label)
        .filter((value, index, values) => values.indexOf(value) === index)
        .slice(0, 4);
      if (!examples.length) {
        this.examplesEl.hidden = true;
        return;
      }
      this.examplesEl.innerHTML = `<small>Try searching for</small>${examples.map((value) => `<button type="button" data-search-example="${escape(value)}">${escape(value)}</button>`).join("")}`;
    }
  }

  function boot() {
    document.querySelectorAll("[data-relationship-graph]").forEach((container) => {
      const source = document.getElementById(container.dataset.graphSource);
      if (!source) return;
      try {
        const graph = JSON.parse(source.textContent);
        if (!graph.nodes || !graph.nodes.length) {
          container.innerHTML = '<div class="graph-empty">No graph data is available.</div>';
          return;
        }
        container.__relationshipGraph = new RelationshipGraph(container, graph);
      } catch (error) {
        container.innerHTML = '<div class="graph-empty">Graph data could not be rendered.</div>';
        console.error(error);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", boot);
  window.addEventListener("resize", () => {
    document.querySelectorAll("[data-relationship-graph]").forEach((container) => {
      const instance = container.__relationshipGraph;
      if (!instance) return;
      instance.layout();
      instance.render();
      instance.fit([...instance.visibleNodeIds()]);
    });
  });
})();
