(function () {
  const endpoint = "/api/chatbot/graph/";
  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const sendButton = document.getElementById("sendButton");
  const conversation = document.getElementById("conversation");
  const status = document.getElementById("connectionStatus");
  const modelSwitcher = document.querySelector("[data-dev-model-switcher]");
  const modelSelect = document.querySelector("[data-model-select]");
  const modelBadge = document.querySelector("[data-model-badge]");
  const avatarTemplate = document.getElementById("atlas-avatar-svg-template");
  let pending = false;
  const history = [];
  let conversationState = {};

  function addAtlasAvatar(target, variant) {
    if (!avatarTemplate) {
      return null;
    }
    const avatar = document.createElement("span");
    avatar.className = `atlas-avatar ${variant}`;
    avatar.setAttribute("aria-hidden", "true");
    const svg = avatarTemplate.content.querySelector("svg").cloneNode(true);
    svg.setAttribute("aria-hidden", "true");
    avatar.appendChild(svg);
    target.appendChild(avatar);
    return avatar;
  }

  document.querySelectorAll("[data-welcome-avatar]").forEach((slot) => {
    const svg = avatarTemplate && avatarTemplate.content.querySelector("svg");
    if (svg) {
      slot.appendChild(svg.cloneNode(true));
    }
  });

  function csrfToken() {
    const match = document.cookie.match(/(?:^|; )csrftoken=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  function setPending(value) {
    pending = value;
    input.disabled = value;
    sendButton.disabled = value;
    const sendIcon = document.createElement("span");
    sendIcon.setAttribute("aria-hidden", "true");
    if (value) {
      sendIcon.className = "send-spinner";
    } else {
      sendIcon.textContent = "→";
    }
    sendButton.replaceChildren(sendIcon);
    sendButton.setAttribute("aria-label", value ? "Sending message" : "Send message");
    status.textContent = value ? "Thinking" : "Ready";
    status.classList.toggle("busy", value);
  }

  function scrollToLatest() {
    conversation.scrollTop = conversation.scrollHeight;
  }

  function appendTextWithLineBreaks(parent, text) {
    String(text).split("\n").forEach((line, index) => {
      if (index > 0) {
        parent.appendChild(document.createElement("br"));
      }
      if (line) {
        parent.appendChild(document.createTextNode(line));
      }
    });
  }

  function removeCitationMarkers(text) {
    return String(text || "")
      .replace(/[ \t]*\[(?:S\d+)\][ \t]*/g, " ")
      .split("\n")
      .map((line) => line.replace(/[ \t]{2,}/g, " ").replace(/ +([.,;:!?])/g, "$1").trim())
      .filter(Boolean)
      .join("\n");
  }

  function appendInlineFormattedText(parent, text, options) {
    const source = String(text || "");
    const allowBold = !(options && options.disableBold);
    let cursor = 0;

    while (cursor < source.length) {
      const boldIndex = allowBold ? source.indexOf("**", cursor) : -1;
      const nextIndex = boldIndex;

      if (nextIndex < 0) {
        appendTextWithLineBreaks(parent, source.slice(cursor));
        break;
      }

      if (nextIndex > cursor) {
        appendTextWithLineBreaks(parent, source.slice(cursor, nextIndex));
      }

      const closeIndex = source.indexOf("**", boldIndex + 2);
      if (closeIndex < 0) {
        appendTextWithLineBreaks(parent, source.slice(boldIndex, boldIndex + 2));
        cursor = boldIndex + 2;
        continue;
      }

      const strongText = source.slice(boldIndex + 2, closeIndex);
      if (strongText.trim()) {
        const strong = document.createElement("strong");
        appendInlineFormattedText(strong, strongText, { disableBold: true });
        parent.appendChild(strong);
      } else {
        appendTextWithLineBreaks(parent, source.slice(boldIndex, closeIndex + 2));
      }
      cursor = closeIndex + 2;
    }
  }

  function sentenceChunks(text) {
    return text
      .replace(/\s+/g, " ")
      .split(/(?<=[.!?])\s+(?=[A-ZÀ-Ý"])/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function legacySectionizeAssistantText(text) {
    const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
    if (!normalized) {
      return [{ type: "paragraph", text: "" }];
    }

    const explicitLines = normalized.split("\n").map((line) => line.trim()).filter(Boolean);
    if (explicitLines.some((line) => /^([-*•]|\d+[.)])\s+/.test(line))) {
      const sections = [];
      let list = null;
      explicitLines.forEach((line) => {
        const bullet = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
        if (bullet) {
          if (!list) {
            list = { type: "list", items: [] };
            sections.push(list);
          }
          list.items.push(bullet[2]);
        } else {
          list = null;
          sections.push({ type: "paragraph", text: line });
        }
      });
      return sections;
    }

    const labels = [
      "Useful confirmed context",
      "Useful confirmed information",
      "Confirmed information",
      "What you can do",
      "Important",
      "Next step",
    ];
    const labelPattern = new RegExp(`\\b(${labels.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")}):`, "gi");
    const marked = normalized.replace(labelPattern, "\n$1:\n");
    const blocks = marked.split("\n").map((block) => block.trim()).filter(Boolean);
    const sections = [];
    blocks.forEach((block) => {
      if (labels.some((label) => block.toLowerCase() === `${label.toLowerCase()}:`)) {
        sections.push({ type: "label", text: block.slice(0, -1) });
        return;
      }
      const sentences = sentenceChunks(block);
      if (sections.length && sections[sections.length - 1].type === "label" && sentences.length > 1) {
        sections.push({ type: "list", items: sentences });
      } else if (sentences.length >= 4) {
        sections.push({ type: "paragraph", text: sentences.slice(0, 2).join(" ") });
        sections.push({ type: "paragraph", text: sentences.slice(2).join(" ") });
      } else {
        sections.push({ type: "paragraph", text: block });
      }
    });
    return sections;
  }

  function sectionizeAssistantText(text) {
    let normalized = String(text || "").replace(/\r\n?/g, "\n").replace(/^\n+|\n+$/g, "");
    if (!normalized.trim()) {
      return [{ type: "paragraph", text: "" }];
    }
    if (!normalized.includes("\n")) {
      const inlineNumberedItems = normalized.match(/\b\d+[.)]\s+/g) || [];
      if (inlineNumberedItems.length >= 2) {
        normalized = normalized.replace(/\s+(?=\d+[.)]\s+)/g, "\n");
      }
      normalized = normalized.replace(/(\[S\d+\])\s+(?=[A-Z][^.!?\n]{2,80}:)/g, "$1\n\n");
    }

    const sections = [];
    const paragraphLines = [];
    let activeList = null;

    function flushParagraph() {
      const paragraphText = removeCitationMarkers(paragraphLines.join("\n"));
      if (paragraphText) {
        sections.push({ type: "paragraph", text: paragraphText });
      }
      paragraphLines.length = 0;
    }

    function closeList() {
      activeList = null;
    }

    normalized.split("\n").forEach((line) => {
      const trimmed = line.trim();
      const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
      const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);

      if (!trimmed) {
        flushParagraph();
        closeList();
        return;
      }

      const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        flushParagraph();
        closeList();
        sections.push({ type: "heading", level: Math.min(headingMatch[1].length + 2, 4), text: removeCitationMarkers(headingMatch[2]) });
        return;
      }

      if (unorderedMatch || orderedMatch) {
        const type = unorderedMatch ? "unordered-list" : "ordered-list";
        const itemText = removeCitationMarkers((unorderedMatch || orderedMatch)[1]);
        flushParagraph();
        if (!itemText) {
          return;
        }
        if (!activeList || activeList.type !== type) {
          activeList = { type, items: [] };
          sections.push(activeList);
        }
        activeList.items.push(itemText);
        return;
      }

      closeList();
      paragraphLines.push(trimmed);
    });

    flushParagraph();
    return sections;
  }

  function renderAssistantBody(text, options) {
    const body = document.createElement("div");
    body.className = "message-body assistant-response";
    if (options && options.loading) {
      body.classList.add("assistant-response--loading");
    }

    sectionizeAssistantText(text).forEach((section) => {
      if (section.type === "label") {
        const label = document.createElement("div");
        label.className = "assistant-section-label";
        label.textContent = section.text;
        body.appendChild(label);
        return;
      }
      if (section.type === "heading") {
        const heading = document.createElement(section.level <= 3 ? "h" + section.level : "h4");
        heading.className = "assistant-heading";
        appendInlineFormattedText(heading, section.text);
        body.appendChild(heading);
        return;
      }
      if (section.type === "list" || section.type === "unordered-list" || section.type === "ordered-list") {
        const list = document.createElement(section.type === "ordered-list" ? "ol" : "ul");
        list.className = "assistant-list";
        section.items.forEach((item) => {
          const li = document.createElement("li");
          appendInlineFormattedText(li, item);
          list.appendChild(li);
        });
        body.appendChild(list);
        return;
      }
      const paragraph = document.createElement("p");
      appendInlineFormattedText(paragraph, section.text);
      body.appendChild(paragraph);
    });
    const citations = options && Array.isArray(options.citations) ? options.citations : [];
    if (citations.length) {
      const sources = document.createElement("section");
      sources.className = "assistant-sources";
      const title = document.createElement("h4");
      title.textContent = "Sources";
      sources.appendChild(title);
      const list = document.createElement("ul");
      list.className = "assistant-sources__list";
      citations.forEach((citation) => {
        if (!citation || typeof citation !== "object") return;
        const item = document.createElement("li");
        const label = citation.source_title || citation.document_id || citation.citation_id || "Verified source";
        const url = typeof citation.source_url === "string" ? citation.source_url.trim() : "";
        if (/^https?:\/\//i.test(url)) {
          const link = document.createElement("a");
          link.href = url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = label;
          item.appendChild(link);
        } else {
          item.textContent = label;
        }
        list.appendChild(item);
      });
      sources.appendChild(list);
      body.appendChild(sources);
    }
    return body;
  }

  function renderPlainBody(text) {
    const body = document.createElement("p");
    body.className = "message-body";
    body.textContent = text;
    return body;
  }

  function addMessage(role, text, options) {
    const message = document.createElement("article");
    message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
    if (options && options.error) {
      message.classList.add("error-message");
    }

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "ATLAS assistant";

    const body = role === "assistant" ? renderAssistantBody(text, options) : renderPlainBody(text);

    if (role === "assistant") {
      addAtlasAvatar(message, "atlas-avatar--message");
    }
    message.append(label, body);
    conversation.appendChild(message);
    scrollToLatest();
    return message;
  }

  function updateMessage(message, text, options) {
    const oldBody = message.querySelector(".message-body");
    const role = message.classList.contains("assistant-message") ? "assistant" : "user";
    const newBody = role === "assistant" ? renderAssistantBody(text, options) : renderPlainBody(text);
    oldBody.replaceWith(newBody);
    message.classList.toggle("error-message", Boolean(options && options.error));
    scrollToLatest();
  }

  function modelLabel(model) {
    if (model === "phi4-mini:latest" || model === "phi4-mini") {
      return "Phi";
    }
    if (model === "qwen2.5:7b-instruct" || model === "qwen2.5:7b") {
      return "Qwen";
    }
    return model || "Unknown";
  }

  function updateModelBadge(debugModels) {
    if (!modelBadge || !debugModels) {
      return;
    }
    modelBadge.textContent = `Classifier: ${modelLabel(debugModels.classifier)} · Generator: ${modelLabel(debugModels.generator)}`;
  }

  function resetChatContext() {
    history.splice(0, history.length);
    conversationState = {};
  }

  function backendMessage(payload, fallback) {
    if (payload && typeof payload.answer === "string" && payload.answer.trim()) {
      return payload.answer;
    }
    if (payload && payload.error && typeof payload.error.message === "string") {
      return payload.error.message;
    }
    if (payload && typeof payload.message === "string" && payload.message.trim()) {
      return payload.message;
    }
    return fallback;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || pending) {
      return;
    }

    addMessage("user", text);
    input.value = "";
    setPending(true);
    const loading = addMessage("assistant", "Thinking...");

    try {
      const headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
      };
      const token = csrfToken();
      if (token) {
        headers["X-CSRFToken"] = token;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        credentials: "same-origin",
        headers,
        body: JSON.stringify({ message: text, history: history.slice(-6), conversation_state: conversationState }),
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }

      if (!response.ok) {
        updateMessage(loading, backendMessage(payload, "The backend returned an error."), { error: true });
        return;
      }

      const answer = backendMessage(payload, "The backend returned an empty response.");
      updateMessage(loading, answer, { error: Boolean(payload && payload.failure_code && !payload.answer), citations: payload && payload.citations });
      if (payload && payload.conversation_state && typeof payload.conversation_state === "object") {
        conversationState = payload.conversation_state;
      }
      if (payload && payload.debug_models) {
        updateModelBadge(payload.debug_models);
      }
      history.push({ role: "user", content: text });
      history.push({ role: "assistant", content: answer });
    } catch (error) {
      updateMessage(loading, "Network error while contacting the chatbot backend.", { error: true });
    } finally {
      setPending(false);
      input.focus();
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage();
  });

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  if (modelSwitcher && modelSelect) {
    modelSelect.addEventListener("change", async function () {
      const configuration = modelSelect.value;
      modelSelect.disabled = true;
      status.textContent = "Switching";
      status.classList.add("busy");
      try {
        const headers = {
          "Content-Type": "application/json",
          "Accept": "application/json",
        };
        const token = csrfToken();
        if (token) {
          headers["X-CSRFToken"] = token;
        }
        const response = await fetch(modelSwitcher.dataset.endpoint, {
          method: "POST",
          credentials: "same-origin",
          headers,
          body: JSON.stringify({ configuration }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "model_switch_failed");
        }
        resetChatContext();
        updateModelBadge(payload.debug_models);
        addMessage("assistant", "Model configuration switched. Starting a fresh comparison context.");
      } catch (error) {
        addMessage("assistant", "Could not switch model configuration.", { error: true });
      } finally {
        modelSelect.disabled = false;
        status.textContent = "Ready";
        status.classList.remove("busy");
        input.focus();
      }
    });
  }
})();
