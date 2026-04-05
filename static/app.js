(function () {
  const $ = (id) => document.getElementById(id);

  const questionEl = $("question");
  const submitBtn = $("submit");
  const statusEl = $("status");
  const errorBox = $("error-box");
  const resultEl = $("result");

  function setLoading(loading) {
    submitBtn.disabled = loading;
    statusEl.textContent = loading ? "正在请求后端…" : "";
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function renderTags(container, items) {
    container.innerHTML = "";
    if (!items || items.length === 0) {
      const span = document.createElement("span");
      span.className = "tag empty";
      span.textContent = "（无）";
      container.appendChild(span);
      return;
    }
    items.forEach((t) => {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = String(t);
      container.appendChild(span);
    });
  }

  function renderFewShots(container, list) {
    container.innerHTML = "";
    if (!list || list.length === 0) {
      const p = document.createElement("p");
      p.className = "tag empty";
      p.textContent = "（无，可在 data/few_shot.jsonl 中补充）";
      container.appendChild(p);
      return;
    }
    list.forEach((ex, i) => {
      const card = document.createElement("div");
      card.className = "fewshot-card";
      const sim = ex.similarity;
      const simEl = document.createElement("div");
      simEl.className = "sim";
      simEl.textContent =
        sim != null ? `示例 ${i + 1} · 相似度 ${Number(sim).toFixed(4)}` : `示例 ${i + 1}`;
      const qEl = document.createElement("p");
      qEl.className = "q";
      qEl.textContent = ex.question || "";
      const pre = document.createElement("pre");
      pre.textContent = ex.cypher || "";
      card.appendChild(simEl);
      card.appendChild(qEl);
      card.appendChild(pre);
      container.appendChild(card);
    });
  }

  function renderTable(thead, tbody, rows) {
    thead.innerHTML = "";
    tbody.innerHTML = "";
    if (!rows || rows.length === 0) {
      return;
    }
    const keys = new Set();
    rows.forEach((row) => {
      Object.keys(row).forEach((k) => keys.add(k));
    });
    const cols = Array.from(keys);
    const trh = document.createElement("tr");
    cols.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      cols.forEach((c) => {
        const td = document.createElement("td");
        const v = row[c];
        td.textContent =
          v != null && typeof v === "object" ? JSON.stringify(v) : String(v ?? "");
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  async function runQuery() {
    const q = (questionEl.value || "").trim();
    if (!q) {
      showError("请输入问题。");
      return;
    }
    clearError();
    resultEl.classList.add("hidden");
    setLoading(true);

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const detail =
          data.detail != null
            ? typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail)
            : res.statusText;
        throw new Error(detail || `HTTP ${res.status}`);
      }

      renderTags($("kw-exact"), data.exact_match_keywords);
      renderTags($("kw-vector"), data.vector_schema_keywords);
      renderFewShots($("fewshots"), data.few_shot_retrieved);
      $("cypher").textContent = data.generated_cypher || "";

      const st = data.execution_status || "";
      $("exec-status").textContent = st;
      $("exec-status").style.color = st === "success" ? "var(--ok)" : "var(--danger)";

      const rows = data.execution_data || [];
      $("exec-count").textContent = String(rows.length);
      $("failure-count").textContent = String(data.failure_count ?? 0);

      const errEl = $("exec-err");
      if (data.execution_error) {
        errEl.textContent = "错误：" + data.execution_error;
        errEl.classList.remove("hidden");
      } else {
        errEl.textContent = "";
        errEl.classList.add("hidden");
      }

      const table = $("data-table");
      const thead = $("data-thead");
      const tbody = $("data-tbody");
      if (rows.length > 0) {
        renderTable(thead, tbody, rows);
        table.classList.remove("hidden");
      } else {
        table.classList.add("hidden");
      }

      resultEl.classList.remove("hidden");
    } catch (e) {
      showError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  submitBtn.addEventListener("click", runQuery);
  questionEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      runQuery();
    }
  });
})();
