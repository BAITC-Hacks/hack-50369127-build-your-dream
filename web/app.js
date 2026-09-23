"use strict";

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];
const app = { state: { dataset: null, model: null, last_run: null, backtest: null }, hours: 48, busy: false, showAllRows: false, view: "overview" };
const numberFormat = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });
const integerFormat = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
const dateFormat = new Intl.DateTimeFormat("ru-RU", { timeZone: "UTC", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
const shortDateFormat = new Intl.DateTimeFormat("ru-RU", { timeZone: "UTC", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
const timeFormat = new Intl.DateTimeFormat("ru-RU", { timeZone: "UTC", hour: "2-digit", minute: "2-digit" });
const stages = [
  { keys: ["weather", "fetch", "weather_fetch", "retrieve_weather"], title: "Получение погоды", note: "Прогноз по координатам ВЭС" },
  { keys: ["prepare", "preparation", "data", "prepare_data", "validate"], title: "Подготовка данных", note: "Проверка и сбор признаков" },
  { keys: ["model", "predict", "prediction", "forecast", "inference"], title: "Расчёт модели", note: "Почасовая мощность турбин" },
  { keys: ["analyze", "analyse", "analysis", "quality", "evaluate"], title: "Анализ результата", note: "Покрытие и предупреждения" },
  { keys: ["save", "export", "complete", "update", "refresh", "persist", "reuse"], title: "Сохранение прогноза", note: "Результат и история запуска" },
];
const reportLabels = {
  total_rows: "Всего строк", rows: "Строк", accepted_rows: "Принято строк", valid_rows: "Корректных строк",
  invalid_rows: "Некорректных строк", rejected_rows: "Отклонено строк", dropped_rows: "Исключено строк",
  excluded_future_rows: "Исключено строк после отсечки", excluded_after_cutoff: "Исключено после отсечки",
  duplicate_rows: "Дубликатов", duplicates: "Дубликатов", missing_hours: "Пропущено часов",
  start: "Начало периода", end: "Конец периода", min_time: "Начало периода", max_time: "Конец периода",
  train_rows: "Строк обучения", training_rows: "Строк обучения", validation_rows: "Строк валидации",
  mae: "MAE", rmse: "RMSE", validation_mae: "MAE на валидации", validation_rmse: "RMSE на валидации",
  model: "Модель", model_type: "Тип модели", method: "Метод", training_cutoff: "Отсечка обучения",
  timezone_offset_hours: "Смещение UTC при импорте", power_scale: "Масштаб исходной мощности",
  turbine_id: "Турбина", turbine: "Турбина", warnings: "Предупреждения", source: "Источник",
  coverage_hours: "Покрытие, часов", expected_hours: "Ожидается часов", horizon_hours: "Горизонт, часов",
  mean_power: "Средняя нормализованная мощность", max_power: "Пик нормализованной мощности",
  min_power: "Минимум нормализованной мощности", mean_wind_speed: "Средний ветер, м/с",
  stale: "Устарел", reused: "Использован кэш", calibration_rows: "Строк для диапазона",
  cutoff: "Отсечка", first_timestamp: "Первая отметка времени", last_timestamp: "Последняя отметка времени",
  input_rows: "Исходных строк", hours_complete: "Полных часов", hours_total: "Всего часов", hours_target_usable: "Пригодных часов",
  missing_10min_slots: "Пропущено 10-минутных интервалов", hours_without_readings: "Часов без наблюдений",
  exact_duplicates_removed: "Удалено точных дублей", source_timezone: "Часовой пояс источника", timestamp_convention: "Положение временной метки",
  min_valid_samples: "Минимум отсчётов в часе", turbines: "Турбины", raw_rows: "Исходных записей",
};

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function icon(name) { return '<svg aria-hidden="true"><use href="#i-' + name + '"/></svg>'; }
function finite(value) { return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)); }
function n(value, integer = false) { return finite(value) ? (integer ? integerFormat : numberFormat).format(Number(value)) : "—"; }
function timezoneOffset() {
  const offset = Number(app.state.config?.timezone_offset_hours ?? 5);
  return Number.isFinite(offset) && Math.abs(offset) <= 14 ? offset : 5;
}
function timezoneLabel() { const offset = timezoneOffset(); return offset === 0 ? "UTC" : "UTC" + (offset > 0 ? "+" : "−") + Math.abs(offset); }
function timezoneSuffix() {
  const offset = Math.round(timezoneOffset() * 60);
  return (offset < 0 ? "-" : "+") + String(Math.floor(Math.abs(offset) / 60)).padStart(2, "0") + ":" + String(Math.abs(offset) % 60).padStart(2, "0");
}
function displayDate(value) { return new Date(new Date(value).getTime() + timezoneOffset() * 3600000); }
function isProject() { return app.state.dataset?.kind === "project_archive"; }
function isProjectModel() { return app.state.model?.kind === "catboost_archive"; }
function executionOf(run) { return run?.execution || run?.engine?.execution || run?.status?.execution || ""; }
function isSavedArchive(run = app.state.last_run) { return executionOf(run) === "saved_archive"; }
function fmtDate(value, short = false) {
  if (!value) return "—";
  const date = displayDate(value);
  return Number.isNaN(date.getTime()) ? String(value) : (short ? shortDateFormat : dateFormat).format(date);
}
function mean(values) {
  const valid = values.filter(finite).map(Number);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}
function safeText(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Да" : "Нет";
  if (Array.isArray(value)) return value.map(safeText).join(", ");
  if (typeof value === "object") return Object.entries(value).map(([k,v]) => (reportLabels[k] || k) + ": " + safeText(v)).join("; ");
  return String(value);
}
function sourceRow(label, value, html = false) {
  return '<div class="source-row"><span>' + escapeHTML(label) + '</span><span>' + (html ? value : escapeHTML(safeText(value))) + '</span></div>';
}
function note(text, warning = false) {
  return '<div class="analysis-note' + (warning ? " warning" : "") + '">' + icon(warning ? "info" : "check") + "<span>" + escapeHTML(text) + "</span></div>";
}
function showNotice(message, type = "") {
  $("global-notice").textContent = message;
  $("global-notice").className = "notice" + (type ? " " + type : "");
  $("global-notice").hidden = !message;
}
function setView(view, updateHash = true) {
  if (!["overview", "data", "backtest", "about"].includes(view)) view = "overview";
  app.view = view;
  all(".page-view").forEach((el) => { el.hidden = el.id !== "view-" + view; });
  all(".nav-item").forEach((button) => {
    const selected = button.dataset.view === view;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("current-page").textContent = ({ overview: "Обзор станции", data: "Исторические данные", backtest: "Ретроспективный тест", about: "Как это работает" })[view];
  if (updateHash && location.hash !== "#" + view) history.replaceState(null, "", "#" + view);
}
async function api(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (_) {
    throw new Error("Нет соединения с локальным сервером. Проверьте, что приложение запущено, и повторите действие.");
  }
  let data;
  try { data = await response.json(); } catch (_) { throw new Error("Сервер вернул неожиданный ответ. Проверьте журнал приложения."); }
  if (!response.ok || data.error) throw new Error(data.error || "Не удалось выполнить запрос (" + response.status + ").");
  return data;
}
function updateButtons() {
  const hasData = Boolean(app.state.dataset);
  $("forecast-button").disabled = app.busy || !hasData;
  $("backtest-button").disabled = app.busy || !hasData;
  $("refresh-button").disabled = app.busy || !app.state.last_run;
  $("export-forecast").disabled = app.busy || !app.state.last_run?.rows?.length;
  $("demo-button").disabled = app.busy;
  $("import-submit").disabled = app.busy;
  all(".import-trigger").forEach((button) => { button.disabled = app.busy; });
  all("[data-horizon],#as-of,#weather-mode,#turbine-select,#power-scale,#data-timezone,#csv-file").forEach((el) => { el.disabled = app.busy; });
  if ($("export-backtest")) $("export-backtest").disabled = app.busy;
}
async function withBusy(button, label, action) {
  if (app.busy) return;
  app.busy = true;
  const original = button.innerHTML;
  button.innerHTML = icon("refresh") + escapeHTML(label);
  button.classList.add("is-busy");
  button.setAttribute("aria-busy", "true");
  const bar = document.createElement("div");
  bar.className = "loading-bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-label", label);
  document.body.appendChild(bar);
  updateButtons();
  $("agent-status").textContent = "РАБОТАЕТ";
  $("agent-status").className = "tiny-badge running";
  try { await action(); }
  catch (error) {
    showNotice(error.message || String(error), "error");
    if ($("import-dialog").open) {
      $("import-error").textContent = error.message || String(error);
      $("import-error").hidden = false;
    }
  } finally {
    app.busy = false;
    button.innerHTML = original;
    button.classList.remove("is-busy");
    button.removeAttribute("aria-busy");
    bar.remove();
    updateButtons();
    renderPipeline();
  }
}
async function loadState(initialize = false) {
  const previousKind = app.state.dataset?.kind;
  app.state = await api("/api/state");
  const datasetChanged = previousKind !== app.state.dataset?.kind;
  if (initialize && app.state.last_run) {
    const run = app.state.last_run;
    if (run.as_of) {
      const shifted = displayDate(run.as_of);
      if (!Number.isNaN(shifted.getTime())) $("as-of").value = shifted.toISOString().slice(0, 16);
    }
    if ([24, 48].includes(Number(run.hours))) app.hours = Number(run.hours);
    $("weather-mode").value = isProject() ? "project" : run.mode === "demo" ? "demo" : "archive";
  } else if (initialize || datasetChanged) {
    $("weather-mode").value = isProject() ? "project" : app.state.dataset?.demo ? "demo" : "archive";
    const cutoff = app.state.config?.training_cutoff;
    if (cutoff) {
      const shifted = displayDate(cutoff);
      if (!Number.isNaN(shifted.getTime())) $("as-of").value = shifted.toISOString().slice(0, 16);
    }
  }
  if (initialize || datasetChanged) {
    $("data-timezone").value = String(timezoneOffset());
  }
  render();
  if (initialize && app.state.last_run && ["archive", "project"].includes(app.state.last_run.mode)) resultNotice(app.state.last_run);
}
function selectedRows() {
  const rows = app.state.last_run?.rows || [];
  const turbine = $("turbine-select").value;
  if (turbine !== "farm") return rows.filter((row) => String(row.turbine_id) === turbine).sort((a, b) => new Date(a.valid_time) - new Date(b.valid_time));
  const groups = new Map();
  rows.forEach((row) => {
    if (!groups.has(row.valid_time)) groups.set(row.valid_time, []);
    groups.get(row.valid_time).push(row);
  });
  return [...groups.entries()].map(([time, group]) => ({
    valid_time: time, turbine_id: "farm", turbines: new Set(group.map((row) => String(row.turbine_id))).size,
    wind_speed: mean(group.map((r) => r.wind_speed)), temperature: mean(group.map((r) => r.temperature)),
    power_pred: mean(group.map((r) => r.power_pred)), lower: mean(group.map((r) => r.lower)), upper: mean(group.map((r) => r.upper)),
  })).sort((a, b) => new Date(a.valid_time) - new Date(b.valid_time));
}
function render() {
  const { dataset, config, last_run: run } = app.state;
  all("[data-display-timezone]").forEach((element) => { element.textContent = timezoneLabel(); });
  const projectOption = $("weather-mode").querySelector('option[value="project"]');
  if (projectOption) { projectOption.hidden = !isProject(); projectOption.disabled = !isProject(); }
  if (!app.busy) $("refresh-button").innerHTML = icon("refresh") + (isProject() ? "Проверить кеш проекта" : "Проверить обновления");
  $("refresh-button").title = isProject() ? "Проверяет локальный кеш и пересчитывает результат; новые погодные данные не загружаются" : "Проверяет изменения входных данных и обновляет прогноз";
  $("welcome-card").hidden = Boolean(dataset);
  const badge = $("environment-badge");
  badge.className = "badge " + (!dataset ? "neutral" : dataset.demo || run?.mode === "demo" ? "demo" : "");
  badge.innerHTML = '<span class="dot"></span>' + (!dataset ? "Ожидание данных" : dataset.demo || run?.mode === "demo" ? "Синтетическое демо" : isProject() ? "Реальные данные · архив проекта" : "Данные загружены");
  const turbines = config?.turbines || [];
  if (turbines.length) {
    $("station-coordinates").textContent = Number(turbines[0].latitude).toFixed(4) + "° N, " + Number(turbines[0].longitude).toFixed(4) + "° E";
    const current = $("turbine-select").value;
    $("turbine-select").innerHTML = '<option value="farm">Среднее двух турбин</option>' + turbines.map((t) => '<option value="' + escapeHTML(t.id) + '">' + escapeHTML(t.name || "Турбина " + t.id) + "</option>").join("");
    if ([...$("turbine-select").options].some((o) => o.value === current)) $("turbine-select").value = current;
  }
  all("[data-horizon]").forEach((button) => {
    const selected = Number(button.dataset.horizon) === app.hours;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderForecast();
  renderPipeline();
  renderSources();
  renderAnalysis();
  renderDataset();
  renderBacktest();
  updateButtons();
}
function renderForecast() {
  const rows = selectedRows();
  const run = app.state.last_run;
  const dataset = app.state.dataset;
  const selected = $("turbine-select").value;
  const power = mean(rows.map((r) => r.power_pred));
  $("metric-power").innerHTML = (power === null ? "—" : n(power * 100)) + "<small>%</small>";
  $("metric-wind").innerHTML = n(mean(rows.map((r) => r.wind_speed))) + "<small>м/с</small>";
  $("metric-hours").innerHTML = n(run?.hours || app.hours, true) + "<small>часов</small>";
  $("metric-data").innerHTML = n(app.state.model?.training_rows ?? dataset?.rows ?? dataset?.report?.accepted_rows, true) + "<small>строк</small>";
  $("metric-power-note").textContent = selected === "farm" ? "Среднее двух турбин · равные веса" : "Нормализованная мощность турбины";
  $("metric-wind-note").textContent = run ? (run.mode === "demo" ? "Синтетическая погодная траектория" : run.mode === "project" ? "Кеш проекта · ECMWF" : "Архивный погодный прогноз") : "Из выбранного погодного прогноза";
  $("metric-data-note").textContent = !dataset ? "Исторические данные не загружены" : dataset.demo ? "Синтетическая история · демо" : isProjectModel() ? "Обучающая выборка CatBoost из проекта" : "Загруженная история станции";
  $("metric-hours-note").textContent = run && Number(run.hours) !== app.hours ? "Выбрано " + app.hours + " ч. Применится после запроса прогноза." : run ? "Выпуск на " + fmtDate(run.as_of) + " " + timezoneLabel() : "Почасовой шаг · 2 турбины";
  $("chart-period").textContent = rows.length ? fmtDate(rows[0].valid_time) + " — " + fmtDate(rows[rows.length - 1].valid_time) + " · " + timezoneLabel() + (isSavedArchive(run) ? " · сохранённый расчёт" : "") : "Подготовьте данные, чтобы увидеть прогноз";
  $("chart-empty").hidden = rows.length > 0;
  const hasInterval = rows.some((row) => finite(row.lower) && finite(row.upper));
  $("interval-legend").hidden = rows.length > 0 && !hasInterval;
  $("chart-caption").textContent = rows.length && !hasInterval ? "Нормализованная мощность. Диапазон неопределённости отсутствует в результате." : selected === "farm" ? "Среднее нормализованной мощности. Диапазон — среднее границ турбин." : "Эмпирический диапазон модели; не гарантированный доверительный интервал.";
  if (!rows.length) $("chart-caption").textContent = "Мощность нормализована от 0 до 100%. Это не энергия в МВт·ч.";
  renderChart(rows);
  renderTable(rows);
}
function renderChart(rows) {
  const width = 720, height = 260;
  const pad = { left: 42, right: 20, top: 12, bottom: 36 };
  const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
  const x = (i) => pad.left + (i / Math.max(1, rows.length - 1)) * plotW;
  const y = (v) => pad.top + (1 - Math.min(1, Math.max(0, Number(v)))) * plotH;
  let svg = '<svg viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none" role="img" aria-label="' + escapeHTML(rows.length ? "Почасовой прогноз мощности, от 0 до 100 процентов. Подробные значения в таблице ниже." : "Область графика. Прогноз пока не рассчитан.") + '"><defs><linearGradient id="power-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#61aa8e" stop-opacity=".17"/><stop offset="100%" stop-color="#61aa8e" stop-opacity=".02"/></linearGradient></defs>';
  [0, 25, 50, 75, 100].forEach((tick) => {
    const yp = y(tick / 100);
    svg += '<line x1="' + pad.left + '" x2="' + (width - pad.right) + '" y1="' + yp + '" y2="' + yp + '" stroke="#edf0e9" stroke-width="1" stroke-dasharray="3 5"/><text x="' + (pad.left - 12) + '" y="' + (yp + 3) + '" text-anchor="end" fill="#a6b09f" stroke="none" font-size="9" font-family="inherit">' + tick + "</text>";
  });
  if (rows.length) {
    const line = rows.map((r, i) => (i ? "L" : "M") + x(i).toFixed(2) + " " + y(r.power_pred).toFixed(2)).join(" ");
    const polygon = rows.map((r, i) => x(i).toFixed(2) + "," + y(finite(r.upper) ? r.upper : r.power_pred).toFixed(2)).concat(rows.map((r, i) => x(i).toFixed(2) + "," + y(finite(r.lower) ? r.lower : r.power_pred).toFixed(2)).reverse()).join(" ");
    svg += '<polygon points="' + polygon + '" fill="#deeee2" fill-opacity=".72" stroke="none"/>';
    svg += '<path d="' + line + " L" + x(rows.length - 1) + " " + y(0) + " L" + x(0) + " " + y(0) + ' Z" fill="url(#power-fill)" stroke="none"/>';
    svg += '<path d="' + line + '" fill="none" stroke="#429778" stroke-width="2.5" vector-effect="non-scaling-stroke"/>';
    const ticks = [...new Set([0, ...[1, 2, 3, 4, 5].map((i) => Math.round(i * (rows.length - 1) / 5))])];
    ticks.forEach((i) => {
      const date = displayDate(rows[i].valid_time);
      const label = Number.isNaN(date.getTime()) ? "" : timeFormat.format(date);
      svg += '<text x="' + x(i) + '" y="' + (height - 13) + '" text-anchor="middle" fill="#9ca997" stroke="none" font-size="9" font-family="inherit">' + escapeHTML(label) + "</text>";
    });
    svg += '<line id="hover-line" x1="0" x2="0" y1="' + pad.top + '" y2="' + y(0) + '" stroke="#91b49d" stroke-dasharray="4 4" stroke-width="1" visibility="hidden"/>';
    svg += '<circle id="hover-dot" cx="0" cy="0" r="4" fill="#429778" stroke="white" stroke-width="2" visibility="hidden"/>';
    svg += '<rect id="chart-hit" x="' + pad.left + '" y="' + pad.top + '" width="' + plotW + '" height="' + plotH + '" fill="transparent" stroke="none"/>';
  } else {
    ["00:00", "08:00", "16:00", "00:00", "08:00", "16:00"].forEach((label, i) => {
      svg += '<text x="' + (pad.left + i * plotW / 5) + '" y="' + (height - 13) + '" text-anchor="middle" fill="#b1baa9" stroke="none" font-size="9" font-family="inherit">' + label + "</text>";
    });
  }
  $("forecast-chart").innerHTML = svg + "</svg>";
  $("chart-tooltip").hidden = true;
  if (!rows.length) return;
  const hit = $("chart-hit"), chart = $("forecast-chart").querySelector("svg");
  hit.addEventListener("pointermove", (event) => {
    const bounds = chart.getBoundingClientRect();
    const position = (event.clientX - bounds.left) / bounds.width * width;
    const index = Math.max(0, Math.min(rows.length - 1, Math.round((position - pad.left) / plotW * (rows.length - 1))));
    const row = rows[index];
    const line = $("hover-line"), dot = $("hover-dot");
    line.setAttribute("x1", x(index)); line.setAttribute("x2", x(index)); line.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(index)); dot.setAttribute("cy", y(row.power_pred)); dot.setAttribute("visibility", "visible");
    const tooltip = $("chart-tooltip");
    tooltip.innerHTML = escapeHTML(fmtDate(row.valid_time)) + "<br><strong>" + n(Number(row.power_pred) * 100) + "%</strong> мощности<br><small>Ветер: " + n(row.wind_speed) + " м/с</small>";
    tooltip.hidden = false;
    tooltip.style.left = Math.max(8, Math.min(bounds.width - tooltip.offsetWidth - 8, event.clientX - bounds.left + 12)) + "px";
    tooltip.style.top = Math.max(5, event.clientY - bounds.top - tooltip.offsetHeight - 10) + "px";
  });
  hit.addEventListener("pointerleave", () => {
    $("chart-tooltip").hidden = true;
    $("hover-line").setAttribute("visibility", "hidden");
    $("hover-dot").setAttribute("visibility", "hidden");
  });
}
function renderTable(rows = selectedRows()) {
  if (!rows.length) {
    $("forecast-table").innerHTML = '<tr><td colspan="5" class="empty-table">Рассчитайте первый прогноз, чтобы увидеть почасовые значения</td></tr>';
    $("table-pagination").hidden = true;
    return;
  }
  const visible = app.showAllRows ? rows : rows.slice(0, 8);
  $("forecast-table").innerHTML = visible.map((r) => '<tr><td>' + escapeHTML(fmtDate(r.valid_time)) + '</td><td>' + n(r.wind_speed) + '</td><td>' + n(r.temperature) + '</td><td><span class="power-cell"><span class="power-bar" aria-hidden="true" style="--power:' + Math.max(0, Math.min(100, Number(r.power_pred) * 100)) + '%"></span>' + n(Number(r.power_pred) * 100) + '</span></td><td>' + (finite(r.lower) && finite(r.upper) ? n(Number(r.lower) * 100) + "–" + n(Number(r.upper) * 100) : "—") + "</td></tr>").join("");
  $("table-pagination").hidden = false;
  $("table-count").textContent = "Показано " + visible.length + " из " + rows.length + " часов";
  $("table-more").hidden = rows.length <= 8;
  $("table-more").innerHTML = (app.showAllRows ? "Свернуть таблицу" : "Показать все часы") + icon("chevron");
  $("table-description").textContent = $("turbine-select").value === "farm" ? "Арифметическое среднее двух турбин · CSV содержит обе турбины" : $("turbine-select").selectedOptions[0].textContent + " · CSV содержит обе турбины";
}
function renderPipeline() {
  const run = app.state.last_run;
  const saved = isSavedArchive(run);
  const events = saved ? [] : run?.events || [];
  const shownStages = saved ? [
    { title: "Архив проекта", note: "Источник — импортированный проект" },
    { title: "Погодные данные", note: "Сведения о выпуске взяты из архива" },
    { title: "Сохранённый прогноз", note: "Модель в этом запросе не запускалась" },
    { title: "Ограничения", note: "Доступность погоды требует подтверждения" },
    { title: "Экспорт результата", note: "Архивные значения доступны в CSV" },
  ] : stages;
  let used = new Set();
  $("pipeline-list").innerHTML = shownStages.map((stage, index) => {
    let event = events.find((e, i) => !used.has(i) && stage.keys?.includes(String(e.stage).toLowerCase()));
    if (event) used.add(events.indexOf(event));
    const status = String(event?.status || "").toLowerCase();
    const failed = ["error", "failed", "failure"].includes(status);
    const done = Boolean(event) && !failed && !["running", "pending", "started"].includes(status);
    const message = event ? safeText(event.message || event.stage) : stage.note;
    return '<li class="' + (failed ? "failed" : done ? "done" : "") + '" title="' + escapeHTML(message) + '"><span class="stage-icon">' + (done ? icon("check") : failed ? icon("info") : String(index + 1).padStart(2, "0")) + '</span><div><h3>' + stage.title + "</h3><p>" + escapeHTML(message) + "</p></div></li>";
  }).join("");
  const hasRun = Boolean(run);
  $("agent-title").innerHTML = '<span class="live-dot"></span>' + (saved ? "Архивный расчёт" : "Журнал агента");
  $("agent-subtitle").textContent = saved ? "Сохранённый расчёт из архива" : "Прозрачный цикл прогнозирования";
  $("agent-status").textContent = app.busy ? "РАБОТАЕТ" : saved ? "ИЗ АРХИВА" : hasRun ? run.reused ? "ИЗ КЭША" : "ЗАВЕРШЁН" : "ОЖИДАНИЕ";
  $("agent-status").className = "tiny-badge" + (app.busy ? " running" : "");
  $("agent-footer-text").innerHTML = saved ? "Модель повторно не запускалась.<br>Показан результат из архива." : hasRun && run.mode === "demo" ? "Демонстрационный цикл<br>на синтетических данных" : "Контроль времени выпуска<br>и происхождения погоды";
}
function renderSources() {
  const { dataset, last_run: run, model } = app.state;
  const provenance = Array.isArray(run?.provenance) ? run.provenance : run?.provenance ? [run.provenance] : [];
  const first = provenance[0] || {};
  const source = first.source || first.provider || first.model || (run?.mode === "demo" ? "Синтетическая погода" : run ? "Архив погодных запусков" : "Не запрошена");
  let html = sourceRow("История ВЭС", dataset ? dataset.demo ? '<span class="tag warning">Синтетический пример</span>' : escapeHTML(dataset.source || "Загруженный CSV") : "Не загружена", Boolean(dataset));
  html += sourceRow("Погодные данные", source);
  if (isProject()) html += sourceRow("Обновление", "Проверка кеша · без загрузки погоды");
  const modelNames = { empirical_curve: "Эмпирическая кривая", constant: "Постоянный прогноз", persistence: "Последняя мощность" };
  const kinds = [...new Set(Object.values(model?.diagnostics || {}).map((d) => modelNames[d?.selected_kind] || d?.selected_kind).filter(Boolean))];
  html += sourceRow("Модель", model ? model.name || model.model_type || model.method || kinds.join(" / ") || "Обучена на загруженной истории" : "Не обучена");
  html += sourceRow("Момент прогноза", run ? fmtDate(run.as_of) + " " + timezoneLabel() : "—");
  const issue = first.issue_time || first.issued_at || first.run_time || first.model_run || first.init_time;
  if (issue) html += sourceRow("Выпуск погоды", fmtDate(issue) + " " + timezoneLabel());
  if (["archive", "project"].includes(run?.mode)) html += sourceRow("Доступность в прошлом", '<span class="tag warning">Требует подтверждения</span>', true);
  if (run?.mode === "project" || isSavedArchive(run)) html += sourceRow("Получение результата", isSavedArchive(run) ? "Сохранённый расчёт из архива" : executionOf(run) === "recomputed" ? "Модель выполнена заново" : "Результат проекта");
  if (run?.id) html += sourceRow("Идентификатор", run.id);
  if (provenance.length) {
    html += '<details class="model-details"><summary>Источники по каждой турбине</summary>' + provenance.map((p) => '<div class="provenance-item">' + sourceRow("Турбина", p.turbine_id) + sourceRow("Источник", p.source || p.provider || "—") + sourceRow("Выпуск", fmtDate(p.run_time || p.issue_time) + " " + timezoneLabel()) + sourceRow("Доступность по метаданным", fmtDate(p.available_at) + " " + timezoneLabel()) + sourceRow("Основание доступности", p.availability_basis || "Не указано") + sourceRow("Версия данных", p.sha256 || p.weather_sha256 || "Не указана") + '</div>').join("") + "</details>";
  }
  $("source-details").innerHTML = html;
}
function renderAnalysis() {
  const run = app.state.last_run;
  if (!run) {
    $("analysis-details").innerHTML = '<div class="quiet-empty">После расчёта здесь появятся покрытие горизонта, предупреждения и сведения о модели.</div>';
    return;
  }
  const rows = selectedRows();
  const runRows = run.rows || [];
  const uniqueHours = new Set(runRows.map((r) => r.valid_time)).size;
  const turbines = new Set(runRows.map((r) => String(r.turbine_id))).size;
  const fullCoverage = uniqueHours === Number(run.hours) && turbines === 2 && runRows.length === Number(run.hours) * 2;
  let html = isSavedArchive(run) ? note("Сохранённый расчёт из архива. Показаны готовые прогнозные значения; модель в этом запросе не запускалась.", true) : run.mode === "project" && executionOf(run) === "recomputed" ? note("CatBoost выполнен заново на погодных признаках из кеша проекта.") : "";
  html += note("Получено " + uniqueHours + " часов, турбин: " + turbines + ". " + (fullCoverage ? "Горизонт заполнен полностью." : "Проверьте полноту горизонта и наличие обеих турбин."), !fullCoverage);
  if (run.mode === "demo" || app.state.dataset?.demo) html += note("Синтетическое демо. Эти результаты не оценивают реальную выработку ВЭС за февраль.", true);
  if (["archive", "project"].includes(run.mode)) html += note("Архив: доступность требует подтверждения. Время фактической публикации запусков не подтверждено; результат не готов для конкурсной оценки.", true);
  if (run.eligibility?.reason) html += note(run.eligibility.reason, run.eligibility.competition_ready !== true);
  const warnings = [...new Set((run.warnings || []).map(safeText))];
  html += warnings.map((warning) => note(warning, true)).join("");
  if (rows.some((row) => $("turbine-select").value === "farm" && row.turbines !== 2)) html += note("В части часов отсутствует одна из турбин. Среднее рассчитано только по доступным строкам.", true);
  if (run.reused) html += note("Входные данные не изменились: повторно использован сохранённый расчёт.");
  $("analysis-details").innerHTML = html;
}
function reportItems(report, depth = 0) {
  if (!report || typeof report !== "object") return [];
  const items = [];
  Object.entries(report).forEach(([key, value]) => {
    if (value === null || value === undefined || key === "rows" && Array.isArray(value)) return;
    const label = reportLabels[key] || key.replaceAll("_", " ");
    if (typeof value === "object" && !Array.isArray(value) && depth < 1) {
      reportItems(value, depth + 1).forEach((item) => items.push(label + " · " + item));
    } else if (!Array.isArray(value) || value.length < 8) {
      items.push(label + ": " + safeText(value));
    } else {
      items.push(label + ": " + value.length + " записей");
    }
  });
  return items;
}
function renderProjectModel(model) {
  const metrics = Array.isArray(model.january_metrics) ? model.january_metrics : [];
  let html = '<div class="project-model-section"><h3 class="diagnostic-title">Модель из проекта</h3>';
  html += sourceRow("Модель", model.name || "CatBoost");
  html += sourceRow("Строк обучения", n(model.training_rows, true));
  if (model.training_cutoff) html += sourceRow("Отсечка обучения", fmtDate(model.training_cutoff) + " " + timezoneLabel());
  html += '<p class="small-muted model-context">Импортирована модель CatBoost. Повторное обучение при подключении архива не выполняется; способ получения каждого прогноза указан рядом с результатом.</p>';
  if (model.independent_retraining?.status === "complete") html += note("Независимое обучение воспроизведено: все 2 928 январских и 2 784 прогнозных значения 29 выпусков совпали с архивом. Февральские факты для оценки ошибки отсутствуют.");
  if (metrics.length) {
    html += '<h3 class="diagnostic-title">Январь 2026 · проверка прогноза мощности</h3>';
    html += '<p class="small-muted">MAE и RMSE приведены в процентных пунктах нормализованной мощности. Это отдельная проверка по январским прогнозам, не оценка февральской выработки.</p>';
    html += note(model.independently_verified_metrics ? "MAE и RMSE CatBoost независимо пересчитаны по сохранённым январским прогнозам и фактической мощности. Метрики базовых моделей импортированы из отчёта." : model.imported_report ? "Метрики импортированы из отчёта проекта. Независимый пересчёт метрик не подтверждён." : "Показаны январские метрики, сохранённые в проекте.", !model.independently_verified_metrics);
    html += '<div class="table-scroll"><table class="diagnostic-table"><thead><tr><th>Модель</th><th>Турбина</th><th>Горизонт</th><th>Строк</th><th>MAE, п. п.</th><th>RMSE, п. п.</th></tr></thead><tbody>';
    const modelNames = { selected_model: "CatBoost", historical_mean: "Средняя мощность", weather_power_curve: "Кривая мощности", frozen_persistence: "Последняя мощность", catboost: "CatBoost", catboost_global: "CatBoost", physics: "Физическая модель", power_curve: "Кривая мощности", persistence: "Последняя мощность", constant: "Средняя мощность" };
    for (const metric of metrics) {
      const factor = ["percentage_points", "percent", "pct"].includes(metric.unit || model.metrics_unit) ? 1 : 100;
      const horizon = metric.horizon ?? metric.horizon_hours;
      const turbine = metric.turbine_id ?? metric.turbine;
      const turbineLabel = turbine === "ALL" ? "Обе турбины" : /^T?[12]$/.test(String(turbine)) ? "Турбина " + String(turbine).replace(/^T/, "") : safeText(turbine);
      const horizonLabel = horizon === "ALL" ? "Все часы" : finite(horizon) ? n(horizon, true) + " ч" : /^\d+-\d+$/.test(String(horizon)) ? String(horizon).replace("-", "–") + " ч" : safeText(horizon);
      html += '<tr><td>' + escapeHTML(modelNames[metric.model] || metric.model || "CatBoost") + '</td><td>' + escapeHTML(turbineLabel) + '</td><td>' + escapeHTML(horizonLabel) + '</td><td>' + n(metric.n ?? metric.rows, true) + '</td><td>' + (finite(metric.mae) ? n(Number(metric.mae) * factor) : "—") + '</td><td>' + (finite(metric.rmse) ? n(Number(metric.rmse) * factor) : "—") + '</td></tr>';
    }
    html += '</tbody></table></div>';
  } else {
    html += note("В импортированном состоянии нет январских метрик модели. Числовая оценка качества не показывается.", true);
  }
  if (model.validation) html += note(safeText(model.validation));
  if (model.interval?.description) html += note(model.interval.description, !model.interval.calibrated);
  html += (model.warnings || []).map((warning) => note(safeText(warning), true)).join("");
  return html + '</div>';
}
function renderDataset() {
  const { dataset, model } = app.state;
  if (!dataset) {
    $("dataset-details").innerHTML = '<div class="large-empty">' + icon("data") + '<h2>История пока не загружена</h2><p>Добавьте реальные наблюдения CSV<br>или откройте синтетическое демо на странице обзора.</p></div>';
    return;
  }
  const turbineCount = Array.isArray(dataset.turbines) ? dataset.turbines.length : typeof dataset.turbines === "object" && dataset.turbines ? Object.keys(dataset.turbines).length : dataset.turbines || 2;
  let html = '<div class="data-section-title"><span>' + escapeHTML(dataset.source || "История станции") + '</span><span class="badge' + (dataset.demo ? " demo" : "") + '">' + (dataset.demo ? "Синтетическое демо" : isProject() ? "Реальные данные · архив проекта" : "Загруженный CSV") + "</span></div>";
  html += '<div class="data-stats"><div class="data-stat"><span>Строк данных</span><strong>' + n(dataset.rows, true) + '</strong></div><div class="data-stat"><span>Турбин</span><strong>' + n(turbineCount, true) + '</strong></div><div class="data-stat"><span>Модель</span><strong style="font-size:18px">' + (isProjectModel() ? "Импортирована" : model ? "Обучена" : "Не обучена") + "</strong></div></div>";
  if (dataset.demo) html += note("История создана программно. Для реального прогноза импортируйте статистику вашей ВЭС.", true);
  if (isProject()) html += note("Реальная телеметрия и модель импортированы из проекта. Время отображается в " + timezoneLabel() + "; часовой пояс исходной телеметрии принят из конфигурации архива и требует подтверждения.", true);
  html += (dataset.warnings || []).map((warning) => note(safeText(warning), true)).join("");
  const report = reportItems(dataset.report);
  if (report.length) html += '<h3 style="font-size:12px;margin-top:22px">Проверка исходных данных</h3><ul class="report-list">' + report.map((line) => "<li>" + escapeHTML(line) + "</li>").join("") + "</ul>";
  if (isProjectModel()) {
    html += renderProjectModel(model);
  } else if (model) {
    const modelNames = { empirical_curve: "Эмпирическая кривая", constant: "Постоянный прогноз", persistence: "Последняя мощность" };
    html += '<details class="model-details" open><summary>Диагностика модели</summary>';
    html += '<p class="small-muted">Ошибки рассчитаны на исторической валидации, в процентных пунктах нормализованной мощности. Это не точность прогноза за февраль.</p>';
    for (const [id, diagnostic] of Object.entries(model.diagnostics || {})) {
      html += '<h3 class="diagnostic-title">Турбина ' + escapeHTML(id) + ' · ' + escapeHTML(modelNames[diagnostic.selected_kind] || diagnostic.selected_kind || "модель") + '</h3>';
      html += '<div class="table-scroll"><table class="diagnostic-table"><thead><tr><th>Модель</th><th>MAE, п. п.</th><th>RMSE, п. п.</th></tr></thead><tbody>';
      for (const [label, metrics] of [["Выбранная модель", diagnostic.selected], ["Средняя мощность", diagnostic.constant_baseline], ["Последнее наблюдение", diagnostic.persistence_baseline]]) {
        if (!metrics) continue;
        html += '<tr><td>' + label + '</td><td>' + (finite(metrics.mae) ? n(Number(metrics.mae) * 100) : "—") + '</td><td>' + (finite(metrics.rmse) ? n(Number(metrics.rmse) * 100) : "—") + '</td></tr>';
      }
      html += '</tbody></table></div><p class="small-muted">Обучение: ' + n(diagnostic.fit_rows, true) + ' строк; валидация: ' + n(diagnostic.holdout_rows, true) + ' строк.' + (diagnostic.holdout_start ? ' Период: ' + escapeHTML(fmtDate(diagnostic.holdout_start)) + ' — ' + escapeHTML(fmtDate(diagnostic.holdout_end)) + '.' : '') + '</p>';
      if (diagnostic.validation_description) html += '<p class="small-muted">' + escapeHTML(diagnostic.validation_description) + '</p>';
    }
    if (model.validation) html += note(model.validation);
    if (model.interval?.description) html += note(model.interval.description, !model.interval.calibrated);
    html += (model.warnings || []).map((warning) => note(safeText(warning), true)).join("");
    html += "</details>";
  }
  $("dataset-details").innerHTML = html;
}
function renderBacktest() {
  const report = app.state.backtest;
  if (!report) {
    $("backtest-result").innerHTML = '<div class="large-empty">' + icon("history") + '<h2>Тест ещё не запускался</h2><p>Сначала подготовьте обучающие данные.<br>Демо проверяет работу процесса на синтетической погоде.</p></div>';
    return;
  }
  const runs = Array.isArray(report.runs) ? report.runs.length : report.runs;
  const rowCount = Array.isArray(report.rows) ? report.rows.length : report.rows;
  const coverage = typeof report.coverage_hours === "object" ? Math.min(...Object.values(report.coverage_hours)) : report.coverage_hours;
  let html = '<div class="data-section-title"><span>Результат за февраль 2026</span><button id="export-backtest" class="button button-secondary button-small">' + icon("download") + 'Скачать CSV</button></div>';
  html += '<div class="data-stats"><div class="data-stat"><span>Запусков</span><strong>' + n(runs, true) + '</strong></div><div class="data-stat"><span>Покрытие месяца</span><strong>' + n(coverage, true) + '<small style="font-size:12px"> / ' + n(report.expected_hours || 672, true) + '</small></strong></div><div class="data-stat"><span>Строк прогноза</span><strong>' + n(rowCount, true) + "</strong></div></div>";
  if (isSavedArchive(report)) html += note("Сохранённый расчёт из архива. Показаны результаты исторических запусков; модель повторно не выполнялась.", true);
  else if (report.mode === "project" && executionOf(report) === "recomputed") html += note("Прогнозы CatBoost рассчитаны заново на погодных признаках из кеша проекта.");
  html += note(report.mode === "demo" ? "Демонстрационный тест на синтетических входах. Он проверяет работоспособность полного цикла, а не качество реального прогноза." : "Архив: доступность требует подтверждения. Перед использованием результатов в конкурсе необходим аудит погодного источника.", true);
  html += note("Без фактической выработки за февраль MAE и RMSE тестового месяца не вычисляются.");
  if (report.eligibility?.reason) html += note(report.eligibility.reason, true);
  html += (report.warnings || []).map((warning) => note(safeText(warning), true)).join("");
  $("backtest-result").innerHTML = html;
  $("export-backtest").addEventListener("click", () => downloadCSV("backtest"));
}
function runPayload(refresh = false) {
  const date = $("as-of").value;
  if (!date) throw new Error("Укажите момент формирования прогноза.");
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(date)) throw new Error("Проверьте формат даты и времени.");
  return { as_of: date + ":00" + timezoneSuffix(), hours: app.hours, mode: $("weather-mode").value, refresh };
}
function resultNotice(run) {
  if (isSavedArchive(run)) showNotice("Сохранённый расчёт из архива. Модель в этом запросе не запускалась. Историческая доступность погодных выпусков требует подтверждения.", "warning");
  else if (run?.mode === "project") showNotice((executionOf(run) === "recomputed" ? "CatBoost выполнен заново на признаках из кеша проекта. " : "Прогноз из проекта загружен. ") + "Архив: доступность требует подтверждения.", "warning");
  else if (run?.mode === "archive") showNotice("Архив: доступность требует подтверждения. Проверьте предупреждения и происхождение погодных запусков перед конкурсной оценкой.", "warning");
  else showNotice(run?.reused ? "Входные данные не изменились. Сохранённый прогноз использован повторно." : "Демонстрационный прогноз рассчитан. Погода и результаты этого режима — синтетический пример.", "warning");
}
async function calculate(refresh = false) {
  const button = refresh ? $("refresh-button") : $("forecast-button");
  await withBusy(button, refresh ? "Проверяем…" : "Получаем прогноз…", async () => {
    showNotice($("weather-mode").value === "project" ? "Проверяем кеш проекта и получаем прогноз. В результате будет указан новый запуск модели или сохранённый расчёт." : refresh ? "Агент проверяет входные данные. Если они изменились, прогноз будет пересчитан." : "Агент получает погоду, подготавливает признаки и рассчитывает прогноз. Архивный запрос может занять несколько минут.");
    const result = await api("/api/forecast", runPayload(refresh));
    app.showAllRows = false;
    await loadState();
    if (!app.state.last_run && (result.rows || result.run?.rows)) { app.state.last_run = result.run || result; render(); }
    resultNotice(app.state.last_run);
  });
}
async function downloadCSV(kind) {
  try {
    const response = await fetch("/api/export?kind=" + encodeURIComponent(kind));
    if (!response.ok) {
      let message = "Не удалось скачать файл.";
      try { message = (await response.json()).error || message; } catch (_) {}
      throw new Error(message);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = kind === "backtest" ? "wind-agent-february-2026.csv" : "wind-agent-forecast.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) { showNotice(error.message || "Не удалось скачать CSV.", "error"); }
}

all(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelector(".brand").addEventListener("click", (event) => { event.preventDefault(); setView("overview"); });
window.addEventListener("hashchange", () => setView(location.hash.slice(1), false));
all("[data-horizon]").forEach((button) => button.addEventListener("click", () => {
  app.hours = Number(button.dataset.horizon);
  render();
}));
$("turbine-select").addEventListener("change", () => { renderForecast(); renderAnalysis(); });
$("forecast-button").addEventListener("click", () => calculate(false));
$("refresh-button").addEventListener("click", () => calculate(true));
$("table-more").addEventListener("click", () => { app.showAllRows = !app.showAllRows; renderTable(); });
$("export-forecast").addEventListener("click", () => downloadCSV("forecast"));
$("demo-button").addEventListener("click", () => withBusy($("demo-button"), "Готовим демо…", async () => {
  showNotice("Создаём синтетическую историю, обучаем модель и запускаем демонстрационный прогноз.");
  await api("/api/demo", {});
  $("weather-mode").value = "demo";
  await loadState();
  await api("/api/forecast", runPayload(false));
  await loadState();
  resultNotice(app.state.last_run);
}));
all(".import-trigger").forEach((button) => button.addEventListener("click", () => {
  $("import-error").hidden = true;
  $("data-timezone").value = String(timezoneOffset());
  $("import-dialog").showModal();
}));
function closeImport() { if (!app.busy) $("import-dialog").close(); }
$("close-import").addEventListener("click", closeImport);
$("cancel-import").addEventListener("click", closeImport);
$("import-dialog").addEventListener("cancel", (event) => { if (app.busy) event.preventDefault(); });
$("csv-file").addEventListener("change", () => { $("file-label").textContent = $("csv-file").files[0]?.name || "Выберите CSV-файл"; });
$("import-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const file = $("csv-file").files[0];
  if (!file) { $("csv-file").focus(); return; }
  if (!file.name.toLowerCase().endsWith(".csv")) {
    $("import-error").textContent = "Выберите файл в формате CSV. Файл Excel можно предварительно сохранить как CSV UTF-8.";
    $("import-error").hidden = false;
    return;
  }
  withBusy($("import-submit"), "Обучаем модель…", async () => {
    $("import-error").hidden = true;
    await api("/api/import", { csv: await file.text(), filename: file.name, timezone_offset_hours: Number($("data-timezone").value), power_scale: Number($("power-scale").value) });
    $("weather-mode").value = "archive";
    app.showAllRows = false;
    await loadState();
    $("import-dialog").close();
    setView("data");
    showNotice("История загружена, модель подготовлена. Ознакомьтесь с отчётом проверки перед расчётом прогноза.");
  });
});
$("backtest-button").addEventListener("click", () => withBusy($("backtest-button"), "Выполняем тест…", async () => {
  showNotice($("weather-mode").value === "project" ? "Получаем февральские результаты из проекта. Способ получения прогнозов будет указан в отчёте." : "Запускаем ежедневные прогнозы за февраль. Для архивного режима нужны погодные запуски по каждому дню; расчёт может занять несколько минут.");
  const result = await api("/api/backtest", { mode: $("weather-mode").value, hours: 48 });
  await loadState();
  if (!app.state.backtest) { app.state.backtest = result.report || result; renderBacktest(); }
  showNotice((isSavedArchive(app.state.backtest) ? "Сохранённый февральский расчёт загружен из архива. " : "Результаты ретроспективного теста получены. ") + "Проверьте покрытие месяца и ограничения источника погоды.", app.state.backtest.mode === "demo" || app.state.backtest.eligibility?.competition_ready !== true ? "warning" : "");
}));
setView(location.hash.slice(1) || "overview", false);
render();
loadState(true).catch((error) => { showNotice(error.message, "error"); });
