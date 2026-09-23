from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.utils.common import atomic_write, load_config


def style(fig: go.Figure, title: str) -> None:
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#121c2c", plot_bgcolor="#121c2c",
                      font=dict(family="Arial", color="#dfe9f4"), height=420,
                      margin=dict(l=55, r=25, t=65, b=45), yaxis=dict(range=[0, 1], title="Нормализованная мощность"),
                      legend=dict(orientation="h", y=-0.18))


def build_report(root: Path) -> Path:
    report = json.loads((root / "reports/training_report.json").read_text())
    replay = json.loads((root / "reports/replay_report.json").read_text())
    metrics = pd.read_csv(root / "reports/january_metrics.csv")
    january = pd.read_csv(root / "reports/january_predictions.csv")
    forecast = pd.read_csv(root / "outputs/forecast_all_issues.csv")
    selection = pd.read_csv(root / "reports/validation_leaderboard.csv")
    chosen = metrics[(metrics.model == "selected_model") & (metrics.turbine_id == "ALL")].iloc[0]
    base = metrics[(metrics.model == "weather_power_curve") & (metrics.turbine_id == "ALL")].iloc[0]
    improvement = 100 * (1 - chosen.mae / base.mae)
    bars = go.Figure()
    totals = metrics[(metrics.turbine_id == "ALL")]
    bars.add_bar(x=totals.model, y=totals.mae, name="MAE", marker_color=["#35d5a7", "#647b99", "#6ea1ec", "#9b87cf"])
    bars.update_layout(template="plotly_dark", paper_bgcolor="#121c2c", plot_bgcolor="#121c2c", height=320,
                       margin=dict(l=55, r=25, t=35, b=55), yaxis_title="MAE · ниже лучше")
    history = go.Figure()
    for turbine in ["T1", "T2"]:
        sub = january[(january.turbine_id == turbine) & (january.lead_hours <= 24)].sort_values("valid_time")
        history.add_scatter(x=sub.valid_time, y=sub.target_power_norm, name=f"{turbine} факт", line=dict(width=1, color="#9cacc1"), visible=turbine == "T1")
        history.add_scatter(x=sub.valid_time, y=sub.prediction, name=f"{turbine} прогноз", line=dict(width=1.6, color="#35d5a7"), visible=turbine == "T1")
    history.update_layout(updatemenus=[dict(buttons=[dict(label=t, method="update", args=[{"visible": [t == "T1", t == "T1", t == "T2", t == "T2"]}]) for t in ["T1", "T2"]], x=1, y=1.15)])
    style(history, "Январь 2026 · независимая проверка, горизонт 1–24 часа")
    pred = go.Figure()
    groups = list(forecast.groupby(["issue_time", "turbine_id"], sort=True))
    buttons = []
    for index, ((issue, turbine), sub) in enumerate(groups):
        sub = sub.sort_values("valid_time")
        pred.add_scatter(x=sub.valid_time, y=sub.lower_80, name="Нижняя граница", line=dict(width=0), showlegend=False, visible=index == 0)
        pred.add_scatter(x=sub.valid_time, y=sub.upper_80, name="Эмпирический интервал", line=dict(width=0),
                         fill="tonexty", fillcolor="rgba(53,213,167,0.17)", visible=index == 0)
        pred.add_scatter(x=sub.valid_time, y=sub.power_norm, name="Прогноз мощности", line=dict(color="#35d5a7", width=2.5), visible=index == 0)
        visibility = [False] * (len(groups) * 3)
        visibility[index * 3:index * 3 + 3] = [True] * 3
        buttons.append(dict(label=f"{issue[:10]} · {turbine}", method="update", args=[{"visible": visibility}]))
    pred.update_layout(updatemenus=[dict(buttons=buttons, x=1, y=1.18)])
    style(pred, "Ежедневный прогноз на 48 часов · выберите дату выпуска и турбину")
    table = metrics[metrics.turbine_id != "ALL"].round(4).fillna("").to_html(index=False, border=0, classes="table")
    validation = selection.round(4).to_html(index=False, border=0, classes="table")
    diagrams = [fig.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False), config={"responsive": True, "displaylogo": False})
                for i, fig in enumerate([bars, history, pred])]
    document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wind Forecast · HackAlem AI</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0a1220;color:#dfe9f4;font:16px/1.6 Arial,sans-serif}}
main{{max-width:1240px;margin:auto;padding:36px 24px}}header{{border-bottom:1px solid #27364b;padding-bottom:24px}}
.label{{color:#35d5a7;letter-spacing:3px;font-size:12px}}h1{{font-size:42px;line-height:1.15;margin:15px 0}}h2{{font-size:22px}}
.muted{{color:#9bacc3}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}
.card,section{{background:#121c2c;border:1px solid #27364b;border-radius:14px;padding:20px;margin-bottom:18px}}
.value{{font-size:32px;color:#35d5a7}}.note{{border-left:3px solid #e2bb6a;padding:12px 20px;background:#1f2430}}
.table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #27364b}}
.scroll{{overflow-x:auto}}code{{color:#80c0ff}}a{{color:#80c0ff}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:29px}}}}
</style></head><body><main><header><div class="label">HACKALEM AI / WIND FORECAST</div><h1>Прогноз ветровой генерации<br>с проверяемой историей расчёта</h1>
<p class="muted">Две турбины · архивные выпуски ECMWF IFS · почасовой прогноз на 48 часов</p></header>
<div class="cards"><div class="card"><div class="muted">MAE на январе</div><div class="value">{chosen.mae:.4f}</div><small>в долях номинальной мощности</small></div>
<div class="card"><div class="muted">Снижение MAE</div><div class="value">{improvement:.1f}%</div><small>относительно кривой мощности</small></div>
<div class="card"><div class="muted">Выпусков февраля*</div><div class="value">{replay['issue_count']}</div><small>*включая 31 января</small></div>
<div class="card"><div class="muted">Часов февраля</div><div class="value">{replay['february_hours']}</div><small>для каждой из двух турбин</small></div></div>
<p class="note">Часовой пояс телеметрии UTC и задержка публикации погоды 8 часов приняты как предположения.
Происхождение архива как исходных операционных выпусков требует подтверждения. Метрики относятся к январю; фактических данных февраля нет.</p>
<section><h2>Сравнение на независимом месяце</h2><p class="muted">Выбор модели завершён на декабре. Январь использован для оценки без подбора гиперпараметров.</p>{diagrams[0]}</section>
<section>{diagrams[1]}</section><section>{diagrams[2]}<p class="muted">Интервалы рассчитаны по историческим остаткам с целевым покрытием 80%.
Фактическое покрытие в январе: {chosen.coverage:.1%}; средняя ширина: {chosen.interval_width:.3f}. Это не гарантия покрытия в феврале.</p></section>
<section><h2>Как выполняется ежедневный цикл</h2><p>План → выбор доступного выпуска → проверка погоды → прогноз → анализ скачков и ширины интервалов → сохранение результата.</p>
<p class="muted">При сбое проверяется более ранний выпуск. Повтор неизменных входов использует сохранённый прогноз. Все решения записываются в SQLite.
Обычный режим использует явную политику; дополнительный LLM-планировщик подключается флагом <code>--llm</code> и API-ключом.</p></section>
<section><h2>Метрики по турбинам и горизонтам</h2><div class="scroll">{table}</div></section>
<section><h2>Выбор на декабре</h2><p>Выбрана модель: <b>{html.escape(report['selected_model'])}</b>.</p><div class="scroll">{validation}</div></section>
<footer class="muted">Источник погоды: Open-Meteo / ECMWF. Результаты получены на предоставленных CSV ВЭУ.
Самодостаточный отчёт: интернет для просмотра графиков не требуется.</footer></main></body></html>'''
    path = root / "reports/demo.html"
    atomic_write(path, document)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    path = build_report(load_config(args.config)["_root"])
    print(path)


if __name__ == "__main__":
    main()
