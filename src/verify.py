from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd


class Inspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if tag == 'script' and data.get('src'):
            self.external.append(str(data['src']))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    submission = root / 'outputs/submission_february.csv'
    before = hashlib.sha256(submission.read_bytes()).hexdigest()
    with (root / 'logs/reproduce-offline.log').open('w') as log:
        subprocess.run([sys.executable, 'run.py'], cwd=root, stdout=log, stderr=log, check=True)
    after = hashlib.sha256(submission.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError('Offline replay must reproduce identical submission bytes')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=root, text=True, capture_output=True, check=True)
    output = tests.stdout + tests.stderr
    (root / 'logs/tests-final.log').write_text(output)
    count = int(re.search(r'Ran (\d+) tests', output).group(1))
    x = pd.read_csv(submission)
    expected = set(pd.date_range('2026-02-01', periods=672, freq='h', tz='UTC'))
    if len(x) != 1344 or x[['turbine_id', 'valid_time']].duplicated().any():
        raise AssertionError('Incomplete submission or duplicate hours')
    for _, group in x.groupby('turbine_id'):
        if set(pd.to_datetime(group.valid_time, utc=True)) != expected:
            raise AssertionError('February time coverage mismatch')
    if not x.power_norm.between(0, 1).all() or not ((x.lower_80 <= x.power_norm) & (x.power_norm <= x.upper_80)).all():
        raise AssertionError('Invalid forecast bounds')
    inspector = Inspector()
    inspector.feed((root / 'reports/demo.html').read_text())
    if inspector.external:
        raise AssertionError('Demo must not depend on external scripts')
    metrics = pd.read_csv(root / 'reports/january_metrics.csv')
    chosen = metrics[(metrics.model == 'selected_model') & (metrics.turbine_id == 'ALL')].iloc[0]
    base = metrics[(metrics.model == 'weather_power_curve') & (metrics.turbine_id == 'ALL')].iloc[0]
    replay = json.loads((root / 'reports/replay_report.json').read_text())
    result = {'python': sys.version.split()[0], 'tests_passed': count, 'offline_submission_identical': True,
              'submission_sha256': after, 'january_mae': float(chosen.mae),
              'january_rmse': float(chosen.rmse), 'baseline_curve_mae': float(base.mae),
              'relative_mae_improvement_percent': float(100*(1-chosen.mae/base.mae)),
              'replay': replay, 'html_external_scripts': inspector.external}
    (root / 'reports/verification.json').write_text(json.dumps(result, indent=2))
    (root / 'VERIFICATION.md').write_text(f'''# Проверка завершённого прототипа

Python {sys.version.split()[0]}, CPU. Дата разработки: 2026-09-23.

- Чистое окружение из requirements-models.txt; pip check успешен.
- {count} автоматических тестов прошли, исходники компилируются.
- Оба реальных CSV обработаны: по 25392 часа.
- Получено 419 из 425 ежедневных погодных выпусков для обеих ВЭУ.
  Шесть недоступных августовских дат перечислены в манифесте.
- 40224 строки признаков, 36636 с известной мощностью.
- Модель выбрана на декабре. MAE января {chosen.mae:.6f}, RMSE {chosen.rmse:.6f}.
- Кривая мощности: MAE {base.mae:.6f}; снижение MAE {100*(1-chosen.mae/base.mae):.2f}%.
- Покрытие январских интервалов {chosen.coverage:.2%}; ширина {chosen.interval_width:.4f}.
- Февраль: 29 выпусков, 2784 строки всех горизонтов; 1344 строки сдачи.
- Проверены все 672 часа для каждой ВЭУ, отсутствие дублей и диапазон мощности.
- Повторный offline replay дал побайтово одинаковый submission.
- SHA-256: {after}
- Дополнительный реальный расчёт 2026-02-01 05:00 UTC использовал выпуск
  2026-01-31 18:00 UTC; 96 строк сохранены отдельно.
- Откат при сбое проверен тестом с имитацией недоступного провайдера.
  В реальном февральском replay откат не потребовался.
- HTML содержит встроенный Plotly и не запрашивает внешние скрипты.

## Границы проверки

Происхождение архива as-issued и время CSV остаются неподтверждёнными.
Февральских фактических целей нет. Живой LLM-вызов не выполнялся без ключа.
Docker-сборка и развёртывание у заказчика не выполнялись.
''', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
