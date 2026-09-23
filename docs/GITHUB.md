# GitHub

## Репозиторий команды

Основной репозиторий: https://github.com/BAITC-Hacks/hack-50369127-build-your-dream

Он закрытый: для клонирования нужен аккаунт с доступом к команде.

```bash
git clone https://github.com/BAITC-Hacks/hack-50369127-build-your-dream.git wind-forecast
cd wind-forecast
```

Установка зависимостей и запуск описаны в README.

В репозиторий включены код, исходные измерения, погодный кэш, обученная модель,
февральские прогнозы и отчёты. Поэтому после клонирования доступен offline replay.
Виртуальные окружения, ключи API, временные файлы и логи исключены через `.gitignore`.
Файлы CSV сохраняют исходные байты при checkout на Windows и Linux.

## Публикация отдельной копии в новом репозитории

Требуются Git и GitHub CLI (`gh`). Эти команды нужны только для отдельной копии,
у которой ещё нет удалённого репозитория `origin`.
Авторизация проходит в браузере; токен не нужно вставлять в исходники или чат.

```bash
gh auth login --hostname github.com --git-protocol https --web
gh repo create wind-forecast-hackathon --private --source=. --remote=origin --push
gh repo view --web
```

Эти команды предназначены для нового репозитория, когда `origin` ещё не настроен.
Если репозиторий с таким именем уже существует, сначала проверьте его содержимое.
Не применяйте `--force` для замены существующей истории.

## Восстановление из Git bundle

Файл `wind-forecast-git.bundle` содержит ветку `main`, все файлы и историю коммитов.
Его можно клонировать без сети:

```bash
git clone --branch main wind-forecast-git.bundle wind-forecast
cd wind-forecast
git remote remove origin
```

Затем выполните команды создания репозитория выше.
Проверка после установки зависимостей по README:

```bash
python -m unittest discover -s tests -v
python -m src.verify
```
