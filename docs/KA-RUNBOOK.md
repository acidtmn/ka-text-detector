# Локальная инструкция

Репозитории `kodalexandrova.ru` и `ka-text-detector` лежат рядом. Команды ниже — из папки сайта, не production. `.env.detector` игнорируется Git, содержит `DETECTOR_API_URL=http://ka-text-detector:8020` и случайный `DETECTOR_SERVICE_TOKEN` длиной ≥32 символов.

```sh
docker compose -f compose.yaml -f compose.detector.yaml build ka-text-detector
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-prepare run --rm detector-prepare
# Опциональный более тяжёлый алгоритм v1:
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-prepare run --rm detector-prepare python -m ka_detector.prepare --versions v2 v3 v4 --with-v1
docker compose -f compose.yaml -f compose.detector.yaml up -d ka-text-detector app
docker compose -f compose.yaml -f compose.detector.yaml exec app pnpm prisma migrate deploy
docker compose -f compose.yaml -f compose.detector.yaml exec app node scripts/setup-text-detector-local.mjs
```

Сеть overlay `10.251.38.0/24` проверена на этой машине; на другой проверь пересечения. Host port детектора не опубликован, прежний порт сайта 3217 сохранён. Сайт должен быть пересобран с кодом интеграции. Платежи/лицензии не изменяются.

Лаборатория: `http://localhost:3217/tools/text-lab`, детектор: `/tools/ai-detector`, панель: `/admin/text-detector`, история: `/account/text-checks`. Вход — существующий локальный механизм студии.

Остановка только сервиса: `docker compose -f compose.yaml -f compose.detector.yaml stop ka-text-detector`. Не использовать `down -v`, prune или очистку пользовательского storage. Модели — в `models`, результаты исследования — в `runs`.

## Исследование

```sh
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-training run --rm detector-train python -m ka_detector.seed_corpus /runs/validation.jsonl
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-training run --rm detector-train python -m ka_detector.research validate /runs/validation.jsonl
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-training run --rm detector-train python -m ka_detector.research evaluate /runs/validation.jsonl --models upstream-v2 upstream-v3 upstream-v4 --output /runs/evaluation.json
docker compose -f compose.yaml -f compose.detector.yaml --profile detector-training run --rm detector-train python -m ka_detector.research train-smoke --output /runs/gradient-unique-run
```

Seed-корпус явно получает общественное достояние через сеть только в исследовательском контейнере; inference Интернет не получает. Это маленький тест, не обучающий датасет. Для прикрепления evaluation в prepare-контейнер монтируется `runs:/runs:ro`, затем `python -m ka_detector.register attach-evaluation /runs/evaluation.json`. Fingerprint обязан совпасть с весами.

Полное обучение: свой JSONL в `corpora/private`, команда `research train /corpora/private/corpus.jsonl --output /runs/candidate-unique`. Не менее 200 train / 40 validation / 60 test, два класса в каждой части: лишь технический предохранитель, не достаточное условие product-quality. Seed 27, CPU batch 2, максимум две эпохи / 10 минут. Незавершённый запуск не регистрируется.

Оператор регистрирует кандидата `register candidate /runs/candidate-unique --id ka-candidate-name` с RW mount моделей. Smoke отклоняется. Далее нужны отдельный evaluation report, ручной выбор в админке и проверка прав; автоматического продвижения нет.

JSONL: `id,text,label,genre,language,source,sourceGroup,license,provenance,generator,edited,addedAt,split`. label=`human/ai/mixed`, language=`ru`, split=`train/validation/test`. Источники разделяются до нарезки; exact/near duplicates между split запрещены. Тексты посетителей не используются для обучения.

## Проверки и безопасные артефакты

Python: `python -m pytest ka_tests -q -p no:cacheprovider` внутри контейнера с моделями. HTTP-тест требует живого API и env-secret. Сайт: `tests/api/text-detector.integration.test.ts`, отдельная локальная `*_store_test` БД. Браузер: `KA_DETECTOR_REAL_TESTS=true`, иначе явный skip, а не мок модели.

**Никогда не монтировать storage, проект или домашний каталог в outputDir тестового раннера.** Результаты Playwright пишутся во временную файловую систему тестового контейнера; после завершения копируются из остановленного контейнера в отдельный новый каталог.
