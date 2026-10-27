"""Explicit, small research corpus, never visitor data or product accuracy evidence."""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SOURCES = [
    ("pushkin", "Капитанская дочка (Пушкин)/1978 (СО)", "А. С. Пушкин, 1836"),
    ("chekhov", "Хамелеон (Чехов)/Версия 2", "А. П. Чехов, 1884"),
]


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.current = []
        self.items = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.depth += 1
        if tag == "sup":
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "sup":
            self.skip = max(0, self.skip - 1)
        if tag == "p" and self.depth:
            self.depth -= 1
            if not self.depth:
                self.items.append("".join(self.current))
                self.current = []

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.current.append(data)


# These are authored by the Codex assistant for this engineering task, not human samples.
AI_TEXTS = [
    "Утром мастерская открылась раньше обычного. На столе лежало письмо, которое никто не помнил. Анна прочитала первые строки и остановилась: знакомый почерк напоминал о лете у реки. За окном тихо стучал дождь, а часы показывали без десяти восемь. Она решила сначала поставить чайник, затем позвонить брату. Но телефон разрядился. Пришлось искать зарядку среди инструментов и старых фотографий. В ящике обнаружилась ещё одна записка. На этот раз в ней был только адрес. Анна положила письмо в карман и вышла на улицу. В городе начинался обыкновенный рабочий день, хотя для неё он уже перестал быть обыкновенным.",
    "Когда поезд остановился у маленькой станции, пассажиры не сразу заметили перемену погоды. Только мальчик у окна увидел первый снег. Он попросил отца выйти на платформу, но тот продолжал искать билет. Проводница объяснила, что стоянка продлится пять минут. За это время они успели купить горячий чай и поговорить с продавцом газет. Человек рассказал о старом мосте за посёлком, который недавно отремонтировали. Мальчик внимательно слушал и представлял, как по мосту идут люди. Потом раздался свисток. Отец взял его за руку, и они вернулись в вагон. Белые хлопья ещё долго кружились за стеклом.",
    "Вечером библиотекарь обнаружил на полке книгу без инвентарного номера. Обложка была потёртой, но страницы сохранились хорошо. Между ними лежал рисунок маленького дома с зелёной крышей. Внизу было написано имя, которое сотрудник раньше не встречал. Он решил не отправлять книгу в хранилище, а показать её посетителям на следующий день. Первой пришла пожилая женщина. Она узнала дом и рассказала, что когда-то жила рядом. По её словам, рисунок сделал школьный учитель перед самым переездом. После этого рассказа библиотекарь понял, что обычная находка может стать началом отдельной выставки о памяти города.",
    "Небольшой команде полезно начинать разработку сервиса с описания конкретной задачи пользователя. Сначала нужно выяснить, какие действия выполняются вручную и где возникают ошибки. Затем можно составить короткий список требований и определить границы первого выпуска. Важно отдельно обсудить хранение данных, права сотрудников и восстановление после сбоев. Интерфейс должен показывать не только успешный результат, но и состояние ожидания, отказа или повторной попытки. Проверки стоит проводить на реалистичных данных без персональной информации. После запуска полезно измерять фактическое использование функций, а не количество добавленных экранов. Такой порядок не гарантирует успех, однако помогает вовремя обнаружить неверные предположения.",
    "Для проверки новой информационной системы необходимо определить воспроизводимый протокол испытаний. Каждый сценарий должен содержать исходные условия, входные данные и ожидаемое наблюдаемое поведение. Отдельное внимание следует уделить отрицательным случаям: отсутствию доступа, повреждённому файлу и превышению времени обработки. Набор испытаний рекомендуется разделять на функциональные и нагрузочные проверки. Результаты нужно сопровождать описанием среды и версии программного обеспечения. Сравнение разных выпусков имеет смысл только при одинаковых условиях. Если данных недостаточно, вывод формулируется как ограниченное наблюдение, а не как подтверждение универсальной надёжности решения. Повторный запуск должен сохранять историю предыдущих измерений.",
    "Выбор программы для рабочего компьютера начинается с понимания условий её использования. Следует проверить совместимость с операционной системой, требования к памяти и способы установки обновлений. Для небольшой организации удобен понятный интерфейс и возможность быстро получить поддержку. Перед покупкой полезно испытать основную задачу на собственном тестовом примере, не загружая конфиденциальные документы в неизвестный сервис. В описании продукта должны быть ясно указаны ограничения пробного режима и условия получения новых версий. Стоимость сама по себе не показывает качество. Гораздо важнее, насколько предсказуемо инструмент выполняет нужную работу и что происходит при ошибке.",
]


def seed(output):
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for author, title, citation in SOURCES:
        params = urlencode(
            {"action": "parse", "page": title, "prop": "text|revid", "format": "json"}
        )
        request = Request(
            f"https://ru.wikisource.org/w/api.php?{params}",
            headers={
                "User-Agent": "KA-Text-Detector/0.1 local research (kodalexandrova.ru)"
            },
        )
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)["parse"]
        parser = Paragraphs()
        parser.feed(payload["text"]["*"])
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in parser.items]
        paragraphs = [
            p
            for p in paragraphs
            if len(p) > 250
            and not re.search(
                r"1281|лиценз|произведение находится|текст доступен|Оригинал:|Источник:",
                p,
                re.I,
            )
        ]
        for index, text in enumerate(paragraphs[:6]):
            rows.append(
                {
                    "id": f"{author}-{index}",
                    "text": text,
                    "label": "human",
                    "genre": "historical-prose",
                    "language": "ru",
                    "source": f"https://ru.wikisource.org/w/index.php?oldid={payload['revid']}",
                    "sourceGroup": author,
                    "license": "public-domain-work; Wikimedia transcription attribution retained",
                    "provenance": citation,
                    "generator": "human",
                    "edited": False,
                    "addedAt": now,
                    "split": "test",
                }
            )
    for index, text in enumerate(AI_TEXTS):
        rows.append(
            {
                "id": f"codex-{index}",
                "text": text,
                "label": "ai",
                "genre": "generated-prose" if index < 3 else "generated-informational",
                "language": "ru",
                "source": "KA engineering session 2026-10-27",
                "sourceGroup": f"codex-prompt-{index}",
                "license": "owner-authorized task output; research only",
                "provenance": "Written by the Codex assistant as a labelled synthetic sample; not passed off as human",
                "generator": "Codex-session-2026-10-27",
                "edited": False,
                "addedAt": now,
                "split": "test",
            }
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "samples": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "representative": False,
            }
        )
    )


if __name__ == "__main__":
    import sys

    seed(sys.argv[1])
