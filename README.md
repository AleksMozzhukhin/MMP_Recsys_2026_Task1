# Домашнее задание 1: рекомендательные системы

**Мозжухин Александр Васильевич**

Датасет: **Amazon Reviews 2014 — Video Games**, полный ratings-only файл.
[Описание данных UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) ·
[CSV](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/ratings_Video_Games.csv).

Основная работа — [experiments.ipynb](experiments.ipynb): данные, EDA, протокол, подбор на validation,
сравнение на test и выводы.

## Результаты

Обучение — на отзывах до 2014 года, test — январь–июль 2014 (3254 пользователя, каталог 3998 товаров).
Настройки выбраны на 2013 годе. Основная метрика — NDCG@10.

| Модель | NDCG@10 | Recall@10 |
|---|---:|---:|
| Fusion: 0.7·FM + 0.3·CustomALS | **0.0753** | **0.1217** |
| Popularity за 30 дней | 0.0680 | 0.1044 |
| FM с возрастом товара | 0.0609 | 0.1014 |
| CustomALS | 0.0572 | 0.0944 |
| iALS | 0.0563 | 0.0914 |
| EASE | 0.0530 | 0.0885 |
| ItemKNN | 0.0477 | 0.0823 |

Главный сигнал в данных — время. Признак возраста товара на момент события даёт FM +0.019 NDCG@10,
затухание по давности даёт CustomALS +0.011. Подробности и доверительные интервалы — в ноутбуке.

## Структура

| Путь | Содержимое |
|---|---|
| [models/](models/) | ItemKNN, iALS, EASE, FM, CustomALS — по файлу на модель; Popularity и Fusion — ориентиры |
| [pipeline/](pipeline/) | загрузка и отбор данных, метрики, подбор на validation и финальный прогон |
| [results/](results/) | все конфигурации validation, выбранные настройки, результаты test, графики |

## Запуск

Python 3.13, CPU. Проверено на macOS arm64; полный прогон занимает около 15 минут.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter execute experiments.ipynb --inplace --timeout=-1
```

CSV скачивается в `.cache/raw/` и проверяется по SHA-256; данные в Git не хранятся.
