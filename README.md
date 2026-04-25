# MLOps Homework 5: Воспроизводимые эксперименты

## Цель проекта
Демонстрация MLOps-контура с DVC и MLflow.

## Структура проекта
├── data/ # Данные (версионируются через DVC)

├── models/ # Сохраненные модели

├── src/ # Скрипты подготовки и обучения

├── params.yaml # Параметры экспериментов

├── dvc.yaml # DVC-пайплайн

├── HW5_MLOps_Шумакова_Валентина.ipynb # Блокнот с заданием

└── requirements.txt # Зависимости

## Работа выполнена в двух средах:

1. **Git-репозиторий** — создана правильная структура проекта, 
   написаны скрипты prepare.py и train.py, настроены dvc.yaml и params.yaml.
   Ссылка: [ссылка на репозиторий]

2. **Colab Notebook** — выполнены практические запуски DVC пайплайна,
   MLflow эксперимента и Feast. Все логи сохранены.

## Как запустить проект

### 1. Клонировать репозиторий
git clone https://github.com/valentinashumakova01-ctrl/hw5_mlops_Shumakova_Valentina.git

cd hw5_mlops_Shumakova_Valentina

### 2. Создать и активировать виртуальное окружение
python -m venv venv
venv\Scripts\activate   # Windows

### 3. Установить зависимости
pip install -r requirements.txt

### 4. Воспроизвести весь пайплайн (подготовка данных + обучение)
dvc repro

### 5. Запустить MLflow UI для просмотра экспериментов
mlflow ui --host 0.0.0.0 --port 5000

## Краткое описание пайплайна

Пайплайн состоит из двух стадий, описанных в dvc.yaml:

1. prepare — Подготовка данных
Вход: src/prepare.py

Параметры: prepare.test_size=0.2, prepare.random_state=42

Выход: data/raw.csv, data/processed/ (train/test split)

Что делает: Загружает датасет Iris, разделяет на обучающую (80%) и тестовую (20%) выборки, сохраняет в CSV

2. train — Обучение модели
Зависимости: src/train.py, data/processed/, params.yaml

Параметры: train.n_estimators=100, train.max_depth=3

Выход: models/model.pkl, metrics.json

Что делает: Обучает Random Forest Classifier, логирует метрики (accuracy, precision, recall) в MLflow, сохраняет модель

Результат: Accuracy на тестовой выборке = 1.0 (100%) — ожидаемо для простого датасета Iris.


## Где смотреть MLflow UI

### Локальный запуск (после выполнения mlflow ui)
Откройте браузер и перейдите по адресу: http://localhost:5000

Вы увидите эксперимент iris_experiment

Нажмите на него, чтобы просмотреть:

Параметры: n_estimators, max_depth, random_state, model_type

Метрики: accuracy, precision, recall

Артефакты: model.pkl (сохранённая модель)

### В GitHub Codespaces
Если вы запускаете проект в Codespace (как в процессе выполнения ДЗ):

После запуска mlflow ui --host 0.0.0.0 --port 5000

Нажмите на иконку Ports в нижней панели VS Code

Найдите порт 5000 и нажмите на значок "Open in Browser"
