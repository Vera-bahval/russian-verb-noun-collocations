"""
Скрипт для анализа коллокаций с использованием Yandex GPT API
"""

import aiohttp
import asyncio
import json
import csv
import pandas as pd
import requests
import re
import argparse
import os
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# Глобальные переменные для API
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def setup_args():
    """Настройка аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Анализ коллокаций с помощью Yandex GPT')
    parser.add_argument('--vars', default='config.json', help='Путь к файлу конфигурации')
    parser.add_argument('--dataset', default='collocations_dataset_final_version_v2.csv', 
                       help='Путь к датасету коллокаций')
    parser.add_argument('--output-dir', default='results', help='Директория для результатов')
    return parser.parse_args()

def load_config(config_path):
    """Загрузка конфигурации из JSON файла"""
    try:
        with open(config_path, "r", encoding="utf-8") as fin:
            config = json.load(fin)
            return config["folder_id"], config["iamToken"]
    except FileNotFoundError:
        print(f"Файл конфигурации {config_path} не найден!")
        print("Создайте файл config.json с содержимым:")
        print('{"folder_id": "your_folder_id", "iamToken": "your_iam_token"}')
        exit(1)
    except KeyError as e:
        print(f"Отсутствует ключ {e} в файле конфигурации")
        exit(1)

def get_metrics(y_true_test, y_pred_test):
    """Вычисление метрик точности, полноты и F1-score"""
    precision = precision_score(y_true_test, y_pred_test)
    recall = recall_score(y_true_test, y_pred_test)
    f1 = f1_score(y_true_test, y_pred_test)
    return precision, recall, f1

def load_dataset(dataset_path):
    """Загрузка и обработка датасета"""
    try:
        df = pd.read_csv(dataset_path, encoding="utf8")
        print(f"Загружен датасет: {len(df)} строк")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Датасет {dataset_path} не найден!")
        exit(1)

def extract_fill_masks(df):
    """Извлечение масок заполнения из датасета"""
    gap_sents = df[df['sentence_with_gap'].notna()]['sentence_with_gap'].to_list()
    fill_sents = df[df['sentence_filled'].notna()]['sentence_filled'].to_list()
    
    true_fill_masks = []
    for gap_sent, fill_sent in zip(gap_sents, fill_sents):
        parts = gap_sent.split('[MASK]')
        part_1 = parts[0]
        part_2 = parts[1] if len(parts) == 2 else ''
        st = len(part_1)
        end = len(fill_sent) - len(part_2)
        ex_col = fill_sent[st:end]
        true_fill_masks.append(ex_col)
    
    return gap_sents, fill_sents, true_fill_masks

async def async_full_mask(prompt, iam_token, folder_id, model_name):
    """Асинхронный запрос к Yandex GPT API"""
    data = {
        "modelUri": "gpt://" + folder_id + "/" + model_name,
        "completionOptions": {"temperature": 1, "maxTokens": 10},
        "messages": [{"role": "user", "text": prompt}]
    }
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["result"]["alternatives"][0]["message"]["text"]
            else:
                return f"Error: API returned status code {response.status} - {await response.text()}"

def get_prompt_template():
    """Получение шаблона промпта для оценки"""
    return """
    Ты - лингвистический эксперт, оценивающий смысловую и грамматическую близость двух отрывков текстов.

    Первый отрывок: "{mask_ans}"
    Второй отрывок: "{true_fill_mask}"

    Оцени, насколько похожи эти отрывки по следующим критериям (0-4 балла):

    4 - отрывки полностью совпадают
    3 - отрывки ОЧЕНЬ близки по смыслу, в них примерно одинаковое количество слов и их можно заменить друг на друга в одном и том же предложении (например, "принять на себя ответственность" и "взять вину на себя")
    2 - отрывки состоят из ОДИНАКОВЫХ слов, но в разных грамматических формах (например, "бить тревогу" и "забить тревогу")
    1 - отрывки отдаленно связаны друг с другом (например, "держать учет" и "беречь деньги")
    0 - отрывки не связаны по смыслу друг с другом (например, "отойти вперед" и "взять верх")
    0 - первый отрывок - одно слово (например, "вспоминать")
    0 - первый отрывок - не словосочетание, а состоит из большого количества слов (например, "всадники летели сломя голову впереди эскадрона и г")
    0 - первый отрывок содержит нерусские вставки (например, "凝聚围绕", "seem" и т.п.)

    В ответе напиши итоговый балл.
    Твой ответ должен содержать только число, больше ничего. НЕ приводи в ответе НИКАКИЕ рассуждения.
    """

def write_to_csv_fill_mask(file_path, row):
    """Запись строки в CSV файл"""
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
        csvfile.flush()

def create_csv_fill_mask(file_path):
    """Создание CSV файла с заголовками"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['collocation', 'true_fill', 'pred_fill', 'evaluation'])
        csvfile.flush()

async def async_run_judge_sents_with_csv(prompt, cols, fill_mask_ans, true_fill_masks, file_path, iam_token, folder_id, model_name):
    """Асинхронная оценка предложений с записью в CSV"""
    judge_results_test = []
    
    for i, (col, mask_ans, true_fill_mask) in enumerate(zip(cols, fill_mask_ans, true_fill_masks)):
        mask_ans = mask_ans.lower()
        formatted_prompt = prompt.format(mask_ans=mask_ans, true_fill_mask=true_fill_mask)
        
        print(f"Обработка {i+1}/{len(cols)}: {col}")
        
        result = await async_full_mask(formatted_prompt, iam_token, folder_id, model_name)
        row = [col, true_fill_mask, mask_ans, result]
        write_to_csv_fill_mask(file_path, row)
        judge_results_test.append(result)
        
        print(f"col: {col}, mask_ans: {mask_ans}, cor: {true_fill_mask}")
        print(f"result: {result}")
        
        # Небольшая задержка для избежания превышения лимитов API
        await asyncio.sleep(0.5)
    
    return judge_results_test

def get_labels_fill_sent(pred_evals):
    """Преобразование оценок в бинарные метки"""
    pred_labels, true_labels = [], []
    
    for pred_eval in pred_evals:
        if pred_eval in [1, 0, 2, 3, 4, '1', '0', '2', '3', '4']:
            pred_eval = int(pred_eval)
            pred_labels.append(1 if pred_eval >= 2 else 0)
            true_labels.append(1)
    
    return true_labels, pred_labels

def get_results_fill_sent(file_path):
    """Получение результатов из CSV файла"""
    df = pd.read_csv(file_path, encoding='utf-8')
    pred_evals = df['evaluation'].to_list()
    true_eval, pred_eval = get_labels_fill_sent(pred_evals)
    pr, rec, f1 = get_metrics(true_eval, pred_eval)
    
    print(f"\nРезультаты для {os.path.basename(file_path)}:")
    print(f"Precision: {pr:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return pr, rec, f1

def get_items(file_path):
    """Получение элементов из файла (если есть дополнительные данные)"""
    cols, sents, preds = [], [], []
    
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return cols, sents, preds
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('@')
            if len(parts) < 3:
                print(f"Неполная строка: {line}")
            if len(parts) >= 3:
                cols.append(parts[0])
                sents.append(parts[1])
                preds.append(parts[2])
    
    print(f"Загружено: {len(cols)} коллокаций, {len(sents)} предложений, {len(preds)} предсказаний")
    return cols, sents, preds

def get_mjs(file_path):
    """Вычисление MJS метрики"""
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return 0
    
    if 'xlsx' in file_path:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    results = df['evaluation'].to_list()
    total_sum = 0
    valid_results = 0
    
    for res in results:
        try:
            json_res = json.loads(res)
            if 'Сумма' in json_res:
                pred_label_value = json_res["Сумма"]
                if isinstance(pred_label_value, int):
                    total_sum += pred_label_value
                    valid_results += 1
                elif isinstance(pred_label_value, str):
                    pattern = r"[0-6]"
                    match = re.search(pattern, pred_label_value)
                    if match:
                        pred_label = int(match.group())
                        total_sum += pred_label
                        valid_results += 1
        except json.JSONDecodeError as e:
            print(f"Ошибка декодирования JSON: {str(res)[:50]}...")
    
    if valid_results > 0:
        mjs_score = total_sum / valid_results
        print(f"\nMJS для {os.path.basename(file_path)}: {mjs_score:.4f}")
        return mjs_score
    else:
        print("Не найдено валидных результатов для MJS")
        return 0

async def main():
    """Основная функция"""
    args = setup_args()
    
    # Загрузка конфигурации
    folder_id, iam_token = load_config(args.vars)
    
    # Загрузка датасета
    df = load_dataset(args.dataset)
    
    # Извлечение масок
    gap_sents, fill_sents, true_fill_masks = extract_fill_masks(df)
    
    # Настройки модели
    model_name = 'yandexgpt-lite/latest'
    prompt_template = get_prompt_template()
    
    # Создание директории для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Пути к файлам результатов
    file_paths = {
        'zero': output_dir / 'zero_shot_results.csv',
        'one': output_dir / 'one_shot_results.csv',
        'few': output_dir / 'few_shot_results.csv'
    }
    
    # Создание CSV файлов
    for file_path in file_paths.values():
        create_csv_fill_mask(file_path)
    
    # Здесь должны быть определены cols_*, preds_* переменные
    # Для демонстрации создам примеры
    print("ВНИМАНИЕ: Используются тестовые данные!")
    print("Необходимо определить переменные cols_llama_zero, preds_llama_zero и т.д.")
    
    # Пример данных (замените на реальные)
    cols_example = ['тест коллокация'] * min(3, len(true_fill_masks))
    preds_example = ['тест предсказание'] * min(3, len(true_fill_masks))
    true_fill_masks_example = true_fill_masks[:min(3, len(true_fill_masks))]
    
    # Запуск оценки (только для примера с ограниченными данными)
    if len(true_fill_masks_example) > 0:
        print("\nЗапуск тестовой оценки...")
        results = await async_run_judge_sents_with_csv(
            prompt_template, 
            cols_example, 
            preds_example, 
            true_fill_masks_example, 
            file_paths['zero'],
            iam_token,
            folder_id,
            model_name
        )
        
        # Анализ результатов
        print("\nАнализ результатов:")
        get_results_fill_sent(file_paths['zero'])

if __name__ == "__main__":
    # Запуск асинхронной основной функции
    asyncio.run(main())
