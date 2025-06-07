import torch
import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import re

def write_to_file(col, pred, output_dir):
    """Записывает результат в файл"""
    with open(output_dir, 'a', encoding='utf-8') as f:
        f.write(col + '@' + pred + '\n')
        f.flush()

def make_fs_prompt(prompt, samples):
    """Создает промпт с примерами для few-shot обучения"""
    answer = prompt + " Ниже приведены примеры."
    for sample in samples:
        answer += "\n\nСочетание: {source}\nОтвет: {target}".format(**sample)
    return answer

def get_answer_qwen(model, tokenizer, system_prompt, user_text):
    """Получает ответ от модели Qwen"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    # Применяем chat template для Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Токенизируем
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем только новые токены
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def clean_prediction(pred):
    """Очищает предсказание, оставляя только цифры 0 или 1"""
    # Ищем первую цифру 0 или 1
    match = re.search(r'[01]', pred)
    if match:
        return match.group()
    return "0"  # Возвращаем 0 по умолчанию если не найдено

def compute_metrics(true_labels, pred_labels):
    """Вычисляет метрики precision, recall, f1"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', pos_label='1'
    )
    return precision, recall, f1

def load_qwen_model():
    """Загружает модель Qwen"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"Загрузка модели {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Модель загружена успешно!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None

def test_qwen_model():
    """Тестирует модель Qwen на небольшом наборе данных"""
    model, tokenizer = load_qwen_model()
    
    if model is None or tokenizer is None:
        return None, None, None
    
    # Системный промпт
    system_prompt_classification = '''Ниже дано сочетание глагола и существительного.
Определи, является ли оно коллокацией. Ответь только 1 (если сочетание - коллокация) или только 0 (если сочетание - не коллокация).
В твоем ответе должна быть ТОЛЬКО соответствующая цифра, больше ничего.'''

    # Few-shot примеры
    sample_few_shot_class = [
        {"source": "выносить выговор", "target": "1"},
        {"source": "выражать интересы", "target": "0"},
        {"source": "гнать в шею", "target": "1"},
        {"source": "жить без забот", "target": "0"},
        {"source": "держаться особняком", "target": "1"},
        {"source": "закусить бутербродом", "target": "0"},
    ]
    
    system_prompt_few_shot = make_fs_prompt(system_prompt_classification, sample_few_shot_class)
    
    # Тестовые данные
    test_cols = ["делать работу", "бить баклуши", "читать книгу", "взять верх", "поднять тост"]
    test_labels = ["0", "1", "0", "1", "1"]
    
    print("\nТестирование модели:")
    results = []
    
    for col, expected in zip(test_cols, test_labels):
        print(f"\nОбработка: {col}")
        result = get_answer_qwen(model, tokenizer, system_prompt_few_shot, col)
        cleaned_result = clean_prediction(result)
        results.append(cleaned_result)
        
        print(f"Сочетание: {col}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: {cleaned_result}")
        print(f"Ожидаемый: {expected}")
        print(f"Правильно: {'✓' if cleaned_result == expected else '✗'}")
        print("-" * 50)
    
    # Вычисляем метрики
    if len(results) == len(test_labels):
        precision, recall, f1 = compute_metrics(test_labels, results)
        print(f"\nМетрики на тестовых данных:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
    
    return model, tokenizer, system_prompt_few_shot

def process_dataset(model, tokenizer, dataset_path, output_dirs):
    """Обрабатывает весь датасет"""
    try:
        # Загружаем данные
        print(f"Загрузка данных из {dataset_path}...")
        df = pd.read_csv(dataset_path, encoding="utf-8")
        cols = df['collocation'].to_list()
        print(f"Загружено {len(cols)} сочетаний")

        # Определяем промпты
        base_prompt = '''Ниже дано сочетание глагола и существительного.
Определи, является ли оно коллокацией. Ответь только 1 (если сочетание - коллокация) или только 0 (если сочетание - не коллокация).
В твоем ответе должна быть ТОЛЬКО соответствующая цифра, больше ничего.'''

        one_shot_examples = [
            {"source": "выносить выговор", "target": "1"},
            {"source": "выражать интересы", "target": "0"}
        ]

        few_shot_examples = [
            {"source": "выносить выговор", "target": "1"},
            {"source": "выражать интересы", "target": "0"},
            {"source": "гнать в шею", "target": "1"},
            {"source": "жить без забот", "target": "0"},
            {"source": "держаться особняком", "target": "1"},
            {"source": "закусить бутербродом", "target": "0"}
        ]

        # Создаем промпты
        prompts = [
            base_prompt,  # zero-shot
            make_fs_prompt(base_prompt, one_shot_examples),  # one-shot
            make_fs_prompt(base_prompt, few_shot_examples)   # few-shot
        ]
        
        regimes = ['zero-shot', 'one-shot', 'few-shot']

        # Обрабатываем каждый режим
        for prompt_idx, (prompt, regime, output_dir) in enumerate(zip(prompts, regimes, output_dirs), 1):
            if output_dir is None:
                print(f"Пропускаем {regime} - не указан output_dir")
                continue
                
            print(f"\nОбработка {regime} (prompt {prompt_idx}/{len(prompts)})...")
            print(f"Результаты будут сохранены в: {output_dir}")
            
            # Очищаем файл результатов
            with open(output_dir, 'w', encoding='utf-8') as f:
                f.write("")

            correct_predictions = 0
            total_predictions = 0
            
            for i, col in enumerate(tqdm(cols, desc=f"{regime}", leave=False)):
                try:
                    pred = get_answer_qwen(model, tokenizer, prompt, col)
                    cleaned_pred = clean_prediction(pred)
                    write_to_file(col, cleaned_pred, output_dir)
                    
                    # Если есть истинные метки, можно подсчитать точность
                    if 'label' in df.columns:
                        true_label = str(df.iloc[i]['label'])
                        if cleaned_pred == true_label:
                            correct_predictions += 1
                        total_predictions += 1
                    
                except Exception as e:
                    print(f"Ошибка при обработке '{col}': {e}")
                    write_to_file(col, "0", output_dir)

            print(f"{regime} завершен!")
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"Точность на {regime}: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    argument_parser = ArgumentParser(description="Классификация коллокаций с помощью Qwen2.5-7B-Instruct")
    argument_parser.add_argument("--dataset", default="collocations_dataset_final_version.csv", 
                               help="Путь к CSV файлу с данными")
    argument_parser.add_argument("--output_dir_zero_shot", 
                               help="Путь для сохранения результатов zero-shot")
    argument_parser.add_argument("--output_dir_one_shot", 
                               help="Путь для сохранения результатов one-shot")
    argument_parser.add_argument("--output_dir_few_shot", 
                               help="Путь для сохранения результатов few-shot")
    argument_parser.add_argument("--test_only", action="store_true", 
                               help="Запустить только тест модели")

    args = argument_parser.parse_args()

    # Если запрошен только тест
    if args.test_only:
        test_qwen_model()
        return

    # Загружаем модель
    model, tokenizer = load_qwen_model()
    if model is None or tokenizer is None:
        print("Не удалось загрузить модель")
        return

    # Проверяем, что указан хотя бы один output_dir
    output_dirs = [args.output_dir_zero_shot, args.output_dir_one_shot, args.output_dir_few_shot]
    if all(output_dir is None for output_dir in output_dirs):
        print("Сначала протестируем модель...")
        test_qwen_model()
        return

    # Обрабатываем датасет
    process_dataset(model, tokenizer, args.dataset, output_dirs)
    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()