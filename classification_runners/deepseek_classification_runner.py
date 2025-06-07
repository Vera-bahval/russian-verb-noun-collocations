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
    answer = prompt + "\n\nПримеры:"
    for sample in samples:
        answer += "\n\nСочетание: {source}\nОтвет: {target}".format(**sample)
    answer += "\n\nТеперь ответь на следующий вопрос:"
    return answer

def create_deepseek_prompt(system_prompt, user_text, examples=None):
    """Создает промпт для базовой модели DeepSeek без chat template"""
    if examples:
        prompt = make_fs_prompt(system_prompt, examples)
        prompt += f"\n\nСочетание: {user_text}\nОтвет:"
    else:
        prompt = f"{system_prompt}\n\nСочетание: {user_text}\nОтвет:"
    
    return prompt

def get_answer_deepseek(model, tokenizer, system_prompt, user_text, examples=None):
    """Получает ответ от базовой модели DeepSeek"""
    # Создаем промпт для базовой модели
    prompt = create_deepseek_prompt(system_prompt, user_text, examples)
    
    # Токенизируем
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Ограничиваем количество новых токенов
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Декодируем только новые токены
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def clean_prediction(pred):
    """Очищает предсказание, оставляя только цифры 0 или 1"""
    # Убираем лишние символы и пробелы
    pred = pred.strip()
    
    # Ищем первую цифру 0 или 1
    match = re.search(r'[01]', pred)
    if match:
        return match.group()
    
    # Попробуем найти слова "ноль"/"один" или "да"/"нет"
    pred_lower = pred.lower()
    if any(word in pred_lower for word in ["да", "yes", "один", "коллокация", "правильно"]):
        return "1"
    elif any(word in pred_lower for word in ["нет", "no", "ноль", "не коллокация", "неправильно"]):
        return "0"
    
    return "0"  # Возвращаем 0 по умолчанию

def compute_metrics(true_labels, pred_labels):
    """Вычисляет метрики precision, recall, f1"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', pos_label='1'
    )
    return precision, recall, f1

def load_deepseek_model():
    """Загружает базовую модель DeepSeek"""
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    
    print(f"Загрузка модели {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
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

def test_deepseek_model():
    """Тестирует модель DeepSeek на небольшом наборе данных"""
    model, tokenizer = load_deepseek_model()
    
    if model is None or tokenizer is None:
        return None, None, None
    
    # Системный промпт для базовой модели
    system_prompt_classification = '''Задача: определить, является ли сочетание глагола и существительного коллокацией.
Коллокация - это устойчивое словосочетание, где слова часто употребляются вместе.
Ответь только цифрой: 1 если сочетание является коллокацией, 0 если не является.'''

    # Few-shot примеры
    sample_few_shot_class = [
        {"source": "выносить выговор", "target": "1"},
        {"source": "выражать интересы", "target": "0"},
        {"source": "гнать в шею", "target": "1"},
        {"source": "жить без забот", "target": "0"},
        {"source": "держаться особняком", "target": "1"},
        {"source": "закусить бутербродом", "target": "0"},
    ]
    
    # Тестовые данные
    test_cols = ["делать работу", "бить баклуши", "читать книгу", "взять верх", "поднять тост"]
    test_labels = ["0", "1", "0", "1", "1"]
    
    print("\nТестирование модели:")
    print("="*60)
    
    # Тестируем zero-shot
    print("\n--- ZERO-SHOT ---")
    zero_shot_results = []
    for col, expected in zip(test_cols, test_labels):
        result = get_answer_deepseek(model, tokenizer, system_prompt_classification, col)
        cleaned_result = clean_prediction(result)
        zero_shot_results.append(cleaned_result)
        
        print(f"Сочетание: {col}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный: {cleaned_result} | Ожидаемый: {expected} | {'✓' if cleaned_result == expected else '✗'}")
        print("-" * 40)
    
    # Тестируем few-shot
    print("\n--- FEW-SHOT ---")
    few_shot_results = []
    for col, expected in zip(test_cols, test_labels):
        result = get_answer_deepseek(model, tokenizer, system_prompt_classification, col, sample_few_shot_class)
        cleaned_result = clean_prediction(result)
        few_shot_results.append(cleaned_result)
        
        print(f"Сочетание: {col}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный: {cleaned_result} | Ожидаемый: {expected} | {'✓' if cleaned_result == expected else '✗'}")
        print("-" * 40)
    
    # Вычисляем метрики
    print("\n--- МЕТРИКИ ---")
    if len(zero_shot_results) == len(test_labels):
        precision, recall, f1 = compute_metrics(test_labels, zero_shot_results)
        print(f"Zero-shot - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    if len(few_shot_results) == len(test_labels):
        precision, recall, f1 = compute_metrics(test_labels, few_shot_results)
        print(f"Few-shot - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    return model, tokenizer, system_prompt_classification

def process_dataset(model, tokenizer, dataset_path, output_dirs):
    """Обрабатывает весь датасет"""
    try:
        # Загружаем данные
        print(f"Загрузка данных из {dataset_path}...")
        df = pd.read_csv(dataset_path, encoding="utf-8")
        cols = df['collocation'].to_list()
        print(f"Загружено {len(cols)} сочетаний")

        # Определяем промпт для базовой модели
        base_prompt = '''Задача: определить, является ли сочетание глагола и существительного коллокацией.
Коллокация - это устойчивое словосочетание, где слова часто употребляются вместе.
Ответь только цифрой: 1 если сочетание является коллокацией, 0 если не является.'''

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

        # Настройки для каждого режима
        configs = [
            ("zero-shot", None),
            ("one-shot", one_shot_examples),
            ("few-shot", few_shot_examples)
        ]
        
        regimes = ['zero-shot', 'one-shot', 'few-shot']

        # Обрабатываем каждый режим
        for config_idx, ((regime, examples), output_dir) in enumerate(zip(configs, output_dirs), 1):
            if output_dir is None:
                print(f"Пропускаем {regime} - не указан output_dir")
                continue
                
            print(f"\nОбработка {regime} ({config_idx}/{len(configs)})...")
            print(f"Результаты будут сохранены в: {output_dir}")
            
            # Очищаем файл результатов
            with open(output_dir, 'w', encoding='utf-8') as f:
                f.write("")

            correct_predictions = 0
            total_predictions = 0
            
            for i, col in enumerate(tqdm(cols, desc=f"{regime}", leave=False)):
                try:
                    pred = get_answer_deepseek(model, tokenizer, base_prompt, col, examples)
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
    argument_parser = ArgumentParser(description="Классификация коллокаций с помощью DeepSeek-LLM-7B-Base")
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
        test_deepseek_model()
        return

    # Загружаем модель
    model, tokenizer = load_deepseek_model()
    if model is None or tokenizer is None:
        print("Не удалось загрузить модель")
        return

    # Проверяем, что указан хотя бы один output_dir
    output_dirs = [args.output_dir_zero_shot, args.output_dir_one_shot, args.output_dir_few_shot]
    if all(output_dir is None for output_dir in output_dirs):
        print("Сначала протестируем модель...")
        test_deepseek_model()
        return

    # Обрабатываем датасет
    process_dataset(model, tokenizer, args.dataset, output_dirs)
    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()