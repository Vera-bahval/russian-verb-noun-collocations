import torch
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def write_to_file(col, pred, syn, output_dir):
    """Записывает результат в файл в формате: коллокация@предсказание@синоним"""
    with open(output_dir, 'a', encoding='utf-8') as f:
        f.write(col + '@' + pred + '@' + syn + '\n')
        f.flush()

def make_zero_shot_prompt(col):
    """Создает промпт для zero-shot с принудительной генерацией начального токена"""
    prompt = f"Вопрос: Как одним глаголом заменить выражение '{col}'?\nОтвет: Выражение '{col}' можно заменить глаголом '"
    return prompt

def make_one_shot_prompt(col):
    """Создает промпт для one-shot с парным сравнением"""
    examples = {
        "вскрывать причины": "выяснять"
    }
    prompt = '''Замени сочетание глагола и существительного одним глаголом.
Примеры:
'''
    for key, val in examples.items():
        prompt += f"{key} = {val}\n"
    prompt += f"{col} ="
    return prompt

def make_few_shot_prompt(col):
    """Создает промпт для few-shot с парным сравнением"""
    examples = {
        'вскрывать причины': 'выяснять',
        'вставать во главе': 'возглавлять',
        'брать за правило': 'следовать',
        'бросаться обещаниями': 'обещать',
        'схватывать суть': 'понять'
    }
    prompt = '''Замени сочетание глагола и существительного одним глаголом.
Примеры:
'''
    for key, val in examples.items():
        prompt += f"{key} = {val}\n"
    prompt += f"{col} ="
    return prompt

def get_answer_deepseek(model, tokenizer, prompt):
    """Получает ответ от модели DeepSeek"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=7,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            temperature=0.7,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )

    # Декодируем только новые токены (ответ модели)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response.strip()

def clean_paraphrase(pred):
    """Очищает предсказание, оставляя только глагол"""
    pred = pred.strip()
    
    # Убираем возможные лишние символы и знаки препинания
    pred = re.sub(r'[^\w\s-]', '', pred)
    
    # Берем только первое слово (должен быть глагол)
    words = pred.split()
    if words:
        return words[0].lower()
    
    return pred.lower()

def load_deepseek_model():
    """Загружает модель deepseek-llm-7b-base"""
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    
    print(f"Загрузка модели {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        return None, None
    
    # Тестовые данные
    test_cols = ["делать работу", "бить баклуши", "читать книгу", "взять верх", "поднять тост"]
    expected_paraphrases = ["работать", "бездельничать", "читать", "победить", "тостовать"]
    
    print("\nТестирование модели на задаче перефразирования:")
    print("="*70)
    
    # Тестируем zero-shot
    print("\n--- ZERO-SHOT ---")
    for col, expected in zip(test_cols, expected_paraphrases):
        prompt = make_zero_shot_prompt(col)
        result = get_answer_deepseek(model, tokenizer, prompt)
        cleaned_result = clean_paraphrase(result)
        
        print(f"Сочетание: {col}")
        print(f"Промпт: {prompt}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 50)
    
    # Тестируем one-shot
    print("\n--- ONE-SHOT ---")
    for col, expected in zip(test_cols, expected_paraphrases):
        prompt = make_one_shot_prompt(col)
        result = get_answer_deepseek(model, tokenizer, prompt)
        cleaned_result = clean_paraphrase(result)
        
        print(f"Сочетание: {col}")
        print(f"Промпт: {prompt}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 50)
    
    # Тестируем few-shot
    print("\n--- FEW-SHOT ---")
    for col, expected in zip(test_cols, expected_paraphrases):
        prompt = make_few_shot_prompt(col)
        result = get_answer_deepseek(model, tokenizer, prompt)
        cleaned_result = clean_paraphrase(result)
        
        print(f"Сочетание: {col}")
        print(f"Промпт: {prompt}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 50)
    
    return model, tokenizer

def process_dataset(model, tokenizer, dataset_path, output_dirs):
    """Обрабатывает весь датасет"""
    try:
        # Загружаем данные
        print(f"Загрузка данных из {dataset_path}...")
        df = pd.read_csv(dataset_path, encoding="utf-8")
        
        # Проверяем наличие необходимых столбцов
        required_columns = ['collocation', 'synonym']
        for col in required_columns:
            if col not in df.columns:
                print(f"Ошибка: столбец '{col}' не найден в датасете")
                print(f"Доступные столбцы: {list(df.columns)}")
                return
        
        cols = df['collocation'].to_list()
        syns = df['synonym'].to_list()
        print(f"Загружено {len(cols)} сочетаний")

        # Определяем функции для создания промптов
        prompt_functions = [
            make_zero_shot_prompt,  # zero-shot
            make_one_shot_prompt,   # one-shot
            make_few_shot_prompt    # few-shot
        ]
        
        regimes = ['zero-shot', 'one-shot', 'few-shot']

        # Обрабатываем каждый режим
        for prompt_idx, (prompt_func, regime, output_dir) in enumerate(zip(prompt_functions, regimes, output_dirs), 1):
            if output_dir is None:
                print(f"Пропускаем {regime} - не указан output_dir")
                continue
                
            print(f"\nОбработка {regime} (prompt {prompt_idx}/{len(prompt_functions)})...")
            print(f"Результаты будут сохранены в: {output_dir}")
            
            # Очищаем файл результатов
            with open(output_dir, 'w', encoding='utf-8') as f:
                f.write("")

            processed_count = 0
            
            for col, syn in tqdm(zip(cols, syns), total=len(cols), desc=f"{regime}", leave=False):
                try:
                    prompt = prompt_func(col)
                    pred = get_answer_deepseek(model, tokenizer, prompt)
                    cleaned_pred = clean_paraphrase(pred)
                    write_to_file(col, cleaned_pred, syn, output_dir)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Ошибка при обработке '{col}': {e}")
                    write_to_file(col, "неизвестно", syn, output_dir)

            print(f"{regime} завершен! Обработано {processed_count}/{len(cols)} сочетаний")

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    argument_parser = ArgumentParser(description="Перефразирование коллокаций с помощью deepseek-llm-7b-base")
    argument_parser.add_argument("--dataset", default="collocations_dataset_final_version.csv", 
                               help="Путь к CSV файлу с данными (должен содержать столбцы 'collocation' и 'synonym')")
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