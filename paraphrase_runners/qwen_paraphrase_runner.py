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

    # Применяем шаблон чата Qwen
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=True, 
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
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

def load_qwen_model():
    """Загружает модель Qwen2.5-7B-Instruct"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"Загрузка модели {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Для Qwen обычно pad_token уже установлен, но проверим
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # Может потребоваться для Qwen
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
    
    # Промпт для перефразирования
    prompt_paraphrase = """Ниже дано сочетание глагола и существительного. Замени все сочетание на один глагол с таким же значением.
В ответе напиши ТОЛЬКО глагол, больше ничего. Твой ответ должен быть ТОЛЬКО на русском языке."""

    # Few-shot примеры
    few_shot_examples_paraphrases = [
        {'source': 'вскрывать причины', 'target': 'выяснять'},
        {'source': 'вставать во главе', 'target': 'возглавлять'},
        {'source': 'брать за правило', 'target': 'следовать'},
        {'source': 'бросаться обещаниями', 'target': 'обещать'},
        {'source': 'схватывать суть', 'target': 'понимать'}
    ]
    
    system_prompt_few_shot = make_fs_prompt(prompt_paraphrase, few_shot_examples_paraphrases)
    
    # Тестовые данные
    test_cols = ["делать работу", "бить баклуши", "читать книгу", "взять верх", "поднять тост"]
    expected_paraphrases = ["работать", "бездельничать", "читать", "победить", "тостовать"]
    
    print("\nТестирование модели на задаче перефразирования:")
    print("="*70)
    
    # Тестируем zero-shot
    print("\n--- ZERO-SHOT ---")
    for col, expected in zip(test_cols, expected_paraphrases):
        result = get_answer_qwen(model, tokenizer, prompt_paraphrase, col)
        cleaned_result = clean_paraphrase(result)
        
        print(f"Сочетание: {col}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 50)
    
    # Тестируем few-shot
    print("\n--- FEW-SHOT ---")
    for col, expected in zip(test_cols, expected_paraphrases):
        result = get_answer_qwen(model, tokenizer, system_prompt_few_shot, col)
        cleaned_result = clean_paraphrase(result)
        
        print(f"Сочетание: {col}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 50)
    
    return model, tokenizer, system_prompt_few_shot

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

        # Определяем промпты
        prompt_paraphrase = """Ниже дано сочетание глагола и существительного. Замени все сочетание на один глагол с таким же значением.
В ответе напиши ТОЛЬКО глагол, больше ничего. Твой ответ должен быть ТОЛЬКО на русском языке."""

        one_shot_examples_paraphrases = [
            {'source': 'вскрывать причины', 'target': 'выяснять'}
        ]

        few_shot_examples_paraphrases = [
            {'source': 'вскрывать причины', 'target': 'выяснять'},
            {'source': 'вставать во главе', 'target': 'возглавлять'},
            {'source': 'брать за правило', 'target': 'следовать'},
            {'source': 'бросаться обещаниями', 'target': 'обещать'},
            {'source': 'схватывать суть', 'target': 'понимать'}
        ]

        # Создаем промпты
        prompts = [
            prompt_paraphrase,  # zero-shot
            make_fs_prompt(prompt_paraphrase, one_shot_examples_paraphrases),  # one-shot
            make_fs_prompt(prompt_paraphrase, few_shot_examples_paraphrases)   # few-shot
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

            processed_count = 0
            
            for col, syn in tqdm(zip(cols, syns), total=len(cols), desc=f"{regime}", leave=False):
                try:
                    pred = get_answer_qwen(model, tokenizer, prompt, col)
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
    argument_parser = ArgumentParser(description="Перефразирование коллокаций с помощью Qwen2.5-7B-Instruct")
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