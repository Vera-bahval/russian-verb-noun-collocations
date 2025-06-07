import torch
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import time

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

def make_prompt_classification(base_prompt, examples):
    """Альтернативная функция для создания промпта (используется в main)"""
    return make_fs_prompt(base_prompt, examples)

def get_answer_mistral(model, tokenizer, system_prompt, user_text):
    """Получает ответ от модели Mistral"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

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
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только ответ после [/INST]
    if '[/INST]' in response:
        response = response.split('[/INST]')[1].strip()
    
    return response

def clean_prediction(pred):
    """Очищает предсказание, оставляя только цифры"""
    import re
    # Ищем первую цифру 0 или 1
    match = re.search(r'[01]', pred)
    if match:
        return match.group()
    return pred.strip()

def compute_metrics(true_labels, pred_labels):
    """Вычисляет метрики precision, recall, f1"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', pos_label='1'
    )
    return precision, recall, f1

def test_mistral_model():
    """Тестирует модель Mistral на небольшом наборе данных"""
    print("Загрузка модели Mistral...")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Модель загружена успешно!")
        
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
        test_cols = ["делать работу", "бить баклуши", "читать книгу"]
        test_labels = ["0", "1", "0"]
        
        print("\nТестирование модели:")
        results = []
        
        for col, expected in zip(test_cols, test_labels):
            result = get_answer_mistral(model, tokenizer, system_prompt_few_shot, col)
            cleaned_result = clean_prediction(result)
            results.append(cleaned_result)
            print(f"Сочетание: {col}")
            print(f"Результат: {result}")
            print(f"Очищенный результат: {cleaned_result}")
            print(f"Ожидаемый: {expected}")
            print("-" * 50)
        
        return model, tokenizer, system_prompt_few_shot
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None, None

def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--dataset", default="collocations_dataset_final_version.csv", 
                               help="Путь к CSV файлу с данными")
    argument_parser.add_argument("--vars", help="Путь к JSON файлу с токенами")
    argument_parser.add_argument("--output_dir_zero_shot", help="Путь для сохранения результатов zero-shot")
    argument_parser.add_argument("--output_dir_one_shot", help="Путь для сохранения результатов one-shot")
    argument_parser.add_argument("--output_dir_few_shot", help="Путь для сохранения результатов few-shot")
    argument_parser.add_argument("--test_mistral", action="store_true", 
                               help="Запустить тест модели Mistral")
    argument_parser.add_argument("--use_yandex", action="store_true",
                               help="Использовать Yandex GPT вместо Mistral")

    args = argument_parser.parse_args()

    # Если запрошен тест Mistral
    if args.test_mistral:
        test_mistral_model()
        return

    # Проверяем наличие обязательных аргументов для основной работы
    if not args.dataset:
        print("Ошибка: необходимо указать путь к датасету")
        return

    try:
        # Загружаем данные
        print(f"Загрузка данных из {args.dataset}...")
        df = pd.read_csv(args.dataset, encoding="utf-8")
        cols = df['collocation'].to_list()
        print(f"Загружено {len(cols)} сочетаний")

        # Определяем промпты
        prompt_classification = '''Ниже дано сочетание глагола и существительного.
Определи, является ли оно коллокацией. Ответь только 1 (если сочетание - коллокация) или только 0 (если сочетание - не коллокация).
В твоем ответе должна быть ТОЛЬКО соответствующая цифра, больше ничего'''

        one_shot_examples_classification = [
            {"source": "выносить выговор", "target": "1"},
            {"source": "выражать интересы", "target": "0"}
        ]

        few_shot_examples_classification = [
            {"source": "выносить выговор", "target": "1"},
            {"source": "выражать интересы", "target": "0"},
            {"source": "гнать в шею", "target": "1"},
            {"source": "жить без забот", "target": "0"},
            {"source": "держаться особняком", "target": "1"},
            {"source": "закусить бутербродом", "target": "0"}
        ]

        # Создаем промпты
        prompt_classification_one_shot = make_prompt_classification(
            prompt_classification, one_shot_examples_classification
        )
        prompt_classification_few_shot = make_prompt_classification(
            prompt_classification, few_shot_examples_classification
        )

        regimes = ['zero_shot', 'one_shot', 'few_shot']
        prompts_classification = [
            prompt_classification, 
            prompt_classification_one_shot, 
            prompt_classification_few_shot
        ]
        output_dirs = [
            args.output_dir_zero_shot, 
            args.output_dir_one_shot, 
            args.output_dir_few_shot
        ]

        model, tokenizer, _ = test_mistral_model()
            
        if model is None:
                print("Не удалось загрузить модель Mistral")
                return

        for prompt_idx, (prompt, regime, output_dir) in enumerate(
                zip(prompts_classification, regimes, output_dirs), 1
            ):
                if output_dir is None:
                    continue
                    
                print(f"\nОбработка prompt {prompt_idx}/{len(prompts_classification)} ({regime})...")

                for col in tqdm(cols, desc=f"Prompt {prompt_idx}", leave=False):
                    pred = get_answer_mistral(model, tokenizer, prompt, col)
                    cleaned_pred = clean_prediction(pred)
                    write_to_file(col, cleaned_pred, output_dir)

                print(f"Prompt {prompt_idx} ({regime}) обработан")

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()