import torch
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def write_to_file(sentence, pred, target, output_dir):
    """Записывает результат в файл в формате: предложение@предсказание@правильный_ответ"""
    with open(output_dir, 'a', encoding='utf-8') as f:
        f.write(sentence + '@' + pred + '@' + target + '\n')
        f.flush()

def make_fs_prompt(prompt, samples):
    """Создает промпт с примерами для few-shot обучения"""
    answer = prompt + " Ниже приведены примеры."
    for sample in samples:
        answer += "\n\nПредложение: {source}\nОтвет: {target}".format(**sample)
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
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # Декодируем только новые токены (ответ модели)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response.strip()

def clean_collocation(pred):
    """Очищает предсказание, оставляя только коллокацию"""
    pred = pred.strip()
    
    # Убираем возможные лишние символы в начале и конце
    pred = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', pred)
    
    # Убираем лишние пробелы
    pred = ' '.join(pred.split())
    
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
    
    # Промпт для заполнения пропуска коллокацией
    prompt_fill_collocation = """Ты - помощник, специализирующийся на русских устойчивых словосочетаниях. Твоя задача - заполнить пропуск [MASK] в предложении УСТОЙЧИВЫМ СОЧЕТАНИЕМ, состоящим из ГЛАГОЛА и СУЩЕСТВИТЕЛЬНОГО.
ОБРАТИ ВНИМАНИЕ: Твой ответ ВСЕГДА должен содержать ДВА элемента:
1. ГЛАГОЛ (например: разбудить, вести, отойти)
2. СУЩЕСТВИТЕЛЬНОЕ (например: интерес, учет, сторону)
ВАЖНО: Между глаголом и существительным может быть предлог, но это необязательно.
ЗАПОМНИ СТРУКТУРУ ОТВЕТА:
[ГЛАГОЛ] + [(предлог)] + [СУЩЕСТВИТЕЛЬНОЕ]
Примеры правильных ответов:
- разбудить интерес
- вести учет
- держать слово
- уступить место
- отойти в сторону
- впасть в отчаяние
Примеры НЕПРАВИЛЬНЫХ ответов (нельзя давать только глагол):
- пожертвовать (нет существительного)
- выйти (нет существительного)
- раздаваться (нет существительного)
ТАКЖЕ ВАЖНО: Сочетание должно быть устойчивым и естественным для русского языка и должно подходить в предложении по СМЫСЛУ и ГРАММАТИЧЕСКИ.
В ответе ОБЯЗАТЕЛЬНО должны быть ГЛАГОЛ И СУЩЕСТВИТЕЛЬНОЕ (и предлог между ними, если нужно).
Твой ответ должен быть ТОЛЬКО на РУССКОМ языке."""

    # Few-shot примеры
    one_shot_fill_sent_examples = [
        {
            "source": "Удивительный рассказ лектора смог мгновенно [MASK] у студентов, которые уже начали зевать.",
            "target": "разбудить интерес"
        }
    ]

    few_shot_fill_sent_examples = [
        {
            "source": "Удивительный рассказ лектора смог мгновенно [MASK] у студентов, которые уже начали зевать.",
            "target": "разбудить интерес"
        },
        {
            "source": "Комиссия признала ситуацию исчерпанной и решила окончательно [MASK] по этой проблеме.",
            "target": "закрыть вопрос"
        },
        {
            "source": "Когда времени на раздумья не осталось, они решительно [MASK], несмотря на численное превосходство врага.",
            "target": "пошли в атаку"
        },
        {
            "source": "Мечтатели часто [MASK] и предаются меланхолии вместо того, чтобы что-то делать.",
            "target": "разводят сантименты"
        },
        {
            "source": "Полководцу просто надо знать и понимать, кого и когда [MASK] и кого беречь для более тяжелых боев.",
            "target": "бросать в бой"
        }
    ]
    
    system_prompt_one_shot = make_fs_prompt(prompt_fill_collocation, one_shot_fill_sent_examples)
    system_prompt_few_shot = make_fs_prompt(prompt_fill_collocation, few_shot_fill_sent_examples)
    
    # Тестовые данные
    test_sentences = [
        "Он решил [MASK] и рассказать всю правду о случившемся.",
        "После долгих споров депутаты наконец смогли [MASK] по важному вопросу.",
        "Студентам нужно [MASK] при подготовке к экзамену по математике.",
        "Директор попросил всех сотрудников [MASK] в новом проекте.",
        "Артист сумел [MASK] зрителей своим блестящим выступлением."
    ]
    expected_collocations = [
        "взять слово",
        "прийти к соглашению",
        "приложить усилия",
        "принять участие",
        "произвести впечатление"
    ]
    
    print("\nТестирование модели на задаче заполнения пропуска коллокацией:")
    print("="*80)
    
    # Тестируем zero-shot
    print("\n--- ZERO-SHOT ---")
    for sentence, expected in zip(test_sentences, expected_collocations):
        result = get_answer_qwen(model, tokenizer, prompt_fill_collocation, sentence)
        cleaned_result = clean_collocation(result)
        
        print(f"Предложение: {sentence}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 70)
    
    # Тестируем one-shot
    print("\n--- ONE-SHOT ---")
    for sentence, expected in zip(test_sentences, expected_collocations):
        result = get_answer_qwen(model, tokenizer, system_prompt_one_shot, sentence)
        cleaned_result = clean_collocation(result)
        
        print(f"Предложение: {sentence}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 70)
    
    # Тестируем few-shot
    print("\n--- FEW-SHOT ---")
    for sentence, expected in zip(test_sentences, expected_collocations):
        result = get_answer_qwen(model, tokenizer, system_prompt_few_shot, sentence)
        cleaned_result = clean_collocation(result)
        
        print(f"Предложение: {sentence}")
        print(f"Полный ответ: '{result}'")
        print(f"Очищенный результат: '{cleaned_result}'")
        print(f"Ожидаемый: '{expected}'")
        print("-" * 70)
    
    return model, tokenizer, system_prompt_one_shot, system_prompt_few_shot

def process_dataset(model, tokenizer, dataset_path, output_dirs):
    """Обрабатывает весь датасет"""
    try:
        # Загружаем данные
        print(f"Загрузка данных из {dataset_path}...")
        df = pd.read_csv(dataset_path, encoding="utf-8")
        
        # Проверяем наличие необходимых столбцов
        required_columns = ['sentence', 'target_collocation']
        for col in required_columns:
            if col not in df.columns:
                print(f"Ошибка: столбец '{col}' не найден в датасете")
                print(f"Доступные столбцы: {list(df.columns)}")
                return
        
        sentences = df['sentence'].to_list()
        targets = df['target_collocation'].to_list()
        print(f"Загружено {len(sentences)} предложений")

        # Определяем промпт
        prompt_fill_collocation = """Ты - помощник, специализирующийся на русских устойчивых словосочетаниях. Твоя задача - заполнить пропуск [MASK] в предложении УСТОЙЧИВЫМ СОЧЕТАНИЕМ, состоящим из ГЛАГОЛА и СУЩЕСТВИТЕЛЬНОГО.
ОБРАТИ ВНИМАНИЕ: Твой ответ ВСЕГДА должен содержать ДВА элемента:
1. ГЛАГОЛ (например: разбудить, вести, отойти)
2. СУЩЕСТВИТЕЛЬНОЕ (например: интерес, учет, сторону)
ВАЖНО: Между глаголом и существительным может быть предлог, но это необязательно.
ЗАПОМНИ СТРУКТУРУ ОТВЕТА:
[ГЛАГОЛ] + [(предлог)] + [СУЩЕСТВИТЕЛЬНОЕ]
Примеры правильных ответов:
- разбудить интерес
- вести учет
- держать слово
- уступить место
- отойти в сторону
- впасть в отчаяние
Примеры НЕПРАВИЛЬНЫХ ответов (нельзя давать только глагол):
- пожертвовать (нет существительного)
- выйти (нет существительного)
- раздаваться (нет существительного)
ТАКЖЕ ВАЖНО: Сочетание должно быть устойчивым и естественным для русского языка и должно подходить в предложении по СМЫСЛУ и ГРАММАТИЧЕСКИ.
В ответе ОБЯЗАТЕЛЬНО должны быть ГЛАГОЛ И СУЩЕСТВИТЕЛЬНОЕ (и предлог между ними, если нужно).
Твой ответ должен быть ТОЛЬКО на РУССКОМ языке."""

        one_shot_fill_sent_examples = [
            {
                "source": "Удивительный рассказ лектора смог мгновенно [MASK] у студентов, которые уже начали зевать.",
                "target": "разбудить интерес"
            }
        ]

        few_shot_fill_sent_examples = [
            {
                "source": "Удивительный рассказ лектора смог мгновенно [MASK] у студентов, которые уже начали зевать.",
                "target": "разбудить интерес"
            },
            {
                "source": "Комиссия признала ситуацию исчерпанной и решила окончательно [MASK] по этой проблеме.",
                "target": "закрыть вопрос"
            },
            {
                "source": "Когда времени на раздумья не осталось, они решительно [MASK], несмотря на численное превосходство врага.",
                "target": "пошли в атаку"
            },
            {
                "source": "Мечтатели часто [MASK] и предаются меланхолии вместо того, чтобы что-то делать.",
                "target": "разводят сантименты"
            },
            {
                "source": "Полководцу просто надо знать и понимать, кого и когда [MASK] и кого беречь для более тяжелых боев.",
                "target": "бросать в бой"
            }
        ]

        # Создаем промпты
        prompts = [
            prompt_fill_collocation,  # zero-shot
            make_fs_prompt(prompt_fill_collocation, one_shot_fill_sent_examples),  # one-shot
            make_fs_prompt(prompt_fill_collocation, few_shot_fill_sent_examples)   # few-shot
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
            
            for sentence, target in tqdm(zip(sentences, targets), total=len(sentences), desc=f"{regime}", leave=False):
                try:
                    pred = get_answer_qwen(model, tokenizer, prompt, sentence)
                    cleaned_pred = clean_collocation(pred)
                    write_to_file(sentence, cleaned_pred, target, output_dir)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Ошибка при обработке '{sentence}': {e}")
                    write_to_file(sentence, "неизвестно", target, output_dir)

            print(f"{regime} завершен! Обработано {processed_count}/{len(sentences)} предложений")

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    argument_parser = ArgumentParser(description="Заполнение пропусков коллокациями с помощью Qwen2.5-7B-Instruct")
    argument_parser.add_argument("--dataset", default="collocation_sentences_dataset.csv", 
                               help="Путь к CSV файлу с данными (должен содержать столбцы 'sentence' и 'target_collocation')")
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