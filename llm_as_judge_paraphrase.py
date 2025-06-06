import requests
import json
import csv
import re
import time
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict, Counter

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

class SynonymEvaluator:
    def __init__(self, iam_token, folder_id, model_name='yandexgpt-lite/latest'):
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.model_name = model_name
        self.stats = defaultdict(int)
        self.evaluation_scores = []
        self.error_count = 0
        self.start_time = None
        
    def judge_synonym(self, prompt, col, ans):
        """Оценка синонимичности"""
        
        data = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_name}",
            "completionOptions": {"temperature": 1, "maxTokens": 100},
            "messages": [{"role": "user", "text": prompt}]
        }
        
        try:
            response = requests.post(
                URL,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.iam_token}"
                },
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            text = result["result"]["alternatives"][0]["message"]["text"]
            return text
                
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}, collocation: {col}, model's answer: {ans}")
            print(f"Response content: {response.text if 'response' in locals() else 'Response unavailable'}")
            return "Error"
        except Exception:
            return "Error"

    def parse_evaluation_result(self, result_text):
        """Парсинг JSON результата оценки"""
        json_match = re.search(r'\{.*?\}', result_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            try:
                json_data = json.loads(json_string)
                # Валидация структуры JSON
                required_keys = ["Семантическая близость", "Грамматическая близость", "Сумма"]
                if all(key in json_data for key in required_keys):
                    # Конвертация в числа
                    for key in required_keys:
                        json_data[key] = float(json_data[key])
                    
                    self.evaluation_scores.append(json_data["Сумма"])
                    return json.dumps(json_data, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
                
        return '-'

    def write_to_csv(self, file_path, row):
        """Запись строки в CSV файл"""
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
            csvfile.flush()

    def run_evaluation(self, prompt, cols, answers, true_syns, output_file):
        """Основной цикл оценки"""
        self.start_time = time.time()
        judge_results = []
        
        for col, ans, true_syn in tqdm(zip(cols, answers, true_syns), total=len(cols), desc="Judging answers"):
            formatted_prompt = prompt.format(ans=ans, true_syn=true_syn)
            result = self.judge_synonym(formatted_prompt, col, ans)
            json_res = self.parse_evaluation_result(result)
            
            row = [col, ans, true_syn, json_res]
            self.write_to_csv(output_file, row)
            judge_results.append(json_res)
        
        self._print_final_stats(len(cols))
        return judge_results

def load_input_data(file_path):
    """Загрузка входных данных"""
    cols, answers, true_syns = [], [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('@')
            if len(parts) != 3:
                continue
                
            cols.append(parts[0])
            answers.append(parts[1])
            true_syns.append(parts[2])
    
    return cols, answers, true_syns

def initialize_output_file(output_path):
    """Инициализация выходного CSV файла"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['collocation', 'model_answer', 'true_synonym', 'evaluation'])

# Промпт для оценки (сохранен без изменений)
prompt_llm_as_judge_paraphrase = """
    Ты - лингвистический эксперт, оценивающий семантическую и грамматическую близость двух глаголов.

    Первый глагол: "{ans}"
    Второй глагол: "{true_syn}"

    Оцени, насколько семантически и грамматически близки эти два глагола, по следующим критериям:

    1. Семантическая близость (0-3 балла)
       3 - глаголы полностью совпадают
       2 - глаголы - очень близкие синонимы, их можно заменить друг на друга в одном предложении (например, "взглянуть" и "посмотреть")
       1 - есть общее значение, но есть заметные отличия (например, "защищать" и "ценить")
       0 - глаголы очень отдалённо связаны друг с другом или совершенно не связаны по смыслу (например, "устанавливать" и "следовать") или одно из слов не на русском языке/содержит нерусские вставки (например, "-lnуть")

    2. Грамматическая близость (0-3 балла)
       3 - формы глаголов полностью совпадают
       2 - незначительные отличия в формах (разные формы вида или возвратности глаголов)
       1 - существенные различия в формах
       0 - одно из слов не является глаголом или одно из слов не на русском языке/содержит нерусские вставки

    Ответ дай в формате JSON:
    {{
      "Семантическая близость": "число",
      "Грамматическая близость": "число",
      "Сумма": "число" (сумма Семантической близости и Грамматической близости)
    }}

    Твой ответ должен содержать только json объект, больше ничего. НЕ приводи в ответе НИКАКИЕ рассуждения.
    """

def main():
    argument_parser = ArgumentParser(description="Оценка синонимичности глаголов с помощью LLM")
    argument_parser.add_argument("--model_answers_zero_shot", required=True, 
                               help="Путь к файлу с ответами модели (zero-shot)")
    argument_parser.add_argument("--model_answers_one_shot", 
                               help="Путь к файлу с ответами модели (one-shot)")
    argument_parser.add_argument("--model_answers_few_shot", 
                               help="Путь к файлу с ответами модели (few-shot)")
    argument_parser.add_argument("--vars", required=True, 
                               help="Путь к файлу с конфигурацией (folder_id, iamToken)")
    argument_parser.add_argument("--output_csv_eval_zero_shot", required=True,
                               help="Путь к выходному CSV файлу (zero-shot)")
    argument_parser.add_argument("--output_csv_eval_one_shot", 
                               help="Путь к выходному CSV файлу (one-shot)")
    argument_parser.add_argument("--output_csv_eval_few_shot", 
                               help="Путь к выходному CSV файлу (few-shot)")
    
    args = argument_parser.parse_args()
    
    # Загрузка конфигурации
    with open(args.vars, "r", encoding="utf-8") as fin:
        config = json.load(fin)
        folder_id = config["folder_id"]
        iam_token = config["iamToken"]
    
    model_name = 'yandexgpt-lite/latest'
    
    # Подготовка списков входных и выходных файлов
    input_files = []
    output_files = []
    
    if args.model_answers_zero_shot and args.output_csv_eval_zero_shot:
        input_files.append(args.model_answers_zero_shot)
        output_files.append(args.output_csv_eval_zero_shot)
        
    if args.model_answers_one_shot and args.output_csv_eval_one_shot:
        input_files.append(args.model_answers_one_shot)
        output_files.append(args.output_csv_eval_one_shot)
        
    if args.model_answers_few_shot and args.output_csv_eval_few_shot:
        input_files.append(args.model_answers_few_shot)
        output_files.append(args.output_csv_eval_few_shot)
    
    # Инициализация выходных файлов
    for output_file in output_files:
        initialize_output_file(output_file)
    
    # Обработка каждой пары файлов
    for input_file, output_file in zip(input_files, output_files):
        # Создание экземпляра оценщика для каждого файла
        evaluator = SynonymEvaluator(iam_token, folder_id, model_name)
        
        # Загрузка данных
        cols, answers, true_syns = load_input_data(input_file)
        
        # Запуск оценки
        judge_results = evaluator.run_evaluation(
            prompt_llm_as_judge_paraphrase, 
            cols, answers, true_syns, 
            output_file
        )

if __name__ == "__main__":
    main()
