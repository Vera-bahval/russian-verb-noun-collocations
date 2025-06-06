import requests
import json
import csv
import re
from argparse import ArgumentParser
from tqdm import tqdm

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def judge_synonym(prompt, col, ans, iam_token, folder_id, model_name):

    data = {}
    data["modelUri"] = "gpt://"+ folder_id + '/' + model_name
    data["completionOptions"] = {"temperature": 1, "maxTokens": 100}
    messages = [
            {"role": "user", "text": prompt}
            ]
    data['messages'] = messages
    response = requests.post(
        URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {iam_token}"
        },
        json=data,
    ).text

    try:
      text = json.loads(response)["result"]["alternatives"][0]["message"]["text"]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}, collocation: {col}, model's answer: {ans}")
        print(f"Response content: {response}")
        text = "Error"
    return text

def write_to_csv(file_path, row):
  with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(row)
    csvfile.flush()

def run_judge_with_csv(prompt, cols, answers, true_syns, file_path, iam_token, folder_id, model_name):
  
  judge_results = []

  for col, ans, true_syn in tqdm(zip(cols, answers, true_syns), total=len(cols), desc="Judging answers"):
    formatted_prompt = prompt.format(ans=ans, true_syn=true_syn)
    result = judge_synonym(formatted_prompt, col, ans, iam_token, folder_id, model_name)
    json_match = re.search(r'\{.*?\}', result, re.DOTALL)
    if json_match:
      json_string = json_match.group(0)
      json_data = json.loads(json_string)
      json_res = json.dumps(json_data, ensure_ascii=False, indent=2)
    else:
      json_res = '-'
    row = [col, ans, true_syn, json_res]
    write_to_csv(file_path, row)
    judge_results.append(json_res)

  return judge_results

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

argument_parser = ArgumentParser()
argument_parser.add_argument("--model_answers_zero_shot")
argument_parser.add_argument("--model_answers_one_shot")
argument_parser.add_argument("--model_answers_few_shot")
argument_parser.add_argument("--vars")
argument_parser.add_argument("--output_csv_eval_zero_shot")
argument_parser.add_argument("--output_csv_eval_one_shot")
argument_parser.add_argument("--output_csv_eval_few_shot")

if __name__ == "__main__":
    args = argument_parser.parse_args()

    with open(args.vars, "r", encoding="utf-8") as fin:
      folder_id = json.load(fin)["folder_id"]
    with open(args.vars, "r", encoding="utf-8") as fin:
      iam_token = json.load(fin)["iamToken"]

    model_name = 'yandexgpt-lite/latest'
    input_dirs = [args.model_answers_zero_shot, args.model_answers_one_shot, args.model_answers_few_shot]
    output_dirs = [args.output_csv_eval_zero_shot, args.output_csv_eval_one_shot, args.output_csv_eval_few_shot]

    for dir in output_dirs:
        with open(dir, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['collocation', 'model_answer', 'true_synonym', 'evaluation'])

    for input_dir, output_dir in zip(tqdm(input_dirs, desc="Processing input/output pairs"), output_dirs):
        cols, answers, true_syns = [], [], []
    
        with open(input_dir, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('@')
                cols.append(line[0])
                answers.append(line[1])
                true_syns.append(line[2])
    
        judge_results = run_judge_with_csv(prompt_llm_as_judge_paraphrase, cols, answers, true_syns, output_dir, iam_token, folder_id, model_name)