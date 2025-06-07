import requests
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def write_to_file(col, pred, output_dir):
    with open(output_dir, 'a') as f:
        f.write(col + '@' + pred + '\n')
        f.flush()

def run(iam_token, folder_id, model_name, system_prompt, user_text):
    data = {}
    data["modelUri"] = "gpt://"+ folder_id + "/" + model_name
    data["completionOptions"] = {"temperature": 1, "maxTokens": 10}
    messages = [

            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_text}

    ]
    data["messages"] = messages
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
        print(f"Error decoding JSON: {e}, query: {user_text}")
        print(f"Response content: {response}")
        text = "Error"
    return text

def make_prompt_classification(system_prompt, samples=[]):
    answer = system_prompt
    if len(samples) > 0:
        answer += "\n\nПримеры:"
    for sample in samples:
            answer += "\n\nСочетание: {source}\nЗамена: {target}".format(**sample)
    return answer

argument_parser = ArgumentParser()
argument_parser.add_argument("--dataset", default="collocations_dataset_final_version.csv")
argument_parser.add_argument("--vars")
argument_parser.add_argument("--output_dir_zero_shot")
argument_parser.add_argument("--output_dir_one_shot")
argument_parser.add_argument("--output_dir_few_shot")

if __name__ == "__main__":
    args = argument_parser.parse_args()

    df = pd.read_csv(args.dataset, encoding = "utf8")
    cols = df['collocation'].to_list()

    with open(args.vars, "r", encoding="utf-8") as fin:
      folder_id = json.load(fin)["folder_id"]
    with open(args.vars, "r", encoding="utf-8") as fin:
      iam_token = json.load(fin)["iamToken"]
    
    prompt_classification = '''Ниже дано сочетание глагола и существительного.
Определи, является ли оно коллокацией. Ответь только 1 (если сочетание - коллокация) или только 0 (если сочетание - не коллокация).
В твоем ответе должна быть ТОЛЬКО соответствующая цифра, больше ничего'''


    one_shot_examples_classification =  [
        {
        "source": "выносить выговор",
        "target": "1"
    },
    {
        "source": "выражать интересы",
        "target": "0"
    }
]

    few_shot_examples_classification = [
 {
        "source": "выносить выговор",
        "target": "1"
    },
    {
        "source": "выражать интересы",
        "target": "0"
    },
    {
        "source": "гнать в шею",
        "target": "1"
    },
    {
        "source": "жить без забот",
        "target": "0"
    },
    {
        "source": "держаться особняком",
        "target": "1"
    },

    {
        "source": "закусить бутербродом",
        "target": "0"
    }

]

    prompt_classification_one_shot = make_prompt_classification(prompt_classification, one_shot_examples_classification)
    prompt_classification_few_shot = make_prompt_classification(prompt_classification, few_shot_examples_classification)
    regimes = ['zero_shot', 'one_shot', 'few_shot']
    prompts_classification = [prompt_classification, prompt_classification_one_shot, prompt_classification_few_shot]
    model_name = 'yandexgpt/latest'
    output_dirs = [args.output_dir_zero_shot, args.output_dir_one_shot, args.output_dir_few_shot]

    for prompt_idx, (prompt, regime, output_dir) in enumerate(zip(prompts_classification, regimes, output_dirs), 1):
        print(f"\nОбработка prompt {prompt_idx}/{len(prompts_classification)}...")

        for col, syn in tqdm(zip(cols), total=len(cols), desc=f"Prompt {prompt_idx}", leave=False):
            pred = run(iam_token, folder_id, model_name, prompt, col)
            write_to_file(col, pred, output_dir)

        print(f"Prompt {prompt_idx} обработан")