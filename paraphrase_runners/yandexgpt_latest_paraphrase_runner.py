import requests
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def write_to_file(col, pred, syn, output_dir):
    with open(output_dir, 'a') as f:
        f.write(col + '@' + pred + '@' + syn + '\n')
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

def make_prompt_paraphrase(system_prompt, samples=[]):
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
    cols = df[df['label'] == 1]['collocation'].to_list()
    syns = df[df['label'] == 1]['synonym'].to_list()

    with open(args.vars, "r", encoding="utf-8") as fin:
      folder_id = json.load(fin)["folder_id"]
    with open(args.vars, "r", encoding="utf-8") as fin:
      iam_token = json.load(fin)["iamToken"]
    
    prompt_paraphrase = """Ниже дано сочетание глагола и существительного. Замени все сочетание на один глагол с таким же значением.
    В ответе напиши ТОЛЬКО глагол, больше ничего."""

    one_shot_examples_paraphrases =  [
        {
            'source': 'вскрывать причины',
            'target': 'выяснять'
        }
    ]

    few_shot_examples_paraphrases = [
        {
            'source': 'вскрывать причины',
            'target': 'выяснять'
        },
        {
            'source': 'вставать во главе',
            'target': 'возглавлять'
        },
        {
            'source': 'брать за правило',
            'target': 'сдедовать'
        },
        {
            'source': 'бросаться обещаниями',
            'target': 'обещать'
        },
        {
            'source': 'схватывать суть ',
            'target': 'понять'
        }
    ]

    prompt_paraphrase_one_shot = make_prompt_paraphrase(prompt_paraphrase, one_shot_examples_paraphrases)
    prompt_paraphrase_few_shot = make_prompt_paraphrase(prompt_paraphrase, few_shot_examples_paraphrases)
    regimes = ['zero_shot', 'one_shot', 'few_shot']
    prompts_paraphrase = [prompt_paraphrase, prompt_paraphrase_one_shot, prompt_paraphrase_few_shot]
    model_name = 'yandexgpt/latest'
    output_dirs = [args.output_dir_zero_shot, args.output_dir_one_shot, args.output_dir_few_shot]
    
    for prompt_idx, (prompt, regime, output_dir) in enumerate(zip(prompts_paraphrase, regimes, output_dirs), 1):
        print(f"\nОбработка prompt {prompt_idx}/{len(prompts_paraphrase)}...")

        for col, syn in tqdm(zip(cols, syns), total=len(cols), desc=f"Prompt {prompt_idx}", leave=False):
            pred = run(iam_token, folder_id, model_name, prompt, col)
            write_to_file(col, pred, syn, output_dir)

        print(f"Prompt {prompt_idx} обработан")
