import requests
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def write_to_file(col, sent, pred, output_dir):
    with open(output_dir, 'a') as f:
        f.write(col + '@' + sent + '@' + pred + '\n')
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

def make_prompt_fill_sent(prompt, samples=[]):
    samples = samples or []
    answer = prompt
    if len(samples) > 0:
      answer += '\n\nПримеры:'
    for sample in samples:
        answer += "\n\nПредложение: {source}\nОтвет: {target}".format(**sample)
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
    gap_sent = df[df['label'] == 1]['sentence_with_gap'].to_list()

    with open(args.vars, "r", encoding="utf-8") as fin:
      folder_id = json.load(fin)["folder_id"]
    with open(args.vars, "r", encoding="utf-8") as fin:
      iam_token = json.load(fin)["iamToken"]
    
    prompt_fill_sent = '''Ниже дано предложение. На месте [MASK] пропущено устойчивое сочетание русских глагола и существительного.
    Напиши это сочетание. Сочетание, которое ты подберешь, должно подходит в предложении по смыслу и грамматически. Между глаголом и существительным может быть предлог.
    В ответе напиши ТОЛЬКО сочетание глагола и существительного (и предлог между ниими, если нужно), больше ничего.'''

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

    prompt_fill_sent_one_shot = make_prompt_fill_sent(prompt_fill_sent, one_shot_fill_sent_examples)
    prompt_fill_sent_few_shot = make_prompt_fill_sent(prompt_fill_sent, few_shot_fill_sent_examples)
    regimes = ['zero_shot', 'one_shot', 'few_shot']
    prompts_fill_sent = [prompt_fill_sent, prompt_fill_sent_one_shot, prompt_fill_sent_few_shot]
    model_name = 'yandexgpt/latest'
    output_dirs = [args.output_dir_zero_shot, args.output_dir_one_shot, args.output_dir_few_shot]
    
    for prompt_idx, (prompt, regime, output_dir) in enumerate(zip(prompts_fill_sent, regimes, output_dirs), 1):
        print(f"\nОбработка prompt {prompt_idx}/{len(prompts_fill_sent)}...")

        for col, sent in tqdm(zip(cols, gap_sent), total=len(cols), desc=f"Prompt {prompt_idx}", leave=False):
            pred = run(iam_token, folder_id, model_name, prompt, sent)
            write_to_file(col, sent, pred, output_dir)

        print(f"Prompt {prompt_idx} обработан")