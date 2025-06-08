from openai import OpenAI
import json

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="{}", # please replace {} with the your API obtained from OpenAI
    base_url="{}" # please replace {} with the url of OpenAI API
)

def gpt_35_api(messages: list):

    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return completion.choices[0].message.content
    #print(completion)

def gpt_35_api_stream(messages: list):

    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == '__main__':
    i = 0
    dataset = '{}' # please replace {} with the name of the dataset
    print(dataset)
    t = 'three' # four
    with open(dataset + '_aux_attr_init_content.jsonl', 'w', encoding='utf-8') as wf:
        with open(dataset + '_pair.txt', 'r', encoding='utf-8') as rf:
            for line in rf:
                content = line.strip()
                messages = [{'role': 'user','content': f'Please give me {t} adjectives that can describe the visual feature of a photo of a/an {content} well. Please strictly follow the format: give me just {t} words separated by commas and do not say more.'}]
                answer = gpt_35_api(messages)

                #gpt_35_api_stream(messages)
                dic = {content: answer}
                json.dump(dic, wf)
                wf.write('\n')
                i = i + 1
                print(i)