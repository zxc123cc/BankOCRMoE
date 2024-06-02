import pandas as pd
import string

def remove_prefix(text,prefix):
    sub_list = text.split(prefix)
    if len(sub_list) > 1:
        text = ''.join(sub_list[1:])
    else:
        text = sub_list[0]
    return text

def remove(text):
    #case 1
    text = text.replace("\"",'')
    #case 2
    if '用途' in text[:3]:
        text = remove_prefix(text,'用途')
    if '其他' in text[:3]:
        text = remove_prefix(text,'其他')
    text = remove_prefix(text,'用途 ')
    text = remove_prefix(text,'用途:')
    text = remove_prefix(text,'其他 ')
    text = remove_prefix(text,'其他:')
    text = text.strip()
    for x in [',','，',':','：',';','；']:
        text = text.strip(x)

    pre_word = ''
    result = ''
    ignore_list = [str(i) for i in range(0,10)]
    ignore_list += list(string.ascii_lowercase)
    ignore_list += list(string.ascii_uppercase)
    ignore_list += ['市','汉']
    for word in text:
        if word == pre_word and word not in ignore_list:
            pass
        else:
            result = result + word
        pre_word =word
    text = result

    return text

def replace(text):
    text = text.replace('大炬','火炬')
    text = text.replace('CHIA','CHINA')
    text = text.replace('CHIN','CHINA')
    text = text.replace('CHINAA','CHINA')
    text = text.replace('中国市','中国')
    text = text.replace('支行支行','支行')
    text = text.replace('北京北京','北京')

    for x in ['请岛','清岛','情岛','晴岛','蜻岛']:
        text = text.replace(x,'青岛')
    for x in ['青华','请岛','情华','晴华','蜻华']:
        text = text.replace(x,'清华')

    text = text.replace('行行','行')
    return text


def append(text):
    if ')' in text and '(' not in text:
        if '中国' in text and ('自贸' in text or '自由贸易' in text):
            text = '中国' + '(' + text[2:]

    return text

def post_one(text):
    text = remove(text)
    text = replace(text)
    text = append(text)
    return text

def process_post(test_tmp_path,out_path):
    with open(test_tmp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    file_name_list = []
    raw_results = []
    post_results = []
    for line in lines[1:]:
        file_name = line.strip().split(",")[0]
        file_name_list.append(file_name)
        result = ','.join(line.strip().split(",")[1:])
        raw_results.append(result)

        post_result = post_one(result)
        post_results.append(post_result)


    result_df = pd.DataFrame()
    result_df['file_name'] = file_name_list
    result_df['result'] = post_results
    result_df[['file_name', 'result']].to_csv(out_path, encoding='utf8', index=False)


def get_post_result(results):
    post_results = []
    for result in results:
        post_result = post_one(result)
        post_results.append(post_result)
    return post_results


if __name__ == '__main__':
    process_post('./submit/text_sub_smooth.csv',out_path='./submit/text_sub_smooth_post.csv')