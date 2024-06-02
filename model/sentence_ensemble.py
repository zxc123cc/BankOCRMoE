from datasets import load_metric
import pandas as pd

cer_metric = load_metric("./cer.py")

def compute_metrics(pred_str,label_str):
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer

def select_one(sentence_list):
    min_score = 100000000
    sentence = sentence_list[0]
    for i in range(len(sentence_list)):
        hypothesis = sentence_list[i]
        hypothesis_list = [hypothesis for _ in range(len(sentence_list)-1)]
        reference_list = []
        for j in range(len(sentence_list)):
            if j==i:
                continue
            reference_list.append(sentence_list[j])
        score = compute_metrics(hypothesis_list,reference_list)
        print(score)
        if score < min_score:
            min_score = score
            sentence = hypothesis

    return sentence


def get_results(test_tmp_path):
    with open(test_tmp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    file_name_list = []
    raw_results = []
    for line in lines[1:]:
        file_name = line.strip().split(",")[0]
        file_name_list.append(file_name)
        result = ','.join(line.strip().split(",")[1:])
        raw_results.append(result)
    return raw_results


def run_sentence_ensemble(results_list):
    num = len(results_list[0])
    answers = []
    for i in range(num):
        tmp = [results[i] for results in results_list]
        answers.append(select_one(tmp))
    return answers


if __name__ == '__main__':
    # text
    test_tmp_path = './submit_tmp/text_sub.csv'

    with open(test_tmp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    images_names = []
    for line in lines[1:]:
        file_name = line.strip().split(",")[0]
        images_names.append(file_name)

    results1 = get_results('./submit/text_sub_1_2_post.csv')
    results2 = get_results('./submit/text_sub_1_3_post.csv')
    results3 = get_results('./submit/text_sub_2_3_post.csv')
    results4 = get_results('./submit/text_sub_ensemble3_post.csv')

    results = run_sentence_ensemble([results1,results2,results3,results4])


    result_df = pd.DataFrame()
    result_df['file_name'] = images_names
    result_df['result'] = results
    result_df[['file_name', 'result']].to_csv('./submit/text_sub_ensemble_tmp.csv', encoding='utf8', index=False)

