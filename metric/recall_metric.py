
from typing import List
from operator import itemgetter

def recall_hit_rate(labels: List[int], scores: List[float], querys: List[str], pos_label_from=1, top_k=10):


    def compute_recall(query_data, top_k):

        items = query_data['item']

        items = sorted(items, key=itemgetter('score'),reverse=True)
        top_k_items = items[0:top_k]

        recalls = 0
        for item in top_k_items:

            if item['label'] == 1:
                recalls += 1

        return recalls


    query_map = {}

    for label, score, query in zip(labels,scores, querys) :

        label = 1 if label >=pos_label_from else 0

        if query not in query_map:
            query_map[query] = {'all_count':0,'pos_count':0, 'item':[]}

        query_map[query]['item'].append({'label':label,'score':score})
        query_map[query]['all_count'] += 1
        query_map[query]['pos_count'] += label

    all_pos_count = 0
    all_recall_count = 0
    all_pred_count = 0
    for query, value in query_map.items():

        all_pos_count += (value['pos_count'] if value['pos_count'] < top_k else top_k)
    
        recall = compute_recall(value, top_k)
        all_recall_count += recall
        all_pred_count += (0 if value['pos_count']==0 else top_k)
    
    if all_pos_count ==0:
        return 0,0
    else:
        recall_ratio =  all_recall_count/all_pos_count
        hit_ratio = all_recall_count/all_pred_count
        return recall_ratio,hit_ratio
