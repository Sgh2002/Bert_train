from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metric import metric
from metric.recall_metric import recall_hit_rate
import numpy as np

from scipy.special import softmax, expit 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            torch.set_printoptions(profile="full")
            print(message, flush=True)
    else:
        torch.set_printoptions(profile="full")
        print(message, flush=True)

# [sunliuying debug] 这个函数暂时没有修改
def compute_metrics_with_querys(all_querys, all_data_source,all_urls, all_float_labels):   
    def compute_metrics(eval_preds):
    
        # compute will recive all eval sample     
        labels = eval_preds.label_ids
        preds, qtc_score = eval_preds.predictions
        preds = preds.argmax(-1)
        
        source_group ={} 

        print("all_float_labels")
        print(all_float_labels)
        for label, pred, score,query,url, source in zip(all_float_labels.tolist(), preds.tolist(),qtc_score.tolist(), all_querys,all_urls, all_data_source):

            if source not in source_group:
                source_group[source] = {'labels':[], 'preds':[], 'qtc_scores':[], 'querys':[], 'urls':[]}

            source_group[source]['labels'].append(label)
            source_group[source]['qtc_scores'].append(score)
            source_group[source]['querys'].append(query)
            source_group[source]['urls'].append(url)
            source_group[source]['preds'].append(pred)
        
        metric_values = {}
        
        for key, values in source_group.items():
            labels = values['labels']
            qtc_scores = values['qtc_scores']
            querys = values['querys']
            urls = values['urls']
            preds = values['preds']

            #accuracy = accuracy_score(labels, preds)     
            # Calculate precision, recall, and F1-score
            #precision = precision_score(labels, preds, average=None)
            #recall = recall_score(labels, preds, average=None)
            #f1 = f1_score(labels, preds, average=None) 

            pairacc, pairacc_hard = metric.UserMetric.seq_pair(labels, qtc_scores, querys, urls=urls)
            #dcg3 = metric.UserMetric.dcg(labels, qtc_scores, querys,urls=urls,dcg_pos=3, dcg_type='dcg')
            #ndcg3 = metric.UserMetric.dcg(labels, qtc_scores, querys,urls=urls,dcg_pos=3, dcg_type='ndcg')
            #dcg20 = metric.UserMetric.dcg(labels, qtc_scores, querys,urls=urls,dcg_pos=20, dcg_type='dcg')
            #ndcg20 = metric.UserMetric.dcg(labels, qtc_scores, querys,urls=urls,dcg_pos=20, dcg_type='ndcg')

            label1_recall_rate10,label1_hitratio_rate10 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=1, top_k=10)
            label2_recall_rate10,label2_hitratio_rate10 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=2, top_k=10)

            label1_recall_rate20,label1_hitratio_rate20 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=1, top_k=20)
            #label2_recall_rate20,label2_hitratio_rate20 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=2, top_k=20)

            metric_values[key] = {
                    'label>=1,recall@10':label1_recall_rate10, 
                    #'label>=2,recall@10':label2_recall_rate10, 

                    'label>=1,recall@20':label1_recall_rate20, 
                    #'label>=2,recall@20':label2_recall_rate20, 

                    #'label>=1,hit@10':label1_hitratio_rate10, 
                    #'label>=2,hit@10':label2_hitratio_rate10, 

                    #'label>=1,hit@20':label1_hitratio_rate20, 
                    #'label>=2,hit@20':label2_hitratio_rate20, 

                    'pairacc':pairacc, 
                    'pairacc_hard':pairacc_hard,
                    #'ndcg3':ndcg3,
                    #'ndcg20':ndcg20,
                    #'precision':precision.tolist(),
                    #'recall':recall.tolist(),
                    }
            
            flatten_metric = {}
            
            for key, values in metric_values.items():

                for metric_name, metric_value in values.items():

                    flatten_metric[key+'_'+metric_name]=metric_value



        
        return flatten_metric
    
    return compute_metrics


def calc_qtc_score(logits: torch.Tensor, num_classes) -> torch.Tensor:

    x = torch.softmax(logits,dim=-1)
    y = torch.FloatTensor(list(range(0, num_classes))).to(x.device)

    score = torch.sum(x*y,dim=-1)
    
    return score


def compute_metrics_with_querys_elie(all_querys, all_data_source, all_urls, all_float_labels):
    def compute_metrics(eval_preds):
    
        # compute will recive all eval sample     
        labels, elie_labels = eval_preds.label_ids
        merge_sims, ts_merge_sims, elie_sims, auth_sims = eval_preds.predictions
        
        pred_logits = merge_sims
        preds = pred_logits.argmax(-1)
        qtc_score = calc_qtc_score(torch.from_numpy(merge_sims), num_classes=5)
        
        elie_pred_logits = elie_sims
        elie_preds = elie_pred_logits.argmax(-1)
     
        source_group ={}
        bad_predict = 0
        for idx, (label, pred, score, elie_label, elie_pred, elie_pred_logit, query, url, source) in enumerate(zip(
            all_float_labels, 
            preds.tolist(),
            qtc_score.tolist(), 
            elie_labels.tolist(),
            elie_preds.tolist(),
            elie_pred_logits.tolist(),
            all_querys,
            all_urls, 
            all_data_source)):

            if source not in source_group:
                source_group[source] = {
                    'labels':[], 
                    'preds':[], 
                    'qtc_scores':[], 
                    'querys':[], 
                    'urls':[],
                    'elie_labels':[],
                    'elie_preds':[],
                    'elie_pred_logits':[],
                    }
            # check 
            if source in ('elie','vip') and elie_label==1 and pred==0 and elie_pred==0:
                # print('ac score correct, but elie_pred wrong')
                # elie_pred = 1
                bad_predict += 1
            
            source_group[source]['labels'].append(label)
            source_group[source]['qtc_scores'].append(score)
            source_group[source]['querys'].append(query)
            source_group[source]['urls'].append(url)
            source_group[source]['preds'].append(pred)
            # source_group[source]['elie_losses'].append(a_elie_loss)
            source_group[source]['elie_labels'].append(elie_label)
            source_group[source]['elie_preds'].append(elie_pred)
            source_group[source]['elie_pred_logits'].append(elie_pred_logit)
            # source_group[source]['ac_losses'].append(a_ac_loss)

        print(f'elie bad_predict: {bad_predict}')
        
        metric_values = {}
        
        for idx, (key, values) in enumerate(source_group.items()):
            labels = values['labels']
            qtc_scores = values['qtc_scores']
            querys = values['querys']
            urls = values['urls']
            preds = values['preds']
            # elie_losses = values['elie_losses']
            elie_labels = values['elie_labels']
            elie_preds = values['elie_preds']
            elie_pred_logits = values['elie_pred_logits']
            # ac_losses = values['ac_losses']

            metric_values[key] = {}

            if sum(labels) >0: # it is ac eval set
                int_labels = [int(v) for v in labels]
                t = classification_report(int_labels, preds)
                print_rank_0(t)
                print_rank_0(confusion_matrix(int_labels, preds))

                pairacc, pairacc_hard = metric.UserMetric.seq_pair(labels, qtc_scores, querys, urls=urls)
            
                label1_recall_rate10,label1_hitratio_rate10 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=1, top_k=10)
                label2_recall_rate10,label2_hitratio_rate10 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=2, top_k=10)

                label1_recall_rate20,label1_hitratio_rate20 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=1, top_k=20)
                #label2_recall_rate20,label2_hitratio_rate20 = recall_hit_rate(labels, qtc_scores,querys, pos_label_from=2, top_k=20)
                
                # ac_loss = float(np.array(ac_losses).mean()) 

                if key not in ('qc_new2', 'tongshi', 'elie'):
                    metric_values[key]['label>=1,recall@10']=label1_recall_rate10
                
                metric_values[key]['pairacc'] = pairacc
                metric_values[key]['pairacc/h'] = pairacc_hard
                # metric_values[key]['ac_loss'] = ac_loss
            
            elif sum(elie_labels) >0: # it is elie eval set
          
                elie_precision = precision_score(elie_labels, elie_preds)
                elie_recall = recall_score(elie_labels, elie_preds)
                f1_scores = f1_score(elie_labels, elie_preds)

                metric_values[key]['elie_precision'] = elie_precision
                metric_values[key]['elie_recall'] = elie_recall
                metric_values[key]['f1'] = f1_scores

                # metric_values[key]['elie_losses']=float(np.array(elie_losses).mean())

                # add pair acc 
                elie_labels = list(map(lambda x: 0 if x==1 else 1, elie_labels))
                elie_score = []
    
                for elie_logit in elie_pred_logits:
                    elie_logit = [elie_logit[1], elie_logit[0]]
                    score = calc_qtc_score(torch.tensor(elie_logit), num_classes=2)
                    score = float(score.numpy())
                    elie_score.append(score)
                
                pairacc, pairacc_hard = metric.UserMetric.seq_pair(elie_labels, elie_score, querys, urls=urls)
                metric_values[key]['pairacc'] = pairacc
                metric_values[key]['pairacc/h'] = pairacc_hard

        flatten_metric = {}
        for key, values in metric_values.items():
            for metric_name, metric_value in values.items():
                flatten_metric[key+'_'+metric_name]=metric_value        
        return flatten_metric
    
    return compute_metrics
