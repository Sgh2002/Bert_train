DEBUG=False
import os
if DEBUG==True:
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['ODPS_CONFIG_FILE_PATH'] = './utils/odps_config_pub.ini'
else:
    os.environ['ODPS_CONFIG_FILE_PATH'] = './utils/odps_config_inc.ini'
from dataclasses import dataclass, field
from typing import Optional
from transformers.utils.versions import require_version
from transformers import TrainerCallback
from datetime import datetime
from accelerate import Accelerator
import sys
import logging
import time
import json
import transformers
from transformers import AutoTokenizer
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed
)
from data_training_arguments import DataTrainingArguments
from model_arguments import ModelArguments

from dataset_loader_single import load_dataset_from_oss_files,load_dataset_from_local_files,load_dataset_from_odps_files,load_dataset_from_local_files_v2

from transformers import BertConfig, BertTokenizer,BertForPreTraining,BertModel, BertTokenizerFast,BatchEncoding
transformers.logging.set_verbosity_error()

from model.modeling_l4qtc import L4QtcElieModel
# [debug sunliuying]
from model.modeling_l4qtc_only_single import L4QtcOnlyElieModel


from model.configuration_l4qtc import L4qtcConfig
from model.configuration_l4qtc_only import L4qtcOnlyConfig

from urllib import parse
import torch
import torch.nn.functional as F
from typing import Dict,List
import copy 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from metric.metric_func_test import compute_metrics_with_querys, compute_metrics_with_querys_elie

from utils.oss_utils import OssUtils
import numpy as np
from utils.cache import Cache
from collections.abc import Iterable

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            torch.set_printoptions(profile="full")
            print(message, flush=True)
    else:
        torch.set_printoptions(profile="full")
        print(message, flush=True)


def init_peft(model):
    from peft import get_peft_model
    from peft import LoraConfig, TaskType   
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, peft_config)
    return model

class TimeStampLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 获取当前时间
        now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        # 打印日志（你可以用 logging.info 或 print，也可以加别的字段）
        print(f"{now} - {logs}", flush=True)
        # 如果用logger，自动带时间戳
        # logging.info(logs)

@dataclass
class DataCollatorForQtcDataset(object):
    """Collate examples for supervised fine-tuning."""

    return_tensors: str = "pt"

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
    
        out_put = {}
        for item in instances:
            for k,v in item.items():
                if k not in out_put:
                    out_put[k] = []
                out_put[k].append(v)
        
        for k, v in out_put.items():
            if k in ("distil_logits",'elie_distil_probs','term_match_score','elie_distil_logits', \
                'pair_distill_score', 'auth_distill_score') :
                out_put[k] = torch.FloatTensor(v)
            else:
                out_put[k] = torch.LongTensor(v)
        return out_put


def get_model_by_type(
        model_args,
        model_type,  
        qtc_init_checkpoint,
        model_checkpoint_path=None,
        bert_config=None,
        is_export=False):
    
    if model_type.startswith("l4_qtc_only"):
        config = L4qtcOnlyConfig()
        if model_checkpoint_path==None:
            model = L4QtcOnlyElieModel(config, qtc_init_checkpoint)
        else:
            bert_config = '/mnt/data/sunliuying/models/qtc_clsonly_v266_72b_3y_new_elie_auth/config.json'
            model = L4QtcOnlyElieModel.from_pretrained(model_checkpoint_path, 
                                                config=config, bert_config=bert_config, is_export=is_export)
    else:
        raise ValueError(f'{model_type} not support.')

    return model


def merge_title_body_summary(titles, bodys, elapsedmonths=None, summarys=None, urls=None):
    merge = []
    # 加入了时效特征
    if summarys not in (None,'None'):
        for title, body, elapsedmonth, summary, url in zip(titles, bodys, elapsedmonths, summarys, urls):
            host = parse.urlparse(url).hostname
            tcs = f"标题：{title}{host}。距离发文时间过去:{elapsedmonth}月。摘要：{summary}。正文：{body}"
            merge.append(tcs)
    else:
        for title, body, elapsedmonth in zip(titles, bodys, elapsedmonths):
            tcs = f"标题：{title}。距离发文时间过去:{elapsedmonth}月。正文：{body}"
            merge.append(tcs)
    return merge


def tokenize_function_l4_only(examples, tokenizer: BertTokenizer, data_args: DataTrainingArguments, model_type: str):
    output = {}
    tcs = merge_title_body_summary(examples[data_args.title_col], examples[data_args.body_col], 
            examples[data_args.elapsedmonth_col], examples[data_args.summary_col], examples[data_args.url_col])

    query_body = tokenizer(examples[data_args.query_col], tcs, padding="max_length", max_length=data_args.tc_max_len, truncation=True)
    output['qtc_id'] = query_body['input_ids']
    output['qtc_segment_id'] = query_body['token_type_ids']
    output['qtc_attention_mask'] = query_body['attention_mask']

    # point
    if data_args.label_col not in (None,'None') and data_args.label_col in examples:
        #output['labels'] = list(map(int, examples[data_args.label_col]))
        output['labels'] = list(map(float, examples[data_args.label_col]))
    if data_args.logit_col not in (None,'None') and data_args.logit_col in examples:
        output['distil_logits'] = examples[data_args.logit_col]
    # pair
    if data_args.qtc_pair_score_col not in (None,'None') and data_args.qtc_pair_score_col in examples:
        output['pair_distill_score'] = examples[data_args.qtc_pair_score_col]
    # elie
    if data_args.elie_label_col not in (None,'None') and data_args.elie_label_col in examples:
        output['elie_labels'] = list(map(int,examples[data_args.elie_label_col]))
        # print('output elie_labels raw', examples[data_args.elie_label_col])
        # print('output elie_labels ', output['elie_labels'])
    if data_args.elie_prob_col not in (None,'None') and data_args.elie_prob_col in examples:
        output['elie_distil_probs'] = examples[data_args.elie_prob_col]
    # auth
    if data_args.auth_score_col not in (None,'None') and data_args.auth_score_col in examples:
        output['auth_distill_score'] = examples[data_args.auth_score_col]
    
    
    return output


def merge_texts(querys, titles, bodys):
    res = []
    for query,title,body in zip(querys, titles, bodys):
        tcs = f"标题：{title}\n正文：{body}\n"
        res.append([query, tcs])
    return res


def main():
    if DEBUG==True:
        parser = HfArgumentParser((ModelArguments,DataTrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args,data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args,data_args = parser.parse_args_into_dataclasses()
        
        #override training_args for debug
        training_args = TrainingArguments(
            output_dir="./check_point/debug",
            do_train=True,
            do_eval=True,
            #do_predict=True,
            #self.bf16=True,
            #self.tf32=True,
            #self.torch_compile=True,
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            dataloader_num_workers=4,
            logging_steps=1,
            max_steps=10000000,
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,
            push_to_hub=False,
            overwrite_output_dir=True,
            eval_steps=100,
            save_steps=10,
            save_safetensors=False,
            dataloader_persistent_workers=True
        )
    else:
        parser = HfArgumentParser((ModelArguments,DataTrainingArguments,TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args,data_args,training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args,data_args,training_args = parser.parse_args_into_dataclasses()
    #setup logger ---------------------------------------------------------------------------------------------------------------------##

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    rank = os.getenv('RANK', 'null')
    logger.warning(
        f"Process rank: {training_args.local_rank}, global rank : {rank}, world size : {training_args.world_size}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set seed
    # Set seed before initializing model. ---------------------------------------------------------------------------------------------------------------------##
    set_seed(training_args.seed)
    
    #load dataset ---------------------------------------------------------------------------------------------------------------------##
    if data_args.label_col not in (None,'None') and data_args.logit_col not in (None,'None'):            
        logger.warning('we only accept one of label or label_logits, if both were provided, only use label_logits to compute distil loss')
    
    if training_args.do_train and (data_args.label_col is None and  data_args.logit_col is None):
        raise ValueError('in training mode , one of label_col,logit_col should be provided.')
    
    cache = Cache(data_args.oss_ak, data_args.oss_sk,data_args.oss_bucket, data_args.oss_endpoint, model_args.cache_dir)
    qtc_init_checkpoint = cache.get_cached_file_path(model_args.qtc_init_checkpoint)   
    bert_config = cache.get_cached_file_path(model_args.bert_config)
    vocab = cache.get_cached_file_path(model_args.bert_vocab)
    # sunliuying debug_sly
    vocab = '/mnt/data/sunliuying/models/qtc_clsonly_v266_72b_3y_new_elie_auth/'

    # [debug sunliuying] get bert tokenizer
    if qtc_init_checkpoint not in (None,'None'):
        tokenizer = BertTokenizerFast.from_pretrained(qtc_init_checkpoint)
    elif vocab not in (None,'None'):
        # use vocab 
        tokenizer = BertTokenizerFast.from_pretrained(vocab)
    else:
        raise ValueError('in order to init tokenizer , one of qt_init_checkpoint and bert_vocab parameter must be provided.')
    
    ##### 使用的[debug sunliuying] tokenizer的方式
    tokenize_func = tokenize_function_l4_only    
    # Preprocessing the datasets.----------------------------------------------------------------------------------------------------------##
    if data_args.train_file not in (None,'None') and training_args.do_train:
        # [debug sunliuying] 训练数据加载
        if data_args.training_data_format =='ids':
            selected_col = ['query_bert_id', 'query_segment_id', 'tc_bert_id','tc_segment_id','t_bert_id','t_segment_id', 'ac24l_rdrop_qc8948','ac24l_rdrop_qc8957','ac24l_rdrop_qc8981']
        else:
            selected_col = [data_args.query_col, data_args.url_col, data_args.title_col, data_args.body_col, data_args.summary_col]
            if data_args.logit_col not in (None,'None'):
                selected_col.append(data_args.logit_col)
            # 需要加上时间特征信号
            if data_args.elapsedmonth_col not in (None,'None'):
                selected_col.append(data_args.elapsedmonth_col)
            # [debug sunliuying] 可能采用多阶段训练的方式；每一阶段训练的头不一样，加载的训练数据不一样
            # 加载 qtc_pair_score 的标签数据
            if data_args.qtc_pair_score_col not in (None,'None'):
                selected_col.append(data_args.qtc_pair_score_col)
            # 加载 elie 相关的标签数据
            if 'elie' in model_args.model_type:
                if data_args.elie_prob_col not in (None,'None'):
                    selected_col.append(data_args.elie_prob_col)
                if  data_args.elie_logit_col not in (None,'None'):
                    selected_col.append(data_args.elie_logit_col)
                # if data_args.elie_label_col not in (None,'None'):
                #     selected_col.append(data_args.elie_label_col)
            # 加载 auth 相关的标签数据
            if data_args.auth_score_col not in (None,'None'):
                selected_col.append(data_args.auth_score_col)            

        if data_args.train_file.startswith('odps:'):
            train_dataset = load_dataset_from_odps_files(
                data_args.train_file,
                selected_col,shuffle=True, 
                fixed_slice_count=training_args.dataloader_num_workers*8,
                buffer_size=data_args.shuffle_buffer_size, seed=training_args.seed)
            
        elif data_args.train_file[0] not in ('/','.'):
            train_dataset= load_dataset_from_oss_files(
                data_args.train_file ,
                data_args.oss_ak, 
                data_args.oss_sk,
                data_args.oss_bucket,
                data_args.oss_endpoint,
                selected_col,
                shuffle=True,
                buffer_size=data_args.shuffle_buffer_size,
                seed=training_args.see
                )
        else:
            if data_args.train_file_v2:
                train_dataset = load_dataset_from_local_files_v2(data_args.train_file ,selected_col, shuffle=True, buffer_size=data_args.shuffle_buffer_size,seed=training_args.seed)
            else:
                train_dataset = load_dataset_from_local_files(data_args.train_file ,selected_col, shuffle=True, buffer_size=data_args.shuffle_buffer_size,seed=training_args.seed)

        map_func = tokenize_func
    
        train_dataset = train_dataset.map(
            map_func,
            batched=True,
            batch_size=1000,
            remove_columns=selected_col,
            fn_kwargs={'tokenizer':tokenizer,'data_args':data_args,'model_type':model_args.model_type}
        )        
    else:
        train_dataset=None
        
    if data_args.validation_file not in (None,'None') and training_args.do_eval:
        # validation file must have label
        selected_col = [data_args.query_col, data_args.title_col, data_args.body_col, data_args.url_col, data_args.label_col, data_args.source_col, data_args.summary_col]
        # [deubg sunliuying] 这里的 selected_col 加上时效特征
        if data_args.elapsedmonth_col not in (None,'None'):
            selected_col.append(data_args.elapsedmonth_col)
        # [deubg sunliuying] 需要评估哪些指标？？？
        if 'elie' in model_args.model_type:
            print("eval elie")
            if data_args.elie_label_col is None:
                raise ValueError(f'in {model_args.model_type} eval mode , elie_label_col must provied')
            selected_col.append(data_args.elie_label_col)

            # if data_args.elie_prob_col not in (None,'None'):
            #     selected_col.append(data_args.elie_prob_col)
        
            # if  data_args.elie_logit_col not in (None,'None'):
            #     selected_col.append(data_args.elie_logit_col)

        if data_args.validation_file.startswith('odps:'):
            validation_dataset = load_dataset_from_odps_files(data_args.validation_file, selected_col, is_iterable=False, fixed_slice_count=4)
        elif data_args.validation_file[0]  not in ('/','.'):
            validation_dataset = load_dataset_from_oss_files(
                                                            data_args.validation_file,
                                                            data_args.oss_ak, 
                                                            data_args.oss_sk,
                                                            data_args.oss_bucket,
                                                            data_args.oss_endpoint,
                                                            selected_col,
                                                            is_iterable=False
                                                            )
        else:
            print("load file from local")
            validation_dataset = load_dataset_from_local_files(data_args.validation_file, selected_col, is_iterable=False)

        all_querys = validation_dataset[data_args.query_col]
        all_sources = validation_dataset[data_args.source_col]
        all_urls = validation_dataset[data_args.url_col]
        all_float_labels = validation_dataset[data_args.label_col]
        #print(all_float_labels)

        with training_args.main_process_first(desc="dataset map tokenization"):
            print("eval tokenizer here")
            validation_dataset = validation_dataset.map(
                                            tokenize_func,
                                            batched=True,
                                            batch_size=1000,
                                            num_proc=4,
                                            load_from_cache_file=False,
                                            keep_in_memory=False,
                                            remove_columns=selected_col,
                                            fn_kwargs={'tokenizer':tokenizer,'data_args':data_args, 'model_type':model_args.model_type}
                                        )
            #print(validation_dataset["labels"])
    else:
        validation_dataset=None

    if  data_args.prediction_file not in (None,'None') and training_args.do_predict:
        # prediction input do not need label
        # [deubg sunliuying] 这里的 selected_col 需要加上新的特征
        selected_col = [data_args.query_col, data_args.url_col, data_args.title_col, data_args.body_col, data_args.summary_col]
        if data_args.elapsedmonth_col not in (None,'None'):
            selected_col.append(data_args.elapsedmonth_col)

        if data_args.prediction_file.startswith('odps:'):
            org_predict_dataset = load_dataset_from_odps_files(data_args.prediction_file,selected_col,is_iterable=False, fixed_slice_count=training_args.dataloader_num_workers)
   
        elif data_args.prediction_file[0]  not in ('/','.'):
            org_predict_dataset = load_dataset_from_oss_files(
                data_args.prediction_file,
                data_args.oss_ak, 
                data_args.oss_sk,
                data_args.oss_bucket,
                data_args.oss_endpoint,
                selected_col,
                is_iterable=False
                )  
        else:
            org_predict_dataset = load_dataset_from_local_files(data_args.prediction_file,selected_col,is_iterable=False)
        
        all_query = org_predict_dataset[data_args.query_col]
        all_urls = org_predict_dataset[data_args.url_col]
        with training_args.main_process_first(desc="dataset map tokenization"):
            
            predict_dataset = org_predict_dataset.map(
                                        tokenize_func,
                                        batched=True,
                                        batch_size=1000,
                                        remove_columns=selected_col,
                                        fn_kwargs={'tokenizer':tokenizer,'data_args':data_args,'model_type':model_args.model_type}
                                    )
    else:
        predict_dataset=None
    
    #init model         
    if model_args.model_name_or_path not in (None,'None'):        
        model_name_or_path = cache.get_cached_file_path(model_args.model_name_or_path)
    else:
        model_name_or_path = None
    
    model = get_model_by_type(
        model_args,
        model_args.model_type,
        qtc_init_checkpoint,
        model_checkpoint_path=model_name_or_path,
        bert_config=bert_config,
        is_export=False
        )  
    # 打印所有可训练参数名  
    all_param_names = [name for name, param in model.named_parameters() if param.requires_grad]  
    #print("sunliuying debug all_param_names", all_param_names) 
    def print_params_grad(model):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad not in (None,'None'):
                print(f"{name} 梯度范数: {param.grad.norm().item()}")
            else:
                print(f"{name} 未被更新")
    # 训练循环前打印初始值
    #print("===== 训练前 =====")
    #print_params_grad(model)
    
    if training_args.do_train and train_dataset is None:
        raise ValueError("--do_train requires a train dataset")

    if training_args.do_eval and validation_dataset is None:
        raise ValueError("--do_eval requires a validation dataset")
    
    if training_args.do_predict and predict_dataset is None:
        raise ValueError("--do_predict requires a predict dataset")
        
    # Initialize our Trainer        
    # accelerator = Accelerator()
    # model = accelerator.prepare(model)

    data_collator = DataCollatorForQtcDataset()
    # [debug sunliuying] compute metric
    metrics_func = compute_metrics_with_querys_elie if 'elie' in model_args.model_type else compute_metrics_with_querys

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        callbacks=[TimeStampLogCallback()],
        compute_metrics=metrics_func(all_querys, all_sources, all_urls, all_float_labels) if training_args.do_eval else None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint not in (None,'None'):
            checkpoint = cache.get_cached_file_path(training_args.resume_from_checkpoint)
            checkpoint = training_args.resume_from_checkpoint
            print("debug checkpoint", checkpoint)
            
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # 保存训练前参数  
        pre_train_param = {n: p.detach().clone() for n, p in model.named_parameters()}
        train_result = trainer.train()
        # 对比参数变化
        print("===== 训练后 =====")
        for name, param in model.named_parameters():
            if not torch.equal(param, pre_train_param[name]):
                print(f"{name} 发生了更新")
            else:
                print(f"{name} 未更新") 
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        #different evaluate metric compare to the point wise evaluation metric in the trainer loop 
        #given a query  and doc list , compute the pair acc , ndcg, dcg metric globaly on the data_args.validation_file
        trainer.evaluate()
    
    if training_args.do_predict:
        predictions, _, _ = trainer.predict(predict_dataset)
        print_rank_0("*** Predict ***")
        # [debug sunliuying]new_add  
        merge_sim, ts_merge_sim, elie_sim, auth_sim = predictions 
        
        merge_sim = torch.tensor(merge_sim, dtype=torch.float32)  # 转换为 PyTorch 张量 
        ts_merge_sim = torch.tensor(ts_merge_sim, dtype=torch.float32)
        elie_sim = torch.tensor(elie_sim, dtype=torch.float32)
        auth_sim = torch.tensor(auth_sim, dtype=torch.float32)

        def calc_prob(sim):
            if sim.shape[-1] == 1:
                return sim
            elif sim.shape[-1] == 2:
                return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1.]), dim=-1)
            elif sim.shape[-1] == 3:
                return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1., 2.]), dim=-1)
            elif sim.shape[-1] == 6:
                return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 0., 1., 2., 3., 4.]), dim=-1)
            else:
                return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1., 2., 3., 4.]), dim=-1)

        merge_sim_prob = F.softmax(merge_sim, dim=-1)
        merge_sim_pred_labels = merge_sim.argmax(-1)
        
        elie_prob = F.softmax(elie_sim, dim=-1)
        elie_pred_labels = elie_sim.argmax(-1)
            
        merge_sim_prob_new = torch.cat([
            (merge_sim_prob[:, 0] * elie_prob[:, 0]).unsqueeze(-1),
            (merge_sim_prob[:, 0] * elie_prob[:, 1]).unsqueeze(-1),
            merge_sim_prob[:, 1:]
        ], dim=-1)

        examples = {
            "query": all_query,
            "url": all_urls,
            "0_merge": calc_prob(merge_sim),
            "1_ts_merge": calc_prob(ts_merge_sim),
            
            "logits_0_merge": merge_sim_prob_new,
            "logits_1_ts_merge": ts_merge_sim,
            "logits_6_elie_sim": elie_prob,
            "logits_8_auth_sim": auth_sim,
            "merge_argmax_label": torch.argmax(merge_sim, dim=-1).float()
        }

        def convert_tensor(data):
            if isinstance(data, torch.Tensor):
                if data.numel() == 1:
                    return data.item()
                return data.cpu().numpy().tolist()
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
                return [convert_tensor(item) for item in data]
            return data

        if trainer.is_world_process_zero():
            all_ouputs = []
            # 一致性验证
            batch_sizes = [len(v) for v in examples.values()]
            # if len(set(batch_sizes)) > 1:
            #     raise ValueError(f"不一致的批次大小: {batch_sizes}")
            # 按索引逐条展开
            all_outputs = [
                json.dumps({k: convert_tensor(v[i]) for k, v in examples.items()}, ensure_ascii=False)
                for i in range(batch_sizes[0])
            ]
            # print(all_outputs)
            all_outputs = '\n'.join(all_outputs)
            # print(all_outputs)

            if data_args.predict_output_file[0] not in ('/','.'):
                #oss file
                oss = OssUtils(
                        data_args.oss_ak,
                        data_args.oss_sk,
                        data_args.oss_bucket,
                        data_args.oss_endpoint)
                f = oss.open_oss_file(data_args.predict_output_file, 'w')
                f.write(all_outputs)       
            else:
                #local file
                with open(data_args.predict_output_file,'w') as f:
                    f.write(all_outputs)
            print_rank_0("*** Predict  Done***")
        

if __name__ == "__main__":
    main()
