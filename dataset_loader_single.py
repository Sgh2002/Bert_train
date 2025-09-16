from datasets import IterableDataset, IterableDatasetDict, load_dataset, Dataset
from data_training_arguments import DataTrainingArguments
from utils.oss_utils import OssUtils
import io
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist

from typing import Tuple, Dict,List
import os
from datasets.distributed import split_dataset_by_node
import numpy as np
import random
import common_io


logger = logging.getLogger(__name__)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def oss_gen(shards: List[str] , oss_kwargs: Dict[str, str], selected_col: Tuple[str]):
        
        rank = os.getenv('RANK', 'null')
        local_rank = os.getenv('LOCAL_RANK', 'null')

        oss = OssUtils(
        oss_kwargs['oss_ak'],
        oss_kwargs['oss_sk'],
        oss_kwargs['oss_bucket'],
        oss_kwargs['oss_endpoint']) 

        if len(shards) < 100:
            print(f'oss shards:{shards}------------------------------------')
            
        for shard in shards:


            oss_file = oss.open_oss_file(shard,selected_col=selected_col) 
            print(f'rank {rank},local rank {local_rank}:start to read local file {shard}')

            for line in oss_file:
                try:
                   line = json.loads(line)
                except:
                    #logger.warning(f"data row has invalid json format: {line}")
                    continue

                if selected_col is not None:
                    is_valid_record = True
                    for col in  selected_col:

                        if col not in line:
                            #print(f'warning :skip record : column {col} not in : {line}')
                            is_valid_record=False
                            break
                
                    if is_valid_record==False:
                        print(f'warning skip records: {line}')
                        
                        continue
                else:
                    new_line = row

                yield line

def odps_gen(shards: List[str], selected_col: Tuple[str], table_name: str,slice_count: int):

    
    rank = os.getenv('RANK', 'null')
    local_rank = os.getenv('LOCAL_RANK', 'null')

    if len(shards) < 100:
        print(f"rank {rank}, local rank {local_rank}:shards:{shards}------------------------------------")
    

    for shard in shards:
        reader = common_io.table.TableReader(
                                                table_name,
                                                selected_cols="json_content",
                                                slice_id=shard,
                                                slice_count=slice_count,
                                                num_threads=8,
                                                capacity=1024)


        # try:
        #     reader = common_io.table.TableReader(
        #                                         table_name,
        #                                         selected_cols="json_content",
        #                                         slice_id=shard,
        #                                         slice_count=slice_count,
        #                                         num_threads=8,
        #                                         capacity=1024)
        # except Exception as e:
        #     print(f"Error: {str(e)}")

        
        print(f'rank {rank},local rank {local_rank}:start to read odps slice {shard},total row: {reader.get_row_count()}')

        while True:
            try:
                lines = reader.read(num_records=32, allow_smaller_final_batch=True)

                for row in lines:
                    row = row[0]
                    try:
                        row = json.loads(row)
                    except Exception as e:
                        logger.error(f"{str(e)}, row={row}")
                        #logger.warning(f"data row has invalid json format: {line}")
                        continue
                    new_line = {}
                   
                    if selected_col is not None:
                        is_valid_record = True
                        for col in  selected_col:

                            if col not in row:
                                print(f'warning :skip record : column {col} not in : {row}')
                                is_valid_record=False
                                break
                            new_line[col] = row[col]
                    
                        if is_valid_record==False:
                            print(f'warning skip records')
                            continue
                    else:
                        new_line = row

                    yield new_line

            except common_io.exception.OutOfRangeException:
                reader.close()
                break
        
        reader.close()

def local_file_gen(shards: List[str], selected_col: Tuple[str], resume_point:Tuple[str] = None):

    rank = os.getenv('RANK', 'null')
    local_rank = os.getenv('LOCAL_RANK', 'null')

    if len(shards) < 100:
        print(f"rank {rank}, local rank {local_rank}:shards:{shards}------------------------------------")
    
    for shard in shards:
        """   
        if resume_point is not None:

            if is_start == False:
                if Path(shard).name in resume_point:
                    is_start =True
                    print(f'rank {rank},start from shard :{shard}')


            if is_start == False:
                continue
        """        
        with open(shard, 'rb') as f:
            print(f'rank {rank},local rank {local_rank}:start to read local file {shard}')
            for idx,line in enumerate(f):
                try:
                    line = line.decode('utf-8')
                except:
                    # logger.warning(f"data row has invalid character: {line}")
                    continue
                # try:
                line = json.loads(line)
                ##### sunliuying 临时测试加入
                if "elie_label" not in line:
                    line["elie_label"] = 0
                else:
                    # 二分类 标签却为1 / 2
                    line["elie_label"] = line["elie_label"] - 1
                    if "label" not in line:
                        line["label"] = line["elie_label"]
                # 数据有问题，打补丁
                if "logits" in line and len(line["logits"]) != 5:
                    continue
                if "summary" not in line:
                    if 'sumary' in line:
                        line["summary"] = line["sumary"]
                    elif "wenda_summary" in line:
                        line["summary"] = line["wenda_summary"]
                    else:
                        line["summary"] = ""
                if 'label' not in line and "pred_label" in line:
                    line['label'] = line["pred_label"]
                if "label" in line:
                    line["label"] = float(line["label"])
                if "elapsedDays" not in line:
                    if "elapsedMonth" in line:
                        line["elapsedDays"] = int(line["elapsedMonth"])
                    else:
                        line["elapsedDays"] = 38
                else:
                    if line["elapsedDays"] == -999:
                        line["elapsedDays"] = 38
                if "data_source" not in line:
                    if "filename" in line:
                        line["data_source"] = line["filename"]
                else:
                    line["data_source"] = 'train'
                # except:
                #     # logger.warning(f"data row has invalid json format: {line}")
                #     # print(f"data row has invalid json format: {line}")
                #     continue
                new_line = {}
               
                # print('selected_col', selected_col)
                if selected_col is not None:
                    is_valid_record = True
                    for col in selected_col:

                        if col not in line:
                            print(f'warning :skip record : column {col} not in : {col}')
                            is_valid_record=False
                            break
                        # # sunliuying, 针对测试数据临时加的[0]因为在数据读取的时候会出问题，变成 'elie_probs': ([0.9252299570920386, 0.07477004290796137],)
                        # if col == 'elie_probs':
                        #     new_line[col] = line[col][0]
                        # else:
                        #     new_line[col] = line[col]
                        new_line[col] = line[col]
                        if line[col] in (None, 'None'):
                            print(line[col])      
                    if is_valid_record==False:
                        #print(f'warning skip records')
                        continue
                else:
                    new_line=line

                # print("new_line", new_line)
                yield new_line
  

     
def local_file_gen_v2(shards: List[str], selected_col: Tuple[str], resume_point:Tuple[str] = None):

    rank = os.getenv('RANK', 'null')
    local_rank = os.getenv('LOCAL_RANK', 'null')

    if len(shards) < 100:
        print(f"rank {rank}, local rank {local_rank}:shards:{shards}------------------------------------")
    
    for shard in shards:   
        with open(shard, 'rb') as f:
            print(f'rank {rank},local rank {local_rank}:start to read local file {shard}')

            for idx,line in enumerate(f):
                try:
                    line = line.decode('utf-8')
                except:
                    logger.warning(f"data row has invalid character")
                    continue

                # try:
                #    items = line.strip().split('\t')
                #    query, url, jl_str, pair_str = items
                #    jl = json.loads(jl_str)
                #    line = {'query': query, 
                #            'url': url, 
                #            'title': jl['title'], 
                #            'body': jl['body'], 
                #            'logits': jl['logits'], 
                #            'elie_probs':[-1.0,-1.0],
                #            'elie_label':-1,
                #            'data_source': 'youzhi'}
                # except:
                #     logger.warning(f"data row has invalid format: {line}")
                #     continue
                try:
                    url_idx = line.index('/thttp')
                    jl_idx = line.index('/t{"')
                    pair_idx = line[jl_idx:].index('}/t')
                    query = line[:url_idx]
                    url = line[url_idx+2:jl_idx]
                    jl_str = line[jl_idx+2:jl_idx+pair_idx+1]
                    jl = json.loads(jl_str)
                    line = {'query': query, 
                            'url': url, 
                            'title': jl['title'], 
                            'body': jl['body'],
                            'elapsedDays': 12,  
                            'logits': jl['logits'], 
                            'elie_probs':[0.1,0.9],
                            'elie_label':0,
                            # 'pair_score': -1.35546875,
                            'auth_logits': 0.9388889670372009,
                            'data_source': 'youzhi'}
                    if len(line['logits']) != 5:
                        # print('logits size is not 5', line['logits'])
                        continue
                except Exception as e:
                    # print('data row has invalid format')
                    continue


                new_line = {}

               
                if selected_col is not None:
                    is_valid_record = True
                    for col in  selected_col:

                        if col not in line:
                            #print(f'warning :skip record : column {col} not in : {line}')
                            is_valid_record=False
                            break
                        new_line[col] = line[col]
                
                    if is_valid_record==False:
                        #print(f'warning skip records')
                        continue
                else:
                    new_line=line

                yield new_line

def load_dataset_from_local_files_v2(data_path: str,selected_col: Tuple=None,is_iterable=True,resume_point : str=None, ext='json', shuffle=False,buffer_size=1,seed=42):

    def list_files(file_path: str) -> List[str]:

        path = Path(file_path)

        if path.is_dir():
            files = list(path.rglob('*'))
        else:
            files = [str(path)]
        
        return files

    shards = list_files(data_path)

    if len(shards) == 0:
        raise ValueError('no shard to process.')


    #convert list to tuple to avoid shards error 
    if selected_col is not None:
        selected_col = tuple(selected_col)

    if resume_point is not None:
        resume_point=tuple(resume_point.split(','))
        print(resume_point)
    
    

    if is_iterable:
        raw_dataset = IterableDataset.from_generator(local_file_gen_v2, gen_kwargs={'shards':shards,'selected_col':selected_col,'resume_point':resume_point})
        """
                        There are two ways an iterable dataset can be split by node:

                if the number of shards is a factor of number of GPUs: in that case the shards are evenly distributed per GPU
                otherwise, each GPU iterate on the data and at the end keeps 1 sample out of n(GPUs) - skipping the others.
                In case 2. it's therefore possible to have the same examples passed to prepare_dataset for each GPU.

                This doesn't sound optimized though, because it runs the preprocessing on samples that won't be used in the end.
        """

        if shuffle:
            raw_dataset=raw_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    else:
        raw_dataset = Dataset.from_generator(local_file_gen_v2, gen_kwargs={'shards':shards,'selected_col':selected_col})

        if shuffle:
            raw_dataset=raw_dataset.shuffle(seed=seed)
    
    return raw_dataset
def load_dataset_from_local_files(data_path: str,selected_col: Tuple=None,is_iterable=True,resume_point : str=None, ext='json', shuffle=False,buffer_size=1,seed=42):

    def list_files(file_path: str) -> List[str]:

        path = Path(file_path)

        if path.is_dir():
            files = list(path.rglob('*'))
        else:
            files = [str(path)]
        
        return files

    shards = list_files(data_path)

    if len(shards) == 0:
        raise ValueError('no shard to process.')


    #convert list to tuple to avoid shards error 
    if selected_col is not None:
        selected_col = tuple(selected_col)

    if resume_point is not None:
        resume_point=tuple(resume_point.split(','))
        print(resume_point)
    
    

    if is_iterable:
        raw_dataset = IterableDataset.from_generator(local_file_gen, gen_kwargs={'shards':shards,'selected_col':selected_col,'resume_point':resume_point})
        """
                        There are two ways an iterable dataset can be split by node:

                if the number of shards is a factor of number of GPUs: in that case the shards are evenly distributed per GPU
                otherwise, each GPU iterate on the data and at the end keeps 1 sample out of n(GPUs) - skipping the others.
                In case 2. it's therefore possible to have the same examples passed to prepare_dataset for each GPU.

                This doesn't sound optimized though, because it runs the preprocessing on samples that won't be used in the end.
        """

        if shuffle:
            raw_dataset=raw_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    else:
        raw_dataset = Dataset.from_generator(local_file_gen, gen_kwargs={'shards':shards,'selected_col':selected_col})

        if shuffle:
            raw_dataset=raw_dataset.shuffle(seed=seed)
    
   

    return raw_dataset


def load_dataset_from_oss_files(
                            data_path:str,
                            oss_ak:str,
                            oss_sk:str, 
                            oss_bucket:str, 
                            oss_endpoint:str,
                            selected_col : Tuple[str]=None, 
                            is_iterable=True, 
                            resume_point : str=None,
                            shuffle=False,
                            buffer_size=1,
                            seed=42
    ):

    #get files from oss_folder
    oss = OssUtils(
        oss_ak,
        oss_sk,
        oss_bucket,
        oss_endpoint)
    
    oss_kwargs = {'oss_ak':oss_ak,'oss_sk':oss_sk,'oss_bucket':oss_bucket,'oss_endpoint':oss_endpoint}
    
    #convert list to tuple to avoid shards error 
    if selected_col is not None:
        selected_col = tuple(selected_col)

    shards = oss.list_objects(data_path)
    if is_iterable:
        raw_dataset = IterableDataset.from_generator(oss_gen, gen_kwargs={'shards':shards,'oss_kwargs':oss_kwargs,'selected_col':selected_col})
        #raw_dataset = split_dataset_by_node(raw_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        if shuffle:
            raw_dataset=raw_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    else:
        raw_dataset = Dataset.from_generator(oss_gen, gen_kwargs={'shards':shards,'oss_kwargs':oss_kwargs,'selected_col':selected_col})
        #raw_dataset = split_dataset_by_node(raw_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        if shuffle:
            raw_dataset=raw_dataset.shuffle(seed=seed)


    return  raw_dataset

def load_dataset_from_odps_files(
                            data_path:str,
                            selected_col : Tuple[str]=None, 
                            is_iterable=True, 
                            resume_point : str=None,
                            shuffle=False,
                            fixed_slice_count=32,
                            buffer_size=1,
                            seed=42
    ):

    if selected_col is not None:
        selected_col = tuple(selected_col)

    shards = list(range(0, fixed_slice_count))
    
    if is_iterable:
        raw_dataset = IterableDataset.from_generator(odps_gen, gen_kwargs={'shards':shards,'selected_col':selected_col,'table_name':data_path,'slice_count':fixed_slice_count})
        #raw_dataset = split_dataset_by_node(raw_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        if shuffle:
            raw_dataset=raw_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    else:
        raw_dataset = Dataset.from_generator(odps_gen, gen_kwargs={'shards':shards,'selected_col':selected_col,'table_name':data_path,'slice_count':fixed_slice_count})
        #raw_dataset = split_dataset_by_node(raw_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        if shuffle:
            raw_dataset=raw_dataset.shuffle(seed=seed)
    
    return raw_dataset


if __name__ == "__main__":    
    # from transformers import (
    # TrainingArguments,
    # HfArgumentParser,
    # Trainer,
    # default_data_collator,
    # set_seed
    # )

    # from torch.utils.data.dataloader import DataLoader

    # parser = HfArgumentParser((DataTrainingArguments))
    # data_args = parser.parse_args_into_dataclasses()[0]
    
    # selected_col = [data_args.query_col, data_args.title_col, data_args.body_col]

    # dataset = load_dataset_from_oss_files(data_args,selected_col)
    selected_col = ['query', 'url', 'json_content']
    #odps://${your_projectname}/tables/${table_name}
    dataset = load_dataset_from_odps_files('odps://aliyun_searchlog/tables/gx_qtc_distill_3e_v266_all_logits_shuf',selected_col)
    for data in dataset:
        print(data)

    # dataloader = DataLoader(dataset['train'].with_format("torch"), num_workers=4)

    # for data in dataloader:
    #     print(data)
