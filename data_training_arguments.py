from dataclasses import dataclass, field
from typing import Optional
from transformers.utils.versions import require_version
import sys
import os

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default='odps://aliyun_searchlog/tables/L3_L4_training_data_final/ds=elie_v2_1800w_training', metadata={"help": "The input training data file (a text file)."})
    training_data_format: Optional[str] = field(default='raw', metadata={"help": "the data format: raw, ids(for the training data converted from stdkv format)."})
    validation_file: Optional[str] = field(default='odps://aliyun_searchlog/tables/L3_L4_eval_data_final/ds=elie_vip',metadata={"help": "validation file."})
    prediction_file: Optional[str] = field(default='odps://aliyun_searchlog/tables/L3_L4_eval_data_final/ds=elie_vip',metadata={"help": "prediction file"})
    predict_output_file: Optional[str] = field(default='QTC/yongbo/eval_data/elie_vip_predict.json',metadata={"help": "prediction file"})

    # oss_ak
    # oss_sk
    oss_bucket:Optional[str] = field(default='quark-search-tech', metadata={"help": "The oss bucket."})
    oss_endpoint:Optional[str] = field(default='http://oss-cn-wulanchabu.aliyuncs.com', metadata={"help": "The oss endpoint."})
    #cn-wulanchabu.oss.aliyuncs.com

    query_col:Optional[str] = field(default='query',metadata={"help": "query column"})
    query_max_len:Optional[int] = field(default=30,metadata={"help": "query ids max len"})
    title_col:Optional[str] = field(default='title',metadata={"help": "title column"})
    body_col:Optional[str] = field(default='body',metadata={"help": "body column"})
    summary_col:Optional[str] = field(default='summary',metadata={"help": "summary column"})
    url_col:Optional[str] = field(default='url',metadata={"help": "body column"})
    source_col:Optional[str] = field(default='data_source',metadata={"help": "body column"})
    tc_max_len:Optional[int] = field(default=512,metadata={"help": "tc ids max len"})
    t_max_len:Optional[int] = field(default=30,metadata={"help": "title ids max len"})
    qt_max_len:Optional[int] = field(default=64,metadata={"help": "title ids max len"})

    # add by sunliuying 新增特征
    elapsedmonth_col: Optional[str] = field(default='elapsedDays',metadata={"help": "elapsedmonth time featue"})

    label_col:Optional[str] = field(default='label',metadata={"help": "body column"})
    logit_col:Optional[str] = field(default='logits',metadata={"help": "body column"})
    elie_label_col:Optional[str] = field(default='elie_label',metadata={"help": "elie label column"})
    elie_prob_col:Optional[str] = field(default='elie_probs',metadata={"help": "elie distil probs column"})
    elie_logit_col:Optional[str] = field(default=None,metadata={"help": "elie distil logits column"})
    # add by sunliuying
    qtc_pair_score_col: Optional[str] = field(default='pair_score', metadata={"help": "qtc pair distill score column"})
    auth_score_col: Optional[str] = field(default='auth_logits', metadata={"help": "auth distill score column"})
    
    train_file_v2: Optional[bool] = field(default=False, metadata={"help": "train file is v2"})

    #data_type:Optional[str] = field(default='oss',metadata={"help": "oss, local, odps"})
    
    shuffle_buffer_size:Optional[int] = field(default=256,metadata={"help": "shffle_buffer_size"})


    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either training or validation file.")
        else:
            pass
