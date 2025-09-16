from dataclasses import dataclass, field
from typing import Optional
from transformers.utils.versions import require_version
import sys
import os

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    model_type: Optional[str] = field(
        default='q_t_tc_elie_v8',
        metadata={
            "help": (
                "model type : q_tc, q_t_tc,q_t_tc_v2,q_t_tc_elie, l4_qtc_elie"
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=True,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )

    query_encoder_export_path:  Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "save path of query bert branch"
            )
        },
    )

    doc_encoder_export_path:  Optional[str] = field(
        #default='./cache/export_model/doc_encoder.pt',
        default=None,
        metadata={
            "help": (
                "save path of title bert branch"
            )
        },
    )

    cross_export_path:  Optional[str] = field(
        #default='./cache/export_model/cross_layer.pt',
        default=None,
        metadata={
            "help": (
                "save path of title bert branch"
            )
        },
    )


    model_export_path:  Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "save path of the model"
            )
        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    bert_config: Optional[str] = field(
        default=None, metadata={"help": "bert config name or path if not the same as model_name"}
    )
    bert_vocab: Optional[str] = field(
        default=None, metadata={"help": "bert config name or path if not the same as model_name"}
    )

    qt_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )
    t_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    qtc_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    qtc_rerank_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    qtc_qwen_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    qwen_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained qwen"}
    )

    quark_init_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained qwen"}
    )

    q_perceiver_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    t_perceiver_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    tc_perceiver_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "pretrained bert with query and title"}
    )

    doc_compress: Optional[bool] = field(
        default=False, metadata={"help": "pretrained bert with query and title"}
    )

    cache_dir: Optional[str] = field(
        default='./cache',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
       pass