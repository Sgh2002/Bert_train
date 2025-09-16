from transformers.modeling_utils import PreTrainedModel
from model.configuration_l4qtc_only import L4qtcOnlyConfig
from transformers import BertModel,BertForMaskedLM,AutoConfig,AutoModel
from transformers import BertConfig, BertTokenizer, BertForPreTraining, BertModel
from torch import nn
import torch
from typing import Optional
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.utils import ModelOutput
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class L4QtcOnlyElieModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    merge_sim: Optional[torch.FloatTensor] = None
    ts_merge_sim:Optional[torch.FloatTensor] = None
    elie_sim: Optional[torch.FloatTensor] = None
    auth_sim: Optional[torch.FloatTensor] = None


class MLP(nn.Module):

    def __init__(self, config: L4qtcOnlyConfig, inputdim: int):
        super().__init__()
        self.hidden_layer1 =  nn.Linear(inputdim, config.ml_intermediate_size_L1, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.hidden_layer2 =  nn.Linear(config.ml_intermediate_size_L1, config.mlp_intermediate_size_L2, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        x = self.hidden_layer1(input)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.hidden_layer2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return x

class CrossLayer(nn.Module):

    def __init__(self, config: L4qtcOnlyConfig, input_dim, num_classes=5):
        super().__init__()
        self.mlp = MLP(config, input_dim)
        self.logit_layer = nn.Linear(config.mlp_intermediate_size_L2, num_classes, bias=True)    
    
    def forward(self, qtc_embedding : torch.Tensor):
        mlp_input = qtc_embedding
        mlp_output =  self.mlp(mlp_input)
        logit = self.logit_layer(mlp_output)
        return logit


class L4QtcOnlyElieModel(PreTrainedModel):

    def __init__(self, config: L4qtcOnlyConfig, qtc_checkpoint=None, bert_config=None, is_export=False) -> None:
        super().__init__(config)

        self.is_export = is_export
        self.cross_layer = CrossLayer(config, config.hidden_size, config.num_classes)
        self.pair_corss_layer = CrossLayer(config, config.hidden_size, 1)
        self.elie_cross_layer = CrossLayer(config, config.hidden_size, 2)
        self.auth_cross_layer = CrossLayer(config, config.hidden_size, 1)

        self.num_classes = config.num_classes
        self.apply(self._init_weights) 

        if qtc_checkpoint:
            self.qtc_bert = BertModel.from_pretrained(qtc_checkpoint)
        else:
            config2 = AutoConfig.from_pretrained(bert_config) 
            self.qtc_bert = BertModel(config2)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            initializer_range = 0.02
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   

    def elie_distil_loss(self, predict_logit : torch.Tensor, label_logit : torch.Tensor, is_logit=False) -> torch.Tensor:
        def distil_cross_entropy(predict_logit, label_logit, temperature=1.0):

            predict_prob = torch.softmax(predict_logit/temperature,dim=-1)

            if is_logit:
                label_prob = torch.softmax(label_logit/temperature, dim=-1)
            else:
                label_prob = label_logit

            return -torch.sum(label_prob* torch.log(predict_prob),dim=-1,keepdim=True)
    
        loss = distil_cross_entropy(predict_logit, label_logit)
        singal_loss = loss
        loss = torch.mean(loss)
        return loss, singal_loss
    
    def distil_loss(self, predict_logit : torch.Tensor, label_logit : torch.Tensor, is_logit=True) -> torch.Tensor:
        def distil_cross_entropy(predict_logit, label_logit, temperature=1.0):

            predict_prob = torch.softmax(predict_logit/temperature,dim=-1)

            if is_logit:
                label_prob = torch.softmax(label_logit/temperature, dim=-1)
            else:
                label_prob = label_logit

            return -torch.sum(label_prob* torch.log(predict_prob),dim=-1,keepdim=True)
        
        T = 2.0
        S = 0.95

        loss = distil_cross_entropy(predict_logit, label_logit)
        loss_with_T = distil_cross_entropy(predict_logit, label_logit,temperature=T)

        toal_loss = loss*(1-S) + loss_with_T*S
        singal_loss = toal_loss
        toal_loss = torch.mean(toal_loss)
        return toal_loss, singal_loss
    
    def qtc_score(self, logits: torch.Tensor) -> torch.Tensor:

        x = torch.softmax(logits,dim=-1)
        y = torch.FloatTensor(list(range(0,self.num_classes))).to(x.device)

        score = torch.sum(x*y,dim=-1)
        return score

    def calc_prob(self, sim):
        if sim.shape[-1] == 1:
            return sim
        elif sim.shape[-1] == 2:
            return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1.], dtype=sim.dtype, device=sim.device), dim=-1)
        elif sim.shape[-1] == 3:
            return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1., 2.]), dim=-1)
        elif sim.shape[-1] == 6:
            return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 0., 1., 2., 3., 4.], dtype=sim.dtype, device=sim.device), dim=-1)
        else:
            return torch.sum(F.softmax(sim, dim=-1) * torch.tensor([0., 1., 2., 3., 4.], dtype=sim.dtype, device=sim.device), dim=-1)
        
        return result

    def calc_prob_elie(self, sim):
        softmax_sim = F.softmax(sim, dim=-1)
        weights = torch.tensor([0., 1.], dtype=sim.dtype, device=sim.device)
        result = torch.sum(softmax_sim * weights, dim=-1)
        
        return result

    def forward(
            self, 
            qtc_id: Optional[torch.Tensor] = None,
            qtc_attention_mask: Optional[torch.Tensor] = None,
            qtc_segment_id: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            distil_logits: Optional[torch.Tensor] = None,
            elie_labels: Optional[torch.LongTensor] = None,
            elie_distil_probs: Optional[torch.Tensor] = None,
            pair_distill_score: Optional[torch.Tensor] = None,
            auth_distill_score: Optional[torch.Tensor] = None
     ) -> L4QtcOnlyElieModelOutput:

        qtc_bert_out = self.qtc_bert(qtc_id, qtc_attention_mask, qtc_segment_id)
        # output heads
        merge_sim = self.cross_layer(qtc_bert_out.pooler_output)
        ts_merge_sim = self.pair_corss_layer(qtc_bert_out.pooler_output)
        elie_sim = self.elie_cross_layer(qtc_bert_out.pooler_output)
        auth_sim = self.auth_cross_layer(qtc_bert_out.pooler_output)

        # Export 是最终部署的模型的预估结果
        if self.is_export:
            fn = lambda sim: torch.reshape(self.calc_prob(sim), [-1, 1])
            fn_elie = lambda sim: torch.reshape(self.calc_prob_elie(sim), [-1, 1])
            fn_argmax = lambda sim: torch.reshape(torch.argmax(sim, dim=-1).float(), [-1, 1])
            fn_prompt = lambda sim: torch.reshape(sim, [-1, 1])

            elie_prob = F.softmax(elie_sim, dim=-1)
            merge_sim_prob = F.softmax(merge_sim, dim=-1)

            # 计算merge_sim_prob_new
            merge_sim_prob_new = torch.cat([
                torch.reshape(merge_sim_prob[:, 0] * elie_prob[:, 0], [-1, 1]),
                torch.reshape(merge_sim_prob[:, 0] * elie_prob[:, 1], [-1, 1]),
                merge_sim_prob[:, 1:]
            ], dim=-1)

            # 拼接所有张量
            result = torch.cat([
                fn(merge_sim), fn(merge_sim), fn(merge_sim),
                fn(merge_sim), fn(merge_sim), fn(ts_merge_sim),
                fn_argmax(merge_sim_prob_new),
                fn_prompt(auth_sim), fn_prompt(auth_sim), merge_sim_prob_new,
                fn_elie(elie_sim), fn(merge_sim), fn(merge_sim),
                fn(merge_sim), fn(merge_sim)
            ], dim=-1)

            return result


        loss = None
        pair_loss = None
        elie_loss = None
        auth_loss = None

        singal_loss = None
        singal_elie_loss = None

        #################### point_loss
        if distil_logits is not None :
            loss, singal_loss = self.distil_loss(merge_sim, distil_logits)
        elif labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            # merge_sim needs not to be softmaxed
            singal_loss = loss_fct(merge_sim, labels)
            loss = singal_loss.mean()
        #################### pair_loss
        if pair_distill_score is not None:
            pair_distill_score = pair_distill_score.view(-1, 1)
            mse_loss = torch.nn.MSELoss()
            pair_loss = mse_loss(ts_merge_sim, pair_distill_score)
        #################### elie_loss
        # print('elie_distil_probs', elie_distil_probs)
        # print('elie_sim', elie_sim)
        if elie_distil_probs is not None:
            # print("elie_loss",11111)
            elie_loss, singal_elie_loss = self.elie_distil_loss(elie_sim, elie_distil_probs, is_logit=False)
        elif elie_labels is not None:
            # print("elie_loss",22222)
            # print('elie_sim.shape', elie_sim.shape)
            # print('elie_labels.shape', elie_labels.shape)
            # print(elie_sim, elie_sim.type())
            # print(elie_labels, elie_labels.type())
            loss_fct = CrossEntropyLoss(reduction='none')
            singal_elie_loss = loss_fct(elie_sim, elie_labels)
            elie_loss = singal_elie_loss.mean()

        # elie_loss = loss
        # singal_elie_loss = singal_loss
        # # #################### auth_loss
        if auth_distill_score is not None:
            auth_distill_score = auth_distill_score.view(-1, 1)
            mse_loss = nn.MSELoss()  # 默认会做均值（reduction='mean'） 
            auth_loss = mse_loss(auth_sim, auth_distill_score)  
        #################### head output
        qtc_score = self.qtc_score(merge_sim)
        total_loss = loss

        # if pair_loss is None:
        #     total_loss = loss
        # else:
        #     total_loss = 0.65 * loss + 0.35 * pair_loss
        # # print('total_loss', total_loss)

        # losses = [loss, pair_loss, elie_loss, auth_loss]
        # filtered_losses = [item for item in losses if item is not None]
        # total_loss = sum(filtered_losses) if filtered_losses else None

        # print(f'qtc_id:{qtc_id.tolist()}')
        # print(f'qtc_attention_mask:{qtc_attention_mask.tolist()}')
        # print(f'qtc_segment_id:{qtc_segment_id.tolist()}')
        # print(f'qtc_score:{qtc_score.tolist()}')
        # print()

        # [debug sunliuying] 输出要修改一下
        return L4QtcOnlyElieModelOutput(
            loss=total_loss,
            merge_sim=merge_sim,
            ts_merge_sim=ts_merge_sim,
            elie_sim=elie_sim,
            auth_sim=auth_sim            
        )
