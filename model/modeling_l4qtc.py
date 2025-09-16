from transformers.modeling_utils import PreTrainedModel
from model.configuration_l4qtc import L4qtcConfig
from transformers import BertConfig, BertTokenizer,BertForPreTraining,BertModel
from torch import nn
import torch
from typing import Optional
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import ModelOutput
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class L4qtcElieModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    qtc_score:Optional[torch.FloatTensor] = None
    ac_loss: Optional[torch.FloatTensor] = None
    elie_loss: Optional[torch.FloatTensor] = None
    elie_logits:Optional[torch.FloatTensor] = None

class MLP(nn.Module):

    def __init__(self, config: L4qtcConfig, inputdim: int):
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

    def __init__(self, config: L4qtcConfig, input_dim, num_classes=5):
        super().__init__()

        self.mlp = MLP(config, input_dim)

        self.logit_layer = nn.Linear(config.mlp_intermediate_size_L2, num_classes, bias=True)
    
    
    def forward(self, q_embedding : torch.Tensor,qt_embedding : torch.Tensor, qtc_embedding : torch.Tensor):

        mlp_input = torch.cat([q_embedding,qt_embedding,qtc_embedding],dim=-1)
        mlp_output =  self.mlp(mlp_input)
        logit = self.logit_layer(mlp_output)
        return logit

class L4QtcElieModel(PreTrainedModel):

    def __init__(self, config: L4qtcConfig, qt_checkpoint,qtc_checkpoint) -> None:
        super().__init__(config)

        self.cross_layer = CrossLayer(config, config.hidden_size*3,config.num_classes)
        self.elie_cross_layer = CrossLayer(config, config.hidden_size*3,2)


        self.num_classes = config.num_classes     
        self.apply(self._init_weights) 

        self.q_bert = BertModel.from_pretrained(qt_checkpoint)
        self.qt_bert = BertModel.from_pretrained(qt_checkpoint)
        self.qtc_bert = BertModel.from_pretrained(qtc_checkpoint)
    
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


    def forward(
            self, 
            query_input_ids: Optional[torch.Tensor] = None,
            query_attention_mask: Optional[torch.Tensor] = None,
            query_token_type_ids: Optional[torch.Tensor] = None,
            qt_input_ids: Optional[torch.Tensor] = None,
            qt_attention_mask: Optional[torch.Tensor] = None,
            qt_token_type_ids: Optional[torch.Tensor] = None,
            qtc_input_ids: Optional[torch.Tensor] = None,
            qtc_attention_mask: Optional[torch.Tensor] = None,
            qtc_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            distil_logits: Optional[torch.Tensor] = None,
            elie_distil_probs: Optional[torch.Tensor] = None,
            elie_labels: Optional[torch.LongTensor] = None
     ) -> L4qtcElieModelOutput:

        #print(f'query_input_ids:{query_input_ids}')
    
        

        q_bert_out = self.q_bert(query_input_ids,query_attention_mask,query_token_type_ids)
        qt_bert_out = self.qt_bert(qt_input_ids,qt_attention_mask,qt_token_type_ids)
        qtc_bert_out = self.qtc_bert(qtc_input_ids,qtc_attention_mask,qtc_token_type_ids)

        logits = self.cross_layer(q_bert_out.pooler_output,qt_bert_out.pooler_output,qtc_bert_out.pooler_output)
        
       

        elie_logits =  self.elie_cross_layer(q_bert_out.pooler_output,qt_bert_out.pooler_output,qtc_bert_out.pooler_output)

        loss = None
        elie_loss = None
        singal_loss = None
        singal_elie_loss = None

        if distil_logits is not None :
            loss, singal_loss = self.distil_loss(logits, distil_logits)
        elif labels is not None:
            assert(elie_labels is not None)
            loss_fct = CrossEntropyLoss(reduction='none')
            singal_loss = loss_fct(logits,labels)
            loss = singal_loss.mean()

        if elie_distil_probs is not None:
            elie_loss, singal_elie_loss =  self.elie_distil_loss(elie_logits, elie_distil_probs, is_logit=False)
        elif elie_labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            singal_elie_loss = loss_fct(elie_logits,elie_labels)
            elie_loss = singal_elie_loss.mean()

        
        qtc_score = self.qtc_score(logits)

        if loss is not None and elie_loss is not None:
            total_loss = elie_loss
        elif loss is not None:
            total_loss = loss
        else:
            total_loss=None

        return L4qtcElieModelOutput(
            loss=loss,
            logits=logits,
            qtc_score=qtc_score,
            ac_loss=singal_loss,
            elie_loss=singal_elie_loss,
            elie_logits=elie_logits
        )






    





