import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead

# VarCLR 추가
import os
import gdown
from transformers import PreTrainedModel, AutoConfig
# from varclr.models import urls_pretrained_model
PRETRAINED_TOKENIZER = "microsoft/codebert-base"

PRETRAINED_CODEBERT_URL = (
    "https://drive.google.com/uc?id=1xl8kdQtJ7ke4jyv5kHDiOc5dScPTTKzg"
)
PRETRAINED_CODEBERT_FOLDER = "varclr_bert"
PRETRAINED_CODEBERT_MD5 = "3844bd6e76a928084b0d742ac120a91c"


from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y):
        sim = self.cos(x, y)
        self.record = sim.detach()
        min_size = min(self.record.shape[0], self.record.shape[1])
        num_item = self.record.shape[0] * self.record.shape[1]
        self.pos_avg = self.record.diag().sum() / min_size
        self.neg_avg = (self.record.sum() - self.record.diag().sum()) / (num_item - min_size)
        return sim / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config) if not cls.model_args.batchnorm else ProjectionMLP(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    cls.generator = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased') if cls.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(cls.model_args.generator_name)
    cls.electra_acc = 0.0
    cls.electra_rep_acc = 0.0
    cls.electra_fix_acc = 0.0

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    cls_token=101,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if not cls.model_args.before_mlp:
        if cls.pooler_type == "cls":
            pooler_output = pooler_output.view((batch_size*num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
            pooler_output = cls.mlp(pooler_output)
            pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # Produce MLM augmentations and perform conditional ELECTRA using the discriminator
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        with torch.no_grad():
            g_pred = cls.generator(mlm_input_ids, attention_mask)[0].argmax(-1)
        g_pred[:, 0] = cls_token
        replaced = (g_pred != input_ids) * attention_mask
        e_inputs = g_pred * attention_mask

        mlm_outputs = cls.discriminator(
            e_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            cls_input=pooler_output.view((-1, pooler_output.size(-1))),
        )

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.before_mlp:
        if cls.pooler_type == "cls":
            pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for conditional ELECTRA
    if mlm_outputs is not None and mlm_labels is not None:
        # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        e_labels = replaced.view(-1, replaced.size(-1))
        # prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        prediction_scores = cls.electra_head(mlm_outputs.last_hidden_state)
        rep = (e_labels == 1) * attention_mask
        fix = (e_labels == 0) * attention_mask
        prediction = prediction_scores.argmax(-1)
        cls.electra_rep_acc = float((prediction*rep).sum()/rep.sum())
        cls.electra_fix_acc = float(1.0 - (prediction*fix).sum()/fix.sum())
        cls.electra_acc = float(((prediction == e_labels) * attention_mask).sum()/attention_mask.sum())
        # masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        masked_lm_loss = loss_fct(prediction_scores.view(-1, 2), e_labels.view(-1))
        loss = loss + cls.model_args.lambda_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        self.lm_head = BertLMPredictionHead(config)
        self.discriminator = BertModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=101,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.lm_head = RobertaLMHead(config)
        self.discriminator = RobertaModel(config, add_pooling_layer=False)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=0,
            )

# VarCLR 가중치 로드
class VarclrForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.discriminator = RobertaModel(config, add_pooling_layer=False)
        cl_init(self, config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        BertPreTrainedModel.from_pretrained 오버라이드
        Google Drive에서 사전학습된 CodeBERT 모델을 다운로드하고 압축 해제한 뒤,
        config를 불러와서 VarclrForCL 객체를 생성.
        """
        save_path = kwargs.get("cache_dir", "./")
        varclr_model_dir = os.path.join(save_path, "varclr_bert")  # 압축 해제 폴더
        varclr_model_path = os.path.join(varclr_model_dir, "pytorch_model.bin")  # 실제 가중치 파일

        # 가중치 다운로드 및 압축 해제
        gdown.cached_download(
            PRETRAINED_CODEBERT_URL,
            os.path.join(save_path, "varclr.zip"),
            md5=PRETRAINED_CODEBERT_MD5,
            postprocess=gdown.extractall,
        )

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 인스턴스 생성, varclr_bert 안의 config 대신 전달 받은 config 사용
        instance = cls(config=config, *model_args, **kwargs)

        # 가중치 로드
        if os.path.exists(varclr_model_path):
            state_dict = torch.load(varclr_model_path, map_location="cpu")
            instance.bert.load_state_dict(state_dict, strict=False)
            instance.discriminator.load_state_dict(state_dict, strict=False)
        else:
            print(f"[VarCLR] Warning: {varclr_model_path} not found. Using randomly initialized weights.")

        return instance
    
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.bert.embeddings.word_embeddings = new_embeddings

    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=101,
            )

# # VarCLR 를 huggingface 모델 래핑해보려 했으나 varclr.encoders 내부에서 config를 사용하지 않고있어 불가능
'''
class VarclrForCL(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.varclr = VarclrBertWrapper(config, add_pooling_layer=False)

        self.lm_head = VarclrLMPredictionHead(config)
        self.discriminator = VarclrBertWrapper(config, add_pooling_layer=False)

        cl_init(self, config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Google Drive에서 사전학습된 CodeBERT 모델을 다운로드하고 압축 해제한 뒤,
        config를 불러와서 VarclrForCL 객체를 생성합니다.
        """
        save_path = kwargs.get("cache_dir", "./")
        varclr_model_path = os.path.join(save_path, "varclr_bert")  # PRETRAINED_CODEBERT_FOLDER

        # 모델 다운로드 및 압축 해제
        gdown.cached_download(
            urls_pretrained_model.PRETRAINED_CODEBERT_URL,  # PRETRAINED_CODEBERT_URL
            os.path.join(save_path, "bert.zip"),
            md5=urls_pretrained_model.PRETRAINED_CODEBERT_MD5,  # PRETRAINED_CODEBERT_MD5
            postprocess=gdown.extractall,
        )

        # HuggingFace와 호환되는 config 로드
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # VarclrForCL 객체 초기화
        instance = cls(config=config, *model_args, **kwargs)

        # 실제로 가중치 로드 (VarclrBertWrapper 내부의 varclr_encoder에)
        # bert_model_path는 압축 해제된 모델 디렉토리 경로여야 함
        instance.varclr.varclr_encoder = instance.varclr.varclr_encoder.load(varclr_model_path)
        instance.discriminator.varclr_encoder = instance.discriminator.varclr_encoder.load(varclr_model_path)

        return instance

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.varclr,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.varclr,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=101,
            )
'''