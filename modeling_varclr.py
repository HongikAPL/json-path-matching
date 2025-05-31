'''
VarCLR BERT 모델을 HuggingFace Transformers BERT 인터페이스로 래핑하는 클래스
하지만 VarCLR BERT는 HuggingFace Transformers BERT와는 다르게 모델 생성시 Config 객체를 사용하지 않음 -> Config 전달 불가능
고로 사용안함
'''


# import torch
# import torch.nn as nn
# from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
# from varclr.models.model import Encoder as VarclrEncoder

# class VarclrBertWrapper(nn.Module):
#     """
#     Wraps a VarCLR Encoder (BERT) to present a HuggingFace-like BERT interface.
#     """
#     def __init__(self, config, add_pooling_layer=False, *args, **kwargs):
#         super().__init__()
#         self.varclr_encoder = VarclrEncoder.from_pretrained("varclr-codebert")
#         self.add_pooling_layer = add_pooling_layer  # (옵션) 실제로 pooling이 필요하면 활용

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=True,
#         **kwargs,
#     ):
#         # VarCLR BERT only uses input_ids, attention_mask
#         pooled, (all_hids, attn_mask) = self.varclr_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         # all_hids: tuple of (layer, batch, seq, hidden)
#         # transformers: last_hidden_state = all_hids[-1] (batch, seq, hidden)
#         last_hidden_state = all_hids[-1]
#         pooler_output = pooled if self.add_pooling_layer else None  # [CLS] 임베딩

#         if not return_dict:
#             # transformers BERT: (last_hidden_state, pooler_output, all_hidden_states, attentions)
#             return (last_hidden_state, pooler_output, all_hids, None, None)
#         return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooler_output,
#             hidden_states=all_hids,
#             attentions=None,
#             cross_attentions=None,
#             past_key_values=None,
#         )
    

# class VarclrPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.transform_act_fn = getattr(torch.nn.functional, config.hidden_act, torch.nn.functional.gelu)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states


# class VarclrLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = VarclrPredictionHeadTransform(config)
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states