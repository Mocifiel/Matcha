import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from typing import Union, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TextInput = str
PreTokenizedInput = List[str]


def token_to_phone_expand(text_emb, syl_phone_num, batch_phone_num):
    num_syls = syl_phone_num.shape[1]
    num_tokens = text_emb.shape[1]
    text_emb_pad = text_emb[:, :num_syls, :] if num_tokens > num_syls else F.pad(text_emb, (0, 0, 0,
                                                                                            num_syls - num_tokens))
    phone_emb_stack = []
    for uid in range(text_emb_pad.shape[0]):
        phone_emb = torch.repeat_interleave(text_emb_pad[uid, ...], syl_phone_num[uid, ...], dim=0)
        if batch_phone_num < phone_emb.shape[0]:
            phone_emb_stack.append(phone_emb[:phone_emb.shape[0], :])
        else:
            phone_emb_stack.append(F.pad(phone_emb, (0, 0, 0, batch_phone_num - phone_emb.shape[0])))

    phone_emb_stack = torch.stack(phone_emb_stack)

    return phone_emb_stack


class TextEmb(torch.nn.Module):
    """On-the-fly spectrogram extractor."""

    def __init__(
        self,
        model_name_or_path
    ):
        super().__init__()
        try:
            # You can use model in hugginface, just give name, such as roberta-base, bert-base-uncased,
            # bert-base-chinese, etc, or you can pass model path
            if model_name_or_path.strip():
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModel.from_pretrained(model_name_or_path)
        except Exception as msg:
            print("Model init error %s, please check if the model name or path %s is correct." %
                  (msg, model_name_or_path))

    def forward(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
                device='cpu'):

        # text = 'Hello, my dog is cute'
        inputs = self.tokenizer(text, return_tensors="pt", padding='longest').to(device)
        inputs.data['input_ids'] = inputs.data['input_ids'][:, :self.model.config.max_position_embeddings - 2]
        inputs.data['attention_mask'] = inputs.data['attention_mask'][:, :self.model.config.max_position_embeddings - 2]
        outputs = self.model(**inputs)

        # sentence embedding
        token_emb_masked = outputs.last_hidden_state * inputs.data['attention_mask'].unsqueeze(2)
        # skip cls/sep
        total_non_padded_tokens_per_batch = torch.sum(inputs.data['attention_mask'], dim=1).unsqueeze(-1) - 2
        sentence_mean_embedding = torch.sum(token_emb_masked[:, 1:-1, :], dim=1) / total_non_padded_tokens_per_batch

        return outputs.last_hidden_state, sentence_mean_embedding
