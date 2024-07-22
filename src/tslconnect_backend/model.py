import torch.nn as nn
from transformers import T5ForConditionalGeneration


class Sign2ThaiT5(nn.Module):
    def __init__(self, t5_model="t5-small", vocab_size=119, input_dim=483):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.t5.resize_token_embeddings(vocab_size)

        self.projection = nn.Linear(input_dim, self.t5.config.d_model, bias=False)

    def forward(
        self, landmarks, attention_mask=None, decoder_input_ids=None, labels=None
    ):
        # Input shape: (batch_size, frames_len, landmarks_xyz)
        # Project landmarks to T5's d_model dimension
        projected_landmarks = self.projection(landmarks)

        # Pass the projected landmarks to T5
        outputs = self.t5(
            inputs_embeds=projected_landmarks,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def generate(self, landmarks, attention_mask, **generate_kwargs):
        projected_landmarks = self.projection(landmarks)

        return self.t5.generate(
            inputs_embeds=projected_landmarks,
            attention_mask=attention_mask,
            **generate_kwargs,
        )


def Thai2SignT5(vocab_size=119, sign_vocab_size=44):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.resize_token_embeddings(vocab_size)
    model.decoder.embed_tokens = nn.Embedding(sign_vocab_size, model.config.d_model)
    model.lm_head = nn.Linear(model.config.d_model, sign_vocab_size, bias=False)
    return model
