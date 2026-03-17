"""Inference pipeline with greedy decoding."""

import torch
from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.model.transformer import GeneticTransformer
from genetic_transformer.tokenizer.genetic_tokenizer import GeneticTokenizer
from genetic_transformer.tokenizer.vocab import SOS_IDX, EOS_IDX, PAD_IDX


class GeneticTranslator:
    """High-level inference interface for the trained Genetic Transformer."""

    def __init__(self, model: GeneticTransformer, tokenizer: GeneticTokenizer,
                 config: GeneticTransformerConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model.eval()

    def translate(self, input_text: str, max_len: int = 128) -> str:
        """Translate input text (auto-detects domain). Returns decoded output string."""
        output_ids, _ = self._greedy_decode(input_text, max_len)
        return self.tokenizer.decode(output_ids, skip_special=True)

    def translate_with_attention(self, input_text: str, max_len: int = 128) -> tuple[str, dict, list[str], list[str]]:
        """Translate and return attention maps + source/target tokens for visualization.

        Returns:
            output_text: decoded output string
            attention_maps: dict with encoder/decoder attention weight tensors
            src_tokens: list of source token strings
            tgt_tokens: list of generated target token strings
        """
        output_ids, attention_maps = self._greedy_decode(input_text, max_len)
        output_text = self.tokenizer.decode(output_ids, skip_special=True)

        # Get source tokens for labeling
        src_tokens = self.tokenizer.tokenize(input_text)
        src_tokens = ["<SOS>"] + src_tokens + ["<EOS>"]

        # Get target tokens
        tgt_tokens = ["<SOS>"]
        for idx in output_ids:
            if idx == EOS_IDX:
                tgt_tokens.append("<EOS>")
                break
            if idx == PAD_IDX:
                break
            token = self.tokenizer.vocab.decode(idx)
            tgt_tokens.append(token)

        return output_text, attention_maps, src_tokens, tgt_tokens

    @torch.no_grad()
    def _greedy_decode(self, input_text: str, max_len: int) -> tuple[list[int], dict]:
        """Autoregressive greedy decoding."""
        # Encode source
        src_indices = self.tokenizer.encode(input_text, max_len=self.config.max_seq_len)
        src = torch.tensor([src_indices], dtype=torch.long, device=self.device)
        src_mask = self.model.make_src_mask(src)

        # Encode source through encoder
        enc_out, enc_attn = self.model.encoder(src, src_mask)

        # Start with SOS token
        tgt_indices = [SOS_IDX]
        all_attention_maps = {"encoder_self": enc_attn, "decoder_self": [], "decoder_cross": []}

        # Cap max_len to avoid exceeding positional encoding buffer
        max_decode_len = min(max_len, self.config.max_seq_len - 1)

        for _ in range(max_decode_len):
            tgt = torch.tensor([tgt_indices], dtype=torch.long, device=self.device)
            tgt_mask = self.model.make_tgt_mask(tgt)

            # Decode
            logits, dec_attn = self.model.decoder(tgt, enc_out, src_mask, tgt_mask)

            # Get next token (greedy: argmax of last position)
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            tgt_indices.append(next_token)

            if next_token == EOS_IDX:
                break

        # Collect final attention maps
        all_attention_maps["decoder_self"] = dec_attn.get("self_attn", [])
        all_attention_maps["decoder_cross"] = dec_attn.get("cross_attn", [])

        # Return generated tokens (excluding SOS)
        return tgt_indices[1:], all_attention_maps

    def batch_translate(self, inputs: list[str]) -> list[str]:
        """Translate a batch of inputs sequentially."""
        return [self.translate(text) for text in inputs]
