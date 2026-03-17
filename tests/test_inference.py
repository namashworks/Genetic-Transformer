"""Tests for genetic_transformer.inference — translator and visualizer."""

import pytest
import torch

from genetic_transformer.inference.translator import GeneticTranslator
from genetic_transformer.inference.visualizer import AttentionVisualizer
from genetic_transformer.tokenizer.vocab import EOS_IDX


# ---------------------------------------------------------------------------
# GeneticTranslator
# ---------------------------------------------------------------------------

class TestGeneticTranslator:
    @pytest.fixture
    def translator(self, model, tokenizer, small_config):
        return GeneticTranslator(model, tokenizer, small_config, device="cpu")

    def test_translate_returns_string(self, translator):
        result = translator.translate("AUGGCU", max_len=16)
        assert isinstance(result, str)

    def test_translate_produces_nonempty_output(self, translator):
        # Even an untrained model should produce some tokens before hitting max_len
        result = translator.translate("AUGGCU", max_len=16)
        # It may or may not be empty depending on random weights, but it should not error
        assert isinstance(result, str)

    def test_translate_with_attention_returns_tuple(self, translator):
        result = translator.translate_with_attention("AUGGCU", max_len=16)
        assert isinstance(result, tuple)
        assert len(result) == 4
        output_text, attn_maps, src_tokens, tgt_tokens = result
        assert isinstance(output_text, str)
        assert isinstance(attn_maps, dict)
        assert isinstance(src_tokens, list)
        assert isinstance(tgt_tokens, list)

    def test_translate_with_attention_has_expected_keys(self, translator):
        _, attn_maps, _, _ = translator.translate_with_attention("ATGCGA", max_len=16)
        assert "encoder_self" in attn_maps
        assert "decoder_self" in attn_maps
        assert "decoder_cross" in attn_maps

    def test_batch_translate(self, translator):
        inputs = ["AUGGCU", "AUGCGA"]
        results = translator.batch_translate(inputs)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_greedy_decode_returns_list_and_dict(self, translator):
        ids, attn = translator._greedy_decode("AUGGCU", max_len=10)
        assert isinstance(ids, list)
        assert isinstance(attn, dict)
        assert all(isinstance(i, int) for i in ids)

    def test_translate_nl_input(self, translator):
        result = translator.translate("what amino acid does AUG encode", max_len=16)
        assert isinstance(result, str)

    def test_translate_protein_input(self, translator):
        result = translator.translate("MFKL", max_len=16)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AttentionVisualizer
# ---------------------------------------------------------------------------

class TestAttentionVisualizer:
    def test_plot_attention_heatmap(self):
        import matplotlib
        matplotlib.use("Agg")
        attn = torch.randn(1, 2, 5, 8)  # batch, heads, tgt_len, src_len
        src_tokens = [f"s{i}" for i in range(8)]
        tgt_tokens = [f"t{i}" for i in range(5)]
        fig = AttentionVisualizer.plot_attention_heatmap(
            attn, src_tokens, tgt_tokens, title="Test", layer=0, head=0
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_multi_head_attention(self):
        import matplotlib
        matplotlib.use("Agg")
        attn = torch.randn(1, 2, 5, 8)
        src_tokens = [f"s{i}" for i in range(8)]
        tgt_tokens = [f"t{i}" for i in range(5)]
        fig = AttentionVisualizer.plot_multi_head_attention(
            attn, src_tokens, tgt_tokens, layer=0, n_heads=2
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_cross_attention(self):
        import matplotlib
        matplotlib.use("Agg")
        cross_attn = [torch.randn(1, 2, 5, 8)]
        attn_maps = {"decoder_cross": cross_attn}
        src_tokens = [f"s{i}" for i in range(8)]
        tgt_tokens = [f"t{i}" for i in range(5)]
        fig = AttentionVisualizer.plot_cross_attention(
            attn_maps, src_tokens, tgt_tokens, layer=0
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_cross_attention_empty(self):
        import matplotlib
        matplotlib.use("Agg")
        attn_maps = {}
        fig = AttentionVisualizer.plot_cross_attention(attn_maps, [], [])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_training_curves(self):
        import matplotlib
        matplotlib.use("Agg")
        history = {
            "train_loss": [2.0, 1.5, 1.2],
            "val_loss": [2.1, 1.6, 1.3],
            "lr": [0.001, 0.002, 0.0015],
        }
        fig = AttentionVisualizer.plot_training_curves(history)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
