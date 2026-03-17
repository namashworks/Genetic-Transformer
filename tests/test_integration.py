"""End-to-end integration tests for the Genetic Transformer pipeline."""

import pytest
import torch

from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.tokenizer.genetic_tokenizer import GeneticTokenizer
from genetic_transformer.model.transformer import GeneticTransformer
from genetic_transformer.data.synthetic_generator import SyntheticDataGenerator
from genetic_transformer.data.dataset import GeneticDataset, create_dataloaders
from genetic_transformer.training.trainer import Trainer
from genetic_transformer.training.losses import LabelSmoothingCrossEntropy
from genetic_transformer.training.scheduler import WarmupScheduler
from genetic_transformer.inference.translator import GeneticTranslator
from genetic_transformer.tokenizer.vocab import PAD_IDX


class TestEndToEnd:
    """Full pipeline: generate data -> build model -> train 1 epoch -> translate."""

    @pytest.fixture
    def pipeline(self):
        """Set up a complete tiny pipeline."""
        tokenizer = GeneticTokenizer()
        vocab_size = tokenizer.get_vocab_size()

        config = GeneticTransformerConfig(
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            max_seq_len=32,
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-4,
            warmup_steps=5,
            label_smoothing=0.1,
            grad_clip=1.0,
            num_train_samples=16,
            num_val_samples=8,
            seed=42,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
        )

        model = GeneticTransformer(config)
        train_loader, val_loader = create_dataloaders(config, tokenizer)
        trainer = Trainer(model, config, train_loader, val_loader, device="cpu")

        return {
            "tokenizer": tokenizer,
            "config": config,
            "model": model,
            "trainer": trainer,
            "train_loader": train_loader,
            "val_loader": val_loader,
        }

    def test_full_pipeline_runs(self, pipeline):
        """The entire pipeline should execute without errors."""
        trainer = pipeline["trainer"]

        # Train one epoch
        train_loss = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert train_loss > 0

        # Validate
        val_loss = trainer.validate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_translate_after_training(self, pipeline):
        """After training, the translator should produce output."""
        trainer = pipeline["trainer"]
        trainer.train_epoch()

        translator = GeneticTranslator(
            pipeline["model"],
            pipeline["tokenizer"],
            pipeline["config"],
            device="cpu",
        )

        # Translate a simple RNA sequence
        result = translator.translate("AUGGCU", max_len=16)
        assert isinstance(result, str)

    def test_translate_with_attention_after_training(self, pipeline):
        """Attention maps should be available after translation."""
        trainer = pipeline["trainer"]
        trainer.train_epoch()

        translator = GeneticTranslator(
            pipeline["model"],
            pipeline["tokenizer"],
            pipeline["config"],
            device="cpu",
        )

        output_text, attn_maps, src_tokens, tgt_tokens = (
            translator.translate_with_attention("AUGGCU", max_len=16)
        )
        assert isinstance(output_text, str)
        assert "encoder_self" in attn_maps
        assert len(src_tokens) > 0
        assert len(tgt_tokens) > 0

    def test_data_generation_feeds_model(self, pipeline):
        """Data from the generator can be consumed by the model forward pass."""
        model = pipeline["model"]
        train_loader = pipeline["train_loader"]
        model.eval()

        batch = next(iter(train_loader))
        with torch.no_grad():
            logits, attn_maps = model(batch["src"], batch["tgt"])

        assert logits.shape[0] == batch["src"].shape[0]
        assert logits.shape[2] == pipeline["config"].src_vocab_size

    def test_loss_computes_on_model_output(self, pipeline):
        """Loss function should work on model output tensors."""
        model = pipeline["model"]
        config = pipeline["config"]
        train_loader = pipeline["train_loader"]

        criterion = LabelSmoothingCrossEntropy(
            config.tgt_vocab_size, PAD_IDX, config.label_smoothing
        )

        model.eval()
        batch = next(iter(train_loader))
        with torch.no_grad():
            logits, _ = model(batch["src"], batch["tgt"])
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch["tgt_y"].view(-1),
            )
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_checkpoint_save_load_roundtrip(self, pipeline, tmp_path):
        """Model can be saved and loaded, producing the same output."""
        trainer = pipeline["trainer"]
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        config = pipeline["config"]

        trainer.train_epoch()

        ckpt_path = str(tmp_path / "integration_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)

        # Create a fresh model and load the checkpoint
        fresh_model = GeneticTransformer(config)
        fresh_trainer = Trainer(
            fresh_model, config,
            pipeline["train_loader"], pipeline["val_loader"],
            device="cpu",
        )
        fresh_trainer.load_checkpoint(ckpt_path)

        # Both models should produce the same output
        model.eval()
        fresh_model.eval()
        test_src = torch.randint(1, config.src_vocab_size, (1, 8))
        test_tgt = torch.randint(1, config.tgt_vocab_size, (1, 6))
        with torch.no_grad():
            out1, _ = model(test_src, test_tgt)
            out2, _ = fresh_model(test_src, test_tgt)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_multiple_input_domains(self, pipeline):
        """Translator handles inputs from different domains without error."""
        trainer = pipeline["trainer"]
        trainer.train_epoch()

        translator = GeneticTranslator(
            pipeline["model"],
            pipeline["tokenizer"],
            pipeline["config"],
            device="cpu",
        )

        inputs = [
            "ATGCGA",                              # DNA
            "AUGGCUUAA",                           # RNA
            "MFKL",                                # protein
            "what amino acid does AUG encode",     # NL
        ]
        for text in inputs:
            result = translator.translate(text, max_len=16)
            assert isinstance(result, str)
