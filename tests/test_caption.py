import json

import keras
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tounum_caption import config, data, evaluation, features, model, pipeline
from tounum_cls import modeling as cls_modeling
from tounum_denoise import modeling as dn_modeling

CAPTIONS = [f"A {c} dog runs on the {p}." for c in ("brown", "black", "white") for p in ("grass", "beach")]


def _coco(tmp_path):
    """Tiny COCO-like folder: 6 train2017 images and 3 val2017 images, 2 captions each."""
    for split, ids in (("train2017", range(1, 7)), ("val2017", range(101, 104))):
        anns = {"images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in ids],
                "annotations": [{"image_id": i, "caption": CAPTIONS[(i + k) % 6]} for i in ids for k in (0, 1)]}
        (tmp_path / "annotations").mkdir(exist_ok=True)
        (tmp_path / "annotations" / f"captions_{split}.json").write_text(json.dumps(anns))
    return tmp_path


def _toy_model_data(n_img=12, seed=0):
    rng = np.random.default_rng(seed)
    vocab = data.build_vocabulary([data.clean_caption(c) for c in CAPTIONS])
    feats = rng.standard_normal((n_img, config.N_REGIONS, config.FEATURE_DIM)).astype(np.float16)
    caps = [data.clean_caption(CAPTIONS[i % 6]) for i in range(n_img) for _ in range(2)]
    return vocab, feats, data.encode(caps, vocab), np.repeat(np.arange(n_img), 2)


def test_clean_caption_and_vocabulary():
    assert data.clean_caption("A man's surf-board, 2 waves!") == "startseq a man s surf board waves endseq"
    vocab = data.build_vocabulary(["startseq a dog endseq", "startseq a cat endseq"], vocab_size=5)
    assert vocab[:2] == ["", "[UNK]"] and len(vocab) == 5 and vocab[2] == "a"


def test_encode_decode_roundtrip_and_truncation():
    vocab = data.build_vocabulary([data.clean_caption(c) for c in CAPTIONS])
    ids = data.encode([data.clean_caption("a brown dog runs on the zebra")], vocab, max_len=6)
    assert ids.shape == (1, 6) and ids[0, -1] == vocab.index(config.END) and 1 not in ids[0, :5]
    full = data.encode([data.clean_caption("a brown dog runs on the zebra")], vocab)
    assert data.decode(full[0], vocab) == "a brown dog runs on the [UNK]"


def test_sampling_exclusion_captions_and_references(tmp_path):
    coco = _coco(tmp_path)
    images = data.sample_images(n_train=4, n_val=2, n_test=3, coco_dir=coco)
    assert list(images["split"].value_counts().sort_index()) == [3, 4, 2]       # test, train, val
    assert not set(images.query("split == 'train'").image_id) & set(images.query("split == 'val'").image_id)
    assert set(images.query("split == 'test'").coco_split) == {"val2017"}
    dropped = data.exclude_images(images, [images.image_id.iloc[0]])
    assert len(dropped) == len(images) - 1
    caps = data.captions_for(dropped, coco_dir=coco)
    assert len(caps) == 2 * len(dropped) and caps["clean"].str.startswith(config.START).all()
    refs = data.references(caps, "test")
    assert len(refs) == 3 and all(len(r) == 2 and not r[0].startswith(config.START) for r in refs)


def test_encoder_output_shape():
    enc = features.build_encoder(weights=None)
    img = features.prepare(np.zeros((50, 80, 3), np.uint8))
    assert img.shape == (299, 299, 3)
    out = features.encode(enc, img[None])
    assert out.shape == (1, config.N_REGIONS, config.FEATURE_DIM) and out.dtype == np.float16


def test_extract_features_cache(tmp_path):
    for i in range(3):
        Image.fromarray(np.full((40, 60, 3), 40 * i, np.uint8)).save(tmp_path / f"{i}.jpg")
    enc = features.build_encoder(weights=None)
    paths = sorted(tmp_path.glob("*.jpg"))
    f1 = features.extract_features(enc, paths, tmp_path / "f.npy", chunk=2)
    assert f1.shape == (3, config.N_REGIONS, config.FEATURE_DIM) and (tmp_path / "f.npy").exists()
    assert not (tmp_path / "f.part.npy").exists()
    assert np.array_equal(features.extract_features(None, paths, tmp_path / "f.npy"), f1)   # read from cache


def test_masked_accuracy_ignores_padding():
    y_true = np.array([[3, 4, 0, 0]])
    logits = np.eye(6)[[[3, 5, 1, 1]]].astype("float32")    # right, wrong, then padding
    assert float(model.masked_accuracy(y_true, logits)) == pytest.approx(0.5)


def test_caption_batches_one_caption_per_image():
    vocab, feats, tokens, index = _toy_model_data()
    batches = model.CaptionBatches(feats, tokens, index, batch_size=5, one_per_image=True)
    assert len(batches) == 3 and sorted(index[batches.rows]) == list(range(12))
    x, y = batches[0]
    assert x["image_features"].shape == (5, 64, 2048) and x["tokens"].shape == (5, config.MAX_LEN - 1)
    assert np.array_equal(x["tokens"][:, 1:], y[:, :-1])                    # targets = inputs shifted by one
    assert len(model.CaptionBatches(feats, tokens, index, batch_size=5)) == 5


@pytest.mark.parametrize("attention", [True, False])
def test_captioner_learns_generates_and_reloads(attention, tmp_path):
    keras.utils.set_random_seed(0)
    vocab, feats, tokens, index = _toy_model_data()
    m = model.build_captioner(vocab_size=len(vocab), attention=attention, embed_dim=32, units=32)
    assert ("attention" in [l.name for l in m.layers]) == attention
    batches = model.CaptionBatches(feats, tokens, index, batch_size=8)
    hist = model.train(m, batches, batches, epochs=25, patience=25, verbose=0,
                       callbacks=[model.BleuCallback(feats[:4], [["a dog"]] * 4, vocab)])
    assert {"epoch", "loss", "masked_accuracy", "val_loss", "val_bleu4"} <= set(hist.columns)
    assert hist["val_loss"].iloc[-1] < hist["val_loss"].iloc[0]
    captions = model.generate(m, feats[:3], vocab)
    assert len(captions) == 3 and all(isinstance(c, str) for c in captions)
    m.save(tmp_path / "c.keras")
    m2 = keras.models.load_model(tmp_path / "c.keras")
    assert model.generate(m2, feats[:3], vocab) == captions
    if attention:
        att = model.attention_maps(model.attention_model(m2), feats[:3], captions, vocab)
        assert att.shape == (3, config.MAX_LEN - 1, config.N_REGIONS)
        assert np.allclose(att.sum(-1), 1.0, atol=1e-4)


def test_bleu_perfect_and_partial():
    refs = [["a dog runs on the grass"], ["a cat sits on a red sofa"]]
    assert model.bleu(refs, ["a dog runs on the grass", "a cat sits on a red sofa"])["BLEU-4"] == pytest.approx(1.0)
    partial = model.bleu(refs, ["a dog runs on the beach", "a cat sits on a sofa"])
    assert 0 < partial["BLEU-4"] < partial["BLEU-1"] < 1


def test_pipeline_end_to_end(tmp_path):
    folder = tmp_path / "in"
    (folder / "sub").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("a.jpg", "sub/b.png", "c.JPG"):
        Image.fromarray(rng.integers(0, 255, (60, 90, 3), dtype=np.uint8)).save(folder / name)
    (folder / "broken.jpg").write_bytes(b"not an image")
    (folder / "notes.txt").write_text("skip me")
    assert [p.name for p in pipeline.list_images(folder)] == ["a.jpg", "broken.jpg", "c.JPG", "b.png"]

    vocab, *_ = _toy_model_data()
    classifier = cls_modeling.build_cnn(filters=(4,), dense_units=4)
    classifier.get_layer("output").set_weights([np.zeros((4, 1)), np.array([5.0])])   # always "photo"
    pipe = pipeline.CaptionPipeline(
        classifier=classifier, denoiser=dn_modeling.build_autoencoder(filters=(4, 8)), noise_threshold=0.0,
        encoder=features.build_encoder(weights=None),
        captioner=model.build_captioner(vocab_size=len(vocab), embed_dim=16, units=16), vocab=vocab)
    out = pipe.run(folder)
    assert list(out["file"]) == ["a.jpg", "broken.jpg", "c.JPG", "b.png"]
    ok = out[out["file"] != "broken.jpg"]
    assert ok["is_photo"].all() and ok["denoised"].all() and ok["caption"].map(lambda c: isinstance(c, str)).all()
    assert out.loc[out["file"] == "broken.jpg", "caption"].iloc[0].startswith("[unreadable")
    assert set(pipe.timings) == {"read + filter", "denoise", "encode", "caption"}


def test_plots():
    hist = pd.DataFrame({"epoch": [1, 2], "loss": [3.0, 2.5], "val_loss": [3.1, 2.7], "masked_accuracy": [0.3, 0.4],
                         "val_masked_accuracy": [0.3, 0.38], "val_bleu4": [0.05, 0.1]})
    assert evaluation.plot_history(hist) is not None
    assert evaluation.draw_architecture() is not None and evaluation.draw_workflow(4.0) is not None
    img = np.zeros((40, 60, 3), np.uint8)
    assert evaluation.show_captions([img, img], ["a dog", "a cat"], [["a dog runs"], ["a cat"]]) is not None
    assert evaluation.show_attention(img, "a dog runs", np.full((21, 64), 1 / 64)) is not None
