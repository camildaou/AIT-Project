"""LSTM output dimensions, preset labels and checkpoint compatibility."""
import pandas as pd
import pytest

pytest.importorskip("torch")
from emotion_ait.lstm_classifier import LSTMClassifier, torch


@pytest.mark.parametrize("classes", [4, 27])
def test_class_count_training_validation_and_checkpoint(classes, tmp_path):
    frame = pd.DataFrame({"text": ["", "happy", "sad", "unknown"],
                          "label": [0, classes - 1, 1, 0]})
    validation = pd.DataFrame({"text": ["validation_only"], "label": [classes - 1]})
    model = LSTMClassifier(num_classes=classes, embedding_dim=4, hidden_dim=5,
                           batch_size=2, max_epochs=1).fit(frame, validation)
    assert model._model.output.out_features == classes
    assert "validation_only" not in model._vocab
    predictions = model.predict(["", "happy", "unseen words"])
    assert all(0 <= label < classes for label in predictions)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = LSTMClassifier.load(path)
    assert restored.num_classes == classes
    assert restored.predict(["", "happy", "unseen words"]) == predictions
    if classes == 4:
        payload = torch.load(path, weights_only=False)
        del payload["params"]["num_classes"]
        torch.save(payload, path)
        assert LSTMClassifier.load(path).predict(["", "happy", "unseen words"]) == predictions


@pytest.mark.parametrize("classes", [0, 1, 2.5, True])
def test_rejects_invalid_class_count(classes):
    with pytest.raises(ValueError):
        LSTMClassifier(num_classes=classes)


def test_rejects_out_of_range_training_and_validation_labels():
    model = LSTMClassifier(num_classes=27, max_epochs=1)
    good = pd.DataFrame({"text": ["valid"], "label": [26]})
    bad = pd.DataFrame({"text": ["invalid"], "label": [27]})
    with pytest.raises(ValueError):
        model.fit(bad)
    with pytest.raises(ValueError):
        model.fit(good, bad)
