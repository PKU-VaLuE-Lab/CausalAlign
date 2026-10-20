from __future__ import annotations

import json

import torch

from causalalign.data.records import RecordDataset


def test_record_dataset_merges_target_and_selects_transition(tmp_path):
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    torch.save({"clean_latent": torch.zeros(2, 3, 4, 5), "transitions": {"l10_s398": torch.arange(18).reshape(3, 6)}}, source)
    torch.save({"teacher_token": torch.ones(128)}, target)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"scene_id": "x", "file": "source.pt", "target_file": "target.pt", "pair_index": 1}) + "\n", encoding="utf-8")
    item = RecordDataset(index)[0]
    assert item["transitions"]["l10_s398"].shape == (6,)
    assert item["teacher_token"].shape == (128,)
    assert int(item["pair_index"]) == 1

