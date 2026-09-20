"""Option/probability mapping and variant construction without loading the model."""

from unittest.mock import patch

import torch

from jevc import infer
from jevc.schema import Item, Option, Source


def item():
    return Item(id="i1", group_id="g1", family="clinc", output_type="choice", template_id="t", domain="d",
                state="turn the lights off", question="Which intent?",
                options=[Option("smart_home", "smart home", "control a device"), Option("alarm", "alarm", "set an alarm"),
                         Option("timer", "timer", "set a timer")],
                gold="smart_home", source=Source("ds", "c", "r", "validation", 0, "cc-by-3.0", "overlap_unknown"))


def test_variants():
    it = item()
    assert infer.variant_view(it, "no_state")[1] == "" and infer.variant_view(it, "no_question")[0] == ""
    ro = infer.variant_view(it, "reordered")[2]
    assert {o.id for o in ro} == {o.id for o in it.options} and [o.id for o in ro] != [o.id for o in it.options]


def test_build_text_places_descriptions_before_state():
    t = infer.build_text("STATE", item().options)
    assert t.startswith("smart home: control a device alarm: set an alarm") and t.endswith(" STATE")
    assert infer.build_text("", item().options) == "smart home: control a device alarm: set an alarm timer: set a timer"


def test_input_modes_keep_model_visible_semantics_explicit():
    opts = item().options
    assert infer.model_labels(opts, "official_labels") == ["smart home", "alarm", "timer"]
    assert infer.model_labels(opts, "described_text") == ["smart home", "alarm", "timer"]
    assert infer.model_labels(opts, "described_labels") == [o.description for o in opts]
    assert infer.build_text("STATE", opts, "official_labels") == "STATE"
    assert infer.build_text("STATE", opts, "pairwise_descriptions") == "STATE"


def test_probs_map_back_to_option_ids_under_reordering():
    """Fake scorer: logit for an option depends only on its identity, so probabilities must be identical after reordering."""
    it = item()
    logit_by_id = {"smart_home": 2.0, "alarm": 0.0, "timer": -1.0}

    def fake_score(rt, views, input_mode="described_text"):
        out = []
        for _, _, opts in views:
            p = torch.softmax(torch.tensor([logit_by_id[o.id] for o in opts]), -1).tolist()
            out.append({"probs": dict(zip([o.id for o in opts], p)), "input_tokens": 5, "full_tokens": 5, "truncated": False})
        return out

    with patch.object(infer, "score_batch", fake_score):
        base = infer.run_items(None, [it], "full")[0]
        reo = infer.run_items(None, [it], "reordered")[0]
    assert base["presented_order"] != reo["presented_order"]
    assert base["probs"] == reo["probs"] and base["pred"] == reo["pred"] == "smart_home" and base["correct"]
    assert abs(sum(base["probs"].values()) - 1) < 1e-6
