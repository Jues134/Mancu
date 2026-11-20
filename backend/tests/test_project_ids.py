"""Project-wide id allocation and repair of records created by older builds."""

from types import SimpleNamespace

from app.doc.agents_six_impl import (
    _merge_manifest,
    _rebind_board_asset_ids,
    _safe_previous_ids,
    _sequence_for,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def test_sequences_scan_all_segment_manifests_and_are_shared_live():
    ctx = _ctx(
        {
            "asset_index": {
                "seg1": {
                    "character_assets": [
                        {"character_id": "CHAR-005", "name": "甲"}
                    ]
                },
                "seg2": {
                    "character_assets": [
                        {"character_id": "CHAR-009", "name": "乙"}
                    ]
                },
            }
        }
    )

    first = _sequence_for(ctx, "character", "character_assets")
    second = _sequence_for(ctx, "character", "character_assets")

    assert first is second
    assert first.next() == "CHAR-010"
    assert second.next() == "CHAR-011"


def test_conflicting_saved_id_is_not_preserved_for_retry_segment():
    ctx = _ctx(
        {
            "asset_index": {
                "approved": {
                    "character_assets": [
                        {"character_id": "CHAR-006", "name": "毛泽东"}
                    ]
                },
                "retry": {
                    "character_assets": [
                        {"character_id": "CHAR-006", "name": "与会代表"}
                    ]
                },
            }
        }
    )

    retained = _safe_previous_ids(
        ctx, "retry", "character_assets", "character_id", canonical=True
    )

    assert retained == {}
    assert _sequence_for(ctx, "character", "character_assets").next() == "CHAR-007"


def test_reissued_character_id_updates_shot_by_character_name():
    board = {
        "shots": [
            {
                "shot_id": "SB-008",
                "characters": ["与会代表"],
                "character_ids": ["CHAR-006"],
                "prop_ids": [],
            }
        ]
    }
    previous = {
        "character_assets": [
            {"character_id": "CHAR-006", "name": "与会代表"}
        ]
    }
    current = {
        "character_assets": [
            {"character_id": "CHAR-007", "name": "与会代表"}
        ]
    }

    _rebind_board_asset_ids(board, previous, current)

    assert board["shots"][0]["character_ids"] == ["CHAR-007"]


def test_partial_merge_replaces_named_asset_when_collision_changes_its_id():
    previous = {
        "segment_id": "retry",
        "character_assets": [
            {"character_id": "CHAR-006", "name": "与会代表", "image_path": "old.png"}
        ],
        "scene_assets": [],
        "prop_assets": [],
        "image_generation_tasks": [
            {"asset_id": "CHAR-006", "local_path": "old.png", "status": "done"}
        ],
    }
    rebuilt = {
        "segment_id": "retry",
        "character_assets": [
            {"character_id": "CHAR-007", "name": "与会代表"}
        ],
        "scene_assets": [],
        "prop_assets": [],
        "image_generation_tasks": [
            {"asset_id": "CHAR-007", "status": "pending"}
        ],
    }

    merged = _merge_manifest(previous, rebuilt, ["CHAR-006"])

    assert [x["character_id"] for x in merged["character_assets"]] == ["CHAR-007"]
    assert [x["asset_id"] for x in merged["image_generation_tasks"]] == ["CHAR-007"]


def test_partial_merge_disambiguates_two_names_that_shared_one_bad_id():
    previous = {
        "segment_id": "seg1",
        "character_assets": [
            {"character_id": "CHAR-006", "name": "毛泽东"},
            {"character_id": "CHAR-006", "name": "与会代表"},
        ],
        "scene_assets": [],
        "prop_assets": [],
        "image_generation_tasks": [],
    }
    rebuilt = {
        "segment_id": "seg1",
        "character_assets": [
            {"character_id": "CHAR-006", "name": "毛泽东"},
            {"character_id": "CHAR-007", "name": "与会代表"},
        ],
        "scene_assets": [],
        "prop_assets": [],
        "image_generation_tasks": [],
    }

    merged = _merge_manifest(previous, rebuilt, ["CHAR-006"])

    assert [
        (x["character_id"], x["name"]) for x in merged["character_assets"]
    ] == [("CHAR-006", "毛泽东"), ("CHAR-007", "与会代表")]

