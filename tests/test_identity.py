"""患者号更正（归并）单元测试：映射链、冲突/环/缺失校验、亲属对改写。"""
import pytest

from app.identity import (
    apply_patient_mapping,
    parse_mapping_csv,
    resolve_patient_mapping,
)
from app.models import ActiveInput, ValidationError
from app.splitter import Case, RelativePair


def _input():
    return ActiveInput(
        cases=[
            Case("C1", "P1", 1),
            Case("C2", "P2", 1),
            Case("C3", "P3", 0),
            Case("C4", "P4", 0),
        ],
        relatives=[
            RelativePair("P1", "P2"),
            RelativePair("P3", "P4"),
            RelativePair("P1", "P3"),
        ],
        quotas={(1, 0): 1, (1, 1): 1, (2, 0): 1, (2, 1): 1},
        locks={"C1": 1, "C2": 2},
    )


def test_mapping_chain_resolves_to_final_canonical():
    m = resolve_patient_mapping(
        [("P1", "P2"), ("P2", "P3")], ["P1", "P2", "P3"]
    )
    assert m.final_id("P1") == "P3"
    assert m.final_id("P2") == "P3"
    assert m.final_id("P3") == "P3"


def test_multiple_old_ids_to_same_canonical_allowed():
    m = resolve_patient_mapping(
        [("P1", "P9"), ("P2", "P9")], ["P1", "P2"]
    )
    assert m.final_id("P1") == m.final_id("P2") == "P9"


def test_old_id_missing_from_cases_rejects_whole_batch():
    with pytest.raises(ValidationError) as exc:
        resolve_patient_mapping([("P1", "P2"), ("P999", "P2")], ["P1", "P2"])
    assert any("P999" in e for e in exc.value.errors)


def test_same_old_id_different_targets_rejects():
    with pytest.raises(ValidationError) as exc:
        resolve_patient_mapping([("P1", "P2"), ("P1", "P3")], ["P1", "P2", "P3"])
    assert any("同时指向不同规范号" in e for e in exc.value.errors)


def test_cycle_rejected():
    with pytest.raises(ValidationError) as exc:
        resolve_patient_mapping([("P1", "P2"), ("P2", "P1")], ["P1", "P2"])
    assert any("成环" in e for e in exc.value.errors)


def test_self_mapping_is_cycle_rejected():
    with pytest.raises(ValidationError) as exc:
        resolve_patient_mapping([("P1", "P1")], ["P1"])
    assert any("成环" in e for e in exc.value.errors)


def test_apply_rewrites_patients_and_relatives_only():
    active = _input()
    _, new_input, detail = apply_patient_mapping(
        active, [("P2", "P1"), ("P4", "P1")]
    )
    # 病例号、顺序、标签不变；患者号按链改写
    assert [(c.case_id, c.patient_id, c.label) for c in new_input.cases] == [
        ("C1", "P1", 1),
        ("C2", "P1", 1),
        ("C3", "P3", 0),
        ("C4", "P1", 0),
    ]
    # 配额与锁定保持原样
    assert new_input.quotas == active.quotas
    assert new_input.locks == {"C1": 1, "C2": 2}
    # P1-P2 变自亲属删除；P3-P4 -> P3-P1；P1-P3 重复去掉
    assert new_input.relatives == [RelativePair("P3", "P1")]
    assert detail["dropped_self_relatives"] == [
        {"patient_a": "P1", "patient_b": "P2", "merged_patient_id": "P1"}
    ]
    assert detail["dropped_duplicate_relatives"] == [
        {"patient_a": "P1", "patient_b": "P3"}
    ]
    assert detail["affected_case_count"] == 2
    assert detail["relatives_before"] == 3
    assert detail["relatives_after"] == 1


def test_merge_that_creates_quota_conflict_is_still_applied_at_data_level():
    # 把 P2 并入 P1 后，C1 锁定组1、C2 锁定组2 变成同一患者却分属两组 -> 必然无解，
    # 但归并本身不做求解，照常写回。
    active = _input()
    _, new_input, _ = apply_patient_mapping(active, [("P2", "P1")])
    assert {c.patient_id for c in new_input.cases} == {"P1", "P3", "P4"}
    assert new_input.locks == {"C1": 1, "C2": 2}


def test_duplicate_identical_rows_deduplicated():
    m = resolve_patient_mapping([("P1", "P2"), ("P1", "P2")], ["P1", "P2"])
    assert m.rows == [("P1", "P2")]


def test_parse_mapping_csv_aliases_and_validation():
    rows = parse_mapping_csv("旧患者号,规范患者号\nP1,P2\n".encode())
    assert rows == [("P1", "P2")]
    rows2 = parse_mapping_csv("old_patient_id,canonical_patient_id\nP1,P2\n".encode())
    assert rows2 == [("P1", "P2")]
    with pytest.raises(ValidationError) as exc:
        parse_mapping_csv("旧患者号,规范患者号\nP1,\n".encode())
    assert any("规范患者号为空" in e for e in exc.value.errors)
    with pytest.raises(ValidationError):
        parse_mapping_csv("旧患者号\nP1\n".encode())
