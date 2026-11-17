"""患者号更正的 API 全链路测试：预览/提交、过期与写入失败不改状态、
旧预览与清单永久失效、归并导致配额无解仍保存。"""
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 可求解数据：两个连通分量，各含一个标签1与一个标签0
CASES_CSV = (
    "病例号,患者号,标签\n"
    "C1,P1,1\nC2,P2,0\nC3,P3,1\nC4,P4,0\n"
)
REL_CSV = "患者A,患者B\nP1,P2\nP3,P4\n"

# 用于验证亲属对改写的数据：P2 并入 P1 后会产生自亲属与重复关系
MERGE_CASES_CSV = (
    "病例号,患者号,标签\n"
    "C1,P1,1\nC2,P2,1\nC3,P3,0\nC4,P4,0\n"
)
MERGE_REL_CSV = "患者A,患者B\nP1,P2\nP3,P4\nP2,P3\nP1,P3\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLIT_DB_PATH", str(tmp_path / "test.db"))
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    main.db.close()


def _upload_base(client, cases_csv=CASES_CSV, rel_csv=REL_CSV):
    r = client.post("/api/upload/cases",
                    files={"file": ("cases.csv", cases_csv.encode(), "text/csv")})
    assert r.status_code == 200, r.text
    if rel_csv is not None:
        r = client.post("/api/upload/relatives",
                        files={"file": ("rel.csv", rel_csv.encode(), "text/csv")})
        assert r.status_code == 200, r.text
    r = client.put("/api/quotas", json={"g1_label0": 1, "g1_label1": 1,
                                        "g2_label0": 1, "g2_label1": 1})
    assert r.status_code == 200, r.text
    r = client.put("/api/locks", json={"locks": [{"case_id": "C1", "group": 1}]})
    assert r.status_code == 200, r.text
    return client.get("/api/state").json()["dataset_version"]


def _setup(client):
    return _upload_base(client)


def _setup_merge(client):
    return _upload_base(client, MERGE_CASES_CSV, MERGE_REL_CSV)


def _preview(client, csv, version):
    return client.post(
        "/api/patient-mapping/preview",
        files={"file": ("map.csv", csv.encode(), "text/csv")},
        data={"expected_version": str(version)},
    )


def _commit(client, token, version):
    return client.post(
        "/api/patient-mapping/commit",
        data={"preview_token": token, "expected_version": str(version)},
    )


def _state(client):
    return client.get("/api/state").json()


def test_preview_returns_chain_and_does_not_write(client):
    v = _setup(client)
    r = _preview(client, "旧患者号,规范患者号\nP1,P2\nP2,P5\n", v)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset_version"] == v
    resolved = {m["old_patient_id"]: m["canonical_patient_id"]
                for m in body["resolved"]}
    assert resolved == {"P1": "P5", "P2": "P5"}
    # P1,P2 两患者的病例均被改写
    assert body["detail"]["affected_case_count"] == 2
    # 预览不写库
    assert _state(client)["dataset_version"] == v
    assert _state(client)["input"]["cases"][0]["patient_id"] == "P1"


def test_commit_applies_mapping_bumps_version_and_keeps_other_inputs(client):
    v = _setup(client)
    # P2->P1：C2 改属 P1；P1-P2 成自亲属删除
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    r = _commit(client, body["preview_token"], v)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["dataset_version"] == v + 1
    assert out["previous_version"] == v

    state = _state(client)
    assert state["dataset_version"] == v + 1
    cases = state["input"]["cases"]
    # 病例号、顺序、标签不变；患者号已改写
    assert [(c["case_id"], c["label"]) for c in cases] == [
        ("C1", 1), ("C2", 0), ("C3", 1), ("C4", 0)]
    assert [c["patient_id"] for c in cases] == ["P1", "P1", "P3", "P4"]
    # 配额、锁定保持不变（C1 仍在锁定表中）
    assert state["input"]["quotas"] == {"g1_label0": 1, "g1_label1": 1,
                                        "g2_label0": 1, "g2_label1": 1}
    assert state["input"]["locks"] == {"C1": 1}
    # 亲属对：P1-P2 自亲属删除；P3-P4 保留
    rels = {(r["patient_a"], r["patient_b"]) for r in state["input"]["relatives"]}
    assert rels == {("P3", "P4")}

    log = client.get("/api/version-log").json()["log"]
    assert log[0]["change_type"] == "merge_patient_ids"
    assert log[0]["dataset_version"] == v + 1
    assert log[0]["detail"]["affected_case_count"] == 1


def test_commit_removes_self_and_duplicate_relatives(client):
    v = _setup_merge(client)
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    detail = body["detail"]
    assert detail["relatives_before"] == 4
    # P1-P2 -> 自亲属；P2-P3 与 P1-P3 均变成 P1-P3 -> 去重
    assert detail["dropped_self_relatives"] == [
        {"patient_a": "P1", "patient_b": "P2", "merged_patient_id": "P1"}]
    assert detail["dropped_duplicate_relatives"] == [
        {"patient_a": "P1", "patient_b": "P3"}]
    assert _commit(client, body["preview_token"], v).status_code == 200
    state = _state(client)
    rels = [(r["patient_a"], r["patient_b"]) for r in state["input"]["relatives"]]
    assert rels == [("P3", "P4"), ("P1", "P3")]
    # 病例号/标签/锁定不变
    assert [(c["case_id"], c["label"]) for c in state["input"]["cases"]] == [
        ("C1", 1), ("C2", 1), ("C3", 0), ("C4", 0)]
    assert state["input"]["locks"] == {"C1": 1}


def test_commit_invalidates_old_preview_and_manifests(client):
    v = _setup(client)
    solve_token = client.post("/api/solve").json()["preview_token"]
    mid = client.post("/api/publish", json={"preview_token": solve_token}).json()["manifest_id"]
    assert client.get(f"/api/manifests/{mid}/download").status_code == 200

    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    assert _commit(client, body["preview_token"], v).status_code == 200

    # 旧求解预览与新身份更正预览都不能再用
    assert client.post("/api/publish", json={"preview_token": solve_token}).status_code == 409
    assert _commit(client, body["preview_token"], v + 1).status_code == 409
    # 旧清单永久失效
    detail = client.get(f"/api/manifests/{mid}").json()
    assert detail["expired"] is True and detail["invalidated"] is True
    assert client.get(f"/api/manifests/{mid}/download").status_code == 409


def test_stale_expected_version_commit_changes_nothing(client):
    v = _setup(client)
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    # 另一个窗口变动了输入
    client.put("/api/locks", json={"locks": []})
    after = _state(client)["dataset_version"]
    before_snapshot = _state(client)["input"]

    r = _commit(client, body["preview_token"], v)
    assert r.status_code == 409
    # 现行状态完全不变
    state = _state(client)
    assert state["dataset_version"] == after
    assert state["input"] == before_snapshot
    assert state["input"]["cases"][1]["patient_id"] == "P2"


def test_stale_expected_version_preview_changes_nothing(client):
    v = _setup(client)
    client.put("/api/locks", json={"locks": []})
    after = _state(client)["dataset_version"]
    r = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v)
    assert r.status_code == 409
    assert _state(client)["dataset_version"] == after
    assert _state(client)["input"]["cases"][1]["patient_id"] == "P2"


def test_invalid_mapping_rejects_whole_batch(client):
    v = _setup(client)
    cases = [
        "旧患者号,规范患者号\nP2,P1\nP999,P1\n",   # 旧号不存在
        "旧患者号,规范患者号\nP1,P3\nP1,P4\n",      # 同旧号不同规范号
        "旧患者号,规范患者号\nP1,P2\nP2,P1\n",      # 成环
        "旧患者号,规范患者号\nP1,P1\n",             # 自环
    ]
    for csv in cases:
        r = _preview(client, csv, v)
        assert r.status_code == 400, (csv, r.text)
        assert r.json()["errors"]
    state = _state(client)
    assert state["dataset_version"] == v
    assert state["input"]["cases"][0]["patient_id"] == "P1"
    assert len(state["input"]["relatives"]) == 2


def test_success_creates_new_identity_even_if_quotas_become_unsolvable(client):
    v = _setup(client)
    # 把 P2 并入 P1 后 C1(P1,标签1) 与 C2(P1,标签0) 同患者必须同组；
    # 再锁定 C1 到组1、C2 到组2 -> 新身份关系让原配额无解，但归并必须保存
    client.put("/api/locks", json={"locks": [{"case_id": "C1", "group": 1},
                                             {"case_id": "C2", "group": 2}]})
    v = _state(client)["dataset_version"]
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    assert _commit(client, body["preview_token"], v).status_code == 200

    state = _state(client)
    assert state["dataset_version"] == v + 1
    assert [c["patient_id"] for c in state["input"]["cases"]] == ["P1", "P1", "P3", "P4"]
    r = client.post("/api/solve").json()
    assert r["result"]["status"] == "unsat"
    assert r["preview_token"] is None
    assert r["result"]["mus_verified"] is True
    kinds = {c["kind"] for c in r["result"]["conflicts"]}
    assert "lock" in kinds


def test_unknown_token_changes_nothing(client):
    v = _setup(client)
    before = _state(client)["input"]
    r = _commit(client, "does-not-exist", v)
    assert r.status_code == 409
    assert _state(client)["dataset_version"] == v
    assert _state(client)["input"] == before


def test_database_write_failure_rolls_back_and_keeps_current_state(client):
    """模拟提交写入中途失败：事务整体回滚，版本、输入、清单均不变。"""
    import app.main as main

    v = _setup(client)
    solve_token = client.post("/api/solve").json()["preview_token"]
    mid = client.post("/api/publish", json={"preview_token": solve_token}).json()["manifest_id"]
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    before = _state(client)

    # 在版本日志写入处制造真实的数据库失败（此前 state 更新已执行）
    main.db._conn.execute(
        "CREATE TRIGGER fail_log BEFORE INSERT ON version_log "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )
    r = _commit(client, body["preview_token"], v)
    assert r.status_code == 500
    assert "已整体回滚" in r.json()["detail"]
    main.db._conn.execute("DROP TRIGGER fail_log")

    after = _state(client)
    assert after["dataset_version"] == before["dataset_version"] == v
    assert after["input"] == before["input"]
    # 旧清单未受影响，仍可下载
    detail = client.get(f"/api/manifests/{mid}").json()
    assert detail["expired"] is False
    assert client.get(f"/api/manifests/{mid}/download").status_code == 200
    # 预览未被成功消费之外的任何状态未变，重新提交可成功
    assert _commit(client, body["preview_token"], v).status_code == 200
    assert _state(client)["dataset_version"] == v + 1


def test_multiple_old_ids_merge_to_one_canonical(client):
    v = _setup_merge(client)
    body = _preview(client, "旧患者号,规范患者号\nP2,P9\nP4,P9\n", v).json()
    resolved = {m["old_patient_id"]: m["canonical_patient_id"] for m in body["resolved"]}
    assert resolved == {"P2": "P9", "P4": "P9"}
    assert _commit(client, body["preview_token"], v).status_code == 200
    state = _state(client)
    assert [c["patient_id"] for c in state["input"]["cases"]] == ["P1", "P9", "P3", "P9"]


def test_string_ids_with_leading_zeros_preserved(client):
    cases = "病例号,患者号,标签\nC1,001,0\nC2,002,1\nC3,003,0\n"
    _upload_base(client, cases, "患者A,患者B\n001,002\n")
    v = _state(client)["dataset_version"]
    body = _preview(client, "旧患者号,规范患者号\n002,001\n", v).json()
    assert body["resolved"] == [
        {"old_patient_id": "002", "canonical_patient_id": "001"}]
    assert _commit(client, body["preview_token"], v).status_code == 200
    assert [c["patient_id"] for c in _state(client)["input"]["cases"]] == ["001", "001", "003"]


def test_new_identity_alone_makes_quota_unsolvable_but_is_saved(client):
    """仅靠新身份关系（无锁定）就让原配额无解，归并仍须保存，求解如实报告。"""
    cases = "病例号,患者号,标签\nC1,P1,1\nC2,P1,1\nC3,P2,1\nC4,P2,1\n"
    _upload_base(client, cases, None)
    # 组1 标签1=2、组2 标签1=2 —— 归并前可行（P1 与 P2 两组）
    client.put("/api/quotas", json={"g1_label0": 0, "g1_label1": 2,
                                    "g2_label0": 0, "g2_label1": 2})
    v = _state(client)["dataset_version"]
    assert client.post("/api/solve").json()["result"]["status"] == "sat"

    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    assert _commit(client, body["preview_token"], v).status_code == 200
    state = _state(client)
    assert state["dataset_version"] == v + 1
    assert [c["patient_id"] for c in state["input"]["cases"]] == ["P1"] * 4
    assert state["input"]["quotas"] == {"g1_label0": 0, "g1_label1": 2,
                                        "g2_label0": 0, "g2_label1": 2}

    r = client.post("/api/solve").json()
    assert r["result"]["status"] == "unsat"
    assert r["preview_token"] is None
    kinds = {c["kind"] for c in r["result"]["conflicts"]}
    assert "kinship" in kinds and "quota" in kinds


def test_concurrent_commit_of_same_preview_applies_exactly_once(client):
    """两个窗口同时提交同一预览：只允许一次生效，另一次因版本已变而 409。"""
    import concurrent.futures

    v = _setup(client)
    body = _preview(client, "旧患者号,规范患者号\nP2,P1\n", v).json()
    token = body["preview_token"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _commit(client, token, v), range(2)))
    codes = sorted(r.status_code for r in results)
    assert codes == [200, 409], [r.text for r in results]

    state = _state(client)
    assert state["dataset_version"] == v + 1
    assert [c["patient_id"] for c in state["input"]["cases"]] == ["P1", "P1", "P3", "P4"]
    # 版本日志只记录一次归并
    merges = [e for e in client.get("/api/version-log").json()["log"]
              if e["change_type"] == "merge_patient_ids"]
    assert len(merges) == 1


def test_persistence_of_merge_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLIT_DB_PATH", str(tmp_path / "persist.db"))
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c1:
        v = _setup(c1)
        body = _preview(c1, "旧患者号,规范患者号\nP2,P1\n", v).json()
        assert _commit(c1, body["preview_token"], v).status_code == 200
    main.db.close()

    importlib.reload(main)
    with TestClient(main.app) as c2:
        state = c2.get("/api/state").json()
        assert state["dataset_version"] == v + 1
        assert [c["patient_id"] for c in state["input"]["cases"]] == ["P1", "P1", "P3", "P4"]
        # 重启后旧更正预览不可提交
        assert _commit(c2, body["preview_token"], v + 1).status_code == 409
    main.db.close()
