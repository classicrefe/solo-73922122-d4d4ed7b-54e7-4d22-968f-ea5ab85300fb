"""患者号更正：旧患者号 -> 规范患者号 的解析、映射链归并与校验。

规则：
- CSV 列：``旧患者号,规范患者号``；
- 同一旧号重复出现且指向不同规范号 -> 整批拒绝；
- 映射沿链传递（A->B、B->C 时 A、B 都归并为 C）；
- 映射成环（含 A->A 自环）-> 整批拒绝；
- 旧号必须存在于当前病例数据中 -> 否则整批拒绝；
- 多个旧号可以归并到同一规范号；
- 归并只改写患者号与亲属对；病例号、标签、配额、锁定保持不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from .models import (
    MAX_MAPPING_ROWS,
    MAPPING_HEADER_ALIASES,
    ActiveInput,
    ValidationError,
    _read_csv_rows,
)
from .splitter import Case, RelativePair


def parse_mapping_csv(raw: bytes) -> List[Tuple[str, str]]:
    """解析并做逐行格式校验；返回 (旧号, 规范号) 列表（保留重复行，后续去重）。"""
    header, body = _read_csv_rows(raw)
    cols = [MAPPING_HEADER_ALIASES.get(h) for h in header]
    missing = [
        name
        for name in ("old_patient_id", "canonical_patient_id")
        if name not in (cols or [])
    ]
    if missing:
        raise ValidationError([
            "患者号更正 CSV 表头缺少列：" + "、".join(missing)
            + f"。当前表头：{', '.join(header)}"
        ])
    if len(body) > MAX_MAPPING_ROWS:
        raise ValidationError([
            f"患者号更正数据最多 {MAX_MAPPING_ROWS} 行，当前 {len(body)} 行。"
        ])
    if not body:
        raise ValidationError(["患者号更正 CSV 没有数据行。"])

    i_old = cols.index("old_patient_id")
    i_new = cols.index("canonical_patient_id")
    rows: List[Tuple[str, str]] = []
    errors: List[str] = []
    for line_no, row in enumerate(body, start=2):
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        old_id = row[i_old].strip()
        new_id = row[i_new].strip()
        if not old_id:
            errors.append(f"第 {line_no} 行：旧患者号为空")
            continue
        if not new_id:
            errors.append(f"第 {line_no} 行：规范患者号为空（旧号 {old_id}）")
            continue
        rows.append((old_id, new_id))
    if errors:
        raise ValidationError(errors)
    return rows


@dataclass(frozen=True)
class PatientMapping:
    """已通过校验的映射。``rows`` 为去重后的直接映射，``resolved`` 为链式终值。"""

    rows: List[Tuple[str, str]]
    resolved: Dict[str, str]

    def final_id(self, patient_id: str) -> str:
        return self.resolved.get(patient_id, patient_id)


def resolve_patient_mapping(
    rows: Sequence[Tuple[str, str]], known_patients: Sequence[str]
) -> PatientMapping:
    """校验并解析映射链；任何错误都抛 ValidationError（调用方整体拒绝）。"""
    direct: Dict[str, str] = {}
    errors: List[str] = []
    unique_rows: List[Tuple[str, str]] = []
    seen_rows = set()
    for old_id, new_id in rows:
        key = (old_id, new_id)
        if key not in seen_rows:
            seen_rows.add(key)
            unique_rows.append(key)
        if old_id in direct and direct[old_id] != new_id:
            errors.append(
                f"旧患者号 {old_id} 同时指向不同规范号："
                f"{direct[old_id]} 与 {new_id}"
            )
            continue
        direct[old_id] = new_id
    if errors:
        raise ValidationError(errors)

    # 映射链求终值，同时检测环（A->A 也视为环）。
    resolved: Dict[str, str] = {}
    blocked = set()
    reported_cycles = set()
    for start in sorted(direct):
        if start in resolved or start in blocked:
            continue
        path: List[str] = []
        position: Dict[str, int] = {}
        cur = start
        final: str | None = None
        while True:
            if cur in resolved:
                final = resolved[cur]
                break
            if cur in position:
                cycle = path[position[cur]:]
                cycle_key = frozenset(cycle)
                if cycle_key not in reported_cycles:
                    reported_cycles.add(cycle_key)
                    errors.append("映射成环：" + " → ".join(cycle + [cur]))
                blocked.update(path)
                break
            if cur not in direct:
                final = cur
                break
            position[cur] = len(path)
            path.append(cur)
            cur = direct[cur]
        if final is not None:
            for node in path:
                resolved[node] = final

    known = set(known_patients)
    for old_id in sorted(direct):
        if old_id not in known:
            errors.append(f"旧患者号 {old_id} 不存在于当前病例数据中")
    if errors:
        raise ValidationError(errors)

    return PatientMapping(rows=unique_rows, resolved=resolved)


def _relative_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def apply_patient_mapping(
    active: ActiveInput, rows: Sequence[Tuple[str, str]]
) -> Tuple[PatientMapping, ActiveInput, dict]:
    """在内存中应用归并，返回 (映射, 新输入, 版本日志详情)。

    只改写患者号与亲属对：
    - 病例顺序、病例号、标签原样保留；
    - 配额、锁定原样保留；
    - 亲属对两端同步改写，删除归并产生的自亲属，并按无向关系去重。
    """
    mapping = resolve_patient_mapping(rows, [c.patient_id for c in active.cases])

    new_cases: List[Case] = []
    affected_case_ids: List[str] = []
    cases_per_old: Dict[str, int] = {}
    for c in active.cases:
        new_pid = mapping.final_id(c.patient_id)
        if new_pid != c.patient_id:
            affected_case_ids.append(c.case_id)
            cases_per_old[c.patient_id] = cases_per_old.get(c.patient_id, 0) + 1
        new_cases.append(Case(case_id=c.case_id, patient_id=new_pid, label=c.label))

    new_rels: List[RelativePair] = []
    seen = set()
    dropped_self = []
    dropped_duplicate = []
    for r in active.relatives:
        a = mapping.final_id(r.patient_a)
        b = mapping.final_id(r.patient_b)
        if a == b:
            dropped_self.append({
                "patient_a": r.patient_a,
                "patient_b": r.patient_b,
                "merged_patient_id": a,
            })
            continue
        key = _relative_key(a, b)
        if key in seen:
            dropped_duplicate.append({"patient_a": a, "patient_b": b})
            continue
        seen.add(key)
        new_rels.append(RelativePair(patient_a=a, patient_b=b))

    new_input = ActiveInput(
        cases=new_cases,
        relatives=new_rels,
        quotas=dict(active.quotas),
        locks=dict(active.locks),
    )

    merged_ids = [
        {
            "old_patient_id": old_id,
            "direct_target": direct_target,
            "canonical_patient_id": mapping.resolved[old_id],
            "affected_cases": cases_per_old.get(old_id, 0),
        }
        for old_id, direct_target in mapping.rows
    ]
    detail = {
        "mapping_rows": len(mapping.rows),
        "mappings": merged_ids,
        "merged_patient_ids": sorted(
            old for old in mapping.resolved if mapping.resolved[old] != old
        ),
        "affected_case_count": len(affected_case_ids),
        "affected_case_ids": sorted(affected_case_ids),
        "relatives_before": len(active.relatives),
        "relatives_after": len(new_rels),
        "dropped_self_relatives": dropped_self,
        "dropped_duplicate_relatives": dropped_duplicate,
        "unchanged": {
            "case_ids": True,
            "labels": True,
            "quotas": True,
            "locks": True,
        },
    }
    return mapping, new_input, detail
