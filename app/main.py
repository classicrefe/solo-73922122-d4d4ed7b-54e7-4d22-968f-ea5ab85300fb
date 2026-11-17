"""FastAPI 入口：路由、预览令牌缓存、失效管控。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import Database, StaleVersionError
from .identity import apply_patient_mapping, parse_mapping_csv
from .models import (
    ActiveInput,
    ValidationError,
    parse_cases_csv,
    parse_relatives_csv,
    validate_locks,
    validate_quotas,
)
from .splitter import SplitterError, solve_split

import os

DB_PATH = Path(os.environ.get("SPLIT_DB_PATH",
                              Path(__file__).resolve().parent.parent / "data" / "app.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="病例切分平台", version="1.0.0")
db = Database(str(DB_PATH))


# --------------------------------------------------------------------------- #
# 预览缓存：仅存内存。重启即清空——重启后不存在任何“旧预览”可被发布。
# 每个预览绑定其求解时的数据版本；版本一旦变动，令牌立即作废。
# --------------------------------------------------------------------------- #
_PREVIEW_MAX = 64
_previews: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_preview_lock = threading.Lock()

# 患者号更正预览：与求解预览一样仅存内存，并绑定预览时的输入版本。
# 版本一旦变动（包括提交归并本身），令牌即失效，必须重新上传预览。
_mapping_previews: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _add_mapping_preview(version: int, preview: dict) -> str:
    token = uuid.uuid4().hex
    with _preview_lock:
        _mapping_previews[token] = {"version": version, "preview": preview}
        _mapping_previews.move_to_end(token)
        while len(_mapping_previews) > _PREVIEW_MAX:
            _mapping_previews.popitem(last=False)
    return token


def _get_mapping_preview(token: str, expected_version: int) -> Dict[str, Any]:
    with _preview_lock:
        preview = _mapping_previews.get(token)
        if preview is None:
            raise StaleVersionError(
                "该身份更正预览不存在或已因重启/过期被清除，请重新上传 CSV 预览。"
            )
        current_version = db.get_version()
        if preview["version"] != expected_version or expected_version != current_version:
            raise StaleVersionError(
                f"身份更正预览依据的输入版本 v{preview['version']} 或网页所见版本"
                f" v{expected_version} 已变动为 v{current_version}，"
                "该预览已永久失效，本次未写入任何数据，请刷新并重新预览。"
            )
        return preview["preview"]


def _consume_mapping_preview(token: str, new_version: int) -> None:
    """身份更正成功后立即清理旧令牌：该更正预览被消费，其余绑定旧版本的预览
    也一并移除，保证旧令牌永久失效（数据库版本核对是第二道防线）。"""
    with _preview_lock:
        _mapping_previews.pop(token, None)
        for key in [k for k, v in _mapping_previews.items() if v["version"] != new_version]:
            _mapping_previews.pop(key, None)
        for key in [k for k, v in _previews.items() if v["version"] != new_version]:
            _previews.pop(key, None)


def _add_preview(version: int, input_json: dict, result: dict) -> str:
    token = uuid.uuid4().hex
    with _preview_lock:
        _previews[token] = {
            "version": version,
            "input_json": input_json,
            "result": result,
            "created_at": time.time(),
        }
        _previews.move_to_end(token)
        while len(_previews) > _PREVIEW_MAX:
            _previews.popitem(last=False)
    return token


def _get_preview(token: str) -> Dict[str, Any]:
    with _preview_lock:
        preview = _previews.get(token)
        if preview is None:
            raise StaleVersionError("该预览不存在或已因重启/过期被清除，不能发布，请重新求解。")
        current_version = db.get_version()
        if preview["version"] != current_version:
            raise StaleVersionError(
                f"预览依据的输入版本 v{preview['version']} 已变动为"
                f" v{current_version}，该预览已永久失效，不能发布，请重新求解。"
            )
        return preview


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class QuotasPayload(BaseModel):
    g1_label0: int = Field(ge=0)
    g1_label1: int = Field(ge=0)
    g2_label0: int = Field(ge=0)
    g2_label1: int = Field(ge=0)


class LocksPayload(BaseModel):
    locks: list  # [{"case_id": str, "group": 1|2}, ...]


class PublishPayload(BaseModel):
    preview_token: str


# --------------------------------------------------------------------------- #
# 异常处理
# --------------------------------------------------------------------------- #
@app.exception_handler(ValidationError)
async def _validation_handler(_request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": "输入校验失败", "errors": exc.errors})


@app.exception_handler(SplitterError)
async def _splitter_handler(_request, exc: SplitterError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(StaleVersionError)
async def _stale_handler(_request, exc: StaleVersionError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# 状态与上传
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "version": db.get_version()}


@app.get("/api/state")
def get_state():
    version, active = db.get_state()
    label_totals = {0: 0, 1: 0}
    for c in active.cases:
        label_totals[c.label] += 1
    return {
        "dataset_version": version,
        "label_totals": label_totals,
        "patient_count": len({c.patient_id for c in active.cases}),
        "input": active.to_json(),
        "manifests": db.list_manifests(),
    }


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise ValidationError([f"上传文件 {file.filename!r} 为空。"])
    if len(raw) > 2 * 1024 * 1024:
        raise ValidationError(["上传文件超过 2MB 限制。"])
    return raw


@app.post("/api/upload/cases")
async def upload_cases(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    # 解析/校验全部完成且无异常后才写库；失败不触碰源数据
    cases = parse_cases_csv(raw)
    new_version = db.replace_cases(cases)
    return {"dataset_version": new_version, "case_count": len(cases),
            "notice": "病例已替换；亲属关系与指向已删除病例的锁定已清空，需要时请重新设置。"}


@app.post("/api/upload/relatives")
async def upload_relatives(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    _, active = db.get_state()
    if not active.cases:
        raise ValidationError(["请先上传病例 CSV，再上传亲属关系。"])
    relatives = parse_relatives_csv(raw, [c.patient_id for c in active.cases])
    new_version = db.replace_relatives(relatives)
    return {"dataset_version": new_version, "pair_count": len(relatives)}


# --------------------------------------------------------------------------- #
# 患者号更正（旧号 -> 规范号 归并）
# 两步式：先预览（返回绑定输入版本的预览令牌），再按网页所见版本提交。
# 预览不改动任何数据；提交时在同一个 IMMEDIATE 事务内核对版本、应用映射、
# 递增版本并使旧清单失效——过期或失败时现行状态完全不变。
# --------------------------------------------------------------------------- #
@app.post("/api/patient-mapping/preview")
async def preview_patient_mapping(
    file: UploadFile = File(...),
    expected_version: int = Form(...),
):
    raw = await _read_upload(file)
    current_version, active = db.get_state()
    if not active.cases:
        raise ValidationError(["还没有病例数据，无法做患者号更正。"])
    if expected_version != current_version:
        raise StaleVersionError(
            f"网页所见输入版本 v{expected_version} 已变为 v{current_version}，"
            "请刷新后再上传更正表。"
        )
    rows = parse_mapping_csv(raw)
    mapping, new_input, detail = apply_patient_mapping(active, rows)
    preview = {
        "input_version": expected_version,
        "mapping_rows": [(old, new) for old, new in mapping.rows],
        "mapping_row_count": len(mapping.rows),
        "resolved": [
            {
                "old_patient_id": old,
                "canonical_patient_id": mapping.resolved[old],
            }
            for old in sorted(mapping.resolved)
        ],
        "detail": detail,
        "relatives_after": [
            {"patient_a": r.patient_a, "patient_b": r.patient_b}
            for r in new_input.relatives
        ],
        "cases_after": [
            {"case_id": c.case_id, "patient_id": c.patient_id, "label": c.label}
            for c in new_input.cases
        ],
    }
    token = _add_mapping_preview(expected_version, preview)
    return {
        "preview_token": token,
        "dataset_version": expected_version,
        "mapping_rows": [
            {"old_patient_id": old, "canonical_patient_id": new}
            for old, new in preview["mapping_rows"]
        ],
        "mapping_row_count": preview["mapping_row_count"],
        "resolved": preview["resolved"],
        "detail": preview["detail"],
        "relatives_after": preview["relatives_after"],
        "cases_after": preview["cases_after"],
    }


@app.post("/api/patient-mapping/commit")
async def commit_patient_mapping(
    preview_token: str = Form(...),
    expected_version: int = Form(...),
):
    preview = _get_mapping_preview(preview_token, expected_version)
    # 完整映射链在事务内重新解析（以现行数据为准），版本核对不通过则整体回滚
    try:
        new_version, detail = db.apply_patient_mapping(
            preview["mapping_rows"], expected_version
        )
    except (StaleVersionError, ValidationError):
        # 版本过期 -> 409；映射在事务内复核失败 -> 400；两者均已整体回滚
        raise
    except Exception as exc:  # 写入失败：事务已回滚，现行状态未变
        raise HTTPException(
            status_code=500,
            detail=f"身份更正写入失败，已整体回滚，现行状态未变：{exc}",
        )
    _consume_mapping_preview(preview_token, new_version)
    _, active = db.get_state()
    return {
        "dataset_version": new_version,
        "previous_version": expected_version,
        "detail": detail,
        "case_count": len(active.cases),
        "patient_count": len({c.patient_id for c in active.cases}),
        "relatives": [
            {"patient_a": r.patient_a, "patient_b": r.patient_b}
            for r in active.relatives
        ],
    }


@app.put("/api/quotas")
def update_quotas(payload: QuotasPayload):
    _, active = db.get_state()
    totals = {0: 0, 1: 0}
    for c in active.cases:
        totals[c.label] += 1
    quotas = validate_quotas(payload.model_dump(), totals)
    new_version = db.update_quotas(quotas)
    return {"dataset_version": new_version, "quotas": payload.model_dump()}


@app.put("/api/locks")
def update_locks(payload: LocksPayload):
    _, active = db.get_state()
    locks = validate_locks(payload.locks, [c.case_id for c in active.cases])
    new_version = db.update_locks(locks)
    return {"dataset_version": new_version, "lock_count": len(locks)}


# --------------------------------------------------------------------------- #
# 求解（预览）与发布
# --------------------------------------------------------------------------- #
@app.post("/api/solve")
def solve():
    # 先取输入快照与版本；z3 计算在数据库锁之外进行，计算期间允许其他请求改输入
    version, active = db.get_state()
    if not active.cases:
        raise ValidationError(["还没有病例数据，请先上传病例 CSV。"])
    input_snapshot = active.to_json()
    result = solve_split(
        active.cases, active.relatives, active.quotas, active.locks
    )
    # 计算结束后复核版本：若期间输入被改动，本次结果对应旧版本，拒绝生成预览
    current_version = db.get_version()
    if current_version != version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"计算期间输入已从 v{version} 变动到 v{current_version}，"
                "本次结果已作废且不会生成预览，请重新求解。"
            ),
        )
    token = None
    if result["status"] == "sat":
        token = _add_preview(version, input_snapshot, result)
    return {
        "dataset_version": version,
        "preview_token": token,
        "result": result,
        "input_snapshot": input_snapshot,
    }


@app.post("/api/publish")
def publish(payload: PublishPayload):
    preview = _get_preview(payload.preview_token)  # 过期/失效 -> 409
    result = preview["result"]
    if result.get("status") != "sat":
        raise HTTPException(status_code=400, detail="无解的预览不能发布。")
    manifest = db.publish_manifest(
        expected_version=preview["version"],
        input_data=preview["input_json"],
        result=result,
    )
    return manifest


@app.get("/api/manifests")
def manifests():
    return {"dataset_version": db.get_version(), "manifests": db.list_manifests()}


@app.get("/api/manifests/{manifest_id}")
def manifest_detail(manifest_id: str):
    manifest = db.get_manifest(manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="清单不存在。")
    return manifest


@app.get("/api/manifests/{manifest_id}/download")
def manifest_download(manifest_id: str):
    manifest = db.get_manifest(manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="清单不存在。")
    if manifest["expired"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "该清单依据的输入已变动，清单已永久失效，禁止下载。"
                "可在“已发布清单”中查看其历史记录与失效原因。"
            ),
        )
    result = manifest["result"]
    assignments = result["assignments"]
    locks = manifest["input"].get("locks", {})
    lines = ["病例号,患者号,标签,分组编号,分组名称,是否锁定"]
    for c in manifest["input"]["cases"]:
        cid = c["case_id"]
        grp = assignments[cid]
        locked = "是" if cid in locks else "否"
        lines.append(
            f"{cid},{c['patient_id']},{c['label']},{grp},"
            f"{'训练集' if grp == 1 else '评估集'},{locked}"
        )
    csv_text = "﻿" + "\n".join(lines) + "\n"
    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="split_manifest_{manifest_id[:8]}_v{manifest["dataset_version"]}.csv"'
            )
        },
    )


@app.get("/api/version-log")
def version_log():
    return {"current_version": db.get_version(), "log": db.version_log()}


# 静态前端（最后挂载，避免覆盖 /api 路由）
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
