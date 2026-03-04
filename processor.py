from __future__ import annotations

import math
import importlib.util
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd

MONTH_PATTERN = re.compile(
    r"^\s*(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)\s*[-/ ]\s*(\d{2}|\d{4})\s*$",
    re.IGNORECASE,
)
HEADER_DATE_PATTERN = re.compile(r"\bDATE\b", re.IGNORECASE)

METADATA_TOKENS = {
    "date",
    "day",
    "from",
    "to",
    "town",
    "place",
    "km",
    "distance",
    "remark",
    "remarks",
    "description",
    "details",
    "narration",
    "mode",
    "total",
    "grand",
    "grand total",
    "actual",
    "real",
    "call",
    "employee",
    "employee name",
    "name",
    "code",
    "employee code",
    "id",
    "month",
    "year",
    "designation",
}

EXCLUDED_HEAD_PATTERNS = (
    "daily actual exp",
    "daily acutal exp",
    "daily actual expense",
    "daily acutal expense",
    "calls done",
    "call done",
)

POLICY = {
    "BM & SR. BM": {"da_per_day": 240.0, "os_per_day": 500.0, "fare_rate": 3.0, "fare_max_km": 150.0, "internet_cap": 375.0},
    "RM & SR. RM": {"da_per_day": 265.0, "os_per_day": 1200.0, "fare_rate": 3.0, "fare_max_km": 150.0, "internet_cap": 1000.0},
    "DY ZM": {"da_per_day": 275.0, "os_per_day": 1750.0, "fare_rate": 4.0, "fare_max_km": 200.0, "internet_cap": 1000.0},
    "ZM & SR. ZM": {"da_per_day": 285.0, "os_per_day": 3000.0, "fare_rate": 6.0, "fare_max_km": 200.0, "internet_cap": 2500.0},
    "NSM": {"da_per_day": 700.0, "os_per_day": 4500.0, "fare_rate": 9.0, "fare_max_km": 200.0, "internet_cap": math.inf},
}


@dataclass
class FileProcessResult:
    filename: str
    row: dict[str, Any] | None
    expense_heads: set[str]
    error: str | None = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_header(value: Any, fallback_index: int) -> str:
    cleaned = _clean_text(value)
    if not cleaned or cleaned.lower().startswith("unnamed"):
        return f"column_{fallback_index + 1}"
    return cleaned


def _canonical_column_key(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()


def _is_metadata_column(column_name: str) -> bool:
    key = _canonical_column_key(column_name)
    if key in METADATA_TOKENS:
        return True
    if any(pattern in key for pattern in EXCLUDED_HEAD_PATTERNS):
        return True
    return any(f" {token} " in f" {key} " for token in ("remark", "description", "total", "distance", "km"))


def _parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _find_date_column(columns: list[str]) -> str | None:
    for col in columns:
        if HEADER_DATE_PATTERN.search(_clean_text(col)):
            return col
    return None


def _find_km_column(columns: list[str]) -> str | None:
    for col in columns:
        key = _canonical_column_key(col)
        if " km " in f" {key} " or "distance" in key:
            return col
    return None


def _find_claim_column(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    for col in columns:
        key = _canonical_column_key(col)
        if all(word in key for word in keywords):
            return col
    return None


def _eligible_allowance_mask(txn_df: pd.DataFrame) -> pd.Series:
    """Exclude non-working rows (e.g., Sunday/Leave) from allowance-day eligibility."""
    if txn_df.empty:
        return pd.Series(False, index=txn_df.index)

    from_col = _find_claim_column(list(txn_df.columns), ("from",))
    to_col = _find_claim_column(list(txn_df.columns), ("to",))
    exclusion_tokens = ("sunday", "leave", "off", "holiday")

    mask = pd.Series(True, index=txn_df.index)
    for col in (from_col, to_col):
        if not col:
            continue
        text = txn_df[col].astype(str).str.lower().str.strip()
        row_excluded = text.apply(lambda x: any(token in x for token in exclusion_tokens))
        mask &= ~row_excluded

    return mask


def _is_summary_row(row: pd.Series) -> bool:
    for value in row.tolist():
        text = _clean_text(value).lower()
        if text and any(marker in text for marker in ("total", "grand total", "closing", "net", "overall")):
            return True
    return False


def _extract_month(raw_df: pd.DataFrame, header_row_idx: int) -> str | None:
    max_rows = min(max(header_row_idx, 1), 30)
    for i in range(max_rows):
        for value in raw_df.iloc[i].tolist():
            text = _clean_text(value)
            if MONTH_PATTERN.fullmatch(text):
                return text.upper().replace("/", "-").replace(" ", "-")
    return None


def _extract_employee_details(raw_df: pd.DataFrame, filename: str) -> tuple[str, str, str]:
    employee_name = ""
    employee_code = ""
    designation = "UNKNOWN"

    for i in range(min(len(raw_df), 25)):
        row_values = [_clean_text(v) for v in raw_df.iloc[i].tolist()]
        lowered = [v.lower() for v in row_values]
        for idx, cell in enumerate(lowered):
            if not employee_name and ("employee name" in cell or cell == "name"):
                employee_name = next((v for v in row_values[idx + 1 :] if v and v.lower() != "nan"), employee_name)
            if not employee_code and any(k in cell for k in ("employee code", "emp code", "employee id", "emp id", "code")):
                employee_code = next((v for v in row_values[idx + 1 :] if v and v.lower() != "nan"), employee_code)
            if designation == "UNKNOWN" and "designation" in cell:
                designation = next((v for v in row_values[idx + 1 :] if v and v.lower() != "nan"), designation)

    return employee_name or filename.rsplit(".", 1)[0], employee_code or "UNKNOWN", designation


def _row_has_date_header(values: list[Any]) -> bool:
    return any(HEADER_DATE_PATTERN.search(_clean_text(v)) for v in values if _clean_text(v))


def _find_header_row(df: pd.DataFrame) -> int:
    best_row, best_score = None, -1
    for idx in range(min(len(df), 120)):
        values = df.iloc[idx].tolist()
        if not _row_has_date_header(values):
            continue
        cleaned = [_clean_text(v) for v in values if _clean_text(v)]
        score = len(set(c.lower() for c in cleaned))
        if score > best_score:
            best_score, best_row = score, idx
    if best_row is None:
        raise ValueError("Could not detect header row containing DATE.")
    return best_row


def _choose_best_sheet(excel_file: pd.ExcelFile) -> tuple[str, pd.DataFrame, int]:
    candidates, failures = [], []
    for sheet_name in excel_file.sheet_names:
        raw_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, dtype=object)
        try:
            header_row = _find_header_row(raw_df)
            candidates.append((int(raw_df.iloc[header_row].notna().sum()), sheet_name, raw_df, header_row))
        except Exception as exc:
            failures.append(f"{sheet_name}: {exc}")
    if not candidates:
        raise ValueError("No valid sheet found. " + " | ".join(failures))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, sheet_name, raw_df, header_row = candidates[0]
    return sheet_name, raw_df, header_row


def _dedupe_headers(headers: list[str]) -> list[str]:
    seen, out = {}, []
    for header in headers:
        key = header.strip() or "column"
        seen[key] = seen.get(key, 0) + 1
        out.append(key if seen[key] == 1 else f"{key}__{seen[key]}")
    return out


def _normalize_designation(designation: str) -> str:
    raw = _canonical_column_key(designation)
    d = raw.replace(" ", "")

    # Normalize very common spelling mistakes/variants.
    d = (
        d.replace("maneger", "manager")
        .replace("manger", "manager")
        .replace("mng", "manager")
        .replace("sale", "sales")
    )

    tokens = [t for t in raw.split() if t]
    initials = "".join(token[0] for token in tokens if token)

    def has_initials(*targets: str) -> bool:
        return any(t.lower() in initials.lower() for t in targets)

    if "nsm" in d or "nationalsalesmanager" in d or has_initials("nsm"):
        return "NSM"

    # Dy ZM / Deputy Zonal Manager (+ typo support via initials)
    if "dyzm" in d or "deputyzonalmanager" in d or has_initials("dzm", "dyzm"):
        return "DY ZM"

    # ZM / Sr ZM / Zonal Manager / Senior Zonal Manager
    if any(token in d for token in ("zmsrzm", "seniorzonalmanager", "srzonalmanager")) or has_initials("szm"):
        return "ZM & SR. ZM"
    if ("zm" in d and "dy" not in d) or "zonalmanager" in d or has_initials("zm"):
        return "ZM & SR. ZM"

    # RM / Sr RM / Regional Manager / Senior Regional Manager
    if any(token in d for token in ("rmsrrm", "seniorregionalmanager", "srregionalmanager")) or has_initials("srm"):
        return "RM & SR. RM"
    if "regionalmanager" in d or ("rm" in d and "dy" not in d) or has_initials("rm"):
        return "RM & SR. RM"

    # BM / Sr BM / Business Manager / Senior Business Manager
    if any(token in d for token in ("bmsrbm", "seniorbusinessmanager", "srbusinessmanager")) or has_initials("sbm"):
        return "BM & SR. BM"
    if "businessmanager" in d or "bm" in d or has_initials("bm"):
        return "BM & SR. BM"

    return "UNKNOWN"


def _compute_policy_allowance(designation: str, txn_df: pd.DataFrame, numeric_candidates: list[str]) -> dict[str, float | str | None]:
    normalized = _normalize_designation(designation)
    policy = POLICY.get(normalized)
    if not policy:
        return {
            "__policy_designation": normalized,
            "__allowable_da": None,
            "__allowable_os": None,
            "__allowable_fare": None,
            "__allowable_internet": None,
        }

    eligible_mask = _eligible_allowance_mask(txn_df)

    da_col = _find_claim_column(numeric_candidates, ("da",))
    os_col = _find_claim_column(numeric_candidates, ("os", "bill"))
    fare_col = _find_claim_column(numeric_candidates, ("fare",))
    km_col = _find_km_column(list(txn_df.columns))

    allowable_da = None
    if da_col:
        da_series = _parse_numeric(txn_df[da_col]).fillna(0)
        da_days = int(((da_series > 0) & eligible_mask).sum())
        allowable_da = round(da_days * float(policy["da_per_day"]), 2)

    allowable_os = None
    if os_col:
        os_series = _parse_numeric(txn_df[os_col]).fillna(0)
        os_days = int(((os_series > 0) & eligible_mask).sum())
        allowable_os = round(os_days * float(policy["os_per_day"]), 2)

    allowable_fare = None
    if fare_col:
        if km_col and policy["fare_rate"] is not None and policy["fare_max_km"] is not None:
            # Sheet KM is one-way; policy should be applied for both-way allowance.
            km_series = _parse_numeric(txn_df[km_col]).fillna(0)
            both_way_km = km_series * 2
            capped_km = both_way_km.clip(upper=float(policy["fare_max_km"]) * 2)
            allowable_fare = round(float((capped_km[eligible_mask] * float(policy["fare_rate"])).sum()), 2)
        else:
            allowable_fare = None

    internet_cap = policy["internet_cap"]
    allowable_internet = "NO LIMIT" if internet_cap is not None and math.isinf(float(internet_cap)) else internet_cap

    return {
        "__policy_designation": normalized,
        "__allowable_da": allowable_da,
        "__allowable_os": allowable_os,
        "__allowable_fare": allowable_fare,
        "__allowable_internet": allowable_internet,
    }


def _open_excel_with_fallback(file_name: str, file_bytes: bytes) -> pd.ExcelFile:
    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

    has_calamine = (
        importlib.util.find_spec("python_calamine") is not None
        or importlib.util.find_spec("calamine") is not None
    )
    has_openpyxl = importlib.util.find_spec("openpyxl") is not None
    has_xlrd = importlib.util.find_spec("xlrd") is not None

    engine_candidates: list[str | None] = []
    if ext == "xlsx":
        if has_calamine:
            engine_candidates.append("calamine")
        if has_openpyxl:
            engine_candidates.append("openpyxl")
    elif ext == "xls":
        if has_calamine:
            engine_candidates.append("calamine")
        if has_xlrd:
            engine_candidates.append("xlrd")
    else:
        if has_calamine:
            engine_candidates.append("calamine")
        if has_openpyxl:
            engine_candidates.append("openpyxl")
        if has_xlrd:
            engine_candidates.append("xlrd")
        engine_candidates.append(None)

    # keep order but drop duplicates
    seen = set()
    deduped_candidates = []
    for eng in engine_candidates:
        if eng not in seen:
            deduped_candidates.append(eng)
            seen.add(eng)

    errors: list[str] = []
    for engine in deduped_candidates:
        try:
            return pd.ExcelFile(BytesIO(file_bytes), engine=engine)
        except Exception as exc:
            errors.append(f"{engine or 'auto'}: {exc}")

    if not deduped_candidates:
        raise ValueError(
            "No Excel reader engine is available. Install dependencies from requirements.txt "
            "(openpyxl/xlrd/python-calamine)."
        )

    raise ValueError("Unable to read workbook with available engines. " + " | ".join(errors))


def process_expense_file(file_name: str, file_bytes: bytes) -> FileProcessResult:
    try:
        if not file_bytes:
            return FileProcessResult(file_name, None, set(), "Empty file.")

        excel = _open_excel_with_fallback(file_name, file_bytes)
        _, raw_df, header_row = _choose_best_sheet(excel)
        header_values = _dedupe_headers([_normalize_header(v, idx) for idx, v in enumerate(raw_df.iloc[header_row].tolist())])

        data_df = raw_df.iloc[header_row + 1 :].copy()
        data_df.columns = header_values
        data_df = data_df.dropna(how="all")
        if data_df.empty:
            return FileProcessResult(file_name, None, set(), "No transactional rows after header.")

        employee_name, employee_code, designation = _extract_employee_details(raw_df, file_name)
        month = _extract_month(raw_df, header_row) or "UNKNOWN"

        metadata_cols = [col for col in data_df.columns if _is_metadata_column(col)]
        numeric_candidates = [col for col in data_df.columns if col not in metadata_cols]

        keyword_summary_mask = data_df.apply(_is_summary_row, axis=1)
        date_col = _find_date_column(list(data_df.columns))
        parsed_dates = None
        likely_footer_mask = pd.Series(False, index=data_df.index)
        if date_col is not None:
            parsed_dates = pd.to_datetime(data_df[date_col], errors="coerce", dayfirst=True)
            numeric_cells_per_row = pd.Series(0, index=data_df.index, dtype="int64")
            for col in numeric_candidates:
                numeric_cells_per_row += _parse_numeric(data_df[col]).notna().astype(int)
            likely_footer_mask = parsed_dates.isna() & (numeric_cells_per_row >= 2)

        summary_mask = keyword_summary_mask | likely_footer_mask
        footer_df = data_df.loc[summary_mask].copy()
        txn_df = data_df.loc[~summary_mask].copy()
        if date_col is not None and parsed_dates is not None:
            txn_df = txn_df.loc[parsed_dates.loc[txn_df.index].notna()].copy()

        if txn_df.empty and footer_df.empty:
            return FileProcessResult(file_name, None, set(), "No valid transactional rows after filtering totals/footer rows.")

        expense_totals: dict[str, float] = {}
        for col in numeric_candidates:
            footer_numeric = _parse_numeric(footer_df[col]).dropna() if not footer_df.empty else pd.Series(dtype="float64")
            if not footer_numeric.empty:
                total = float(footer_numeric.iloc[-1])
            else:
                numeric = _parse_numeric(txn_df[col])
                if int(numeric.notna().sum()) == 0:
                    continue
                total = float(numeric.fillna(0).sum())
            if abs(total) < 1e-9:
                continue
            expense_totals[col] = round(total, 2)

        if not expense_totals:
            return FileProcessResult(file_name, None, set(), "No dynamic expense columns detected.")

        row = {
            "Employee Name": employee_name,
            "Employee Code": employee_code,
            "Designation": designation,
            "Month": month,
            **expense_totals,
            **_compute_policy_allowance(designation, txn_df, numeric_candidates),
        }
        return FileProcessResult(file_name, row, set(expense_totals), None)
    except Exception as exc:
        return FileProcessResult(file_name, None, set(), str(exc))


def build_consolidated_dataframe(rows: list[dict[str, Any]], expense_heads: set[str]) -> pd.DataFrame:
    ordered_heads = sorted(expense_heads)
    ordered_columns = ["Employee Name", "Employee Code", "Designation", "Month", *ordered_heads]

    if not rows:
        return pd.DataFrame(columns=[*ordered_columns, "Grand Total"])

    normalized_rows = [{col: row.get(col, 0 if col in ordered_heads else "") for col in ordered_columns} for row in rows]
    consolidated = pd.DataFrame(normalized_rows, columns=ordered_columns)
    if ordered_heads:
        consolidated[ordered_heads] = consolidated[ordered_heads].apply(pd.to_numeric, errors="coerce").fillna(0)

    dedupe_cols = ["Employee Name", "Employee Code", "Designation", "Month", *ordered_heads]
    consolidated = consolidated.drop_duplicates(subset=dedupe_cols, keep="first")

    group_cols = ["Employee Name", "Employee Code", "Designation", "Month"]
    consolidated = consolidated.groupby(group_cols, as_index=False).agg({head: "sum" for head in ordered_heads})
    consolidated["Grand Total"] = consolidated[ordered_heads].sum(axis=1).round(2) if ordered_heads else 0.0
    return consolidated[["Employee Name", "Employee Code", "Designation", "Month", *ordered_heads, "Grand Total"]]


def build_policy_comparison_dataframe(rows: list[dict[str, Any]], consolidated_df: pd.DataFrame) -> pd.DataFrame:
    if consolidated_df.empty:
        return pd.DataFrame(
            columns=[
                "Employee Name", "Employee Code", "Designation", "Month",
                "Claimed DA", "Allowed DA", "DA Status",
                "Claimed OS", "Allowed OS", "OS Status",
                "Claimed Fare", "Allowed Fare", "Fare Status",
                "Claimed Internet", "Allowed Internet", "Internet Status",
            ]
        )

    base_cols = ["Employee Name", "Employee Code", "Designation", "Month"]
    policy_rows = []
    for row in rows:
        policy_rows.append({
            "Employee Name": row.get("Employee Name", ""),
            "Employee Code": row.get("Employee Code", ""),
            "Designation": row.get("Designation", ""),
            "Month": row.get("Month", ""),
            "Allowed DA": row.get("__allowable_da"),
            "Allowed OS": row.get("__allowable_os"),
            "Allowed Fare": row.get("__allowable_fare"),
            "Allowed Internet": row.get("__allowable_internet"),
        })
    allowance_df = pd.DataFrame(policy_rows)
    if not allowance_df.empty:
        allowance_df = allowance_df.groupby(base_cols, as_index=False).agg({
            "Allowed DA": "sum",
            "Allowed OS": "sum",
            "Allowed Fare": "sum",
            "Allowed Internet": lambda s: "NO LIMIT" if (s == "NO LIMIT").any() else pd.to_numeric(s, errors="coerce").fillna(0).sum(),
        })

    cols = consolidated_df.columns.tolist()
    da_col = _find_claim_column(cols, ("da",))
    os_col = _find_claim_column(cols, ("os", "bill"))
    fare_col = _find_claim_column(cols, ("fare",))
    internet_col = _find_claim_column(cols, ("internet",))

    report = consolidated_df[base_cols].copy()
    report["Claimed DA"] = consolidated_df[da_col] if da_col else 0
    report["Claimed OS"] = consolidated_df[os_col] if os_col else 0
    report["Claimed Fare"] = consolidated_df[fare_col] if fare_col else 0
    report["Claimed Internet"] = consolidated_df[internet_col] if internet_col else 0

    report = report.merge(allowance_df, on=base_cols, how="left")

    def status(claimed: float, allowed: Any) -> str:
        if isinstance(allowed, str) and allowed.upper() == "NO LIMIT":
            return "Within Policy"
        if pd.isna(allowed):
            return "Policy Missing"
        return "Exceeds" if float(claimed) > float(allowed) else "Within Policy"

    report["DA Status"] = report.apply(lambda r: status(r["Claimed DA"], r["Allowed DA"]), axis=1)
    report["OS Status"] = report.apply(lambda r: status(r["Claimed OS"], r["Allowed OS"]), axis=1)
    report["Fare Status"] = report.apply(lambda r: status(r["Claimed Fare"], r["Allowed Fare"]), axis=1)
    report["Internet Status"] = report.apply(lambda r: status(r["Claimed Internet"], r["Allowed Internet"]), axis=1)

    return report[
        [
            "Employee Name", "Employee Code", "Designation", "Month",
            "Claimed DA", "Allowed DA", "DA Status",
            "Claimed OS", "Allowed OS", "OS Status",
            "Claimed Fare", "Allowed Fare", "Fare Status",
            "Claimed Internet", "Allowed Internet", "Internet Status",
        ]
    ]
