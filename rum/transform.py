import pprint
import re
import streamlit as st
from datetime import datetime, timedelta
from dateutil import tz
from typing import Dict, Any, List
import pandas as pd
from collections import defaultdict

# ─────────────────────────────────────────
# 데이터 변환 함수
# ─────────────────────────────────────────


def iso_to_kst_ms(iso_str: str, tz_name: str = "Asia/Seoul") -> str:
    """ISO 8601 형식의 시간 문자열을 KST 시간(ms 단위 포함)으로 변환합니다."""
    if not iso_str:
        return ""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    kst = tz.gettz(tz_name)
    k = dt.astimezone(kst)
    return k.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(k.strftime('%f'))//1000:03d} KST"


def flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    """중첩된 딕셔너리나 리스트를 평탄화하여 단일 딕셔너리로 만듭니다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        out[prefix] = ", ".join(map(str, obj))
    else:
        out[prefix] = obj


def process_rum_events(all_events: List[Dict[str, Any]], tz_name="Asia/Seoul") -> List[Dict[str, Any]]:
    """
    RUM 이벤트 목록을 평탄화된 행(딕셔너리)의 목록으로 변환합니다.
    - 중첩된 속성을 'a.b.c' 형태로 평탄화합니다.
    - 타임스탬프를 KST로 변환합니다.
    - 여러 형태의 Call ID를 단일 필드('Call ID')로 정규화합니다.
    """
    processed_rows: List[Dict[str, Any]] = []
    for event in all_events:
        attrs = event.get("attributes", {}) or {}
        flat_row: Dict[str, Any] = {}
        flatten("", attrs, flat_row)

        # 'usr' 객체 평탄화 추가
        usr_info = event.get("usr")
        if usr_info:
            flatten("usr", usr_info, flat_row)

        # Add tags
        tags = event.get("tags")
        if tags:
            flat_row["tags"] = tags

        flat_row["timestamp(KST)"] = iso_to_kst_ms(attrs.get("timestamp"), tz_name)

        call_id_val = (
            flat_row.get("attributes.context.callID")
            or flat_row.get("attributes.context.callId")
            # or flat_row.get("attributes.context.CallIDs")
        )

        if call_id_val is not None:
            flat_row["Call ID"] = call_id_val
            flat_row.pop("attributes.context.callID", None)
            flat_row.pop("attributes.context.callId", None)
            # flat_row.pop("attributes.context.CallIDs", None)

        processed_rows.append(flat_row)
    return processed_rows


def summarize_calls(flat_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    RUM 이벤트를 Call ID별로 그룹화하고 통화 정보를 요약합니다.
    - 종료 사유, 패킷 정보, 주요 이벤트 상태 코드를 추출하여 요약 테이블을 생성합니다.
    """
    calls = defaultdict(list)
    for row in flat_rows:
        call_id = row.get("Call ID")
        if call_id:
            calls[call_id].append(row)

    if not calls:
        return pd.DataFrame()

    summaries = []
    for call_id, events in calls.items():
        termination_reason = None
        bye_reason = None
        send_packets = []
        receive_health_check = []
        request_status, accept_status, reject_status, end_status = None, None, None, None
        media_svr_ip, media_svr_port = None, None
        active_ts, stopping_ts = None, None
        local_addr, local_port = None, None

        overall_end_time_str = events[0].get("timestamp(KST)")
        overall_start_time_str = events[-1].get("timestamp(KST)")

        for event in events:
            path = event.get("attributes.resource.url_path")
            status_code = event.get("attributes.resource.status_code")
            timestamp_str = event.get("timestamp(KST)")

            if event.get("attributes.context.method") == "BYE":
                bye_reason = event.get("attributes.context.reason")

            if path == "/res/SDK_CALL_STATUS_ACTIVE" and active_ts is None:
                active_ts = timestamp_str
            elif path == "/res/SDK_CALL_STATUS_STOPPING":
                if stopping_ts is None:
                    stopping_ts = timestamp_str
                if termination_reason is None:
                    event_type = event.get("attributes.context.eventType")
                    event_detail = event.get("attributes.context.eventDetail")
                    
                    parts = []
                    if event_type:
                        parts.append(str(event_type))
                    if event_detail:
                        parts.append(f"({event_detail})")

                    if parts:
                        termination_reason = " ".join(parts)

            elif path == "/res/ENGINE_startCall":
                if media_svr_ip is None:
                    media_svr_ip = event.get("attributes.context.mediaSvrIP")
                if media_svr_port is None:
                    media_svr_port = event.get("attributes.context.mediaSvrPort")
            elif path == "/res/ENGINE_getLocalAddrInfo":
                if local_addr is None:
                    local_addr = event.get("attributes.context.local_addr")
                if local_port is None:
                    local_port = event.get("attributes.context.local_port")
            elif path == "/res/requestVoiceCall" and request_status is None:
                request_status = status_code
            elif path == "/res/acceptCall" and accept_status is None:
                accept_status = status_code
            elif path == "/res/rejectCall" and reject_status is None:
                reject_status = status_code
            elif path == "/res/endCall" and end_status is None:
                end_status = status_code
            elif path == "/res/ENGINE_SendPackets" and len(send_packets) < 3:
                count = event.get("attributes.context.totalCount")
                if count is not None:
                    send_packets.append(count)
            elif path == "/res/ENGINE_ReceiveHealthCheck" and len(receive_health_check) < 3:
                count = event.get("attributes.context.totalCount")
                if count is not None:
                    receive_health_check.append(count)
        
        duration_str = ""
        if stopping_ts is None:
            duration_str = "STOPPING 없음"
        elif active_ts is None:
            duration_str = "ACTIVE 없음"
        else:
            try:
                stop_dt = datetime.strptime(stopping_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                active_dt = datetime.strptime(active_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                duration_seconds = (stop_dt - active_dt).total_seconds()
                duration_str = f"{duration_seconds:.1f} 초"
            except (ValueError, TypeError):
                duration_str = "시간 포맷 오류"

        summaries.append({
            "Call ID": call_id,
            "Start Time (KST)": overall_start_time_str,
            "End Time (KST)": overall_end_time_str,
            "Duration": duration_str,
            "종료 사유": termination_reason,
            "BYE reason": bye_reason,
            "requestVoiceCall_status_code": request_status,
            "acceptCall_status_code": accept_status,
            "rejectCall_status_code": reject_status,
            "endCall_status_code": end_status,
            "SendPackets 수": send_packets,
            "ReceiveHealthCheck 수": receive_health_check,
            "MediaSvr IP": media_svr_ip,
            "MediaSvr Port": media_svr_port,
            "Local Addr": local_addr,
            "Local Port": local_port,
        })

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty and "Start Time (KST)" in summary_df.columns:
        summary_df = summary_df.sort_values("Start Time (KST)", ascending=False).reset_index(drop=True)

    return summary_df


def to_base_dataframe(flat_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """평탄화된 행 목록으로부터 DataFrame을 생성하고, 데이터 타입을 변환한 후 시간순으로 정렬합니다."""
    df = pd.DataFrame(flat_rows)

    for col in ["attributes.resource.status_code", "attributes.resource.duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    if "timestamp(KST)" in df.columns:
        parsed_ts = pd.to_datetime(
            df["timestamp(KST)"].str.replace(" KST", "", regex=False),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce"
        )
        df = df.assign(_ts=parsed_ts).sort_values("_ts", ascending=False).drop(columns=["_ts"])
    return df


def apply_view_filters(
    df_view: pd.DataFrame,
    auto_hide_sparse: bool = False,
    sparse_threshold: int = 5,
    hidden_cols: List[str] = None,
) -> pd.DataFrame:
    """
    데이터프레임에 뷰 관련 필터(희소 열 숨김, 사용자 지정 숨김)를 적용합니다.
    - auto_hide_sparse: True일 경우, 값이 거의 없는 열(희소 열)을 자동으로 숨깁니다. 기본값은 False입니다.
    """
    hidden_cols = hidden_cols or []
    if auto_hide_sparse and sparse_threshold > 0 and not df_view.empty:
        non_empty_ratio = (df_view.notna() & (df_view != "")).mean(numeric_only=False)
        keep_cols_sparse = [
            c for c in df_view.columns
            if (non_empty_ratio.get(c, 0) * 100) >= sparse_threshold or c == "timestamp(KST)"
        ]
        df_view = df_view[keep_cols_sparse]

    drops = [c for c in hidden_cols if c in df_view.columns and c != "timestamp(KST)"]
    if drops:
        df_view = df_view.drop(columns=drops, errors="ignore")

    return df_view


def analyze_rtp_timeouts(flat_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    RTP Timeout이 발생한 통화를 분석합니다.
    - 'rtp' 또는 'RTP'가 포함된 reason을 가진 이벤트를 기반으로 Call ID를 필터링합니다.
    - 통화 유지 시간 (ACTIVE ~ STOPPING)을 계산합니다.
    - BYE 메시지의 출처(longRes, restReq 등)를 분석합니다.
    """
    rtp_timeout_call_ids = set()
    for row in flat_rows:
        reason = row.get("attributes.context.reason", "")
        if isinstance(reason, str) and "rtp" in reason.lower():
            call_id = row.get("Call ID")
            if call_id:
                rtp_timeout_call_ids.add(call_id)

    if not rtp_timeout_call_ids:
        return pd.DataFrame()

    calls = defaultdict(list)
    for row in flat_rows:
        call_id = row.get("Call ID")
        if call_id in rtp_timeout_call_ids:
            calls[call_id].append(row)

    summaries = []
    for call_id, events in calls.items():
        # events are assumed to be sorted descending by timestamp from the dataframe
        active_ts, stopping_ts = None, None
        bye_method_source = "N/A"
        rtp_timeout_reason = "N/A"
        send_packets = []
        receive_health_check = []
        media_svr_ip, media_svr_port = None, None
        local_addr, local_port = None, None

        audio_session_active_500_count = 0
        audio_session_deactive_count = 0
        audio_session_interrupt_count = 0
        audio_session_routed_count = 0

        has_firstTx = ""
        
        usr_id = "N/A"
        first_version = "N/A"

        if events:
            # Extract first_version from tags. Tags should be consistent across events for a call.
            tags_str = events[0].get("tags", "")
            if tags_str:
                match = re.search(r"first_version:([^,]+)", tags_str)
                if match:
                    first_version = match.group(1).strip()
            
            for event in events:
                if event.get("attributes.usr.id"):
                    usr_id = event.get("attributes.usr.id")
                    break

        overall_end_time_str = events[0].get("timestamp(KST)")
        overall_start_time_str = events[-1].get("timestamp(KST)")

        for event in events:
            path = event.get("attributes.resource.url_path")
            timestamp_str = event.get("timestamp(KST)")
            reason = event.get("attributes.context.reason", "")
            status_code = event.get("attributes.resource.status_code")

            if path == "/res/audioSessionActive" and status_code == 500 :
                audio_session_active_500_count += 1

            if path == "/res/audioSessionDeactive" :
                audio_session_deactive_count += 1

            if path == "/res/audioSessionInterrupt" :
                audio_session_interrupt_count += 1

            if path == "/res/routedAudioSession" :
                audio_session_routed_count += 1

            if isinstance(reason, str) and "rtp" in reason.lower() and rtp_timeout_reason == "N/A":
                rtp_timeout_reason = reason

            # 통화 시간을 계산하는 로직
            # CALL_STATUS_ACTIVE - CALL_STATUS_STOPPING
            if path == "/res/SDK_CALL_STATUS_ACTIVE" and active_ts is None:
                active_ts = timestamp_str
            elif path == "/res/SDK_CALL_STATUS_STOPPING" and stopping_ts is None:
                stopping_ts = timestamp_str

            # Media Server(미디어 서버) IP, Port 추출
            if path == "/res/ENGINE_startCall":
                if media_svr_ip is None:
                    media_svr_ip = event.get("attributes.context.mediaSvrIP")
                if media_svr_port is None:
                    media_svr_port = event.get("attributes.context.mediaSvrPort")

            # 단말의 IP, Port 추출
            if path == "/res/ENGINE_getLocalAddrInfo":
                if local_addr is None:
                    local_addr = event.get("attributes.context.local_addr")
                if local_port is None:
                    local_port = event.get("attributes.context.local_port")

            # firstTx 추출
            if path == "/res/ENGINE_firstTx":
                has_firstTx = "있음"

            # BYE message 송수신 분석
            if event.get("attributes.context.method") == "BYE":
                # This is a guess based on user's request.
                # We assume the source information is in the url_path.
                url_path = event.get("attributes.resource.url_path", "").lower()
                if "longres" in url_path:
                    bye_method_source = "longRes"
                elif "restreq" in url_path:
                    bye_method_source = "restReq"
                elif "sendmessage" in url_path:
                    bye_method_source = "sendMessage"
                elif "recvmessage" in url_path:
                    bye_method_source = "recvMessage"
                else:
                    bye_method_source = "Unknown"
            
            elif path == "/res/ENGINE_SendPackets" and len(send_packets) < 3:
                count = event.get("attributes.context.totalCount")
                if count is not None:
                    send_packets.append(count)
            elif path == "/res/ENGINE_ReceiveHealthCheck" and len(receive_health_check) < 3:
                count = event.get("attributes.context.totalCount")
                if count is not None:
                    receive_health_check.append(count)

        duration_str = ""
        if stopping_ts and active_ts:
            try:
                stop_dt = datetime.strptime(stopping_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                active_dt = datetime.strptime(active_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                duration_seconds = (stop_dt - active_dt).total_seconds()
                duration_str = f"{duration_seconds:.1f}"
            except (ValueError, TypeError):
                duration_str = "시간 포맷 오류"
        elif not active_ts:
            duration_str = "ACTIVE 없음"
        elif not stopping_ts:
            duration_str = "STOPPING 없음"

        summaries.append({
            "Call ID": call_id,
            "App Version": first_version,
            "Start Time (KST)": overall_start_time_str,
            "End Time (KST)": overall_end_time_str,
            "통화 시간(초)": duration_str,
            "BYE Reason": rtp_timeout_reason,
            "BYE 전달": bye_method_source,
            "firstTx 유무" : has_firstTx,
            "SendPackets 수": send_packets,
            "ReceiveHealthCheck 수": receive_health_check,
            "usr.id": usr_id,
            "MediaSvr IP": media_svr_ip,
            "MediaSvr Port": media_svr_port,
            "Local Addr": local_addr,
            "Local Port": local_port,
            # "audioSessionActive(500)": audio_session_active_500_count,
            # "audioSessionDeactive": audio_session_deactive_count,
            # "audioSessionInterrupt": audio_session_interrupt_count,
            # "routedAudioSession" : audio_seesion_routed_count,
        })

    if not summaries:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Start Time (KST)", ascending=False).reset_index(drop=True)

    return summary_df


def categorize_rtp_timeouts(summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    RTP Timeout 요약 데이터프레임을 분석하여 차트용 데이터를 생성합니다.
    - 사용자별 발생 건수
    - 앱 버전별 발생 건수
    - 통화 시간대별 발생 건수
    """
    if summary_df.empty:
        return {
            "by_user": pd.DataFrame(),
            "by_version": pd.DataFrame(),
            "by_duration": pd.DataFrame(),
        }

    # 1. 유저 분석
    if "usr.id" in summary_df.columns:
        user_counts = summary_df["usr.id"].value_counts().head(10).reset_index()
        user_counts.columns = ["User ID", "Count"]
    else:
        user_counts = pd.DataFrame(columns=["User ID", "Count"])

    # 2. App Version 분석
    if "App Version" in summary_df.columns:
        version_counts = summary_df["App Version"].value_counts().reset_index()
        version_counts.columns = ["App Version", "Count"]
    else:
        version_counts = pd.DataFrame(columns=["App Version", "Count"])

    # 3. 통화 시간 분석
    duration_categories = []
    if "통화 시간(초)" in summary_df.columns:
        for duration in summary_df["통화 시간(초)"]:
            try:
                duration_sec = float(duration)
                if 0 <= duration_sec <= 11:
                    duration_categories.append("0-11초")
                elif 12 <= duration_sec <= 16:
                    duration_categories.append("12-16초")
                else:
                    duration_categories.append("17초 이상")
            except (ValueError, TypeError):
                duration_categories.append("알 수 없음")

    duration_counts = pd.Series(duration_categories).value_counts().reindex(["0-11초", "12-16초", "17초 이상", "알 수 없음"], fill_value=0).reset_index()
    duration_counts.columns = ["Duration Category", "Count"]

    return {
        "by_user": user_counts,
        "by_version": version_counts,
        "by_duration": duration_counts,
    }


def filter_dataframe(df: pd.DataFrame, column: str, filter_text: str, is_and: bool) -> pd.DataFrame:
    """
    주어진 조건에 따라 데이터프레임을 필터링합니다.
    """
    if column not in df.columns:
        st.warning(f"'{column}' 컬럼이 없어 필터링할 수 없습니다.")
        return df

    keywords = [kw.strip() for kw in filter_text.split(',') if kw.strip()]
    if not keywords:
        return df

    series = df[column].fillna('')
    
    if is_and:
        condition = pd.Series(True, index=df.index)
        for kw in keywords:
            condition &= series.str.contains(re.escape(kw), case=False, regex=True)
    else:
        regex_pattern = '|'.join(re.escape(kw) for kw in keywords)
        condition = series.str.contains(regex_pattern, case=False, regex=True)
    
    return df[condition]
