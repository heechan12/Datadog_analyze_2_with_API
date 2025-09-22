import streamlit as st
from datetime import datetime, timedelta
import pytz
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# RUM modules
from rum.api_client import DatadogAPIClient
from rum.config import get_settings, get_default_hidden_columns
from rum.datadog_api import search_rum_events
from rum.transform import process_rum_events, to_base_dataframe, apply_view_filters, summarize_calls, analyze_rtp_timeouts
from rum.ui import render_sidebar, render_main_view, effective_hidden, sanitize_pin_slots

# TODO 1. Log 분석해서 RUM 데이터와 결합하여 분석하기 -> callId 기반으로 검색해서 데이터를 얻을 수 있을지 검토 필요
# TODO 2. 대시보드 데이터 보여주기
# TODO 3. Fast API + cron -> 시간 별로 RTP timeout 분석 자동화

# ─────────────────────────────────────────
# Constants & Settings
# ─────────────────────────────────────────
FIXED_PIN = "attributes.resource.url_path"  # 테이블에서 항상 고정할 열 이름
PIN_COUNT = 10  # 고정할 수 있는 최대 열 개수

# ─────────────────────────────────────────
# Session State & Data Processing
# ─────────────────────────────────────────
def initialize_session_state():
    """Streamlit의 세션 상태를 초기화합니다."""
    ss = st.session_state
    # 기본값 설정
    defaults = {
        "df_base": None,                                # 원본 데이터프레임
        "df_view": None,                                # 필터링 및 정렬된 뷰 데이터프레임
        "df_summary": None,                             # 통화 요약 데이터프레임
        "df_rtp_summary": None,                         # RTP Timeout 분석 결과 데이터프레임
        "hide_defaults": get_default_hidden_columns(),  # 기본적으로 숨길 열 목록
        "hidden_cols_user": [],                         # 사용자가 선택한 숨길 열 목록
        "table_height": 900,                            # 테이블 높이
        "pin_slots": [""] * PIN_COUNT,                  # 고정 열 슬롯
        "unique_call_ids": [],                          # 고유 통화 ID 목록
        "custom_query": "",                             # Custom Query 저장
        "analysis_type": "User ID 분석",                 # 현재 분석 유형
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value

    # UI 상태 초기화 (선택 대기 값)
    if "pending_hidden_cols_user" not in ss:
        ss.pending_hidden_cols_user = ss.hidden_cols_user.copy()
    if "pending_pin_slots" not in ss:
        ss.pending_pin_slots = ss.pin_slots.copy()

    # 시간 범위 초기화 (KST 기준)
    kst = pytz.timezone("Asia/Seoul")
    if "start_dt" not in ss:
        ss.start_dt = datetime.now(kst) - timedelta(minutes=10)
    if "end_dt" not in ss:
        ss.end_dt = datetime.now(kst)


# ─────────────────────────────────────────
# User ID 기반 RUM 데이터 검색
# ─────────────────────────────────────────
def handle_user_id_based_rum_search(client: DatadogAPIClient, params: dict):
    """User ID 기반 RUM 데이터 검색"""
    ss = st.session_state
    ss.df_rtp_summary = None # 다른 분석 결과 초기화

    usr_id_value = params.pop("usr_id_value", None)
    params.pop("analysis_type", None)
    params.pop("custom_query", None)  # custom_query는 이 분석에서 사용하지 않으므로 제거

    if not usr_id_value:
        query = "*"
    else:
        safe_usr_id = usr_id_value.replace('"', '\"')
        query = f'@usr.id:"{safe_usr_id}"'  # usr.id 쿼리 수행

    with st.spinner("검색 중..."):
        raw_events = search_rum_events(client=client, query=query, **params)
    st.success(f"가져온 이벤트: {len(raw_events)}건")

    if not raw_events:
        ss.df_base = ss.df_view = ss.df_summary = None
        ss.unique_call_ids = []
        st.info("검색 결과가 없습니다.")
        return

    with st.spinner("이벤트 데이터 변환 중..."):
        flat_rows = process_rum_events(raw_events, tz_name="Asia/Seoul")
    
    with st.spinner("통화 정보 요약 중..."):
        ss.df_summary = summarize_calls(flat_rows)

    ss.df_base = to_base_dataframe(flat_rows)
    all_cols = [c for c in ss.df_base.columns if c != "timestamp(KST)"]
    
    ss.hidden_cols_user = [c for c in ss.hidden_cols_user if c in all_cols]
    ss.pending_hidden_cols_user = ss.hidden_cols_user.copy()
    
    visible_cols = [c for c in all_cols if c not in effective_hidden(all_cols, ss.pending_hidden_cols_user, ss.hide_defaults, FIXED_PIN)]
    ss.pin_slots = sanitize_pin_slots(ss.pin_slots, visible_cols, PIN_COUNT, FIXED_PIN)
    ss.pending_pin_slots = ss.pin_slots.copy()

    eff_hidden_applied = effective_hidden(all_cols, ss.hidden_cols_user, ss.hide_defaults, FIXED_PIN)
    ss.df_view = apply_view_filters(ss.df_base.copy(), hidden_cols=eff_hidden_applied)

    if "Call ID" in ss.df_base.columns:
        ss.unique_call_ids = sorted(ss.df_base["Call ID"].dropna().unique().tolist())
    else:
        ss.unique_call_ids = []


# ─────────────────────────────────────────
# Custom Query 페이지
# ─────────────────────────────────────────
def handle_custom_query_rum_search(client: DatadogAPIClient, params: dict):
    """Custom query 검색을 실행하고 결과를 처리하여 세션 상태에 저장"""
    ss = st.session_state
    ss.df_summary = None
    ss.df_rtp_summary = None

    query = params.pop("custom_query", "*")
    params.pop("analysis_type", None)
    params.pop("usr_id_value", None)

    if not query.strip():
        query = "*"

    with st.spinner(f"검색 중... (query: {query})"):
        raw_events = search_rum_events(client=client, query=query, **params)
    st.success(f"가져온 이벤트: {len(raw_events)}건")

    if not raw_events:
        ss.df_base = ss.df_view = ss.df_summary = None
        ss.unique_call_ids = []
        st.info("검색 결과가 없습니다.")
        return

    with st.spinner("이벤트 데이터 변환 중..."):
        flat_rows = process_rum_events(raw_events, tz_name="Asia/Seoul")

    ss.df_base = to_base_dataframe(flat_rows)
    all_cols = [c for c in ss.df_base.columns if c != "timestamp(KST)"]

    ss.hidden_cols_user = [c for c in ss.hidden_cols_user if c in all_cols]
    ss.pending_hidden_cols_user = ss.hidden_cols_user.copy()

    visible_cols = [c for c in all_cols if c not in effective_hidden(all_cols, ss.pending_hidden_cols_user, ss.hide_defaults, FIXED_PIN)]
    ss.pin_slots = sanitize_pin_slots(ss.pin_slots, visible_cols, PIN_COUNT, FIXED_PIN)
    ss.pending_pin_slots = ss.pin_slots.copy()

    eff_hidden_applied = effective_hidden(all_cols, ss.hidden_cols_user, ss.hide_defaults, FIXED_PIN)
    ss.df_view = apply_view_filters(ss.df_base.copy(), hidden_cols=eff_hidden_applied)


# ─────────────────────────────────────────
# RTP Timeout 검색 및 2차 분석
# ─────────────────────────────────────────
def handle_rum_based_rtp_analysis(client: DatadogAPIClient, params: dict):
    """RTP Timeout 통화에 대한 2단계 분석을 수행"""
    ss = st.session_state
    ss.df_summary = None # 통화 요약은 사용하지 않으므로 초기화

    api_params = params.copy()
    usr_id_value = api_params.pop("usr_id_value", None)
    version_value = api_params.pop("version_value", None)
    build_version_value = api_params.pop("build_version_value", None)
    api_params.pop("analysis_type", None)
    api_params.pop("custom_query", None) # custom_query 파라미터 제거

    # 1단계: RTP Timeout이 발생한 Call ID 수집
    query_parts = ["@context.reason:(*RTP* OR *rtp*)"]

    def build_or_query_part(field, value_str):
        values = [v.strip() for v in value_str.split(',') if v.strip()]
        if not values:
            return None
        # Python 3.12+ f-string 제약으로 인해 SyntaxError를 피하기 위해 문자열을 직접 조합합니다.
        # f-string의 {} 표현식 안에는 '\'를 사용할 수 없습니다.
        # 예: @usr.id:"user\"1"
        escaped_values = [v.replace('"', '\\"') for v in values]
        query_parts = [f'{field}:{val}' for val in escaped_values]
        return f'({" OR ".join(query_parts)})'

    if usr_id_value:
        query_parts.append(build_or_query_part("@usr.id", usr_id_value))
    if version_value:
        query_parts.append(build_or_query_part("version", version_value))
    if build_version_value:
        query_parts.append(build_or_query_part("@build_version", build_version_value))

    final_query = " AND ".join(p for p in query_parts if p)

    with st.spinner(f"1/2: RTP Timeout 이벤트 검색 중... (query: {final_query})"):
        rtp_timeout_events = search_rum_events(client=client, query=final_query, **api_params) # 이벤트 검색
    
    if not rtp_timeout_events:
        st.info("해당 기간에 RTP Timeout으로 기록된 통화가 없습니다.")
        ss.df_rtp_summary = pd.DataFrame()
        ss.df_base = ss.df_view = None
        return

    # process_rum_events를 사용하여 Call ID를 통합
    flat_rtp_rows = process_rum_events(rtp_timeout_events, tz_name="Asia/Seoul")
    
    call_ids = set()
    for row in flat_rtp_rows:
        call_id = row.get("Call ID")
        if call_id:
            call_ids.add(call_id)
    
    st.toast(f"1/2: {len(call_ids)}개의 RTP Timeout 통화 ID를 찾았습니다.")

    if not call_ids:
        st.info("RTP Timeout 이벤트에서 Call ID를 찾을 수 없었습니다.")
        ss.df_rtp_summary = pd.DataFrame()
        ss.df_base = ss.df_view = None
        return

    # 2단계: 수집된 Call ID로 전체 이벤트 검색
    all_raw_events = []
    call_id_list = list(call_ids)
    batch_size = 50  # 쿼리 길이 제한을 피하기 위해 ID를 50개씩 나누어 처리

    with st.spinner(f"2/2: {len(call_ids)}개 통화의 전체 이벤트 병렬 검색 중..."):
        batches = [call_id_list[i:i+batch_size] for i in range(0, len(call_id_list), batch_size)]
        
        # Use a managed ThreadPoolExecutor to prevent conflicts with pyarrow's internal pool
        # during interpreter shutdown phases.
        executor = ThreadPoolExecutor()
        try:
            # Create API call tasks for each batch.
            future_to_batch = {
                executor.submit(
                    search_rum_events,
                    client=client,
                    query=f'(@context.callID:({" OR ".join(f"{cid}" for cid in batch)}) OR @context.callId:({" OR ".join(f"{cid}" for cid in batch)}))',
                    **api_params
                ): batch for batch in batches
            }
            # Collect results as they complete.
            failed_batches = 0
            for future in as_completed(future_to_batch):
                try:
                    all_raw_events.extend(future.result())
                except Exception as e:
                    failed_batches += 1
                    st.error(f"이벤트 검색 중 오류 발생: {e}")
            if failed_batches > 0:
                st.warning(f"{failed_batches}개 배치 검색에 실패했습니다. 결과가 일부 누락될 수 있습니다.")
        finally:
            executor.shutdown(wait=True) # Ensure the executor is shut down before proceeding.

    raw_events = all_raw_events # 이미 extend로 추가되었으므로 이 줄은 사실상 all_raw_events를 가리킴
    st.toast(f"2/2: 총 {len(raw_events)}개의 관련 이벤트를 가져왔습니다.")

    if not raw_events:
        st.warning("RTP Timeout 통화 ID로 이벤트를 조회했으나, 결과를 가져오지 못했습니다.")
        ss.df_rtp_summary = pd.DataFrame()
        ss.df_base = ss.df_view = None
        return

    with st.spinner("이벤트 데이터 변환 및 분석 중..."):
        flat_rows = process_rum_events(raw_events, tz_name="Asia/Seoul")
        ss.df_rtp_summary = analyze_rtp_timeouts(flat_rows)
        
        # 원본 로그 데이터프레임도 생성
        ss.df_base = to_base_dataframe(flat_rows)
        all_cols = [c for c in ss.df_base.columns if c != "timestamp(KST)"]

        ss.hidden_cols_user = [c for c in ss.hidden_cols_user if c in all_cols]
        ss.pending_hidden_cols_user = ss.hidden_cols_user.copy()

        visible_cols = [c for c in all_cols if c not in effective_hidden(all_cols, ss.pending_hidden_cols_user, ss.hide_defaults, FIXED_PIN)]
        ss.pin_slots = sanitize_pin_slots(ss.pin_slots, visible_cols, PIN_COUNT, FIXED_PIN)
        ss.pending_pin_slots = ss.pin_slots.copy()

        eff_hidden_applied = effective_hidden(all_cols, ss.hidden_cols_user, ss.hide_defaults, FIXED_PIN)
        ss.df_view = apply_view_filters(ss.df_base.copy(), hidden_cols=eff_hidden_applied)

        if "Call ID" in ss.df_base.columns:
            ss.unique_call_ids = sorted(ss.df_base["Call ID"].dropna().unique().tolist())

    if ss.df_rtp_summary.empty:
        st.info("분석 결과 RTP Timeout 통화가 없습니다.")
    else:
        st.toast(f"RTP Timeout 통화 {len(ss.df_rtp_summary)}건을 분석했습니다.")

# ─────────────────────────────────────────
# Main App Logic
# ─────────────────────────────────────────
def main():
    """메인 애플리케이션 실행 함수"""
    st.set_page_config(page_title="Datadog RUM 분석기", layout="wide")
    st.title("Datadog RUM 분석 Tool (API 기반)")

    try:
        settings = get_settings() # .streamlit/secrets.toml
        client = DatadogAPIClient(settings.api_key, settings.app_key, settings.site)
        st.write(f"**Site:** `{settings.site}`")
    except (KeyError, FileNotFoundError):
        st.error("설정 파일(.streamlit/secrets.toml)을 찾을 수 없거나 키가 누락되었습니다.")
        st.stop()

    initialize_session_state()
    
    ss = st.session_state
    run_user_id_search, run_rtp_analysis, run_custom_query, search_params = render_sidebar(ss, PIN_COUNT, FIXED_PIN)

    if run_user_id_search:
        handle_user_id_based_rum_search(client, search_params)
    
    if run_rtp_analysis:
        handle_rum_based_rtp_analysis(client, search_params)

    if run_custom_query:
        handle_custom_query_rum_search(client, search_params)
    
    render_main_view(ss, FIXED_PIN)

if __name__ == "__main__":
    main()
