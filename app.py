import streamlit as st
from datetime import datetime, timedelta
import pytz
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# RUM modules
from rum.api_client import DatadogAPIClient
from rum.config import get_settings, get_default_hidden_columns
from rum.datadog_api import search_rum_events
from rum.transform import process_rum_events, to_base_dataframe, apply_view_filters, summarize_calls, analyze_rtp_timeouts, categorize_rtp_timeouts, analyze_call_sessions
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
        "df_rtp_categorized": None,                     # RTP Timeout 분석 차트용 데이터
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
    params.pop("version_value", None)
    params.pop("build_version_value", None)

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
    params.pop("version_value", None)
    params.pop("build_version_value", None)

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
    ss.df_rtp_categorized = None # 차트 데이터 초기화

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
        ss.df_rtp_categorized = None
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
        ss.df_rtp_categorized = None
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
        ss.df_rtp_categorized = None
        ss.df_base = ss.df_view = None
        return

    with st.spinner("이벤트 데이터 변환 및 분석 중..."):
        flat_rows = process_rum_events(raw_events, tz_name="Asia/Seoul")
        ss.df_rtp_summary = analyze_rtp_timeouts(flat_rows)
        
        ss.df_rtp_categorized = categorize_rtp_timeouts(ss.df_rtp_summary)
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
# ─────────────────────────────────────────
# RTP Timeout 검색 및 2차 분석 (New - User ID 기반 시간대 검색)
# ─────────────────────────────────────────
def handle_rum_based_rtp_analysis_new(client: DatadogAPIClient, params: dict):
    """
    RTP Timeout 통화에 대한 2단계 분석을 수행 (User ID 기반 확장 검색)
    1. RTP Timeout 발생 Call ID 식별
    2. 해당 Call ID의 통화 시작/종료 시간 및 User ID 파악
    3. (Start Time ~ End Time) 구간의 해당 User ID의 모든 로그 검색
    """
    ss = st.session_state
    ss.df_summary = None
    ss.df_rtp_categorized = None

    api_params = params.copy()
    usr_id_value = api_params.pop("usr_id_value", None)
    version_value = api_params.pop("version_value", None)
    build_version_value = api_params.pop("build_version_value", None)
    api_params.pop("analysis_type", None)
    api_params.pop("custom_query", None)

    # 1단계: RTP Timeout이 발생한 Call ID 수집
    query_parts = ["@context.reason:(*RTP* OR *rtp*)"]

    def build_or_query_part(field, value_str):
        values = [v.strip() for v in value_str.split(',') if v.strip()]
        if not values:
            return None
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

    with st.spinner(f"1/3: RTP Timeout 이벤트 검색 중... (query: {final_query})"):
        rtp_timeout_events = search_rum_events(client=client, query=final_query, **api_params)
    
    if not rtp_timeout_events:
        st.info("해당 기간에 RTP Timeout으로 기록된 통화가 없습니다.")
        ss.df_rtp_summary = pd.DataFrame()
        ss.df_rtp_categorized = None
        ss.df_base = ss.df_view = None
        return

    # Call ID 식별
    # process_rum_events를 사용하여 Call ID를 통합
    flat_rtp_rows = process_rum_events(rtp_timeout_events, tz_name="Asia/Seoul")
    
    call_ids = set()
    for row in flat_rtp_rows:
        call_id = row.get("Call ID")
        if call_id:
            call_ids.add(call_id)
    
    if not call_ids:
        st.info("RTP Timeout 이벤트에서 Call ID를 찾을 수 없었습니다.")
        ss.df_rtp_summary = pd.DataFrame()
        ss.df_rtp_categorized = None
        ss.df_base = ss.df_view = None
        return

    st.toast(f"1/3: {len(call_ids)}개의 RTP Timeout 통화 ID 식별.")

    # 2단계: 식별된 Call ID들의 전체 로그(Call ID 기준)를 가져와서 시작/종료 시간 및 User ID 파악
    # 이는 정확한 세션 시간을 알기 위함임.
    call_id_list = list(call_ids)
    batch_size = 50
    boundary_events = []

    with st.spinner(f"2/3: {len(call_ids)}개 통화의 시간 범위 파악 중..."):
        batches = [call_id_list[i:i+batch_size] for i in range(0, len(call_id_list), batch_size)]
        executor = ThreadPoolExecutor()
        try:
            future_to_batch = {
                executor.submit(
                    search_rum_events,
                    client=client,
                    query=f'(@context.callID:({" OR ".join(f"{cid}" for cid in batch)}) OR @context.callId:({" OR ".join(f"{cid}" for cid in batch)}))',
                    **api_params
                ): batch for batch in batches
            }
            for future in as_completed(future_to_batch):
                try:
                    boundary_events.extend(future.result())
                except Exception as e:
                    st.error(f"시간 범위 파악 중 오류: {e}")
        finally:
            executor.shutdown(wait=True)

    if not boundary_events:
        st.warning("RTP Timeout 통화의 상세 이벤트를 가져오지 못했습니다.")
        return

    # User ID 및 시간 범위 추출
    flat_boundary_rows = process_rum_events(boundary_events, tz_name="Asia/Seoul")
    sessions = analyze_call_sessions(flat_boundary_rows)  # Returns list of dicts with Call ID, User ID, Start Time, End Time

    # analyze_call_sessions는 Start/End 이벤트가 명확해야만 세션을 반환함.
    # RTP Timeout의 경우 비정상 종료라 End 이벤트가 없을 수도 있음.
    # 따라서 analyze_call_sessions 결과에 없는 Call ID에 대해서는 flat_boundary_rows에서 min/max 시간을 계산해서 보완해야 함.
    
    session_map = {s["Call ID"]: s for s in sessions}
    final_sessions = []
    
    # Group events by call_id manually to fallback
    events_by_call = defaultdict(list)
    for row in flat_boundary_rows:
        cid = row.get("Call ID")
        if cid:
            events_by_call[cid].append(row)

    for cid in call_ids:
        if cid in session_map:
            final_sessions.append(session_map[cid])
        else:
            # Fallback: Use min/max timestamp of specific call logs
            if cid in events_by_call:
                rows = events_by_call[cid]
                # Sort by timestamp string (KST) - simplistic but usually works for finding bounds
                # Better to use original timestamp if available, but process_rum_events converts it.
                # Assuming ISO strings sort order matches time order.
                rows_sorted = sorted(rows, key=lambda x: x.get("timestamp(KST)", ""))
                
                # Find User ID
                u_id = None
                for r in rows_sorted:
                    if r.get("attributes.usr.id"):
                        u_id = r.get("attributes.usr.id")
                        if u_id: break # found
                
                start_ts = rows_sorted[0].get("timestamp(KST)")
                end_ts = rows_sorted[-1].get("timestamp(KST)")
                
                if start_ts and end_ts:
                    final_sessions.append({
                        "Call ID": cid,
                        "User ID": u_id,
                        "Start Time": start_ts,
                        "End Time": end_ts
                    })

    if not final_sessions:
        st.error("통화 세션 정보를 추출할 수 없습니다.")
        return

    st.toast(f"2/3: {len(final_sessions)}개의 통화 세션 시간 범위 파악 완료.")

    # 3단계: (User ID + Time Range)로 전체 로그 조회
    all_user_events = []
    with st.spinner(f"3/3: User ID 기반 전체 상세 로그 조회 중 ({len(final_sessions)} 세션)..."):
        executor = ThreadPoolExecutor()
        future_to_cid = {}
        
        for sess in final_sessions:
            call_id = sess["Call ID"]
            u_id = sess["User ID"]
            start_ts = sess["Start Time"]
            end_ts = sess["End Time"]
            
            if not u_id:
                # User ID 모르면 Call ID로라도 조회했던 boundary_events 사용 (혹은 스킵)
                continue

            try:
                # 타임스탬프 변환 (KST -> ISO UTC)
                s_dt = datetime.strptime(start_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                e_dt = datetime.strptime(end_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                
                kst = pytz.timezone("Asia/Seoul")
                s_dt_aware = kst.localize(s_dt)
                e_dt_aware = kst.localize(e_dt)
                
                # 범위 약간 여유 있게? (앞뒤 1초)
                s_search = (s_dt_aware - timedelta(seconds=1)).astimezone(pytz.utc).isoformat()
                e_search = (e_dt_aware + timedelta(seconds=1)).astimezone(pytz.utc).isoformat()
                
                sess_params = api_params.copy()
                sess_params["from_ts"] = s_search
                sess_params["to_ts"] = e_search
                
                # 쿼리: 해당 유저의 모든 로그
                q = f'@usr.id:"{u_id}"'
                
                future = executor.submit(search_rum_events, client, q, **sess_params)
                future_to_cid[future] = call_id

            except Exception as e:
                st.error(f"상세 조회 쿼리 구성 실패 ({call_id}): {e}")

        for f in as_completed(future_to_cid):
            cid = future_to_cid[f]
            try:
                evts = f.result()
                # Inject context call_id
                for evt in evts:
                    evt["_session_context_call_id"] = cid
                all_user_events.extend(evts)
            except Exception as e:
                 # 개별 실패는 무시하고 진행
                 pass
        executor.shutdown(wait=True)

    if not all_user_events:
        st.warning("상세 로그 조회 결과가 없습니다. (기존 Call ID 로그만 표시됩니다)")
        all_user_events = boundary_events # Fallback

    st.toast(f"3/3: 총 {len(all_user_events)}개의 상세 이벤트 로드 완료.")

    # 데이터 변환 및 시각화 준비
    with st.spinner("데이터 변환 및 분석 중..."):
        flat_rows = process_rum_events(all_user_events, tz_name="Asia/Seoul")
        
        # RTP 분석 결과는 "Call ID가 있는 로그" 만으로 분석해야 정확할 수 있음.
        # 하지만 analyze_rtp_timeouts 함수는 Call ID가 있는 행만 골라서 처리하므로,
        # User ID로 긁어온 로그 중 Call ID가 없는 로그는 무시됨 -> 이게 맞음.
        ss.df_rtp_summary = analyze_rtp_timeouts(flat_rows)
        ss.df_rtp_categorized = categorize_rtp_timeouts(ss.df_rtp_summary)
        
        # 원본 로그 테이블 (여기에는 User ID로 긁어온 Call ID 없는 로그도 포함됨 -> 이게 목표)
        ss.df_base = to_base_dataframe(flat_rows)
        
        # 뷰 업데이트
        update_view_state(ss)

    if ss.df_rtp_summary.empty:
        st.info("분석 결과 RTP Timeout 통화가 없습니다.")
    else:
        st.success(f"분석 완료: {len(ss.df_rtp_summary)}건의 통화, 총 {len(flat_rows)}건의 이벤트")


def main():
    """메인 애플리케이션 실행 함수"""
    st.set_page_config(page_title="Datadog RUM 분석기", layout="wide")
    st.title("Datadog RUM 분석 Tool (API 기반)")

    try:
        settings = get_settings() # .streamlit/secrets.toml
        client = DatadogAPIClient(settings.api_key, settings.app_key, settings.site, verify=settings.verify_ssl)
        st.write(f"**Site:** `{settings.site}` (SSL Verify: `{settings.verify_ssl}`)")
    except (KeyError, FileNotFoundError):
        st.error("설정 파일(.streamlit/secrets.toml)을 찾을 수 없거나 키가 누락되었습니다.")
        st.stop()

    initialize_session_state()
    
    ss = st.session_state
    run_user_id_search, run_rtp_analysis, run_custom_query, run_call_session_analysis, search_params = render_sidebar(ss, PIN_COUNT, FIXED_PIN)

    if run_user_id_search:
        handle_user_id_based_rum_search(client, search_params)
    
    if run_rtp_analysis:
        if ss.analysis_type == "RTP Timeout 분석(New)":
            handle_rum_based_rtp_analysis_new(client, search_params)
        else:
            handle_rum_based_rtp_analysis(client, search_params)

    if run_custom_query:
        handle_custom_query_rum_search(client, search_params)
    
    if run_call_session_analysis:
        handle_call_session_analysis(client, search_params)
    
    render_main_view(ss, FIXED_PIN)


# ─────────────────────────────────────────
# Call Session Based Analysis (2-Step)
# ─────────────────────────────────────────
def handle_call_session_analysis(client: DatadogAPIClient, params: dict):
    """
    1. Call Boundary(Start/End) 이벤트를 검색하여 세션을 식별
    2. 식별된 세션(User ID + Time Range)별로 정밀 검색 수행
    3. 결과 병합 및 시각화
    """
    ss = st.session_state
    ss.df_summary = None
    ss.df_rtp_summary = None
    ss.df_rtp_categorized = None

    api_params = params.copy()
    usr_id_filter = api_params.pop("usr_id_value", None)
    api_params.pop("analysis_type", None)
    api_params.pop("custom_query", None)
    api_params.pop("version_value", None)
    api_params.pop("build_version_value", None)

    # Step 1: Search for Boundary Events
    # requestVoiceCall (Start Out), pushReceived (Start In), notifyPushCall (Start In - API), SDK_CALL_STATUS_STOPPING (End)
    boundary_query = '(@resource.url_path:(*requestVoiceCall* OR *notifyPushCall* OR *SDK_CALL_STATUS_STOPPING*) OR @message:*pushReceived*)'
    
    if usr_id_filter:
        boundary_query += f' AND @usr.id:"{usr_id_filter}"'

    with st.spinner("1/2: 통화 세션 탐색 중 (Start/End 이벤트)..."):
        boundary_events = search_rum_events(client=client, query=boundary_query, **api_params)
    
    if not boundary_events:
        st.info("해당 기간에 통화 관련 이벤트(Start/End)가 없습니다.")
        ss.df_base = ss.df_view = None
        return

    # Step 2: Analyze Sessions
    flat_boundary_rows = process_rum_events(boundary_events, tz_name="Asia/Seoul")
    sessions = analyze_call_sessions(flat_boundary_rows)
    
    if not sessions:
        st.warning("이벤트를 찾았으나, 유효한 통화 세션(Start & End 쌍)을 식별하지 못했습니다.")
        # 디버깅을 위해 찾은 이벤트라도 보여줄 수 있음
        ss.df_base = to_base_dataframe(flat_boundary_rows)
        # 뷰 업데이트 로직 태움
        update_view_state(ss)
        return

    st.toast(f"1/2: {len(sessions)}개의 통화 세션 식별됨.")
    
    # Step 3: Detailed Search per Session
    all_detailed_events = []
    
    # 병렬 처리 준비
    # Datadog API rate limit 주의. 
    # 시간 범위가 서로 다르므로 각각 쿼리해야 함.
    
    with st.spinner(f"2/2: {len(sessions)}개 세션에 대해 상세 로그 조회 중..."):
        executor = ThreadPoolExecutor()
        futures = []
        
        for sess in sessions:
            call_id = sess["Call ID"]
            u_id = sess["User ID"]
            start_ts = sess["Start Time"] # KST string "YYYY-MM-DD HH:MM:SS.mmm KST"
            end_ts = sess["End Time"]
            
            if not u_id:
                continue

            # Convert KST time string back to ISO format (UTC) for API
            try:
                # remove " KST" and parse
                s_dt = datetime.strptime(start_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                e_dt = datetime.strptime(end_ts.replace(" KST", ""), "%Y-%m-%d %H:%M:%S.%f")
                
                # Assume strictly sequential in local time, apply time zone
                kst = pytz.timezone("Asia/Seoul")
                s_dt_aware = kst.localize(s_dt)
                e_dt_aware = kst.localize(e_dt)
                
                # Buffer time? Maybe +/- 1 second to catch boundary?
                s_search = (s_dt_aware - timedelta(seconds=1)).astimezone(pytz.utc).isoformat()
                e_search = (e_dt_aware + timedelta(seconds=1)).astimezone(pytz.utc).isoformat()
                
                # Query for this user in this time range
                # We can filter by CallID too if we want ONLY that call's logs, but user asked for "usr_id based query"
                # "각 Call ID 별로 파악되는 usr_id 를 기반으로 다시 Datadog 쿼리를 할거야"
                # "call id 의 시작시간과 종료시간 까지의 usr id 의 데이터가 되게끔 수정해줘"
                # So query: @usr.id:"{u_id}"
                
                # Copy params and override time
                sess_params = api_params.copy()
                sess_params["from_ts"] = s_search
                sess_params["to_ts"] = e_search
                
                q = f'@usr.id:"{u_id}"'
                
                futures.append(executor.submit(search_rum_events, client, q, **sess_params))
                
            except Exception as e:
                st.error(f"시간 변환 오류 (Call {call_id}): {e}")
        
        # Collect results
        for f in as_completed(futures):
            try:
                evts = f.result()
                all_detailed_events.extend(evts)
            except Exception as e:
                st.error(f"상세 조회 실패: {e}")
        
        executor.shutdown()

    if not all_detailed_events:
        st.warning("상세 로그를 찾을 수 없습니다. (검색된 구간에 로그가 없을 수 있음)")
        # Boundary 이벤트라도 보여주기 위해
        all_detailed_events = boundary_events

    st.success(f"총 {len(all_detailed_events)}건의 상세 이벤트 로드 완료")
    
    # Process and Display
    flat_rows = process_rum_events(all_detailed_events, tz_name="Asia/Seoul")
    
    # Summary could be useful too
    ss.df_summary = summarize_calls(flat_rows)
    
    ss.df_base = to_base_dataframe(flat_rows)
    update_view_state(ss)


def update_view_state(ss):
    """공통 DataFrame View 업데이트 로직 (df_base -> df_view)"""
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


if __name__ == "__main__":
    main()

