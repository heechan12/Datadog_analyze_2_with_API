import streamlit as st
from datetime import datetime
import pytz
import pandas as pd
import re

from .transform import apply_view_filters, analyze_rtp_timeouts, filter_dataframe

# 헬퍼 함수들을 rum/ui.py로 이동
def effective_hidden(all_cols: list[str], user_hidden: list[str], hide_defaults: list[str], fixed_pin: str) -> list[str]:
    """
    실제로 숨겨야 할 열 목록을 계산합니다.
    기본 숨김 목록과 사용자가 선택한 숨김 목록을 합치고, 고정 열은 제외합니다.
    """
    hidden = (set(hide_defaults) | set(user_hidden)) & set(all_cols)
    if fixed_pin in hidden:
        hidden.remove(fixed_pin)
    return sorted(list(hidden))

def sanitize_pin_slots(slot_values: list[str], visible_candidates: list[str], count: int, fixed_pin: str) -> list[str]:
    """
    핀 슬롯 값을 정리하고 유효한 값만 남깁니다.
    중복을 제거하고, 보이는 열 목록에 있는 값만 유지하며, 최대 개수를 맞춥니다.
    """
    allow = set(visible_candidates)
    seen, out = set(), []
    for v in slot_values:
        c = (v or "").strip()
        if c and c in allow and c not in seen and c != fixed_pin:
            out.append(c)
            seen.add(c)
    out += [""] * (count - len(out))
    return out[:count]

def reorder_for_pinned(df: pd.DataFrame, fixed_second: str, pin_slots: list[str]) -> pd.DataFrame:
    """
    고정 핀 설정에 따라 데이터프레임의 열 순서를 재정렬합니다.
    """
    if df is None or df.empty:
        return df
    
    pins, seen = [], set()
    if "timestamp(KST)" in df.columns:
        pins.append("timestamp(KST)")
        seen.add("timestamp(KST)")
    if fixed_second in df.columns and fixed_second not in seen:
        pins.append(fixed_second)
        seen.add(fixed_second)
    for c in pin_slots:
        if c and c in df.columns and c not in seen:
            pins.append(c)
            seen.add(c)
    rest = [c for c in df.columns if c not in seen]
    return df[pins + rest]

def apply_row_highlighting(df: pd.DataFrame, red_kws: str, blue_kws: str, yellow_kws: str):
    """
    사용자가 입력한 키워드에 따라 행 전체에 배경색과 글자색 하이라이트를 적용합니다.
    - 우선순위: 빨강 > 파랑 > 노랑
    - 키워드 매칭은 행 전체의 텍스트를 대상으로 하며, 대소문자를 구분하지 않습니다.
    """
    if not any([red_kws, blue_kws, yellow_kws]):
        return df.style

    r_kws = [kw.strip().lower() for kw in red_kws.split(',') if kw.strip()]
    b_kws = [kw.strip().lower() for kw in blue_kws.split(',') if kw.strip()]
    y_kws = [kw.strip().lower() for kw in yellow_kws.split(',') if kw.strip()]

    # 하이라이트 스타일 정의 (배경색 + 글자색)
    colors = {
        'red': 'background-color: #ffcccc; color: black;',
        'blue': 'background-color: #cce6ff; color: black;',
        'yellow': 'background-color: #ffffcc; color: black;'
    }

    def highlight_logic(row):
        full_row_text = ' '.join(row.astype(str)).lower()
        style = [''] * len(row)

        if r_kws and any(kw in full_row_text for kw in r_kws):
            style = [colors['red']] * len(row)
        elif b_kws and any(kw in full_row_text for kw in b_kws):
            style = [colors['blue']] * len(row)
        elif y_kws and any(kw in full_row_text for kw in y_kws):
            style = [colors['yellow']] * len(row)
            
        return style

    return df.style.apply(highlight_logic, axis=1)

def render_sidebar(ss, pin_count, fixed_pin):
    """
    사이드바 UI를 렌더링하고 검색 파라미터를 반환합니다.
    """
    with st.sidebar:
        st.markdown("### 분석 유형")
        ss.analysis_type = st.radio(
            "분석 유형 선택",
            ["User ID 분석", "RTP Timeout 분석", "RTP Timeout 분석(New)", "Custom Query 분석", "Call Session 분석"],
            label_visibility="collapsed",
            key="analysis_type_radio" # 위젯 키를 통해 상태 유지
        )

        st.divider()

        usr_id = ""
        version_value = ""
        build_version_value = ""
        if ss.analysis_type == "User ID 분석":
            st.markdown("### 검색 조건")
            usr_id = st.text_input("usr.id", value="", placeholder="예: user_1234 (비우면 전체 *)")
        elif ss.analysis_type == "RTP Timeout 분석" or ss.analysis_type == "RTP Timeout 분석(New)":
            st.markdown("### 검색 조건")
            usr_id = st.text_input("usr.id (쉼표로 구분)", value="", placeholder="예: id1,id2 (비우면 전체)")
            version_value = st.text_input("version (쉼표로 구분)", value="", placeholder="예: 1.0.0,1.0.1 (비우면 전체)")
            build_version_value = st.text_input("build_version (쉼표로 구분)", value="", placeholder="예: 100,101 (비우면 전체)")
        elif ss.analysis_type == "Call Session 분석":
             st.markdown("### 검색 조건")
             usr_id = st.text_input("usr.id (Optional for filtering)", value="", placeholder="예: user_1234 (비우면 모든 통화 검색)")

        kst = pytz.timezone("Asia/Seoul")
        st.markdown("##### 검색 기간 (KST)")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("시작 날짜", value=ss.start_dt.date())
        start_time = col1.time_input("시작 시간", value=ss.start_dt.time())
        end_date = col2.date_input("종료 날짜", value=ss.end_dt.date())
        end_time = col2.time_input("종료 시간", value=ss.end_dt.time())

        ss.start_dt = kst.localize(datetime.combine(start_date, start_time))
        ss.end_dt = kst.localize(datetime.combine(end_date, end_time))

        is_valid_time = ss.start_dt < ss.end_dt
        if not is_valid_time:
            st.error("시작 시간은 종료 시간보다 빨라야 합니다.")

        run_search = False
        run_rtp_analysis = False
        run_custom_query = False
        run_call_session_analysis = False

        if ss.analysis_type == "User ID 분석":
            run_search = st.button("조회", disabled=not is_valid_time)
        elif ss.analysis_type == "RTP Timeout 분석" or ss.analysis_type == "RTP Timeout 분석(New)":
            run_rtp_analysis = st.button("RTP Timeout 분석", disabled=not is_valid_time)
        elif ss.analysis_type == "Custom Query 분석":
            run_custom_query = st.button("조회", disabled=not is_valid_time)
        elif ss.analysis_type == "Call Session 분석":
            run_call_session_analysis = st.button("Call Session 분석", disabled=not is_valid_time)

        st.divider()
        st.markdown("### 표시 옵션")
        # 모든 분석 유형에서 df_base가 있으면 표시 옵션을 렌더링
        if ss.df_base is not None:
            render_options_sidebar(ss, pin_count, fixed_pin)
        else:
            st.info("먼저 '조회' 또는 '분석'을 실행하세요.")

    search_params = {
        "usr_id_value": usr_id,
        "version_value": version_value,
        "build_version_value": build_version_value,
        "custom_query": ss.custom_query,
        "from_ts": ss.start_dt.astimezone(pytz.utc).isoformat(),
        "to_ts": ss.end_dt.astimezone(pytz.utc).isoformat(),
        "limit_per_page": 1000,
        "max_pages": 20,
        "analysis_type": ss.analysis_type
    }
    return run_search, run_rtp_analysis, run_custom_query, run_call_session_analysis, search_params

def render_options_sidebar(ss, pin_count, fixed_pin):
    """사이드바에 표시 옵션(열 숨김, 핀 설정 등)을 렌더링합니다."""
    ss.table_height = st.slider("표 높이(px)", 400, 2000, ss.table_height, 50)

    all_cols = [c for c in ss.df_base.columns if c != "timestamp(KST)"]
    defaults_set = set(ss.hide_defaults)
    options_for_hide = sorted([c for c in all_cols if c not in defaults_set and c != fixed_pin])

    st.markdown("### 숨길 컬럼(선택)")
    ss.pending_hidden_cols_user = st.multiselect(
        "숨길 컬럼", options_for_hide, default=ss.pending_hidden_cols_user, label_visibility="collapsed"
    )

    st.markdown("### 핀(왼쪽 고정) 순서")
    eff_hidden_proposed = effective_hidden(all_cols, ss.pending_hidden_cols_user, ss.hide_defaults, fixed_pin)
    visible_candidates = [c for c in all_cols if c not in eff_hidden_proposed and c != fixed_pin]
    ss.pending_pin_slots = sanitize_pin_slots(ss.pending_pin_slots, visible_candidates, pin_count, fixed_pin)
    slot_options = [""] + visible_candidates

    for i in range(pin_count):
        ss.pending_pin_slots[i] = st.selectbox(
            f"핀 #{i+1}", options=slot_options,
            index=slot_options.index(ss.pending_pin_slots[i]) if ss.pending_pin_slots[i] in slot_options else 0,
            key=f"pin_{i}"
        )

    if st.button("보기 새로고침"):
        ss.hidden_cols_user = ss.pending_hidden_cols_user.copy()
        ss.pin_slots = sanitize_pin_slots(ss.pending_pin_slots, visible_candidates, pin_count, fixed_pin)
        eff_hidden_applied = effective_hidden(all_cols, ss.hidden_cols_user, ss.hide_defaults, fixed_pin)
        ss.df_view = apply_view_filters(ss.df_base.copy(), hidden_cols=eff_hidden_applied)
        st.rerun()

    if st.button("모두 표시(사용자 숨김 초기화)"):
        ss.pending_hidden_cols_user = []
        ss.hidden_cols_user = []
        eff_hidden_applied = effective_hidden(all_cols, [], ss.hide_defaults, fixed_pin)
        ss.df_view = apply_view_filters(ss.df_base.copy(), hidden_cols=eff_hidden_applied)
        st.rerun()

def render_main_view(ss, fixed_pin):
    """메인 화면(통화 분석, 이벤트 로그)을 렌더링합니다."""
    print("[App] render_main_view called")
    
    # Custom Query 분석 시, 검색창을 메인 뷰에 표시
    if ss.analysis_type == "Custom Query 분석":
        st.markdown("### 검색 조건")
        st.warning("공란으로 검색하지 말아주세요~")
        ss.custom_query = st.text_area("Datadog Query", value=ss.custom_query, placeholder="예: @context.callID:\"...\" AND (*ERROR* OR *FAIL*)", height=100, label_visibility="collapsed")

    if ss.get('df_rtp_summary') is not None and not ss.df_rtp_summary.empty:
        st.markdown("## RTP Timeout 분석 결과")
        st.dataframe(ss.df_rtp_summary, use_container_width=True, height=800)

        # RTP 분석 차트
        if ss.get('df_rtp_categorized') is not None:
            st.markdown("### RTP Timeout 통계")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 사용자 별 발생 건 수")
                st.bar_chart(ss.df_rtp_categorized['by_user'].set_index('User ID'))
            with col2:
                st.markdown("##### 앱 버전 별 발생 건 수")
                st.bar_chart(ss.df_rtp_categorized['by_version'].set_index('App Version'))
            with col3:
                st.markdown("##### 통화 시간 별 발생 건 수")
                st.bar_chart(ss.df_rtp_categorized['by_duration'].set_index('Duration Category'))


        st.divider()

    if ss.df_view is not None:
        if ss.df_summary is not None and not ss.df_summary.empty:
            st.markdown("## 통화 분석 결과")
            st.dataframe(ss.df_summary, use_container_width=True)
            st.divider()

        st.markdown("## RUM 로그")

        # --- 필터 및 하이라이트 UI ---
        # url path 필터
        col1, col2 = st.columns([9, 1])
        filter_text = col1.text_input("URL Path 필터", placeholder="쉼표(,)로 구분하여 여러 개 입력")
        is_and = col2.checkbox("AND 조건", help="모든 키워드를 포함하는 로그만 필터링")
        # Call ID 필터
        if ss.unique_call_ids:
            selected_call_id = st.selectbox(
                "Call ID 필터",
                options=["전체"] + ss.unique_call_ids,
                help="특정 통화에 해당하는 로그만 필터링합니다."
            )
        else:
            selected_call_id = "전체"
        
        # Call ID 선택 시 피드백 (Toast & Log)
        if ss.analysis_type == "RTP Timeout 분석(New)":
            if selected_call_id != "전체":
                if ss.get("last_selected_call_id") != selected_call_id:
                    msg = f"선택한 Call ID({selected_call_id})로 로그를 상세 필터링 중입니다..."
                    st.toast(msg)
                    print(f"[App] {msg}")
                    ss.last_selected_call_id = selected_call_id
            else:
                 ss.last_selected_call_id = "전체"
        
        st.markdown("##### 행 하이라이트 (쉼표로 구분, OR 조건)")
        h_col1, h_col2, h_col3 = st.columns(3)
        red_kws = h_col1.text_input("🔴 빨강", placeholder="빨강으로 강조할 키워드")
        blue_kws = h_col2.text_input("🔵 파랑", placeholder="파랑으로 강조할 키워드")
        yellow_kws = h_col3.text_input("🟡 노랑", placeholder="노랑으로 강조할 키워드")
        # --- 필터 및 하이라이트 UI 끝 ---

        df_render = reorder_for_pinned(ss.df_view, fixed_pin, ss.pin_slots)
        
        # Call ID 필터링 적용
        if selected_call_id != "전체":
            print(f"[App] Filtering by Call ID: {selected_call_id}")
            if "Related Call ID" in df_render.columns:
                cond1 = df_render["Call ID"] == selected_call_id
                cond2 = df_render["Related Call ID"] == selected_call_id
                df_render = df_render[cond1 | cond2]
            else:
                df_render = df_render[df_render["Call ID"] == selected_call_id]
            print(f"[App] Filtered rows: {len(df_render)}")

        if filter_text:
            df_render = filter_dataframe(df_render, fixed_pin, filter_text, is_and)

        # Row Limit for Performance (Prevent WebSocket Crash)
        MAX_ROWS = 5000
        if len(df_render) > MAX_ROWS:
            st.warning(f"⚠️ 데이터가 너무 많아({len(df_render):,}행) 상위 {MAX_ROWS:,}개만 표시합니다. 전체 데이터 확인은 JSON 보기를 이용해주세요.")
            df_render = df_render.head(MAX_ROWS)

        # 하이라이트 적용
        print("[App] Applying highlights/styles...")
        if len(df_render) > 3000 and any([red_kws, blue_kws, yellow_kws]):
             st.warning(f"데이터가 많아({len(df_render)}행) 하이라이트 기능이 비활성화됩니다.")
             styler = df_render.style
        else:
             styler = apply_row_highlighting(df_render, red_kws, blue_kws, yellow_kws)
        print("[App] Style applied. Rendering dataframe...")

        if df_render.size > 262144:  # default is 2**18
            pd.set_option("styler.render.max_elements", df_render.size + 1)

        st.dataframe(styler, use_container_width=True, height=ss.table_height)
        print("[App] Dataframe rendered.")
        
        with st.expander("원본 이벤트(JSON) 보기"):
            if st.checkbox("JSON 변환/표시"):
                st.json(df_render.head(50).to_dict(orient="records"))
    elif ss.get('df_rtp_summary') is None and ss.df_view is None:
        if ss.analysis_type == "Custom Query 분석":
            st.caption("쿼리 입력 후 '조회'를 실행하세요.")
        else:
            st.caption("조회 실행 후 결과가 나타납니다.")
