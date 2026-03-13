import os
import streamlit as st
from dataclasses import dataclass

@dataclass
class Settings:
    """애플리케이션 설정값을 저장하는 데이터 클래스"""
    api_key: str
    app_key: str
    site: str  # 예: "ap1.datadoghq.com", "datadoghq.com", "eu.datadoghq.com"
    verify_ssl: bool = True

def get_settings() -> Settings:
    """
    Streamlit secrets 또는 환경 변수에서 Datadog 설정값을 로드합니다.
    Streamlit secrets에 값이 있으면 우선적으로 사용하고, 없으면 환경 변수에서 찾습니다.
    """
    api_key = st.secrets.get("DD_API_KEY", os.getenv("DD_API_KEY", ""))
    app_key = st.secrets.get("DD_APP_KEY", os.getenv("DD_APP_KEY", ""))
    site = st.secrets.get("DD_SITE", os.getenv("DD_SITE", "datadoghq.com"))
    
    # SSL 검증 설정 (기본값 True)
    # .streamlit/secrets.toml 예시: DD_VERIFY_SSL = false
    verify_ssl = st.secrets.get("DD_VERIFY_SSL", os.getenv("DD_VERIFY_SSL", True))
    
    # st.secrets에서 읽어오면 boolean일 수 있고, env에서 읽으면 string일 수 있음
    if isinstance(verify_ssl, str):
        verify_ssl = verify_ssl.lower() in ("true", "1", "yes")
        
    return Settings(api_key=api_key, app_key=app_key, site=site, verify_ssl=verify_ssl)

def get_api_base(site: str) -> str:
    """Datadog API 기본 URL을 생성합니다."""
    return f"https://api.{site}"

def get_search_url(site: str) -> str:
    """RUM 이벤트 검색 API의 전체 URL을 생성합니다."""
    return f"{get_api_base(site)}/api/v2/rum/events/search"

# ─────────────────────────────────────────
# 기본 숨김 컬럼 목록
# ─────────────────────────────────────────
# 이 목록에 포함된 열은 UI의 '숨길 컬럼' 선택 목록에 나타나지 않으며,
# 기본적으로 테이블 뷰에서 숨겨집니다.
DEFAULT_HIDDEN_COLUMNS = [
    "session.id","usr.id","attribute.os.build",
    "type","application.id","session.type","view.url","view.referrer",
    "usr.name","usr.email","action.type","action.target.name",
    "resource.type","resource.url","error.message","error.source","error.stack",
    "device.type","os.name","browser.name",
    "attributes.os.build","attributes.os.name","attributes.os.version_major",
    "attributes.resource.size","attributes.resource.method",
    "attributes.resource.provider.domain","attributes.resource.provider.name","attributes.resource.provider.type",
    "attributes.resource.type","attributes.resource.id","attributes.resource.url",
    "attributes.resource.url_host","attributes.resource.url_scheme","attributes.resource.url_path_group",
    "attributes.session.matching_retention_filter.name","attributes.session.matching_retention_filter.id",
    "attributes.session.type","attributes.session.plan","attributes.type",
    "attributes.geo.continent","attributes.geo.country","attributes.geo.as.domain","attributes.geo.as.name","attributes.geo.as.type",
    "attributes.geo.country_iso_code","attributes.geo.city","attributes.geo.latitude","attributes.geo.continent_code",
    "attributes.geo.subdivision_iso_code","attributes.geo.country_subdivision_iso_code","attributes.geo.location",
    "attributes.geo.country_subdivision","attributes.geo.longitude",
    "attributes.view.url_path_group","attributes.view.id","attributes.view.url","attributes.view.url_path",
    "attributes.application.name","attributes.application.short_name","attributes.application.id",
    "attributes.connectivity.cellular.carrier_name","attributes.connectivity.status",
    "attributes.usr.anonymous_id","attributes.usr.usr_id","attributes.service",
    "attributes.device.name",
    "attributes.device.model",
    "attributes.device.type",
    "attributes.device.brand",
    "attributes.device.architecture",
    "attributes._dd.format_version",
    "tags",
    "attributes.usr.id",
    "timestamp",

]

def get_default_hidden_columns() -> list[str]:
    """기본 숨김 열 목록의 복사본을 반환합니다."""
    # 원본 리스트가 외부에서 수정되는 것을 방지하기 위해 복사본을 반환합니다.
    return list(DEFAULT_HIDDEN_COLUMNS)
