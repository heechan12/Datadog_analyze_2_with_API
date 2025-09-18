import json
import requests
from typing import List, Dict, Any

from .api_client import DatadogAPIClient

def search_rum_events(
    client: DatadogAPIClient,
    query: str,
    from_ts: str,
    to_ts: str,
    limit_per_page: int = 200,
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """
    Datadog RUM 이벤트 검색 API를 호출하여 지정된 조건에 맞는 이벤트를 가져옵니다.

    - 기간: from_ts ~ to_ts (UTC, ISO 8601 형식)
    - 필터: query (Datadog 검색 쿼리)
    - 페이지네이션: 커서를 따라가며 max_pages까지 수집

    Args:
        client: DatadogAPIClient 인스턴스
        query: 검색할 Datadog 쿼리
        from_ts: 검색 시작 시간 (UTC, ISO 8601)
        to_ts: 검색 종료 시간 (UTC, ISO 8601)
        limit_per_page: 페이지당 가져올 이벤트 수
        max_pages: 최대 페이지 수

    Returns:
        검색된 RUM 이벤트의 원본(raw) 목록
    """
    body = {
        "filter": {"from": from_ts, "to": to_ts, "query": query},
        "page": {"limit": limit_per_page},
        "sort": "-timestamp",
    }
    
    all_events: List[Dict[str, Any]] = []
    cursor = None

    for _ in range(max_pages):
        if cursor:
            body["page"]["cursor"] = cursor

        data = client.post("/api/v2/rum/events/search", body)
        # data = resp.json()
        events = data.get("data", [])
        all_events.extend(events)

        cursor = data.get("meta", {}).get("page", {}).get("after")
        if not cursor:
            break

    return all_events
