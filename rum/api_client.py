# -*- coding: utf-8 -*-
import requests
import streamlit as st
from typing import Dict, Any, Optional


import urllib3

class DatadogAPIClient:
    """Datadog API 요청을 위한 중앙 클라이언트"""

    def __init__(self, api_key: str, app_key: str, site: str = "datadoghq.com", verify: bool = True):
        if not (api_key and app_key):
            raise ValueError("Datadog API_KEY와 APP_KEY는 필수입니다.")
        self.api_key = api_key
        self.app_key = app_key
        self.site = site
        self.base_url = f"https://api.{self.site}"
        self.verify = verify

        if not self.verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Use requests.Session for connection pooling (Keep-Alive)
        self.session = requests.Session()

    @property
    def _headers_v1(self) -> Dict[str, str]:
        return {"DD-API-KEY": self.api_key, "DD-APPLICATION-KEY": self.app_key}

    @property
    def _headers_v2_json(self) -> Dict[str, str]:
        return {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @st.cache_data(show_spinner=False, ttl=30)
    def get(_self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET 요청을 보냅니다. Streamlit 캐싱을 위해 self를 사용하지 않습니다."""
        url = f"{_self.base_url}{path}"
        # Use session if available (though _self is the instance)
        r = _self.session.get(url, headers=_self._headers_v1, params=params, timeout=30, verify=_self.verify)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST 요청을 보냅니다."""
        url = f"{self.base_url}{path}"
        r = self.session.post(url, headers=self._headers_v2_json, json=body, timeout=30, verify=self.verify)
        try:
            resp_json = r.json()
        except requests.exceptions.JSONDecodeError:
            resp_json = {"raw_text": r.text}
        r.raise_for_status()
        return resp_json