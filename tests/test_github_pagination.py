"""Tests for paginated GitHub requests."""

import httpx

from src.config.settings import Settings
from src.github.client import GitHubClient


def create_settings() -> Settings:
    """Create isolated GitHub test settings."""

    return Settings(
        github_token="test-token",
        github_api_base_url="https://api.github.test",
        github_default_page_size=2,
        github_max_pages=10,
        github_max_retries=0,
    )


def test_paginated_list_endpoint() -> None:
    """Confirm list endpoints are paginated."""

    requested_pages: list[int] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))

        requested_pages.append(page)

        if page == 1:
            payload = [
                {"number": 1},
                {"number": 2},
            ]
        elif page == 2:
            payload = [
                {"number": 3},
            ]
        else:
            payload = []

        return httpx.Response(
            status_code=200,
            json=payload,
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_settings(),
        transport=transport,
    ) as client:
        records = client.get_all_pages(
            path="/repos/pallets/flask/pulls",
        )

    assert records == [
        {"number": 1},
        {"number": 2},
        {"number": 3},
    ]

    assert requested_pages == [1, 2]


def test_search_endpoint_item_key() -> None:
    """Confirm search endpoints use the items field."""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "total_count": 1,
                "items": [
                    {
                        "number": 101,
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)

    with GitHubClient(
        settings=create_settings(),
        transport=transport,
    ) as client:
        records = client.get_all_pages(
            path="/search/issues",
            params={"q": "repo:pallets/flask is:pr"},
            item_key="items",
            max_pages=1,
        )

    assert records == [
        {
            "number": 101,
        }
    ]
