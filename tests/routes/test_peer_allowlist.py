"""Tests for the workspace-level ``allowed_ai_peers`` peer auto-creation guardrail.

Background
----------
When a remote profile (e.g. an OpenCode plugin started with `--open-profile
silverwolf`) ends up writing a session to the wrong workspace (e.g. ``theherta``)
because of a profile-mixing bug, Honcho's ``add_peers_to_session`` API silently
auto-creates the peer in the wrong workspace. This guardrail lets a workspace
owner pin down which peer names may be auto-created in their workspace via
``configuration.allowed_ai_peers``.

Backward compatibility
----------------------
When the field is unset, null, or empty, peer auto-creation is unrestricted.
Already-existing peers are always permitted regardless of the allowlist, so
re-adding a peer to a session is never blocked.
"""

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Workspace


def _new_peer_name(prefix: str) -> str:
    """Generate a unique-ish peer name to avoid cross-test interference."""
    return f"{prefix}-{generate_nanoid(size=8)}"


def _set_workspace_allowlist(
    client: TestClient, workspace_name: str, allowed: list[str] | None
) -> None:
    """Configure ``configuration.allowed_ai_peers`` on a workspace."""
    payload: dict = {"configuration": {}}
    if allowed is not None:
        payload["configuration"]["allowed_ai_peers"] = allowed
    response = client.put(f"/v3/workspaces/{workspace_name}", json=payload)
    assert response.status_code == 200, response.text


def _seed_peer(client: TestClient, workspace_name: str, peer_name: str) -> None:
    """Create a peer in the workspace directly so it's on the allowlist implicitly."""
    response = client.post(
        f"/v3/workspaces/{workspace_name}/peers", json={"name": peer_name}
    )
    assert response.status_code in (200, 201), response.text


# -----------------------------------------------------------------------
# Backward compatibility: no allowlist → unrestricted
# -----------------------------------------------------------------------


def test_no_allowlist_allows_any_peer_name(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """Without ``allowed_ai_peers`` set, peer auto-creation works for any name."""
    test_workspace, _ = sample_data
    # Defensive: make sure the workspace has no allowlist.
    _set_workspace_allowlist(client, test_workspace.name, None)

    new_peer = _new_peer_name("rogue")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers", json={"name": new_peer}
    )
    assert response.status_code in (200, 201)
    assert response.json()["id"] == new_peer


def test_empty_allowlist_acts_as_unrestricted(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """An explicit empty list is treated as "no restriction"."""
    test_workspace, _ = sample_data
    _set_workspace_allowlist(client, test_workspace.name, [])

    new_peer = _new_peer_name("rogue")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers", json={"name": new_peer}
    )
    assert response.status_code in (200, 201)


# -----------------------------------------------------------------------
# Allowlist rejection paths
# -----------------------------------------------------------------------


def test_allowlist_rejects_new_peer_in_session_add(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """``add_peers_to_session`` rejects a NEW peer name not on the allowlist.

    This is the exact shape of the OpenCode cross-workspace pollution: a
    session in workspace A asks to add a peer named for workspace B.
    """
    test_workspace, _ = sample_data

    # Two existing peers on the allowlist, one allowed-new (Foo) and one
    # rejected-new (Bar).
    existing = _new_peer_name("Existing")
    allowed = _new_peer_name("Foo")
    rejected = _new_peer_name("Bar")
    _seed_peer(client, test_workspace.name, existing)
    _set_workspace_allowlist(
        client, test_workspace.name, [existing, allowed]
    )

    session_id = _new_peer_name("sess")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={existing: {}, allowed: {}, rejected: {}},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "PeerNotAllowedException" in detail or rejected in detail
    # The full message should name the rejected peer(s) so operators can see
    # exactly what got blocked.
    assert rejected in detail


def test_allowlist_rejects_new_peer_in_explicit_create(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """``POST /v3/workspaces/{ws}/peers`` also enforces the allowlist."""
    test_workspace, _ = sample_data
    allowed = _new_peer_name("Foo")
    _set_workspace_allowlist(client, test_workspace.name, [allowed])

    rejected = _new_peer_name("Rogue")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers", json={"name": rejected}
    )
    assert response.status_code == 422


def test_allowlist_rejects_multiple_new_peers_at_once(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """All rejected peer names are listed in the 422 detail."""
    test_workspace, _ = sample_data
    allowed = _new_peer_name("Foo")
    _set_workspace_allowlist(client, test_workspace.name, [allowed])

    rogue_a = _new_peer_name("RogueA")
    rogue_b = _new_peer_name("RogueB")
    session_id = _new_peer_name("sess")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={rogue_a: {}, rogue_b: {}},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert rogue_a in detail
    assert rogue_b in detail


# -----------------------------------------------------------------------
# Allowlist does not block existing peers (the "re-add" semantic)
# -----------------------------------------------------------------------


def test_allowlist_does_not_block_existing_peers(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """A peer already in the workspace can be re-added to a session even when
    the allowlist doesn't mention it.

    Rationale: the allowlist only gates NEW peer creation. This avoids a
    churn problem where editing the allowlist would silently break existing
    sessions.
    """
    test_workspace, _ = sample_data
    existing = _new_peer_name("Existing")
    _seed_peer(client, test_workspace.name, existing)
    # Allowlist does NOT include `existing` — only a different name.
    _set_workspace_allowlist(
        client, test_workspace.name, [_new_peer_name("DifferentName")]
    )

    session_id = _new_peer_name("sess")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={existing: {}},
    )
    assert response.status_code == 200, response.text


def test_allowlist_does_not_block_partial_new_request(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """If ANY peer in a batch is a rejected new peer, the whole batch fails.

    This is the atomic-write guarantee: we never want to partially create
    half the requested peers before discovering a rejection.
    """
    test_workspace, _ = sample_data
    allowed = _new_peer_name("Foo")
    existing = _new_peer_name("Existing")
    _seed_peer(client, test_workspace.name, existing)
    _set_workspace_allowlist(client, test_workspace.name, [existing, allowed])

    rogue = _new_peer_name("Rogue")
    session_id = _new_peer_name("sess")
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={existing: {}, allowed: {}, rogue: {}},
    )
    assert response.status_code == 422
    # Confirm none of the would-be new peers were created by checking
    # workspace peer list. The session was also not created.
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list", json={}
    )
    assert response.status_code == 200
    names = {p["id"] for p in response.json()["items"]}
    assert rogue not in names


# -----------------------------------------------------------------------
# Realistic pollution regression test (the ThinkPad case)
# -----------------------------------------------------------------------


def test_pollution_regression_silverwolf_into_theherta(
    client: TestClient, sample_data: tuple[Workspace, object]
):
    """Reproduces the actual pollution case from the live deployment:

    A session in workspace A is asked to add peers ``Trailblazer`` (allowed)
    and ``SilverWolf`` (the AI peer from a sibling profile — NOT on A's
    allowlist). The call must be rejected with 422 and no ``SilverWolf``
    peer may appear in workspace A.
    """
    test_workspace, _ = sample_data
    # Seed the legitimate workspace peers and configure the allowlist so that
    # only those peers may be auto-created. Mirrors the real theherta config.
    theherta_legit = [
        "TheHerta",
        "Trailblazer",
        "HertaBuild",
        "HertaExplore",
        "HertaReviewer",
        "HertaQuickFix",
        "HertaDebugger",
        "HertaTester",
        "HertaArchivist",
    ]
    for peer_name in theherta_legit:
        _seed_peer(client, test_workspace.name, peer_name)
    _set_workspace_allowlist(client, test_workspace.name, theherta_legit)

    session_id = "20260706_000925_01de58-pollution-regression"
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={"Trailblazer": {}, "SilverWolf": {}},
    )
    assert response.status_code == 422
    assert "SilverWolf" in response.json()["detail"]

    # The session was NOT created (entire request rejected).
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/list", json={}
    )
    items = response.json()["items"]
    assert all(s["id"] != session_id for s in items)

    # And no SilverWolf peer leaked into the workspace.
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list", json={}
    )
    names = {p["id"] for p in response.json()["items"]}
    assert "SilverWolf" not in names
