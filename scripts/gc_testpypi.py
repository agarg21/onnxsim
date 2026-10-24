#!/usr/bin/env python3
"""Garbage-collect old ``.dev`` releases from the TestPyPI project.

Once a day (if ``master`` has new commits), plus any manually dispatched run,
the full wheel matrix is uploaded to TestPyPI as a new ``X.Y.Z.devN`` release
(see ``.github/workflows/build-and-test.yml``, ``upload_pypi``). These
pile up quickly -- the recent ``0.6.5.dev*`` builds are ~375 MB each -- and
TestPyPI enforces a 10 GB per-project limit. This script prunes the old dev
builds so the project stays under the cap.

By default it deletes every ``.dev`` release *except* the newest few, and never
touches real tagged releases (``0.4.x``, ``0.5.0``, ...).

TestPyPI has no delete API (OIDC / trusted-publishing tokens are upload-only),
so deletion has to go through the authenticated web form. TestPyPI also
requires 2FA, which rules out scripted username/password login. Instead this
script reuses your *browser session cookie*, so you log in (and clear 2FA) in
the browser once and paste the cookie here.

Getting the cookie
------------------
1. Log in at https://test.pypi.org/ in your browser.
2. Open DevTools -> Network, click any request to test.pypi.org, and copy the
   whole ``Cookie:`` request header. (Or in the Application/Storage tab, copy
   the ``session_id`` cookie value and pass ``session_id=<value>``.)
3. Provide it via ``--cookie`` or the ``TESTPYPI_COOKIE`` environment variable.
   Quote it -- it contains ``;`` and spaces.

Usage
-----
    # See what would be deleted (no changes made):
    python scripts/gc_testpypi.py --dry-run

    # Actually delete, keeping the 3 newest dev builds, asking for confirmation:
    export TESTPYPI_COOKIE='session_id=...; ...'
    python scripts/gc_testpypi.py --keep 3

    # Non-interactive:
    python scripts/gc_testpypi.py --keep 3 --yes --cookie "session_id=..."

Only ``requests`` is required (``packaging`` is used if present for correct
version sorting, otherwise a built-in fallback is used).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import requests

BASE = "https://test.pypi.org"
DEFAULT_PROJECT = "onnxsim"

# Warehouse renders the CSRF token as a hidden form input on every page.
_CSRF_RE = re.compile(r'name="csrf_token"[^>]*value="([^"]+)"')


def human(nbytes: int) -> str:
    step = 1000.0
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < step or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= step
    return f"{nbytes:.1f} GB"


def _version_key(version: str):
    """Sort key for release versions, newest last.

    Prefers ``packaging.version`` for PEP 440 correctness; falls back to a
    tuple of ints extracted from the string so we degrade gracefully without
    the dependency.
    """
    try:
        from packaging.version import Version

        return (0, Version(version))
    except Exception:
        nums = tuple(int(n) for n in re.findall(r"\d+", version))
        return (1, nums, version)


def fetch_releases(project: str) -> dict[str, list[dict]]:
    """Return {version: [file-info, ...]} from the public JSON API."""
    url = f"{BASE}/pypi/{project}/json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["releases"]


def select_targets(releases: dict[str, list[dict]], keep: int, prune_tagged: bool):
    """Split releases into (to_delete, kept_dev, tagged) lists of versions."""
    dev = sorted((v for v in releases if ".dev" in v), key=_version_key)
    tagged = sorted((v for v in releases if ".dev" not in v), key=_version_key)

    kept_dev = dev[-keep:] if keep > 0 else []
    to_delete = [v for v in dev if v not in set(kept_dev)]

    if prune_tagged and tagged:
        # Keep only the newest tagged release; delete the rest.
        to_delete += tagged[:-1]

    return to_delete, kept_dev, tagged


_REAUTH_TITLE_RE = re.compile(r"<title>\s*Confirm password", re.IGNORECASE)


class ReauthRequired(RuntimeError):
    """Raised when Warehouse demands a fresh password/2FA confirmation."""


class TestPyPISession:
    def __init__(self, cookie: str, project: str):
        self.project = project
        self.s = requests.Session()
        self.s.headers["Cookie"] = cookie.strip()
        self.s.headers["User-Agent"] = "onnxsim-gc-testpypi/1.0"
        self._csrf: str | None = None

    def _releases_url(self) -> str:
        # The per-version "release" URL below is POST-only (it 404s on GET),
        # so the CSRF token has to come from the releases list page instead --
        # that page embeds one hidden csrf_token input per release's delete
        # form, all sharing the same session-wide token value.
        return f"{BASE}/manage/project/{self.project}/releases/"

    def _manage_url(self, version: str) -> str:
        return f"{BASE}/manage/project/{self.project}/release/{version}/"

    def _ensure_csrf(self) -> str:
        if self._csrf:
            return self._csrf
        url = self._releases_url()
        resp = self.s.get(url, timeout=30, allow_redirects=True)
        if "/account/login" in resp.url:
            raise RuntimeError(
                "Redirected to the login page -- the session cookie is missing "
                "or expired. Re-copy it from a logged-in browser."
            )
        resp.raise_for_status()
        if _REAUTH_TITLE_RE.search(resp.text):
            raise ReauthRequired(
                "TestPyPI wants a fresh password/2FA confirmation before it will "
                f"show the releases page. Open {url} in your browser, complete "
                "that prompt, then immediately re-copy the Cookie header and "
                "retry -- a cookie from ordinary browsing isn't enough."
            )
        m = _CSRF_RE.search(resp.text)
        if not m:
            raise RuntimeError(
                f"Could not find a CSRF token on {url}. The page may not exist "
                "or the cookie is not for a maintainer of this project."
            )
        self._csrf = m.group(1)
        return self._csrf

    def delete_release(self, version: str) -> None:
        """Submit the delete form for one release. Raises on failure.

        Does NOT verify the release is actually gone afterwards -- the old
        per-version GET used for that always 404s (it's a POST-only route)
        regardless of deletion state, so it can't distinguish success from
        failure. Verify via the public JSON API after the whole batch instead.
        """
        csrf = self._ensure_csrf()
        url = self._manage_url(version)
        resp = self.s.post(
            url,
            data={
                "csrf_token": csrf,
                "acknowledge_delete_files": "on",
                "acknowledge_install_break": "on",
                "acknowledge_no_reupload": "on",
                "acknowledge_irreversible": "on",
                "acknowledge_admins_cannot_undo": "on",
                "confirm_delete_version": version,
            },
            headers={"Referer": self._releases_url(), "Origin": BASE},
            timeout=30,
            allow_redirects=True,
        )
        if "/account/login" in resp.url:
            raise RuntimeError(
                "Redirected to the login page mid-run -- the session cookie "
                "expired. Re-copy it from a logged-in browser and retry."
            )
        if _REAUTH_TITLE_RE.search(resp.text):
            raise ReauthRequired(
                f"Reauth lapsed while deleting {version} -- re-confirm your "
                "password/2FA in the browser, re-copy the cookie, and retry."
            )
        resp.raise_for_status()
        # Warehouse flashes an error (and does NOT delete) if the confirmation
        # value doesn't match, and redirects back to the release page instead
        # of the project page on some failures.
        if "Could not delete release" in resp.text:
            raise RuntimeError(f"Server refused to delete {version} (flash error).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help=f"TestPyPI project name (default: {DEFAULT_PROJECT})")
    p.add_argument("--keep", type=int, default=3,
                   help="Number of newest .dev releases to keep (default: 3)")
    p.add_argument("--prune-tagged", action="store_true",
                   help="Also delete all but the newest real tagged release.")
    p.add_argument("--cookie", default=os.environ.get("TESTPYPI_COOKIE"),
                   help="TestPyPI session cookie (or set TESTPYPI_COOKIE).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted without deleting anything.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive confirmation prompt.")
    args = p.parse_args(argv)

    releases = fetch_releases(args.project)
    to_delete, kept_dev, tagged = select_targets(
        releases, args.keep, args.prune_tagged
    )

    def size_of(v):
        return sum(f["size"] for f in releases.get(v, []))

    total = sum(size_of(v) for v in releases)
    freed = sum(size_of(v) for v in to_delete)

    print(f"Project: {args.project}   TestPyPI limit: 10 GB")
    print(f"Current: {len(releases)} releases, {human(total)}\n")

    print(f"Keeping {len(tagged)} tagged release(s) and "
          f"{len(kept_dev)} newest dev build(s):")
    for v in tagged[-5:]:
        print(f"  keep  {v:<24} {human(size_of(v))}")
    if len(tagged) > 5:
        print(f"  ...   (+{len(tagged) - 5} more tagged)")
    for v in kept_dev:
        print(f"  keep  {v:<24} {human(size_of(v))}")

    print(f"\nDeleting {len(to_delete)} release(s), freeing {human(freed)} "
          f"-> {human(total - freed)} remaining:")
    for v in sorted(to_delete, key=_version_key):
        print(f"  DROP  {v:<24} {human(size_of(v))}")

    if not to_delete:
        print("\nNothing to delete. Already clean.")
        return 0

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return 0

    if not args.cookie:
        print("\nERROR: no session cookie. Pass --cookie or set TESTPYPI_COOKIE.",
              file=sys.stderr)
        print("See the module docstring for how to obtain it.", file=sys.stderr)
        return 2

    if not args.yes:
        ans = input(f"\nDelete these {len(to_delete)} releases? This is "
                    f"IRREVERSIBLE. [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    session = TestPyPISession(args.cookie, args.project)
    submitted, errored = [], 0
    for v in sorted(to_delete, key=_version_key):
        try:
            session.delete_release(v)
            print(f"  submitted {v}")
            submitted.append(v)
        except ReauthRequired as exc:
            # The whole session shares one fresh-auth window -- once it
            # lapses, every remaining attempt will fail the same way.
            print(f"\nERROR: {exc}", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED  {v}: {exc}", file=sys.stderr)
            errored += 1
        time.sleep(1)  # be polite to the server

    # The old per-version GET used to "confirm" a delete always 404s
    # regardless of outcome (it's a POST-only route), so verify against the
    # public JSON API instead -- it's the only reliable signal of what's
    # actually gone.
    print("\nVerifying against the public index...")
    still_present = set(fetch_releases(args.project)) & set(submitted)
    deleted = [v for v in submitted if v not in still_present]

    print(f"\nDone. {len(deleted)} deleted, {len(still_present)} submitted but "
          f"still present, {errored} errored outright. "
          f"Freed ~{human(freed - sum(size_of(v) for v in still_present))} "
          "(best case).")
    if still_present:
        print("Still present despite a submitted delete (reauth likely "
              "lapsed before the server processed it):", file=sys.stderr)
        for v in sorted(still_present, key=_version_key):
            print(f"  {v}", file=sys.stderr)
    return 1 if (errored or still_present or len(deleted) < len(to_delete)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
