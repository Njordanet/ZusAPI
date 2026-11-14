#!/usr/bin/env python3
"""
Admin CLI for ZüsAPI Founder & Lifetime System.
Uses Python standard library only.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional


def load_env_admin_key() -> str:
    # 1. Check existing environment variable
    if os.getenv("ZUS_ADMIN_API_KEY"):
        return os.environ["ZUS_ADMIN_API_KEY"]

    # 2. Candidate .env paths
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / "zus-api" / ".env",
    ]

    for p in candidates:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k == "ZUS_ADMIN_API_KEY" and v:
                            return v
            except Exception:
                pass

    return ""


def make_admin_request(
    endpoint: str,
    payload: Dict[str, any],
    api_base: str,
    admin_key: str,
) -> Dict[str, any]:
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    data_bytes = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Zus-Admin-Key": admin_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"❌ HTTP Error {e.code}: {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Request failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_create_codes(args, api_base: str, admin_key: str):
    payload = {
        "count": args.count,
        "max_redemptions": args.max_uses,
        "grant_tier": args.tier,
        "note": args.note,
    }
    if args.expires_days:
        payload["expires_days"] = args.expires_days

    res = make_admin_request("admin/founder-codes", payload, api_base, admin_key)
    codes = res.get("codes", [])
    print(f"✅ Generated {len(codes)} founder redemption code(s) [{res.get('grant_tier')} tier, max uses: {res.get('max_redemptions')}]:\n")
    for c in codes:
        print(f"  {c}")
    print()


def cmd_grant_user(args, api_base: str, admin_key: str):
    payload = {
        "user_id": args.user_id,
        "grant_tier": args.tier,
    }
    res = make_admin_request("admin/grant-founder", payload, api_base, admin_key)
    print(f"✅ Granted {res.get('grant_tier')} status to user {args.user_id} (updated: {res.get('updated_count')})")


def cmd_grant_early_adopters(args, api_base: str, admin_key: str):
    if not args.before_date and not args.before_version:
        print("❌ Error: Must specify at least one of --before-date or --before-version", file=sys.stderr)
        sys.exit(1)

    payload = {
        "before_created_at": args.before_date,
        "before_version": args.before_version,
        "grant_tier": args.tier,
    }
    res = make_admin_request("admin/grant-founder", payload, api_base, admin_key)
    print(f"✅ Grandfathered {res.get('updated_count')} early adopter account(s) to {res.get('grant_tier')} status.")


def main():
    parser = argparse.ArgumentParser(description="Züs Founder System Admin Tool")
    parser.add_argument(
        "--url",
        default=os.getenv("ZUS_API_BASE", "http://127.0.0.1:8000/api/v1/accounts"),
        help="Base URL for accounts endpoint (default: http://127.0.0.1:8000/api/v1/accounts)",
    )
    parser.add_argument(
        "--key",
        default="",
        help="Züs Admin API Key (overrides ZUS_ADMIN_API_KEY from .env)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # create-codes
    p_create = subparsers.add_parser("create-codes", help="Generate single or batch founder invitation codes")
    p_create.add_argument("--count", type=int, default=1, help="Number of codes to generate (1-200)")
    p_create.add_argument("--max-uses", type=int, default=1, help="Max redemptions allowed per code")
    p_create.add_argument("--tier", choices=["founder", "lifetime"], default="founder", help="Grant tier")
    p_create.add_argument("--note", type=str, default=None, help="Descriptive memo for admin auditing")
    p_create.add_argument("--expires-days", type=int, default=None, help="Days until code expiration")

    # grant-user
    p_user = subparsers.add_parser("grant-user", help="Directly grant founder or lifetime tier to a specific user UUID")
    p_user.add_argument("--user-id", required=True, type=str, help="Supabase user UUID")
    p_user.add_argument("--tier", choices=["founder", "lifetime"], default="founder", help="Grant tier")

    # grant-early-adopters
    p_early = subparsers.add_parser("grant-early-adopters", help="Grandfather early adopters prior to date or app version")
    p_early.add_argument("--before-date", type=str, default=None, help="ISO cutoff date (e.g. 2026-12-31T23:59:59Z)")
    p_early.add_argument("--before-version", type=str, default=None, help="Semver cutoff version (e.g. 2.28.0)")
    p_early.add_argument("--tier", choices=["founder", "lifetime"], default="founder", help="Grant tier")

    args = parser.parse_args()

    admin_key = args.key or load_env_admin_key()
    if not admin_key:
        print("❌ Error: ZUS_ADMIN_API_KEY not found in environment or .env files.", file=sys.stderr)
        print("Set ZUS_ADMIN_API_KEY in .env or pass --key <key>", file=sys.stderr)
        sys.exit(1)

    if args.subcommand == "create-codes":
        cmd_create_codes(args, args.url, admin_key)
    elif args.subcommand == "grant-user":
        cmd_grant_user(args, args.url, admin_key)
    elif args.subcommand == "grant-early-adopters":
        cmd_grant_early_adopters(args, args.url, admin_key)


if __name__ == "__main__":
    main()
