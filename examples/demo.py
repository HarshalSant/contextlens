"""
ContextLens demo — no API key required.

Simulates a realistic ~30-turn agent loop with canned data, then runs
the full ContextLens analysis and generates:
  1. A terminal report (printed here)
  2. demo_report.html (open in any browser)

Run:
    python examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running directly from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextlens.analyzer import analyze_trace
from contextlens.capture import _dict_to_trace
from contextlens.cli import _print_report
from contextlens.reporter import render_html_report


# ---------------------------------------------------------------------------
# Canned content — realistic enough to trigger all four waste detectors
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional software engineering assistant helping a team build a SaaS application.
You have access to tools for searching code, reading files, running tests, and querying databases.
Always be precise, cite relevant code when possible, and suggest tests for any changes you make.
Follow the team's coding standards: Python 3.11+, typed, ruff-clean, pytest for tests.
Do not make breaking changes without explicit approval.
""".strip()

TOOL_SCHEMAS = [
    {
        "name": "search_code",
        "description": "Search the codebase for a pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or literal to search for"},
                "path": {"type": "string", "description": "Directory to search in"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_tests",
        "description": "Execute the test suite or a subset of tests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Test file or module"},
                "flags": {"type": "string", "description": "Extra pytest flags"},
            },
        },
    },
    {
        "name": "query_database",
        "description": "Run a read-only SQL query against the production database.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "SQL query"}},
            "required": ["sql"],
        },
    },
    # This tool is defined but NEVER called — triggers unused_tool_schema detector
    {
        "name": "send_email",
        "description": "Send an email notification to a team member.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

# A large retrieval chunk that gets injected at turn 3 and never referenced again
STALE_RETRIEVAL_CHUNK = """
retrieved document: Architecture Decision Record #42
source: docs/adr/0042-database-strategy.md
excerpt:
We evaluated three database strategies for the user data store: (1) PostgreSQL with row-level
security, (2) MongoDB with document-level permissions, and (3) a hybrid approach using
PostgreSQL for structured data and Redis for session caching.
After a 2-week proof-of-concept, we selected PostgreSQL with row-level security because it
aligns with our existing operational expertise, supports complex relational queries needed for
billing, and integrates cleanly with our SQLAlchemy ORM layer.
Migration path: existing MySQL tables will be migrated using Alembic with zero-downtime
blue-green deploys. Target completion: Q3 2024.
Performance target: p99 query latency < 50ms for all user-facing endpoints.
""".strip()

# A large tool result that gets added at turn 5 and re-billed many times
SEARCH_RESULT = """
search_code result for pattern='class UserService':
  File: src/services/user_service.py, line 42
  class UserService:
      def __init__(self, db: Database, cache: RedisCache) -> None:
          self.db = db
          self.cache = cache
          self._logger = logging.getLogger(__name__)

      def get_user(self, user_id: int) -> User | None:
          cached = self.cache.get(f"user:{user_id}")
          if cached:
              return User.model_validate(cached)
          user = self.db.query(User).filter_by(id=user_id).first()
          if user:
              self.cache.set(f"user:{user_id}", user.model_dump(), ttl=300)
          return user

      def create_user(self, payload: CreateUserRequest) -> User:
          if self.db.query(User).filter_by(email=payload.email).first():
              raise DuplicateEmailError(payload.email)
          user = User(**payload.model_dump())
          self.db.add(user)
          self.db.commit()
          self.cache.invalidate(f"user:*")
          return user
""".strip()

# A short user query that repeats verbatim across several turns — triggers duplicate detector
REPEATED_USER_PREFIX = "Please continue working on the authentication module."


def _make_turn(
    turn_index: int,
    timestamp: datetime,
    messages: list[dict],
    tool_calls_made: list[str],
    total_tokens: int,
) -> dict:
    return {
        "turn_index": turn_index,
        "timestamp": timestamp.isoformat(),
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "total_tokens": total_tokens,
        "tool_names_defined": [t["name"] for t in TOOL_SCHEMAS],
        "tool_names_called": tool_calls_made,
        "raw_request": {
            "model": "claude-3-5-sonnet-20241022",
            "system": SYSTEM_PROMPT,
            "tools": TOOL_SCHEMAS,
            "messages": messages,
        },
    }


def build_demo_trace() -> dict:
    """Construct a realistic 30-turn agent trace with known waste patterns."""
    turns = []
    base_time = datetime(2024, 6, 1, 10, 0, 0)
    messages: list[dict] = []

    for i in range(30):
        ts = base_time + timedelta(minutes=i * 2)
        tool_calls: list[str] = []

        if i == 0:
            messages = [
                {"role": "user", "content": "Help me refactor the authentication module to use JWT tokens."},
                {"role": "assistant", "content": "I'll start by searching for the existing auth code to understand the current structure."},
            ]
            tool_calls = ["search_code"]

        elif i == 1:
            messages = messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_001", "content": SEARCH_RESULT},
                    ],
                },
                {"role": "assistant", "content": "I found the UserService. Let me also read the current auth implementation."},
            ]
            tool_calls = ["read_file"]

        elif i == 2:
            messages = messages + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_002",
                            "content": "File: src/auth/session.py\n\nclass SessionAuth:\n    def authenticate(self, token: str) -> User | None:\n        record = self.db.query(Session).filter_by(token=token).first()\n        if not record or record.expires_at < datetime.utcnow():\n            return None\n        return record.user\n",
                        }
                    ],
                },
                {"role": "assistant", "content": "Now I have a clear picture. The existing auth uses database-backed sessions. I'll design the JWT migration."},
            ]
            tool_calls = []

        elif i == 3:
            # Inject the stale retrieval chunk — never referenced again
            messages = messages + [
                {
                    "role": "user",
                    "content": f"{STALE_RETRIEVAL_CHUNK}\n\n{REPEATED_USER_PREFIX}",
                },
                {"role": "assistant", "content": "I'll implement the JWT token service now."},
            ]

        elif i == 4:
            messages = messages + [
                {"role": "user", "content": REPEATED_USER_PREFIX},
                {
                    "role": "assistant",
                    "content": (
                        "Here is the JWTService implementation:\n\n"
                        "```python\nimport jwt\nfrom datetime import datetime, timedelta\n\n"
                        "class JWTService:\n    def __init__(self, secret: str, algorithm: str = 'HS256'):\n"
                        "        self.secret = secret\n        self.algorithm = algorithm\n\n"
                        "    def encode(self, user_id: int, exp_minutes: int = 60) -> str:\n"
                        "        payload = {'sub': user_id, 'exp': datetime.utcnow() + timedelta(minutes=exp_minutes)}\n"
                        "        return jwt.encode(payload, self.secret, algorithm=self.algorithm)\n\n"
                        "    def decode(self, token: str) -> dict:\n"
                        "        return jwt.decode(token, self.secret, algorithms=[self.algorithm])\n```"
                    ),
                },
            ]
            tool_calls = ["run_tests"]

        elif i in (5, 6, 7, 8):
            # Run tests, iterate on failures — SEARCH_RESULT re-billed from turn 1
            messages = messages + [
                {"role": "user", "content": REPEATED_USER_PREFIX},
                {"role": "assistant", "content": f"Running tests (iteration {i - 4})... adjusting token expiry handling."},
            ]
            tool_calls = ["run_tests"]

        elif i == 9:
            messages = messages + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_009",
                            "content": "Test run results:\n  PASSED: test_jwt_encode, test_jwt_decode, test_jwt_expiry\n  FAILED: test_refresh_token (AttributeError: 'JWTService' has no attribute 'refresh')\n  3 passed, 1 failed in 0.42s",
                        }
                    ],
                },
                {"role": "assistant", "content": "One failure: need to add refresh token support."},
            ]
            tool_calls = ["read_file"]

        elif i in range(10, 20):
            # Continuation turns — all carrying the full message history
            # including SEARCH_RESULT and STALE_RETRIEVAL_CHUNK from early turns
            step_msgs = [
                "Adding refresh token endpoint.",
                "Updating middleware to validate JWT.",
                "Writing integration tests for the new auth flow.",
                "Fixing edge case: expired token during refresh.",
                "Updating documentation strings.",
                "Running full test suite.",
                "Addressing review comments on the PR.",
                "Squashing migrations.",
                "Final cleanup pass.",
                "All tests green. Preparing summary.",
            ]
            messages = messages + [
                {"role": "user", "content": REPEATED_USER_PREFIX},
                {"role": "assistant", "content": step_msgs[i - 10]},
            ]
            if i in (10, 13, 16):
                tool_calls = ["run_tests"]
            elif i in (11, 14):
                tool_calls = ["search_code"]
            elif i == 12:
                tool_calls = ["query_database"]

        else:
            # Turns 20-29: wrap-up and documentation
            messages = messages + [
                {"role": "user", "content": REPEATED_USER_PREFIX},
                {
                    "role": "assistant",
                    "content": (
                        f"Wrap-up turn {i}: finalizing the authentication migration. "
                        "JWT tokens are now validated on every request via middleware. "
                        "Session table will be deprecated after a 30-day migration window."
                    ),
                },
            ]

        # Approximate token count (system + tools + messages)
        system_tokens = len(SYSTEM_PROMPT) // 4
        tools_tokens = sum(len(json.dumps(t)) // 4 for t in TOOL_SCHEMAS)
        msgs_tokens = sum(len(json.dumps(m)) // 4 for m in messages)
        total_tokens = system_tokens + tools_tokens + msgs_tokens

        turns.append(
            _make_turn(
                turn_index=i,
                timestamp=ts,
                messages=list(messages),  # snapshot of current context
                tool_calls_made=tool_calls,
                total_tokens=total_tokens,
            )
        )

    return {
        "run_id": "demo-001",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "created_at": base_time.isoformat(),
        "turns": turns,
    }


def main() -> None:
    print("ContextLens Demo")
    print("=" * 60)
    print("Building simulated 30-turn agent trace...")

    trace_dict = build_demo_trace()
    trace = _dict_to_trace(trace_dict)

    print(f"  {len(trace.turns)} turns, model={trace.model}")
    print("\nRunning analysis pipeline...")

    report = analyze_trace(trace)

    print("\n")
    _print_report(report, top=15)

    # Write HTML report
    out_dir = Path(__file__).parent
    html_path = out_dir / "demo_report.html"
    html = render_html_report(report)
    html_path.write_text(html, encoding="utf-8")

    # Save trace JSON for CLI testing
    trace_path = out_dir / "demo_trace.json"
    trace_path.write_text(json.dumps(trace_dict, indent=2), encoding="utf-8")

    print(f"\n[HTML report written] {html_path}")
    print(f"[Trace JSON written]  {trace_path}")
    print("\nTip: contextlens analyze examples/demo_trace.json")
    print("     contextlens report examples/demo_trace.json -o my_report.html")


if __name__ == "__main__":
    main()
