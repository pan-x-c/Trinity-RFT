"""Thin MCP server that wraps docker/sync.sh and docker/remote_run.sh.

Running as an MCP server (a regular child process) instead of through the
Bash tool avoids Claude Code's sandbox, which blocks outbound SSH connections.
"""

import asyncio
import os
import re

from mcp.server.fastmcp import FastMCP

DOCKER_DIR = os.path.dirname(os.path.abspath(__file__))

mcp = FastMCP("trinity-remote-test")

# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_TEST_NOISE_RE = re.compile(
    "|".join(
        [
            r"\[transformers\] Flash Attention 2",
            r"Monkey patch _flash_attention_forward",
            r"Skipping monkey patch for \w+",
            r"is_fx_tracing will return true",
            r"Using blocking ray\.get inside async actor",
            r"Before FSDP, memory",
            r"After FSDP, memory",
            r"\w+ForCausalLM contains.*parameters",
            r"colocated worker base class",
            r"W\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ torch/",
            r"INFO worker\.py:\d+ -- (Using address|Connecting to existing|Connected to Ray|Calling ray\.init)",
            r"sys:\d+: DeprecationWarning",
        ]
    )
)

_REPEATED_SUFFIX_RE = re.compile(r"\x1b\[32m\s*\[repeated \d+x across cluster\][^\x1b]*\x1b\[0m")

_RSYNC_ITEMIZE_PREFIX_RE = re.compile(r"^[<>ch.*][fdLDS][.+cstTpoguax]* ")


def _clean_output(raw: str) -> str:
    """Remove noise from combined sync + test output."""
    lines = raw.split("\n")
    result: list[str] = []
    in_warnings_section = False
    prev_blank = False

    for line in lines:
        plain = _ANSI_RE.sub("", line).strip()

        # ── pytest warnings summary section: skip entirely ──
        if "= warnings summary =" in plain:
            in_warnings_section = True
            continue
        if in_warnings_section:
            if re.match(r"^(\d+ passed|FAILED|ERROR|=)", plain):
                in_warnings_section = False
                result.append(line)
            continue

        # ── rsync --itemize-changes: show only transferred files ──
        m = _RSYNC_ITEMIZE_PREFIX_RE.match(plain)
        if m:
            if plain[1] == "d":
                continue
            line = "  " + plain[m.end() :]

        # ── test noise filter ──
        if plain and _TEST_NOISE_RE.search(plain):
            continue

        # Strip "[repeated Nx across cluster]" ANSI suffix from kept lines
        line = _REPEATED_SUFFIX_RE.sub("", line)

        # Collapse consecutive blank lines
        is_blank = not plain
        if is_blank and prev_blank:
            continue
        prev_blank = is_blank

        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------


async def _run_script(cmd: list[str], timeout: int) -> str:
    """Run a shell script and return combined stdout+stderr."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.path.join(DOCKER_DIR, ".."),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Command timed out after {timeout} seconds."

    output = stdout.decode(errors="replace") if stdout else ""
    if proc.returncode != 0:
        output += f"\n[exit code: {proc.returncode}]"
    return output


@mcp.tool()
async def sync_code(dry_run: bool = False) -> str:
    """Sync local code to the remote GPU server via rsync over SSH.

    Only git-tracked files are synced. Untracked files are listed as warnings.

    Args:
        dry_run: If true, show what would be transferred without making changes.
    """
    cmd = ["bash", os.path.join(DOCKER_DIR, "sync.sh")]
    if dry_run:
        cmd.append("--dry-run")
    return _clean_output(await _run_script(cmd, timeout=120))


@mcp.tool()
async def run_tests(
    module: str = "",
    keyword: str = "",
    quiet: bool = True,
    no_sync: bool = False,
    timeout: int = 600,
) -> str:
    """Sync code and run pytest on the remote GPU server inside Docker.

    Args:
        module: Test target under tests/ (directory or file). Empty runs all tests.
        keyword: Pytest -k expression to filter tests.
        quiet: Run pytest in quiet mode.
        no_sync: Skip the rsync step if code is already up to date.
        timeout: Timeout in seconds for the entire operation.
    """
    cmd = ["bash", os.path.join(DOCKER_DIR, "remote_run.sh")]
    if module:
        cmd.extend(["--module", module])
    if keyword:
        cmd.extend(["--keyword", keyword])
    if quiet:
        cmd.append("--quiet")
    if no_sync:
        cmd.append("--no-sync")
    cmd.extend(["--timeout", str(timeout)])
    return _clean_output(await _run_script(cmd, timeout=timeout + 30))


@mcp.tool()
async def check_status() -> str:
    """Check Docker container and Ray cluster status on the remote GPU server."""
    cmd = [
        "bash",
        "-c",
        f"source {os.path.join(DOCKER_DIR, 'common.sh')} && "
        "load_remote_env && "
        'ssh -p "$TRINITY_REMOTE_SSH_PORT" '
        "-o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=10 "
        '"$TRINITY_REMOTE_HOST" '
        f'"cd $TRINITY_REMOTE_WORKSPACE && bash docker/status.sh"',
    ]
    return await _run_script(cmd, timeout=30)


if __name__ == "__main__":
    mcp.run(transport="stdio")
