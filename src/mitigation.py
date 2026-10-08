"""Automated, reversible firewall mitigation engine for Linux nftables.

Translates incident mitigation directives emitted by ``RiskEngine`` (``ALLOW``,
``LOG_RATE_LIMIT``, ``QUARANTINE_TEMP``, ``QUARANTINE_ISOLATE``) into safe,
dynamic, and reversible containment actions using native Linux ``nftables``.

Architecture & Netfilter Rules
------------------------------
The engine manages a dedicated table and set hierarchy to isolate malicious
hosts dynamically without flushing existing system firewall rules:

::

    nft add table inet idps_filter
    nft add chain inet idps_filter input { type filter hook input priority 0; policy accept; }
    nft add chain inet idps_filter forward { type filter hook forward priority 0; policy accept; }
    nft add set inet idps_filter quarantine_v4 { type ipv4_addr; flags timeout; }
    nft add rule inet idps_filter input ip saddr @quarantine_v4 drop
    nft add rule inet idps_filter forward ip saddr @quarantine_v4 drop

Native Timeouts (TTL)
---------------------
When an IP is added to the ``quarantine_v4`` set, a kernel-level timeout is
attached (e.g., ``timeout 60s``). This guarantees that even if the NIDS process
crashes or loses state, the isolated host is automatically restored by the
kernel after the lease expires, preventing permanent lockouts.

Dry-Run / Simulation Mode
-------------------------
If the script is not executed with root privileges (or if ``nft`` is missing),
it automatically falls back to ``dry_run = True``. In this mode, all state is
tracked in-memory, and the exact ``nft`` CLI commands are logged instead of
being executed, allowing full pipeline testing on standard user environments.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Final

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.mitigation")

#: The default strict isolation lease duration (5 minutes).
DEFAULT_LEASE_SECONDS: Final[int] = 300

__all__: Final[list[str]] = ["MitigationEngine"]


# --------------------------------------------------------------------------- #
# Mitigation Engine                                                            #
# --------------------------------------------------------------------------- #


class MitigationEngine:
    """Automated mitigation engine mapping risk actions to nftables rules.

    Manages the dynamic quarantine of malicious IP addresses via Linux
    ``nftables``. Supports automatic TTL expiration, manual overrides, and
    a safe dry-run mode for testing without root privileges.

    Args:
        dry_run: Force dry-run mode. If ``None``, auto-detects based on
            root privileges and ``nft`` availability.
        default_lease_seconds: The TTL applied to strict isolation
            (``QUARANTINE_ISOLATE``) actions. Defaults to 300 seconds.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        default_lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        """Initialize the engine, verify permissions, and setup nftables."""
        self.default_lease_seconds = default_lease_seconds
        self._active_blocks: dict[str, dict[str, Any]] = {}

        # Auto-detect privileges if dry_run is not explicitly set
        if dry_run is None:
            is_root = os.geteuid() == 0
            nft_exists = shutil.which("nft") is not None
            self.dry_run = not (is_root and nft_exists)
            if self.dry_run:
                LOGGER.warning(
                    "MitigationEngine operating in DRY-RUN mode (Root: %s, nft available: %s).",
                    is_root, nft_exists,
                )
        else:
            self.dry_run = dry_run
            if not self.dry_run and os.geteuid() != 0:
                LOGGER.warning("dry_run=False requested but not running as root. Forcing dry_run=True.")
                self.dry_run = True

        # Initialize the firewall base rules if running in live mode
        if not self.dry_run:
            self._initialize_firewall()

    def _is_valid_ip(self, src_ip: str) -> bool:
        """Validate that the provided string is a valid IPv4 or IPv6 address."""
        try:
            ipaddress.ip_address(src_ip)
            return True
        except ValueError:
            LOGGER.error("Invalid IP address provided: %s", src_ip)
            return False

    def _run_nft(self, cmd_str: str) -> bool:
        """Execute or log an ``nft`` command safely.

        Args:
            cmd_str: The full ``nft`` command string (excluding the ``nft`` prefix).

        Returns:
            ``True`` if the command succeeded (or if in dry-run), ``False`` otherwise.
        """
        if self.dry_run:
            LOGGER.info("[dry-run] nft %s", cmd_str)
            return True

        try:
            # NOTE: shell=True is safe here because src_ip is strictly validated
            # via ipaddress.ip_address() before reaching this method.
            result = subprocess.run(
                f"nft {cmd_str}",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stderr:
                LOGGER.debug("nft stderr: %s", result.stderr.strip())
            return True
        except subprocess.CalledProcessError as e:
            LOGGER.error("nft command failed: '%s'. stderr: %s", e.cmd, e.stderr.strip())
            return False
        except FileNotFoundError:
            LOGGER.critical("nft binary not found. Disabling live mitigation.")
            self.dry_run = True
            return False

    def _initialize_firewall(self) -> None:
        """Create the dedicated nftables table, chains, set, and drop rules.

        This is idempotent: if the table already exists, ``nft`` will simply
        return an error which is caught and logged at DEBUG level.
        """
        commands = [
            "add table inet idps_filter",
            'add chain inet idps_filter input { type filter hook input priority 0; policy accept; }',
            'add chain inet idps_filter forward { type filter hook forward priority 0; policy accept; }',
            'add set inet idps_filter quarantine_v4 { type ipv4_addr; flags timeout; }',
            'add rule inet idps_filter input ip saddr @quarantine_v4 drop',
            'add rule inet idps_filter forward ip saddr @quarantine_v4 drop',
        ]
        for cmd in commands:
            # We don't fail the whole init if the table already exists
            if not self._run_nft(cmd):
                LOGGER.debug("Firewall init command skipped or failed: %s", cmd)
        LOGGER.info("nftables quarantine hierarchy initialized.")

    def _add_quarantine(self, src_ip: str, lease_seconds: int, reason: str) -> bool:
        """Add an IP to the nftables timed set and update internal tracking."""
        # Escape curly braces for the f-string by doubling them
        cmd = f"add element inet idps_filter quarantine_v4 {{ {src_ip} timeout {lease_seconds}s }}"
        success = self._run_nft(cmd)

        if success:
            now = time.time()
            self._active_blocks[src_ip] = {
                "action": "QUARANTINE",
                "blocked_at": now,
                "expires_at": now + lease_seconds,
                "reason": reason or "Unspecified threat",
            }
        return success

    def apply_action(
        self,
        action: str,
        src_ip: str,
        lease_seconds: int | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Execute a mitigation directive from the RiskEngine.

        Args:
            action: The mitigation directive (``ALLOW``, ``LOG_RATE_LIMIT``,
                ``QUARANTINE_TEMP``, ``QUARANTINE_ISOLATE``).
            src_ip: The source IP address to target.
            lease_seconds: Optional custom TTL override.
            reason: Human-readable threat context for audit logging.

        Returns:
            A structured execution audit dictionary.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if action == "ALLOW":
            if src_ip in self._active_blocks:
                self.release_ip(src_ip)
            return {
                "status": "ALLOWED",
                "src_ip": src_ip,
                "action_applied": "ALLOW",
                "lease_seconds": 0,
                "dry_run": self.dry_run,
                "timestamp": timestamp,
            }

        if not self._is_valid_ip(src_ip):
            return {
                "status": "INVALID_IP",
                "src_ip": src_ip,
                "action_applied": action,
                "lease_seconds": 0,
                "dry_run": self.dry_run,
                "timestamp": timestamp,
            }

        if action == "LOG_RATE_LIMIT":
            LOGGER.warning("Rate limit alert for %s. Reason: %s", src_ip, reason)
            return {
                "status": "RATE_LIMITED",
                "src_ip": src_ip,
                "action_applied": "LOG_RATE_LIMIT",
                "lease_seconds": 0,
                "dry_run": self.dry_run,
                "timestamp": timestamp,
            }

        if action == "QUARANTINE_TEMP":
            lease = 60 if lease_seconds is None else lease_seconds
            success = self._add_quarantine(src_ip, lease, reason)
            status = "QUARANTINED_TEMP" if success else "QUARANTINE_FAILED"
            return {
                "status": status,
                "src_ip": src_ip,
                "action_applied": "QUARANTINE_TEMP",
                "lease_seconds": lease,
                "dry_run": self.dry_run,
                "timestamp": timestamp,
            }

        if action == "QUARANTINE_ISOLATE":
            lease = self.default_lease_seconds if lease_seconds is None else lease_seconds
            success = self._add_quarantine(src_ip, lease, reason)
            status = "QUARANTINED_ISOLATE" if success else "QUARANTINE_FAILED"
            return {
                "status": status,
                "src_ip": src_ip,
                "action_applied": "QUARANTINE_ISOLATE",
                "lease_seconds": lease,
                "dry_run": self.dry_run,
                "timestamp": timestamp,
            }

        # Unknown action fallback
        LOGGER.error("Received unknown mitigation action: %s", action)
        return {
            "status": "UNKNOWN_ACTION",
            "src_ip": src_ip,
            "action_applied": action,
            "lease_seconds": 0,
            "dry_run": self.dry_run,
            "timestamp": timestamp,
        }

    def release_ip(self, src_ip: str) -> bool:
        """Manual operator override to instantly lift isolation for an IP.

        Args:
            src_ip: The IP address to remove from quarantine.

        Returns:
            ``True`` if successfully unblocked, ``False`` otherwise.
        """
        if src_ip not in self._active_blocks:
            LOGGER.warning("Attempted to release IP %s, but it is not actively quarantined.", src_ip)
            return False

        cmd = f"delete element inet idps_filter quarantine_v4 {{ {src_ip} }}"
        success = self._run_nft(cmd)

        if success:
            del self._active_blocks[src_ip]
            LOGGER.info("Successfully released quarantine for %s.", src_ip)
            return True

        LOGGER.error("Failed to release quarantine for %s via nftables.", src_ip)
        return False

    def get_active_quarantine(self) -> list[dict[str, Any]]:
        """Retrieve a list of currently quarantined IPs with live timing metrics.

        Also cleans up local tracking entries whose TTLs have naturally
        expired in the kernel.

        Returns:
            A list of dictionaries containing tracking info for each active block.
        """
        now = time.time()
        active_list = []
        expired_ips = []

        for ip, data in self._active_blocks.items():
            if data["expires_at"] <= now:
                expired_ips.append(ip)
                continue

            elapsed = now - data["blocked_at"]
            remaining = data["expires_at"] - now

            active_list.append({
                "src_ip": ip,
                "blocked_at": datetime.fromtimestamp(data["blocked_at"], timezone.utc).isoformat(),
                "expires_at": datetime.fromtimestamp(data["expires_at"], timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 2),
                "remaining_seconds": round(remaining, 2),
                "reason": data["reason"],
            })

        # Clean up naturally expired local tracking
        for ip in expired_ips:
            LOGGER.info("Lease naturally expired for %s, removing from local tracking.", ip)
            del self._active_blocks[ip]

        return active_list

    def flush_all(self) -> None:
        """Purges all active quarantine elements for clean system teardown.

        Iterates through all locally tracked blocked IPs, issues the delete
        command to ``nftables``, and clears the internal tracking state.
        """
        if not self._active_blocks:
            LOGGER.info("Flush called, but no active quarantines to clear.")
            return

        # Create a list copy to safely delete from dict while iterating
        ips_to_release = list(self._active_blocks.keys())
        for ip in ips_to_release:
            cmd = f"delete element inet idps_filter quarantine_v4 {{ {ip} }}"
            self._run_nft(cmd)

        self._active_blocks.clear()
        LOGGER.info("Flushed all quarantine elements. System teardown complete.")


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """Validate the mitigation engine logic in a safe dry-run environment.

    Instantiates the engine in ``dry_run=True`` mode to simulate firewall
    commands without requiring root privileges or modifying the system.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("=" * 78)
    LOGGER.info("MitigationEngine smoke test — Dry-Run Firewall Simulation")
    LOGGER.info("=" * 78)

    try:
        # Force dry_run=True for safe execution as a standard user
        engine = MitigationEngine(dry_run=True)

        # 1. Simulate applying QUARANTINE_TEMP to 192.168.1.105 (60s lease)
        LOGGER.info("Applying QUARANTINE_TEMP to 192.168.1.105...")
        res1 = engine.apply_action(
            action="QUARANTINE_TEMP",
            src_ip="192.168.1.105",
            reason="High-risk DDoS flow detected"
        )
        assert res1["status"] == "QUARANTINED_TEMP"
        assert res1["lease_seconds"] == 60
        assert res1["dry_run"] is True

        # 2. Simulate applying QUARANTINE_ISOLATE to 10.0.0.99 (300s lease)
        LOGGER.info("Applying QUARANTINE_ISOLATE to 10.0.0.99...")
        res2 = engine.apply_action(
            action="QUARANTINE_ISOLATE",
            src_ip="10.0.0.99",
            reason="Critical Botnet C&C beacon"
        )
        assert res2["status"] == "QUARANTINED_ISOLATE"
        assert res2["lease_seconds"] == 300

        # 3. Check active quarantine tracking
        active = engine.get_active_quarantine()
        assert len(active) == 2, f"Expected 2 active blocks, got {len(active)}"
        assert active[0]["src_ip"] in ["192.168.1.105", "10.0.0.99"]
        LOGGER.info("Verified active quarantine tracking: %d IPs isolated.", len(active))

        # 4. Simulate manual release via release_ip
        LOGGER.info("Simulating manual release of 192.168.1.105...")
        success = engine.release_ip("192.168.1.105")
        assert success is True

        # 5. Validate state table updated correctly
        active_after_release = engine.get_active_quarantine()
        assert len(active_after_release) == 1, f"Expected 1 active block post-release, got {len(active_after_release)}"
        assert active_after_release[0]["src_ip"] == "10.0.0.99"
        LOGGER.info("Verified manual release. Remaining isolated: %d", len(active_after_release))

        # 6. Test ALLOW action clearing remaining state
        LOGGER.info("Applying ALLOW to 10.0.0.99 to clear state...")
        res3 = engine.apply_action(
            action="ALLOW",
            src_ip="10.0.0.99"
        )
        assert res3["status"] == "ALLOWED"
        assert len(engine.get_active_quarantine()) == 0
        LOGGER.info("ALLOW action correctly purged remaining block.")

        # 7. Test invalid IP rejection
        res4 = engine.apply_action(
            action="QUARANTINE_TEMP",
            src_ip="999.999.999.999",
            reason="Invalid test"
        )
        assert res4["status"] == "INVALID_IP"

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — Mitigation logic verified without system impact.")
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())