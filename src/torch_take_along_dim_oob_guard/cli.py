"""Command-line interface: run the from-scratch diagnosis of the
torch.take_along_dim(..., dim=<int>) out-of-bounds silent-wrap bug
against the currently installed torch build, using the shared
semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-take-along-dim-oob-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.take_along_dim(input, indices, dim=<int>) silently "
            "wraps out-of-bounds indices modulo the dimension size "
            "instead of raising (pytorch/pytorch#196106), and verify "
            "the safe_take_along_dim() guard raises IndexError on "
            "genuinely out-of-range indices while matching a NumPy "
            "oracle on valid ones. Never trusts a cached or "
            "previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-take-along-dim-oob-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_silently_wrong"]:
        print(status_headline(style, "fail", "torch.take_along_dim(dim=...) silent OOB-wrap reproduced on this host"))
    else:
        print(status_headline(style, "info", "no silent OOB-wrap reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard matches the raise-on-OOB / NumPy-oracle contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one case"))

    section("cases (description -> native vs guard vs oracle)")
    for c in report["cases"]:
        native_flag = "SILENT-WRONG" if c["native_silently_wrong"] else ("raised" if c["native_raised"] else "ok")
        guard_flag = "guard-ok" if c["guard_correct"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["description"][:52],
                    f"native={native_flag:12s}  guard_raised={str(c['guard_raised']):5s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
