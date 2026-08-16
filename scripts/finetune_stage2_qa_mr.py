#!/usr/bin/env python3
"""Deprecated entrypoint.

Use scripts/finetune_stage2_qa_mr_ukr_mix.py (canonical Stage-2 trainer).
This shim keeps older docs/commands from breaking.
"""

from finetune_stage2_qa_mr_ukr_mix import main


if __name__ == "__main__":
    main()
