from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from myrl.finetune.finetune import main  # noqa: E402


if __name__ == "__main__":
    main()

