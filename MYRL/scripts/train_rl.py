from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

if __name__ == "__main__":
    from myrl.rl.train_rl import main  # noqa: E402

    main()
