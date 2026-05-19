# Notice

Open Quant Agent is a standalone public extraction and refactor of multi-agent quant research components originally prototyped inside the local `autotrade` project.

The public repository intentionally excludes:

- local experiment JSONL files,
- local checkpoints,
- generated candidate logs,
- LaunchAgent files with machine-specific paths,
- private environment files,
- API provider configuration,
- broker or exchange integration code.

This project keeps the research architecture, factor validation ideas, portfolio strategy promotion layer, and offline synthetic-data path, while replacing project-specific imports with standalone modules.
