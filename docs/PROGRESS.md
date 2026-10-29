# Progress

## Phase 0: Setup and feasibility (in progress)
* Machine: Ryzen 5 7535HS, 7.32 GB RAM (about 2.3 GB free at inspection time), RTX 3050 Laptop 4096 MB (driver 617.42), **Windows 10 Home** (not 11, see D2), 73 GB free.
* Ollama 0.40.12 present with llama3.2, qwen2.5:3b, qwen2.5:7b, nomic-embed-text. Pulled `qwen3:4b` and `moondream`.
* Python: only 3.14 installed; provisioned 3.12.16 with uv (D1).
* Skeleton, config system, logging, event bus, state machine, policy/gate, tools, brain, audio, STT, TTS, overlay, screen, control, evals harness and tests have been written.
* **Blocker encountered:** PyPI's file host (`files.pythonhosted.org`) is delivering about 40 KB/s per connection from this network (measured with curl: 948 KB in 25 s; one of four parallel connections hit 1.7 MB/s, so speed is very uneven). The first two `uv pip install` attempts stalled for 30+ minutes on av, ctranslate2, piper-tts, opencv-python. Mitigation: short timeouts, many retries, parallel downloads (`UV_HTTP_TIMEOUT=15 UV_HTTP_RETRIES=30`). This delays verification of everything that needs the venv.
