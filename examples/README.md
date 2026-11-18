# Examples

Each example: what it does, what it needs, how to run it, and what to expect. All are light (a few GiB of copies and simple reads and writes), except example 3, which uses an LLM server you already run.

| # | file | what it shows | needs | run | expect |
|---|---|---|---|---|---|
| 1 | [01_any_command.sh](01_any_command.sh) | measure any program from outside, with an idle baseline, CSV and JSON | pravaha | `bash examples/01_any_command.sh` (or add your own command after it) | the test program copies 2 GiB into the GPU: run - idle row, copy about 2 GiB |
| 2 | [02_python_phases.py](02_python_phases.py) | use pravaha inside your own Python code, with three named steps (phases) | PyTorch with CUDA | `python examples/02_python_phases.py` | load: copy about 1 GiB; compute: kread about 10 GiB; save: copy about 0.4 GiB (see below) |
| 3 | [03_llm_bytes_per_token.py](03_llm_bytes_per_token.py) | how many bytes an LLM server reads per generated token, observed from outside | a running OpenAI compatible server (for example vLLM) | `python examples/03_llm_bytes_per_token.py --tokens 400` | kernel reads per token about the size of the model's weights |
| 4 | [04_join_with_other_data.py](04_join_with_other_data.py) | line up pravaha's samples with other data by time (power, energy, logs) | the CSV from example 1 | `python examples/04_join_with_other_data.py /tmp/pravaha_lesson1.csv` | C2C MiB per second next to each reading |
| t | [known_bytes.py](../python/pravaha_mem/known_bytes.py) | the test program: moves an exact amount of data | CUDA | `python -m pravaha_mem.known_bytes --help` | see its help: direction, times, size |

## Example 2 in words
PyTorch is the most widely used Python library for AI; vLLM and most LLM code are built on it. The example creates 1 GiB of numbers in the computer's memory and then, while pravaha observes:

1. **load**: moves the 1 GiB into the GPU (`.cuda()`): copy engines, so the **copy** column.
2. **compute**: the GPU adds up all the numbers ten times; the 1 GiB cannot fit in the GPU's L2 cache, so each pass reads it from memory: about 10 GiB in **kread**.
3. **save**: moves a quarter (0.25 GiB) back to the computer's memory (`.cpu()`). PyTorch copies into new, pageable memory through a staging step, so pravaha sees about 0.4 GiB of copies: what the copy engines really moved.

It samples every 1 ms because these steps last only milliseconds.

## Example 3 in words
Start any OpenAI compatible LLM server. The example asks it for one answer of N tokens while pravaha observes the whole GPU, then divides the traffic by N. During decoding a model reads all its weights once per token, so kernel reads per token come out near the weight size (7.52 GiB per token for Qwen3 4B, whose weights are 7.49 GiB).
