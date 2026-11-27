# Contributing

## Report results from your GB10
The most useful contribution: run pravaha on your machine and tell us what happened.
```bash
pravaha status
pravaha verify --json verify.json
python -m pytest -m gpu tests          # optional, from a clone of the repository
pravaha verify --schedule --json verify_schedule.json   # optional: the schedule runner
```
Open an issue titled "Test report: <your machine>" with the `status` output and `verify.json`. We add it to [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md). All of these commands are light (a few GiB of copies and simple writes and reads).

## Report a problem
Include `pravaha status`, the exact command, its full output, and (if a measurement looks wrong) the `--json` result.

## Change the code
* `python -m pytest tests` must pass (no GPU needed); on a GB10 also `python -m pytest -m gpu tests`.
* Keep the rules in [docs/DESIGN.md](docs/DESIGN.md): bytes and translations never mixed; observer only; least privilege; leave no trace; every command and output explains itself; light test workloads only.
* Update the docs and the test matrix in the same change.
