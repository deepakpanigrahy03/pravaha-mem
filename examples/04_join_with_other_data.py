"""Lesson 4: line up pravaha samples with any other time series.

pravaha timestamps are the system clock in ns (time.time_ns()). Any log or
meter that records the same clock (power, energy, request logs) joins directly.
Here the other series is a simple log of events with timestamps; replace it
with your power meter or application log.

Run after lesson 1:  python examples/04_join_with_other_data.py /tmp/pravaha_lesson1.csv
"""
import csv
import sys

trace = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pravaha_lesson1.csv"
rows = list(csv.DictReader(open(trace)))
t_first, t_last = int(rows[0]["t_start_ns"]), int(rows[-1]["t_end_ns"])

# your other data: (timestamp_ns, value). Here: one made up reading per second.
other = [(t, f"reading_{i}") for i, t in enumerate(range(t_first, t_last, 1_000_000_000))]

# join: for each reading, the C2C bytes in the second before it
print(f"{'time from start (s)':>20} {'other':>12} {'C2C MiB in last second':>24}")
for t, value in other[1:]:
    b = sum(int(r["all_read"]) + int(r["all_write"]) for r in rows
            if t - 1_000_000_000 <= int(r["t_start_ns"]) < t)
    print(f"{(t - t_first) / 1e9:20.1f} {value:>12} {b / 2**20:24.1f}")
