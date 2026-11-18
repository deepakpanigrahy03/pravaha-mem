#!/usr/bin/env bash
# Lesson 1: measure any program from outside. Nothing in the program changes.
# Usage: bash examples/01_any_command.sh [command ...]
#   default: pravaha's test program known_bytes: h2d = host to device (computer memory into the GPU),
#   8 times 256 MiB = 2 GiB, moved by the copy engines

if [ $# -eq 0 ]; then
  set -- python3 -m pravaha_mem.known_bytes h2d 8 256
fi

# 2 s idle baseline first, then the command; one row per 10 ms sample in the CSV
pravaha run --idle 2 --csv /tmp/pravaha_lesson1.csv --json /tmp/pravaha_lesson1.json -- "$@"

echo
echo "CSV  : /tmp/pravaha_lesson1.csv  (first rows below)"
head -3 /tmp/pravaha_lesson1.csv
echo "JSON : /tmp/pravaha_lesson1.json"
