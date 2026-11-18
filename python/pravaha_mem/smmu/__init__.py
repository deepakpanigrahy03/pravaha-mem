"""Address translations of the GPU, from the GB10's SMMU counters (Linux perf PMCG units).

These count translation requests and translation misses, never bytes. They are for
studying how the GPU translates addresses (page sizes, translation cache reach).
Bytes per translation vary with allocation type, page size and history, so they are
not converted to bytes.
"""
