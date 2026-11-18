# pravaha-mem (Author: Deepak Panigrahy). make build | make lib | make test | make smoke | make validate | make clean
NVCC ?= nvcc
ARCH ?= sm_121
BIN  := build/copy_calib build/zc build/zc_pg build/zc_time build/gpu_info build/mixed build/cpu_stream build/pattern
CC   ?= cc
CFLAGS ?= -O2 -std=gnu99 -Wall -Wextra

build: $(BIN)
build/cpu_stream: src/cpu_stream.c | build/.dir
	$(CC) $(CFLAGS) $< -o $@
build/%: src/%.cu | build/.dir
	$(NVCC) -O2 -arch=$(ARCH) $< -o $@

# pravaha C library: lib/libpravaha.a and lib/libpravaha.so
lib: lib/libpravaha.a lib/libpravaha.so
lib/pravaha.o: lib/pravaha.c lib/pravaha.h
	$(CC) $(CFLAGS) -fPIC -c $< -o $@
lib/libpravaha.a: lib/pravaha.o
	ar rcs $@ $<
lib/libpravaha.so: lib/pravaha.o
	$(CC) -shared $< -lm -o $@
build/copy_counter: examples/copy_counter.c lib/libpravaha.a | build/.dir
	$(CC) $(CFLAGS) -Ilib $< lib/libpravaha.a -lm -o $@

# the reader for use without sudo; an administrator then runs, once:
#   sudo install -m 0755 build/pravaha-pmcg /usr/local/bin/ && sudo setcap cap_perfmon=ep /usr/local/bin/pravaha-pmcg
helper: build/pravaha-pmcg
build/pravaha-pmcg: python/pravaha_mem/pravaha_pmcg.c | build/.dir
	$(CC) $(CFLAGS) $< -o $@

build/.dir:
	mkdir -p build && touch $@

# tests that need no GPU (analysis, ncu parsing, PMCG discovery on a fake sysfs)
test:
	@python3 -c "import pytest" 2>/dev/null || { echo "make test needs pytest: pip install '.[test]'"; exit 1; }
	python3 -m pytest -q tests
	bash tests/test_find_pmcg.sh

# quick check on a GB10: tools present, PMCG found, one copy and one ncu kernel counted
smoke: build
	bash tests/smoke.sh

# the full validation (about 15 minutes, needs sudo)
validate: build
	./validate/run_validation.sh

clean:
	rm -rf build lib/*.o lib/*.a lib/*.so

.PHONY: build lib helper test smoke validate clean
