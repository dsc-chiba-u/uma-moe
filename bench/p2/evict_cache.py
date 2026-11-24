"""Drop the page cache of every regular file of 16 MB or more under the given directories (posix_fadvise DONTNEED),
the unprivileged stand-in for writing to /proc/sys/vm/drop_caches. Pages mapped by a running process stay.

    python bench/p2/evict_cache.py DIR...
"""
import os
import sys


def cached_gib():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("Cached:")) / 2**20


before = cached_gib()
n = 0
for root in sys.argv[1:]:
    for d, _, files in os.walk(os.path.expanduser(root)):
        for f in files:
            p = os.path.join(d, f)
            try:
                if os.path.islink(p) or os.path.getsize(p) < 16 << 20:
                    continue
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
print(f"evicted {n} files: Cached {before:.1f} -> {cached_gib():.1f} GiB")
