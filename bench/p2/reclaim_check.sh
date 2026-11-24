#!/bin/bash
# Is the kernel reclaiming memory? Run it before measuring, and again once the engine has loaded: twice we saw the
# main configuration decode 10-20 times slower; in the episode we monitored, kswapd reclaimed pages during every slow
# request (results/pagecache). Prints the pages kswapd reclaimed over N seconds (default 5) and exits 1 if it reclaimed any.
n=${1:-5}
a=$(awk '/^pgsteal_kswapd / {print $2}' /proc/vmstat)
sleep "$n"
b=$(awk '/^pgsteal_kswapd / {print $2}' /proc/vmstat)
free=$(awk '/^MemFree:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
d=$(( b - a ))
if [ "$d" -gt 0 ]; then
  r=$(awk -v d="$d" -v n="$n" 'BEGIN {printf "%.1f", d / n}')
  echo "kswapd reclaimed $d pages in ${n} s ($r pages/s; MemFree $free GiB): wait until it stops, or drop the page cache, before measuring"
  exit 1
fi
echo "kswapd idle over ${n} s (MemFree $free GiB)"
