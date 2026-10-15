#!/usr/bin/env python3
"""Get deleted ROS bags back from an exFAT drive whose files were deleted (not formatted), reading the drive only.

    sudo python3 exfat_bag_rescue.py scan    [DEVICE] [--survey]
    sudo python3 exfat_bag_rescue.py recover [DEVICE] --out DIR [--only A-1,A-2 | --all] [--carve]
    python3 exfat_bag_rescue.py check  BAG
    python3 exfat_bag_rescue.py repair BAG --out FIXED.bag

DEVICE is the exFAT partition, by default /dev/disk/by-label/ExFAT. It is opened read-only and must not be mounted
read-write: whatever is written to it can land on the deleted data (unmount it: udisksctl unmount -b DEVICE).
DIR must be on another disk, with room for the bags. Only the Python standard library is used.

exFAT keeps a deleted file's directory entry (name, first cluster, size, contiguous or not) until its slot is used
again, and the file's data until its clusters are written again (an SSD also loses them to TRIM). So:

  scan     walks every directory, the deleted ones too, lists the bags in them and checks each against its own
           index (bag header, every chunk's header, first and last message, index records): whole, damaged where
           (recording time spans), or gone. --survey also reads all free space for bag chunks, which finds the
           pieces of bags whose directory entries are gone too; it takes as long as reading the drive.
  recover  copies each whole bag byte for byte, and rebuilds a damaged one from its intact chunks (a valid bag with
           gaps). --carve also rebuilds, from the chunks in free space, the chosen recordings that no directory
           entry leads to. Without --only/--all it takes the A-course bags deleted on 2026-10-11.
  check    the same check for a bag file.
  repair   rebuilds a bag file from its intact chunks (index records and the index are made anew).

A recording is named by its directory entry, else by its messages' time: KNOWN below lists the recordings deleted
from the ExFAT drive on 2026-10-11 (name -> seconds recorded); other times give carved_<start time>.bag.
"""
import argparse
import bisect
import calendar
import json
import os
import re
import shutil
import struct
import sys
import time

MAGIC = b"#ROSBAG V2.0\n"
OP_MSG, OP_HEADER, OP_INDEX, OP_CHUNK, OP_INFO, OP_CONN = 2, 3, 4, 5, 6, 7
HEADER_LEN = 4096                   # a bag's header record (fields + padding) after the magic line
MAX_CHUNK_SPAN = 120 * 10**9        # a T-Car chunk holds milliseconds of recording
ZERO_STEP = 64                      # free space checked for zeros (TRIM) every 64th cluster
KST = 9 * 3600

KNOWN = {
    "A-1_2026-09-28-14-40-58": 252, "A-2_2026-09-28-14-45-11": 205, "A-4_2026-09-28-14-51-38": 238,
    "A-5_2026-09-28-14-55-45": 295, "A-6_2026-09-28-15-02-17": 359, "A-7_2026-09-28-15-08-17": 283,
    "A-8_2026-09-28-15-13-01": 366, "A-9_2026-09-28-15-19-08": 309,
    "A-3_2026-09-28-16-32-23": 288, "A-10_2026-09-28-16-03-14": 474,
    "D-3_2026-09-29-17-20-32": 756,
    "F-1_2026-10-06-15-36-52": 428, "F-2_2026-10-06-15-51-37": 383, "F-3_2026-10-06-16-05-27": 116,
    "F-4_2026-10-06-16-10-18": 500, "F-5_2026-10-06-16-18-39": 393,
    "G-1_2026-10-12-16-21-54": 215, "G-2_2026-10-12-16-26-33": 153, "G-3_2026-10-12-16-29-07": 173,
}
WANTED = ["A-1_", "A-2_", "A-4_", "A-5_", "A-6_", "A-7_", "A-8_", "A-9_"]


def u16(b, i=0):
    return struct.unpack_from("<H", b, i)[0]


def u32(b, i=0):
    return struct.unpack_from("<I", b, i)[0]


def u64(b, i=0):
    return struct.unpack_from("<Q", b, i)[0]


def say(*a):
    print(*a, flush=True)


def stamp(v):
    """8-byte ROS time -> ns, None when it is not a plausible time (2020-2035)."""
    if len(v) != 8:
        return None
    s, ns = struct.unpack("<II", v)
    return s * 10**9 + ns if ns < 10**9 and 1577836800 <= s <= 2051222400 else None


def pack_time(ns):
    return struct.pack("<II", ns // 10**9, ns % 10**9)


def kst(ns):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ns // 10**9 + KST)) if ns else "-"


def name_start(name):
    """Start of a recording from its name (local time, KST) in ns."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", name)
    return (calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")) - KST) * 10**9 if m else None


_STARTS = sorted((name_start(n), n, d) for n, d in KNOWN.items())


def recording_of(ns):
    """The known recording a time belongs to: the last one started by then, if it still ran."""
    best = None
    for t0, name, dur in _STARTS:
        if t0 - 10 * 10**9 <= ns <= t0 + (dur + 60) * 10**9:
            best = name
    return best


def gb(n):
    return f"{n / 1e9:.1f} GB"


class Progress:
    def __init__(self, what, total):
        self.what, self.total, self.t0, self.last = what, total, time.time(), 0

    def __call__(self, done, extra=""):
        now = time.time()
        if now - self.last < 2 and done < self.total:
            return
        self.last = now
        rate = done / max(now - self.t0, 1e-6)
        eta = (self.total - done) / rate if rate else 0
        sys.stderr.write(f"\r  {self.what}: {gb(done)} / {gb(self.total)}  {rate / 1e6:.0f} MB/s  "
                         f"남은 시간 {eta / 60:.0f}분 {extra}    ")
        sys.stderr.flush()

    def end(self):
        sys.stderr.write("\n")


# ------------------------------------------------------------------ the device, read only
class Disk:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.lseek(self.fd, 0, os.SEEK_END)
        self.errors = 0
        self.hook = None                                      # hook(offset, bytes) sees everything read

    def read(self, off, n):
        out = []
        while n > 0:
            try:
                b = os.pread(self.fd, min(n, 1 << 26), off)
            except OSError:
                b = self._salvage(off, min(n, 1 << 20))
            if not b:
                break
            if self.hook:
                self.hook(off, b)
            out.append(b)
            off += len(b)
            n -= len(b)
        return b"".join(out)

    def _salvage(self, off, n):
        """A read that failed: 4 KB at a time, zeros where the drive cannot read."""
        out = []
        for k in range(0, n, 4096):
            try:
                out.append(os.pread(self.fd, min(4096, n - k), off + k))
            except OSError:
                self.errors += 1
                if self.errors <= 10:
                    say(f"  [!] 읽기 오류 (offset {off + k}), 0으로 채움")
                out.append(bytes(min(4096, n - k)))
        return b"".join(out)


class File:
    """A regular file, with the same read(off, n) / size as an Extent."""
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size

    def read(self, off, n):
        n = max(0, min(n, self.size - off))
        out = []
        while n > 0:
            b = os.pread(self.fd, min(n, 1 << 26), off)
            if not b:
                break
            out.append(b)
            off += len(b)
            n -= len(b)
        return b"".join(out)


class Extent:
    """A file's bytes on the device: its cluster runs [(first cluster, count)], size bytes."""
    def __init__(self, fs, runs, size):
        self.fs, self.size, self.starts, self.devs = fs, size, [], []
        pos = 0
        for c, n in runs:
            self.starts.append(pos)
            self.devs.append((fs.off(c), n * fs.cs))
            pos += n * fs.cs
        self.mapped = pos

    def read(self, off, n):
        n = max(0, min(n, self.size - off, self.mapped - off))
        out = []
        while n > 0:
            k = bisect.bisect_right(self.starts, off) - 1
            dev, length = self.devs[k]
            inner = off - self.starts[k]
            m = min(n, length - inner)
            out.append(self.fs.d.read(dev + inner, m))
            off += m
            n -= m
        return b"".join(out)


# ------------------------------------------------------------------ exFAT
_BITS = ["".join("1" if x >> k & 1 else "0" for k in range(8)) for x in range(256)]


def entry_checksum(s):
    c = 0
    for k, x in enumerate(s):
        if k not in (2, 3):
            c = ((((c & 1) << 15) | (c >> 1)) + x) & 0xFFFF
    return c


class ExFat:
    def __init__(self, disk):
        self.d = disk
        b = disk.read(0, 512)
        if len(b) < 512 or b[3:11] != b"EXFAT   ":
            sys.exit(f"{disk.path}: exFAT이 아닙니다 (boot sector)")
        bps = 1 << b[0x6C]
        self.cs = bps << b[0x6D]
        self.ncl = u32(b, 0x5C)
        flags = u16(b, 0x6A)
        self.active = 1 if b[0x6E] == 2 and flags & 1 else 0
        self.fat_off = (u32(b, 0x50) + self.active * u32(b, 0x54)) * bps
        self.heap = u32(b, 0x58) * bps
        self.root = u32(b, 0x60)
        self.dirty = bool(flags & 2)
        self._fat = {}
        self.bits = self._bitmap()

    def off(self, c):
        return self.heap + (c - 2) * self.cs

    def valid(self, c):
        return 2 <= c < self.ncl + 2

    def next(self, c):
        page = c >> 16
        t = self._fat.get(page)
        if t is None:
            t = self._fat[page] = self.d.read(self.fat_off + (page << 18), 1 << 18)
        i = (c & 0xFFFF) * 4
        return u32(t, i) if i + 4 <= len(t) else 0xFFFFFFFF

    def runs(self, first, nofat, size=None):
        """A file's clusters as runs: contiguous ones, or the FAT chain (size bytes; None: to the chain's end)."""
        if not self.valid(first):
            return []
        want = -(-size // self.cs) if size else None
        if nofat:
            n = min(want or 1, self.ncl + 2 - first)
            return [(first, n)]
        out, c, seen, n = [], first, set(), 0
        while self.valid(c) and c not in seen and (want is None or n < want):
            seen.add(c)
            if out and out[-1][0] + out[-1][1] == c:
                out[-1] = (out[-1][0], out[-1][1] + 1)
            else:
                out.append((c, 1))
            n += 1
            c = self.next(c)
        return out

    def stale_runs(self, first, limit=1 << 22):
        """The free clusters linked from first in the FAT: a deleted fragmented file's chain (the FAT keeps it
        until its clusters are used again)."""
        out, c, seen = [], first, set()
        while self.valid(c) and not self.allocated(c) and c not in seen and len(seen) < limit:
            seen.add(c)
            if out and out[-1][0] + out[-1][1] == c:
                out[-1] = (out[-1][0], out[-1][1] + 1)
            else:
                out.append((c, 1))
            c = self.next(c)
        return out

    def read_runs(self, runs, limit=64 << 20):
        out, n = [], 0
        for c, k in runs:
            m = min(k * self.cs, limit - n)
            if m <= 0:
                break
            out.append(self.d.read(self.off(c), m))
            n += m
        return b"".join(out)

    def _bitmap(self):
        root = self.read_runs(self.runs(self.root, False), 4 << 20)
        for i in range(0, len(root) - 31, 32):
            if root[i] == 0x81 and (root[i + 1] & 1) == self.active:
                first, length = u32(root, i + 20), u64(root, i + 24)
                runs = self.runs(first, False, length)
                data = self.read_runs(runs, length) if sum(k for _, k in runs) * self.cs >= length else b""
                if len(data) < length:                        # no usable FAT chain: contiguous
                    data = self.d.read(self.off(first), length)
                return "".join(_BITS[x] for x in data)[:self.ncl]
        sys.exit("allocation bitmap을 찾지 못했습니다")

    def allocated(self, c):
        return self.bits[c - 2] == "1"

    def used(self, runs):
        """How many of these clusters are in use now (written again, or about to be)."""
        return sum(self.bits[c - 2:c - 2 + k].count("1") for c, k in runs)

    def free_runs(self):
        return [(m.start() + 2, m.end() - m.start()) for m in re.finditer("0+", self.bits)]

    def entries(self, data):
        """File entry sets in directory bytes, deleted ones too."""
        i, n = 0, len(data)
        while i + 96 <= n:
            t = data[i]
            sc = data[i + 1]
            end = i + 32 * (sc + 1)
            if t & 0x7F != 0x05 or not 2 <= sc <= 18 or end > n or data[i + 32] & 0x7F != 0x40:
                i += 32
                continue
            st = i + 32
            nl = data[st + 3]
            chars = b"".join(data[k + 2:k + 32] for k in range(st + 32, end, 32) if data[k] & 0x7F == 0x41)
            try:
                name = chars[:2 * nl].decode("utf-16-le") if nl and len(chars) >= 2 * nl else None
            except UnicodeDecodeError:
                name = None
            if not name:
                i += 32
                continue
            s = bytearray(data[i:end])
            for k in range(0, len(s), 32):
                s[k] |= 0x80                                  # as it was before deletion
            yield {"name": name, "attr": u16(data, i + 4), "first": u32(data, st + 20), "size": u64(data, st + 24),
                   "nofat": bool(data[st + 1] & 2), "deleted": not t & 0x80,
                   "intact": entry_checksum(s) == u16(data, i + 2)}
            i = end

    def walk(self):
        """Every file entry, in live and in deleted folders: dicts with path."""
        todo = [("", self.runs(self.root, False), False)]
        seen = {self.root}
        while todo:
            path, runs, gone = todo.pop(0)
            for e in self.entries(self.read_runs(runs)):
                e["path"] = f"{path}/{e['name']}"
                e["gone"] = gone or e["deleted"]
                yield e
                c = e["first"]
                if e["attr"] & 0x10 and e["intact"] and self.valid(c) and c not in seen:
                    seen.add(c)
                    if e["gone"] and self.allocated(c):
                        continue                              # the deleted folder's cluster holds something else now
                    todo.append((e["path"], self.runs(c, e["nofat"], min(e["size"] or self.cs, 64 << 20)),
                                 e["gone"]))


# ------------------------------------------------------------------ ROS bag records
class Rec:
    __slots__ = ("op", "fields", "hl", "dl")

    @property
    def total(self):
        return 8 + self.hl + self.dl


def parse(buf, i=0):
    """The record at buf[i:] when its header parses (fields k=v, an op), else None. Data may lie beyond buf."""
    if i + 8 > len(buf):
        return None
    hl = u32(buf, i)
    if not 6 <= hl <= 4096 or i + 8 + hl > len(buf):
        return None
    f, j, end = {}, i + 4, i + 4 + hl
    while j < end:
        if j + 4 > end:
            return None
        n = u32(buf, j)
        if not 2 <= n <= end - j - 4:
            return None
        k, eq, v = buf[j + 4:j + 4 + n].partition(b"=")
        if not eq or not k or len(k) > 20:
            return None
        f[k] = v
        j += 4 + n
    op = f.get(b"op")
    if op is None or len(op) != 1 or not OP_MSG <= op[0] <= OP_CONN:
        return None
    r = Rec()
    r.op, r.fields, r.hl, r.dl = op[0], f, hl, u32(buf, end)
    need = {OP_MSG: (b"conn", b"time"), OP_HEADER: (b"index_pos", b"conn_count", b"chunk_count"),
            OP_INDEX: (b"conn", b"count"), OP_CHUNK: (b"compression", b"size"),
            OP_INFO: (b"chunk_pos", b"start_time", b"end_time", b"count"), OP_CONN: (b"conn", b"topic")}[r.op]
    if any(k not in f for k in need):
        return None
    return r


def field(k, v):
    return struct.pack("<I", len(k) + 1 + len(v)) + k + b"=" + v


def record(fields, data):
    h = b"".join(field(k, v) for k, v in fields)
    return struct.pack("<I", len(h)) + h + struct.pack("<I", len(data)) + data


class Chunk:
    """An intact chunk: its record as found, the index records made from its messages, its times and counts."""
    __slots__ = ("pos", "raw", "index", "entries", "start", "end", "counts", "conns")


def chunk_contents(data):
    """Messages per connection [(time, offset)] and connection records in uncompressed chunk data; None if the
    records do not fill it exactly."""
    i, n, msgs, conns = 0, len(data), {}, {}
    while i < n:
        r = parse(data, i)
        if r is None or i + r.total > n:
            return None
        if r.op == OP_MSG:
            t = stamp(r.fields[b"time"])
            if t is None or len(r.fields[b"conn"]) != 4:
                return None
            msgs.setdefault(u32(r.fields[b"conn"]), []).append((t, i))
        elif r.op == OP_CONN:
            conns[u32(r.fields[b"conn"])] = data[i:i + r.total]
        else:
            return None
        i += r.total
    return (msgs, conns) if msgs else None


def make_chunk(pos, raw, hl):
    data = raw[8 + hl:]
    got = chunk_contents(data)
    if got is None:
        return None
    msgs, conns = got
    ch = Chunk()
    ch.pos, ch.raw, ch.conns = pos, raw, conns
    ch.counts = {c: len(v) for c, v in msgs.items()}
    ch.start = min(t for v in msgs.values() for t, _ in v)
    ch.end = max(t for v in msgs.values() for t, _ in v)
    if ch.end - ch.start > MAX_CHUNK_SPAN:
        return None             # pieces of two bags that happen to line up (scattered clusters of similar bags)
    ch.entries = {c: sorted(v) for c, v in msgs.items()}
    ch.index = b"".join(record([(b"conn", struct.pack("<I", c)), (b"count", struct.pack("<I", len(v))),
                                (b"op", bytes([OP_INDEX])), (b"ver", struct.pack("<I", 1))],
                               b"".join(pack_time(t) + struct.pack("<I", o) for t, o in v))
                        for c, v in sorted(msgs.items()))
    return ch


def record_at(rd, pos, end):
    """(kind, length, item) for an intact record at pos, None if there is none. A chunk's length takes in the index
    records that follow it."""
    b = rd.read(pos, 4096 + 16)
    if b.startswith(MAGIC):
        return "magic", len(MAGIC), None
    r = parse(b)
    if r is None or pos + r.total > end:
        return None
    count = r.fields.get(b"count", b"")
    if (r.op == OP_INDEX and (len(count) != 4 or r.dl != 12 * u32(count))
            or r.op == OP_INFO and (len(count) != 4 or r.dl != 8 * u32(count))
            or r.op == OP_HEADER and r.dl > HEADER_LEN or r.op == OP_MSG and r.dl > 1 << 27):
        return None
    if r.op == OP_CHUNK:
        size = r.fields[b"size"]
        if r.fields[b"compression"] != b"none" or len(size) != 4 or u32(size) != r.dl or r.dl > 1 << 30:
            return None
        raw = rd.read(pos, r.total)
        ch = make_chunk(pos, raw, r.hl) if len(raw) == r.total else None
        if ch is None:
            return None
        q, found = pos + r.total, {}
        blk, at = b"", q
        while q < end:                                        # the index records written after it
            if q + 512 > at + len(blk):
                blk, at = rd.read(q, min(1 << 16, end - q)), q
            r2 = parse(blk, q - at)
            if r2 is None or r2.op != OP_INDEX or q + r2.total > end or len(r2.fields[b"conn"]) != 4:
                break
            if q - at + r2.total > len(blk):
                blk, at = rd.read(q, max(1 << 16, r2.total)), q
            body = blk[q - at + 8 + r2.hl:q - at + r2.total]
            found[u32(r2.fields[b"conn"])] = sorted((stamp(body[k:k + 8]), u32(body, k + 8))
                                                    for k in range(0, len(body) - 11, 12))
            q += r2.total
        if found != ch.entries:
            return None             # its index records are gone (so may be its end) or disagree (two files' bytes)
        return "chunk", q - pos, ch
    if r.op == OP_CONN:
        raw = rd.read(pos, r.total)
        if len(raw) != r.total or len(r.fields[b"conn"]) != 4 or b"type=" not in raw:
            return None
        return "conn", r.total, (u32(r.fields[b"conn"]), raw)
    if r.op in (OP_MSG, OP_INDEX):
        return None             # only inside or right after a chunk: here they are the remains of a broken one
    return {OP_HEADER: "header", OP_INFO: "info"}[r.op], r.total, None


SIGS = (b"compression=", b"op=\x07", MAGIC)


def resync(rd, pos, end, block=16 << 20):
    """The next place at or after pos where an intact chunk, connection record or bag starts (end if none)."""
    while pos < end:
        b = rd.read(pos, min(block, end - pos) + 512)
        cands = set()
        for sig in SIGS:
            q = b.find(sig)
            while 0 <= q < block:
                if sig == MAGIC:
                    cands.add(q)
                else:                                         # the record starts a little before its field
                    for s in range(max(0, q - 4 - 4 - 300), q - 7):
                        hl = u32(b, s)
                        if s + 4 + hl > q and hl <= 4096 and 6 <= hl:
                            cands.add(s)
                q = b.find(sig, q + 1)
        for s in sorted(cands):
            if record_at(rd, pos + s, end) is not None and (b[s:s + 13] == MAGIC or parse(b, s) is not None):
                return pos + s
        pos += block
    return end


def walk(rd, start=0, end=None, progress=None, base=0):
    """What is intact in rd[start:end], in order: ("chunk", Chunk) and ("conn", id, raw); ("gap", a, b) for the
    bytes that are not."""
    end = rd.size if end is None else end
    pos = start
    while pos < end:
        got = record_at(rd, pos, end)
        if got is None:
            nxt = resync(rd, pos + 1, end)
            yield "gap", pos, nxt
            pos = nxt
        else:
            kind, length, item = got
            if kind == "chunk":
                yield "chunk", item
            elif kind == "conn":
                yield ("conn",) + item
            elif kind == "magic":
                yield "magic", pos
            pos += length
        if progress:
            progress(base + pos - start)


class BagWriter:
    """A new bag: chunks as found, index records made anew, then the index."""
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w+b")
        self.f.write(MAGIC + self._header(0, 0, 0))
        self.infos, self.conns, self.bytes = {}, {}, 0      # chunk key -> [pos, length, trusted, start, end,
        #                                                       counts]; conn id -> (record, trusted)

    @staticmethod
    def _header(index_pos, n_conn, n_chunk):
        h = b"".join(field(k, v) for k, v in [(b"chunk_count", struct.pack("<I", n_chunk)),
                                                (b"conn_count", struct.pack("<I", n_conn)),
                                                (b"index_pos", struct.pack("<Q", index_pos)), (b"op", b"\x03")])
        return struct.pack("<I", len(h)) + h + struct.pack("<I", HEADER_LEN - len(h)) + b" " * (HEADER_LEN - len(h))

    def add_conn(self, cid, raw, trusted=False):
        if cid not in self.conns or trusted and not self.conns[cid][1]:
            self.conns[cid] = (raw, trusted)

    def add_chunk(self, ch, trusted=False):
        """Adds a chunk unless it is there already. A trusted one (read along its file's own clusters) takes the
        place of a copy read otherwise that differs from it."""
        key = (ch.start, ch.end, tuple(sorted(ch.counts.items())))
        old = self.infos.get(key)
        if old is not None:
            if not trusted or old[2]:
                return False
            self.f.seek(old[0])
            if self.f.read(old[1]) == ch.raw:
                old[2] = True
                return False
        self.f.seek(0, os.SEEK_END)
        pos = self.f.tell()
        self.f.write(ch.raw)
        self.f.write(ch.index)
        self.infos[key] = [pos, len(ch.raw), trusted, ch.start, ch.end, ch.counts]
        self.bytes += len(ch.raw)
        for cid, raw in ch.conns.items():
            self.add_conn(cid, raw, trusted)
        return old is None

    def drop_untrusted(self, t0, t1):
        """Leaves out the chunks between t0 and t1 that were not read along the file's own clusters."""
        for key in [k for k, v in self.infos.items() if not v[2] and t0 <= v[3] <= t1]:
            del self.infos[key]

    def close(self):
        """Writes the index; returns the connections used without a connection record."""
        self.f.seek(0, os.SEEK_END)
        index_pos = self.f.tell()
        used = set()
        for cid in sorted(self.conns):
            self.f.write(self.conns[cid][0])
        for pos, _, _, t0, t1, counts in sorted(self.infos.values(), key=lambda v: v[3]):
            used.update(counts)
            self.f.write(record([(b"chunk_pos", struct.pack("<Q", pos)), (b"count", struct.pack("<I", len(counts))),
                                 (b"end_time", pack_time(t1)), (b"op", bytes([OP_INFO])),
                                 (b"start_time", pack_time(t0)), (b"ver", struct.pack("<I", 1))],
                                b"".join(struct.pack("<II", c, k) for c, k in sorted(counts.items()))))
        self.f.seek(len(MAGIC))
        self.f.write(self._header(index_pos, len(self.conns), len(self.infos)))
        self.f.close()
        own(self.path)
        return sorted(used - set(self.conns))


# ------------------------------------------------------------------ checking a bag against its index
def read_index(rd, ip, n_conn=None, n_chunk=None):
    """Connections {id: raw} and chunk infos [(pos, start, end, {conn: count})] from the index at ip; None if it
    does not parse to the end of the bag."""
    if not 0 < ip < rd.size or rd.size - ip > 1 << 30:
        return None
    return parse_index(rd.read(ip, rd.size - ip), n_conn, n_chunk)


def parse_index(b, n_conn=None, n_chunk=None):
    i, conns, infos = 0, {}, []
    while i < len(b):
        r = parse(b, i)
        if r is None or i + r.total > len(b):
            return None
        data = b[i + 8 + r.hl:i + r.total]
        if r.op == OP_CONN and not infos:
            conns[u32(r.fields[b"conn"])] = b[i:i + r.total]
        elif r.op == OP_INFO:
            t0, t1 = stamp(r.fields[b"start_time"]), stamp(r.fields[b"end_time"])
            if t0 is None or t1 is None or len(data) != 8 * u32(r.fields[b"count"]):
                return None
            infos.append((u64(r.fields[b"chunk_pos"]), t0, t1,
                          {u32(data, k): u32(data, k + 4) for k in range(0, len(data), 8)}))
        else:
            return None
        i += r.total
    if not infos or (n_conn is not None and (len(conns), len(infos)) != (n_conn, n_chunk)):
        return None
    return conns, infos


def find_index(rd, window=128 << 20):
    """The index when the bag header that points at it is lost: where in the bag's tail the records parse to
    its end."""
    lo = max(0, rd.size - window)
    b = rd.read(lo, rd.size - lo)
    q = b.find(b"op=\x07")
    while q >= 0:
        for s in range(max(0, q - 4 - 4 - 300), q - 7):
            hl = u32(b, s)
            if 6 <= hl <= 4096 and s + 4 + hl > q:
                r = parse(b, s)
                if r is not None and r.op == OP_CONN:
                    got = parse_index(b[s:])
                    if got:
                        return lo + s, got
        q = b.find(b"op=\x07", q + 1)
    return None


def chunk_ok(rd, pos, t0, t1, counts):
    """True if the chunk at pos is the one the index describes (header, first and last message, index records);
    "zero" if the place reads as zeros."""
    b = rd.read(pos, 8192)
    r = parse(b)
    if r is None or r.op != OP_CHUNK:
        return "zero" if b and not b.strip(b"\0") else False
    m = parse(b, 8 + r.hl)
    if m is None or m.op not in (OP_MSG, OP_CONN):
        return False
    if m.op == OP_MSG and not t0 - 10**9 <= (stamp(m.fields[b"time"]) or 0) <= t1 + 10**9:
        return False
    blk = rd.read(pos + r.total, sum(80 + 12 * n for n in counts.values()))   # its index records, in one read
    q, last = 0, None
    for _ in counts:
        r2 = parse(blk, q)
        if r2 is None or r2.op != OP_INDEX or q + r2.total > len(blk):
            return False
        c, n = u32(r2.fields[b"conn"]), u32(r2.fields[b"count"])
        if counts.get(c) != n or r2.dl != 12 * n:
            return False
        tail = blk[q + r2.total - 12:q + r2.total]
        if last is None or u32(tail, 8) > last[1]:
            last = (stamp(tail[:8]), u32(tail, 8))
        q += r2.total
    if last:                                                  # the chunk's last message is where its index says
        m = parse(rd.read(pos + 8 + r.hl + last[1], 512))
        if m is None or m.op != OP_MSG or stamp(m.fields[b"time"]) != last[0]:
            return False
    return True


def check(rd):
    """What is left of a bag, from its own index."""
    r = {"size": rd.size, "header": False, "index": False, "chunks": 0, "bad": 0, "zero": 0, "start": None,
         "end": None, "lost": [], "conns": 0, "ok": False}
    head = rd.read(0, len(MAGIC) + 8 + HEADER_LEN)
    got = None
    if head.startswith(MAGIC):
        h = parse(head, len(MAGIC))
        if h is not None and h.op == OP_HEADER:
            r["header"] = True
            got = read_index(rd, u64(h.fields[b"index_pos"]), u32(h.fields[b"conn_count"]),
                             u32(h.fields[b"chunk_count"]))
    elif head and not head.strip(b"\0"):
        r["zero_head"] = True
    if got is None:
        found = find_index(rd)
        got = found[1] if found else None
    if got is None:
        r["why"] = "인덱스를 찾지 못함 (repair로 chunk를 직접 모아야 함)"
        return r
    conns, infos = got
    r.update(index=True, conns=len(conns), chunks=len(infos), start=min(i[1] for i in infos),
             end=max(i[2] for i in infos))
    lost = []
    for pos, t0, t1, counts in sorted(infos):
        ok = chunk_ok(rd, pos, t0, t1, counts)
        if ok is not True:
            r["bad"] += 1
            r["zero"] += ok == "zero"
            lost.append([t0, t1])
    lost.sort()
    for a, b in lost:                                         # merge into time spans
        if r["lost"] and a <= r["lost"][-1][1] + 2 * 10**9:
            r["lost"][-1][1] = max(r["lost"][-1][1], b)
        else:
            r["lost"].append([a, b])
    r["ok"] = r["header"] and r["bad"] == 0
    return r


def describe(r):
    if not r.get("index"):
        return r.get("why", "bag이 아님")
    span = (r["end"] - r["start"]) / 1e9 if r["start"] else 0
    if r["ok"]:
        return f"온전함 · chunk {r['chunks']:,}개 · {span:.0f}초 ({kst(r['start'])}~)"
    lost_s = sum(b - a for a, b in r["lost"]) / 1e9
    s = (f"손상 · chunk {r['chunks']:,}개 중 {r['bad']:,}개 깨짐 (약 {lost_s:.0f}초 / {span:.0f}초 잃음)"
         + ("" if r["header"] else " · bag 헤더 없음"))
    if r["zero"]:
        s += f" · 그중 {r['zero']:,}개는 0으로 지워짐 (TRIM?)"
    return s


# ------------------------------------------------------------------ commands
def own(path):
    """Files written under sudo go to the user who ran it."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


def open_device(dev):
    real = os.path.realpath(dev)
    try:
        for line in open("/proc/self/mounts"):
            src, mnt, _, opts = line.split()[:4]
            if os.path.realpath(src) == real and "rw" in opts.split(","):
                sys.exit(f"{dev} ({real})가 {mnt}에 읽기-쓰기로 마운트돼 있습니다. 지운 데이터가 덮이지 않게 먼저 "
                         f"해제하세요:\n  udisksctl unmount -b {real}")
    except OSError:
        pass
    try:
        disk = Disk(dev)
    except PermissionError:
        sys.exit(f"{dev}: 읽을 권한이 없습니다. sudo로 실행하세요")
    except FileNotFoundError:
        sys.exit(f"{dev}가 없습니다. lsblk -o NAME,SIZE,FSTYPE,LABEL 로 exFAT 파티션을 찾아 경로를 주세요")
    fs = ExFat(disk)
    free = fs.bits.count("0") * fs.cs
    say(f"{dev} ({real}): exFAT, 클러스터 {fs.cs // 1024} KB × {fs.ncl:,}개, 빈 공간 {gb(free)}"
        + (" · 정상적으로 해제되지 않았던 볼륨" if fs.dirty else ""))
    return fs


def find_bags(fs):
    """Bags in the drive's directories, deleted and live: one per (name, first cluster)."""
    out, seen = [], set()
    for e in fs.walk():
        if e["attr"] & 0x10 or not re.search(r"\.bag(\.active)?$", e["name"], re.I):
            continue
        key = (e["name"], e["first"])
        if key in seen:
            continue
        seen.add(key)
        e["runs"] = fs.runs(e["first"], e["nofat"], e["size"])
        e["mapped"] = sum(k for _, k in e["runs"]) * fs.cs
        e["reused"] = fs.used(e["runs"]) if e["gone"] else 0
        out.append(e)
    return out


def scan_bags(fs, bags):
    for e in bags:
        state = "지워짐" if e["gone"] else "있음"
        layout = "연속" if e["nofat"] else f"조각 {len(e['runs'])}개"
        if not e["intact"]:
            layout += " · 항목 일부 덮임"
        say(f"\n{e['path']}  [{state}] {gb(e['size'])} ({layout})")
        if not e["gone"]:
            continue
        ncl = max(1, -(-e["size"] // fs.cs))
        if e["reused"]:
            say(f"  클러스터 {e['reused']:,} / {ncl:,}개가 다른 파일에 다시 쓰임 ({100 * e['reused'] / ncl:.0f}%)")
        if e["mapped"] < e["size"]:
            say(f"  FAT 체인이 {gb(e['mapped'])}에서 끊김")
        e["check"] = check(Extent(fs, e["runs"], e["size"]))
        say("  " + describe(e["check"]))


def selected(name, only, everything):
    if everything:
        return True
    return any(name.startswith(p) for p in (only or WANTED))


def survey(fs, dry=True, out=None, names=()):
    """Read all free space for intact chunks; tally them per recording (dry) or write those of names into bags.
    A bag start found there whose clusters are scattered is also followed along the FAT chain it was written
    with."""
    runs = fs.free_runs()
    total = sum(k for _, k in runs) * fs.cs
    say(f"\n빈 공간 {gb(total)}에서 bag chunk를 찾습니다 (드라이브를 다 읽는 만큼 걸립니다)")
    prog = Progress("빈 공간 읽기", total)
    tally, writers, keys, done, zero = {}, {}, {}, 0, 0

    def take(ch, trusted=False):
        name = recording_of(ch.start) or "unknown_" + time.strftime("%Y-%m-%d_%H시",
                                                                   time.gmtime(ch.start // 10**9 + KST))
        if not dry and name in names:
            if name not in writers:
                writers[name] = BagWriter(os.path.join(out, name + ".bag"))
            writers[name].add_chunk(ch, trusted)
        key = (ch.start, ch.end, tuple(sorted(ch.counts.items())))
        if key in keys.setdefault(name, set()):
            return name
        keys[name].add(key)
        t = tally.setdefault(name, {"chunks": 0, "bytes": 0, "start": ch.start, "end": ch.end})
        t["chunks"] += 1
        t["bytes"] += len(ch.raw)
        t["start"], t["end"] = min(t["start"], ch.start), max(t["end"], ch.end)
        return name

    def collect(ext, progress=None, count_zeros=False, trusted=False, spans=None):
        nonlocal zero
        last = None                                          # a bag's index follows its last chunk
        for item in walk(ext, progress=progress):
            if item[0] == "chunk":
                last = take(item[1], trusted)
                if spans is not None:
                    sp = spans.setdefault(last, [item[1].start, item[1].end])
                    sp[0], sp[1] = min(sp[0], item[1].start), max(sp[1], item[1].end)
            elif item[0] == "conn":
                if last in writers:
                    writers[last].add_conn(item[1], item[2], trusted)
            elif item[0] == "gap" and count_zeros and item[2] - item[1] >= ZERO_STEP * fs.cs:
                a, b = item[1], item[2]                      # sampled: one cluster in ZERO_STEP
                zero += sum(1 for p in range(a + (-a) % fs.cs, b, ZERO_STEP * fs.cs)
                            if not ext.read(p, 512).strip(b"\0")) * ZERO_STEP * fs.cs

    starts = set()

    def spot(off, data):                                     # a bag's first bytes at the start of a free cluster
        first = fs.heap + -(-(off - fs.heap) // fs.cs) * fs.cs
        for p in range(first, off + len(data) - len(MAGIC) + 1, fs.cs):
            if data[p - off:p - off + len(MAGIC)] == MAGIC:
                starts.add((p - fs.heap) // fs.cs + 2)

    fs.d.hook = spot
    for c, k in runs:
        collect(Extent(fs, [(c, k)], k * fs.cs), lambda p: prog(done + p), dry)
        done += k * fs.cs
    fs.d.hook = None
    prog.end()
    heads = sorted(h for h in starts if not fs.allocated(h))
    scattered, spans = 0, {}
    for h in heads:
        chain = fs.stale_runs(h)
        if len(chain) < 2:
            continue                                         # contiguous: read whole above
        ext = Extent(fs, chain, sum(k for _, k in chain) * fs.cs)
        if record_at(ext, len(MAGIC) + 8 + HEADER_LEN, ext.size) is None:
            continue                                         # the chain does not go on with this bag
        scattered += 1
        collect(ext, trusted=True, spans=spans)
    for name, (t0, t1) in spans.items():                     # a scattered bag read in disk order mixes in others
        if name in writers:
            writers[name].drop_untrusted(t0, t1)
    if scattered:
        say(f"  흩어진 bag {scattered}개는 FAT 체인을 따라 읽었습니다")
    missing = {name: w.close() for name, w in writers.items()}
    return tally, missing, zero, heads


def cmd_scan(a):
    fs = open_device(a.device)
    bags = find_bags(fs)
    gone = [e for e in bags if e["gone"]]
    say(f"bag 항목 {len(bags)}개 (지워진 것 {len(gone)}개)")
    scan_bags(fs, bags)
    report = {"device": a.device, "bags": [{k: v for k, v in e.items() if k != "runs"} for e in bags]}
    if a.survey:
        tally, _, zero, heads = survey(fs)
        say("\n빈 공간에서 찾은 chunk (녹화별):")
        for name in sorted(tally):
            t = tally[name]
            dur = KNOWN.get(name)
            cover = (t["end"] - t["start"]) / 1e9
            say(f"  {name}: chunk {t['chunks']:,}개 · {gb(t['bytes'])} · {kst(t['start'])} ~ {kst(t['end'])}"
                + (f" (녹화 {dur}초 중 {cover:.0f}초 구간)" if dur else ""))
        say(f"  bag 시작 부분(헤더)이 남은 곳 {len(heads)}개")
        if zero:
            say(f"  0으로 채워진 빈 공간: {gb(zero)} (한 번도 안 쓰인 자리, 또는 TRIM으로 지워진 자리)")
        report["survey"] = tally
    path = os.path.abspath(time.strftime("exfat_scan_%m%d_%H%M.json"))
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    own(path)
    say(f"\n결과: {path}")


def cmd_recover(a):
    fs = open_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    own(a.out)
    if os.stat(a.out).st_dev == os.stat(fs.d.path).st_rdev:
        sys.exit("--out이 복구할 드라이브 위에 있습니다. 다른 디스크를 주세요")
    best = {}
    for e in find_bags(fs):                                   # one entry per name: the least overwritten
        if e["gone"] and e["mapped"] and selected(e["name"], a.only, a.all):
            k = e["name"]
            if k not in best or (e["intact"], -e["reused"]) > (best[k]["intact"], -best[k]["reused"]):
                best[k] = e
    need = sum(e["size"] for e in best.values())
    free = shutil.disk_usage(a.out).free
    say(f"디렉터리 항목으로 찾은 bag {len(best)}개, {gb(need)} → {a.out} (남은 공간 {gb(free)})")
    if need > free - (1 << 30):
        sys.exit("공간이 모자랍니다. --only로 나눠서 하거나 더 큰 디스크를 주세요")
    report = []
    for name in sorted(best):
        e = best[name]
        ext = Extent(fs, e["runs"], e["size"])
        say(f"\n{name}: {gb(e['size'])}, 검사 중")
        r = check(ext)
        say("  " + describe(r))
        dst = os.path.join(a.out, name)
        if r["ok"] and not e["reused"] and e["mapped"] >= e["size"]:
            copy(ext, dst)
            r2 = check(File(dst))
            how = "그대로 복사" if r2["ok"] else "복사했지만 검사 실패"
            say(f"  {how}: {describe(r2)}")
        else:
            how = "살아 있는 chunk로 다시 묶음"
            st = repair(ext, dst, window_for(name, r))
            r2 = check(File(dst)) if st["chunks"] else {"ok": False}
            say(f"  {how}: chunk {st['chunks']:,}개, {kst(st['start'])} ~ {kst(st['end'])}"
                + (f" · 다른 녹화의 chunk {st['other']:,}개 뺌" if st["other"] else "")
                + (f" · 연결 정보 없는 토픽 id {st['missing']}" if st["missing"] else "")
                + ("" if r2.get("ok") else " · 결과 검사 실패"))
        report.append({"name": name, "path": e["path"], "how": how, "check_before": r, "check_after": r2})
    if a.carve:                                               # recordings no directory entry leads to
        names = {n for n in KNOWN if selected(n + ".bag", a.only, a.all)} - {
            re.sub(r"\.bag(\.active)?$", "", n) for n in best}
        if names:
            say(f"\n항목이 없는 녹화 {len(names)}개: {', '.join(sorted(names))}")
            tally, missing, _, _ = survey(fs, dry=False, out=a.out, names=names)
            for n in sorted(names & set(tally)):
                t = tally[n]
                say(f"  {n}: 빈 공간의 chunk {t['chunks']:,}개로 다시 묶음 ({kst(t['start'])} ~ {kst(t['end'])}, "
                    f"녹화 {KNOWN[n]}초 중 {(t['end'] - t['start']) / 1e9:.0f}초 구간)"
                    + (f" · 연결 정보 없는 토픽 id {missing.get(n)}" if missing.get(n) else ""))
                report.append({"name": n + ".bag", "how": "빈 공간에서 다시 묶음", "survey": t,
                               "missing_conns": missing.get(n),
                               "check_after": check(File(os.path.join(a.out, n + ".bag")))})
            for n in sorted(names - set(tally)):
                say(f"  {n}: 빈 공간에서도 chunk를 찾지 못함")
    path = os.path.join(a.out, "rescue_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    own(path)
    say(f"\n끝. 보고서: {path}")


def copy(ext, dst):
    prog = Progress("복사", ext.size)
    with open(dst + ".part", "wb") as f:
        for k, (start, (dev, length)) in enumerate(zip(ext.starts, ext.devs)):
            n = min(length, ext.size - start)
            done = 0
            while done < n:
                m = min(32 << 20, n - done)
                f.write(ext.fs.d.read(dev + done, m))
                done += m
                prog(start + done)
        f.flush()
        os.fsync(f.fileno())
    prog.end()
    os.replace(dst + ".part", dst)
    own(dst)


def window_for(name, r=None):
    """The time a bag's chunks must lie in: its index's span, else its name's start and length (KNOWN). Chunks of
    other recordings written over it are left out."""
    if r and r.get("index") and r.get("start"):
        return r["start"] - 60 * 10**9, r["end"] + 60 * 10**9
    stem = re.sub(r"\.bag(\.active)?$", "", os.path.basename(name))
    t0 = name_start(stem)
    return (t0 - 60 * 10**9, t0 + (KNOWN.get(stem, 7200) + 120) * 10**9) if t0 is not None else None


def repair(rd, dst, window=None):
    """A new bag from rd's intact chunks (those within window=(t0, t1) when given)."""
    w = BagWriter(dst + ".part")
    prog = Progress("chunk 모으기", rd.size)
    stats = {"chunks": 0, "other": 0, "gaps": 0, "gap_bytes": 0, "start": None, "end": None}
    ours = True                       # connection records outside chunks (the index) belong with the chunks before
    for item in walk(rd, progress=prog):
        if item[0] == "chunk":
            ch = item[1]
            ours = window is None or window[0] <= ch.start <= window[1]
            if not ours:
                stats["other"] += 1
            elif w.add_chunk(ch):
                stats["chunks"] += 1
                stats["start"] = min(stats["start"] or ch.start, ch.start)
                stats["end"] = max(stats["end"] or ch.end, ch.end)
        elif item[0] == "conn":
            if ours:
                w.add_conn(item[1], item[2])
        elif item[0] == "gap":
            stats["gaps"] += 1
            stats["gap_bytes"] += item[2] - item[1]
    prog.end()
    stats["missing"] = w.close()
    os.replace(dst + ".part", dst)
    return stats


def cmd_check(a):
    r = check(File(a.bag))
    say(f"{a.bag}: {describe(r)}")
    for t0, t1 in r["lost"][:20]:
        say(f"  잃은 구간 {kst(t0)} ~ {kst(t1)} ({(t1 - t0) / 1e9:.1f}초)")


def cmd_repair(a):
    src = File(a.bag)
    st = repair(src, a.out, window_for(a.bag, check(src)))
    say(f"{a.out}: chunk {st['chunks']:,}개, {kst(st['start'])} ~ {kst(st['end'])}; 버린 구간 {st['gaps']}개 "
        f"({gb(st['gap_bytes'])})" + (f"; 다른 녹화의 chunk {st['other']:,}개 뺌" if st["other"] else "")
        + (f"; 연결 정보 없는 토픽 id {st['missing']}" if st["missing"] else ""))
    say(f"  검사: {describe(check(File(a.out)))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("scan", help="지워진 bag을 찾아 상태를 봅니다 (읽기만)")
    s.add_argument("device", nargs="?", default="/dev/disk/by-label/ExFAT")
    s.add_argument("--survey", action="store_true", help="빈 공간 전체에서 chunk도 찾기 (오래 걸림)")
    r = sub.add_parser("recover", help="bag을 다른 디스크로 꺼냅니다")
    r.add_argument("device", nargs="?", default="/dev/disk/by-label/ExFAT")
    r.add_argument("--out", required=True)
    r.add_argument("--only", type=lambda s: [x for x in s.split(",") if x], help="이름 앞부분, 쉼표로 (예: A-1_,F-)")
    r.add_argument("--all", action="store_true", help="지워진 bag 전부")
    r.add_argument("--carve", action="store_true", help="항목이 없는 녹화도 빈 공간의 chunk로 다시 묶기")
    c = sub.add_parser("check", help="bag 파일 검사")
    c.add_argument("bag")
    p = sub.add_parser("repair", help="bag 파일을 살아 있는 chunk로 다시 묶기")
    p.add_argument("bag")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.cmd:
        ap.print_help()
        return
    {"scan": cmd_scan, "recover": cmd_recover, "check": cmd_check, "repair": cmd_repair}[a.cmd](a)


if __name__ == "__main__":
    main()
