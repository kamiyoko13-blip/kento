#!/usr/bin/env python3
"""
Fix concatenated 'False' + 'def send_notification' tokens and dedupe send_notification definitions.
Usage: python fix_send_notification.py ninibo1127.py
Outputs: <input>.fixed
"""
import sys, re, io, os

if len(sys.argv) < 2:
    print("Usage: python fix_send_notification.py <file>")
    sys.exit(2)

fn = sys.argv[1]
txt = open(fn, 'r', encoding='utf-8').read()

# 1) Fix likely concatenations (common paste errors)
# Replace 'Falsesend_notification' and 'Falsedef send_notification' and similar with proper newline(s)
txt_fixed = re.sub(r'False\s*send_notification', 'False\n\nsend_notification', txt)
txt_fixed = re.sub(r'Falsedef\s+send_notification', 'False\n\ndef send_notification', txt_fixed)
txt_fixed = re.sub(r'Falsesend_notification', 'False\n\ndef send_notification', txt_fixed)

# 2) Normalize to ensure 'def send_notification' lines start with 'def ' at column 0
# If function got indented wrongly at top-level, try to bring top-level defs to column 0
# (only for def send_notification occurrences that appear at column > 0 and are not inside class)
lines = txt_fixed.splitlines(keepends=True)
# Build list of indices where a top-level def appears (no leading space)
top_def_idxs = []
for i, line in enumerate(lines):
    if re.match(r'^\s*def\s+send_notification\s*\(', line):
        # consider top-level if indent is 0
        if line.lstrip() == line:
            top_def_idxs.append(i)
        else:
            # if indented but previous non-blank non-comment is 'class ' then skip promoting
            # otherwise, promote to column 0 (remove leading spaces)
            # Heuristic: if previous non-empty line starts with 'class ' within last 5 lines, don't demote.
            prev_range = range(max(0, i-6), i)
            prev_nonblank = None
            for j in prev_range:
                if lines[j].strip():
                    prev_nonblank = lines[j].strip()
                    break
            if prev_nonblank and prev_nonblank.startswith('class '):
                # keep as-is (probably method)
                continue
            # promote by removing leading spaces for this def and following indented block until next top-level def
            lines[i] = re.sub(r'^\s*', '', lines[i])
            top_def_idxs.append(i)

# Rebuild text
txt_fixed = ''.join(lines)

# 3) If more than one top-level def send_notification, keep first and remove subsequent ones (body until next top-level def)
# Find all start positions of '^def send_notification' at column 0
pattern = re.compile(r'(?m)^def\s+send_notification\s*\(')
matches = list(pattern.finditer(txt_fixed))
if len(matches) <= 1:
    out_fn = fn + '.fixed'
    open(out_fn, 'w', encoding='utf-8').write(txt_fixed)
    print(f"No duplicates found (count={len(matches)}). Wrote {out_fn}")
    sys.exit(0)

# locate function boundaries
starts = [m.start() for m in matches]
# find end of each function: the start of next top-level 'def ' (column 0) or EOF
top_level_def_pattern = re.compile(r'(?m)^def\s+\w+\s*\(')
# split into lines to ease removal
lines = txt_fixed.splitlines(keepends=True)
# find indices (line numbers) of top-level def send_notification occurrences
line_starts = []
for i, line in enumerate(lines):
    if re.match(r'^def\s+send_notification\s*\(', line):
        line_starts.append(i)

# Keep the first occurrence; remove the others entirely (function body until next top-level def at col 0)
keep_index = line_starts[0]
to_delete_ranges = []
for idx in line_starts[1:]:
    # start at idx, find end: next line number j where a top-level 'def ' occurs (col 0), excluding current idx
    end = None
    for j in range(idx+1, len(lines)):
        if re.match(r'^def\s+\w+\s*\(', lines[j]):
            end = j
            break
    if end is None:
        end = len(lines)
    to_delete_ranges.append((idx, end))

# Remove ranges from bottom to top to keep indices valid
for a,b in reversed(to_delete_ranges):
    del lines[a:b]

out_text = ''.join(lines)
out_fn = fn + '.fixed'
open(out_fn, 'w', encoding='utf-8').write(out_text)
print(f"Found {len(matches)} send_notification defs, kept the first, wrote {out_fn}")
