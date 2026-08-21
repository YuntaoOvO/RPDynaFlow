#!/usr/bin/env python3
"""Drive ``gmx pdb2gmx -ter`` and auto-select terminal patches by residue type.

CHARMM36 represents nucleic-acid termini as terminal PATCHES (5TER/3TER), not
separate rtp building blocks, so GROMACS does NOT auto-select them — the default
applies the protein patch NH3+ to the 5' nucleotide and crashes ("atom N not
found in building block 1GUA"). This wrapper answers the -ter prompts by residue.

Selection is matched by NAME as a SUBSTRING of the option token (the option list
is inconsistent and residue-prefixed, e.g. GLY N-term shows "0: GLY-NH3+" not
"0: NH3+"; PRO N-term shows "0: PRO-NH2+"):
    nucleotide (A U G C T DA DT DG DC) -> start 5TER , end 3TER
    PRO at N-term                       -> "NH2+"  (PRO-NH2+ patch)
    other amino acid                    -> start NH3+, end COO-
    histidine tautomer prompt           -> 0 (default)

Exit status mirrors pdb2gmx; its output streams to stdout for the caller's log.

Usage:  python3 pdb2gmx_auto.py gmx pdb2gmx <...args incl. -ter -ignh>
"""
import re
import sys

import pexpect

NT = {"A", "U", "G", "C", "T", "DA", "DT", "DG", "DC"}


def want_name(res, start):
    if res in NT:
        return "5TER" if start else "3TER"
    if start and res == "PRO":
        return "NH2+"          # PRO N-terminus -> PRO-NH2+ patch
    return "NH3+" if start else "COO-"


def main():
    cmd = sys.argv[1:]
    if not cmd:
        sys.stderr.write("pdb2gmx_auto.py: no command given\n")
        sys.exit(64)
    child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8", timeout=300)
    child.logfile_read = sys.stdout  # stream pdb2gmx output to caller's log

    prompts = [
        re.compile(r"Select start terminus type for ([A-Z0-9]+)-"),
        re.compile(r"Select end terminus type for ([A-Z0-9]+)-"),
        re.compile(r"protonation state"),
        pexpect.EOF,
        pexpect.TIMEOUT,
    ]
    try:
        while True:
            i = child.expect(prompts)
            if i in (0, 1):
                wn = want_name(child.match.group(1), i == 0)
                # containment match: "0: NH3+", "0: GLY-NH3+", "0: PRO-NH2+", "4: 5TER"
                opt = re.compile(r"(\d+):\s\S*" + re.escape(wn) + r"(?!\S)")
                j = child.expect([opt, pexpect.EOF, pexpect.TIMEOUT])
                if j == 0:
                    child.sendline(child.match.group(1))
                elif j == 1:
                    sys.stderr.write(f"pdb2gmx_auto: EOF before terminus option {wn}\n")
                    sys.exit(2)
                else:
                    sys.stderr.write(f"pdb2gmx_auto: timeout looking for option {wn}\n")
                    child.close(force=True)
                    sys.exit(3)
            elif i == 2:  # histidine tautomer -> default
                child.sendline("0")
            elif i == 3:  # EOF
                break
            else:  # TIMEOUT
                sys.stderr.write("pdb2gmx_auto: timeout waiting for pdb2gmx\n")
                child.close(force=True)
                sys.exit(3)
    finally:
        child.close()
    ec = child.exitstatus
    sys.exit(ec if ec is not None else 1)


if __name__ == "__main__":
    main()
