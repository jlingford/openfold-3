#!/usr/bin/env python3
"""
Cluster Rfam-like FASTA with MMseqs2 and extract:
  - cluster representatives from clusters with >=3 members
  - then filter those reps to sequences with length > 10 nt.

Equivalent to the shell pipeline:

  mmseqs createdb Rfam.fa rfamDB
  mmseqs cluster rfamDB rfamDB_clu tmp --min-seq-id 0.9 -c 0.8 --cov-mode 1
  mmseqs createtsv rfamDB rfamDB rfamDB_clu rfamDB_clu.tsv
  awk '{n[$1]++} END {for (c in n) if (n[c] >= 3) print c}' rfamDB_clu.tsv > reps_ge3.ids
  mmseqs result2repseq rfamDB rfamDB_clu rfamDB_rep
  mmseqs result2flat rfamDB rfamDB rfamDB_rep rfamDB_rep.fa --use-fasta-header
  awk 'NR==FNR {ids[$1]; next} /^>/ {id=substr($1,2); keep=(id in ids)} keep' \
        reps_ge3.ids rfamDB_rep.fa > Rfam_reps_ge3.fa
  awk '...len>10 filter...' Rfam_reps_ge3.fa > Rfam_reps_ge3_lenGT10.fa
"""

import sys              # for command-line args and stderr output
import subprocess       # to run mmseqs commands
from pathlib import Path  # for convenient path manipulations
from collections import Counter  # to count cluster members


def run(cmd):
    """Run a command with subprocess, printing it first; fail on error."""
    print("[RUN]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def count_clusters(tsv_path, min_size, ids_out_path):
    """
    Parse rfamDB_clu.tsv and write representative IDs for clusters
    with size >= min_size to ids_out_path.

    Equivalent to:
      awk '{n[$1]++} END {for (c in n) if (n[c] >= 3) print c}' rfamDB_clu.tsv > reps_ge3.ids
    """
    counts = Counter()
    with open(tsv_path) as f:
        for line in f:
            if not line.strip():
                continue
            rep_id = line.split("\t", 1)[0]  # $1 in awk
            counts[rep_id] += 1

    kept = 0
    with open(ids_out_path, "w") as out:
        for rep_id, n in counts.items():
            if n >= min_size:
                out.write(rep_id + "\n")
                kept += 1

    print(f"[INFO] Clusters with size >= {min_size}: {kept}", file=sys.stderr)
    return kept


def filter_fasta_by_ids(ids_path, fasta_in, fasta_out):
    """
    Keep only records whose header ID is in ids_path.

    Equivalent to:
      awk 'NR==FNR {ids[$1]; next} /^>/ {id=substr($1,2); keep=(id in ids)} keep' \
          reps_ge3.ids rfamDB_rep.fa > Rfam_reps_ge3.fa
    """
    # Load allowed IDs (one per line)
    keep_ids = set(line.strip() for line in open(ids_path) if line.strip())
    print(f"[INFO] Loaded {len(keep_ids):,} IDs to keep", file=sys.stderr)

    with open(fasta_in) as fin, open(fasta_out, "w") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                # Extract ID as first token after '>'
                header_id = line[1:].split()[0]
                write = header_id in keep_ids
            if write:
                fout.write(line)


def filter_fasta_by_length(fasta_in, fasta_out, min_len=10):
    """
    Keep only sequences (possibly multiline) with total length > min_len.

    Equivalent to the long awk block that accumulates seqlen across lines.
    """
    with open(fasta_in) as fin, open(fasta_out, "w") as fout:
        header = None
        seq_chunks = []

        def flush_record():
            """Write current record if length > min_len."""
            if header is None:
                return
            seq = "".join(seq_chunks)
            if len(seq) > min_len:
                fout.write(header)
                # Wrap sequence as in original file (just one line here)
                fout.write(seq + "\n")

        for line in fin:
            if line.startswith(">"):
                # New record: flush previous one
                flush_record()
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())

        # Flush last record
        flush_record()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} Rfam.fa", file=sys.stderr)
        sys.exit(1)

    # Positional input FASTA (only required input)
    fasta_path = Path(sys.argv[1]).resolve()

    # Derive base names and MMseqs DB names from the FASTA
    base = fasta_path.stem           # e.g. "Rfam" from "Rfam.fa"
    db_prefix = base + "DB"          # rfamDB
    db = Path(db_prefix)             # rfamDB
    clu = Path(db_prefix + "_clu")   # rfamDB_clu
    tmp_dir = Path("tmp")            # tmp (as in your command)
    tmp_dir.mkdir(exist_ok=True)

    clu_tsv = Path(str(clu) + ".tsv")    # rfamDB_clu.tsv
    reps_ids = Path("reps_ge3.ids")      # reps_ge3.ids
    rep_db = Path(db_prefix + "_rep")    # rfamDB_rep (DB)
    rep_fa = Path(db_prefix + "_rep.fa") # rfamDB_rep.fa

    reps_ge3_fa = Path(f"{base}_reps_ge3.fa")              # Rfam_reps_ge3.fa
    reps_ge3_len_fa = Path(f"{base}_reps_ge3_lenGT10.fa")  # Rfam_reps_ge3_lenGT10.fa

    # 1) createdb: mmseqs createdb Rfam.fa rfamDB
    run(["mmseqs", "createdb", str(fasta_path), str(db)])

    # 2) cluster: mmseqs cluster rfamDB rfamDB_clu tmp --min-seq-id 0.9 -c 0.8 --cov-mode 1
    run([
        "mmseqs", "cluster",
        str(db), str(clu), str(tmp_dir),
        "--min-seq-id", "0.9",
        "-c", "0.8",
        "--cov-mode", "1",
    ])

    # 3) createtsv: mmseqs createtsv rfamDB rfamDB rfamDB_clu rfamDB_clu.tsv
    run(["mmseqs", "createtsv", str(db), str(db), str(clu), str(clu_tsv)])

    # 4) awk cluster-size >=3 -> reps_ge3.ids
    count_clusters(clu_tsv, min_size=3, ids_out_path=reps_ids)

    # 5) result2repseq: mmseqs result2repseq rfamDB rfamDB_clu rfamDB_rep
    run(["mmseqs", "result2repseq", str(db), str(clu), str(rep_db)])

    # 6) result2flat: mmseqs result2flat rfamDB rfamDB rfamDB_rep rfamDB_rep.fa --use-fasta-header
    run([
        "mmseqs", "result2flat",
        str(db), str(db), str(rep_db), str(rep_fa),
        "--use-fasta-header",
    ])

    # 7) awk ID filter: reps_ge3.ids + rfamDB_rep.fa -> Rfam_reps_ge3.fa
    filter_fasta_by_ids(reps_ids, rep_fa, reps_ge3_fa)

    # 8) awk length > 10 filter: Rfam_reps_ge3.fa -> Rfam_reps_ge3_lenGT10.fa
    filter_fasta_by_length(reps_ge3_fa, reps_ge3_len_fa, min_len=10)

    print(f"[DONE] Reps from clusters >=3 written to: {reps_ge3_fa}", file=sys.stderr)
    print(f"[DONE] Length>10 subset written to:      {reps_ge3_len_fa}", file=sys.stderr)


if __name__ == "__main__":
    main()

