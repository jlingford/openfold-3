# 5. Best Practices and Additional Notes

<https://openfold-3.readthedocs.io/en/latest/precomputed_msa_generation_how_to.html>

*Run proteins and RNA separately:* If you need to align both proteins and RNA, you have to make two separate configs and have two separate calls to the pipeline, one for each modality.

*Run with one database at a time:* Running multiple databases at once will work just fine, but for generating MSAs at scale (for instance, for 100 sequences), we recommend running only a single database at a time - this leads to faster runtimes as the database can be more aggressively cached in memory.

*Use whole-node jobs on HPC clusters:* While snakemake has great support for running individual jobs across a cluster, we find that the optimal way to use our alignment pipeline on a typical academic HPC is to submit independent snakemake jobs that use a whole node at a time. The main reason for this is that alignments generally work best when the alignment databases are stored on node-local SSD based storage. This typically requires copying data each time a job is run on a node as in most clusters node-local storage is not peristent. Therefore a typical workflow involves first copying alignment DBs to a node, and then running snakemake locally on that node. 

# .json config

The available databases are:

```json
"databases":["uniref90", "uniref30", "uniprot", "mgnify", "cfdb", "bfd"],
```

The different fields are:

- input_fasta: path to multi-fasta file
- openfold_env: path to the `of3-aln-env`, NOT the `openfold-3` env used for running the prediction on GPUs
- databases: choose just ONE database for better performance
- base_database_path: path to the download dir of databases (i.e., `alignment_dbs/`)
- output_directory: path to output
- jackhmmer_output_format: "sto" or "a3m". Use "a3m" for the hhblits databses (i.e., "bfd", "uniref30", "cfdb")
- jackhmmer_threads: use a low number of threads for jackhmmer databases (i.e., *.fasta)
- hhblits_threads: *WHAT IS OPTIMAL NUM?*
- tmpdir: path to temp dir (needs lots of disk space and fast access)
- run_template_search: bool **PURPOSE???**

Example:

```json
{
    "input_fasta":"/home/jamesl/rp24_scratch/rp24/jamesl2/MMseqs2_stuff/openfold-3/fastas/msa_input_famoushydgeneneighbours.faa",
    "openfold_env":"/fs04/scratch2/rp24/jamesl2/MMseqs2_stuff/openfold-3/scripts/snakemake_msa/rp24_scratch2/jamesl2/miniconda/conda/envs/of3-aln-env",
    "databases":["uniref30"],
    "base_database_path":"/home/jamesl/rp24_scratch/Database/AF2-db/OpenFold3_DB/alignment_dbs",
    "output_directory":"/home/jamesl/rp24_scratch/rp24/jamesl2/MMseqs2_stuff/msas/of3",
    "jackhmmer_output_format":"a3m",
    "jackhmmer_threads":4,
    "hhblits_threads":16,
    "tmpdir": "/home/jamesl/rp24_scratch/rp24/jamesl2/MMseqs2_stuff/openfold-3/tmp/msa_tmp",
    "run_template_search":false
}
```


