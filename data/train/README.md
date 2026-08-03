# Training Data

Large training prompt files are intentionally excluded from Git. Prepare the
Light-Forcing/Self-Forcing prompt corpus with:

```bash
bash scripts/prepare_hsa_training_data.sh
```

The script is offline by default. It first reuses a validated
`prompts_train.txt`, then looks for a local `source_prompts.txt` or
`vidprom_filtered_extended.txt`. A different local file can be selected with
`SOURCE_FILE=/path/to/vidprom_filtered_extended.txt`. Only
`ALLOW_DOWNLOAD=1 bash scripts/prepare_hsa_training_data.sh` is allowed to
access Hugging Face. The source SHA256 is
`7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5`.
The preparation step deduplicates prompts and removes normalized exact
overlaps with the complete standard and augmented VBench prompt files.

Expected output is 248,217 prompts with SHA256
`c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623`.
The upstream VidProM-derived data should be treated as CC BY-NC 4.0 and used
only where that license is acceptable.
