# Training Data

Large training prompt files are intentionally excluded from Git. Prepare the
Light-Forcing/Self-Forcing prompt corpus with:

```bash
bash scripts/prepare_hsa_training_data.sh
```

This downloads `gdhe17/Self-Forcing/vidprom_filtered_extended.txt`, verifies
SHA256 `7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5`,
deduplicates prompts, and removes normalized exact overlaps with the complete
standard and augmented VBench prompt files.

Expected output is 248,217 prompts with SHA256
`c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623`.
The upstream VidProM-derived data should be treated as CC BY-NC 4.0 and used
only where that license is acceptable.
