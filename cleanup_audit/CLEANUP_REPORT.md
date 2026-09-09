# Project cleanup record

The cleaned distribution is `C:\UNI\BA Final Project\Project_Clean`.
The original project files were preserved. All 713
hashed original files still match the starting inventory.

## Scope and results

- Inventoried 18,193 files, including hidden files, local dependencies and Git history.
- Scanned 236 project-owned text files. Dependency and Git internals were inventoried as infrastructure, not rewritten.
- Exported 212 files (4.49 GB), including the checksum list.
- Excluded 17,982 files (3.22 GB) from the copy; most excluded file counts are virtual-environment and Git internals.
- Modified 36 exported source/documentation files. All retained reports, images, model artifacts, metrics, the PBIX file and database backup match their original SHA-256 hashes.

## Exclusions

Local assistant configuration, Git history, the virtual environment, bytecode,
temporary document builds, rendered QA images, intermediate report versions,
unreferenced draft illustrations, the older database backup, and the explicitly
named Power BI backup were omitted. The selected final reports are
`Project_Report.docx` and `Project_Report.pdf`.

Unreferenced template variants and stylesheets were omitted. Project-level Django
templates shadowed four application-level templates, so the unused copies were
omitted. The current model loaders use revenue version 12, repurchase version 4 and
hybrid version 1; incompatible old cache generations were omitted. The large
collaborative-filtering cache was retained for demonstration performance.

Broken or obsolete one-off helper scripts were omitted. The supported Django
transaction import command remains, with an explicit CSV path required.

## Source changes

Redundant file labels, editing notes, generic tutorial text and promotional
recommendation labels were simplified. Substantive algorithm explanations,
framework migration markers and third-party attributions were retained.
The unused historical churn implementation was removed while its deprecation
guards were preserved.

Three legacy URLs referenced missing templates. They now redirect to the existing
dashboard or basket-analysis pages, retaining login checks. One existing forecast
test used a superseded fixture structure; its fixture now matches the current
per-lead model interface, with the original numerical assertions preserved.

The exported settings accept environment variables for database connection,
allowed hosts, debug mode and the secret key. A temporary secret is generated
when no stable key is supplied. The README documents setup, actual page paths,
current model versions and the date of the included database snapshot.

## Verification

- 37 existing tests passed using a temporary in-memory SQLite database. No live test database was created.
- Django system checks and installed dependency consistency checks passed.
- 103 Python files compiled; 12 JSON files parsed.
- 23 templates compiled; 45 template references and 7 static references resolved.
- 20 source/template script blocks passed JavaScript syntax checks after substituting template values.
- 37 actual server-rendered script blocks passed JavaScript syntax checks.
- Nine main pages and two dashboard endpoints returned HTTP 200 against local SQL Server. The checker blocked database-writing statements.
- SQL Server `RESTORE VERIFYONLY` accepted the copied backup. No restore was performed.
- The selected 221-page PDF was text-extracted. The DOCX package passed its ZIP integrity check and contained no Word comments or tracked insertions/deletions.
- No assistant tool names were found in retained project text or in the inspected final report text/core metadata. This is a literal scan, not an authorship test.

## Limits

Code style cannot establish who wrote a project, and this cleanup does not certify
independent authorship or satisfy an institution's disclosure rules. This was a
distribution cleanup with automated source checks and targeted review, not a
proof that every algorithm, browser interaction or report claim is correct.

The report contents and layout were not edited; full page-by-page visual review
was outside the cleanup. Binary model files were copied and hash-checked; the
models were not all retrained. The database backup is from 11 August 2026 and was
not restored to compare its contents with the current live database. Newer Django
migrations may be needed after restoration.

SQL Server-specific behaviour was smoke-tested on the live database, while the
integration test ran on SQLite. Browser interactions and external CDN/Power BI
availability were not exercised end to end. This remains a local demonstration
configuration, not a completed production security review. Existing CSRF-exempt
API handlers and the legacy dataset-specific CSV importer warrant a separate
review before production use or importing a different dataset.

## Audit files

- `source_inventory.csv`: every original file, with hashes for project-owned files.
- `export_manifest.csv`: the keep/exclude decision and reason for every original file.
- `final_inventory.csv`: final distributed file hashes.
- `changes.json`: source refinements and their reasons.
- `source_review.json`: text scans and Python syntax/structure inventory.
- `verification.json`, `rendered_javascript.json`, `live_smoke.json`: source and runtime checks.
- `isolated_tests.log`, `database_check.json`, `document_inspection.json`: test and artifact evidence.

The audit folder is separate from the distribution. No original files were deleted.
