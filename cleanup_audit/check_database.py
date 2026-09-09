from pathlib import Path
import json
import pyodbc

ROOT = Path(__file__).resolve().parent.parent
result = {}
try:
    connection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=marketdb;Trusted_Connection=yes;Encrypt=no;TrustServerCertificate=yes;', timeout=5, autocommit=True)
    connection.timeout = 60
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(*), MIN(day), MAX(day) FROM transactions')
    result['transactions'] = list(cursor.fetchone())
    cursor.execute('SELECT COUNT(*) FROM product')
    result['products'] = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sys.views WHERE name LIKE 'vw_%'")
    result['reporting_views'] = cursor.fetchone()[0]
    cursor.execute('SELECT app, name FROM django_migrations ORDER BY app, name')
    result['migrations'] = [list(row) for row in cursor.fetchall()]
    backup = str(ROOT / 'Project_Clean/database/marketdb_backup_20260811_191945.bak').replace("'", "''")
    cursor.execute(f"RESTORE HEADERONLY FROM DISK = N'{backup}'")
    names = [d[0] for d in cursor.description]
    rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    result['backup_headers'] = [{k: str(row.get(k)) for k in ['DatabaseName', 'BackupStartDate', 'BackupFinishDate', 'BackupSize', 'HasBackupChecksums']} for row in rows]
    cursor.execute(f"RESTORE VERIFYONLY FROM DISK = N'{backup}'")
    while cursor.nextset():
        pass
    result['backup_verify'] = 'passed'
    connection.close()
except Exception as exc:
    result['error'] = str(exc)
(ROOT / 'cleanup_audit/database_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k,v in result.items() if k != 'migrations'}, indent=2))
