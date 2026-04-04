import re, os
with open('hw13_comprehensive_report.md') as f:
    for i, line in enumerate(f, 1):
        m = re.search(r'!\[.*?\]\(([^)]+)\)', line)
        if m:
            path = m.group(1)
            exists = os.path.isfile(path)
            status = 'OK' if exists else 'MISSING'
            print(f'Line {i}: {status} -> {path}')
