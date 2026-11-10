import sys,openpyxl
wb=openpyxl.load_workbook(sys.argv[1],read_only=True)
for ws in wb.worksheets:
    rows=list(ws.iter_rows(values_only=True)); print('##',ws.title,len(rows))
    for r in rows[:int(sys.argv[2]) if len(sys.argv)>2 else 8]: print(r)
