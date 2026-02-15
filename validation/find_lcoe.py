import openpyxl
wb = openpyxl.load_workbook('oa_hybrid.xlsx', data_only=True)
ws = wb['LCOE']

print("Scanning LCOE sheet for numeric values...")
for i in range(1, 50):
    for j in range(1, 15):
        cell = ws.cell(i, j)
        if cell.value:
            if isinstance(cell.value, str) and 'LCOE' in cell.value.upper():
                print(f"{cell.coordinate}: {cell.value}")
            elif isinstance(cell.value, (int, float)) and 2 < cell.value < 10:
                print(f"{cell.coordinate}: {cell.value:.4f}")
