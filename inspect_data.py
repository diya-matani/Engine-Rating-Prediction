import pandas as pd
try:
    xl = pd.ExcelFile('data.xlsx')
    print(f"Sheet names: {xl.sheet_names}")
    # Load first sheet that looks like data if possible, or just print len
    for sheet in xl.sheet_names:
        df = pd.read_excel('data.xlsx', sheet_name=sheet)
        print(f"Sheet: {sheet}, Columns: {list(df.columns)}")
except Exception as e:
    print(e)
