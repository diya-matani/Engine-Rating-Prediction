import pandas as pd
try:
    df = pd.read_excel('data.xlsx')
    print("Columns in data.xlsx:")
    for col in df.columns:
        print(col)
except Exception as e:
    print(e)
