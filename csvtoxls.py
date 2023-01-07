import pandas as pd

data = pd.read_csv("Book1.csv")

data.to_excel("read_this.xlsx", index=None, header=True)