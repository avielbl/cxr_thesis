import pandas as pd
csv_path = r"C:\projects\CXR_thesis\data_repo\NIH\Data_Entry_2017.csv"
df = pd.read_csv(csv_path)
df.columns = [c.lower() for c in df.columns] #postgres doesn't like capitals or spaces

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:postgres@localhost:5432/nihlabels')

df.to_sql("labels", engine)