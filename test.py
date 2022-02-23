from utils.profile_processor import ProfileReader
from utils.profile_processor import PreStartPoint

file = ProfileReader('data/profileBM.txt')
df = file.profile_df
#print(df)
new_df = file.concat_dfs()
#print(new_df)
dtfrm = df.join(new_df, how='left', rsuffix='_след.')
print(dtfrm)



