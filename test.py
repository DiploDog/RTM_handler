from utils.profile_processor import ProfileReader
from utils.profile_processor import PreStartPoint
import matplotlib.pyplot as plt

file = ProfileReader('data/profileBM.txt')
df = file.profile_df
#print(df)
new_df = file.shift_df()
#print(new_df)
dtfrm = file.widen_dfs()
slope = file.calculate_slope()
#print(slope)
predicted = file._get_widened_df()
print(predicted)
print(list(predicted.iloc[-1, :]))
print('Уклон', predicted['Уклон'].unique())
print('Расстояние', predicted['Расстояние'].unique())
print(file.obj_psp.get_coords_score())
# plt.plot(predicted['Пикет'], predicted['Высота'],
#          predicted['Пикет_след.'], predicted['Высота_след.'], alpha=0.5)
# #plt.plot(predicted['Пикет_след.'], predicted['Высота_след.'])
# plt.show()


