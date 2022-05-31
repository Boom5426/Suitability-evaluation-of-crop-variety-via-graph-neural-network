from cmath import isnan
import pandas as pd
import numpy as np
import io


data =  pd.read_excel('./data_four_years.xlsx', engine='openpyxl')

# print(data[:3])
data_total = ["续试","继续试验","适宜本地区种植","建议续试","继续试验或审定","建议继续试验",\
    "建议继续参试", "晋级", ]

data_select = []
data_target = []
count = 0
cou = 0
# print(len(data.index.values)) # 2327
for row in data.index.values:
    print(data['品种建议'][row])
    if data['品种建议'][row] is not np.nan:
        cou += 1
    if data['品种建议'][row] not in data_total:
        continue

    count += 1
    # print(data['空秆率(%)'][row])

    feature_1 = data['空秆率(%)'][row] if not np.isnan(data['空秆率(%)'][row]) else 0
    # feature_2 = data['比对照增减产(%)'][row] if not np.isnan(data['比对照增减产(%)'][row]) else 0
    # feature_3 = data['褐斑病(级)'][row] if not np.isnan(data['褐斑病(级)'][row]) else 0
    # feature_4 = data['大斑病'][row] if not np.isnan(data['大斑病'][row]) else 0
    # feature_5 = data['弯孢叶斑病'][row] if not np.isnan(data['弯孢叶斑病'][row]) else 0
    # feature_6 = data['倒伏率(%)'][row] if not np.isnan(data['倒伏率(%)'][row]) else 0
    # feature_7 = data['倒折率(%)'][row] if not np.isnan(data['倒折率(%)'][row]) else 0
    # feature_8 = data['小斑病'][row] if not np.isnan(data['小斑病'][row]) else 0

print(count)
print(cou)

