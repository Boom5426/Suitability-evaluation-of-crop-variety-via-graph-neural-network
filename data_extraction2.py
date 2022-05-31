import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv 
import os


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

filename = 'problem.txt'
f_data = open('our_data.csv','w',encoding='utf-8')
csv_writer = csv.writer(f_data)
csv_writer.writerow(["序号","最高气温均值","最高气温方差","平均气温均值","平均气温方差","最低气温均值",\
    "最低气温方差","温差均值","温差方差","地面气压均值","地面气压方差","相对湿度均值","相对湿度方差",\
        "降水量均值","降水量方差","最大风速均值","最大风速方差","平均风速均值","平均风速方差","风向角度均值",\
            "风向角度方差","日照时间均值","日照时间方差","风力等级均值","风力等级方差","大斑病","倒伏率",\
                "倒折率","灰斑病","株高","穗位高","空杆率","生育期","穗腐病","百粒重","穗长","秃尖长","鲜穗果重","亩产",\
                    "增减产", "评审状态"])

# 读取数据
corn_file=open('data_select.csv',encoding='gbk')    #打开玉米文件夹  
csv_reader_lines = csv.reader(corn_file)    #用csv.reader读文件  

path = "./meteorology_data/" #打开气象文件夹目录
files= sorted(os.listdir(path)) #得到文件夹下的所有文件名称
information = []
corn_num = 0
for one_line in csv_reader_lines:  #遍历玉米产量文件夹
    if corn_num==0:
        print(one_line)
        information = one_line
    if corn_num>0:
        TEM_Max = [] #最高温
        TEM_Avg = [] #平均温
        TEM_Min = [] #最低温
        PRS = [] #地面气压
        RHU = [] #相对湿度
        PRE = [] #降水量
        WIN_Max = [] #最大风速
        WIN_S = [] #平均风速
        WIN_deg = [] #风向角度
        SSD = [] #日照时间
        WIN_Scale = [] #风力等级
    
        province = one_line[1] #省
        city = one_line[2] #市
        district = one_line[3] #区
        start_data = one_line[4] #播种期
        end_data = one_line[13] #成熟期
        per_mu_yield = one_line[21] #亩产

        if (start_data == '' or end_data == ''):
            continue
        strlist = start_data.split('/')
        if int(strlist[1])<10:
            strlist[1]='0'+strlist[1]
        if int(strlist[2])<10:
            strlist[2]='0'+strlist[2]
        start_data_int = int(strlist[0]+strlist[1]+strlist[2]) 
        strlist2 = end_data.split('/')
        if int(strlist2[1])<10:
            strlist2[1]='0'+strlist2[1]
        if int(strlist2[2])<10:
            strlist2[2]='0'+strlist2[2]
        start_data_int = int(strlist[0]+strlist[1]+strlist[2])
        end_data_int = int(strlist2[0]+strlist2[1]+strlist2[2])

        for file in files: #遍历气象数据文件夹
            # print(file)
            file_int = int(file[0:6])
            start_1 = int(strlist[0]+strlist[1])
            end_1 = int(strlist2[0]+strlist2[1])
            if (file_int<start_1 or file_int>end_1):
                continue
            file_name = path+file
            f = open(file_name)
            data_reader_lines = csv.reader(f)
            num = 0 
            for one_day_data in data_reader_lines:
                if num > 0:
                    this_province = one_day_data[0]
                    this_city = one_day_data[1]
                    this_district = one_day_data[2]
                    data=one_day_data[3]
                    this_data=data[0:4]+data[5:7]+data[8:10]
                    this_data_int = int(this_data)
                    if this_province == province and this_city == city and this_district == district and this_data_int>=start_data_int and this_data_int<=end_data_int:
                        TEM_Max.append(one_day_data[4])
                        TEM_Avg.append(one_day_data[5])
                        TEM_Min.append(one_day_data[6])
                        PRS.append(one_day_data[7])
                        RHU.append(one_day_data[8])
                        PRE.append(one_day_data[9])
                        WIN_Max.append(one_day_data[10])
                        WIN_S.append(one_day_data[11])
                        WIN_deg.append(one_day_data[12])
                        SSD.append(one_day_data[13])
                        WIN_Scale.append(one_day_data[14])
                num = num + 1  
        if len(TEM_Max)== 0:
            print(one_line[4]+"  "+one_line[5]+"  "+ one_line[6]+"   "+one_line[7]+"   "+one_line[23]+"  "+str(corn_num))
        if len(TEM_Max)> 0:
            TEM_Max = list(filter(None, TEM_Max))
            TEM_Max_num = np.array(list(map(float, TEM_Max)))

            TEM_Avg = list(filter(None, TEM_Avg))
            TEM_Avg_num = np.array(list(map(float, TEM_Avg)))

            TEM_Min = list(filter(None, TEM_Min))
            TEM_Min_num = np.array(list(map(float, TEM_Min)))
            if  TEM_Max_num.shape[0]==TEM_Min_num.shape[0]:
                TEM_Wencha =  TEM_Max_num-TEM_Min_num
            else:
                    TEM_Wencha = np.array([1,2,3,4])

            PRS = list(filter(None, PRS))
            PRS_num = np.array(list(map(float, PRS)))
            
            RHU = list(filter(None, RHU))
            RHU_num = np.array(list(map(float, RHU)))

            PRE = list(filter(None, PRE))
            PRE_num = np.array(list(map(float, PRE)))

            WIN_Max = list(filter(None, WIN_Max))
            WIN_Max_num = np.array(list(map(float, WIN_Max)))

            WIN_S = list(filter(None, WIN_S))
            WIN_S_num = np.array(list(map(float, WIN_S)))

            WIN_deg = list(filter(None, WIN_deg))
            WIN_deg_num = np.array(list(map(float, WIN_deg)))

            SSD = list(filter(None, SSD))
            SSD_num = np.array(list(map(float, SSD)))

            WIN_Scale = list(filter(None, WIN_Scale))
            WIN_Scale_num = np.array(list(map(float, WIN_Scale)))

            TEM_Max_num_mean = np.mean(TEM_Max_num) #最高气温的均值
            TEM_Max_num_std = np.std(TEM_Max_num)   #最高气温的方差

            TEM_Avg_num_mean = np.mean(TEM_Avg_num) #平均气温的均值
            TEM_Avg_num_std = np.std(TEM_Avg_num)   #平均气温的方差

            TEM_Min_num_mean = np.mean(TEM_Min_num) #最高气温的均值
            TEM_Min_num_std = np.std(TEM_Min_num)   #最高气温的方差

            TEM_Wencha_mean = np.mean(TEM_Wencha) #温差的均值
            TEM_Wencha_std = np.std(TEM_Wencha)   #温差的方差

            PRS_num_mean = np.mean(PRS_num) #地面气压的均值
            PRS_num_std = np.std(PRS_num)   #地面气压的方差

            RHU_num_mean = np.mean(RHU_num) #相对湿度的均值
            RHU_num_std = np.std(RHU_num)   #相对湿度的方差

            PRE_num_mean = np.mean(PRE_num) #降水量的均值
            PRE_num_std = np.std(PRE_num)   #降水量的方差

            WIN_Max_num_mean = np.mean(WIN_Max_num) #最大风速的均值
            WIN_Max_num_std = np.std(WIN_Max_num)   #最大风速的方差

            WIN_S_num_mean = np.mean(WIN_S_num) #平均风速的均值
            WIN_S_num_std = np.std(WIN_S_num)   #平均风速的方差

            WIN_deg_num_mean = np.mean(WIN_deg_num) #风向角度的均值
            WIN_deg_num_std = np.std(WIN_deg_num)   #风向角度的方差

            SSD_num_mean = np.mean(SSD_num) #日照时间的均值
            SSD_num_std = np.std(SSD_num)   #日照时间的方差
            
            WIN_Scale_num_mean = np.mean(WIN_Scale_num) #风力等级的均值
            WIN_Scale_num_std = np.std(WIN_Scale_num)   #风力等级的方差

            leafblight = one_line[7] #大斑病
            lodging_rate = one_line[8] #倒伏率
            reversal_rate = one_line[9] #倒折率
            grey_speck_disease = one_line[10] #灰斑病
            plant_height = one_line[11] #株高
            Ear_height = one_line[12] #穗位高

            # 单位不统一处理
            # if float(plant_height) < 10:
            #     plant_height = float(plant_height) * 100

            # if Ear_height<10:
            #     Ear_height *= 100

            miss_rate = one_line[14] #空杆率
            growth_duration = one_line[15] #生育期
            panicle_rot = one_line[16] #穗腐病
            grains_weigh  = one_line[17] #百粒重
            spike_length = one_line[18] #穗长
            Bald_tip_length = one_line[19] #秃尖长
            fruit_spike_weight = one_line[20] #果穗鲜重
            yield_per_unit_area = one_line[21] #亩产
            production_compared = one_line[22] # 比对照组增减产
            Review_status = one_line[23] # 评审状态
            csv_writer.writerow([corn_num,TEM_Max_num_mean,TEM_Max_num_std,TEM_Avg_num_mean,TEM_Avg_num_std,\
                TEM_Min_num_mean,TEM_Min_num_std,TEM_Wencha_mean,TEM_Wencha_std,PRS_num_mean,PRS_num_std,\
                    RHU_num_mean,RHU_num_std,PRE_num_mean,PRE_num_std,WIN_Max_num_mean,WIN_Max_num_std,\
                        WIN_S_num_mean,WIN_S_num_std,WIN_deg_num_mean,WIN_deg_num_std,SSD_num_mean,SSD_num_std,\
                            WIN_Scale_num_mean,WIN_Scale_num_std,leafblight,lodging_rate,reversal_rate,\
                                grey_speck_disease,plant_height,Ear_height,miss_rate,growth_duration,panicle_rot,\
                                    grains_weigh,spike_length,Bald_tip_length,fruit_spike_weight,yield_per_unit_area,\
                                        production_compared, Review_status])
    corn_num = corn_num+1
    print(corn_num)
    
