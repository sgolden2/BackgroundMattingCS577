from os import listdir, rename
from os.path import isfile, join
# alpha_path = "kaggle\\kagglealpha"
fgr_path = "kaggle\\kaggleimg"
# alpha_files = [f for f in listdir(alpha_path)if isfile(join(alpha_path,f))]
fgr_files = [f for f in listdir(fgr_path)if isfile(join(fgr_path,f))]
for i in range(len(fgr_files)):
    # alpha_files[i] = fgr_files[i][:-4]
    fgr_files[i] = fgr_files[i][:-4]
# print(alpha_files[:10],fgr_files[:10])
for i,x in enumerate(fgr_files):
    offset = 254748
    # alpha_org = "kaggle\\kagglealpha\\"+x+".png"
    # alpha_fin = "output3\\pha\\"+str(offset+i)+".jpg"
    fgr_org = "kaggle\\kaggleimg\\"+x+".jpg"
    fgr_fin = "output3\\fgr\\"+str(offset+i)+".jpg"
    # rename(alpha_org,alpha_fin)
    rename(fgr_org,fgr_fin)