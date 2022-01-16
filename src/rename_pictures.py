import os
os.getcwd()
collection = "/home/zz/Closet/database/raw_data/other_google_tie/"
for i, filename in enumerate(os.listdir(collection)):
    os.rename(collection + filename, collection + "other_ti" + str(i)  + ".jpg")
