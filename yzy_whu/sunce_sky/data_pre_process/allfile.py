import os
import csv

address = "./expression"

#count=0
#dicta=
shotlist=[]
for root,dirs,files in os.walk(address): 
    for file_name in dirs: 
        #count+=1
        #num=0
        #dicta['others']=dicta['others']+1
        #dicta['crying']=dicta['crying']+1
        #dicta['shouting']=dicta['shouting']+1
        #dicta['others']=dicta['others']+1
        for x, ys, txt_names in os.walk(os.path.join(address,file_name)):
            for txt_name in txt_names: 
                #if num%4==0:
                    #dicta['others']=dicta['others']+1
                #num+=1
                txt_path=os.path.join(file_name, txt_name)
                txt_path=file_name+'/'+txt_name
                shotlist.append(txt_path)
                
                
with open('predict.csv', mode='w', newline='') as csv_p:
    fieldnames = ['shot','shot2']
    writer = csv.DictWriter(csv_p, fieldnames=fieldnames)
    writer.writeheader()
    for i in shotlist:
        print("shot: {}".format(i))
        writer.writerow({'shot':i,'shot2':i})
    csv_p.close()


