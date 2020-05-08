import pydicom
import numpy as np
import pandas as pd
import os

len_size=100

def read_lable():
    labels0_df = pd.read_excel('./lable/TCIA CTC no polyp found.xls')

    labels0_df.rename(columns={'TCIA Patient ID': 'TCIA Number'}, inplace=True)

    labels1_df = pd.read_excel('./lable/TCIA CTC 6 to 9 mm polyps.xls')

    labels2_df = pd.read_excel('./lable/TCIA CTC large 10 mm polyps.xls')


    labels_df = pd.concat([labels0_df, labels1_df, labels2_df], axis=0, ignore_index=True)

    labels_df = labels_df.iloc[:, [0, 7]]
    labels_df.fillna(0, inplace=True)
    labels_df.rename(columns={1.5: 'Label'}, inplace=True)
    labels_df['TCIA Number'] = labels_df['TCIA Number'].str.replace('.', '_')

    # Get names of indexes for which column Age has value 16 98 88
    indexNames = labels_df[(labels_df['Label'] == 16.0) | (labels_df['Label'] == 88.0) | (labels_df['Label'] == 98.0)].index
    # Delete these row indexes from dataFrame
    labels_df.drop(indexNames, inplace=True)

    labels_df['Class'] = np.where(labels_df['Label'] == 0.0, 0, 1)
    labels_df
    dic={}
    s=len(labels_df.values)
    for i in range(len(labels_df.values)):
        dic[labels_df.values[i][0]]=labels_df.values[i][2]
    return dic


def read_data2txt(path_path,lable_data,save_path):
    if os.path.isdir(path_path):
        for name in os.listdir(path_path):
            if name.endswith('dcm'):
                if path_path.lower().find('supine')!=-1 or path_path.lower().find('3d')!=-1\
                        or path_path.lower().find('colon')!=-1:
                    if len(os.listdir(path_path))>100:
                        path=path_path.split('/')[4].replace('.', '_')
                        lable=str(lable_data[path])
                        output=open(save_path,mode='a')
                        output.write(path_path+'/'+lable+'/'+str(len(os.listdir(path_path)))+'\n')
                        output.close()
                break
            else:
                if path_path.lower().find('__macosx')==-1:
                    read_data2txt(os.path.join(path_path,name), lable_data,save_path)


def read_data(path):
    ds = pydicom.read_file(path)
    img = ds.pixel_array
    return img

def get_frame_data(video):
    label = int(video.split('/')[-2])
    video_name = '/'.join(video.split('/')[:-2])
    image_data = []
    data=sorted(os.listdir(video_name))
    for i in range(len_size):
        image_name=data[int(i*int((len(data)-1)/len_size))]
        image = np.resize(read_data(os.path.join(video_name, image_name)),new_shape=(112,112))
        image = np.array(image)
        image=image[:,:,np.newaxis]
        image_data.append(image)
    return image_data,label

def get_data(lines,batch_size,i):
    video_path = lines[i*batch_size:(i+1)*batch_size]
    label_data = []
    video_data = []
    for video in video_path:
        image_data,label=get_frame_data(video)
        label_data.append(label)
        video_data.append(np.array(image_data).transpose(3,0,1,2))
    return video_data,label_data
