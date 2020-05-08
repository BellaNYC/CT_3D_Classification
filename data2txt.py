import read_data

lable_data=read_data.read_lable()
read_data.read_data2txt('./data/train/',lable_data,'./data/train_data.txt')
read_data.read_data2txt('./data/test/',lable_data,'./data/test_data.txt')
read_data.read_data2txt('./data/val/',lable_data,'./data/val_data.txt')