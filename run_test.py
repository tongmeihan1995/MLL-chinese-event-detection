#conding=utf8  
import os 
g = os.walk(r"data/ace_data_LINYANKAI/type_test")  
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        print(os.path.join(path, file_name))
        nn=os.path.join(path,file_name)
        os.system("CUDA_VISIBLE_DEVICES=5 python3 main.py --status test  --savemodel 'data/MLL_saved_model.30.model' --savedset 'data/MLL_saved_model.dset' --test "+nn)
