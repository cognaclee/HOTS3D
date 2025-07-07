# file path
input_PNG_file_path = '/home/lwh/code/PureCLIPNeRF/logs/shapenet/png'  
input_text_file_path = '/home/lwh/code/shap-e/PureCLIPNeRF200.txt'
output_PNG_name_file_path ='/home/lwh/code/shap-e/PureCLIPNeRF_PNG_Name200.txt'  
output_Text_name_file_path ='/home/lwh/code/shap-e/PureCLIPNeRF_Text_Name200.txt' 

with open(input_text_file_path, "r") as text_file:
    text_data = text_file.read().splitlines()


with open(output_PNG_name_file_path, "w") as fp:
    for k in range(0,200):
        for i in range(0,20):      
            fp.write(input_PNG_file_path+'/'+str(k)+'/'+str(i).zfill(3)+'.png'+'\r\n')
        
    

with open(output_Text_name_file_path, "w") as fp:
        for i, line in enumerate(text_data):#50
            for m in range(0,20):
                fp.write(line+'\r\n')              
                