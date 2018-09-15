# coding: utf-8
cropped_files = Path(image_dir).glob('**/*+crop*')

mask_files = []
#check for mask files and make a list without the mask suffix
for i, file in enumerate(Path(image_dir).glob('**/*mask*')):
    #print(file)
    base_fn = str(file).split('_IC_mask')[0]
    #print(base_fn)
    mask_files.append(base_fn)
    
cropped_files = []
#check for mask files and make a list without the mask suffix
for i, file in enumerate(Path(image_dir).glob('**/*+crop*')):
    #print(file)
    base_fn = str(file).split('_IC+crop')[0]
    #print(base_fn)
    cropped_files.append(base_fn)
    
match = set(mask_files) & set(cropped_files)
match
