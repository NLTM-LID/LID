import os # interation with the OS

def extract_BNF(file):
     csv_fname = "./extracted_BNF/feature_extract_file"
     csv_fname_with_ext = csv_fname+".csv"
     #### To remove the old file
     os.remove(csv_fname_with_ext)
     os.system("./BUT/mkhaudio2bottleneck.py BabelMulti {} {}".format(file, csv_fname))
     return csv_fname_with_ext