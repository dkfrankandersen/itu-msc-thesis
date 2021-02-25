import h5py
import os
import sys
import traceback

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def write_attributes(f):
    for attribute in f['attributes'][0]:
        if isinstance(attribute, bytes):
            attribute = attribute.decode('ascii', 'replace')
        (key, datatype, value) = attribute.split(":")
        if datatype == "int":
            val = int(value)
        elif datatype == "float":
            val = float(value)
        elif datatype == "bool":
            val = True if value == "true" else False
        else:
            val = str(value)
        f.attrs[key] = val

def color_text(text):
    return style.GREEN + text + style.YELLOW

def convert():
    print(style.MAGENTA + "###########################################################")
    print("### "+style.GREEN+f"ENTERING PYTHON3 ({os.path.basename(__file__)})"+style.MAGENTA+"           ###")
    if len(sys.argv) < 2:
        print("###    Missing as args, expected a hdf5 file!")
    else:
        path_file = sys.argv[1]        
        if not os.path.exists(path_file):
            print("###    File not found            ###")
        elif not path_file.endswith('.hdf5'):
            print("###    File is not hdf5            ###")
        else:
            try:
                f = h5py.File(path_file, 'r+')
                if "attributes" in f:
                    print("###    "+style.GREEN+"1. Found attributes as dataset in file"+style.MAGENTA+"           ###")
                    
                    print("###    "+style.GREEN+"2. Writing attributes into file as attributes.."+style.MAGENTA+"  ###")
                    write_attributes(f)
                    print("###  ------------------------")
                    print("### "+style.GREEN+f"{dict(f.attrs)}"+style.MAGENTA+"")
                    print("###  ------------------------")
                    print("###    "+style.GREEN+"3. Done writing attributes to file"+style.MAGENTA+"               ###")
                else:
                    print("###    "+style.GREEN+"No attributes dataset in file"+style.MAGENTA+"           ###")
                f.close()
            except:
                print(f"###    "+style.GREEN+"Was unable to read {path_file}"+style.MAGENTA+"            ###")
                traceback.print_exc()  
    print("### "+style.GREEN+"LEAVING PYTHON3"+style.MAGENTA+"                                     ###")
    print("###########################################################" + style.RESET)
convert()