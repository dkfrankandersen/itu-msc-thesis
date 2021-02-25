import h5py
import os
import sys
import traceback

def write_attributes(f, attributes):
    for attribute in attributes:
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

        print(f"{key} : {val} (type: {datatype})")
        f.attrs[key] = val

def convert():
    use_input_path = "no"
    root_path = ""
    if len(sys.argv) < 2:
        print("Full filepath missing as args input")
    else:
        filename = sys.argv[1]
        use_input_path = input(f"Convert file: {root_path}{filename} ? [yes/no]: ")
        
        if use_input_path == "yes":
            path_file = os.path.join(root_path, filename)
            print(path_file)
            if os.path.exists(path_file):
                print("file found")
            if filename.endswith('.hdf5'):
                print("file is hdf5")
                try:
                    f = h5py.File(path_file, 'r+')
                    existing_attrs = dict(f.attrs)
                    print("Attributes in file:")
                    print(existing_attrs)

                    if "attributes" in f:
                        print("Found attributes as dataset in file")
                        attributes = f['attributes'][0]
                        print(attributes)
                        answer = input("Insert attributes? [yes/no]: ")
                        if answer == "yes":
                            print("Writing attributes into file as attributes..")
                            
                            write_attributes(f, attributes)

                            print("Done writing attributes to file")
                        else:
                            print("No work done, exiting...")
                    else:
                        print("No attributes dataset in file")

                    f.close()
                except:
                    print('Was unable to read', path_file)
                    traceback.print_exc()
            else:
                print("Not hdf5?")

convert()