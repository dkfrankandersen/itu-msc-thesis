import h5py
import os
import sys
import traceback

def write_attributes():
    use_input_path = "no"
    path_file = "/mnt/e/repository/itu/thesis/itu-msc-thesis/src/util"
    if len(sys.argv) < 2:
        print("Full filepath missing as args input")
        use_input_path = input(f"Use standard dir: {path_file} ? [yes/no]: ")
        if use_input_path == "no":
            path_file = input("Input path for *.hdf5 files: ")
            use_input_path = input(f"Use input dir: {path_file} ? [yes/no]: ")
    else:
        path_file = sys.argv[1]
        use_input_path = input(f"Use input dir: {path_file} ? [yes/no]: ")

    print(use_input_path)
    if use_input_path == "yes" and not path_file == "":
        print(f"path_file: {path_file}")
        for root, _, files in os.walk(path_file):
            print(root)
            for fn in files:
                print(f"fn: {fn}")
                if os.path.splitext(fn)[-1] != '.hdf5':
                    continue
                try:
                    f = h5py.File(os.path.join(root, fn), 'r+')
                    existing_attrs = dict(f.attrs)
                    print("Attributes in file:")
                    print(existing_attrs)

                    attributes = f['attributes'][0]
                    print("Found attributes as dataset in file")
                    print(attributes)
                    answer = input("Insert attributes? [yes/no]: ")
                    if answer == "yes":
                        print("Writing attributes into file as attributes..")
                        
                        for attribute in attributes:
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

                        print("Done writing attributes to file")
                    else:
                        print("No work done, exiting...")
                    f.close()
                except:
                    print('Was unable to read', path_file)
                    traceback.print_exc()
    else:
        print("No work done, buy...")
write_attributes()