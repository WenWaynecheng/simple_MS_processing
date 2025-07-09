import csv,os


def convet_orbitrap_data(filepath, filename, output_path, min_rt=None, max_rt=None,min_int=1000):
    input = os.path.join(filepath, filename)
    output = os.path.join(output_path, filename.replace(".txt",".csv"))
    print("converting file: " + filename )
    with open(input, 'r') as f, open(output, 'w',newline='') as output_f:
        csv_w = csv.DictWriter(output_f, fieldnames=["No", "polarity", "rt", "mz", "intensity"])
        csv_w.writeheader()
        
        current_scan_number = -1 
        current_retention_time = None
        current_array_type = None
        current_polarity = "positive"


        for i, line in enumerate(f):
            stripped_line = line.strip()
            if stripped_line.startswith("chromatogramList"): #likely the end of chromatogram and remaining are other data such as pump pressure etc that we dont care about
                break

            if stripped_line.startswith("id: controllerType"): #Line looks like: "id: controllerType=0 controllerNumber=1 scan=15"
                    current_scan_number = stripped_line.split(" ")[3][5:]
            #in case polarity switches within a run        
            elif stripped_line.startswith("cvParam: negative scan"):
                current_polarity = "negative"
            elif stripped_line.startswith("cvParam: positive scan"):
                current_polarity = "positive"
            elif stripped_line.startswith("cvParam: scan start time"): #Line looks like: "cvParam: scan start time, 0.033787036, minute"
                    current_retention_time = round(float(stripped_line.split(" ")[4][:-1]) * 60,3) # Convert to second
                    if min_rt is not None and current_retention_time < min_rt:
                        continue
                    if max_rt is not None and current_retention_time > max_rt:
                        continue
            elif stripped_line.startswith("cvParam") and "array" in stripped_line:
                if stripped_line.startswith("cvParam: m/z array, m/z"):
                    current_array_type = "mz" 
                elif stripped_line.startswith("cvParam: intensity array, number of detector counts"):
                    current_array_type = "mz intensity"
                else:
                    current_array_type = None
            elif stripped_line.startswith("binary:") and current_array_type is not None:
                if current_array_type == "mz":
                    current_mz_list = stripped_line.split(" ")[2:]
                elif current_array_type == "mz intensity":
                    current_int_list = stripped_line.split(" ")[2:]
                    if len(current_mz_list) != len(current_int_list):
                        print(len(current_mz_list))
                        print(len(current_int_list))
                        raise RuntimeError("mz and intensity has different length at scan " + str(current_scan_number))
                    
                    for i, intensity in enumerate(current_int_list):
                        if float(intensity) > min_int:
                            csv_w.writerow({"No":current_scan_number, "polarity":current_polarity, "rt":current_retention_time, "mz":current_mz_list[i], "intensity":intensity})
    f.close()
    output_f.close()
        

if __name__ == "__main__":
    filepath = r"G:\My Drive\Master\Data\20260526 high temp\1_Converted\txt"
    output_path = r"G:\My Drive\Master\Data\20260526 high temp\1_Converted\csv"
    os.mkdir(output_path) if not os.path.exists(output_path) else None  # Create output directory if it doesn't exist
    for file in os.listdir(filepath):
        if file.endswith(".txt"):
            convet_orbitrap_data(filepath, file, output_path, min_rt=None, max_rt=None, min_int=1000)

    print("Finished converting files!")
    