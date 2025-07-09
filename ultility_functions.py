import pandas as pd
import numpy as np
import bisect
import molmass
from itertools import product
from pyopenms import ElementDB, EmpiricalFormula



class DictListIndexWrapper(object):
    #function originally from Elvis for IsoMS and back search improvement 
    def __init__(self, dict_list, key, check_sorted=False):
        self.dict_list = dict_list
        self.key = key
        
        if check_sorted:
            if not all(dict_list[i][key] <= dict_list[i+1][key] for i in range(len(dict_list) - 1)):
                self.dict_list.sort(key=lambda d: d[key])
            
    def __getitem__(self, index):
        return self.dict_list[index][self.key]
    def get(self, index, key):
        return self.dict_list[index][key]
    def __len__(self):
        return self.dict_list.__len__()
    
    def _get_boundary_indexes(self, lower_limit, upper_limit, greater_or_equal=False, less_than_or_equal=False):
        '''Return the indexes for self that are at the edge of the lower and upper limits.
        
        Return (-1, -2) if no such indexes exist. (return -2 for upper_index, so that range(-1, -2+1) gives an empty list.)
        '''
        try:
            if not greater_or_equal:
                lower_index = gt_index(self, lower_limit)
            else:
                lower_index = gt_or_eq_index(self, lower_limit)
                
            if not less_than_or_equal:
                upper_index = lt_index(self, upper_limit)
            else:
                upper_index = lt_or_eq_index(self, upper_limit) 
        except ValueError:
            lower_index = -1
            upper_index = -2
        return lower_index, upper_index    
    
    def get_values_between(self, lower_limit, upper_limit, return_generator=False, greater_or_equal=False, less_than_or_equal=False):
        '''
        Get key values between a lower and upper limit.
        
        return_generator - whether to return result as generator or a list. Default False.'''
        lower_index, upper_index = self._get_boundary_indexes(lower_limit, upper_limit, greater_or_equal=greater_or_equal, 
                                                              less_than_or_equal=less_than_or_equal)
        
        if return_generator:
            return (self[i] for i in range(lower_index, upper_index+1))
        else:
            return [self[i] for i in range(lower_index, upper_index+1)]
        
    def get_items_between(self, lower_limit, upper_limit, return_generator=False, greater_or_equal=False, less_than_or_equal=False):
        '''Get original items between a lower and upper limit'''
        lower_index, upper_index = self._get_boundary_indexes(lower_limit, upper_limit, greater_or_equal=greater_or_equal, 
                                                              less_than_or_equal=less_than_or_equal)
        
        if return_generator:
            return (self.dict_list[i] for i in range(lower_index, upper_index+1))
        else:
            return [self.dict_list[i] for i in range(lower_index, upper_index+1)]
        
    def get_indexes_between(self, lower_limit, upper_limit, return_generator=False, greater_or_equal=False, less_than_or_equal=False):
        '''Get original indexes between a lower and upper limit'''
        lower_index, upper_index = self._get_boundary_indexes(lower_limit, upper_limit, greater_or_equal=greater_or_equal, 
                                                              less_than_or_equal=less_than_or_equal)
        
        # Note: self.sorted_indexes contains numpy.int64, convert to int before giving back.
        if return_generator:
            return (i for i in range(lower_index, upper_index+1))
        else:
            return [i for i in range(lower_index, upper_index+1)]
        
    def get_dict_list(self):
        return self.dict_list
    
    def get_closest_value(self, value):
        index = bisect.bisect_left(self, value)
        
        if index == 0:
            return self[0]
        elif index == len(self):
            return self[-1]
        else:
            before = self[index - 1]
            after = self[index]
            
            if after - value < value - before:
                return after
            else:
                return before
            
    def get_closest_item(self, value):
        index = bisect.bisect_left(self, value)
        
        if index == 0:
            return self.dict_list[0]
        elif index == len(self):
            return self.dict_list[-1]
        else:
            before = self[index - 1]
            after = self[index]
            
            if after - value < value - before:
                return self.dict_list[index]
            else:
                return self.dict_list[index - 1]
    
class SortedBisectableWrapper(DictListIndexWrapper):
    '''
    A wrapper for a list of dicts that makes it useable with the bisect library for an unsorted field.
    
    Allows bisect (or other libraries) to treat the unsorted list of dicts as if it was just a sorted list of the key field. 
    
    Note: Right now self.sorted_indexes is converted from numpy.int64 to int when returned. Could alternatively store as int instead.
    Unsure of what which way would be more efficient in practice.
    '''
    def __init__(self, dict_list, key):
        super(SortedBisectableWrapper, self).__init__(dict_list, key, check_sorted=False)
        
        self.sorted_indexes = np.argsort(DictListIndexWrapper(dict_list, key, check_sorted=False)) # These indexes are numpy.int64, should be converted to int when returned.
        
    def __getitem__(self, index):
        return self.dict_list[self.sorted_indexes[index]][self.key]
    
    def get(self, index, key):
        return self.dict_list[self.sorted_indexes[index]][key]
        
    def get_items_between(self, lower_limit, upper_limit, return_generator=False, greater_or_equal=False, less_than_or_equal=False):
        '''Get original items between a lower and upper limit'''
        lower_index, upper_index = self._get_boundary_indexes(lower_limit, upper_limit, greater_or_equal=greater_or_equal, 
                                                              less_than_or_equal=less_than_or_equal)
        
        if return_generator:
            return (self.dict_list[self.sorted_indexes[i]] for i in range(lower_index, upper_index+1))
        else:
            return [self.dict_list[self.sorted_indexes[i]] for i in range(lower_index, upper_index+1)]
        
    def get_indexes_between(self, lower_limit, upper_limit, return_generator=False, greater_or_equal=False, less_than_or_equal=False):
        '''Get original indexes between a lower and upper limit'''
        lower_index, upper_index = self._get_boundary_indexes(lower_limit, upper_limit, greater_or_equal=greater_or_equal, 
                                                               less_than_or_equal=less_than_or_equal)
        
        # Note: self.sorted_indexes contains numpy.int64, convert to int before giving back.
        if return_generator:
            return (int(self.sorted_indexes[i]) for i in range(lower_index, upper_index+1))
        else:
            return [int(self.sorted_indexes[i]) for i in range(lower_index, upper_index+1)]

def lt_index(input_list, x):
    '''Return rightmost index less than x. Expects a sorted list.'''
    i = bisect.bisect_left(input_list, x)
    
    if i:
        return i-1
    raise ValueError   

def gt_index(input_list, x):
    '''Return leftmost index greater than x. Expects a sorted list.'''
    i = bisect.bisect_right(input_list, x)
    
    if i != len(input_list):
        return i
    raise ValueError    
                    
def lt_or_eq_index(input_list, x):
    '''Return rightmost index less than or equal to x. Expects a sorted list.'''
    i = bisect.bisect_right(input_list, x)
    
    if i:
        return i-1
    raise ValueError

def gt_or_eq_index(input_list, x):
    '''Return leftmost index greater than or equal to x. Expects a sorted list.'''
    i = bisect.bisect_left(input_list, x)
    
    if i != len(input_list):
        return i
    raise ValueError



def get_formula_string(counts ,elements=["C", "H", "N", "O", "S"]):
    return ''.join(f"{el}{n}" for el, n in zip(elements, counts) if n > 1)

def compute_formula_from_mz(target, tol=0.005, tol_unit="Da", mode="positive"):
    # start_time = time.time()
    mass_dict = {e: ElementDB().getElement(e).getMonoWeight() for e in ["C", "H", "N", "O", "S"]}
    max_atoms = {'C': 50, 'H': 100, 'N': 10, 'O': 20, 'S': 5}
    elements = list(mass_dict.keys())
    if tol_unit == "ppm":
        tol = target * tol / 1e6

    if mode == "positive":
        target -= 1.00727646
    elif mode == "negative":
        target += 1.00727646
    else:
        raise ValueError("Invalid ionization mode. Must be 'positive' or 'negative'.")

    best = []
    def dfs(level, counts, mass):
        if abs(mass - target) <= tol:
            formula_str = get_formula_string(counts)
            ef = EmpiricalFormula(formula_str)
            best.append((ef.toString(), ef.getMonoWeight()))
            return
        if level == len(elements):
            return
        element = elements[level]
        
        for i in range(max_atoms[element] + 1):
            new_mass = mass + i * mass_dict[element]
            if new_mass - target > tol:
                break
            dfs(level + 1, counts + [i], new_mass)
    dfs(0, [], 0.0)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
    best.sort(key=lambda x: abs(x[1] - target))
    if not best:
        print(f"No formulas found within the specified tolerance for m/z {target}.")
        return [None,None]
    else:
        return best[0]


class custom_aligned_peaks():
    def __init__(self):
        self.mz_tolerance=5.0
        self.mz_tolerance_unit="ppm"
        self.rt_tolerance=5.0

        self.aligned_peaks = []
        self.filenames = []
        self.signal_type="area"
        self.reference_map_index=None
        self.organized = False

    def set_mz_tolerance(self, t):
        self.mz_tolerance = float(t)

    def set_rt_tolerance(self, t):
        self.rt_tolerance = float(t)

    def set_filenames(self, names):
        self.filenames = names

    def set_mz_tolerance_unit(self, unit):
        self.mz_tolerance_unit = unit
        if unit not in ["ppm", "Da"]:
            raise ValueError("Invalid mz_tolerance_unit. Must be 'ppm' or 'Da'.")

    def get_mz_tolerance_da(self, taget_mz):
        if self.mz_tolerance_unit == "ppm":
            return taget_mz * self.mz_tolerance / 1e6
        elif self.mz_tolerance_unit == "Da":
            return self.mz_tolerance

    def get_df(self, fill_na=False, fill_value=np.nan):
        columns = ["Average RT", "Average mz","Missing"] + self.filenames

        if not self.organized:
            self.organize_aligned_peaks()
        df = pd.DataFrame(self.aligned_peaks, columns=columns)
        if fill_na:
            df.fillna(value=fill_value, inplace=True)

        df.rename(columns = {"Average RT": "RT", "Average mz": "mz"}, 
                  inplace=True)
        return df
    
    def get_existing_peak_list(self):
        return self.aligned_peaks

    def initialize_record(self, reference_map, ref_index):
        if isinstance(reference_map, pd.DataFrame):
            reference_map.rename({c:c.lower()for c in reference_map.columns}, inplace=True)
            ref_dict_list = reference_map.to_dict(orient='records')
        elif isinstance(reference_map, list):
            ref_dict_list = reference_map
        elif reference_map == None:
            ref_dict_list = []
        else:
            raise ValueError("Invalid reference input for custom_aligned_peaks")

        if not len(self.filenames):
            Warning("No filename input yet. only input value recorded and no placeholder added in the record!")
            self.filenames = ["Dummy_sample_"+str(ref_index)]

        if len(ref_dict_list):
            for row in ref_dict_list:
                void_rt = [np.nan]* len(self.filenames)  
                void_rt[ref_index] =row["RT"]

                void_mz = [np.nan] * len(self.filenames)
                void_mz[ref_index] =row["mz"]

                r={"RT":void_rt, "mz":void_mz}
                for ind, filename in enumerate(self.filenames):
                    if ind != ref_index:
                        r[filename]= np.nan
                    else:
                        r[filename] = row[self.signal_type]
                self.aligned_peaks.append(r)
        self.reference_map_index = ref_index


    def align_peaks(self, new_dict_list, current_file_index):
        if self.reference_map_index == current_file_index:
            return
        if not isinstance(current_file_index, int):
            try:
                current_file_index = self.filenames.index(current_file_index)
            except ValueError:
                raise ValueError("current_file_index must be an integer or a valid filename from the filenames list")
        if current_file_index > len(self.filenames):
            raise ValueError(" Index over current file number stored. Please check again or update filenames")
        if not "area" in new_dict_list[0].keys():
            if "intensity" in new_dict_list[0].keys():
                Warning("No area record found in the individual data. Intnsity will be used instead!")
                self.signal_type = "intensity"
            else:
                raise ValueError("Incoming data (list of dict) must use either area or intensity as signal value but none was found")
        
        if isinstance(new_dict_list, pd.DataFrame):
            new_dict_list.sort_values(by=["RT","mz"], inplace=True)
            new_dict_list.rename({c:c.lower()for c in new_dict_list.columns}, inplace=True)
            new_dict_list = new_dict_list.to_dict(orient='records')
        elif isinstance(new_dict_list, list):
            if not all(isinstance(i, dict) for i in new_dict_list):
                raise ValueError("Input data must be a list of dictionaries")
            new_dict_list.sort(key=lambda x: (x["RT"], x["mz"]))
        else:
            raise ValueError("Invalid input for align_peaks. Must be a list of dictionaries or a pandas DataFrame.")
        mz_wrapper = SortedBisectableWrapper(new_dict_list, 'mz')

        print(f"Aligning peaks for file {self.filenames[current_file_index]} with {len(new_dict_list)} peaks against {len(self.aligned_peaks)} existing peaks.")

        matched_index = set()
        existing_peak_list = self.aligned_peaks
        for row in existing_peak_list:
            # iterate through all currently existing peaks and try to find match in the incoming data
            # each row is a dictionary of {"RT":void_rt, "mz":void_mz, filename:intensity/area}
            target_mz = np.nanmean(row["mz"])
            target_rt = np.nanmean(row['RT'])

            mz_tolerance = self.get_mz_tolerance_da(target_mz)
            lower_bound = target_mz - mz_tolerance
            upper_bound = target_mz + mz_tolerance

            row_indexes_within_range = mz_wrapper.get_indexes_between(lower_bound, upper_bound, less_than_or_equal=True, greater_or_equal=True)
            rows_within_range = [new_dict_list[i] for i in row_indexes_within_range if 
                                 target_rt- self.rt_tolerance <  new_dict_list[i]["RT"] and new_dict_list[i]["RT"]< target_rt+self.rt_tolerance]

            if rows_within_range:
                if len(rows_within_range) >1:
                    # somehow there's multiple matches within the rt range
                    rt_diff = [abs(r["RT"] - target_rt) for r in rows_within_range]
                    best_match_row = rows_within_range[np.argmin(rt_diff)]
                    row["RT"][current_file_index] = best_match_row["RT"]
                    row["mz"][current_file_index] = best_match_row["mz"]
                    row[self.filenames[current_file_index]] = best_match_row[self.signal_type]
                    matched_index.add(row_indexes_within_range[np.argmin(rt_diff)])

                elif len(rows_within_range) == 1:
                    best_match_row = rows_within_range[0]
                    if target_rt- self.rt_tolerance <  target_rt <target_rt+self.rt_tolerance:
                        row["RT"][current_file_index] = best_match_row["RT"]
                        row["mz"][current_file_index] = best_match_row["mz"]
                        row[self.filenames[current_file_index]] = best_match_row[self.signal_type]
                        matched_index.add(row_indexes_within_range[0])
                else:
                    continue
        
        for ind, peak in enumerate(new_dict_list):
            # iterate through the new peaks and check if they are already in the existing peak list
            if ind not in matched_index:
                # this peak is not matched with any existing peak, so add it as a new peak
                new_peak = {
                    "RT": [np.nan] * len(self.filenames),
                    "mz": [np.nan] * len(self.filenames),
                    self.filenames[current_file_index]: np.nan
                }
                new_peak["RT"][current_file_index] = peak["RT"]
                new_peak["mz"][current_file_index] = peak["mz"]
                new_peak[self.filenames[current_file_index]] = peak[self.signal_type]
                self.aligned_peaks.append(new_peak)

        # Now we have aligned the peaks, we can organize them
        self.organize_aligned_peaks()

                    
    def organize_aligned_peaks(self):
        
        for peak in self.aligned_peaks:
            peak["Average RT"] = round(np.nanmean(peak["RT"]),3)
            peak["Average mz"] = round(np.nanmean(peak["mz"]),3)
            peak["Missing"] = round(peak["RT"].count(np.nan) + peak["RT"].count(0)/len(peak["RT"])*100,2)
            if peak["Missing"] >= 100:
                self.aligned_peaks.remove(peak)

        self.aligned_peaks = sorted(self.aligned_peaks, key=lambda x: (x["Average RT"], x["Average mz"]))
        self.organized = True

# def set_parameters(parameter_handle, base_func, file=None, kwarg={}):
#     """

#     for default settings, currently decided in stored in json file 
#     which is in the format of {function_name: {parameter_name: value}}

#     However, for kwarg, this is only {parameter_name: value},
#     generally this  shouldnt be a problem but not entirely sure
#     """
    
#     if file is not None:
#         if file.endswith(".json"):
#             with open(file, 'r') as f:
#                 default_settings = json.load(f)
#         elif file.endswith(".txt"):
#             with open(file, 'r') as f:
#                 default_settings = f.read()
#         else:
#             raise ValueError("Unsupported file format. Please provide a .json file.")

#     if base_func=="ElutionPeakDetection":
#         if file is not None:
#             params = parameter_handle.getDefaults(file)
#         elif kwarg:
#             params = parameter_handle.getDefaults()
#             for key, value in kwarg.items():
#                 if params.exists(key):
#                     params.setValue(key, value)

#     if base_func =="MassTraceDetection" and "MassTraceDetection" in kwarg.keys():
#         mtd_params = parameter_handle.getDefaults()
#         settings = kwarg["MassTraceDetection"]

#         for s in settings.keys():
#             if not s in kwarg.keys()

#         mtd_params.setValue("mass_error_ppm", 10.0)
#         if "noise_threshold_int" in kwarg.keys():
#             mtd_params.setValue("noise_threshold_int", kwarg{noise_threshold_int})
#         mtd_params.setValue("min_spectra_needed", 2)
#         mtd_params.setValue("trace_termination_criterion", "outlier")

if __name__ == "__main__":
    data = {
        'rt': [10.1, 12.5, 15.2, 11.8, 14.5, 13, 16, 11.5, 12.7, 10.5],
        'mz': [199.998, 200.002, 200.006, 200.001, 200.004, 199.997, 200.1, 199.9, 200.003, 200.005],
        'intensity': [1000, 1500, 2000, 1200, 1800, 900, 1700, 1300, 1100, 1600]
    }
    df = pd.DataFrame(data)
    data_list = df.to_dict(orient='records')
    mz_wrapper = SortedBisectableWrapper(data_list, 'mz')

    target_mz = 200.003
    mz_tolerance = 0.005
    lower_bound = target_mz - mz_tolerance
    upper_bound = target_mz + mz_tolerance
    rows_within_range = mz_wrapper.get_items_between(lower_bound, upper_bound, less_than_or_equal=True, greater_or_equal=True)
    result_df = pd.DataFrame(rows_within_range)

    print(result_df)

