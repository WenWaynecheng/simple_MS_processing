import os, glob, json
import pandas as pd, numpy as np
import pyopenms as oms
from scipy.stats import pearsonr, ttest_ind

from multivariate_analysis_tools import run_pca, run_UMAP
from ultility_functions import custom_aligned_peaks

def process_ms_data(input_path, output_path, group_file,
                    alignment_method="PoseClustering",
                    mass_tolerance=5.0, mass_tolerance_unit="ppm", min_intensity=500,
                    use_isotope_filter=True, use_blank_filter=True,blank_group_name="blank",
                    filter_cv_threhold=0.3, missing_rule="group", max_missing=0.75, do_blank_subtraction=False,
                    rt_tolerance=3, mass_precision=4,isotope_filter_correlation=0.8,
                    do_PCA=True, do_UMAP=False, visualization=True):
    
    """
    Take a folder containing mzML files and a grouping file, process the data to generate a feature map,
    align the feature maps, and perform optional isotope filtering, blank filtering, PCA, and UMAP.
    
    """
    logconfig = oms.LogConfigHandler()
    logconfig.setLogLevel("DEBUG")  # Set log level to ERROR to suppress warnings

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    grouping_dict, inverse_grouping = read_grouping_file(input_path, group_file, save = True, output_path=output_path)
    if alignment_method == "custom":
        mass_trace_df = process_lcms_mzml(input_path, output_path, noise_threshold_int=min_intensity, 
                                          return_mass_trace_only=True)
        print("Processing with custom alignment method...")

        aligned_map = run_custom_alignment(mass_trace_df, output_path, grouping_dict,
                                            mz_diff_window=mass_tolerance, 
                                            mz_diff_unit=mass_tolerance_unit)

    else:
        feature_map = process_lcms_mzml(input_path, output_path, min_intensity)
        aligned_map = align_feature_maps(feature_map, sorted(glob.glob(os.path.join(input_path, "*.mzML"))), output_path,
                                            align_algorithm=alignment_method, mz_diff_window=mass_tolerance, mz_diff_unit=mass_tolerance_unit)

    if use_isotope_filter:
        print("Performing isotope filtering...")
        filtered_feature_map = do_isotope_filter(aligned_map, output_path, rt_tolerance=rt_tolerance, 
                                                mass_tolerance=mass_tolerance, mass_tolerance_unit=mass_tolerance_unit,
                                                mass_precision=mass_precision, isotope_filter_correlation=isotope_filter_correlation,
                                                grouping=inverse_grouping)

    if blank_group_name in inverse_grouping.keys() and use_blank_filter:
        if len(inverse_grouping[blank_group_name]) >= 3 :
            print("Performing blank filtering...")
            filtered_feature_map = do_blank_filter(filtered_feature_map, grouping_dict, output_path, 
                                          blank_group_name=blank_group_name, filter_cv_threhold=filter_cv_threhold,
                                          max_missing=max_missing, do_blank_subtraction=do_blank_subtraction)
        else:
            print("Not enough blank samples to filter out peaks. Skipping blank filtering.")
    if do_PCA:
        print("Performing PCA...")
        run_pca(filtered_feature_map, grouping_dict,components=2, save_path=output_path, visualization=visualization)
        
    if do_UMAP:
        print("Performing UMAP...")
        run_UMAP(filtered_feature_map, grouping_dict, mz_col_index=1, rt_col_index=2,
                n_neighbors=15, min_dist=0.1, n_components=2, save_path=output_path,  visualization=visualization)



    print("processing done!")


def read_grouping_file(input_path, group_file, save=False, output_path = None):

    files = sorted([f.split(".")[0] for f in os.listdir(input_path) if f.endswith(".mzML")])

    if group_file is None:
        grouping_dict = {}
        for f in files:
            if "blank" in f.lower():
                grouping_dict[f] = "blank"
            else:
                grouping_dict[f] = "sample"
    elif group_file.endswith(".csv"):
        df = pd.read_csv(group_file)
        grouping_dict = df.set_index('sample')['group'].to_dict()
    elif group_file.endswith(".json"):
        with open(group_file, 'r') as f:
            grouping_dict = json.load(f)
    
    else:
        raise ValueError("Unsupported file format for group information file. Please provide a .csv or .json file.")
    # if len(info_dict) < len(files):
    #     grouping_dict={f:info_dict[f] for f in files}
    # else:
    #     grouping_dict = info_dict
    # print(grouping_dict)
    inverse_grouping = {}
    for sample, group in grouping_dict.items():
        if group not in inverse_grouping:
            inverse_grouping[group] = [sample]
        else:
            inverse_grouping[group].append(sample)

    if save and output_path:
        with open(os.path.join(output_path,"0_current_grouping_dict.json"), 'w') as f:
            json.dump(grouping_dict, f, indent=4)

    return grouping_dict, inverse_grouping


def process_lcms_mzml(input_path, output_path, noise_threshold_int=500.0, return_mass_trace_only=False, **kwargs):
    """
    Process LC-MS mzML files to detect mass traces and generate feature map for each mzml file.
    
    This was adopted from the OpenMS example for non target analysis
    https://pyopenms.readthedocs.io/en/latest/user_guide/untargeted_metabolomics_preprocessing.html

    However, this bypass the FeatureFindingMetabo() which results in much less peak detected
    instead, feature map is directly generated from the mass traces detected by ElutionPeakDetection()

    This means that we skipped the feature finding step, including isotope filtering,
    which will be handled in the next step
    
    """
    mzml_files = sorted(glob.glob(os.path.join(input_path, "*.mzML")))

    feature_maps = []  # To store FeatureMap objects for alignment

    if not isinstance(noise_threshold_int,float):
        noise_threshold_int = float(noise_threshold_int)

    for file in mzml_files:
        print("Processing file:", os.path.basename(file))
        exp = oms.MSExperiment()
        oms.MzMLFile().load(file, exp)  # load each mzML file to an OpenMS file format (MSExperiment)
        exp.sortSpectra(True)
        # print("Current file:", os.path.basename(file))
        # mass trace detection, this clean up the m/z data in each spectra
        mass_traces = []
        mtd = oms.MassTraceDetection()
        mtd_params = mtd.getDefaults()
        mtd_params.setValue("mass_error_ppm", 10.0)
        mtd_params.setValue("noise_threshold_int", noise_threshold_int)
        # mtd_params.setValue("min_spectra_needed", 2) #seems like an unknown parameter somehow
        mtd_params.setValue("trace_termination_criterion", "outlier")
        mtd.setParameters(mtd_params)
        mtd.run(exp, mass_traces, 0)
        # print("lenth of mass_traces", len(mass_traces))

        #construct the mass traces from each spectrum to basically EIC and peaks
        mass_traces_split = []
        mass_traces_final = []
        epd = oms.ElutionPeakDetection()
        # set_parameters(epd,"ElutionPeakDetection", kwarg=kwargs)
        epd_params = epd.getDefaults()
        epd_params.setValue("chrom_fwhm", 5.0)
        epd_params.setValue("width_filtering", "auto")
        epd_params.setValue("max_fwhm", 120.0)
        epd_params.setValue("min_fwhm", 0.02)
        epd_params.setValue("chrom_peak_snr", 1.0)
        epd.setParameters(epd_params)
        epd.detectPeaks(mass_traces, mass_traces_split)
        # print("lenth of mass_traces_split", len(mass_traces_split))

        if epd.getParameters().getValue("width_filtering") == "auto":
            epd.filterByPeakWidth(mass_traces_split, mass_traces_final)
        else:
            mass_traces_final = mass_traces_split

        rows = []
        for mt in mass_traces_final:
            row = {
                "mz": mt.getCentroidMZ(),
                "RT": mt.getCentroidRT(),
                "fwhm": mt.getFWHM(),
                "area": mt.getIntensity(True),
                "max_intensity": mt.getMaxIntensity(True),
                "QuantMethod": mt.getQuantMethod(),
                "number_of_peaks": mt.getSize()}
            rows.append(row)
        sub_output_path = os.path.join(output_path, "individual_feature_maps")
        os.makedirs(sub_output_path, exist_ok=True)

        if return_mass_trace_only:
            df = pd.DataFrame(rows)
            feature_maps.append(df)
        else:
            # split_tract_df = pd.DataFrame(rows)
            # split_tract_filename = os.path.join(sub_output_path, os.path.basename(file).replace(".mzML", "_mass_trace.csv"))
            # split_tract_df.to_csv(split_tract_filename, index=False)
            fm = oms.FeatureMap()
            for mt in mass_traces_final:
                f = oms.Feature()
                f.setRT(mt.getCentroidRT())
                f.setMZ(mt.getCentroidMZ())
                f.setIntensity(mt.getIntensity(True))  
                hull=mt.getConvexhull()
                f.setConvexHulls([hull])
                fm.push_back(f)

            fm.setUniqueIds()
            fm.setPrimaryMSRunPath([file.encode()])
            feature_maps.append(fm)

            df=fm.get_df(export_peptide_identifications=False)
            df.sort_values(by=["RT", "mz"], inplace=True)
            df.drop(columns=["quality"],inplace=True, errors='ignore')  
            column_names = df.columns.tolist()
            correct_columns = column_names[:5] +  sorted(column_names[5:])
            df = df[correct_columns]

        csv_output_path = os.path.join(sub_output_path, os.path.basename(file).replace(".mzML", "_feature_maps.csv"))
        for c in df.columns:
            if "." in c:
                df.rename(columns={c: c.split(".")[0]}, inplace=True)
        df.to_csv(csv_output_path, index=False)

    return feature_maps

def align_feature_maps(feature_maps, mzml_files, output_path, align_algorithm="PoseClustering", 
                       mz_diff_window =10.0, mz_diff_unit="ppm"):

    # Step 1. Feature Alignment
    # Which means different samples with possible RT shift are now on the same RT axis

    if align_algorithm == "PoseClustering":
        # PoseClustering is a method that clusters features based on their retention time and m/z values
        # however, a reference feature map is required (usually a known control sample or the largest feature map)
        # This is a simple linear alignment (an affine transformation to be exact)
        aligner = oms.MapAlignmentAlgorithmPoseClustering()

        ref_index = feature_maps.index(sorted(feature_maps, key=lambda x: x.size())[-1])

        # parameter optimization
        aligner_par = aligner.getDefaults()
        aligner_par.setValue("max_num_peaks_considered", -1)  
        aligner_par.setValue("pairfinder:distance_MZ:max_difference", 10.0)  
        aligner_par.setValue("pairfinder:distance_MZ:unit", "ppm")
        aligner_par.setValue("pairfinder:distance_RT:max_difference",90.0)  # Maximum RT difference in seconds
        aligner.setParameters(aligner_par)
        aligner.setReference(feature_maps[ref_index])
    
        # Create a copy of feature_maps to avoid modifying the original
        aligned_feature_maps = []
        # trafos = {}
        for i, feature_map in enumerate(feature_maps):
            if i == ref_index:
                # Reference map doesn't need transformation
                aligned_feature_maps.append(feature_map)
                continue
            # Create a copy of the feature map to avoid modifying the original
            feature_map_copy = oms.FeatureMap(feature_map)
            
            trafo = oms.TransformationDescription()
            aligner.align(feature_map_copy, trafo)
            
            # # Store transformation for mzML files
            # file_key = feature_map.getMetaValue("spectra_data")[0].decode()
            # trafos[file_key] = trafo
            
            # Apply transformation to the copied feature map
            transformer = oms.MapAlignmentTransformer()
            transformer.transformRetentionTimes(feature_map_copy, trafo, True)
            aligned_feature_maps.append(feature_map_copy)

        # Transform corresponding mzML files
        # for file in mzml_files:
        #     if file in trafos.keys():
        #         exp = oms.MSExperiment()
        #         oms.MzMLFile().load(file, exp)
        #         exp.sortSpectra(True)
        #         exp.setMetaValue("mzML_path", file)
                
        #         transformer = oms.MapAlignmentTransformer()
        #         trafo_description = trafos[file]
        #         transformer.transformRetentionTimes(exp, trafo_description, True)
                # Optionally save aligned mzML file
                # oms.MzMLFile().store(file[:-5] + "_aligned.mzML", exp)
        
        # Use aligned feature maps for further processing
        feature_maps = aligned_feature_maps
   
    elif align_algorithm == "KD":
        # ===========================================
        # CURRENTLY NOT WORKING
        # ===========================================

        # KD-tree based alignment algorithm
        # This algorithm uses a kd-tree to efficiently compute conflict-free connected components (CCC)
        raise RuntimeError("KD-tree based alignment is not implemented yet. Please use 'PoseClustering' for now.")
        # # Create KDTreeFeatureMaps object and add all feature maps
        # kdtree_feature_maps = oms.KDTreeFeatureMaps()
        # kdtree_feature_maps.addMaps(feature_maps)

        # # Set up the KD alignment algorithm
        # aligner = oms.MapAlignmentAlgorithmKD
        
        # # Align the feature maps
        # aligner.align(kdtree_feature_maps)
        
        # # Extract transformed feature maps
        # transformed_feature_maps = []
        # transformer = oms.MapAlignmentTransformer()

        # for i in range(kdtree_feature_maps.size()):
        #     current_map = kdtree_feature_maps.getFeatureMap(i)
        #     current_trafo = kdtree_feature_maps.getTransformation(i)
            
        #     # Create a copy to avoid modifying the original
        #     map_copy = oms.FeatureMap(current_map)
            
        #     # Apply the transformation to the current feature map
        #     # 'True' stores the original RT as a meta value ('original_RT')
        #     transformer.transformRetentionTimes(map_copy, current_trafo, True)
        #     transformed_feature_maps.append(map_copy)
            
        # # Use transformed feature maps for further processing
        # feature_maps = transformed_feature_maps
    
    else:
        raise ValueError("Unsupported alignment algorithm. Please use 'KD' or 'PoseClustering'.")

    # Step 2. Feature Linking
    # Feature linking is the process of linking similar feature between different runs 
    # the result is then combined into a consensus map

    feature_grouper = oms.FeatureGroupingAlgorithmKD()
    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()

    # Set up file descriptions
    for i, feature_map in enumerate(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_description.filename = os.path.basename(
            feature_map.getMetaValue("spectra_data")[0].decode()
        )
        file_description.size = feature_map.size()
        file_descriptions[i] = file_description

    # Group features across maps
    feature_grouper.group(feature_maps, consensus_map)
    consensus_map.setColumnHeaders(file_descriptions)
    consensus_map.setUniqueIds()

    # Convert to DataFrame and clean up
    consensus_map_df = consensus_map.get_df()
    column_names = consensus_map_df.columns.tolist()
    
    # Reorder columns: first 5 columns, then sorted remaining columns
    if len(column_names) > 5:
        column_names = column_names[:5] + sorted(column_names[5:])
        consensus_map_df = consensus_map_df[column_names]
    
    # Drop unnecessary columns if they exist
    columns_to_drop = ["sequence", "quality"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in consensus_map_df.columns]
    if existing_columns_to_drop:
        consensus_map_df.drop(columns=existing_columns_to_drop, inplace=True, errors='ignore')
    
    # Sort by RT and m/z
    consensus_map_df.sort_values(by=["RT", "mz"], inplace=True)
    consensus_map_df.reset_index(drop=True, inplace=True)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_path, "1_aligned_peaks_from_ElutionPeak.csv")
    consensus_map_df.to_csv(output_file, index=False)
    
    print(f"Alignment completed. Results saved to: {output_file}")
    print(f"Final consensus map contains {len(consensus_map_df)} features")
    
    # Check for negative retention times and warn if found
    if (consensus_map_df['RT'] < 0).any():
        print("Warning: Some features have negative retention times after alignment.")
        print("This might indicate issues with the alignment parameters or reference selection.")
    
    return consensus_map



def run_custom_alignment(feature_maps, output_path, input_filenames, mz_diff_window=10.0, mz_diff_unit="ppm"):
    
    if isinstance(input_filenames, dict):
        if isinstance(input_filenames[next(iter(input_filenames))], str):
            filenames = [v for v in input_filenames.keys()]
        else:
            filenames=[]
            for v in input_filenames.values():
                filenames.extend(v)
    
    reference_map_index = int(np.argmax([fm.shape[0] for fm in feature_maps]))
    CAF = custom_aligned_peaks()
    CAF.set_filenames(filenames)
    CAF.initialize_record(feature_maps[reference_map_index], reference_map_index)
    CAF.set_mz_tolerance(mz_diff_window)
    
    for i, df in enumerate(feature_maps):
        CAF.align_peaks(df.to_dict("records"), int(i))

    aligned_map = CAF.get_df()
    output_file = os.path.join(output_path, "1_Custom_aligned_peaks.csv")
    aligned_map.to_csv(output_file, index=False)
    return aligned_map

def do_isotope_filter(consensus_map, output_path, rt_tolerance=3.0, 
                    mass_tolerance=5.0, mass_tolerance_unit="ppm", mass_precision=4, 
                    isotope_filter_correlation=0.8, grouping={}):
    """
    Filter peaks from the consensus map based on RT and m/z tolerance.
    """
    if isinstance(consensus_map, oms.ConsensusMap):
        df = consensus_map.get_df()
    else:
        df = consensus_map.copy()

    cols=df.columns
    for c in cols:
        if "." in c:
            df.rename(columns={c: c.split(".")[0]}, inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.sort_values(by=["RT", "mz"], inplace=True)
    df.replace(0, np.nan, inplace=True)  # Replace 0 with NaN for filtering

    filtered_index = ()
    # start with the second row, which the table is sorte by rt and mz already
    # i.e., the row will always have larger m/z and maybe rt than the previous
    # if mz, rt , and correlation all meet, the filter the row
    # the only issue is missing values, where 

    for i in range(1,len(df)-1): 
        peak = df.iloc[i]
        rt = peak["RT"]
        mz = peak["mz"]
        prev_peak = df.iloc[i - 1]
        ## dealing with missing values

        if len(grouping):
            valid_data_count_per_group = {}
            for group, samples in grouping.items():
                valid_data_count_per_group[group] = (peak[samples].notna().sum())/len(samples)
                
            if all(count <0.7 for count in valid_data_count_per_group.values()):
                filtered_index += (i,)
                continue
        else:
            # if no grouping is provided, check the number of valid data points
            # we want at least 50% of the data points to be valid
            valid_data_count = df.iloc[i].notna().sum()
            if valid_data_count < len(df.columns)-6 * 0.5:
                filtered_index += (i,)
                continue

        # now we can check the RT and m/z differences is within a reasonable range
        rt_diff = abs(rt - prev_peak["RT"])
        mz_diff = round(abs(mz - 1.00335 - prev_peak["mz"]), mass_precision)
        if mass_tolerance_unit == "ppm":
            mass_diff = mz_diff / mz * 1e6 
        else:   #assuming it's a raw value if not ppm, ie, Da
            mz_diff = mass_diff
            
        if rt_diff <= rt_tolerance and mass_diff <= mass_tolerance:
            valid_data = (~peak.isna()) & (~prev_peak.isna())
            columns_with_non_zero_non_nan = df.columns[valid_data]
            valid_index = [df.columns.get_loc(col) for col in columns_with_non_zero_non_nan]
            valid_sample_ind = [c for c in valid_index if c >= 6]

            if len(valid_sample_ind) < 3:
                continue
            # Check if the previous peak is an isotope of the current peak 
            # in theory, if the current peak is an isotope of the previous peak,
            # the peak intensity should be correlated with the previous peak
            # meanwhile we dont really check the ratio since this can vary a lot due to elemental composition 
            else:
                # Calculate the Pearson correlation coefficient
                corr, _ = pearsonr(prev_peak.iloc[valid_sample_ind].to_list(), peak.iloc[valid_sample_ind].to_list())
                if corr > isotope_filter_correlation and np.mean(prev_peak.iloc[valid_sample_ind].to_numpy()) > np.mean(peak.iloc[valid_sample_ind]):
                    # If the correlation is high, mark the peak for filtering
                    filtered_index += (i,)
    
    # Remove the filtered peaks from the DataFrame
    filtered_df = df.drop(list(filtered_index), axis=0)
    filtered_df.fillna(0, inplace=True)
    filtered_df.to_csv(os.path.join(output_path, "2_data_after_isotope_filter.csv"), index=False)
    return filtered_df
            


def do_blank_filter(df, grouping_dict, output_path, blank_group_name="blank",
                    filter_cv_threhold = 0.3, max_missing=0.75, significance_level=0.05,
                    drop_suspect_blank =True, do_blank_subtraction=True, remove_blank=True):
    """
    Filter out peaks that are present in the blank sample.

    this does a few things:

    1. we shuold have at least 3 blank samples to do the filtering
    2. calculate the coefficient of variation (CV) and missing ratio for each peak across the blank samples
    3. filter out peaks with CV > filter_cv_threhold or missing ratio > max_missing
    4. if do_blank_subtraction is True, subtract the average blank value from each sample and ensure no negative values after subtraction.
       note that we dont check how low the blank value is as long as it pass the previous filtering step

    Parameters:
    df (pd.DataFrame): The DataFrame containing the peak data, i.e., feature map from previous steps
    grouping_dict (dict): {sample:group} dictionary where each sample is mapped to its group.
    output_path (str): The path where the filtered DataFrame will be saved.
    blank_group_name (str): The name of the group representing the blank samples.
    filter_cv_threhold (float): The threshold for the coefficient of variation (CV) to filter peaks.
    max_missing (float): The maximum allowed missing ratio for peaks to be retained.
    do_blank_subtraction (bool): Whether to perform blank subtraction on the data.


    """
    if grouping_dict is None:
        raise ValueError("Grouping dictionary is required for blank filtering.")
    
    # Identify the blank sample
    blank_samples = [samples for samples,group  in grouping_dict.items() if group == blank_group_name]
    actual_samples = [samples for samples,group  in grouping_dict.items() if group != blank_group_name]
    
    if not blank_samples:
        raise ValueError("No blank samples found in the grouping dictionary.")
    
    # Filter out peaks that are present in the blank sample
    filtered_df = df.copy()
    
    if len(blank_samples) <3:
        Warning("Not enough blank samples to filter out peaks. Returning original DataFrame.")
        return filtered_df
    else:
        blank_data = filtered_df.loc[:, blank_samples]
        blank_data["CV"] = blank_data.std() / blank_data.mean()
        blank_data["missing ratio"] = (blank_data[blank_samples].isna().sum(axis=1))/ len(blank_samples)
        blank_data=blank_data[(blank_data["CV"] < filter_cv_threhold) & (blank_data["missing ratio"] < max_missing)]

        if drop_suspect_blank:
            # Drop the blank samples that are not in the filtered data
            for i in blank_data.index:
                sample_values = filtered_df.iloc[i, actual_samples].dropna()
                blank_values = blank_data.loc[i, blank_samples].dropna()
                if len(sample_values) > 1 and len(blank_values) > 1:
                    stat, pval = ttest_ind(blank_values, sample_values, equal_var=False)
                    # If the p-value more than 0.05, this means the sample is not significantly different from the blank
                    # thus we assume that this peak is not a real peak and can be dropped
                    # its not entirelly sure if this might be too conservative, but this is a good star
                    if pval > significance_level:  
                        filtered_df.drop(i, inplace=True)
                        
    if do_blank_subtraction:
        filtered_df["blank_average"] = blank_data.mean(axis=1)
        for sample in grouping_dict.keys():
            if sample in filtered_df.columns and sample not in blank_samples:
                filtered_df[sample] = filtered_df[sample] - filtered_df["blank_average"]
                filtered_df[sample] = filtered_df[sample].clip(lower=0)  # Ensure no negative values after subtraction
                # filtered_df[sample] = filtered_df[sample].fillna(0)  # Fill NaN values with 0 after subtraction

    if remove_blank:
        filtered_df.drop(columns=blank_samples, inplace=True)

    filtered_df.fillna(0, inplace=True)
    filtered_df.to_csv(os.path.join(output_path, "3_data_after_blank_filter.csv"), index=False)

    return filtered_df


if __name__ == "__main__":
    data_path = r"G:\My Drive\Master\Data\20260526 high temp\1_Converted\mzML\pos"
    output_path =r"G:\My Drive\Master\Data\20260526 high temp\3_my code\pos\custom_alignment"

    # group_file = r"D:\20250618 real fire test\mycode\pos\fire_sample_only.json"
    group_file=None
    process_ms_data(data_path, 
                    output_path, 
                    group_file,
                    alignment_method="custom",
                    do_PCA=True, 
                    do_UMAP=False,
                    visualization=True)