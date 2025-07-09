import os ,json
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def run_pca(peak_table, grouping_info_dict, components=2,
            save_path=None, visualization=False, include_blank=False,
            save_name = 'PCA_results.png'):
    if not include_blank:
        sample_cols = [c for c  in peak_table.columns if c in grouping_info_dict.keys() and grouping_info_dict[c]!="blank"]
    else:
        sample_cols = [c for c  in peak_table.columns if c in grouping_info_dict.keys()]
    peak_table.fillna(0, inplace=True)
    
    x = peak_table.loc[:,sample_cols].values
    x=pd.DataFrame(StandardScaler().fit_transform(x))

    pca = PCA(n_components=components)
    x_pca = pca.fit_transform(x)
    x_pca = pd.DataFrame(x_pca)

    # explained_variance_ratio_ is the percentage of variance explained by each of the selected components.
    explained_variance = pca.explained_variance_ratio_

    #components_ is n_component by n_feature matrix
    #Principal axes in feature space, representing the directions of maximum variance in the data.
    raw_loadings = pca.components_.T

    # the true loading represent the feature loadings that directly represent 
    # the correlation between the original features and the principal components
    # instead of just the raw coefficients in pca.components_.
    true_loadings = raw_loadings * np.sqrt(pca.explained_variance_)

    if save_path:
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

        PCA_results = {"sample_names": sample_cols,
                "PC_loadings":raw_loadings.tolist(),
                "True_loadings": true_loadings.tolist(),
                "explained_variance": explained_variance.tolist(),
                "per_var": per_var.tolist()}
        


        output_subfolder = os.path.join(save_path, 'PCA_results')
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        x_pca.to_csv(os.path.join(output_subfolder,'x_pca.csv'), index=False)

        with open(os.path.join(output_subfolder, 'PCA_results.json'), 'w') as f:
            json.dump(PCA_results, f)

    if visualization:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=true_loadings.T[0], y=true_loadings.T[1], hue=sample_cols, s=100)
        plt.title('PCA Results')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
        plt.grid()
        plt.savefig(os.path.join(output_subfolder, 'PCA_plot.png'))

    
def run_UMAP(peak_table, grouping_info_dict, mz_col_index=3, rt_col_index=4, n_neighbors=15, min_dist=0.1,n_components=2, save_path=None, visualization=False):
    sample_cols = [k for k in grouping_info_dict.keys() if k.lower()!='blank' and k in peak_table.columns]

    data = peak_table.loc[:,sample_cols]
    peak_label = peak_table.iloc[:, mz_col_index].astype(str)+"_"+ peak_table.iloc[:, rt_col_index].astype(str)
    y=peak_label.values 

    data.set_index(pd.Index(y, name='Peak Label'), inplace=True)
    x=data.T.values
    x=StandardScaler().fit_transform(x)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, 
                        metric='euclidean', random_state=42)
    umap_embedding = reducer.fit_transform(x)

    umap_df = pd.DataFrame(umap_embedding, columns=['UMAP1', 'UMAP2'])
    umap_df.index = data.index

    # umap_df.reset_index(inplace=True)
    # umap_df.rename(columns={'index': 'Peak Label'}, inplace=True)
    # umap_results = {
    #     "sample_names": sample_cols,
    #     "UMAP_embedding": umap_df
    # }
    if save_path:
        output_subfolder = os.path.join(save_path, 'UMAP_results')
        os.makedirs(output_subfolder, exist_ok=True)
        umap_df.to_csv(os.path.join(output_subfolder, 'UMAP_results.csv'), index=False)
        # with open(os.path.join(output_subfolder, 'UMAP_results.json'), 'w') as f:
        #     json.dump(umap_results, f, indent=4)
    if visualization:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=umap_df['UMAP1'], y=umap_df['UMAP2'], hue=sample_cols, palette='viridis', s=100)
        plt.title('UMAP Results')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(os.path.join(output_subfolder, 'UMAP_plot.png'))



