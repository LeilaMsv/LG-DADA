#!/usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
from scipy import stats
from math import exp
import numpy as np
import SIMLR
import time
import settings
from encoder import Encoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.sparse as sp

import os
import re
import pandas as pd


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# ============   UPDATED CSV LOADING FUNCTION   ==============
# ============================================================

def extract_number(filename):
    """Extract numeric ID from filenames like A_77.csv or B_12.csv"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def load_graphs_from_folder(folder_path):
    """
    Modified loader:
    - Loads CSV adjacency matrices in 2D form.
    - Does NOT flatten them.
    - Returns: list of 2D matrices, list of filenames, number of regions.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(files) == 0:
        raise ValueError("No CSV files found in: {}".format(folder_path))

    files_sorted = sorted(files, key=lambda x: extract_number(x))

    matrices = []
    n_regions = None

    for fname in files_sorted:
        full_path = os.path.join(folder_path, fname)
        mat = pd.read_csv(full_path, header=None).values

        if n_regions is None:
            n_regions = mat.shape[0]

        matrices.append(mat.astype(np.float32))

    return matrices, files_sorted, n_regions


# ============================================================
# ==============  ORIGINAL SIMLR HELPER FUNCTIONS ============
# ============================================================

def _force_simlr_k(simlr_obj, n, k_min=2):
    k_eff = int(max(k_min, min(n - 1, n // 2)))
    for attr in ("k", "K", "n_neighbors", "num_neighbors", "nn", "knn", "knn_k", "kNN"):
        if hasattr(simlr_obj, attr):
            try:
                setattr(simlr_obj, attr, k_eff)
            except Exception:
                pass
    return k_eff


def _try_fit_or_pca(simlr_obj, X, n_clusters, pca_dim=50, seed=0):
    try:
        S, F, val, ind = simlr_obj.fit(X)
        y_pred = simlr_obj.fast_minibatch_kmeans(F, n_clusters)
        return S, F, y_pred
    except Exception as e:
        print("[SIMLR fallback] using PCA+KMeans due to: {}".format(repr(e)))
        d = max(2, min(pca_dim, X.shape[1], X.shape[0]-1))
        Xred = PCA(n_components=d, svd_solver='randomized', random_state=seed).fit_transform(X)
        y_pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Xred)
        return None, Xred, y_pred


def _safe_affinity(X):
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32, copy=False)



def _simlr_fit(simlr_obj, X, k_min=2):
    _force_simlr_k(simlr_obj, X.shape[0], k_min=k_min)
    return simlr_obj.fit(X)

# 17 March
def _pca_then_simlr_affinity(X, simlr_obj, pca_dim=10, seed=0):
    """
    Apply PCA before SIMLR and return the SIMLR affinity matrix.
    X: shape (n_samples, n_features)
    Returns:
        S_sparse : scipy sparse matrix from SIMLR
        X_pca    : PCA-reduced data used as SIMLR input
    """
    d = max(2, min(pca_dim, X.shape[0] - 1, X.shape[1]))
    X_pca = PCA(n_components=d, svd_solver='randomized', random_state=seed).fit_transform(X)
    _force_simlr_k(simlr_obj, X_pca.shape[0], k_min=2)
    S_sparse, _, _, _ = simlr_obj.fit(X_pca)
    return S_sparse, X_pca

# ============================================================
# ====================== MAIN MODEL ==========================
# ============================================================

start = time.time()

plot_results = True


#plots are here

def main(sourceGraph, targetGraph, settings, num_subjects, num_features, input_dim):
    sourceGraph = sourceGraph.astype(np.float32, copy=False)
    targetGraph = targetGraph.astype(np.float32, copy=False)

    print("sourceGraph shape: {}".format(sourceGraph.shape))
    print("targetGraph shape: {}".format(targetGraph.shape))

    subject = num_subjects

    # # #swap K for graph prediction
    # K_MAX = min(10, subject - 1) if subject > 1 else 1
    # K_GRID = list(range(1, K_MAX + 1))
    # pcc_by_k = {k: np.full(subject, np.nan, dtype=np.float32) for k in K_GRID}
    # mae_by_k = {k: np.full(subject, np.nan, dtype=np.float32) for k in K_GRID}

    predictedTargetGraph = np.empty((0, num_features), dtype=np.float32)

    #17MArch
    saved_partB_heatmap = False

    print("======SIMLR for Clustering======")
    c = 2
    simlr = SIMLR.SIMLR_LARGE(c, 50, 0)
    S, F, y_pred = _try_fit_or_pca(simlr, sourceGraph, n_clusters=c, pca_dim=50, seed=0)
    y_pred = y_pred.tolist()

    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    loo = LeaveOneOut()
    loo.get_n_splits(sourceGraph)

    for train_index, test_index in loo.split(sourceGraph):

        rearrangedPredictorView = np.concatenate(
            (np.transpose(sourceGraph[train_index]), np.transpose(sourceGraph[test_index])),
            axis=1
        )
        rearrangedTargetView = np.concatenate(
            (np.transpose(targetGraph[train_index]), np.transpose(targetGraph[test_index])),
            axis=1
        )

        label_of_test_index = y_pred[test_index[0]]
        train_index = get_indexes(label_of_test_index, y_pred)

        if len(train_index) == 0:
            train_index = [i for i in range(len(y_pred)) if i != test_index[0]]
        elif test_index[0] in train_index:
            train_index.remove(test_index[0])

        n_train = len(train_index)
        print("Testing subject number: {} of cluster {}".format(test_index[0], label_of_test_index))

        K_min = 2
        K_simlr_fold = max(K_min, n_train // 2)
        K_simlr_fold = min(K_simlr_fold, max(1, n_train - 1))

        simlr1 = SIMLR.SIMLR_LARGE(1, 10, K_simlr_fold)
        for attr in ("k", "K", "n_neighbors", "nn", "kNN", "knn", "knn_k"):
            if hasattr(simlr1, attr):
                try:
                    setattr(simlr1, attr, int(K_simlr_fold))
                except:
                    pass

        enc = Encoder(settings)
        train__TV_A = targetGraph[train_index]
        SV = sourceGraph[train_index]


    # #17 March################################################################################

    #     print("Encode the target graph...")
    #     try:
    #         Simlarity2, _, _, _ = simlr1.fit(train__TV_A)
    #     except Exception as e:
    #         print("[SIMLR fallback] Simlarity2 via cosine_similarity due to: {}".format(repr(e)))
    #         S2 = _safe_affinity(train__TV_A)
    #         Simlarity2 = sp.csr_matrix(S2)

    # # Right now it sends train__TV_A directly into SIMLR.
    # # I want:

    # # reduce train__TV_A with PCA

    # # run SIMLR on the PCA output

    # # optionally save and inspect the resulting Simlarity2
    # ############################################################


    #17 March
        print("Encode the target graph...")
        try:
            Simlarity2, train__TV_A_pca = _pca_then_simlr_affinity(
        train__TV_A,
        simlr1,
        pca_dim=10,
        seed=0
    )
            print("Part B PCA shape: {} -> {}".format(train__TV_A.shape, train__TV_A_pca.shape))
        except Exception as e:
            print("[SIMLR fallback] Simlarity2 via cosine_similarity due to: {}".format(repr(e)))
            S2 = _safe_affinity(train__TV_A)
            Simlarity2 = sp.csr_matrix(S2)

        S2_dense = np.asarray(Simlarity2.todense())
        print("Part B Simlarity2 stats | shape={} min={:.6f} max={:.6f} mean={:.6f}".format(
            S2_dense.shape, S2_dense.min(), S2_dense.max(), S2_dense.mean()
    ))
        


        if not saved_partB_heatmap:
           plt.figure(figsize=(6, 5))
           plt.imshow(S2_dense, aspect='auto')
           plt.colorbar()
           plt.title("Part B Simlarity2 (first saved fold)")
           plt.tight_layout()
           plt.savefig("partB_Simlarity2_first_fold.png", dpi=150)
           plt.close()
           saved_partB_heatmap = True

##################################################################################


        encode_train__TV_A = enc.erun(
            Simlarity2,
            sourceGraph[train_index],
            "No_hidden_SIMLR",
            targetGraph[train_index],
            None,
            None,
            input_dim
        )

        test__train__SV = np.vstack((sourceGraph[train_index], sourceGraph[test_index]))

        print("Encode the source view of the TRAIN subjects and the TEST subject...")
        try:
            Simlarity1, _, _, _ = simlr1.fit(test__train__SV)
        except Exception as e:
            print("[SIMLR fallback] Simlarity1 via cosine_similarity due to: {}".format(repr(e)))
            S1 = _safe_affinity(test__train__SV)
            Simlarity1 = sp.csr_matrix(S1)

        encode_test__train__SV = enc.erun(
            Simlarity1,
            test__train__SV,
            "Yes_hidden_SIMLR",
            train__TV_A,
            encode_train__TV_A,
            rearrangedTargetView,
            input_dim
        )

        try:
            SALL, _, _, _ = simlr1.fit(encode_test__train__SV)
            sall = np.asarray(SALL.todense())
        except:
            sall = cosine_similarity(encode_test__train__SV)
            np.fill_diagonal(sall, 0.0)

        try:
            SY, _, _, _ = simlr1.fit(encode_train__TV_A)
            sy = np.asarray(SY.todense())
        except:
            sy = cosine_similarity(encode_train__TV_A)
            np.fill_diagonal(sy, 0.0)

        Index_ALL = np.argsort(-sall, axis=1)
        Bvalue_ALL = -np.sort(-sall, axis=1)

        Index_Y = np.argsort(-sy, axis=1)
        Bvalue_Y = -np.sort(-sy, axis=1)

        print("testing subject: {}".format(test_index[0]))
        tSubjectIndex = n_train
        tSubjectOriginalIndex = test_index[0]

        #Fixed K for graph prediction
        k = max(1, min(5, n_train - 1))



        # #swap K for graph prediction

        # K_feasible_max = max(1, min(K_MAX, n_train - 1))


        #Beginnig of the for LOOp wap K for graph prediction
        #  for k in range(1, K_feasible_max + 1):

        row_sorted = np.asarray(
                Index_ALL[tSubjectIndex, :min(Index_ALL.shape[1], 2*k + 10)]
            ).ravel()

        train_cands = [int(ix) for ix in row_sorted if ix < n_train][:k]

        newWeight_TSW = np.zeros(len(train_cands), dtype=np.float32)
        innerPredict_TSW = np.zeros((1, SV.shape[1]), dtype=np.float32)

        for j, neighborIndex in enumerate(train_cands):
                neighborListX = np.asarray(Index_ALL[neighborIndex, 0:k]).ravel()
                neighborListY = np.asarray(Index_Y[neighborIndex, 0:k]).ravel()
                overlap = len(np.intersect1d(neighborListX, neighborListY))

                newWeight_TSW[j] = exp((overlap / float(k)) * Bvalue_ALL[tSubjectIndex, j])

                tr = rearrangedTargetView[:, neighborIndex].reshape(1, -1)
                innerPredict_TSW += tr * newWeight_TSW[j]

        Scale_TSW = float(np.sum(newWeight_TSW))
        innerPredict_TSW = innerPredict_TSW / (Scale_TSW if Scale_TSW > 0 else 1.0)

        x = rearrangedTargetView[:, tSubjectIndex].ravel()
        y = innerPredict_TSW.ravel()
        r = 0.0 if (np.std(x) == 0 or np.std(y) == 0) else stats.pearsonr(x, y)[0]
        iMAE_TSW = mean_absolute_error(x, y)


            # # #swap K for graph prediction 
            # pcc_by_k[k][tSubjectOriginalIndex] = r
            # mae_by_k[k][tSubjectOriginalIndex] = iMAE_TSW


        #END of for Loop for  swap K for graph prediction !!!!


        predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW, axis=0)
        print(test_index[0])

       
    # #swap K for graph prediction
    # pcc_mean_by_k = {k: float(np.nanmean(pcc_by_k[k])) for k in K_GRID}
    # mae_mean_by_k = {k: float(np.nanmean(mae_by_k[k])) for k in K_GRID}
    # print("PCC by K (LOOCV mean): {}".format(pcc_mean_by_k))
    # print("MAE by K (LOOCV mean): {}".format(mae_mean_by_k))


    #fixed K
    # Compute final overall PCC & MAE across all subjects
    true_all = targetGraph.flatten()
    pred_all = predictedTargetGraph.flatten()
    mae = mean_absolute_error(true_all, pred_all)
    pcc = 0.0 if (np.std(true_all) == 0 or np.std(pred_all) == 0) else stats.pearsonr(true_all, pred_all)[0]
    print("Final MAE:", mae)
    print("Final PCC:", pcc)



    ##plotting previous results based on k

    # if plot_results:
    #     import matplotlib
    #     matplotlib.use('TkAgg')
    #     import matplotlib.pyplot as plt

    #     mae_per_subject = mae_by_k[1]
    #     plt.figure()
    #     plt.plot(range(len(mae_per_subject)), mae_per_subject, marker='o')
    #     plt.title("Per-Subject Graph Reconstruction Error")
    #     plt.xlabel("Subject index")
    #     plt.ylabel("Mean Absolute Difference")
    #     plt.legend(["Mean Difference per Subject"])
    #     plt.grid(True)
    #     plt.show()

    #     pcc_means = [pcc_mean_by_k[k] for k in K_GRID]
    #     mae_means = [mae_mean_by_k[k] for k in K_GRID]

    #     plt.figure()
    #     plt.plot(K_GRID, pcc_means, marker='o')
    #     plt.plot(K_GRID, mae_means, marker='o')
    #     plt.title("Learning Trend by K")
    #     plt.xlabel("K (neighbor count)")
    #     plt.ylabel("Metric Value")
    #     plt.legend(["PCC (mean)", "MAE (mean)"])
    #     plt.grid(True)
    #     plt.show()

    return predictedTargetGraph


# ============================================================
# ========== NEW FINAL BLOCK: LOAD CSVs + THRESHOLD ==========
# ============================================================

source_path = "/home/le_mousavi/Documents/GNN/LG/LG-DADA-fork/synthetic data/experiment/b/A"
target_path = "/home/le_mousavi/Documents/GNN/LG/LG-DADA-fork/synthetic data/experiment/b/C"
predicted_output_path = "/home/le_mousavi/Documents/GNN/LG/LG-DADA-fork/synthetic data/experiment/b/predicted_C"

# Threshold parameter (keep strongest X% edges)
percent_threshold = 12.5 # <---- you can change this anytime


# ---- Load raw adjacency matrices in 2D form ----
source_mats, source_files, n_regions = load_graphs_from_folder(source_path)
target_mats, target_files, n_regions2 = load_graphs_from_folder(target_path)

assert n_regions == n_regions2, "Source and target adjacency matrix sizes do not match."


# ---- THRESHOLDING ---- sorting with no abs
# def threshold_matrix(mat, percent_threshold):
#     """
#     Applies percentile threshold:
#     - flatten matrix
#     - compute threshold (100 - percent_threshold)
#     - keep strongest edges
#     - zero rest
#     - return thresholded matrix (2D)
#     """
#     flattened = mat.flatten()
#     thresh_value = np.percentile(flattened, 100 - percent_threshold)
#     return np.where(mat >= thresh_value, mat, 0)


# #---- THRESHOLDING ---- TAKE ABS  assign sign

# def threshold_matrix(mat, percent_threshold):
#     """
#     Applies percentile threshold using absolute values:
#     - flatten abs(mat)
#     - compute threshold (100 - percent_threshold)
#     - keep edges where |weight| >= threshold
#     - keep original sign
#     - zero the rest
#     """
#     flattened_abs = np.abs(mat).flatten()
#     thresh_value = np.percentile(flattened_abs, 100 - percent_threshold)
#     return np.where(np.abs(mat) >= thresh_value, mat, 0)


#---- THRESHOLDING ---- TAKE ABS  NO assign sign
def threshold_matrix(mat, percent_threshold):
    """
    Threshold using absolute values:
    - Take absolute value
    - Compute threshold on abs(mat)
    - Keep |weights| >= threshold
    - Do NOT restore original signs
    - Zero out the rest
    """
    abs_mat = np.abs(mat)
    flattened_abs = abs_mat.flatten()
    thresh_value = np.percentile(flattened_abs, 100 - percent_threshold)
    return np.where(abs_mat >= thresh_value, abs_mat, 0)



sourceGraph_list = []
targetGraph_list = []

for mat in source_mats:
    thr = threshold_matrix(mat, percent_threshold)
    sourceGraph_list.append(thr.reshape(-1))

for mat in target_mats:
    thr = threshold_matrix(mat, percent_threshold)
    targetGraph_list.append(thr.reshape(-1))


# Convert to final arrays expected by the model
sourceGraph = np.array(sourceGraph_list, dtype=np.float32)
targetGraph = np.array(targetGraph_list, dtype=np.float32)


num_subjects = len(source_files)
num_features = n_regions * n_regions
input_dim = num_features

print("Detected subjects: {}".format(num_subjects))
print("Detected regions: {}".format(n_regions))
print("num_features: {}".format(num_features))

model = 'arga_ae'
settings = settings.get_settings_new(model)

predicted_graphs = main(sourceGraph, targetGraph, settings, num_subjects, num_features, input_dim)

print("Predicted target graph shape: {}".format(predicted_graphs.shape))

if not os.path.exists(predicted_output_path):
    os.makedirs(predicted_output_path)

for i, pred_vec in enumerate(predicted_graphs):
    mat = pred_vec.reshape(n_regions, n_regions)
    out_name = "pred_B_{}.csv".format(extract_number(source_files[i]))
    out_path = os.path.join(predicted_output_path, out_name)
    pd.DataFrame(mat).to_csv(out_path, header=False, index=False)

print("Saved predicted CSV files to: {}".format(predicted_output_path))

end = time.time()
print("Total time: {}".format(end - start))