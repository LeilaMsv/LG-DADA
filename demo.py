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




# 8 Oct
def _force_simlr_k(simlr_obj, n, k_min=2):
    """Force k-like attributes on a SIMLR object to a safe value for this X with n rows."""
    k_eff = int(max(k_min, min(n - 1, n // 2)))  # practical rule
    for attr in ("k", "K", "n_neighbors", "num_neighbors", "nn", "knn", "knn_k", "kNN"):
        if hasattr(simlr_obj, attr):
            try:
                setattr(simlr_obj, attr, k_eff)
            except Exception:
                pass
    return k_eff


def _try_fit_or_pca(simlr_obj, X, n_clusters, pca_dim=50, seed=0):

    """Try SIMLR.fit(X); if it still dies (hard-coded k=100), fall back to PCA+KMeans.
    Returns (S, F, y_pred). S or F may be None in the PCA fallback."""
    try:
        S, F, val, ind = simlr_obj.fit(X)
        # SIMLR success
        y_pred = simlr_obj.fast_minibatch_kmeans(F, n_clusters)
        return S, F, y_pred
    except Exception as e:
        print("[SIMLR fallback] using PCA+KMeans due to:", repr(e))
        d = max(2, min(pca_dim, X.shape[1], X.shape[0]-1))
        Xred = PCA(n_components=d, svd_solver='randomized', random_state=seed).fit_transform(X)
        y_pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Xred)
        # No SIMLR S/F available in this path
        return None, Xred, y_pred

def _safe_affinity(X):
    """Cosine similarity fallback to mimic SIMLR’s S matrix."""
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)   # don’t self-count
    return S.astype(np.float32, copy=False)


def _simlr_fit(simlr_obj, X, k_min=2):
    """Force k based on X, then call .fit(X)."""
    _force_simlr_k(simlr_obj, X.shape[0], k_min=k_min)
    return simlr_obj.fit(X)




start = time.time()

def main(sourceGraph,targetGraph,settings, num_subjects, num_features):

    # 2 Oct
    sourceGraph = sourceGraph.astype(np.float32, copy=False)
    targetGraph = targetGraph.astype(np.float32, copy=False)


    # newly added: Print the shapes of input graphs
    print("sourceGraph shape:", sourceGraph.shape)
    print("targetGraph shape:", targetGraph.shape)




    # # initialisation
    # subject = num_subjects
    # overallResult_PCC = np.zeros((subject,32))

    # #Newly added
    # print("overallResult_PCC shape:", overallResult_PCC.shape)

    # overallResult_TSW = np.zeros((subject,32))
    # predictedTargetGraph = np.empty((0, num_features), int)



    # initialisation  24 sep
    subject = num_subjects

    # --- K sweep setup (metrics independent of K/folds) ---
    K_MAX = min(10, subject - 1) if subject > 1 else 1
    K_GRID = list(range(1, K_MAX + 1))
    pcc_by_k = {k: np.full(subject, np.nan, dtype=np.float32) for k in K_GRID}
    mae_by_k = {k: np.full(subject, np.nan, dtype=np.float32) for k in K_GRID}
    print("K sweep grid:", K_GRID)
    # predictions container (keep as float32)
    predictedTargetGraph = np.empty((0, num_features), dtype=np.float32)
   



    

    ## STEP 1: feature extraction and clustering
    print("======SIMLR for Clustering======")


    # c = 2
    # simlr = SIMLR.SIMLR_LARGE(c, 50, 0)
    # S, F, val, ind = simlr.fit(sourceGraph)

    # # 6 OCT
    # c = 2
    # N_total = sourceGraph.shape[0]
    # K_min = 2
    # K_simlr0 = max(K_min, (N_total - 1) // 2)
    # K_simlr0 = min(K_simlr0, N_total - 1)
    # simlr = SIMLR.SIMLR_LARGE(c, 50, K_simlr0)
    # S, F, val, ind = simlr.fit(sourceGraph)




    # # 6 OCT

    # c = 2
    # N_total = sourceGraph.shape[0]
    # K_min = 2
    # K_simlr0 = max(K_min, (N_total - 1) // 2)
    # K_simlr0 = min(K_simlr0, N_total - 1)




    # simlr = SIMLR.SIMLR_LARGE(c, 50, K_simlr0)


    # # --- force k on the instance (some SIMLR versions ignore ctor arg) ---
    # k_eff = int(K_simlr0)
    # for attr in ("k", "K", "n_neighbors", "nn", "kNN", "knn", "knn_k"):
    #     if hasattr(simlr, attr):
    #         setattr(simlr, attr, k_eff)
    # # --------------------------------------------------------------------
    
    
    # S, F, val, ind = simlr.fit(sourceGraph)


    # y_pred = simlr.fast_minibatch_kmeans(F,c)
    
    # y_pred = y_pred.tolist()









    # 8 Oct
    c = 2
    simlr = SIMLR.SIMLR_LARGE(c, 50, 0)  # ctor arg isn’t trusted by this SIMLR version
    # try SIMLR; if it fails, use PCA+KMeans just to get y_pred
    S, F, y_pred = _try_fit_or_pca(simlr, sourceGraph, n_clusters=c, pca_dim=50, seed=0)
    y_pred = y_pred.tolist()












    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    # Split the data into training and testing sets
    loo = LeaveOneOut()
    loo.get_n_splits(sourceGraph)

    for train_index, test_index in loo.split(sourceGraph):

        
        rearrangedPredictorView = np.concatenate((np.transpose(sourceGraph[train_index]), np.transpose(sourceGraph[test_index])),axis = 1)
        rearrangedTargetView = np.concatenate((np.transpose(targetGraph[train_index]),np.transpose(targetGraph[test_index])),axis = 1)
        




        # label_of_test_index = y_pred[test_index[0]]
        # train_index = get_indexes(label_of_test_index,y_pred)
        
        # print("Testing subject number: ", test_index[0]," of cluster ", label_of_test_index)
        # print("Training subjects: ", len(train_index)," of cluster ", label_of_test_index)



        #2 Oct

        label_of_test_index = y_pred[test_index[0]]
        train_index = get_indexes(label_of_test_index, y_pred)

        # ensure we have a valid training set for this fold 
        # If the cluster is empty, fall back to "all except the test subject"
        if len(train_index) == 0:
            train_index = [i for i in range(len(y_pred)) if i != test_index[0]]
        # If the test subject accidentally appears in the cluster list, remove it
        elif test_index[0] in train_index:
            train_index.remove(test_index[0])

        n_train = len(train_index)

        #Dynamic K per fold: at least 1, at most 5, and cannot exceed n_train-1 
        K = 1 if n_train <= 1 else max(1, min(5, n_train - 1))

        # print("Testing subject number:", test_index[0], "of cluster", label_of_test_index)
        # print("Training subjects:", n_train, "of cluster", label_of_test_index, "| K used:", K)



        # 2 oct
        print("Testing subject number:", test_index[0], "of cluster", label_of_test_index)
        print(
              "Training subjects:", n_train,
              "of cluster", label_of_test_index,
              "| K sweep:", list(range(1, max(1, min(K_MAX, n_train - 1)) + 1))
            )


        







    ## STEP 2: Domain Alignment for training samples


        #simlr1 = SIMLR.SIMLR_LARGE(1, 10, 0)


        # # 6 OCT
        # # neighbors for fold's SIMLR
        # K_min = 2
        # K_simlr_fold = max(K_min, n_train // 2)
        # K_simlr_fold = min(K_simlr_fold, n_train)   # cap to available train rows
        # simlr1 = SIMLR.SIMLR_LARGE(1, 10, K_simlr_fold)


        # 6 OCT

        K_min = 2
        K_simlr_fold = max(K_min, n_train // 2)
        K_simlr_fold = min(K_simlr_fold, max(1, n_train - 1))  # ≤ n_train-1

        simlr1 = SIMLR.SIMLR_LARGE(1, 10, K_simlr_fold)
        # --- force k on the instance ---
        k_eff_fold = int(K_simlr_fold)
        for attr in ("k", "K", "n_neighbors", "nn", "kNN", "knn", "knn_k"):
            if hasattr(simlr1, attr):
                setattr(simlr1, attr, k_eff_fold)
        # --------------------------------



        



        enc = Encoder(settings)
    
        train__TV_A = targetGraph[train_index]
        SV = sourceGraph[train_index]





        # print("Encode the target graph...")


        # #Simlarity2, _,_, _ = simlr1.fit(train__TV_A)


        # # 8 Oct
        # try:
        #     _ = simlr1.fit(train__TV_A)
        # except Exception as e:
        #     print("[SIMLR warn] fit(train__TV_A) failed; will use fallback sims later:", repr(e))




        # #encode_train__TV_A = enc.erun(Simlarity2, sourceGraph[train_index],"No_hidden_SIMLR",1,1,1)
        # #encode_train__TV_A = enc.erun(Simlarity2, sourceGraph[train_index], "No_hidden_SIMLR", 1, 1, 1, input_dim)


        # encode_train__TV_A = enc.erun(Simlarity2, sourceGraph[train_index], "No_hidden_SIMLR", targetGraph[train_index], None, None, input_dim)




        # 8 OCt
        print("Encode the target graph...")
        try:
            Simlarity2, _, _, _ = simlr1.fit(train__TV_A)
        except Exception as e:
            print("[SIMLR fallback] Simlarity2 via cosine_similarity due to:", repr(e))
            S2 = _safe_affinity(train__TV_A)     # dense cosine affinity with zeroed diagonal
            Simlarity2 = sp.csr_matrix(S2)       # make it sparse for encoder.format_data_new

        encode_train__TV_A = enc.erun(
           Simlarity2,
            sourceGraph[train_index],
            "No_hidden_SIMLR",
            targetGraph[train_index],
            None,
            None,
            input_dim
        )








        
    ## STEP 3: Dual Adversarial Regularization of source graph embedding for training and testing samples
        test__train__SV = np.vstack((sourceGraph[train_index],sourceGraph[test_index]))





        # print("Encode the source view of the TRAIN subjects and the TEST subject...")


        # #Simlarity1, _,_, _ = simlr1.fit(test__train__SV)


        # # 8 Oct
        # try:
        #     _ = simlr1.fit(test__train__SV)
        # except Exception as e:
        #     print("[SIMLR warn] fit(test__train__SV) failed; will use fallback sims later:", repr(e))



        # #encode_test__train__SV = enc.erun(Simlarity1, test__train__SV,"Yes_hidden_SIMLR", train__TV_A, encode_train__TV_A, rearrangedTargetView)
        # encode_test__train__SV = enc.erun(Simlarity1, test__train__SV,"Yes_hidden_SIMLR", train__TV_A, encode_train__TV_A, rearrangedTargetView, input_dim)



        # 8 OCT
        print("Encode the source view of the TRAIN subjects and the TEST subject...")
        try:
            Simlarity1, _, _, _ = simlr1.fit(test__train__SV)
        except Exception as e:
            print("[SIMLR fallback] Simlarity1 via cosine_similarity due to:", repr(e))
            S1 = _safe_affinity(test__train__SV)  # dense cosine affinity with zeroed diagonal
            Simlarity1 = sp.csr_matrix(S1)        # make it sparse

        encode_test__train__SV = enc.erun(
            Simlarity1,
            test__train__SV,
            "Yes_hidden_SIMLR",
            train__TV_A,
            encode_train__TV_A,
            rearrangedTargetView,
            input_dim
        )  








      
    # ## Connectomic Manifold Learning using SIMLR
    #     SALL, FALL,val, ind = simlr1.fit(encode_test__train__SV)
    #     SY, FY,val, ind = simlr1.fit(encode_train__TV_A)



    #     #2 oct
    #     # # number of neighbors for trust score
    #     # TS_bestNb = 5




    #     # # get best TS_benstNb neighbors for everyone
    #     # sall = SALL.todense()
    #     # Index_ALL = np.argsort(-sall, axis=0)
        
    #     #2 oct
    #     sall = np.asarray(SALL.todense())
    #     Index_ALL = np.argsort(-sall, axis=0)

    #     des = np.sort(-sall, axis=0)
    #     Bvalue_ALL = -des

    #     # sy = SY.todense()
    #     # Index_Y = np.argsort(-sy, axis=0)


    #     sy = np.asarray(SY.todense())
    #     Index_Y = np.argsort(-sy, axis=0)



    #     desy = np.sort(-sy,axis=0)
    #     Bvalue_Y = -desy



    # 8 OCt
    

    # SALL (neighbors on [train; test] encoded source)
        try:
            SALL, _, _, _ = simlr1.fit(encode_test__train__SV)
            sall = np.asarray(SALL.todense())
        except Exception as e:
            print("[SIMLR fallback] SALL via cosine_similarity due to:", repr(e))
            sall = cosine_similarity(encode_test__train__SV)
            np.fill_diagonal(sall, 0.0)

        # SY (neighbors on encoded target of train)
        try:
            SY, _, _, _ = simlr1.fit(encode_train__TV_A)
            sy = np.asarray(SY.todense())
        except Exception as e:
            print("[SIMLR fallback] SY via cosine_similarity due to:", repr(e))
            sy = cosine_similarity(encode_train__TV_A)
            np.fill_diagonal(sy, 0.0)

        # Build neighbor ranks ROW-WISE (axis=1) and positive weights
        Index_ALL = np.argsort(-sall, axis=1)
        Bvalue_ALL = -np.sort(-sall, axis=1)

        Index_Y   = np.argsort(-sy,   axis=1)
        Bvalue_Y  = -np.sort(-sy,  axis=1)









        # ## STEP 4: Target Graph Prediction
        # # make prediction for each testing subject
        # for testingSubject in range(1,2):
        #     #print "testing subject:", test_index[0]
        #     print("testing subject:", test_index[0])

        #     # get this testing subject's rearranged index and original index
        #     tSubjectIndex = (SV.shape[0]-2) + testingSubject
        #     tSubjectOriginalIndex = test_index
        #     # compute Tscore for each neighbor
        #     trustScore = np.ones((TS_bestNb,TS_bestNb))
        #     newWeight_TSW = np.ones(TS_bestNb)

        #     for neighbor in range(0,TS_bestNb):
        #         neighborIndex = Index_ALL[tSubjectIndex,neighbor]
        #         temp_counter = 0
        #         while (neighborIndex  > SV.shape[0]):
        #         # best neighbor is a testing data 
        #             temp_counter = temp_counter + 1
        #             neighborIndex = Index_ALL[tSubjectIndex,(TS_bestNb + temp_counter)]

        #         if (temp_counter != 0):
        #             neighborSequence = TS_bestNb + temp_counter
        #         else:
        #             neighborSequence = neighbor

        #             print('----',neighborIndex)
        #             if (neighborIndex == Index_Y.shape[0]):
        #                 continue
        #             # get top nb neighbors in mappedX
        #             neighborListX = Index_ALL[neighborIndex,0:TS_bestNb]
        #             # get top nb neighbors in mappedY
        #             neighborListY = Index_Y[neighborIndex,0:TS_bestNb]
        #             # calculate trust score
        #             trustScore[TS_bestNb-1,neighbor] = len(np.intersect1d(np.array(neighborListX),np.array(neighborListY)))
        #             # calculate new weight (TS * Similarity)
        #             newWeight_TSW[neighbor] = exp(trustScore[TS_bestNb-1,neighbor] / TS_bestNb * Bvalue_ALL[tSubjectIndex,neighborSequence])
                    
        #     #reconstruct with Tscore and similarity weight
        #     innerPredict_TSW = np.zeros(SV.shape[1])[np.newaxis]
        #     #summing up the best neighbors
        #     for j1 in range(0,TS_bestNb):
        #         tr = (rearrangedTargetView[:,Index_ALL[tSubjectIndex,j1]])[np.newaxis]
        #         if j1 == 0:
        #             innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j1]
        #         else:
        #             innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j1]

        #     # scale weight to 1
        #     Scale_TSW = sum(newWeight_TSW)
        #     innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)
            
        #     # calculate PCC and MAE
        #     tr2 = (rearrangedTargetView[:,tSubjectIndex])[np.newaxis]
        #     resulttsw =abs(tr2.T - innerPredict_TSW)
        #     iMAE_TSW = mean_absolute_error(tr2.T, innerPredict_TSW)
        #     overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] = overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] + iMAE_TSW
        #     [r,p] = stats.pearsonr(tr2.T, innerPredict_TSW)
        #     overallResult_PCC[tSubjectOriginalIndex,TS_bestNb] = overallResult_PCC[tSubjectOriginalIndex,TS_bestNb] + r
            
        #     predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW.T, axis=0)

        #     #print test_index[0]
        #     print(test_index[0])



        # # 2 oct

        # ## STEP 4: Target Graph Prediction (K-sweep)
        # for testingSubject in range(1, 2):
        #     print("testing subject:", test_index[0])

        #     # In test__train__SV = [train; test], the test row is at n_train
        #     tSubjectIndex = n_train
        #     tSubjectOriginalIndex = test_index[0]

        #     # Feasible K for this fold
        #     K_feasible_max = max(1, min(K_MAX, n_train - 1))
        #     for k in range(1, K_feasible_max + 1):

        #         trust_overlap = np.zeros(k, dtype=np.float32)
        #         newWeight_TSW = np.zeros(k, dtype=np.float32)

        #         for j in range(k):



        #             # # while loop
        #             # neighborIndex = Index_ALL[tSubjectIndex, j]

        #             # # if neighbor points to the test row, skip forward
        #             # temp_counter = 0
        #             # while neighborIndex >= n_train:
        #             #     temp_counter += 1
        #             #     col = min(k + temp_counter, Index_ALL.shape[0] - 1)
        #             #     neighborIndex = Index_ALL[tSubjectIndex, col]

        #             # neighborSequence = min(j + temp_counter, Index_ALL.shape[0] - 1)
        #             # # while loop



        #             # 2 oct_ instead of while loop
        #             # get more than enough candidates, filter to training, then take first k
        #             row_sorted = Index_ALL[tSubjectIndex, :min(Index_ALL.shape[0], 2*k + 10)].ravel()
        #             train_cands = [int(ix) for ix in row_sorted if ix < n_train]
        #             train_cands = train_cands[:k]

        #             # now iterate only real training neighbors
        #             innerPredict_TSW = np.zeros((1, SV.shape[1]), dtype=np.float32)
        #             newWeight_TSW = np.zeros(k, dtype=np.float32)

        #             for j, neighborIndex in enumerate(train_cands):
        #                 neighborSequence = j  # similarity lookup column; j aligns with our j-th kept neighbor
        #                 neighborListX = Index_ALL[neighborIndex, 0:k]
        #                 neighborListY = Index_Y[neighborIndex, 0:k]
        #                 overlap = len(np.intersect1d(np.array(neighborListX).ravel(), np.array(neighborListY).ravel()))
        #                 newWeight_TSW[j] = exp((overlap / float(k)) * Bvalue_ALL[tSubjectIndex, neighborSequence])

        #                 idx = neighborIndex  # here idx is a train column in rearrangedTargetView
        #                 tr = rearrangedTargetView[:, idx].reshape(1, -1)
        #                 innerPredict_TSW += tr * newWeight_TSW[j]







        #             # bounds for Index_Y
        #             if neighborIndex >= Index_Y.shape[0]:
        #                 continue

        #             # top-k lists in source/target manifolds
        #             neighborListX = Index_ALL[neighborIndex, 0:k]
        #             neighborListY = Index_Y[neighborIndex, 0:k]

        #             # trust = size of overlap of K-NN in the two spaces
        #             overlap = len(np.intersect1d(np.array(neighborListX).ravel(),
        #                                          np.array(neighborListY).ravel()))
        #             trust_overlap[j] = overlap

        #             # weight = exp( (trust/k) * similarity )
        #             newWeight_TSW[j] = exp((overlap / float(k)) * Bvalue_ALL[tSubjectIndex, neighborSequence])

        #         # if all weights are zero (degenerate small-N), use uniform
        #         if np.all(newWeight_TSW == 0):
        #             newWeight_TSW[:] = 1.0




        #         # # reconstruct with trust/similarity weights
        #         # innerPredict_TSW = np.zeros(SV.shape[1], dtype=np.float32)[np.newaxis, :]
        #         # for j in range(k):
        #         #     idx = Index_ALL[tSubjectIndex, j]
        #         #     # avoid selecting the test column itself
        #         #     if idx >= rearrangedTargetView.shape[1] - 1:
        #         #         continue
        #         #     tr = (rearrangedTargetView[:, idx])[np.newaxis, :]
        #         #     if j == 0:
        #         #         innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j]
        #         #     else:
        #         #         innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j]



        #         # 2 oct
        #         innerPredict_TSW = np.zeros((1, SV.shape[1]), dtype=np.float32)
        #         for j in range(k):
        #             idx = int(Index_ALL[tSubjectIndex, j])
        #             if idx >= rearrangedTargetView.shape[1] - 1:
        #                 continue
        #             tr = rearrangedTargetView[:, idx].reshape(1, -1)  # (1, num_features)
        #             innerPredict_TSW += tr * newWeight_TSW[j]





        #             # explanation: Now innerPredict_TSW stays (1, num_features) the whole time, so later lines work as-is:

        #             # innerPredict_TSW = innerPredict_TSW / max(1.0, float(np.sum(newWeight_TSW)))
        #             # y = innerPredict_TSW.ravel()
        #             # predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW, axis=0)



        #         Scale_TSW = float(np.sum(newWeight_TSW))
        #         if Scale_TSW == 0:
        #             Scale_TSW = 1.0
        #         innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)

        #         # metrics (guard constant vectors for Pearson)
        #         tr2 = (rearrangedTargetView[:, tSubjectIndex])[np.newaxis, :]
        #         x = tr2.T.ravel()
        #         y = innerPredict_TSW.ravel()

        #         if np.std(x) == 0 or np.std(y) == 0:
        #             r = 0.0
        #         else:
        #             r, p = stats.pearsonr(x, y)
        #         iMAE_TSW = mean_absolute_error(x, y)

        #         # record into per-K containers (from Step 1)
        #         pcc_by_k[k][tSubjectOriginalIndex] = r
        #         mae_by_k[k][tSubjectOriginalIndex] = iMAE_TSW

        #     # keep a single predictions array (last k’s prediction)
        #     predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW, axis=0)
        #     print(test_index[0])



        



        # 2 oct

        ## STEP 4: Target Graph Prediction (K-sweep)
        for testingSubject in range(1, 2):
            print("testing subject:", test_index[0])

            # In test__train__SV = [train; test], the test row is at n_train
            tSubjectIndex = n_train
            tSubjectOriginalIndex = test_index[0]

            # Feasible K for this fold
            K_feasible_max = max(1, min(K_MAX, n_train - 1))
            for k in range(1, K_feasible_max + 1):
                # 1) Candidate neighbors for the test node (take >k, then filter to training rows)
                #row_sorted = np.asarray(Index_ALL[tSubjectIndex, :min(Index_ALL.shape[0], 2*k + 10)]).ravel()

                # 8 OCT
                row_sorted = np.asarray(Index_ALL[tSubjectIndex, :min(Index_ALL.shape[1], 2*k + 10)]).ravel()



                
                train_cands = [int(ix) for ix in row_sorted if ix < n_train][:k]

                # 2) Weights and reconstruction accumulator
                newWeight_TSW = np.zeros(len(train_cands), dtype=np.float32)
                innerPredict_TSW = np.zeros((1, SV.shape[1]), dtype=np.float32)

                # 3) Trust-weighted similarity and reconstruction
                for j, neighborIndex in enumerate(train_cands):
                    # top-k lists for this neighbor in both manifolds
                    neighborListX = np.asarray(Index_ALL[neighborIndex, 0:k]).ravel()
                    neighborListY = np.asarray(Index_Y[neighborIndex, 0:k]).ravel()
                    overlap = len(np.intersect1d(neighborListX, neighborListY))

                    # similarity term from Bvalue_ALL; j is our kept-neighbor rank
                    newWeight_TSW[j] = exp((overlap / float(k)) * Bvalue_ALL[tSubjectIndex, j])

                    # add this neighbor's target vector
                    tr = rearrangedTargetView[:, neighborIndex].reshape(1, -1)  # (1, num_features)
                    innerPredict_TSW += tr * newWeight_TSW[j]

                # 4) Normalize (guard zero)
                Scale_TSW = float(np.sum(newWeight_TSW))
                innerPredict_TSW = innerPredict_TSW / (Scale_TSW if Scale_TSW > 0 else 1.0)

                # 5) Metrics (flatten to 1-D; guard constant vectors)
                x = rearrangedTargetView[:, tSubjectIndex].ravel()
                y = innerPredict_TSW.ravel()
                r = 0.0 if (np.std(x) == 0 or np.std(y) == 0) else stats.pearsonr(x, y)[0]
                iMAE_TSW = mean_absolute_error(x, y)

                # 6) Record per-K results (from your Step 1 dicts)
                pcc_by_k[k][tSubjectOriginalIndex] = r
                mae_by_k[k][tSubjectOriginalIndex] = iMAE_TSW

            # (Optional) keep last-k prediction in a flat array
            predictedTargetGraph = np.append(predictedTargetGraph, innerPredict_TSW, axis=0)
            print(test_index[0])









            
    # pcc = np.mean(overallResult_PCC,axis=0)
    # print("Pearson Correlation Coefficient: ", pcc)
    # mae = np.mean(overallResult_TSW,axis=0)
    # print("Mean Absolute Error: ", mae)


    # --- 24 sep
    pcc_mean_by_k = {k: float(np.nanmean(pcc_by_k[k])) for k in K_GRID}
    mae_mean_by_k = {k: float(np.nanmean(mae_by_k[k])) for k in K_GRID}
    print("PCC by K (LOOCV mean):", pcc_mean_by_k)
    print("MAE by K (LOOCV mean):", mae_mean_by_k)




     
    
    return predictedTargetGraph


num_subjects = 50
num_features = 35 * 35  


## Simulate graph data for simply running the code
## in this exemple, the source and target matrices have different statistical distributions
mu, sigma = 0.2226636809, 0.02720207221 # mean and standard deviation
sourceGraph = np.random.normal(mu, sigma, (num_subjects, num_features))
print("SourceGraph shape:", sourceGraph.shape)


mu, sigma = 0.0830806568, 0.01338490182
targetGraph = np.random.normal(mu, sigma, (num_subjects, num_features))
print("TargetGraph shape:", targetGraph.shape)

#newly added
input_dim = sourceGraph.shape[1]
print("Input dimension:", input_dim)   

model = 'arga_ae' #autoencoder/variational autoencoder
settings = settings.get_settings_new(model)
predicted_target_graphs = main(sourceGraph, targetGraph, settings, num_subjects, num_features)
print("Predicted target graph shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:", predicted_target_graphs.shape)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  shape


# #To print from model.py !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# h1_shape, emb_shape, rec_shape = sess.run(
#     [model.hidden1_shape, model.embedding_shape, model.recon_shape],
#     feed_dict=feed_dict
# )
# print("ARGA Hidden1 shape:", h1_shape)
# print("ARGA Embeddings (Z) shape:", emb_shape)
# print("ARGA Reconstructions shape:", rec_shape)




end = time.time()
print(end - start)

# -*- coding: utf-8 -*-
