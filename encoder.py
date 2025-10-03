# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneOut
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from math import exp
import numpy as np
import SIMLR
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import settings
from constructor import get_placeholder, get_model, get_model_2, format_data_new, get_optimizer, get_optimizer_2, update
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Encoder():
    def __init__(self, settings):
        self.iteration = settings['iterations']
        self.model = settings['model']



    # def new_dataset_predicted_TV(self, emb, ztrTV, rearrangedTargetView, original_train_TV):

    #     if ztrTV is None or rearrangedTargetView is None:
    #         return None

    #     simlr = SIMLR.SIMLR_LARGE(1, 10, 0)
    #     loo = LeaveOneOut()
    #     x = emb[:-1]
    #     for train, test in loo.split(x):
    #         test_index = test
    #         new_tr_tst_z_sv = np.vstack((x[train],x[test]))
    #         new_tr_z_tv = ztrTV[train]
    #         SALL, FALL,val, ind = simlr.fit(new_tr_tst_z_sv)
    #         SY, FY,val, ind = simlr.fit(new_tr_z_tv)
    #         # number of neighbors for trust score
    #         TS_bestNb = 5
    #         # get best TS_benstNb neighbors for everyone
    #         sall = SALL.todense()
    #         Index_ALL = np.argsort(-sall, axis=0)
    #         des = np.sort(-sall, axis=0)
    #         Bvalue_ALL = -des
            
    #         sy = SY.todense()
    #         Index_Y = np.argsort(-sy, axis=0)
    #         desy = np.sort(-sy,axis=0)
    #         Bvalue_Y = -desy
            
    #         # make prediction for each testing subject
    #         for testingSubject in range(1,2):
    #             # get this testing subject's rearranged index and original index
    #             tSubjectIndex = (new_tr_z_tv.shape[0]-2) + testingSubject
    #             tSubjectOriginalIndex = test_index
    #             # compute Tscore for each neighbor
    #             trustScore = np.ones((TS_bestNb,TS_bestNb))
    #             newWeight_TSW = np.ones(TS_bestNb)
                
    #             for neighbor in range(1,TS_bestNb+1):
    #                 neighborIndex = Index_ALL[tSubjectIndex,neighbor]
    #                 temp_counter = 0
    #                 while (neighborIndex  > new_tr_z_tv.shape[0]):
    #                 # best neighbor is a testing data 
    #                     temp_counter = temp_counter + 1
    #                     neighborIndex = Index_ALL[tSubjectIndex,(TS_bestNb + temp_counter)]

    #                 if (temp_counter != 0):
    #                     neighborSequence = TS_bestNb + temp_counter
    #                 # else:
    #                 #     neighborSequence = neighbor

    #                 #     print("---",neighborIndex)
    #                 #     # get top nb neighbors in mappedX
    #                 #     neighborListX = Index_ALL[neighborIndex,1:TS_bestNb+1]
    #                 #     # get top nb neighbors in mappedY
    #                 #     neighborListY = Index_Y[neighborIndex,1:TS_bestNb+1]
    #                 #     # calculate trust score
    #                 #     trustScore[TS_bestNb-1,neighbor-1] = len(np.intersect1d(np.array(neighborListX),np.array(neighborListY)))
    #                 #     # calculate new weight (TS * Similarity)
    #                 #     newWeight_TSW[neighbor-1] = exp(trustScore[TS_bestNb-1,neighbor-1] / TS_bestNb * Bvalue_ALL[tSubjectIndex,neighborSequence])

                    
    #                 # i modified this on Aug 12
    #                 else:
    #                     neighborSequence = neighbor

    #                     print("---", neighborIndex)

    #                     # Skip if neighbor index is outside training target-view range
    #                     if neighborIndex >= Index_Y.shape[0]:
    #                         continue

    #                     # get top nb neighbors in mappedX
    #                     neighborListX = Index_ALL[neighborIndex, 1:TS_bestNb+1]
    #                     # get top nb neighbors in mappedY
    #                     neighborListY = Index_Y[neighborIndex, 1:TS_bestNb+1]



                   

                      
                        
    #             # reconstruct with Tscore and similarity weight
    #             innerPredict_TSW = np.zeros(original_train_TV.shape[1])[np.newaxis]
    #             # summing up the best neighbors
    #             for j1 in range(0,TS_bestNb):
    #                 tr = (rearrangedTargetView[:,Index_ALL[tSubjectIndex,j1]])[np.newaxis]
    #                 if j1 == 0:
    #                     innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j1]
    #                 else:
    #                     innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j1]
                        
                   
    #             # scale weight to 1
    #             Scale_TSW = sum(newWeight_TSW)
    #             innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)
    #             if(test==0):
    #                 all_predictedTV_tr = np.c_[innerPredict_TSW]
    #                 #print("=====>>>> First subject in the Hidden SIMLR")
    #             else:
    #                 all_predictedTV_tr = np.c_[all_predictedTV_tr,innerPredict_TSW]
          
    #     #print("=====>>>> Last subject in the Hidden SIMLR")
    #     resul = (all_predictedTV_tr)[np.newaxis]
    #     outputs = all_predictedTV_tr.T
    #     return outputs
        
    


    #  2  OCT
    def new_dataset_predicted_TV(self, emb, ztrTV, rearrangedTargetView, original_train_TV):
        """
        Build a fake batch for the second discriminator by predicting each training
        subject's target-view via an inner LOOCV on the embeddings (train-only).

        Inputs:
          emb:                 (n_train+1, d_z) embeddings for [train; test]   (rows)
          ztrTV:               (n_train,  input_dim) target view (train only)  (rows)
          rearrangedTargetView:(num_features, n_train+1) columns = [train | test]
          original_train_TV:   (n_train,  input_dim) same as ztrTV (rows)

        Return:
          predicted_train_tv:  (n_train, input_dim) – one predicted row per train subject
                           (shape matches real_dist_TV batch in update(...))
        """
        # Basic guards
        if emb is None or ztrTV is None or rearrangedTargetView is None or original_train_TV is None:
            return None

        # Embeddings are stacked as [train; test], so training rows are emb[:-1]
        n_train = ztrTV.shape[0]
        if emb.shape[0] < n_train + 1:
            # Inconsistent inputs; bail out gracefully
            return None

        # For very small n_train, there are no neighbors – fall back to identity
        if n_train <= 1:
            return original_train_TV.astype(np.float32, copy=False)

        simlr = SIMLR.SIMLR_LARGE(1, 10, 0)

        # We will build a prediction for each training subject via inner LOOCV:
        #   hold out one training subject as "inner test", fit SIMLR on the rest,
        #   then reconstruct its TV from its k training neighbors.
        predicted_rows = []

        # Index list of training columns in the outer fold (0..n_train-1)
        outer_train_cols = list(range(n_train))

        for held_out in range(n_train):
            # Inner training indices (exclude held_out)
            inner_keep = [i for i in range(n_train) if i != held_out]

            # Build inner [train; test] in embedding space:
            #   first the kept-train rows, then the held-out row as the last row
            inner_emb = np.vstack([emb[inner_keep, :], emb[held_out:held_out+1, :]])
            # Target-view for inner train (kept rows)
            inner_tv  = ztrTV[inner_keep, :]

            # Fit SIMLR on inner spaces
            SALL, _, _, _ = simlr.fit(inner_emb)
            SY,   _, _, _ = simlr.fit(inner_tv)

            # Convert to ndarray to avoid np.matrix quirks
            sall = np.asarray(SALL.todense())
            sy   = np.asarray(SY.todense())

            Index_ALL = np.argsort(-sall, axis=0)
            Index_Y   = np.argsort(-sy,   axis=0)

            # Positive similarity values for weighting
            des = np.sort(-sall, axis=0)
            Bvalue_ALL = -des

            # In inner_emb, the "test" row is last
            tSubjectIndex = inner_emb.shape[0] - 1  # equals len(inner_keep)

            # Feasible K for inner LOOCV: at least 1, at most 5, and ≤ (#inner_train − 1)
            inner_train_count = len(inner_keep)  # = n_train - 1
            if inner_train_count <= 0:
                # Shouldn't happen because n_train > 1, but guard anyway
                predicted_rows.append(original_train_TV[held_out, :].astype(np.float32, copy=False))
                continue

            k = max(1, min(5, inner_train_count - 1)) if inner_train_count > 1 else 1

            # Candidate neighbors for the inner test (grab >k, then filter to inner training rows)
            row_sorted = np.asarray(Index_ALL[tSubjectIndex, :min(Index_ALL.shape[0], 2*k + 10)]).ravel()
            # Keep only indices < inner_train_count (these are the inner train rows)
            inner_train_cands = [int(ix) for ix in row_sorted if ix < inner_train_count][:k]

            # If we somehow have no candidates, just copy the original row
            if len(inner_train_cands) == 0:
                predicted_rows.append(original_train_TV[held_out, :].astype(np.float32, copy=False))
                continue

            # Weights and reconstruction accumulator
            newWeight_TSW = np.zeros(len(inner_train_cands), dtype=np.float32)
            # Accumulate in TV (feature) space; use the OUTER fold's columns
            # We must map from inner index -> outer training column via inner_keep[]
            innerPredict_TSW = np.zeros((1, original_train_TV.shape[1]), dtype=np.float32)

            for j, inner_neighbor_idx in enumerate(inner_train_cands):
                # Trust overlap using top-k lists in both manifolds
                neighborListX = np.asarray(Index_ALL[inner_neighbor_idx, 0:k]).ravel()
                neighborListY = np.asarray(Index_Y[inner_neighbor_idx,   0:k]).ravel()
                overlap = len(np.intersect1d(neighborListX, neighborListY))

                # Similarity term from Bvalue_ALL; 'j' is the kept-neighbor rank
                newWeight_TSW[j] = exp((overlap / float(k)) * Bvalue_ALL[tSubjectIndex, j])

                # Map inner neighbor row to OUTER training column
                outer_col = inner_keep[inner_neighbor_idx]  # in [0..n_train-1]
                # Take that neighbor's TV vector from rearrangedTargetView (outer columns)
                tr = rearrangedTargetView[:, outer_col].reshape(1, -1)
                innerPredict_TSW += tr * newWeight_TSW[j]

            # Normalize (guard zero)
            Scale_TSW = float(np.sum(newWeight_TSW))
            innerPredict_TSW = innerPredict_TSW / (Scale_TSW if Scale_TSW > 0 else 1.0)

            # Append 1-D row
            predicted_rows.append(innerPredict_TSW.ravel().astype(np.float32, copy=False))

        # Stack to (n_train, input_dim)
        predicted_train_tv = np.vstack(predicted_rows).astype(np.float32, copy=False)
        return predicted_train_tv






    # def erun(self, adj, features, hiddenSIMLR, original_train_TV, ztrTV, rearrangedTargetView, input_dim):


    #     tf.reset_default_graph()
        
    #     model_str = self.model
        
    #     # formatted data
    #     feas = format_data_new(adj, coo_matrix(features))
    #     # newly added
    #     print("Adjacency shape:", feas['adj'].shape)
    #     print("Features shape:", coo_matrix(features).shape)

       
        
    #     # Define placeholders
    #     #placeholders = get_placeholder(feas['adj'])
    #     placeholders = get_placeholder(feas['adj'], input_dim) 



    #     print("Calling get_model with:", feas['num_features'], feas['num_nodes'], feas['features_nonzero'], input_dim)
    #     print("Calling get_model from encoder.py with 6 args")


       
    #     # construct model
    #     #input_dim
    #     d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], input_dim)

        
    #     # Optimizer
    #     opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])





    #     # if(hiddenSIMLR == "No_hidden_SIMLR"):
           
    #     #     # Initialize session
    #     #     sess = tf.Session()
    #     #     sess.run(tf.global_variables_initializer())
           
    #     #     # Train model
    #     #     for epoch in range(self.iteration):
    #     #         #emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, hiddenSIMLR, 1)

    #     #         emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, hiddenSIMLR, 1, input_dim)

    #     #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #     #         if (epoch+1) == 5:
    #     #             break


        
    #     # 2 oct
    #     if (hiddenSIMLR == "No_hidden_SIMLR"):
    #         # Initialize session
    #         sess = tf.Session()
    #         try:
    #             sess.run(tf.global_variables_initializer())

    #             # Train model
    #             for epoch in range(self.iteration):
    #                 emb, avg_cost = update(
    #                     ae_model, opt, sess,
    #                     feas['adj_norm'], feas['adj_label'], feas['features'],
    #                     placeholders, feas['adj'],
    #                     features,                    # prior for real_distribution
    #                     "No_hidden_SIMLR",
    #                     None,                        # new_fake_d not used in this branch
    #                     input_dim
    #                 )
    #                 print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #                 if (epoch + 1) == 5:
    #                     break
    #         finally:
    #             sess.close()





    #     # #(hiddenSIMLR == "Yes_hidden_SIMLR"):
    #     # else:

    #     #     #newly updated
    #     #     d_real_TV, discriminator2, ae_model2 = get_model_2(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], input_dim)
    #     #     #print(tf.trainable_variables())
    #     #     # Optimizer
    #     #     opt2 = get_optimizer_2(model_str, ae_model, discriminator, discriminator2, placeholders, feas['pos_weight'], feas['norm'], d_real_TV, feas['num_nodes'])
            
    #     #     # Initialize session
    #     #     sess = tf.Session()
    #     #     sess.run(tf.global_variables_initializer())



    #     # 2 OCT
    #     else:
    #         # build second discriminator head
    #         d_real_TV, discriminator2, ae_model2 = get_model_2(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], input_dim)
    #         opt2 = get_optimizer_2(model_str, ae_model, discriminator, discriminator2, placeholders, feas['pos_weight'], feas['norm'], d_real_TV, feas['num_nodes'])

    #         # Initialize session
    #         sess = tf.Session()
    #         try:
    #             sess.run(tf.global_variables_initializer())

    #             d_f = None  # will hold the fake batch for D2 (shape: [n_train, input_dim])

    #             for epoch in range(self.iteration):
    #                 if (epoch + 1) == 5:
    #                     break

    #                 if epoch % 2 == 0:
    #                     # Even epochs: update AE + D1 on source space, and build fake batch for D2
    #                     emb, avg_cost = update(
    #                         ae_model, opt, sess,
    #                         feas['adj_norm'], feas['adj_label'], feas['features'],
    #                         placeholders, feas['adj'],
    #                         features,                 # prior for real_distribution in D1
    #                         "No_hidden_SIMLR",
    #                         None,
    #                         input_dim
    #                     )
    #                     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))

    #                     # Build fake batch for the second discriminator from current embeddings
    #                     d_f = self.new_dataset_predicted_TV(
    #                         emb,                       # (n_train+1, d_z)
    #                         ztrTV,                     # (n_train, input_dim)
    #                         rearrangedTargetView,      # (input_dim, n_train+1)
    #                         original_train_TV          # (n_train, input_dim)
    #                     )

    #                 else:
    #                     # Odd epochs: update AE + D2 with target-view supervision (using d_f)
    #                     if d_f is None:
    #                         # Fallback: if we failed to build a fake batch, use zeros of correct shape
    #                         d_f = np.zeros((original_train_TV.shape[0], input_dim), dtype=np.float32)

    #                     emb, avg_cost = update(
    #                         ae_model, opt2, sess,
    #                         feas['adj_norm'], feas['adj_label'], feas['features'],
    #                         placeholders, feas['adj'],
    #                         original_train_TV,         # prior for real_dist_TV
    #                         "Yes_hidden_SIMLR",
    #                         d_f,                       # fake batch for D2
    #                         input_dim
    #                     )
    #                     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #         finally:
    #             sess.close()



    #     # for epoch in range(self.iteration):
    #     #     if (epoch + 1) == 5:
    #     #         break

    #     #     if epoch % 2 == 0:  # even
    #     #         emb, avg_cost = update(
    #     #             ae_model, opt, sess,
    #     #             feas['adj_norm'], feas['adj_label'], feas['features'],
    #     #             placeholders, feas['adj'], features,
    #     #             "No_hidden_SIMLR", features, input_dim  
    #     #         )
    #     #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #     #         d_f = self.new_dataset_predicted_TV(emb, ztrTV, rearrangedTargetView, original_train_TV)

    #     #     # else:  # odd
    #     #     #     emb, avg_cost = update(
    #     #     #         ae_model, opt2, sess,
    #     #     #         feas['adj_norm'], feas['adj_label'], feas['features'],
    #     #     #         placeholders, feas['adj'], original_train_TV,
    #     #     #         hiddenSIMLR, d_f, input_dim  
    #     #     #     )
                


    #     #         emb, avg_cost = update(
    #     #             ae_model, opt2, sess,
    #     #             feas['adj_norm'], feas['adj_label'], feas['features'],
    #     #             placeholders, feas['adj'], original_train_TV,
    #     #             hiddenSIMLR, d_f, input_dim 
    #     #             )
    #     #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))


    #     if hiddenSIMLR != "No_hidden_SIMLR":
    #         for epoch in range(self.iteration):
    #             if (epoch + 1) == 5:
    #                 break

    #             if epoch % 2 == 0:
    #                 emb, avg_cost = update(
    #                     ae_model, opt, sess,
    #                     feas['adj_norm'], feas['adj_label'], feas['features'],
    #                     placeholders, feas['adj'], features,
    #                     "No_hidden_SIMLR", features, input_dim  
    #                 )
    #                 print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #                 d_f = self.new_dataset_predicted_TV(emb, ztrTV, rearrangedTargetView, original_train_TV)

    #             else:
    #                 emb, avg_cost = update(
    #                     ae_model, opt2, sess,
    #                     feas['adj_norm'], feas['adj_label'], feas['features'],
    #                     placeholders, feas['adj'], original_train_TV,
    #                     hiddenSIMLR, d_f, input_dim 
    #                     )
    #                 print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))



           
    #         # # Train model
    #         # for epoch in range(self.iteration):
    #         #     if (epoch+1) == 5:
    #         #             break
    #         #     if(epoch%2==0):#even

    #         #         emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, "No_hidden_SIMLR", 1)


    #         #         #emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features, hiddenSIMLR, features, input_dim)


    #         #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
    #         #         d_f = self.new_dataset_predicted_TV(emb, ztrTV, rearrangedTargetView, original_train_TV)
                    
    #         #     else:#odd
    #         #         emb, avg_cost = update(ae_model, opt2, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], original_train_TV, hiddenSIMLR, d_f)
    #         #         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
                    
        
    #     return emb




    # 2 OCT
    def erun(self, adj, features, hiddenSIMLR, original_train_TV, ztrTV, rearrangedTargetView, input_dim):
        tf.reset_default_graph()
    
        model_str = self.model
    
        # formatted data
        feas = format_data_new(adj, coo_matrix(features))
        print("Adjacency shape:", feas['adj'].shape)
        print("Features shape:", coo_matrix(features).shape)

        # Define placeholders (batch-flexible)
        placeholders = get_placeholder(feas['adj'], input_dim) 

        print("Calling get_model with:", feas['num_features'], feas['num_nodes'], feas['features_nonzero'], input_dim)
        print("Calling get_model from encoder.py with 6 args")

        # construct model / optimizer for the first discriminator (embedding space)
        d_real, discriminator, ae_model = get_model(
            model_str, placeholders,
            feas['num_features'], feas['num_nodes'], feas['features_nonzero'],
            input_dim
        )
        opt = get_optimizer(
            model_str, ae_model, discriminator, placeholders,
            feas['pos_weight'], feas['norm'], d_real, feas['num_nodes']
        )

        if (hiddenSIMLR == "No_hidden_SIMLR"):
            # ------------------------------
            # Phase without hidden SIMLR
            # ------------------------------
            sess = tf.Session()
            try:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.iteration):
                    emb, avg_cost = update(
                        ae_model, opt, sess,
                        feas['adj_norm'], feas['adj_label'], feas['features'],
                        placeholders, feas['adj'],
                        features,            # prior for real_distribution (D1)
                        "No_hidden_SIMLR",
                        None,                # new_fake_d not used in this branch
                        input_dim
                    )
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
                    if (epoch + 1) == 5:
                        break
            finally:
                sess.close()

        else:
            # ------------------------------
            # Phase with hidden SIMLR (dual adversarial)
            # ------------------------------
            d_real_TV, discriminator2, ae_model2 = get_model_2(
                model_str, placeholders,
                feas['num_features'], feas['num_nodes'], feas['features_nonzero'],
                input_dim
            )
            opt2 = get_optimizer_2(
                model_str, ae_model, discriminator, discriminator2, placeholders,
                feas['pos_weight'], feas['norm'], d_real_TV, feas['num_nodes']
            )

            sess = tf.Session()
            try:
                sess.run(tf.global_variables_initializer())
                d_f = None  # fake batch for D2: shape (n_train, input_dim)

                for epoch in range(self.iteration):
                    if (epoch + 1) == 5:
                        break

                    if epoch % 2 == 0:
                        # Even: update AE + D1 (embedding space) and build fake batch for D2
                        emb, avg_cost = update(
                            ae_model, opt, sess,
                            feas['adj_norm'], feas['adj_label'], feas['features'],
                            placeholders, feas['adj'],
                            features,          # prior for real_distribution (D1)
                            "No_hidden_SIMLR",
                            None,
                            input_dim
                        )
                        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))

                        # Build fake batch for D2 from current embeddings
                        d_f = self.new_dataset_predicted_TV(
                            emb,                       # (n_train+1, d_z)
                            ztrTV,                     # (n_train, input_dim)
                            rearrangedTargetView,      # (input_dim, n_train+1) cols=[train|test]
                            original_train_TV          # (n_train, input_dim)
                        )

                    else:
                        # Odd: update AE + D2 (target-view space) with fake batch
                        if d_f is None:
                            d_f = np.zeros((original_train_TV.shape[0], input_dim), dtype=np.float32)

                        emb, avg_cost = update(
                            ae_model, opt2, sess,
                            feas['adj_norm'], feas['adj_label'], feas['features'],
                            placeholders, feas['adj'],
                            original_train_TV,     # prior for real_dist_TV (D2)
                            "Yes_hidden_SIMLR",
                            d_f,                   # fake batch for D2
                            input_dim
                        )
                        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
            finally:
                sess.close()

        return emb
