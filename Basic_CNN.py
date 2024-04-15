# All needed packages
import argparse
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from imblearn.over_sampling import SMOTE
from sklearn.metrics import *
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc


# Import my files
from load_datasets import *
from pathScores_functions import *
from get_Embeddings_FV import *
from training_functions import *
from GIP import *
from snf_code import *
######################################## START MAIN #########################################
#############################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Your description here.")
    parser.add_argument("--data", type=str, help="Your data argument here.")
    # Add other arguments as needed
    return parser.parse_args()


def main():
    # get the parameters from the user
    args = parse_args()
    ## get the start time to report the running time
    t1 = time.time()

    ### Load the input data - return all pairs(X) and its labels (Y)..
    allD, allT, allDsim, allTsim, DrTr, R, X, Y = load_datasets(args.data)

    # create 2 dictionaries for drugs. the keys are their order numbers
    drugID = dict([(d, i) for i, d in enumerate(allD)])
    targetID = dict([(t, i) for i, t in enumerate(allT)])
    #-----------------------------------------

    # CNN Neural Network
    cnn = Sequential()
    cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(16, 1)))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #________________________________________________________________
    # 10-folds Cross Validation...............
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 22)
    skf.get_n_splits(X, Y)
    foldCounter = 1
    # all evaluation lists
    correct_classified = []
    ps = []
    recall = []
    roc_auc = []
    average_precision = []
    f1 = []
    Pre = []
    Rec = []
    AUPR_TEST = []
    TN = []
    FP = []
    FN = []
    TP = []
    all_dt_PredictedScore = []

    #Create file to write the novel interactions based on predicted scores
    novel_DT_file = 'Novel_DTIs/'+args.data+'/'+args.data+'_top_novel_DTIs.csv'
    
    # Start training and testing
    for train_index, test_index in  skf.split(X,Y):

        print("*** Working with Fold %i :***" %foldCounter)
        
        #first thing with R train to remove all edges in test (use it when finding path)
        train_DT_Matrix = Mask_test_index(test_index, X, Y, DrTr, drugID, targetID)
        DrTr_train = train_DT_Matrix.transpose()
        #--------------------------------------------

        DDsim = []
        TTsim = []

        for sim in allDsim:
            DDsim.append(sim)

        for sim in allTsim:
            TTsim.append(sim)

        fused_simDr = SNF(DDsim,K=5,t=3,alpha=1.0)
        fused_simTr = SNF(TTsim,K=5,t=3,alpha=1.0)
        
        #------------------------------ node2vec ------------------------------

        targetFV, drugFV = get_FV_drug_target(foldCounter, allT, allD, args.data)
        
        # Calculate cosine similarity for each drug pair, and for each target pair
        cos_simDD = Cosine_Similarity(drugFV)
        cos_simTT = Cosine_Similarity(targetFV)
        # normalize simiarities to be in positive range [0,1]
        cos_simDD = normalizedMatrix(cos_simDD)
        cos_simTT  = normalizedMatrix(cos_simTT )
        #--------------------------------------------------------------------- 

        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G1)
        sumDDD, maxDDD = DDD_TTT_sim(fused_simDr)
        sumTTT, maxTTT = DDD_TTT_sim(fused_simTr)
        
        sumDDT,maxDDT = metaPath_Dsim_DT(fused_simDr,DrTr_train,2) 
        sumDTT,maxDTT = metaPath_DT_Tsim(fused_simTr,DrTr_train,2)

        sumDDDT,_= metaPath_Dsim_DT(sumDDD,DrTr_train,3)
        _,maxDDDT = metaPath_Dsim_DT(maxDDD,DrTr_train,3)

        sumDTTT,_ = metaPath_DT_Tsim(sumTTT,DrTr_train,3)
        _,maxDTTT = metaPath_DT_Tsim(maxTTT,DrTr_train,3)

        sumDTDT,maxDTDT = metaPath_DTDT(DrTr_train)
        sumDDTT,maxDDTT = metaPath_DDTT(DrTr_train,fused_simDr,fused_simTr)
    #============================================================================== 
        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G2)
        sumDDD2, maxDDD2 = DDD_TTT_sim(cos_simDD)
        sumTTT2, maxTTT2 = DDD_TTT_sim(cos_simTT)
        
        sumDDT2,maxDDT2 = metaPath_Dsim_DT(cos_simDD,DrTr_train,2) 
        sumDTT2,maxDTT2 = metaPath_DT_Tsim(cos_simTT,DrTr_train,2)

        sumDDDT2,_ = metaPath_Dsim_DT(sumDDD2,DrTr_train,3)
        _,maxDDDT2 = metaPath_Dsim_DT(maxDDD2,DrTr_train,3)

        sumDTTT2,_ = metaPath_DT_Tsim(sumTTT2,DrTr_train,3)
        _,maxDTTT2 = metaPath_DT_Tsim(maxTTT2,DrTr_train,3)

        sumDTDT2,maxDTDT2 = metaPath_DTDT(DrTr_train)
        sumDDTT2,maxDDTT2 = metaPath_DDTT(DrTr_train,cos_simDD,cos_simTT)
    #==============================================================================  
    ### Build feature vector and class labels
        DT_score = []
        for i in range(len(allD)):
            for j in range(len(allT)):        
                pair_scores = (allD[i], allT[j],\
                            # path scores from G1
                               sumDDT[i][j],sumDDDT[i][j],\
                               sumDTT[i][j],sumDTTT[i][j], sumDDTT[i][j], sumDTDT[i][j],\
                               maxDDT[i][j],maxDDDT[i][j], \
                               maxDTT[i][j],maxDTTT[i][j],maxDDTT[i][j],maxDTDT[i][j],\
                            # path scores from G2
                               sumDDT2[i][j],sumDDDT2[i][j],\
                               sumDTT2[i][j],sumDTTT2[i][j], sumDDTT2[i][j], sumDTDT2[i][j],\
                               maxDDT2[i][j],maxDDDT2[i][j], \
                               maxDTT2[i][j],maxDTTT2[i][j],maxDDTT2[i][j],maxDTDT2[i][j])
                DT_score.append(pair_scores)
        
        features = []
        class_labels = []
        DT_pair = []
        # Build the feature vector - Concatenate features from G1,G2
        for i in range(len(DT_score)):
            dr = DT_score[i][0]
            tr = DT_score[i][1] 
            edgeScore = DT_score[i][2], DT_score[i][3], DT_score[i][4],DT_score[i][5],\
                        DT_score[i][8],DT_score[i][9], DT_score[i][10], DT_score[i][11],\
                        DT_score[i][14], DT_score[i][15],DT_score[i][16],DT_score[i][17],DT_score[i][18],\
                        DT_score[i][20], DT_score[i][21],DT_score[i][22]
           
            dt = DT_score[i][0], DT_score[i][1]
            DT_pair.append(dt)
            features.append(edgeScore)
            # same label as the begining
            label = R[dr][tr]
            class_labels.append(label)

        ## Start Classification Task
        # featureVector and labels for each pair
        XX = np.asarray(features)
        YY = np.array(class_labels)

        #Apply normalization using MaxAbsolute normlization
        max_abs_scaler = MaxAbsScaler()
        X_train_transformed = max_abs_scaler.fit_transform(XX[train_index])
        X_test_transformed = max_abs_scaler.transform(XX[test_index])

        # Apply different oversampling techniques:
        sm = SMOTE(random_state=10)
        X_res, y_res = sm.fit_resample(X_train_transformed, YY[train_index])
        X_res_reshaped = X_res.reshape(X_res.shape[0], X_res.shape[1], 1)
        cnn.fit(X_res_reshaped, y_res)

        # Reshape the input data to fit the Conv1D input shape
        X_train_transformed_reshaped = X_train_transformed.reshape(X_train_transformed.shape[0], X_train_transformed.shape[1], 1)
        X_test_transformed_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], X_test_transformed.shape[1], 1)
        
        # Fit the model
        # Before fitting the model, slice y_res to match the number of samples in X_train_transformed_reshaped
        y_res_sliced = y_res[:X_train_transformed_reshaped.shape[0]]

        # Fit the model using the sliced target labels
        cnn.fit(X_train_transformed_reshaped, y_res_sliced)


        predictedProbabilities = cnn.predict(X_test_transformed_reshaped)
        # Assuming 'y_pred' is your model's predicted probabilities
        threshold = 0.4  # Default threshold
        
        # Adjust the threshold
        predictedClass = (predictedProbabilities > threshold).astype(int)

        predictedScore = cnn.predict(X_test_transformed_reshaped)
    

        #Find the novel interactions based on predicted scores
        fold_dt_score = []
        for idx, c in zip(test_index,range(0,len(predictedScore))):
            # write drug, target, predicted score of class1, predicted class, actual class
            dtSCORE = str(DT_pair[idx]),predictedScore[c],predictedClass[c],YY[idx]
            all_dt_PredictedScore.append(dtSCORE)



        # ------------------- Print Evaluation metrics for each fold --------------------------------
        print("@@ Validation and evaluation of fold %i @@" %foldCounter)
        print(YY[test_index].shape, predictedClass.shape)


        cm = confusion_matrix(YY[test_index], predictedClass)
        TN.append(cm[0][0])
        FP.append(cm[0][1])
        FN.append(cm[1][0])
        TP.append(cm[1][1])
        print("Confusion Matrix for this fold")
        print(cm)

        print("Correctly Classified Instances: %d" %accuracy_score(Y[test_index], predictedClass, normalize=False))
        correct_classified.append(accuracy_score(Y[test_index], predictedClass, normalize=False))

        #print("Precision Score: %f" %precision_score(Y[test_index], predictedClass))
        ps.append(precision_score(Y[test_index], predictedClass,average='weighted', zero_division=1))

        #print("Recall Score: %f" %recall_score(Y[test_index], predictedClass)
        recall.append(recall_score(Y[test_index], predictedClass, average='weighted'))

        print("F1 Score: %f" %f1_score(Y[test_index], predictedClass, average='weighted'))
        f1.append(f1_score(Y[test_index], predictedClass,average='weighted'))

        print("Area ROC: %f" %roc_auc_score(Y[test_index], predictedScore))
        roc_auc.append(roc_auc_score(Y[test_index], predictedScore))

        # Modify precision_recall_curve with zero_division parameter
        p, r, _ = precision_recall_curve(Y[test_index], predictedScore, pos_label=1)
        aupr = auc(r, p)
        print("AUPR auc(r,p) = %f" %aupr)
        AUPR_TEST.append(aupr)

        Pre.append(p.mean())
        Rec.append(r.mean())
        average_precision.append(average_precision_score(Y[test_index], predictedScore))

        print(classification_report(Y[test_index], predictedClass))
        print('--------------------------------------------------')
        foldCounter += 1
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

    # Write predicted scores into file to find novel interactions:
    dt_df = pd.DataFrame(all_dt_PredictedScore, columns=['DT_pair', 'Predicted_score_class1', 'Predicted_Class', 'Actual_Class'])
    dt_df = dt_df.sort_values(by='Predicted_score_class1', ascending=False)
    
    dt_df = dt_df[dt_df['Predicted_Class']==1]
    novel_dt = dt_df[dt_df['Actual_Class']==0]

    novel_dt.to_csv(novel_DT_file,sep='\t', index=None)
    #--------------------------------------------------------------------
    ############# Evaluation Metrics ####################################
    # Confusion matrix for all folds
    ConfMx = np.zeros((cm.shape[0],cm.shape[0]))
    ConfMx[0][0] = str( np.array(TN).sum() )
    ConfMx[0][1] = str( np.array(FP).sum() )
    ConfMx[1][0] = str( np.array(FN).sum() )
    ConfMx[1][1] = str( np.array(TP).sum() )

    ### Print Evaluation Metrics.......................
    print("Result(Correct_classified): " + str( np.array(correct_classified).sum() ))
    print("Results:precision_score = " + str( np.array(ps).mean().round(decimals=3) ))
    print("Results:recall_score = " + str( np.array(recall).mean().round(decimals=3) ))
    print("Results:f1 = " + str( np.array(f1).mean().round(decimals=3) ))
    print("Results:roc_auc = " + str( np.array(roc_auc).mean().round(decimals=3) ))
    print("Results: AUPR on Testing auc(r,p) = " + str( np.array(AUPR_TEST).mean().round(decimals=3)))
    print("Confusion matrix for all folds")
    print(ConfMx) 
    print('_____________________________________________________________')
    print('Running Time for the whole code:', time.time() - t1)  
    print('_____________________________________________________________')

    import matplotlib.pyplot as plt
    import seaborn as sns
    # Assuming YY contains your class labels
    # Label visualization
    sns.countplot(x=YY)
    plt.title('Class Distribution')
    plt.show()

    # ROC AUC curve visualization
    # Assuming Y[test_index] and predictedScore are your true labels and predicted probabilities
    fpr, tpr, _ = roc_curve(Y[test_index], predictedScore)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Heat map visualization
    # Assuming predictedClass and Y[test_index] are your predicted and true class labels
    sns.heatmap(ConfMx.astype(int), annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    # Assuming 'ps', 'recall', 'f1', 'roc_auc', and 'AUPR_TEST' are lists of scores for each fold
    metrics_scores = {
        'Precision': np.mean(ps),
        'Recall': np.mean(recall),
        'F1 Score': np.mean(f1),
        'ROC AUC': np.mean(roc_auc),
        'AUPR': np.mean(AUPR_TEST)
    }

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_scores.keys(), metrics_scores.values(), color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Average Score')
    plt.title('Average Performance Metrics Scores')
    plt.ylim(0, 1)  # Assuming the scores are normalized between 0 and 1
    plt.show()
#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    main()
#####-------------------------------------------------------------------------------------------------------------
####################### END OF THE CODE ##########################################################################
