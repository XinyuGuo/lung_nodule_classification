# lung_nodule_classification
A 3D Dual-path Resnet for Lung Nodule Classification 

## Introduction
The cryptococcal granulomas are very similar as the small malignant pulmonary nodules in morphology and pulmonary position.  The high similarity can make the early diagnosis of lung cancer challenging, and many patients with cryptococcal granulomas suffer the pain of biopsy confirmation. 

In this repo, three pipelines were established to address the challenging, and facilitate the fast, accurate diagnosis in clinic. A dual-path deep learning (DL) pipeline mimicking the diagnosis process of the radiologists was built to differentiate between the cryptococcal granulomas and the small malignant nodules based on nodule and lung area from the contrast CT scans. The single-path DL pipeline as well as the radiomics pipeline was established respectively to explore the diagnosis performance solely based on the nodule area. The transfer learning training strategy was employed to train both DL pipelines for achieving the better results. 

We conducted a single-center study including 714 contrast CT scans from 295 patients (87 with cryptococcosis and 208 with lung cancer). The disease status of each patient was biopsy confirmed. Three classification pipelines were evaluated by 5-fold cross validation on the train dataset, then the best model was selected to do the performance evaluation on the test dataset. The mean area under the receiver operating characteristic curve (AUC) was the performance metric. Compared to the single-path DL pipeline (mean AUC 0.85, test AUC 0.80) and the radiomics pipeline (mean AUC 0.79, test AUC 0.75), the dual-path DL pipeline achieved the best mean AUC 0.88, and the best test AUC 0.83. The radiomics pipeline can achieve competitive results as the single-path DL pipeline, and 12 important features were selected for the further interpretation. 

Please find more details in manuscript.pdf
