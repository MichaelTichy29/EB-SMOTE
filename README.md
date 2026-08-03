# EB-SMOTE
This Repository contains the source code for my Bachelor's Thesis in Data Science with the titel "An Analysis of SMOTE-Based Oversampling Methods and the Introduction of Enhanced Borderline SMOTE (EB-SMOTE)." Thus the code contains a new oversampling method which has similarities with SL SMOTE or Border SMOTE. 


# Nominal and continuous variant
The EB- SMOTE is presented in two variants. For data that have continuous features only and for data, which have continuous and categorical features. Thus files with the ending "_NC" are bulit for the extension to handle also nominal features. 

# Statistical Tests
The files stat_F, stat_mcnemar, stat_F_NC, stat_mcnemar_NC are built for statistical tests. As they only compare two methods each time they run without the config. The parameters are set in the source code  as this in only a kind of control center to use the methods impelmented before. 
stat_F means to run a 5 x 2 cv combined F-test (with a test on precision and recal) and mcnemar means use a chi square test on the accuracy. (In fact on the probability for wrong classifications, but this is the same in this case)
