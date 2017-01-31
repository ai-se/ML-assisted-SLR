# Student Project for CSC510, spring 2017.

# Literature Review with Machine Assisted Reading (MAR)

## 1. Form a group of 6
 - Deadline: 
 - Submit your group information:

## 2. Identify your topic (GROUP)
 - Decide a topic (such as "software defect prediction")
 - Form a binary search string
   + Example: software AND (("Document Title":defect) OR ("Document Title":fault) OR ("Document Title":error) OR ("Document Title":bug)) AND (("Document Title":predict*) OR ("Document Title":detect*) OR ("Document Title":assess*) OR ("Document Title":pron*) OR ("Document Title":estimat*))
 - Try it on IEEExplorer
 - Refine the binary search string so that:
   + No less than 3000 results are returned
   + Try not to miss possible studies in your target topic
   + Don't be scared for the large number, you won't read all of them.

## 3. Extract candidate studies (GROUP)
 - Extract all results returned by your binary search string from IEEExplorer (including abstracts). [HOW](https://github.com/ai-se/MAR/issues/3)

## 4. Decide your review protocols (GROUP)
 - A mutually agreed rule to decide whether a candidate study is relevant to your project or not.
 - Example: 1. should be about prediction of defect, 2. should not be about debugging, 3... 
 
## 5. Prepare your tool (INDIVIDUAL)
 - Download and setup MAR tool from [here](https://github.com/ai-se/MAR) 

## 6. Use MAR to select studies relevant to your project (INDIVIDUAL) 
 - Each member of the group, use MAR to select relevant studies. Do **NOT** exchange information with other members. [HOW](https://github.com/ai-se/MAR/issues/4)
 - Say we have group member A, B, C, D, E, F. A, B, C start their own review first, stop after 200 studies have been classified (either as relevant or irrelevant). Export data file as "groupID_unitID.csv".
 - Now we have A.csv, B.csv, C.csv. D, E, F start with open A.csv, B.csv, C.csv with MAR respectively. Then review another 200 studies. Export data file as "groupID_unitID.csv".
 - Therefore we will end up with 3 parallel reviews, A+D, B+E, C+F.
 
## 7. Synthesize results (GROUP)
 - Check all your individual results, summarize studies with at least one member code as "relevant". 
 - Discuss on the list and make final decisions on the relevant study list.
 - Make the final list "groupID.csv".
