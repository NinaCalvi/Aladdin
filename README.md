# Aladdin
Informaiton and code for Machine Learning MSc dissertation project at UCL, in collaboration with Aladdin Healthcare Technologies 


A list of possible useful resources:

1. [Drug-drug interaction prediction based on knowledge graph embeddings and convolutional-LSTM network](https://research.vu.nl/en/publications/drug-drug-interaction-prediction-based-on-knowledge-graph-embeddi)
2. [deep-DR (not plublished yet - this would be for drug repurposing)](https://github.com/ChengF-Lab/deepDR)
3. [HetioNet](https://elifesciences.org/articles/26726)
4. [Biomedical knowledge bases](https://openreview.net/pdf?id=B1gGyLFEDV)
5. [DTI](https://www.researchgate.net/publication/321428613_DDR_Efficient_computational_method_to_predict_drug-Target_interactions_using_graph_mining_and_machine_learning_approaches)
6. [dt12](https://www.frontiersin.org/articles/10.3389/fphar.2018.01134/full)
7. [Poster: Knowledge Graph Completion to Predict Polypharmacy Side Effects](https://arxiv.org/pdf/1810.09227.pdf)
8. [You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings](https://openreview.net/forum?id=BkxSmlBFvr)
9. [Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications](https://www.biorxiv.org/content/10.1101/752022v2.full#ref-16)
10. [Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study](https://arxiv.org/abs/2003.08001)
11. [Knowledge Graph Completion to Predict Polypharmacy Side Effects](https://arxiv.org/abs/1810.09227)

12. Sameh's Thesis
- Study KGE and analyse their training pipeline + investigate effects of different trainign components (could take inspiration here to obtain better neural predictors). Different training objectives/negative sampling have different effects in accuracy and scalability 
- **TriVec** model: multiple vectors to model embedding interactions
- When comparing with other SOTA models he uses **MRR and Hits10** metrics. When comparing for the biological tasks, he uses the **AUC** metric.
- Suggests improvements have concentrated on scoring functions but not so much on loss functions, negative sampling etc
- KGE models are extremely sensitive to the different trainign parameters
- TriVec model extend tensor factorisation from ComplEX and DistMult - changes embedding representation and embedding interaction
- mostly inspired by ComplEX - but does not use complex parts - but rather embeddings of three different parts
- interaction of 3 part embeddings: one symmetric interaction, and two asymmetric ones
- showed that squared loss can enhance ComplEX performance on multiple tasks
- **for polypharmacy they use the held out test as defined by the decagon paper, and only report the held out test results which are comparable with state-of-the-art methods**



13. [Modelin Polypharmacy Side Effects Using Graph CNN (DECAGON PAPER)](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770)

14. [Survey of different KG completion methods](https://persagen.com/files/misc/Wang2017Knowledge.pdf)
15. [Evaluation of knowledge graph embedding approaches for drug-drug interaction prediction in realistic settings](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3284-5) 
- DDI predictions report high accuracy however can be due to the systematic bias found in the network - so would report a high accuracy which however would not be realistic
- their method uses disjoint cross validation
- used embeddings from RDF2Vec, TransD and TransE as input to Log Reg, NB, RF etc on DrugBank - RDF2Vec was better
This could be interesting to read with respect to the realistic re-evaluation of kg datasets.
- *adresses evaluation biases by introducing drugwise and pairwise disjoint test classes*
- traditional CV: test pair may share components with training pairs. This could lead to overfitting
- disjoint cv = separates data into grouping according to 'first component of the pair'. Done such as to evaluate predictions on drugs that have no DDI information in the training set (cold start drugs)
- test both for prediciton of interaction between a cold start drug and existing drug, as well as interaction between two cold start drugs
16. [GrEDeL: A Knowledge Graph Embedding Based Method for Drug Discovery From Biomedical Literatures](GrEDeL: A Knowledge Graph Embedding Based Method for Drug Discovery From Biomedical Literatures)
17. [ELPKG: A High-Accuracy Link Prediction Approach for Knowledge Graph Completion](https://www.semanticscholar.org/paper/ELPKG%3A-A-High-Accuracy-Link-Prediction-Approach-for-Ma-Qiao/f9a26d39947c90f7ca432e79f4ce1668061197f9)
18. [Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction](https://arxiv.org/pdf/1911.09419.pdf)
