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
