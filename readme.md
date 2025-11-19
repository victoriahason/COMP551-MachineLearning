# Projects from COMP 551 — Machine Learning Research (McGill)

This repository contains four ML projects completed in COMP551, covering classical ML, regression methods, deep learning for image classification, and transformer-based NLP.

---

## 🩺 1. KNN & Decision Trees From Scratch (Heart Disease & Penguins)
**Technologies:** NumPy, Python, custom KNN + DT implementations  
**Highlights:**  
- Implemented KNN (weighted + multiple distance metrics) and Decision Trees entirely from scratch.  
- Strong performance on Penguins: ~99% accuracy (KNN), ~98–100% (DT).  
- Heart Disease was harder: DT (depth=3) achieved **88.3%**, outperforming KNN (**77%**).  
- Feature selection, mixed-data experiments, and AUROC comparisons included.  
**Key Finding:** DTs perform better on mixed/tabular medical data; KNN excels when classes are well-separated.  

---

## 🍷 2. Linear vs. Logistic Regression (Breast Cancer & Wine)
**Technologies:** NumPy, gradient descent, logistic & multiclass regression from scratch  
**Highlights:**  
- Compared linear regression, logistic regression, and multiclass softmax regression.  
- Logistic regression consistently outperformed linear regression for classification tasks.  
- Breast Cancer: ~0.99 AUC for logistic; Wine: **0.977 test accuracy** for both logistic & multivariate regression.  
- Included analytical vs. numerical gradient checks, CE-loss convergence, and feature-importance analysis.  
**Key Finding:** Even though linear regression can approximate classification, logistic regression yields more stable and interpretable decision boundaries.  

---

## 🈶 3. MLPs vs. CNNs for Japanese Character Recognition (KMNIST)
**Technologies:** Custom MLP (NumPy), PCA, CNN (TensorFlow/Keras)  
**Highlights:**  
- Built MLPs from scratch and evaluated depth, width, activations (ReLU, LeakyReLU, Sigmoid), batch sizes, and L2 regularization.  
- Best MLP: **91.16%** accuracy (2-layer, 256 units, PCA preprocessing).  
- CNN baseline achieved **~93%**, outperforming all MLPs.  
**Key Finding:** CNNs dominate for image tasks due to spatial feature extraction, but PCA-boosted MLPs can be competitive.  

---

## 📰 4. BERT Probing & Fine-Tuning on AG News
**Technologies:** HuggingFace Transformers, PyTorch, BERT-base-uncased  
**Highlights:**  
- Probing: froze BERT and tested CLS, last-token, and mean-pooled embeddings with logistic regression & KNN.  
  - Best probe: **Mean pooling + Logan (88.3%)**.  
- Fine-tuning: Full BERT model reached **92.7% accuracy**, macro-F1 = 0.927.  
- Included attention visualizations across layers → deeper layers focused on class-specific semantic cues.  
**Key Finding:** Frozen embeddings already encode strong semantics, but task-specific fine-tuning gives a significant performance boost.  
