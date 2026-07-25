# Medical Article Classification by Category

An NLP project comparing different text representations and classification models to categorize scientific medical articles (Nutrition, Exercise, Fasting) from a small corpus of PDF documents.

## The real problem behind the project

Beyond "classifying text," the core goal of this project was methodological: **with a very small corpus (16 documents), how reliable are a classification model's results, really?** The project is designed end-to-end to answer that question rigorously, not just to maximize a metric.

## Phase 1 — Corpus exploration and data quality

Original corpus of 20 PDF documents, reduced to 16 after detecting and removing an exact duplicate. The reasoning matters here: keeping a duplicate document in the corpus creates *data leakage* if, by chance, one copy lands in the training set and the other in the test set — the model would memorize the exact text instead of learning to generalize, artificially inflating the final result.

## Phase 2 — Preprocessing (with justification for each technique)

Every text-cleaning technique (lowercasing, special character removal, etc.) was documented by explaining what problem it solves and how it affects the final result — preprocessing wasn't applied "by habit" without understanding its effect.

## Phase 3 — Text representations

Four different representations of the same text were built and compared:

| Representation | Type | Core idea |
|---|---|---|
| **BoW** (Bag of Words) | Sparse | Counts word frequency, simple and robust |
| **TF-IDF** | Sparse | Like BoW, but down-weights words that are common across documents |
| **Word2Vec** | Dense | Semantic embeddings trained on the corpus itself |
| **BERT** | Dense | Pretrained embeddings via transfer learning |

## Phase 4 — Modeling

- **Logistic Regression** on each of the 4 representations (fixed vector per text, no notion of word order)
- **RNN and LSTM** as sequential models, processing text word by word while maintaining an internal state — in theory capable of capturing order and context, but with more parameters to learn from very little available data

## Phase 5 — Evaluation

Accuracy, Precision, Recall and F1 were measured, with one key methodological decision: since the classes are imbalanced (Nutrition ≫ Exercise ≫ Fasting), Precision/Recall/F1 are calculated using **macro averaging**, not weighted or micro. This gives equal weight to each category regardless of how many examples it has, preventing strong performance on the majority class from masking poor performance on minority classes.

## Phase 6 — Results and conclusions

**With a single train/test split:**

| Model | F1 (macro) |
|---|---|
| BERT | 0.849 |
| BoW | 0.834 |
| TF-IDF | 0.631 |
| RNN | 0.294 |
| LSTM | 0.275 |
| Word2Vec | 0.262 |

**Key finding:** LSTM had a relatively high Accuracy (0.701) despite a very low F1 (0.275). This happens because the model collapsed to predicting the majority class ("Nutrition") almost every time, getting it right by sheer frequency without actually distinguishing between categories — a real-world example of why Accuracy alone can be misleading with imbalanced classes, and why macro F1 was prioritized from the experiment's design onward.

**Additional validation with Leave-One-Out Cross-Validation** (evaluating document by document, not with a single split): the real accuracy of BoW and BERT dropped to 0.661 and 0.632 respectively, with very high variability between documents. This confirms that the strong result from the single split was partly influenced by which specific documents landed in train vs. test, and is not a reliable measure of how well the model would generalize to new data.

**Overall conclusion:** with a corpus of only 16 documents, simple and robust representations (BoW) or pretrained ones using transfer learning (BERT) substantially outperformed representations that need to learn from scratch with little data (Word2Vec, RNN, LSTM). A larger corpus would be needed to draw more solid conclusions about which approach is truly "best."

## Tech stack

- **Python** — pandas, numpy, scikit-learn
- **sentence-transformers** (BERT) for pretrained embeddings
- **TensorFlow/Keras** for RNN and LSTM
- **scikit-learn** — `LogisticRegression`, `LeaveOneGroupOut`, classification metrics

## Why it's in my portfolio

Not because of the best model's score, but because of the process: identifying data leakage risk before modeling, justifying every preprocessing decision, choosing the right metric for the problem (macro F1 over Accuracy on an imbalanced dataset), and validating results with a second methodology (LOO-CV) instead of trusting a single train/test split. That discipline of questioning your own results is, in my opinion, more valuable than any single accuracy number.
