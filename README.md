
# 🧠Abstractive Summarization Using GRU & LSTM

This project implements and compares GRU and LSTM-based sequence-to-sequence models with attention mechanisms for the task of **abstractive text summarization**, using the WikiHow dataset.

---

## 🔍 Project Highlights

- 📘 **Models:** GRU and LSTM encoder-decoder architectures with attention
- 📊 **Evaluation:** ROUGE-1, ROUGE-2 and ROUGE-L scores
- 🧪 **Dataset:** WikiHow Abstractive Summarization Dataset (cleaned sample used)
- 📈 **Results:** LSTM with attention slightly outperforms GRU in ROUGE metrics
- 📎 **Outputs:** Summary predictions stored in CSV format for both models

---

## 🗂️ Repository Structure

```
📁 notebooks/          - GRU & LSTM model development and outputs
📁 src/                - Modular Python scripts for model, preprocessing, attention
📁 outputs/            - ROUGE scores and sample predictions
📁 data/               - Sample WikiHow input/output examples
📄 requirements.txt    - Python dependencies
📄 README.md           - Project overview and setup instructions
```

## 📈 Example Results

| Model             | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------|---------|---------|---------|
| GRU + Attention   | 0.3345  | 0.1527  | 0.3079  |
| LSTM + Attention  | 0.3421  | 0.1598  | 0.3185  |

---

## 📦 Sample Output

**Input:** how to make tea  
**GRU Summary:** Boil water, steep bag, serve tea.  
**LSTM Summary:** Boil water, steep tea, add sugar.

---

## 🤝 Authors

- [Anuraag Reddy Kommareddy](mailto:akomm@uic.edu)  
- [Nirmal Kumar Varma Vegesna](mailto:nveges2@uic.edu)
