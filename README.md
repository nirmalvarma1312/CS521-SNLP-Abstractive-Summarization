# CS521 SNLP Abstractive Summarization
Abstractive Summarization Using GRU &amp; LSTM

This project implements and compares GRU and LSTM-based sequence-to-sequence models with attention mechanisms for the task of abstractive text summarization, using the WikiHow dataset.


Project Highlights:


📘 Models: GRU and LSTM encoder-decoder architectures with Bahdanau-style attention

📊 Evaluation: ROUGE-1 and ROUGE-L scores

🧪 Dataset: WikiHow Abstractive Summarization Dataset (230K+ pairs)

📈 Results: LSTM with attention outperforms GRU in both coherence and ROUGE metrics

📎 Visualization: Includes qualitative summary examples and attention heatmaps

📁 notebooks/          - GRU & LSTM model development and outputs

📁 src/                - Modular Python scripts for model, preprocessing, attention

📁 outputs/            - ROUGE scores and sample outputs

📁 data/               - Sample WikiHow input/output examples

📄 requirements.txt    - Python dependencies

📄 README.md           - Project overview and setup instructions


pip install -r requirements.txt

python train.py    # To train the model

python evaluate.py # To evaluate the model


Example Results

Model	            ROUGE-1	      ROUGE-L

GRU + Attention	   0.412	       0.398

LSTM + Attention	 0.471	       0.456


Sample Output


Input: how to make tea

GRU Summary: Boil water, steep bag, serve tea.

LSTM Summary: Boil water, steep tea, add sugar.

Authors

Anuraag Reddy Kommareddy

Nirmal Kumar Varma Vegesna
