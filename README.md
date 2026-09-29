Customer Churn Prediction
A reproducible portfolio project with scikit-learn, PyTorch and TensorFlow/Keras.
Predict customer churn from historical service usage, compare classical models with small neural networks, and produce a ranked list for retention review. This is an educational benchmark, with no claim of production deployment or measured revenue impact.
Results from the included run
Selected model: TensorFlow MLP, selected by validation average precision (AP).
Held-out ROC-AUC: 0.978; AP: 0.907; F1: 0.802.
Data: 3,150 source rows; 300 exact duplicate rows removed; 2,850 retained.
Partitions: 1,708 training / 569 validation / 573 test records.
Threshold: 0.75, chosen using validation F1 only.
Top 20% by score: captures 87.9% of test churners, with 4.38x lift over random selection.
Model	Validation AP	Test ROC-AUC	Test AP	Test precision	Test recall	Test F1
Majority baseline	0.156	0.500	0.159	0.000	0.000	0.000
Logistic Regression	0.742	0.929	0.746	0.503	0.802	0.619
Random Forest	0.878	0.971	0.877	0.669	0.868	0.756
PyTorch MLP	0.910	0.978	0.909	0.800	0.791	0.796
TensorFlow MLP	0.922	0.978	0.907	0.762	0.846	0.802
AP means average precision, not trapezoidal PR-AUC. Test results are from one fixed split, not a guarantee for new customers. Accuracy alone is insufficient for this imbalanced dataset. Every model uses its own validation-selected threshold. The majority baseline's tied ranking has no useful retention interpretation.
![Held-out evaluation](reports/curves.png)
What is implemented
A source checksum and a data quality audit.
Column normalization, exact duplicate removal and two label-free usage ratios.
A roughly 60/20/20 split with identical feature profiles kept in the same partition.
Training-only imputation, scaling and tariff one-hot encoding.
Majority-class baseline, Logistic Regression, Random Forest, PyTorch MLP and TensorFlow MLP.
Class weighting, neural-network early stopping, and validation-only model/threshold selection.
Held-out metrics, confusion matrix, precision-recall curves, top-20% lift and validation permutation importance.
Serialized neural networks, inference CLI, SQL segment analysis, executed notebook and automated checks.
Quick start: Windows / PowerShell (Python 3.12)
Extract the ZIP, open a terminal inside `customer-churn-ml`, and run:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python train.py
.\.venv\Scripts\python predict.py --input data/example_customers.csv --output predictions.csv
.\.venv\Scripts\python -m pytest -q
```
Quick start: Linux (Python 3.12)
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python train.py
.venv/bin/python predict.py --input data/example_customers.csv --output predictions.csv
.venv/bin/python -m pytest -q
```
The first installation downloads both neural-network frameworks. A GPU is not required. The bundled CSV and model artifacts allow offline analysis after packages are installed. macOS users should install native PyTorch/TensorFlow wheels instead of the CPU requirements file; macOS was not tested. Reproduction was tested on Linux with Python 3.12.14.
Notebook
Open `notebooks/customer_churn_walkthrough.ipynb` in VS Code or Jupyter. The included outputs have been executed. Set `RETRAIN = True` in the notebook to rebuild the artifacts, or use `python train.py`. The default walkthrough reads the included verified run.
For command-line execution using the current Python environment:
```bash
python scripts/execute_notebook.py
```
This script executes every Python cell with IPython in the current process and saves the real outputs; it does not need a kernel socket. Normal Jupyter Run All also works.
Neural networks
Both frameworks implement a small dense network: input → 32 ReLU → dropout 0.1 → 16 ReLU → 1 logit. Training uses weighted binary cross-entropy, Adam, batch size 64 and a maximum of 150 epochs. Early stopping uses validation AP with patience 20; the best weights are restored.
PyTorch demonstrates `nn.Module`, `DataLoader`, `BCEWithLogitsLoss`, backpropagation and `state_dict` checkpoints.
TensorFlow demonstrates Keras layers, `GradientTape`, weighted logit loss and `.keras` serialization.
Same architecture/hyperparameters does not imply identical weights: initializers, optimizers and random streams differ. This is a learning comparison, not a speed benchmark.
Evaluation decisions
Exact duplicates are removed before splitting. Groups are hashes of the model inputs, not of the label, so feature-identical rows with different labels also stay together. `StratifiedGroupKFold` assigns fixed partitions; this is not a five-fold cross-validation performance estimate.
`Status` and the derived `Customer Value` are excluded as a conservative feature-scope choice. Age and age-group fields are also omitted. The two ratio features use `calls + 1` to handle zero-call customers. Correlated features can share permutation importance, so that chart is not causal evidence.
All transforms fit on training only. Validation selects the winning model, neural epoch and decision threshold. Test labels are used afterward for reporting, not tuning. The saved winner retains those training-only weights and the fixed threshold; it is not refitted on train + validation.
Scores from class-weighted models are ranking scores, not calibrated probabilities. Top-20% outreach is a capacity illustration; a real retention campaign needs costs, capacity and an experiment to measure impact.
Repository guide
Path	Purpose
`src/churn.py`	Preprocessing, splits, all models, evaluation and inference
`train.py` / `predict.py`	Training and scoring entry points
`notebooks/`	Executed explanation and analysis
`data/raw/`	Unmodified attributed source CSV
`data/example_customers.csv`	Five illustrative rows for inference, without labels
`models/`	Saved model weights, preprocessor and metadata
`reports/`	JSON/CSV metrics, predictions, audit and charts
`sql/retention_analysis.sql`	Complaint/tariff analysis, training partition only
`tests/`	Leakage, preprocessing, schema and serialization checks
`docs/INTERVIEW_GUIDE_HI.md`	Hindi explanations and interview practice
`docs/MODEL_CARD.md`	Scope, assumptions and limits
`train.py` also creates `data/processed/churn.sqlite`. It is intentionally excluded from the ZIP/Git; it can be regenerated. SQL output is bundled in `reports/sql_retention_segments.csv`. Load only model files you trust, because joblib uses pickle-based serialization.
Put this on GitHub
Suggested repository name: `customer-churn-pytorch-tensorflow`.
Create an empty repository, extract this ZIP, and upload the contents of `customer-churn-ml`. Keep `README.md` at the repository root. Do not upload the virtual environment. The package is small enough for a normal repository; a Git client also preserves `.gitignore` automatically.
```bash
git init
git add .
git commit -m "Add reproducible customer churn ML portfolio"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
Replace `YOUR_REPOSITORY_URL` with the URL GitHub gives you. No remote repository has been created or published by this package. After publishing, use the actual repository URL in your resume.
Data and attribution
Iranian Churn, UCI Machine Learning Repository (2020), DOI 10.24432/C5JW3Z, is supplied under CC BY 4.0. UCI describes first-nine-month features and a churn label at month 12. There are no per-record timestamps for a chronological backtest. The raw CSV is unchanged; derivative cleanup is documented above and in `DATA_SOURCE.md`.
Reference documentation
scikit-learn: StratifiedGroupKFold
PyTorch: optimization loop
TensorFlow: custom training loops
Code is MIT-licensed; the dataset retains its separate CC BY 4.0 terms. Developed with AI assistance for Dashrath Kumar's portfolio. Review, run and adapt the code so you can explain the decisions in an interview.
