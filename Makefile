.PHONY: setup data train dashboard notebooks test all clean

PY := .venv/bin/python
JUPYTER := .venv/bin/jupyter

setup:  ## create venv and install dependencies
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --user --name nhanes-diabetes --display-name "Python (nhanes-diabetes)"

data:  ## rebuild artifacts/analysis_cohort.csv from raw NHANES files
	$(PY) -c "from src.data import prepare; print(prepare().shape)"

train:  ## fit + calibrate + evaluate, write models/ and reports/
	$(PY) -m src.pipeline

dashboard:  ## rebuild the Tableau / Power BI extracts in dashboard/data/
	$(PY) -m src.dashboard

notebooks:  ## execute all notebooks in place
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=python3 \
		notebooks/01_data_preparation.ipynb \
		notebooks/02_exploratory_data_analysis.ipynb \
		notebooks/03_modeling.ipynb

test:  ## run the unit tests
	$(PY) -m pytest

all: data train dashboard test  ## full reproduction

clean:  ## remove generated artifacts
	rm -f artifacts/analysis_cohort.csv models/*.joblib reports/model_metrics.json reports/figures/*.png
	rm -rf dashboard/data
