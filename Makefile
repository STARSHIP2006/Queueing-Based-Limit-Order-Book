.PHONY: install test exp1 exp2 exp3 exp4 exp5 exp6 experiments clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

exp1:
	python -m experiments.experiment_1

exp2:
	python -m experiments.experiment_2

exp3:
	python -m experiments.experiment_3

exp4:
	python -m experiments.experiment_4

exp5:
	python -m experiments.experiment_5

exp6:
	python -m experiments.experiment_6

experiments: exp1 exp2 exp3 exp4 exp5 exp6

clean:
	find data -type f ! -name '.gitkeep' -delete
	find results -type f ! -name '.gitkeep' -delete
