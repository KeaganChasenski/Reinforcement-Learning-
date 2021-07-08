VENV := venv

install: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

venv: $(VENV)/bin/activate

ValueIteration: venv
	./$(VENV)/bin/python ValueIteration.py 5 5 -start 0 0 -end 4 4 -k 4 -gamma 0.9

QLearning: venv
	./$(VENV)/bin/python QLearning.py 5 5 -start 0 0 -end 4 4 -k 4  -epochs 350 -learning 0.9 -gamma 0.8

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	find . -type f -name '*.pyc' -delete
	rm -rf *.gif

.PHONY: all venv run clean