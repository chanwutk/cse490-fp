all: install

install:
	pip3 install -r requirements.txt

test:
	python -m pytest tests/