test-all:
	make test-gradient
	make test-reconstruction

test-gradient:
	python -m unittest test/test_gradient.py

test-reconstruction:
	python -m unittest test/test_reconstruction.py
