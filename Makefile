clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf docs/build/

build: clean
	python -m build

check: build
	twine check dist/*

upload-test: check
	twine upload --repository testpypi dist/*

upload: check
	twine upload dist/*
