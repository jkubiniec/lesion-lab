.PHONY: build test demo origin clean
build:
	python3 lab.py build
test:
	python3 lab.py test
demo:
	python3 lab.py demo --with-executive
origin:
	python3 lab.py evolve --with-executive
clean:
	rm -f build/bff_engine
