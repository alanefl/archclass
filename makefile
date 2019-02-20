.PHONY: all figures

all:
	./util_scripts/fetch_tf_hub_modules.sh
	cd data; python build_dataset.py; cd ..

figures:
	./util_scripts/make_figures.sh
