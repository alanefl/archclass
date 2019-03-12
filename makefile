.PHONY: all figures

all:
    pip install -r requirements.txt
	./util_scripts/fetch_tf_hub_modules.sh
	cd data; python build_dataset.py; cd ..

figures:
	./util_scripts/make_figures.sh
