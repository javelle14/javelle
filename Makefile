.PHONY: all data features train clv decide uplift survival
PY=python
all: data features train clv decide uplift

data:      ; $(PY) -m src.generate_data
features:  ; $(PY) -m src.build_features
train:     ; $(PY) -m src.train_churn
clv:       ; $(PY) -m src.clv
decide:    ; $(PY) -m src.decision
uplift:    ; $(PY) -m src.uplift
survival:  ; Rscript r/survival.R
