# Local runs use PYTHONPATH=src (robust to macOS hiding the editable-install .pth file).
PY      ?= .venv/bin/python
RUN     := PYTHONPATH=src $(PY) -m
CUTOFF  ?= 2026-06-30

.PHONY: setup data features backtest research intervals evaluate train shadow dashboard monthly test lint all

setup:            ## create venv and install everything
	uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[research,dev]"

data:             ## download WIM (server-side aggregated), MTA, weather, diesel
	$(RUN) wim_forecast.download && $(RUN) wim_forecast.external

features:         ## daily feature tables (D-1 and nowcast)
	$(RUN) wim_forecast.build_dataset

backtest:         ## baselines + LightGBM setups + ablations (~10 min)
	$(RUN) wim_forecast.backtest --baselines
	$(RUN) wim_forecast.backtest --setup "D-1"
	$(RUN) wim_forecast.backtest --setup "D-1, weather forecast" --weather-forecast
	$(RUN) wim_forecast.backtest --setup "Nowcast 07:00" --table daily_nowcast
	$(RUN) wim_forecast.backtest --ablation
	$(RUN) wim_forecast.backtest --ablation --table daily_nowcast

research:         ## GRU, classical TS, Chronos, N-HiTS (~45 min, needs the [research] extra)
	$(RUN) wim_forecast.train_gru --setup "D-1"
	$(RUN) wim_forecast.train_gru --setup "Nowcast 07:00" --table daily_nowcast
	$(RUN) wim_forecast.stat_baselines
	$(RUN) wim_forecast.foundation --device $${DEVICE:-cpu}
	$(RUN) wim_forecast.hourly_nhits

intervals:        ## conformal prediction intervals
	$(RUN) wim_forecast.conformal && $(RUN) wim_forecast.conformal --table daily_nowcast

evaluate:         ## tables -> reports/, figures -> reports/figures/
	$(RUN) wim_forecast.evaluate

train:            ## train + freeze production models at CUTOFF
	$(RUN) wim_forecast.cli train --cutoff $(CUTOFF)

shadow:           ## score frozen models on days after the cutoff, check drift
	$(RUN) wim_forecast.cli shadow

dashboard:        ## render docs/index.html
	$(RUN) wim_forecast.cli dashboard

monthly:          ## what the scheduled workflow runs
	$(RUN) wim_forecast.cli monthly

test:
	PYTHONPATH=src $(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

all: data features backtest research intervals evaluate train shadow dashboard
