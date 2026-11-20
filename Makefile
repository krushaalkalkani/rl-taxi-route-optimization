.PHONY: install test lint grid train evaluate export calibrate validate web all

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

grid:            ## Stage 1: tabular Q-learning on Gymnasium Taxi-v4
	uv run python -m taxi_rl.taxi_grid

train:           ## Stage 2: MaskablePPO on CityTaxi-v0 (≈30-40 min on a laptop CPU)
	uv run python -m taxi_rl.train --timesteps 3000000 --n-envs 10 --eval-every 150000

evaluate:        ## 240 held-out shifts, all policies
	uv run python -m taxi_rl.evaluate --episodes 240

export:          ## JSON for the website
	uv run python -m taxi_rl.export --out web/public/data

calibrate:       ## Fit traffic to Google Maps (needs GOOGLE_MAPS_API_KEY)
	uv run python -m taxi_rl.gmaps calibrate

validate:        ## Simulator vs Google ETA error
	uv run python -m taxi_rl.gmaps validate --n 40 --hour 18

web:
	cd web && npm install && npm run dev

all: grid train evaluate export
