# Quiet Iron

A survival-first bot for the Builders Mech Royale track. It optimizes for placement: reach the announced next safe circle early, route around static campus geometry, settle near cover, break away from close threats, collect convenient weapons, and take low-risk shots.

## Run locally

```sh
python3 -u bot.py
python3 check/check.py -- python3 -u bot.py
```

The official check exercises protocol handling against a recorded observation stream. It does not simulate matches or evaluate strategy.

## Build and check as the scorer does

```sh
docker build --platform linux/amd64 -t quiet-iron .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 --mount type=bind,source="$PWD/arena",target=/arena,readonly quiet-iron
```

The included `arena/` files are the official static map used for local checks; the scorer mounts its copy read-only at `/arena`. The bot reads the map once at startup and builds a coarse ground-navigation grid. If the map is unavailable, it falls back to direct steering. Runtime code uses only the Python standard library and writes no files.
