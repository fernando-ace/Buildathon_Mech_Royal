# Quiet Iron

A survival-first bot for the Builders Mech Royale track. It optimizes for placement: remain inside the future safe zone, break away from close threats, collect convenient resources, and take low-risk shots.

## Run locally

```sh
python3 -u bot.py
python3 check/check.py -- python3 -u bot.py
```

The official check exercises protocol handling against a recorded observation stream. It does not simulate matches or evaluate strategy.

## Build and check as the scorer does

```sh
docker build --platform linux/amd64 -t quiet-iron .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" quiet-iron
```

The scorer provides the read-only map at `/arena`. Runtime code uses only the Python standard library and writes no files.
