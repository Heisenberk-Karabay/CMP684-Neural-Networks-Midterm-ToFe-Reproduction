#!/usr/bin/env bash
# Robust download of ImageNet-1k parquet shards from the Hugging Face hub.
# usage: HF_TOKEN=... bash src/download_imagenet.sh <split> <first> <last> <total>
split=$1; first=$2; last=$3; total=$4
out="$(dirname "$0")/../data/imagenet"
mkdir -p "$out"
for i in $(seq "$first" "$last"); do
  f=$(printf "%s-%05d-of-%05d.parquet" "$split" "$i" "$total")
  [ -f "$out/$f" ] && { echo "skip $f"; continue; }
  for try in $(seq 1 20); do
    curl -sSL --retry 10 --retry-all-errors --retry-delay 3 -C - \
      -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/$f" -o "$out/$f.part" && break
    echo "retry $try $f"; sleep 3
  done
  mv "$out/$f.part" "$out/$f" && echo "done $f $(du -h "$out/$f" | cut -f1)"
done
