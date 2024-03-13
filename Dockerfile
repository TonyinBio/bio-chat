# syntax=docker/dockerfile:1

# TODO: Setup better development environment
FROM --platform=linux/amd64 nvdia/cuda:12.3.2-devel-ubuntu22.04
WORKDIR /home/code