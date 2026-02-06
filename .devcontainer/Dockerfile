FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspaces
RUN conda config --set ssl_verify false
RUN pip install torch torchvision matplotlib torchviz graphviz
ENTRYPOINT ["bash", "-c"]
CMD ["tail -f /dev/null"]
