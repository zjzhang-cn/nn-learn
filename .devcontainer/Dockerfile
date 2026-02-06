# 使用 Python 3.9 作為基礎映像
FROM us-artifactory.apps.ge-healthcare.net/docker-ext/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 設置工作目錄（容器中的目錄）
WORKDIR /workspaces
RUN conda config --set ssl_verify false
RUN pip install torch torchvision matplotlib torchviz graphviz
# 設定容器啟動後執行的命令
ENTRYPOINT ["bash", "-c"]
CMD ["tail -f /dev/null"]
