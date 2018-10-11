FROM scannerresearch/scanner:cpu-latest
WORKDIR /opt
COPY deps.sh /opt
RUN bash ./deps.sh
