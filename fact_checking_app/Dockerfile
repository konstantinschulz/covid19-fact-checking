FROM konstantinschulz/gbert_base_fang_covid:latest as GBERT_FANG
FROM konstantinschulz/gbert_base:latest as GBERT
FROM python:3.9-slim
RUN apt-get update
# avoid https://github.com/debuerreotype/debuerreotype/issues/10
RUN mkdir -p /usr/share/man/man1
# Install tini and create an unprivileged user
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /sbin/tini
RUN addgroup --gid 1001 "elg" && adduser --disabled-password --gecos "ELG User,,," --home /elg --ingroup elg --uid 1001 elg && chmod +x /sbin/tini
# Everything from here down runs as the unprivileged user account
USER elg:elg
WORKDIR /elg
COPY --from=GBERT_FANG /workspace /elg/streamlit_app
COPY --from=GBERT /workspace /elg/gbert_base
ENV WORKERS=1
# Create a Python virtual environment for the dependencies
RUN python -m venv venv
RUN /elg/venv/bin/pip --no-cache-dir install torch==1.9.0
# Copy in our app, its requirements file and the entrypoint script
COPY --chown=elg:elg requirements.txt docker-entrypoint.sh elg_service.py /elg/
RUN /elg/venv/bin/pip --no-cache-dir install -r requirements.txt
COPY --chown=elg:elg . .
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
