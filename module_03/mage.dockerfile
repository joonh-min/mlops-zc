FROM mageai/mageai:alpha

ARG PROJECT_NAME=homework_03
ARG MAGE_CODE_PATH=/home/src
ARG USER_CODE_PATH=${MAGE_CODE_PATH}/${PROJECT_NAME}

WORKDIR ${MAGE_CODE_PATH}

COPY ${PROJECT_NAME} ${PROJECT_NAME}

ENV USER_CODE_PATH=${USER_CODE_PATH}

RUN pip3 install -r ${USER_CODE_PATH}/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:${MAGE_CODE_PATH}/${PROJECT_NAME}"

CMD ["/bin/sh", "-c", "/app/run_app.sh"]