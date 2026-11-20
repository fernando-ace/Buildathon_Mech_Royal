FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py bot.py ./
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
