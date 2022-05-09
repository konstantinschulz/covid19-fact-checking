from typing import List

from fastapi import FastAPI
from detect_fake_news import calculate_credibility, detect_fake_news

app: FastAPI = FastAPI()


@app.get("/credibility")
def get_credibility(text: str) -> float:
    return calculate_credibility(text)


@app.post("/credibility")
def post_credibility(sentences: List[str]) -> List[float]:
    return detect_fake_news(sentences)
