from datetime import datetime
from pydantic import BaseModel
from finrl.domain.quote import Quote


class MarketState(BaseModel):
    security: str
    quotes: list[Quote]

    def quote_at(self, timestamp: datetime) -> Quote | None:
        valid_quotes = [
            quote
            for quote in self.quotes
            if quote.timestamp <= timestamp
        ]
        if not valid_quotes:
            return None

        return max(valid_quotes, key=lambda quote: quote.timestamp)


