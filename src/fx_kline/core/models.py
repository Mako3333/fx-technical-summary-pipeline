"""
Data models for FX-Kline using Pydantic
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class OHLCRequest(BaseModel):
    """Single OHLC data request"""

    pair: str = Field(..., description="Currency pair (e.g., 'USDJPY', 'EURUSD')")
    interval: str = Field(..., description="Timeframe (1m, 5m, 15m, 1h, 1d, etc.)")
    period: str = Field(..., description="Period (e.g., '30d', '1mo')")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"pair": "USDJPY", "interval": "1h", "period": "30d"}
        }
    )


class BatchOHLCRequest(BaseModel):
    """Batch request for multiple OHLC data fetches"""

    requests: List[OHLCRequest] = Field(..., description="List of OHLC requests")
    exclude_weekends: bool = Field(True, description="Exclude weekend data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {"pair": "USDJPY", "interval": "1d", "period": "30d"},
                    {"pair": "EURUSD", "interval": "1h", "period": "5d"},
                ],
                "exclude_weekends": True,
            }
        }
    )


class FetchError(BaseModel):
    """Error details for failed fetch"""

    pair: str
    interval: str
    period: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OHLCData(BaseModel):
    """OHLC data for a single pair"""

    pair: str
    interval: str
    period: str
    data_count: int
    columns: List[str]
    rows: List[dict] = Field(
        description="OHLC data rows with Datetime, Open, High, Low, Close, Volume"
    )
    timestamp_jst: Optional[datetime] = None
    warnings: List[str] = Field(
        default_factory=list, description="Data quality warnings"
    )


class BatchOHLCResponse(BaseModel):
    """Response for batch OHLC requests"""

    successful: List[OHLCData] = Field(default_factory=list)
    failed: List[FetchError] = Field(default_factory=list)
    total_requested: int
    total_succeeded: int
    total_failed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def summary(self) -> dict:
        """Get summary of the response"""
        return {
            "total_requested": self.total_requested,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "success_rate": (
                f"{(self.total_succeeded / self.total_requested * 100):.1f}%"
                if self.total_requested > 0
                else "N/A"
            ),
        }
