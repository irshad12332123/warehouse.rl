from __future__ import annotations

import os
import uvicorn

from server import app


def main() -> None:
    port: int = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

if __name__ == "__main__":
    main()
