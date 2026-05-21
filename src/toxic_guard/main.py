from fastapi import FastAPI

from toxic_guard.api.routers import router


app = FastAPI(
    title="ToxicGuard API",
    version="1.0.0"
)

app.include_router(router)