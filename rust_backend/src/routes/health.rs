//! Health check endpoint

use actix_web::{get, web, HttpResponse};
use crate::services::qdrant::QdrantService;

#[get("/health")]
pub async fn health_check(
    qdrant: web::Data<QdrantService>,
) -> HttpResponse {
    let qdrant_ok = qdrant.health_check().await.unwrap_or(false);

    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "services": {
            "qdrant": if qdrant_ok { "connected" } else { "disconnected" },
            "groq": "configured",
        },
        "version": env!("CARGO_PKG_VERSION"),
    }))
}
