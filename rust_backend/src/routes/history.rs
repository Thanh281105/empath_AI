//! REST API routes cho Chat History.
//!
//! Endpoints:
//!   GET    /api/sessions                    → Danh sách sessions
//!   GET    /api/sessions/{id}/messages      → Messages của session
//!   DELETE /api/sessions/{id}               → Xoá session

use actix_web::{web, HttpResponse, Responder};
use std::sync::Arc;

use crate::services::database::Database;

/// GET /api/sessions
pub async fn list_sessions(db: web::Data<Arc<Database>>) -> impl Responder {
    match db.list_sessions() {
        Ok(sessions) => HttpResponse::Ok().json(serde_json::json!({
            "sessions": sessions
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Database error: {}", e)
        })),
    }
}

/// GET /api/sessions/{session_id}/messages
pub async fn get_session_messages(
    db: web::Data<Arc<Database>>,
    path: web::Path<String>,
) -> impl Responder {
    let session_id = path.into_inner();

    match db.get_messages(&session_id) {
        Ok(messages) => HttpResponse::Ok().json(serde_json::json!({
            "session_id": session_id,
            "messages": messages
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Database error: {}", e)
        })),
    }
}

/// DELETE /api/sessions/{session_id}
pub async fn delete_session(
    db: web::Data<Arc<Database>>,
    path: web::Path<String>,
) -> impl Responder {
    let session_id = path.into_inner();

    match db.delete_session(&session_id) {
        Ok(true) => HttpResponse::Ok().json(serde_json::json!({
            "deleted": true,
            "session_id": session_id
        })),
        Ok(false) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Session not found"
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Database error: {}", e)
        })),
    }
}
