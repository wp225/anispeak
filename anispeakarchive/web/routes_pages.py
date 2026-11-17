from __future__ import annotations
from flask import Blueprint, render_template, redirect, url_for, session, request

bp = Blueprint("pages", __name__)

@bp.get("/")
def home():
    return redirect(url_for("pages.search_page"))

@bp.get("/search")
def search_page():
    return render_template("pages/search.html")

@bp.get("/paper/<doc_id>")
def paper_page(doc_id: str):
    return render_template("pages/paper.html", doc_id=doc_id)

@bp.get("/analytics")
def analytics_page():
    return render_template("pages/analytics.html")

@bp.get("/about")
def about_page():
    return render_template("pages/about.html")

@bp.get("/ask")
def ask_page():
    if session.get("user_id") is None:
        return redirect(url_for("auth.login_page", next=request.path))
    return render_template("pages/ask.html")
